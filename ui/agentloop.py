"""The editor UI's agent: a tool loop over the local engine that explores the project, asks the user
questions, and edits files only after approval. Events are streamed to the browser as they happen.

Event types (all JSON): run · delta · reasoning · tool · approval · question · tool_result · done · error
"""
import fnmatch
import json
import os
import re
import secrets
import subprocess
import threading
import urllib.error
import urllib.request

MAX_STEPS = 14
MAX_TOOL_OUTPUT = 12000
WAIT_SECONDS = 900          # how long a question/approval may stay unanswered

TOOLS = [
    {"type": "function", "function": {
        "name": "list_dir", "description": "List files under a project folder (recursive, max 300 entries).",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "folder relative to the project, '.' for the root"},
            "pattern": {"type": "string", "description": "optional glob filter such as '*.py'"}}, "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "read_file", "description": "Read a text file (optionally a line range). Lines are numbered.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"}, "start": {"type": "integer"}, "end": {"type": "integer"}}, "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "search", "description": "Search file contents with a regular expression; returns path:line: text.",
        "parameters": {"type": "object", "properties": {
            "pattern": {"type": "string"}, "path": {"type": "string"}, "glob": {"type": "string"}}, "required": ["pattern"]}}},
    {"type": "function", "function": {
        "name": "edit_file", "description": "Replace one exact, unique snippet of a file. The user approves each edit.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"}, "old": {"type": "string"}, "new": {"type": "string"}},
            "required": ["path", "old", "new"]}}},
    {"type": "function", "function": {
        "name": "write_file", "description": "Create or overwrite a file. The user approves each write.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                       "required": ["path", "content"]}}},
    {"type": "function", "function": {
        "name": "run", "description": "Run a shell command in the project (timeout 120 s). The user approves each one.",
        "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}},
    {"type": "function", "function": {
        "name": "ask_user", "description": "Ask the user a clarifying question with 2-4 short options when the "
        "request is ambiguous or a decision is theirs. Prefer this over guessing.",
        "parameters": {"type": "object", "properties": {
            "question": {"type": "string"},
            "options": {"type": "array", "items": {"type": "string"}, "description": "2-4 short answers"}},
            "required": ["question"]}}},
]

CATEGORY = {"list_dir": "explore", "search": "search", "read_file": "read", "edit_file": "edit",
            "write_file": "edit", "run": "command", "ask_user": "question"}


class Run:
    def __init__(self, alive=None):
        self.id = secrets.token_hex(6)
        self.cancelled = False
        self.alive = alive or (lambda: True)     # False once the client has gone away
        self._answers = {}
        self._cv = threading.Condition()

    def answer(self, key, value):
        with self._cv:
            self._answers[key] = value
            self._cv.notify_all()

    def cancel(self):
        with self._cv:
            self.cancelled = True
            self._cv.notify_all()

    def wait(self, key, timeout=WAIT_SECONDS):
        waited = 0.0
        with self._cv:
            while key not in self._answers and not self.cancelled and waited < timeout:
                self._cv.wait(0.5)
                waited += 0.5
                if not self.alive():             # nothing is written while waiting, so a closed tab is
                    self.cancelled = True        # only noticed by looking at the connection itself
            return self._answers.pop(key, None)


RUNS = {}


def system_prompt(project_name, extra=""):
    return ("You are an autonomous coding agent inside a code editor, working on the project "
            f"'{project_name}'. Explore with list_dir, search and read_file before you answer questions "
            "about the code; do not guess file contents. When the request is ambiguous or a choice is the "
            "user's, call ask_user with 2-4 options instead of assuming. Make the smallest correct edit with "
            "edit_file (every edit and command is approved by the user, so say briefly what you are doing). "
            "Never claim you ran or changed something you did not. Answer concisely. Files the user attached are "
            "quoted in their message." + extra)


def _short(text, n=300):
    return text if len(text) <= n else text[:n] + "…"


class Tools:
    def __init__(self, project, run, emit, auto_edit=False):
        self.P, self.run, self.emit, self.auto = project, run, emit, auto_edit

    # ---- helpers
    def _approve(self, call_id, kind, **info):
        if kind in ("edit", "write") and self.auto:
            return True
        self.emit({"type": "approval", "id": call_id, "kind": kind, **info})
        ans = self.run.wait("approval:" + call_id)
        return bool(ans and ans.get("approve"))

    # ---- tool implementations; each returns the text handed back to the model
    def list_dir(self, path=".", pattern=None, **_):
        base = self.P.resolve(path)
        out = []
        for d, dirs, files in os.walk(base):
            dirs[:] = sorted(x for x in dirs if x not in self.P.skip and not x.startswith("."))
            for f in sorted(files):
                if self.P.escapes(os.path.join(d, f)):
                    continue
                if not pattern or fnmatch.fnmatch(f, pattern):
                    out.append(self.P.rel(os.path.join(d, f)))
                    if len(out) >= 300:
                        return "\n".join(out) + "\n…"
        return "\n".join(out) or "(empty)"

    def read_file(self, path, start=None, end=None, **_):
        full = self.P.resolve(path)
        if os.path.isdir(full):
            raise IsADirectoryError(path)
        with open(full, "rb") as f:
            raw = f.read(self.P.max_file + 1)
        if b"\x00" in raw[:8192]:
            return "(binary file)"
        lines = raw.decode("utf-8", errors="replace").splitlines()
        s, e = max(1, int(start or 1)), min(len(lines), int(end or len(lines)))
        return "\n".join(f"{i:5d}  {lines[i - 1]}" for i in range(s, e + 1)) or "(empty file)"

    def search(self, pattern, path=".", glob=None, **_):
        return "\n".join(self.P.search(pattern, path, glob, limit=200)) or "(no matches)"

    def edit_file(self, path, old, new, _call_id="", **_):
        full = self.P.resolve(path)
        try:
            with open(full, encoding="utf-8") as f:
                text = f.read()
        except UnicodeDecodeError:
            return f"error: {path} is not UTF-8 text; it is not edited (saving would damage it)"
        if not old:
            return "error: `old` is empty; give the exact snippet to replace"
        n = text.count(old)
        if n != 1:
            return f"error: the snippet occurs {n} times in {path}; give a unique exact snippet"
        updated = text.replace(old, new, 1)
        if not self._approve(_call_id, "edit", path=self.P.rel(full), old=old, new=new):
            return "the user rejected this edit"
        self.P.write(path, updated)
        return f"edited {path}"

    def write_file(self, path, content, _call_id="", **_):
        full = self.P.resolve(path, must_exist=False)
        old = ""
        if os.path.exists(full):
            with open(full, encoding="utf-8", errors="replace") as f:
                old = f.read()
        if not self._approve(_call_id, "write", path=self.P.rel(full), old=old, new=content):
            return "the user rejected this write"
        self.P.write(path, content)
        return f"wrote {path} ({len(content.splitlines())} lines)"

    def run_command(self, command, _call_id="", **_):
        danger = None
        try:
            from agent import dangerous_reason     # the terminal agent's list of always-confirm commands
            danger = dangerous_reason(command)
        except Exception:
            pass
        if not self._approve(_call_id, "run", command=command, dangerous=danger):
            return "the user declined to run the command"
        from harness import shell
        p = shell.run(command, cwd=self.P.root, timeout=120)
        out = (p.stdout + p.stderr).strip()
        if p.timed_out:
            return f"error: timed out after 120 s (the command was stopped)\n{out}"
        return f"exit code {p.returncode}\n{out}"

    def ask_user(self, question, options=None, _call_id="", **_):
        opts = [str(o) for o in (options or [])][:4]
        self.emit({"type": "question", "id": _call_id, "question": question, "options": opts})
        ans = self.run.wait("question:" + _call_id)
        if ans is None:
            return "the user did not answer"
        return ans.get("text") or "the user skipped the question"

    def call(self, name, args, call_id):
        fn = {"run": self.run_command}.get(name) or getattr(self, name, None)
        if fn is None or name.startswith("_") or name in ("call",):
            return f"error: unknown tool {name}"
        try:
            out = fn(_call_id=call_id, **args)
        except TypeError as e:
            return f"error: bad arguments: {e}"
        except (OSError, ValueError, PermissionError, re.error) as e:
            return f"error: {e}"
        return out if len(out) <= MAX_TOOL_OUTPUT else out[:MAX_TOOL_OUTPUT] + f"\n… [truncated, {len(out)} chars]"


def _stream_chat(engine_port, messages, think, run):
    """Yield ("content"|"reasoning", text) and finally ("calls", [tool calls]). Raises RuntimeError."""
    payload = {"model": "local", "messages": messages, "tools": TOOLS, "stream": True,
               "chat_template_kwargs": {"enable_thinking": bool(think)}}
    req = urllib.request.Request(f"http://127.0.0.1:{engine_port}/v1/chat/completions",
                                 data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    try:
        r = urllib.request.urlopen(req, timeout=3600)
    except urllib.error.HTTPError as e:
        detail = e.read(300).decode("utf-8", "replace")
        e.close()
        raise RuntimeError(f"the model returned {e.code}: {detail}")
    except (urllib.error.URLError, OSError):
        raise RuntimeError("the model is not running — start it first (./llm start)")
    calls = []
    try:
        for raw in r:
            if run.cancelled:
                return
            line = raw.strip()
            if not line.startswith(b"data:") or line == b"data: [DONE]":
                continue
            ev = json.loads(line[5:])
            if ev.get("error"):
                raise RuntimeError(ev["error"].get("message", "model error"))
            for ch in ev.get("choices", []):
                d = ch.get("delta", {})
                if d.get("reasoning_content"):
                    yield "reasoning", d["reasoning_content"]
                if d.get("content"):
                    yield "content", d["content"]
                calls += d.get("tool_calls") or []
    finally:
        r.close()
    yield "calls", calls


def run_agent(project, engine_port, messages, emit, run, think=False, auto_edit=False, system_extra=""):
    """True when the model finished; False when the loop ended on an error event (already emitted)."""
    tools = Tools(project, run, emit, auto_edit)
    convo = [{"role": "system", "content": system_prompt(os.path.basename(project.root), system_extra)}] + messages
    for step in range(MAX_STEPS):
        content, calls = [], []
        for kind, value in _stream_chat(engine_port, convo, think, run):
            if kind == "calls":
                calls = value
            else:
                if kind == "content":
                    content.append(value)
                emit({"type": "delta" if kind == "content" else "reasoning", "text": value})
        if run.cancelled:
            return True
        msg = {"role": "assistant", "content": "".join(content)}
        if not calls:
            convo.append(msg)
            return True
        msg["tool_calls"] = [{"id": c.get("id") or f"call_{step}_{i}", "type": "function", "function": c["function"]}
                             for i, c in enumerate(calls)]
        convo.append(msg)
        for c in msg["tool_calls"]:
            name = c["function"]["name"]
            try:
                args = json.loads(c["function"].get("arguments") or "{}")
            except ValueError:
                args = {}
            if not isinstance(args, dict):
                args = {}
            emit({"type": "tool", "id": c["id"], "name": name, "category": CATEGORY.get(name, "other"),
                  "args": {k: _short(str(v)) for k, v in args.items() if k not in ("content", "old", "new")}})
            result = tools.call(name, args, c["id"])
            emit({"type": "tool_result", "id": c["id"], "name": name, "ok": not result.startswith("error:"),
                  "summary": _short(result.split("\n", 1)[0], 160)})
            convo.append({"role": "tool", "tool_call_id": c["id"], "content": result})
            if run.cancelled:
                return True
    emit({"type": "error", "message": f"stopped after {MAX_STEPS} steps"})
    return False
