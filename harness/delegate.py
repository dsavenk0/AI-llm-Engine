"""Hand a sub-task to another model: a bigger cloud model, another local engine, or a CLI agent.

Config — `.llm/workers.json` in the workspace, else `~/.llm/workers.json`:

    {"workers": {
       "claude": {"type": "openai", "url": "https://api.anthropic.com/v1", "model": "claude-sonnet-5-5",
                  "key_env": "ANTHROPIC_API_KEY",
                  "about": "strong at architecture and review"},
       "big":    {"type": "openai", "url": "http://127.0.0.1:8091/v1", "model": "local",
                  "about": "the 27B on the second engine port"},
       "gemini": {"type": "cli", "command": ["gemini", "-p"], "about": "has its own file tools"}}}

  type "openai" — any OpenAI-compatible chat endpoint (OpenAI, Anthropic, Gemini, another engine);
                  the key is read from the environment variable named in `key_env`, never from the file.
  type "cli"    — a command that takes the task as its last argument and prints the answer; it runs in
                  the workspace and may edit files itself, which the confirmation says.

The worker sees only the task text and the files the agent lists explicitly — never the conversation.
Its answer comes back as a tool result (data, not instructions).
"""
import json
import os
import subprocess
import urllib.error
import urllib.request

MAX_CONTEXT_BYTES = 60_000
MAX_ANSWER = 24_000


def config_path(root):
    for p in (os.path.join(root, ".llm", "workers.json"), os.path.expanduser("~/.llm/workers.json")):
        if os.path.isfile(p):
            return p
    return os.path.join(root, ".llm", "workers.json")


def load(root):
    try:
        with open(config_path(root), encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return {}
    workers = data.get("workers") if isinstance(data, dict) else None
    if not isinstance(workers, dict):
        return {}
    return {n: w for n, w in workers.items() if isinstance(w, dict) and w.get("type") in ("openai", "cli")}


PRESETS = {
    "openrouter": {"type": "openai", "url": "https://openrouter.ai/api/v1", "model": "anthropic/claude-sonnet-5-5",
                   "key_env": "OPENROUTER_API_KEY", "about": "any model on OpenRouter — change `model` to taste"},
}


def add_preset(root, name):
    """Write a ready-made worker into .llm/workers.json (the key itself is read from the environment)."""
    if name not in PRESETS:
        raise ValueError(f"unknown preset {name!r}; available: {', '.join(PRESETS)}")
    p = config_path(root)
    try:
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        data = {}
    if not isinstance(data, dict):
        data = {}
    if not isinstance(data.get("workers"), dict):
        data["workers"] = {}
    data["workers"][name] = dict(PRESETS[name])
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return p


def is_local_url(url):
    """http(s) to this machine. Decided on the parsed host: a prefix test took "127.evil.com" for local."""
    import ipaddress
    import urllib.parse
    host = (urllib.parse.urlsplit(str(url)).hostname or "").lower()
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def describe(workers):
    """What using each worker does, in terms a person can judge: program to run, server and key variable."""
    lines = []
    for n, w in workers.items():
        if w["type"] == "cli":
            lines.append(f"{n}: runs  {' '.join(map(str, w.get('command', [])))}  in the workspace (may edit files)")
        else:
            key = f" with ${w['key_env']}" if w.get("key_env") else ""
            where = "this machine" if is_local_url(w.get("url", "")) else "leaves this machine"
            lines.append(f"{n}: sends tasks to {w.get('url')}{key}  ({where})")
    return lines


def usable(spec):
    """(ok, reason) — a worker whose key is missing is listed but not offered to the model."""
    if spec["type"] == "openai":
        if not spec.get("url") or not spec.get("model"):
            return False, "needs url and model"
        if spec.get("key_env") and not os.environ.get(spec["key_env"]):
            return False, f"set ${spec['key_env']}"
    elif not spec.get("command"):
        return False, "needs command"
    return True, ""


def attach(root, files):
    """The listed workspace files as a context block; refuses paths outside the workspace. A PDF, Office doc,
    image or audio file is recognised (OCR / Whisper / text extraction) instead of read as raw bytes."""
    parts, total = [], 0
    for rel in files or []:
        full = os.path.realpath(os.path.join(root, rel))
        if full != os.path.realpath(root) and not full.startswith(os.path.realpath(root) + os.sep):
            raise ValueError(f"{rel} is outside the workspace")
        if needs_recognition(full):
            from harness import perception
            src = perception.recognize([full])[0]
            text = src["text"] or ("[" + (src["note"] or "unreadable") + "]")
            header = f"### {rel} ({src['kind']})"
        else:
            with open(full, encoding="utf-8", errors="replace") as f:
                text = f.read()
            header = f"### {rel}"
        total += len(text.encode())
        if total > MAX_CONTEXT_BYTES:
            raise ValueError(f"attached files exceed {MAX_CONTEXT_BYTES // 1000} KB; send fewer or smaller files")
        parts.append(f"{header}\n```\n{text}\n```")
    return "\n\n".join(parts)


def needs_recognition(path):
    """True for the file types perception turns into text (audio, images/scans, PDF, Office docs) — a plain
    text or code file is read directly and exactly, never run through OCR."""
    from engine import audio, documents
    ext = os.path.splitext(path)[1].lower()
    return ext in audio.AUDIO_EXTS or ext in documents.IMAGE_EXTS or ext in (".pdf", ".docx", ".pptx")


def build_prompt(task, context):
    return task.strip() + (f"\n\nContext files:\n\n{context}" if context else "")


def ask_openai(spec, prompt, timeout=600):
    headers = {"Content-Type": "application/json", "User-Agent": "llm-agent/1"}
    if spec.get("key_env"):
        headers["Authorization"] = "Bearer " + os.environ.get(spec["key_env"], "")
    body = {"model": spec["model"], "stream": False, "max_tokens": int(spec.get("max_tokens", 4096)),
            "messages": [{"role": "user", "content": prompt}]}
    req = urllib.request.Request(spec["url"].rstrip("/") + "/chat/completions",
                                 data=json.dumps(body).encode(), headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read(4_000_000))
    except urllib.error.HTTPError as e:
        detail = e.read(400).decode("utf-8", "replace")
        e.close()
        raise RuntimeError(f"HTTP {e.code}: {detail}")
    except (urllib.error.URLError, OSError, ValueError) as e:
        raise RuntimeError(str(e))
    try:
        return data["choices"][0]["message"].get("content") or ""
    except (KeyError, IndexError, TypeError):
        raise RuntimeError("unexpected response shape")


def ask_cli(spec, prompt, cwd, timeout=900):
    cmd = [str(c) for c in spec["command"]] + [prompt]
    try:
        p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL)
    except FileNotFoundError:
        raise RuntimeError(f"command not found: {cmd[0]}")
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"no answer within {timeout} s")
    if p.returncode != 0:
        raise RuntimeError(f"exit code {p.returncode}: {(p.stderr or p.stdout).strip()[-400:]}")
    return p.stdout


def ask(name, spec, task, root, files=None):
    prompt = build_prompt(task, attach(root, files))
    answer = ask_openai(spec, prompt) if spec["type"] == "openai" else ask_cli(spec, prompt, root)
    answer = answer.strip() or "(empty answer)"
    if len(answer) > MAX_ANSWER:
        answer = answer[:MAX_ANSWER] + f"\n… [{len(answer) - MAX_ANSWER} more characters]"
    return answer


def summary(workers):
    lines = []
    for n, w in workers.items():
        ok, why = usable(w)
        where = w.get("url") or " ".join(map(str, w.get("command", [])))
        lines.append(f"{'✓' if ok else '✗'} {n}  [{w['type']}] {w.get('model', '')} {where}".rstrip()
                     + (f"  — {why}" if why else "") + (f"  · {w['about']}" if w.get("about") else ""))
    return lines


def tools(root, workers, allow=lambda what: True):
    """{"delegate": (schema, fn)} offering only the usable workers, or {} when there are none."""
    ready = {n: w for n, w in workers.items() if usable(w)[0]}
    if not ready:
        return {}
    catalog = "; ".join(f"{n} — {w.get('about') or w['type']}" for n, w in ready.items())

    MAX_JOBS = 4

    def describe_job(job):
        spec, files = ready[job["worker"]], job["files"]
        sends = f"send this task to “{job['worker']}”" + (f" with {len(files)} file(s): {', '.join(files)}" if files else "")
        if spec["type"] == "cli":
            sends += f" — runs `{' '.join(map(str, spec.get('command', [])))}` in the workspace and may change files itself"
        elif not is_local_url(spec.get("url", "")):
            sends += f" — leaves this machine ({spec.get('url')}" + (f", key ${spec['key_env']}" if spec.get("key_env") else "") + ")"
        return sends + f"\n    task: {job['task'].strip()[:300]}"

    def run_job(job):
        try:
            return ask(job["worker"], ready[job["worker"]], job["task"], root, job["files"])
        except (RuntimeError, OSError, ValueError) as e:
            return f"error: {job['worker']}: {e}"

    def delegate(worker=None, task=None, files=None, workers=None, jobs=None):
        """One worker, the same task for several workers, or different jobs — run at the same time."""
        if isinstance(files, str):
            files = [files]
        if isinstance(workers, str):
            workers = [workers]
        todo = []
        if jobs:
            if not isinstance(jobs, list) or not all(isinstance(j, dict) for j in jobs):
                return "error: jobs must be a list of {worker, task, files}"
            for j in jobs:
                f = j.get("files")
                todo.append({"worker": j.get("worker"), "task": j.get("task") or "", "files": [f] if isinstance(f, str) else f})
        elif workers:
            todo = [{"worker": w, "task": task or "", "files": files} for w in workers]
        elif worker:
            todo = [{"worker": worker, "task": task or "", "files": files}]
        if not todo:
            return "error: name a worker (or workers, or jobs) and a task"
        if len(todo) > MAX_JOBS:
            return f"error: at most {MAX_JOBS} workers at once"
        for job in todo:
            if job["worker"] not in ready:
                return f"error: unknown worker {job['worker']!r}; available: {', '.join(ready)}"
            if not str(job["task"]).strip():
                return f"error: no task for {job['worker']}"
            try:
                attach(root, job["files"])         # validate before asking the user anything
            except (OSError, ValueError) as e:
                return f"error: {e}"
        what = "\n".join(describe_job(j) for j in todo)
        if len(todo) > 1:
            what = f"send {len(todo)} tasks at the same time:\n" + what
            if sum(ready[j["worker"]]["type"] == "cli" for j in todo) > 1:
                what += "\n  (several command-line workers may change files at the same time)"
        if not allow(what):
            return "user declined to delegate"
        if len(todo) == 1:
            return run_job(todo[0])
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=len(todo)) as pool:
            answers = list(pool.map(run_job, todo))
        return "\n\n".join(f"## {j['worker']}\n{a}" for j, a in zip(todo, answers))

    return {"delegate": ({"type": "function", "function": {
        "name": "delegate",
        "description": "Hand a self-contained sub-task to another model and get its answer back. The worker "
                       "does not see this conversation: write a complete task and list the files it needs. "
                       "Use it for what you do poorly (hard reasoning, a second opinion, review) — not for "
                       "things you can do yourself. To get several opinions at once pass `workers` (the same "
                       "task to each, up to 4), or `jobs` for different tasks; they run in parallel and the "
                       "answers come back together, one section per worker. Workers: " + catalog,
        "parameters": {"type": "object", "properties": {
            "worker": {"type": "string", "enum": list(ready)},
            "workers": {"type": "array", "items": {"type": "string", "enum": list(ready)},
                        "description": "ask all of these the same task at the same time"},
            "task": {"type": "string", "description": "complete instructions, with all needed context"},
            "files": {"type": "array", "items": {"type": "string"},
                      "description": "workspace files to attach (max 60 KB total)"},
            "jobs": {"type": "array", "description": "different tasks for different workers, run at the same time",
                     "items": {"type": "object", "properties": {
                         "worker": {"type": "string", "enum": list(ready)}, "task": {"type": "string"},
                         "files": {"type": "array", "items": {"type": "string"}}},
                         "required": ["worker", "task"]}}},
            "required": []}}}, delegate)}
