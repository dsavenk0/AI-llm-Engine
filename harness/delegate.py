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
    return {n: w for n, w in (data.get("workers") or {}).items()
            if isinstance(w, dict) and w.get("type") in ("openai", "cli")}


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
    data.setdefault("workers", {})[name] = dict(PRESETS[name])
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
    """The listed workspace files as a context block; refuses paths outside the workspace."""
    parts, total = [], 0
    for rel in files or []:
        full = os.path.realpath(os.path.join(root, rel))
        if full != os.path.realpath(root) and not full.startswith(os.path.realpath(root) + os.sep):
            raise ValueError(f"{rel} is outside the workspace")
        with open(full, encoding="utf-8", errors="replace") as f:
            text = f.read()
        total += len(text.encode())
        if total > MAX_CONTEXT_BYTES:
            raise ValueError(f"attached files exceed {MAX_CONTEXT_BYTES // 1000} KB; send fewer or smaller files")
        parts.append(f"### {rel}\n```\n{text}\n```")
    return "\n\n".join(parts)


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

    def delegate(worker, task, files=None):
        spec = ready.get(worker)
        if spec is None:
            return f"error: unknown worker {worker!r}; available: {', '.join(ready)}"
        if isinstance(files, str):
            files = [files]
        try:
            attach(root, files)            # validate before asking the user anything
        except (OSError, ValueError) as e:
            return f"error: {e}"
        sends = f"send this task to “{worker}”" + (f" with {len(files)} file(s): {', '.join(files)}" if files else "")
        if spec["type"] == "cli":
            sends += f" — runs `{' '.join(map(str, spec.get('command', [])))}` in the workspace and may change files itself"
        elif not is_local_url(spec.get("url", "")):
            sends += f" — leaves this machine ({spec.get('url')}" + (f", key ${spec['key_env']}" if spec.get("key_env") else "") + ")"
        if not allow(sends + f"\n    task: {task.strip()[:300]}"):
            return "user declined to delegate"
        try:
            return ask(worker, spec, task, root, files)
        except (RuntimeError, OSError, ValueError) as e:
            return f"error: {worker}: {e}"

    return {"delegate": ({"type": "function", "function": {
        "name": "delegate",
        "description": "Hand a self-contained sub-task to another model and get its answer back. The worker "
                       "does not see this conversation: write a complete task and list the files it needs. "
                       "Use it for what you do poorly (hard reasoning, a second opinion, review) — not for "
                       "things you can do yourself. Workers: " + catalog,
        "parameters": {"type": "object", "properties": {
            "worker": {"type": "string", "enum": list(ready)},
            "task": {"type": "string", "description": "complete instructions, with all needed context"},
            "files": {"type": "array", "items": {"type": "string"},
                      "description": "workspace files to attach (max 60 KB total)"}},
            "required": ["worker", "task"]}}}, delegate)}
