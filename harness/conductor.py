"""The Conductor: an optional orchestrator that splits a task and routes the pieces to models by role.

Three roles, each mapped to a model in `.llm/conductor.json` (the person sets this up with `llm conduct`):

    {"roles": {
       "heavy": {"type": "openai", "url": "https://openrouter.ai/api/v1", "model": "anthropic/claude-sonnet-5-5",
                 "key_env": "OPENROUTER_API_KEY"},        # a remote model over the API
       "code":  {"local": "Qwen3-Coder-7B"},             # an installed local model, run on this machine's engine
       "fast":  "quickworker"}}                            # a worker name from .llm/workers.json

A role value is one of: `{"local": "<installed model>"}` (run on the one local engine), an inline worker spec
(same shape as `harness/delegate.py`, a remote/API model), or the NAME of a worker in `.llm/workers.json`.

How they run follows the hardware. A remote (API) model runs over the network, so remote subtasks go **in
parallel**. The local engine holds one model at a time (16 GB), so local roles **take turns on it**: before a
local subtask the engine switches to that role's model, and the Conductor carries the running state (every
subtask's result) forward, so work continues from where it stands after each switch. Local subtasks are grouped
by model to switch as little as possible, and the local chain runs alongside the remote ones.

The flow is harness²'s shape kept small and local — **decompose** (the heavy role writes a plan), **route**
(each subtask on its role's model), **synthesize** (the heavy role composes the answer) — off by default:
nothing runs until `llm conduct "<task>"`. Switching the local model is injected (`switch_local`) so this module
is testable without an engine, and so the CLI can wire it to `./llm use` + restart.
"""
import json
import os

from . import delegate

ROLES = ("heavy", "code", "fast")
MAX_SUBTASKS = 6
MAX_PARALLEL = 3
MAX_DEP_BYTES = 12_000          # how much of a dependency's answer is quoted into a dependent subtask


def config_path(root):
    for p in (os.path.join(root, ".llm", "conductor.json"), os.path.expanduser("~/.llm/conductor.json")):
        if os.path.isfile(p):
            return p
    return os.path.join(root, ".llm", "conductor.json")


def is_local(spec):
    return isinstance(spec, dict) and isinstance(spec.get("local"), str) and bool(spec["local"].strip())


def role_ok(spec):
    """(ok, reason) for a resolved role spec — a local model, or a usable delegation worker."""
    if is_local(spec):
        return True, ""
    if isinstance(spec, dict):
        return delegate.usable(spec)
    return False, "not configured"


def load_mapping(root):
    """The raw {role: value} from the config file (value: a worker name, an inline spec, or {'local': model})."""
    try:
        with open(config_path(root), encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return {}
    raw = data.get("roles") if isinstance(data, dict) else None
    return {r: raw[r] for r in ROLES if isinstance(raw, dict) and r in raw}


def resolve_roles(root, local=None):
    """{role: spec} with worker names resolved and missing roles filled from `local` (the local engine model
    spec, e.g. {'local': '<selected model>'}). Returns also {role: reason} for the roles that stayed unusable."""
    raw = load_mapping(root)
    workers = delegate.load(root)
    roles, dropped = {}, {}
    for role in ROLES:
        spec = raw.get(role)
        if isinstance(spec, str):                       # a worker name from workers.json
            spec = workers.get(spec, {"_missing": spec})
        if not isinstance(spec, dict):
            spec = local
        if not isinstance(spec, dict):
            dropped[role] = "no model configured and no local fallback"
            continue
        if "_missing" in spec:
            dropped[role] = f"worker {spec['_missing']!r} is not in .llm/workers.json"
            continue
        ok, why = role_ok(spec)
        (roles if ok else dropped).__setitem__(role, spec if ok else why)
    return roles, dropped


def set_role(root, role, value):
    """Write one role → value into `.llm/conductor.json` (value: a worker name, {'local': model}, or a spec)."""
    if role not in ROLES:
        raise ValueError(f"role must be one of {', '.join(ROLES)}")
    p = config_path(root)
    try:
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        data = {}
    if not isinstance(data, dict):
        data = {}
    if not isinstance(data.get("roles"), dict):
        data["roles"] = {}
    if value is None:
        data["roles"].pop(role, None)
    else:
        data["roles"][role] = value
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return p


def where(spec):
    """A short label for a role's model, for the routing trace."""
    if is_local(spec):
        return f"{spec['local']} (local)"
    if isinstance(spec, dict) and spec.get("type") == "cli":
        return " ".join(map(str, spec.get("command", []))) or "cli"
    host = (spec.get("url") or "").split("//", 1)[-1].split("/", 1)[0]
    return f"{spec.get('model', '?')} @ {host}" if isinstance(spec, dict) else "?"


def _extract_json(text):
    """The first balanced {...} object in a model's reply (it may wrap it in prose or a code fence)."""
    s = text.find("{")
    while s != -1:
        depth, instr, esc = 0, False, False
        for i in range(s, len(text)):
            c = text[i]
            if instr:
                esc = (c == "\\") and not esc
                if c == '"' and not esc:
                    instr = False
                continue
            if c == '"':
                instr = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[s:i + 1])
                    except ValueError:
                        break
        s = text.find("{", s + 1)
    return None


PLAN_PROMPT = (
    "You are the conductor of a team of models with three roles:\n"
    "  fast  — routine, fast work: finding files, simple edits, formatting, quick checks.\n"
    "  code  — writing and fixing code to a specification.\n"
    "  heavy — architecture, tricky reasoning, resolving conflicts.\n"
    "Break the task below into at most {maxn} atomic subtasks and assign each to the cheapest role that can do it. "
    "A subtask may depend on earlier ones by id. Reply with ONLY a JSON object, no prose:\n"
    '{{"subtasks": [{{"id": "s1", "role": "fast|code|heavy", "task": "<self-contained instruction>", '
    '"depends_on": []}}]}}\n\nTask:\n{task}'
)

SYNTH_PROMPT = (
    "You asked a team to carry out these subtasks. Combine their results into one clear, correct answer to the "
    "original task. Note any subtask that failed or looks wrong rather than hiding it.\n\n"
    "Original task:\n{task}\n\n{bundle}"
)


class _Engine:
    """Runs one prompt on a role's model, switching the local engine when a local role needs a different model.
    `switch_local(model)` loads that model on the engine and returns its OpenAI-compatible spec; it is called only
    from the single local chain, so the global switch is never racy."""

    def __init__(self, switch_local, emit):
        self.switch_local = switch_local
        self.emit = emit
        self.loaded = None          # the model currently in the engine
        self.local_spec = None

    def run(self, spec, prompt, root):
        if is_local(spec):
            model = spec["local"]
            if self.loaded != model:
                if self.switch_local is None:
                    raise RuntimeError("no way to switch the local model here")
                self.emit(f"[switch] engine → {model}")
                self.local_spec = self.switch_local(model)
                self.loaded = model
            return delegate.ask_openai(self.local_spec, prompt)
        if spec.get("type") == "cli":
            return delegate.ask_cli(spec, prompt, root)
        return delegate.ask_openai(spec, prompt)


def decompose(task, heavy, engine, root, available, srctext=""):
    """Ask the heavy role for a plan. Returns a list of validated subtasks. Raises RuntimeError on a bad plan.
    `srctext`: recognised attachments, shown to the planner so it can route work over them."""
    full = task.strip()
    if srctext:
        full += "\n\n--- Attached sources (recognised; plan subtasks that use them) ---\n" + srctext[:SOURCE_PLAN_CAP]
    reply = engine.run(heavy, PLAN_PROMPT.format(maxn=MAX_SUBTASKS, task=full), root)
    data = _extract_json(reply)
    items = (data or {}).get("subtasks") if isinstance(data, dict) else None
    if not isinstance(items, list) or not items:
        raise RuntimeError("the planner did not return a usable list of subtasks")
    out, seen = [], set()
    for i, it in enumerate(items[:MAX_SUBTASKS], 1):
        if not isinstance(it, dict):
            continue
        sid = str(it.get("id") or f"s{i}")
        role = it.get("role") if it.get("role") in available else next(iter(available))
        body = str(it.get("task") or "").strip()
        if not body or sid in seen:
            continue
        seen.add(sid)
        deps = it.get("depends_on")
        out.append({"id": sid, "role": role, "task": body,
                    "depends_on": [str(d) for d in deps] if isinstance(deps, list) else []})
    for s in out:                                       # keep only deps that name a real, earlier subtask
        s["depends_on"] = [d for d in s["depends_on"] if d in seen and d != s["id"]]
    if not out:
        raise RuntimeError("the plan had no valid subtasks")
    return out


def _levels(subtasks):
    """Order subtasks into dependency levels (each level can run together). Breaks cycles by forcing progress."""
    done, levels, remaining = set(), [], list(subtasks)
    while remaining:
        ready = [s for s in remaining if all(d in done for d in s["depends_on"])]
        if not ready:                                   # a cycle or a dangling dep: run the first that is left
            ready = [remaining[0]]
        levels.append(ready)
        done |= {s["id"] for s in ready}
        remaining = [s for s in remaining if s["id"] not in done]
    return levels


def run_subtasks(subtasks, roles, engine, root, emit=lambda _l: None, srctext=""):
    """Run every subtask on its role's model. Within a dependency level, remote (API) subtasks run in parallel
    while the local ones take turns on the engine (grouped by model); the two go on at the same time. `srctext`:
    recognised attachments — each subtask gets the passages relevant to it (TF·IDF, so a long document stays
    small). Returns {id: answer} — the running state the Conductor carries across switches."""
    from concurrent.futures import ThreadPoolExecutor
    results, by_id = {}, {s["id"]: s for s in subtasks}

    def prompt_for(s):
        ctx = ""
        if srctext:
            from engine import documents
            ex = documents.select_excerpts(srctext, s["task"], budget=4000)["text"]
            if ex:
                ctx += "\n\n--- From the attached sources ---\n" + ex
        for d in s["depends_on"]:
            if d in results:
                ctx += f"\n\nResult of {d} ({by_id[d]['role']}):\n{results[d][:MAX_DEP_BYTES]}"
        return s["task"] + ctx

    def run_remote(s):
        emit(f"[route] {s['id']} · {s['role']} → {where(roles[s['role']])} | {s['task'][:80]}")
        try:
            return s["id"], engine.run(roles[s["role"]], prompt_for(s), root)
        except (RuntimeError, OSError, ValueError) as e:
            return s["id"], f"error: {e}"

    def run_local_chain(locals_):
        for s in sorted(locals_, key=lambda s: roles[s["role"]]["local"]):   # group same-model subtasks together
            emit(f"[route] {s['id']} · {s['role']} → {where(roles[s['role']])} | {s['task'][:80]}")
            try:
                results[s["id"]] = engine.run(roles[s["role"]], prompt_for(s), root)
            except (RuntimeError, OSError, ValueError) as e:
                results[s["id"]] = f"error: {e}"

    for level in _levels(subtasks):
        locals_ = [s for s in level if is_local(roles[s["role"]])]
        remotes = [s for s in level if not is_local(roles[s["role"]])]
        if not remotes:
            run_local_chain(locals_)
            continue
        with ThreadPoolExecutor(max_workers=min(MAX_PARALLEL, len(remotes)) + (1 if locals_ else 0)) as pool:
            futures = [pool.submit(run_remote, s) for s in remotes]
            local_future = pool.submit(run_local_chain, locals_) if locals_ else None
            for f in futures:
                sid, ans = f.result()
                results[sid] = ans
            if local_future:
                local_future.result()
    return results


def synthesize(task, subtasks, results, heavy, engine, root):
    bundle = "\n\n".join(f"### {s['id']} ({s['role']}): {s['task']}\n{results.get(s['id'], '(no result)')}"
                         for s in subtasks)
    return engine.run(heavy, SYNTH_PROMPT.format(task=task.strip(), bundle=bundle), root)


SOURCE_PLAN_CAP = 8000     # how much of the attached sources the planner sees; subtasks pull the relevant parts


def conduct(task, root, local=None, switch_local=None, emit=lambda _l: None, sources=()):
    """Perceive → decompose → route → synthesize. `local`: the local-engine model spec for unmapped roles (e.g.
    the selected model). `switch_local(model) -> spec`: loads a model on the engine and returns its endpoint.
    `emit(line)` gets the trace. `sources`: recognised attachments (harness.perception.recognize) — the planner
    sees them and each subtask draws the relevant passages, so the senior/code model works on a scan, a recording
    or a document without a manual step. Returns the final answer text."""
    roles, dropped = resolve_roles(root, local=local)
    if "heavy" not in roles:
        raise RuntimeError("the 'heavy' role needs a model (it plans and synthesizes). "
                           "Set it with: llm conduct set heavy <model-or-worker>")
    from harness import perception
    srctext = perception.as_text(sources) if sources else ""
    engine = _Engine(switch_local, emit)
    if sources:
        emit("[perceive] " + ", ".join(f"{s['name']} ({s['kind']})" for s in sources))
    emit("[plan] " + " · ".join(f"{r} → {where(roles[r])}" for r in ROLES if r in roles))
    subtasks = decompose(task, roles["heavy"], engine, root, set(roles), srctext)
    emit(f"[plan] {len(subtasks)} subtask(s)")
    results = run_subtasks(subtasks, roles, engine, root, emit, srctext)
    emit("[synth] heavy → " + where(roles["heavy"]))
    return synthesize(task, subtasks, results, roles["heavy"], engine, root)
