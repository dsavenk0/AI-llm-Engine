"""Where a task runs: here, or on which node — the decision rule of docs/adar/SCHEDULER.md, as a pure
function of what is known, returning the choice *and* the sentence that explains it.

    local_cost     = load(local) + queue_wait(local) + runtime(local)
    remote_cost(n) = load(n) + transfer(input, link n) + queue_wait(n) + runtime(n) + transfer(output, link n)

The comparison is estimated end-to-end completion time, never processor speed or link rate alone. A node
is a candidate only if it passes every gate (state, pairing scope, handler, exact model, memory, a
*measured* link, a known speed); a node that merely exists is never chosen "because it is there". Nothing
is guessed: a missing measurement removes the candidate and says so.

The numeric inputs are whatever the caller has — measured where a task has run, estimated where not —
and SCHEDULER.md leaves their calibration open until real multi-Mac measurements exist. This module fixes
the shape of the decision, which is what Phase 2 can test.
"""
from dataclasses import dataclass, field

RUNNABLE = ("online", "busy")           # node states a task may be sent to
_PASSING = ("needs ", "node is ", "link not measured", "available memory unknown")   # obstacles that pass


@dataclass
class View:
    """What the scheduler knows about one place a task could run."""
    node_id: str
    label: str                           # what a person calls it ("mini", "studio"), for explanations
    state: str = "online"
    handlers: tuple = ()
    scope: tuple = ()                    # task types this pairing was granted (remote nodes)
    installed: tuple = ()
    loaded: tuple = ()
    available_gb: float | None = None
    busy: int = 0                        # tasks running or queued there now
    tps: float | None = None             # tokens per second on the task's model
    load_s: float | None = None          # cost of a cold start of that model
    rtt_s: float | None = None           # measured; None = the link was never profiled
    bps: float | None = None             # measured payload throughput, bits per second
    local: bool = False


@dataclass
class Decision:
    target: str | None                   # "local", a node id, or None when nothing can run it now
    explanation: str
    cost_s: float | None = None
    wait: str | None = None              # "waiting_for_model" | "waiting_for_resources" when target is None
    rejected: dict = field(default_factory=dict)      # label -> why that candidate was ruled out
    costs: dict = field(default_factory=dict)         # label -> estimated seconds, for the audit trail


def _gate(task, v):
    """None if `v` may run `task`, else the reason it may not."""
    need = task["requirements"].get("model")
    mem = float(task["resource_estimate"].get("peak_memory_gb") or 0.0)
    if not v.local:
        if v.state not in RUNNABLE:
            return f"node is {v.state}"
        if task["type"] not in v.scope:
            return f"this pairing does not permit {task['type']}"
    if task["type"] not in v.handlers:
        return f"no {task['type']} handler"
    if need and need not in v.installed:
        return f"missing checkpoint {need}"
    if v.available_gb is None:
        return "available memory unknown"
    if mem > v.available_gb:
        return f"needs {mem:.1f} GB, {v.available_gb:.1f} GB available"
    if not v.local and (v.rtt_s is None or not v.bps):
        return "link not measured"
    if not v.tps or v.tps <= 0:
        return "no runtime estimate for this model"
    return None


def cost(task, v):
    """(total seconds, parts) for running `task` at `v`. Call only after `_gate` passed."""
    est = task["resource_estimate"]
    need = task["requirements"].get("model")
    runtime = float(est.get("work_tokens") or 1) / v.tps
    load = 0.0 if (not need or need in v.loaded) else float(v.load_s or 0.0)
    wait = v.busy * runtime                       # tasks ahead, assumed alike: an estimate and labelled so
    parts = {"runtime": runtime, "load": load, "queue": wait, "transfer": 0.0}
    if not v.local:
        size = float(est.get("input_bytes") or 0) + float(est.get("output_bytes") or 0)
        parts["transfer"] = 2 * v.rtt_s + size * 8 / v.bps
    return sum(parts.values()), parts


def decide(task, local, nodes):
    """`local`: View of this machine, or None if it cannot run tasks at all. `nodes`: Views of the others."""
    rejected, runnable = {}, []
    remote_ok = bool(task.get("remote_execution_allowed"))
    for v in ([local] if local else []) + (list(nodes) if remote_ok else []):
        why = _gate(task, v)
        if why:
            rejected[v.label] = why
        else:
            total, parts = cost(task, v)
            runnable.append((total, v, parts))
    costs = {v.label: round(t, 3) for t, v, _ in runnable}
    if not runnable:
        reasons = "; ".join(f"{k}: {w}" for k, w in rejected.items()) or "no node available"
        if not remote_ok and nodes:
            reasons += " (the task is local-only: other nodes were not considered)"
        why_all = list(rejected.values())
        # something that can change by itself (a node coming back, memory freed, a link being profiled)
        # parks the task; only when every obstacle is one that waiting cannot remove is it rejected
        if any(w.startswith(_PASSING) for w in why_all):
            return Decision(None, f"Waiting for resources: {reasons}.", wait="waiting_for_resources",
                            rejected=rejected)
        if any("missing checkpoint" in w for w in why_all):
            return Decision(None, f"Waiting for a model: {reasons}.", wait="waiting_for_model", rejected=rejected)
        return Decision(None, f"Rejected: nothing can run this task — {reasons}.", rejected=rejected)
    runnable.sort(key=lambda r: (r[0], not r[1].local, r[1].label))      # ties go to local, then by name
    total, best, parts = runnable[0]
    loc = next(((t, p) for t, v, p in runnable if v.local), None)
    if best.local:
        others = [(t, v, p) for t, v, p in runnable if not v.local]
        if not remote_ok:
            text = "Running locally: the task is marked local-only."
        elif not others:
            text = "Running locally: no other node can take it" + (
                " (" + "; ".join(f"{k}: {w}" for k, w in rejected.items()) + ")." if rejected else ".")
        else:
            rt, rv, rp = others[0]
            overhead = rp["transfer"] + rp["load"] + rp["queue"]
            saving = (parts["runtime"] + parts["load"] + parts["queue"]) - rp["runtime"]
            if saving <= 0:
                text = (f"Running locally: {rv.label} would not be faster (~{rt:.1f}s there against "
                        f"~{total:.1f}s here).")
            else:
                text = (f"Running locally: remote overhead on {rv.label} (~{overhead:.1f}s for transfer, model "
                        f"load and queue) exceeds the estimated compute saving (~{saving:.1f}s).")
        return Decision("local", text, total, rejected=rejected, costs=costs)
    if loc is None:
        text = f"Assigned to {best.label}: this machine cannot run it ({rejected.get(local.label, 'no local worker') if local else 'no local worker'}); estimated ~{total:.1f}s."
    else:
        lt, lp = loc
        reason = []
        if lp["load"] > 0 and parts["load"] == 0:
            reason.append("model already loaded there")
        if lp["queue"] > parts["queue"]:
            reason.append("this machine is busy")
        if lp["runtime"] > parts["runtime"]:
            reason.append("faster on this model")
        text = (f"Assigned to {best.label}: {', '.join(reason) or 'lower estimated completion time'}, "
                f"estimated saving ~{lt - total:.1f}s (~{total:.1f}s there against ~{lt:.1f}s here).")
    return Decision(best.node_id, text, total, rejected=rejected, costs=costs)
