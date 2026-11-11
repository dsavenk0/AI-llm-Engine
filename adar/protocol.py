"""ADAR control-plane messages and the task lifecycle — the shapes of docs/adar/PROTOCOL.md, with the
validation behind them.

Plain JSON, versioned from the start (`"v": 1`), following the one network convention this project already
has (JSON over HTTP, SSE when something streams). A message carrying an unknown version is refused with a
reason naming both versions — never parsed on a guess, which is the whole point of versioning it.
"""
import json
import uuid

VERSION = 1

# docs/adar/PROTOCOL.md, "Lifecycle"
STATES = ("created", "validated", "queued", "scheduled", "dispatched", "accepted", "loading", "running",
          "returning_result", "completed", "rejected", "waiting_for_model", "waiting_for_resources",
          "cancelling", "cancelled", "failed", "timed_out", "interrupted", "recovery_pending")
TERMINAL = frozenset(("completed", "rejected", "cancelled", "failed", "timed_out"))

# the only transitions the task state machine allows; anything else is a bug, not a state to accept
NEXT = {
    "created": ("validated", "rejected"),
    "validated": ("queued", "rejected"),
    "queued": ("scheduled", "cancelled", "rejected", "waiting_for_model", "waiting_for_resources"),
    "waiting_for_model": ("queued", "cancelled", "failed"),
    "waiting_for_resources": ("queued", "cancelled", "failed"),
    "scheduled": ("dispatched", "cancelled", "failed"),
    "dispatched": ("accepted", "rejected", "timed_out", "cancelling", "interrupted"),
    "accepted": ("loading", "running", "cancelling", "failed", "interrupted"),
    "loading": ("running", "failed", "cancelling", "interrupted"),
    "running": ("returning_result", "failed", "timed_out", "cancelling", "interrupted"),
    "returning_result": ("completed", "failed", "interrupted"),
    "interrupted": ("recovery_pending", "failed"),
    "recovery_pending": ("queued", "failed"),
    "cancelling": ("cancelled", "completed", "failed"),
}

# Allowlisted task handlers (docs/adar/SECURITY.md, "Authorization": no task handler may run a command,
# script or binary supplied by another node — enforced by this list being the only dispatch table).
HANDLERS = ("chat.complete", "echo.test")

VERBS = ("capability.push", "task.submit", "task.accept", "task.reject", "task.status", "task.cancel",
         "health.ping", "health.pong", "pair.request", "pair.confirm", "pair.revoke")

MAX_MESSAGE_BYTES = 1 << 20   # mirrors engine/server.py's own MAX_BODY posture: refuse before parsing


class ProtocolError(ValueError):
    pass


def new_task(task_type, payload, *, model=None, backend=None, peak_memory_gb=0.0,
             remote_execution_allowed=False, idempotency_key=None, retry_policy=None):
    """A task message at state `created`. `remote_execution_allowed` defaults False — a task stays local
    unless the submitter explicitly marks it eligible, mirroring harness/delegate.py's rule that nothing
    leaves this machine without a named, explicit approval."""
    if task_type not in HANDLERS:
        raise ProtocolError(f"unknown task type {task_type!r} (handlers: {', '.join(HANDLERS)})")
    return {
        "v": VERSION,
        "task_id": str(uuid.uuid4()),
        "idempotency_key": idempotency_key or str(uuid.uuid4()),
        "type": task_type,
        "schema_version": 1,
        "input": payload,
        "requirements": {"model": model, "backend": backend},
        "resource_estimate": {"peak_memory_gb": float(peak_memory_gb)},
        "data_classification": "local-only",
        "remote_execution_allowed": bool(remote_execution_allowed),
        "retry_policy": retry_policy or {"max_retries": 2, "retryable": True},
        "state": "created",
    }


def advance(task, state):
    """Moves `task` to `state` if the lifecycle allows it, else raises. Returns the task (mutated in place)
    so a caller can chain; a refused transition is a programming error surfaced loudly, not a silent no-op
    that would leave a task wedged in a state nothing ever clears."""
    if state not in STATES:
        raise ProtocolError(f"no such task state: {state!r}")
    now = task["state"]
    if now in TERMINAL:
        raise ProtocolError(f"task {task['task_id']} is already {now} (terminal); refusing to move to {state}")
    if state not in NEXT.get(now, ()):
        raise ProtocolError(f"illegal transition {now} → {state} for task {task['task_id']}")
    task["state"] = state
    return task


def check_version(message, *, what="message"):
    v = message.get("v")
    if v != VERSION:
        raise ProtocolError(f"{what} speaks protocol v{v!r}, this node speaks v{VERSION} — refusing rather "
                            f"than guessing the format")
    return message


def encode(verb, body):
    if verb not in VERBS:
        raise ProtocolError(f"unknown control-plane verb: {verb!r}")
    raw = json.dumps({"v": VERSION, "verb": verb, "body": body}).encode()
    if len(raw) > MAX_MESSAGE_BYTES:
        raise ProtocolError(f"{verb}: message of {len(raw)} bytes exceeds the {MAX_MESSAGE_BYTES}-byte limit")
    return raw


def decode(raw):
    """(verb, body) from a control-plane message. Oversized input is refused on its length alone, before
    any JSON parsing — engine/server.py's own posture for untrusted bodies."""
    if len(raw) > MAX_MESSAGE_BYTES:
        raise ProtocolError(f"message of {len(raw)} bytes exceeds the {MAX_MESSAGE_BYTES}-byte limit")
    try:
        msg = json.loads(raw)
    except ValueError as e:
        raise ProtocolError(f"not valid JSON: {e}") from None
    if not isinstance(msg, dict):
        raise ProtocolError("a control-plane message must be a JSON object")
    check_version(msg)
    verb = msg.get("verb")
    if verb not in VERBS:
        raise ProtocolError(f"unknown control-plane verb: {verb!r}")
    body = msg.get("body")
    if not isinstance(body, dict):
        raise ProtocolError(f"{verb}: body must be a JSON object")
    return verb, body


def validate_task(task):
    """Structural validation of an inbound task, before a handler sees any of it."""
    check_version(task, what="task")
    for field in ("task_id", "idempotency_key", "type", "input", "state"):
        if field not in task:
            raise ProtocolError(f"task is missing {field!r}")
    if task["type"] not in HANDLERS:
        raise ProtocolError(f"no handler for task type {task['type']!r} on this node "
                            f"(allowlisted: {', '.join(HANDLERS)})")
    if not isinstance(task["input"], dict):
        raise ProtocolError("task input must be a JSON object")
    if task["state"] not in STATES:
        raise ProtocolError(f"task arrived in unknown state {task['state']!r}")
    return task
