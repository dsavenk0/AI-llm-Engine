# ADAR — Protocol

The control plane: what nodes say to each other and how a task lives. Implemented in `adar/protocol.py`,
`adar/wire.py` and `adar/capability.py`, and exercised over a simulated network; no message has crossed a
real one yet.

## Stance

The project has one network convention — plain JSON over HTTP, SSE when a response streams — and the
control plane follows it: typed JSON messages, versioned from the start. There is no gRPC, WebSocket or
binary framing in the repo, and none is added here. Large payloads are the one place a binary convention is
justified; that is the data plane, in `NETWORKING.md`.

Every message carries `"v": 1`. A message of another version is refused with a reason naming both — never
parsed on a guess. A message over 1 MB is refused on its length alone, before any parsing.

## Envelope

After pairing, every message travels in an envelope:

```json
{"v": 1, "verb": "task.submit",
 "body": {"from": "ed25519:…", "to": "ed25519:…", "seq": 17, "sig": "…", "data": {…}}}
```

The signature is by the sender's long-term key over the version, verb, sender, recipient, sequence number
and data. A receiver refuses a message that is not from a paired node, whose signature does not match the
key pinned at pairing, that is addressed to another node, or whose sequence number it has already taken
(a sliding window of 64: a message that merely arrives late is accepted, the same one twice is not).

Only `pair.request` and `pair.confirm` travel without an envelope — there is no pinned key yet — and the
pairing flow authenticates those itself (`SECURITY.md`).

## Verbs

| Verb | Direction | Purpose |
|---|---|---|
| `pair.request` / `pair.confirm` | both | pairing, see `SECURITY.md` |
| `pair.revoke` | either | end the pairing; the receiver forgets the sender |
| `capability.push` | node → coordinator | the capability document, after pairing and when it changes |
| `task.submit` | coordinator → node | offer a task; re-sent with the same key if nothing is heard |
| `task.accept` / `task.reject` | node → coordinator | taken, or refused with the reason |
| `task.status` | node → coordinator | each state change; carries the result or the error at the end |
| `task.cancel` | coordinator → node | stop it |
| `health.ping` / `health.pong` | coordinator ↔ node | heartbeat; feeds the node state machine in `SCHEDULER.md` |

Discovery is not a verb: it is an unauthenticated advertisement carrying a key fingerprint, an address and
the protocol version, and nothing else.

## Capability document

Every field says how its value was obtained — measured, static, estimated or unavailable — so a number
nobody measured cannot pass as one that was.

```json
{
  "v": 1,
  "node_id": "ed25519:base64-public-key",
  "simulated": false,
  "static": {
    "unified_memory_gb": {"value": 16.0, "source": "static"},
    "macos_version": {"value": "26.4.1", "source": "static"},
    "chip_generation": {"value": null, "source": "unavailable"},
    "adar_version": {"value": "0.2", "source": "static"},
    "installed_models": {"value": ["Qwen3.5-4B-MLX-4bit"], "source": "static"},
    "task_handlers": {"value": ["echo.test"], "source": "static"}
  },
  "dynamic": {
    "available_memory_gb": {"value": 9.8, "source": "measured"},
    "memory_pressure": {"value": 1, "source": "measured"},
    "loaded_models": {"value": [], "source": "measured"},
    "active_tasks": {"value": 0, "source": "measured"},
    "queue_depth": {"value": 0, "source": "measured"},
    "thermal_state": {"value": null, "source": "unavailable"},
    "disk_free_gb": {"value": 240.0, "source": "measured"}
  }
}
```

`chip_generation` and `thermal_state` are "unavailable" because no code in the repo reads either. A
simulated node's document says `"simulated": true` and never reports this Mac's memory as its own.

A document that contradicts itself is refused and the previous one kept: a field marked unavailable that
carries a value, a handler the protocol does not define, a model listed as loaded but not as installed, or
a document describing a node other than its sender.

## Task

```json
{
  "v": 1,
  "task_id": "uuid",
  "idempotency_key": "uuid",
  "type": "echo.test",
  "schema_version": 1,
  "input": {},
  "requirements": {"model": "Qwen3.5-4B-MLX-4bit", "backend": "native"},
  "resource_estimate": {"peak_memory_gb": 3.2, "work_tokens": 400, "input_bytes": 2000, "output_bytes": 2000},
  "data_classification": "local-only",
  "remote_execution_allowed": false,
  "retry_policy": {"max_retries": 2, "retryable": true},
  "state": "created"
}
```

* **`remote_execution_allowed` defaults to false.** A task stays on the machine that made it unless its
  submitter marks it eligible; the worker checks the flag again on arrival.
* **`type` must be on the allowlist** (`chat.complete`, `echo.test`). There is no way to register a handler
  for anything else, so a peer cannot supply code to run. Only `echo.test` has a handler today.
* **`idempotency_key`** makes a retry safe: a node that already has the key answers with the task's current
  state instead of running it again. This holds while the node stays up — a restarted node has forgotten
  its keys.
* **`resource_estimate`** beyond `peak_memory_gb` is what the scheduler reads; see `SCHEDULER.md`.

### Lifecycle

```
created → validated → queued → scheduled → dispatched → accepted → [loading →] running
        → returning_result → completed
```

plus `rejected`, `waiting_for_model`, `waiting_for_resources`, `cancelling`, `cancelled`, `failed`,
`timed_out`, `interrupted` and `recovery_pending`. The allowed transitions are a table in
`adar/protocol.py`; anything else raises. A task whose node disappears goes `interrupted →
recovery_pending` and back to `queued` if its retry policy allows, otherwise `failed` — never left waiting.
