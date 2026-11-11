# ADAR — Roadmap

**Phases 0 – 2 are done:** the design, the contracts, and a local simulation that runs the real protocol
code. Phases 3 and later need a second Mac on the same LAN and have not started.

| Phase | What it builds | Second Mac | Status |
|---|---|---|---|
| 0 — Audit and design | these documents | no | done |
| 1 — Core contracts | node identity, capability schema, task protocol and lifecycle, handler allowlist | no | done — `adar/identity.py`, `capability.py`, `protocol.py` |
| 2 — Local simulation | several logical nodes on one Mac, the real protocol and scheduler, a deterministic fake network, failure injection | no | done — the rest of `adar/`, 81 tests |
| 3 — LAN discovery and secure pairing | Bonjour / mDNS, a real transport with pinned-key TLS, Keychain identity, the on-disk peer ledger | yes | not started |
| 4 — Node agent | worker lifecycle on a real machine, resource checks, the `chat.complete` handler | yes | not started |
| 5 — Independent distributed jobs | task-level distribution end to end, with a durable task store and benchmarks | yes | not started |
| 6 — Network-aware data plane | transfers attached to tasks, the link profiler on real links | yes | not started |
| 7 — Model replication | artifact distribution on the existing resumable download, replica health, load-aware routing | yes | not started |
| 8 — Distributed inference | pipeline parallelism (`engine/cluster.py`), then tensor parallelism and a distributed KV cache — each only for an architecture and runtime shown to support it | yes | not started |
| 9 — Hardening | security review, diagnostics, recovery validation | yes | not started |

## What Phase 2 settled

Where the design left room, the code decided:

* **The pairing code never crosses the network.** The responder generates it and shows it on its own
  screen; a person types it at the initiator; each side proves it knows the code through the session key.
* **Message signatures stand in for TLS** until there is a socket: every message is attributable to a
  paired identity, unmodified, addressed and not a replay. Confidentiality is the part still missing.
* **A permission belongs to the worker.** It checks each task against its own peer book; the coordinator's
  copy only saves it from offering what would be refused.
* **A task with nowhere to go waits if the obstacle can pass**, and is rejected only when waiting cannot
  help.
* **A pong counts for any ping still remembered**, so a link slower than the heartbeat does not walk a
  live node to `offline`.

## Carried into the later phases

| Gap | Phase |
|---|---|
| a real transport: sockets, pinned-key TLS, Bonjour | 3 |
| Keychain storage; `peers.json` and `audit.log` on disk; connection rate limiting | 3 |
| the real-cryptography path exercised (`cryptography` installed) | 3 |
| a CLI and API surface (`./llm cluster …`) | 3 – 4 |
| a `chat.complete` handler wired to the engine — where "never load N models to imitate a cluster" starts to matter | 4 |
| a durable task store, so a coordinator restart loses nothing | 5 |
| idempotency keys that survive a worker's restart | 5 |
| transfers started by a control-plane verb and attached to tasks | 6 |
| speed and load estimates learned from finished tasks instead of set by hand | 5 – 6 |

## Scope

* **Expert parallelism** comes last and only for the GGUF MoE route: the native engine has no MoE models.
* **A native runtime** is not part of ADAR. The protocol and the scheduler name no backend, so they do not
  change if the engine's own runtime does (`architecture/RUNTIME_EVOLUTION_ROADMAP.md`).
* **CPU / SME2 execution** is not assumed by any phase; the measurement behind that is in the same document.
