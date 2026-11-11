# ADAR — Architecture

Two or more Macs on one trusted local network working as a coordinated AI compute cluster.

**Where it stands:** the control plane — identity, pairing, the task protocol, the scheduler, node health
and the data-plane framing — is written and tested against a simulated network (`adar/`,
`tests/test_adar.py`). Nothing has run between two real machines yet. `ROADMAP.md` has the phases.

## Three levels of cooperation, built in this order

1. **Task-level** — independent requests go to different Macs. The first real milestone.
2. **Model-level** — the same model runs on several Macs; a request goes to whichever replica is free.
3. **Within one inference** — a single model execution split across Macs (pipeline or tensor parallelism,
   a distributed KV cache). This needs a model architecture, a runtime and a communication primitive that
   support it; a fast link alone does not make it possible, and does not show that overhead is not the
   larger term.

Each Mac keeps its own unified memory. ADAR never presents the memory of several Macs as one pool.

## Code map

| Module | Role |
|---|---|
| `adar/identity.py` | a node's long-term key (Ed25519 / X25519 via `cryptography`, or a stdlib stand-in for simulation only) |
| `adar/capability.py` | the capability document: every field labelled measured / static / estimated / unavailable |
| `adar/protocol.py` | message framing, versioning, the task lifecycle, the handler allowlist |
| `adar/wire.py` | signed, addressed, replay-proof envelopes for every message after pairing |
| `adar/pairing.py` | the human-verified pairing flow; the peer book (scopes, revocation) |
| `adar/node.py` | a node: peers, the allowlisted worker, cancellation, the audit trail |
| `adar/coordinator.py` | registry, node state machine, heartbeats, task tracking, recovery, explanations |
| `adar/scheduler.py` | where a task runs, and the sentence that says why — a pure function |
| `adar/transfer.py` | data-plane framing: checksummed chunks, bounded window, final digest, resume |
| `adar/sim.py` | the simulated network and the scripted cluster |

A node is given a transport with three methods — `now()`, `send(src, dst, raw)`, `after(delay, fn)` — and
never learns whether that is a socket and a clock or the simulation. Phase 3 replaces the transport and
nothing above it.

## Components, and what each builds on

| # | Component | State | Builds on |
|---|---|---|---|
| 1 | Cluster manager (cluster-wide config) | not built | the per-project config and trust gate of `harness/trust.py` |
| 2 | Coordinator | simulated | any node can coordinate, and the coordinator also runs tasks itself |
| 3 | Node agent | simulated | allowlisted handlers only, like the terminal agent's command discipline |
| 4 | LAN discovery (Bonjour / mDNS) | not built | no discovery code exists in the repo |
| 5 | Node registry and identity | simulated | the public key is the identity, so it survives a change of hostname or IP |
| 6 | Capability profiler | simulated | `engine/memory.py`'s counters; chip generation and thermal state are not detected anywhere and stay "unavailable" |
| 7 | Resource and network monitor | memory: reused · network: simulated | `engine/memory.py` |
| 8 | Task scheduler | simulated | sits above the server's own `Admission` + `Batcher`: "run locally" is always a candidate |
| 9 | Remote execution service | simulated, `echo.test` only | — |
| 10 | Model and artifact distribution | not built | the resumable, checksummed download in `llm.py` (`hf_pull`) |
| 11 | Result transfer | framing only | `adar/transfer.py`; not yet attached to tasks |
| 12 | Health and recovery | simulated | the single-node `GET /health` |
| 13 | Pairing, authentication, authorization | simulated | see `SECURITY.md` |
| 14 | Event and audit log | in memory | metadata only, never task content |
| 15 | Distributed inference adapter | not built | `engine/cluster.py` — layer-range sharding over MLX's distributed backend, proven only over loopback |

Three existing pieces shaped the design without being reusable as they are: `harness/delegate.py` (naming a
worker and saying, before anything runs, what will leave the machine), the download job in
`app/bar/Bar.swift` (a state machine with a stall watchdog and bounded retries, where a job never silently
disappears) and `ui/agentloop.py` (one SSE stream of typed events). ADAR's task has its own type; it
borrows the first one's approval discipline, the second one's lifecycle and the third one's event shape.

## Standalone guarantee

Every ADAR component is additive and opt-in, like the engine's own "loopback unless you turn it on". With
ADAR off, the server, the CLI, the editor backend and LLMBar behave exactly as `README.md` describes: no new
listener, no new background process, nothing new in `.llm/config.json`. Today that holds trivially — nothing
outside `adar/` and its tests imports it — and it is the rule every later phase is held to.
