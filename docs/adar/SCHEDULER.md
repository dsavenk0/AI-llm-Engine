# ADAR — Scheduler

Where a task runs — here, or on which node — and the sentence that explains it. Implemented in
`adar/scheduler.py` (the decision, a pure function) and `adar/coordinator.py` (the registry it reads from),
with hand-computed cases in `tests/test_adar.py`. The numbers it is fed are uncalibrated until there are
two real machines to measure.

## Above the local scheduler, not instead of it

`engine/server.py` already schedules one machine: `Admission` caps what is in flight and `Batcher` merges
requests that arrive together. The cluster scheduler sits above that. "Run locally" is always a candidate
and is costed like any other — never assumed worse because another machine exists.

## The rule

```
local_cost     = load(local) + queue_wait(local) + runtime(local)
remote_cost(n) = load(n) + transfer(input + output, link n) + queue_wait(n) + runtime(n)

run locally                   if local_cost <= every remote_cost
run on the cheapest node n    otherwise
```

with

| Term | Computed as |
|---|---|
| `runtime` | the task's `work_tokens` ÷ the node's tokens per second on that model |
| `load` | the node's cold-start time for the model, or zero if it is already loaded there |
| `queue_wait` | tasks running or queued there × `runtime` — an estimate that assumes they are alike |
| `transfer` | two round trips + bytes × 8 ÷ the link's measured throughput |

What is compared is estimated end-to-end completion time, never processor speed or link rate alone. Equal
costs stay local.

## Who is a candidate

A node is considered only if every one of these holds; otherwise it is left out and the reason is kept:

| Gate | Reason recorded |
|---|---|
| its state is `online` or `busy` | `node is degraded` / `offline` / … |
| the pairing was granted this task type | `this pairing does not permit <type>` |
| it has a handler for the type | `no <type> handler` |
| it has the exact model | `missing checkpoint <name>` — no substitution |
| its available memory is known and covers `peak_memory_gb` | `needs X GB, Y GB available` |
| its link was measured | `link not measured` |
| its speed on this model is known | `no runtime estimate for this model` |

A task not marked `remote_execution_allowed` is never offered to another node at all. A node that merely
exists and is connected is never chosen because it is there, and nothing is guessed: a missing measurement
removes the candidate.

## When nothing can run it

| Situation | Outcome |
|---|---|
| some obstacle can pass by itself — a node offline or still being profiled, memory short | `waiting_for_resources`; placed again when a node comes back, a capability arrives or a link is measured |
| otherwise, a model that nobody has | `waiting_for_model` |
| otherwise | `rejected`, with every candidate's reason |

## Explanations

Every placement produces one sentence, and they are kept in order — `Coordinator.explain(task_id)`:

- `Running locally: remote overhead on far (~2.1s for transfer, model load and queue) exceeds the estimated compute saving (~0.5s).`
- `Assigned to studio: model already loaded there, faster on this model, estimated saving ~9.5s (~2.5s there against ~12.0s here).`
- `Running locally: the task is marked local-only.`
- `Waiting for a model: coordinator: missing checkpoint huge; mini: missing checkpoint huge.`
- `Interrupted on mini (node went offline); recovering, attempt 2 of 3.`

## Node state machine

```
discovered → awaiting_pairing → paired → online ⇄ busy → degraded → offline
                                   any state → revoked → (a new pairing) → awaiting_pairing
```

Health is a heartbeat, one ping a second:

| Consecutive intervals with no pong | Effect |
|---|---|
| 1 | nothing — a single lost heartbeat does not move a node |
| 2 | `degraded`: it gets no new work |
| 4 | `offline`: its tasks are recovered or failed |

An offline node is still pinged, at an interval that doubles up to 30 s; one pong brings it back. A pong
counts for any ping still remembered, so a link slower than the heartbeat interval is a slow node, not a
dead one.

## Open

The weighting between the terms, and where the speed and load figures come from. Today they are set by
hand (`Coordinator.set_estimate`) and the link throughput by a probe transfer; learning them from finished
tasks needs measurements on real machines.
