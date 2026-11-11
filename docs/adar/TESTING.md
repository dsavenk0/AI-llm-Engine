# ADAR — Testing

```sh
.venv/bin/python -m unittest tests.test_adar     # 81 tests, about a second
python -m adar.sim                               # a scripted cluster, printed
```

No second Mac, no socket, no model and no MLX: nodes are small Python objects, time is a number, and one
seeded generator drives every network effect and every pairing code — so a failing test fails the same way
every time.

## Method

The real protocol, pairing, node, coordinator, scheduler and transfer code runs against a simulated
network (`adar/sim.py`). The scheduler's own internals are not mocked; the network under it is.

The simulated network provides, per link: latency, jitter, loss, corruption, bandwidth (one message at a
time in each direction, which is what makes bandwidth a limit), partitions, and loss in one direction only
— a request that arrives whose answer does not. It also offers a tap, for tests that play an eavesdropper
or replay recorded traffic.

## What is covered

| Area | Held to |
|---|---|
| Versioning | another protocol version is refused by name, in messages, tasks and advertisements |
| Discovery | an advertisement earns a registry row and nothing else; one fingerprint from two places is refused, not merged |
| Pairing | needs the code a person read off the other screen, which never crosses the network; a wrong code fails closed and three void the attempt; a node not in pairing mode ignores requests; the human step cannot be skipped; a recorded exchange replayed proves nothing; a key that does not match its advertised fingerprint is not paired |
| Authorization | pairing alone permits no task; the worker checks the permission itself; a grant for a type with no handler is refused; revocation is immediate |
| Message authentication | unpaired sender, missing envelope, replay, altered body and wrong addressee are each refused; a late message is taken once |
| Capability | every field labelled; a simulated node never reports this Mac's memory; self-contradicting documents are refused and change nothing |
| Scheduler | hand-computed costs for local-wins and remote-wins cases, in both directions; every gate with its recorded reason; local-only tasks; the three outcomes when nothing can run a task; the explanation checked for content, not presence |
| Placement end to end | a faster node gets the work; a tiny task over a slow link stays home; the link figure is measured, below the configured rate; a node short of memory is passed over; a task waits for a model and runs when a node with it appears; a local-only task never reaches the wire |
| Tasks | one execution per idempotency key, including after a lost acknowledgment; cancel before, during and after execution; a result already in flight survives a cancel; deadlines; a failing handler fails the task, not the node; only allowlisted handlers; a node speaks only for its own tasks; the audit trail holds no content |
| Failures | a crashed worker is noticed within the documented bound and its task moves on; with no retries left a task fails rather than vanishing; a task with nowhere to go waits and resumes; partition and heal; one lost heartbeat changes nothing; a lossy, jittery link does not flap a node offline; a link slower than the heartbeat is slow, not dead; an offline node is retried with a bounded backoff |
| Transfer | clean, lossy and corrupting links; damage anywhere in a frame; resume from where it stopped; a resume from the wrong place fails the final check; a chunk out of place is not written; a slow receiver never makes the sender buffer the payload (24 MB moved inside 4 MB); a transfer does not delay heartbeats |
| The simulation itself | every simulated node labelled `SIMULATED` everywhere; the demo loads neither the engine nor a model; ten workers cost kilobytes; the same seed gives the same run |

## What is not covered

| Gap | Why |
|---|---|
| a coordinator restart | tasks in flight live in its memory; there is no durable task store yet |
| exactly-once across a worker's crash | a restarted worker has forgotten its idempotency keys, so a retried task runs again |
| transfers as part of a task | the framing is tested on its own channel; no control-plane verb starts one |
| `chat.complete` | allowlisted, no handler yet |
| the real-cryptography path | `cryptography` is not installed where the tests run; that test is skipped with the reason |
| real links, real Bonjour, real pairing between two Macs | Phase 3 and later; simulation results are not evidence for any of them |
