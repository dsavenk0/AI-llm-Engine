# ADAR — Networking

What carries the messages. The framing of the data plane is written and tested over simulated links
(`adar/transfer.py`); everything about real links is unmeasured, because measuring it takes a second Mac.

## What exists

No discovery, socket or TLS code for ADAR exists in `app/`, `engine/`, `ui/` or `llm.py`. The only
multi-machine code in the repo is `engine/cluster.py`, which rides on MLX's own distributed backend; its
correctness test runs over loopback on one machine, which proves the layer split and says nothing about a
link.

## Transports

| Technology | Standing |
|---|---|
| 10 GbE, 2.5 / 5 GbE | unmeasured — needs the hardware and an `iperf3`-style test, not the port's advertised speed |
| Thunderbolt Bridge | unmeasured — macOS exposes it as an ordinary interface; what it sustains for this traffic is not established |
| Wi-Fi 6 / 6E / 7 | a fallback, never assumed suitable for large transfers |
| Bonjour / mDNS | the standard discovery API (`NWBrowser`); not written yet. What it may advertise is decided in `SECURITY.md` |
| gRPC, QUIC | not planned — JSON over HTTP covers the control plane, and the project has no use of either |

## Control plane and data plane

**Control plane** — pairing, capabilities, task assignment, status, cancellation, heartbeats: small,
frequent, latency-sensitive JSON (`PROTOCOL.md`). It has its own channel, so a large transfer never delays
a heartbeat or a cancellation.

**Data plane** — model artifacts and large task inputs and outputs: binary frames with a fixed header,
outside the control plane's JSON and its 1 MB limit. `adar/transfer.py` holds it to four properties:

| Property | How |
|---|---|
| bounded memory | the sender reads its source a chunk at a time and holds at most a window of unacknowledged chunks; a slow receiver slows the reading instead of growing a buffer |
| a damaged frame is never used | each carries a SHA-256 over its header fields and payload; the receiver asks again for one that does not match |
| the whole is checked too | a final digest over the stream, so a transfer resumed from the wrong place, or truncated, fails at the end |
| resumable | the receiver says how many chunks it holds; a new sender starts there |

The receiver takes chunks in order only (go-back-N), which keeps its own memory at one chunk at the price
of resending what arrives ahead of a gap — the right trade on a LAN, where loss is rare. Compression is not
used: whether its CPU cost pays depends on the link, and that is a measurement to take on real hardware.

Not yet attached to tasks: no control-plane verb starts a transfer, so task inputs and results still travel
inside control messages.

## Link profile

The scheduler is given measured figures, never the interface's advertised rate:

| Figure | Source today |
|---|---|
| round-trip time | the heartbeat |
| payload throughput | a bounded transfer of a synthetic payload, timed end to end |

Still to add: percentiles of the round-trip time rather than the last value, a short-burst figure at the
size of a typical request, timestamps with a time-to-live, and re-measuring periodically.

## Not claimed

No throughput figure for any real link, no zero-copy or RDMA path, no achievable bandwidth for tensor
traffic. Results from the simulation show that the orchestration behaves; they are not evidence about
speed.
