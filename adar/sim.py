"""Several logical nodes on one Mac, the real protocol between them, a network that is a
deterministic fake (docs/adar/TESTING.md).

What is real: every line of adar/protocol.py, wire.py, pairing.py, node.py, coordinator.py, scheduler.py
and transfer.py — the same code a LAN deployment will run. What is simulated: the clock, the links
(latency, jitter, loss, corruption, bandwidth, partitions), the identities' cryptography (adar/identity.py's
"sim" backend) and the work itself — a task "runs" by a timer, never by loading a model. Ten simulated
nodes cost ten small Python objects; nothing here imports MLX or the engine.

Everything simulated says so: `Cluster.describe()` and the demo below label every node SIMULATED, and each
node's capability document carries `"simulated": true`.

    python -m adar.sim            a short scripted run, printed

Determinism: one seeded `random.Random` drives every network effect and every pairing code, and events at
the same instant run in the order they were scheduled — so a failing test fails the same way every time.
"""
import heapq
import random

from . import identity, protocol, transfer
from .coordinator import Coordinator
from .node import Node

LABEL = "SIMULATED"


class _Timer:
    __slots__ = ("cancelled",)

    def __init__(self):
        self.cancelled = False

    def cancel(self):
        self.cancelled = True


class SimNet:
    """A clock and a set of links. Implements the three-method transport adar/node.py asks for."""

    def __init__(self, seed=0, *, latency=0.001, jitter=0.0, loss=0.0, corrupt=0.0, bps=1e9):
        self.t = 0.0
        self.rng = random.Random(seed)
        self._q, self._n = [], 0
        self.endpoints = {}
        self.default = {"latency": latency, "jitter": jitter, "loss": loss, "corrupt": corrupt, "bps": bps, "up": True}
        self.links = {}
        self._free_at = {}                 # (src, dst) -> when that direction of the link is free again
        self.stats = {"sent": 0, "delivered": 0, "dropped": 0, "corrupted": 0, "bytes": 0}
        self.tap = None                    # optional fn(src, dst, raw): a wire-level observer (an "attacker")
        self.blocked = set()               # (src host, dst host): one direction of a link silently eating traffic

    # ───────────── transport interface ─────────────
    def now(self):
        return self.t

    def after(self, delay, fn):
        timer = _Timer()
        self._n += 1
        heapq.heappush(self._q, (self.t + max(0.0, delay), self._n, timer, fn))
        return timer

    def send(self, src, dst, raw):
        self.stats["sent"] += 1
        self.stats["bytes"] += len(raw)
        if self.tap:
            self.tap(src, dst, raw)
        link = self.link_of(src, dst)
        if (dst not in self.endpoints or not link["up"] or (self._host(src), self._host(dst)) in self.blocked
                or self.rng.random() < link["loss"]):
            self.stats["dropped"] += 1
            return
        if link["corrupt"] and self.rng.random() < link["corrupt"]:
            i = self.rng.randrange(len(raw))
            raw = raw[:i] + bytes([raw[i] ^ 0x5A]) + raw[i + 1:]
            self.stats["corrupted"] += 1
        # one direction of a link carries one message at a time: that is what makes bandwidth a limit
        start = max(self.t, self._free_at.get((src, dst), 0.0))
        done = start + len(raw) * 8 / link["bps"]
        self._free_at[(src, dst)] = done
        arrive = done + link["latency"] + (self.rng.uniform(0, link["jitter"]) if link["jitter"] else 0.0)
        self.after(arrive - self.t, lambda: self._deliver(src, dst, raw))

    def _deliver(self, src, dst, raw):
        ep = self.endpoints.get(dst)
        if ep is None or not self.link_of(src, dst)["up"]:
            self.stats["dropped"] += 1
            return
        self.stats["delivered"] += 1
        ep.handle(src, raw)

    # ───────────── the network itself ─────────────
    def attach(self, addr, endpoint):
        self.endpoints[addr] = endpoint

    @staticmethod
    def _host(addr):
        return addr.split("#", 1)[0]       # "mini#data" is another channel of the host "mini"

    def link_of(self, a, b):
        return self.links.get(frozenset((self._host(a), self._host(b))), self.default)

    def link(self, a, b, **changes):
        """Set properties of the link between two hosts: latency, jitter, loss, corrupt, bps, up."""
        key = frozenset((a, b))
        self.links[key] = {**self.links.get(key, self.default), **changes}

    def partition(self, a, b):
        self.link(a, b, up=False)

    def heal(self, a, b):
        self.link(a, b, up=True)

    def block(self, src, dst):
        """Drop what `src` sends to `dst`, and only that: a request that arrives whose answer does not."""
        self.blocked.add((src, dst))

    def unblock(self, src, dst):
        self.blocked.discard((src, dst))

    def run_for(self, seconds, max_events=1_000_000):
        """Advance the clock by `seconds`, running everything that falls due, in order."""
        end = self.t + seconds
        for _ in range(max_events):
            if not self._q or self._q[0][0] > end:
                break
            t, _n, timer, fn = heapq.heappop(self._q)
            self.t = t
            if not timer.cancelled:
                fn()
        else:
            raise RuntimeError("simulation did not settle: an event storm, or max_events is too low")
        self.t = end


class _Pipe:
    """One end of a data-plane channel: hands frames to a Sender or a Receiver and sends back what it says."""

    def __init__(self, net, addr, peer, on_frame, delay=0.0):
        self.net, self.addr, self.peer, self.on_frame, self.delay = net, addr, peer, on_frame, delay
        net.attach(addr, self)

    def handle(self, src, raw):
        if not transfer.is_frame(raw):
            return
        out = self.on_frame(raw)
        frames = [] if out is None else (out if isinstance(out, list) else [out])
        if frames:
            self.net.after(self.delay, lambda: [self.net.send(self.addr, self.peer, f) for f in frames])


def run_transfer(net, src, dst, read, total, write, *, chunk=transfer.CHUNK, window=8, start=0,
                 prefix_digest=None, rto=0.5, consume_s=0.0, limit_s=600.0, tid=None):
    """Move `total` bytes from host `src` to host `dst` over the simulated link between them, on its own
    channel. Returns (sender, receiver, seconds taken). Stops at `limit_s` if the link never lets it finish."""
    tid = tid or bytes(net.rng.getrandbits(8) for _ in range(16))
    a, b = f"{src}#data", f"{dst}#data"
    dig = (lambda: prefix_digest.copy()) if prefix_digest is not None else (lambda: None)
    snd = transfer.Sender(tid, read, total, chunk=chunk, window=window, start=start, prefix_digest=dig())
    rcv = transfer.Receiver(tid, write, total, chunk=chunk, have=start, prefix_digest=dig())
    _Pipe(net, a, b, snd.on_frame)
    _Pipe(net, b, a, rcv.on_frame, delay=consume_s)
    t0 = net.now()
    state = {"base": -1}

    def tick():
        if snd.done or rcv.failed or net.now() - t0 > limit_s:
            return
        if snd.base == state["base"]:              # no progress since the last look: say it again
            for f in snd.resend():
                net.send(a, b, f)
        state["base"] = snd.base
        net.after(rto, tick)

    for f in snd.frames():
        net.send(a, b, f)
    net.after(rto, tick)
    while not (snd.done or rcv.failed) and net.now() - t0 <= limit_s:
        net.run_for(rto)
    return snd, rcv, net.now() - t0


class Cluster:
    """A coordinator and some workers on one SimNet, with the human steps (pairing mode, reading a code off
    the other screen, granting permissions) scripted."""

    def __init__(self, seed=0, *, label="coordinator", **coordinator_kw):
        self.net = SimNet(seed)
        self.seed = seed
        self.nodes = {}
        self.coordinator = self._make(Coordinator, label, coordinator_kw)

    def _make(self, cls, label, kw):
        ident = identity.load_or_create("", simulated=True, seed=f"{self.seed}:{label}")
        node = cls(ident, label, self.net, label=label, simulated=True,
                   rng=random.Random(f"{self.seed}:{label}:codes"), **kw)
        self.net.attach(label, node)
        self.nodes[label] = node
        return node

    def add(self, label, **kw):
        return self._make(Node, label, kw)

    def fingerprint(self, label):
        return self.nodes[label].identity.fingerprint

    def pair(self, label, grant=("echo.test",), code=None):
        """Discovery, the human-verified pairing, and a permission grant. `code` overrides what the person
        types (to get it wrong on purpose)."""
        node, c = self.nodes[label], self.coordinator
        node.pairing.accepting = True                      # a person put the node into pairing mode
        if c.discover(node.advert()) is None:
            return False
        shown = len(node.codes)
        c.pair(node.identity.fingerprint)
        self._until(lambda: len(node.codes) > shown and c.pairing.out is not None and c.pairing.out["peer"] is not None)
        if len(node.codes) == shown or c.pairing.out is None or c.pairing.out["peer"] is None:
            return False
        c.enter_code(code if code is not None else node.codes[-1])
        both = lambda: c.book.is_paired(node.identity.node_id) and node.book.is_paired(c.identity.node_id)  # noqa: E731
        self._until(both, limit=2.0 if code is not None else 10.0)
        node.pairing.accepting = False
        ok = both()
        if ok:
            c.nodes[node.identity.fingerprint]["label"] = label
            # the permission is the worker's to give: its own book is what it checks a task against. The
            # coordinator's copy is only what it believes it was given (to avoid offering what will be refused)
            node.book.grant(c.identity.node_id, grant)
            c.grant(node.identity.fingerprint, grant)
            rec = c.nodes[node.identity.fingerprint]
            self._until(lambda: rec["state"] in ("online", "busy") and rec["capability"] is not None)
        return ok

    def _until(self, done, limit=10.0, step=0.05):
        """Let simulated time pass until `done()` — however slow the link — or `limit` seconds."""
        t0 = self.net.now()
        while not done() and self.net.now() - t0 < limit:
            self.net.run_for(step)

    def profile(self, label, payload=256 * 1024):
        """Measure the link's payload throughput with a bounded synthetic transfer, and tell the scheduler."""
        blob = bytes(payload)
        sink = bytearray(payload)

        def write(off, data):
            sink[off:off + len(data)] = data
        _s, rcv, took = run_transfer(self.net, self.coordinator.addr, label, lambda o, n: blob[o:o + n],
                                     payload, write, chunk=16 * 1024)
        if rcv.complete and took > 0:
            self.coordinator.set_link(self.fingerprint(label), payload * 8 / took)
        return rcv.complete

    def describe(self):
        lines = []
        for label, n in self.nodes.items():
            role = "coordinator" if n is self.coordinator else "worker"
            rec = None if n is self.coordinator else self.coordinator.nodes.get(n.identity.fingerprint)
            lines.append(f"[{LABEL}] {label:<12} {role:<11} {n.identity.fingerprint}  "
                         f"{n.memory_gb:.0f} GB  models: {', '.join(n.models) or '-'}"
                         + (f"  state: {rec['state']}" if rec else ""))
        return lines



def task(model=None, *, work_s=0.1, tokens=100, remote=True, memory_gb=1.0, size=2_000, payload="x", **kw):
    """An `echo.test` task with the estimates the scheduler reads."""
    t = protocol.new_task("echo.test", {"payload": payload, "work_s": work_s}, model=model,
                          peak_memory_gb=memory_gb, remote_execution_allowed=remote, **kw)
    t["resource_estimate"].update(work_tokens=tokens, input_bytes=size, output_bytes=size)
    return t


def main():
    c = Cluster(seed=7, models=["small"], available_gb=10.0)
    c.coordinator.set_estimate("local", "small", tps=40, load_s=1.0)
    c.add("mini", models=["small"], available_gb=10.0)
    c.add("studio", models=["small", "large"], memory_gb=64.0, available_gb=48.0)
    for label, tps in (("mini", 40), ("studio", 160)):
        print(f"[{LABEL}] pairing {label}: {'ok' if c.pair(label) else 'FAILED'}")
        c.profile(label)
        c.coordinator.set_estimate(c.nodes[label].identity.node_id, "small", tps=tps, load_s=1.0)
    c.coordinator.set_estimate(c.nodes["studio"].identity.node_id, "large", tps=30, load_s=4.0)
    print("\n".join(c.describe()))
    jobs = {"a short task": task("small", tokens=20, work_s=0.5),
            "a long task": task("small", tokens=2000, work_s=12.5),
            "needs the large model": task("large", tokens=300, work_s=10.0, memory_gb=30.0),
            "local-only": task("small", tokens=2000, work_s=50.0, remote=False),
            "a model nobody has": task("huge", tokens=10)}
    ids = {name: c.coordinator.submit(t) for name, t in jobs.items()}
    c.net.run_for(120)
    for name, tid in ids.items():
        print(f"[{LABEL}] {name:<22} {c.coordinator.state(tid):<18} {c.coordinator.explain(tid)[-1]}")
    s = c.net.stats
    print(f"[{LABEL}] network: {s['sent']} messages, {s['bytes']} bytes, {s['dropped']} dropped — "
          f"simulated clock at {c.net.now():.1f}s; no model was loaded")


if __name__ == "__main__":
    main()
