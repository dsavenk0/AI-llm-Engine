"""A node: an identity, a book of paired peers, and a worker that runs allowlisted tasks for them.

Everything here is transport-agnostic. A node is given something with three methods —

    now() -> float                    seconds
    send(src_addr, dst_addr, raw)     best effort, like a network
    after(delay_s, fn) -> timer       call fn later; timer.cancel() stops it

— and never learns whether that is a socket and a clock or adar/sim.py's simulated ones. That is what lets
Phase 2 run the *real* protocol code against a deterministic fake network, and what Phase 3 replaces.

What a node does for a peer is bounded by three independent checks, in this order: the message is from a
paired identity and is not a replay (wire.py); the pairing was granted this task type (PeerBook.scope);
the type is one this node ships a handler for (protocol.HANDLERS — registering anything else is refused,
so there is no path by which a peer supplies code to run).
"""
from . import capability, protocol, wire
from .pairing import Pairing, PeerBook
from .protocol import ProtocolError


def echo_handler(task_input, node):
    """echo.test: gives the input back after `work_s` simulated seconds."""
    return float(task_input.get("work_s", 0.0)), {"echo": task_input.get("payload")}


class Node:
    def __init__(self, identity, addr, transport, *, label=None, models=(), memory_gb=16.0, available_gb=12.0,
                 tps=None, load_s=1.0, simulated=True, rng=None, show_code=None):
        self.identity, self.addr, self.net = identity, addr, transport
        self.label = label or addr
        self.simulated = bool(simulated)
        self.models = list(models)
        self.loaded = []                      # models warm in memory (most recent last)
        self.memory_gb, self.available_gb = memory_gb, available_gb
        self.tps = dict(tps or {})            # model -> tokens/s this node really runs at (simulation input)
        self.load_s = load_s
        self.alive = True
        self.book = PeerBook()
        self.codes = []                       # what this node's "screen" has shown
        self.pairing = Pairing(identity, self.book, rng=rng,
                               show_code=show_code or (lambda code, peer: self.codes.append(code)))
        self.handlers = {}
        self.register("echo.test", echo_handler)
        self.tasks = {}                       # task_id -> record
        self.by_key = {}                      # idempotency_key -> task_id
        self.queue = []                       # task ids waiting for the worker
        self.active = None
        self.executions = 0                   # how many times a handler really ran
        self.refused = []                     # (addr, reason) of everything turned away — the audit trail
        self.audit = []

    # ───────────── setup ─────────────
    def register(self, task_type, fn):
        if task_type not in protocol.HANDLERS:
            raise ProtocolError(f"{task_type!r} is not an allowlisted task type ({', '.join(protocol.HANDLERS)})")
        self.handlers[task_type] = fn

    def advert(self):
        """What discovery carries: a fingerprint and a protocol version, nothing a stranger could use."""
        return {"v": protocol.VERSION, "fingerprint": self.identity.fingerprint, "addr": self.addr,
                "instance": self.identity.node_id[-12:]}

    def capability(self):
        return capability.build(self.identity, models=self.models, handlers=sorted(self.handlers),
                                active_tasks=1 if self.active else 0, queue_depth=len(self.queue),
                                simulated=self.simulated, memory_gb=self.memory_gb,
                                available_gb=self.available_gb, loaded_models=self.loaded)

    def log(self, action, **meta):
        """Metadata only — never task input or output (SECURITY.md, audit log)."""
        self.audit.append({"t": round(self.net.now(), 6), "node": self.label, "action": action, **meta})

    # ───────────── sending ─────────────
    def send(self, node_id, verb, body):
        peer = self.book.peers.get(node_id)
        if peer is None:
            raise ProtocolError(f"{verb}: not paired with {node_id}")
        raw = wire.seal(self.identity, node_id, self.book.next_seq(node_id), verb, body)
        self.net.send(self.addr, peer["addr"], raw)

    def _plain(self, addr, verb, body):
        self.net.send(self.addr, addr, protocol.encode(verb, body))

    # ───────────── receiving ─────────────
    def handle(self, src, raw):
        """Entry point for everything that arrives. Whatever is wrong with a message, the outcome is a
        recorded refusal — never an exception escaping into the transport, never a partial action."""
        if not self.alive:
            return
        try:
            verb, body = wire.peek(raw)
            if verb in wire.UNSEALED:
                return self._on_pairing(src, verb, body)
            if body.get("from") in self.book.revoked:
                raise ProtocolError(f"{verb}: sender's pairing was revoked")
            sender, seq, data = wire.unseal(self.identity, verb, body, self.book)
            self.book.peers[sender]["addr"] = src            # a node's address may change; its key does not
            self.on_message(sender, verb, data)
        except ProtocolError as e:
            self.refused.append((src, str(e)))
            self.log("refused", reason=str(e))

    def _on_pairing(self, src, verb, body):
        now = self.net.now()
        if verb == "pair.request":
            self._plain(src, "pair.confirm", self.pairing.on_request(src, body, now))
            self.log("pair.challenge", peer=body["node_id"])
        elif body.get("step") == "challenge":
            self.pairing.on_challenge(src, body)
        elif body.get("step") == "proof":
            peer, reply = self.pairing.on_proof(src, body, now)
            self._plain(src, "pair.confirm", reply)
            self.log("paired", peer=peer)
            self.on_paired(peer)
        elif body.get("step") == "accept":
            peer = self.pairing.on_accept(src, body, now)
            self.log("paired", peer=peer)
            self.on_paired(peer)
        else:
            raise ProtocolError("pair.confirm: unknown step")

    def on_paired(self, peer):
        self.send(peer, "capability.push", self.capability())

    def on_message(self, sender, verb, data):
        if verb == "health.ping":
            self.send(sender, "health.pong", {"n": data.get("n")})
        elif verb == "task.submit":
            self._on_submit(sender, data)
        elif verb == "task.cancel":
            self._on_cancel(sender, data)
        elif verb == "pair.revoke":
            self.book.revoke(sender)
            self.log("revoked_by_peer", peer=sender)
        elif verb in ("capability.push", "health.pong", "task.accept", "task.reject", "task.status"):
            pass                                             # a plain worker has no use for these
        else:
            raise ProtocolError(f"{verb}: not handled by this node")

    # ───────────── pairing, the human side ─────────────
    def pair_with(self, addr):
        self._plain(addr, "pair.request", self.pairing.begin(addr, self.net.now()))

    def enter_code(self, code):
        addr = self.pairing.out["addr"] if self.pairing.out else None
        self._plain(addr, "pair.confirm", self.pairing.prove(code))

    def revoke(self, node_id):
        if self.book.is_paired(node_id):
            try:
                self.send(node_id, "pair.revoke", {})
            except ProtocolError:
                pass
        self.book.revoke(node_id)
        self.log("revoked", peer=node_id)

    # ───────────── the worker ─────────────
    def _status(self, rec, **extra):
        if rec["origin"] is not None and self.alive:
            self.send(rec["origin"], "task.status", {"task_id": rec["task"]["task_id"], "state": rec["state"], **extra})

    def admit(self, task):
        """None if this node can run `task` now, else why not. Shared by remote and local submissions."""
        if task["type"] not in self.handlers:
            return f"no {task['type']} handler on this node"
        need = task["requirements"].get("model")
        if need and need not in self.models:
            return f"missing checkpoint {need}"
        mem = float(task["resource_estimate"].get("peak_memory_gb") or 0.0)
        if mem > self.available_gb:
            return f"needs {mem:.1f} GB, {self.available_gb:.1f} GB available"
        return None

    def _on_submit(self, sender, task):
        try:
            protocol.validate_task(task)
            if not task.get("remote_execution_allowed"):
                raise ProtocolError("task is local-only: it must not run on another node")
            if task["type"] not in self.book.peers[sender]["scope"]:
                raise ProtocolError(f"this pairing does not permit {task['type']}")
        except ProtocolError as e:
            self.log("task.reject", task=task.get("task_id"), reason=str(e))
            return self.send(sender, "task.reject", {"task_id": task.get("task_id"), "reason": str(e)})
        known = self.by_key.get(task["idempotency_key"])
        if known is not None:                        # a retry of something already here: answer, do not redo
            rec = self.tasks[known]
            self.log("task.duplicate", task=known)
            extra = {"result": rec["result"]} if rec["state"] == "completed" else {}
            if rec.get("error"):
                extra["error"] = rec["error"]
            return self.send(sender, "task.status", {"task_id": known, "state": rec["state"], **extra})
        why = self.admit(task)
        if why:
            self.log("task.reject", task=task["task_id"], reason=why)
            return self.send(sender, "task.reject", {"task_id": task["task_id"], "reason": why})
        self.enqueue(task, sender)
        self.send(sender, "task.accept", {"task_id": task["task_id"]})

    def enqueue(self, task, origin, on_change=None):
        rec = {"task": task, "origin": origin, "state": "accepted", "result": None, "error": None,
               "timer": None, "on_change": on_change}
        self.tasks[task["task_id"]] = rec
        self.by_key[task["idempotency_key"]] = task["task_id"]
        self.queue.append(task["task_id"])
        self.log("task.accept", task=task["task_id"], type=task["type"])
        self._pump()
        return rec

    def _move(self, rec, state, **extra):
        rec["state"] = state
        if rec["on_change"]:
            rec["on_change"](rec, extra)
        self._status(rec, **extra)

    def _pump(self):
        if self.active is not None or not self.queue or not self.alive:
            return
        rec = self.tasks[self.queue.pop(0)]
        self.active = rec
        need = rec["task"]["requirements"].get("model")
        if need and need not in self.loaded:
            self._move(rec, "loading")
            rec["timer"] = self.net.after(self.load_s, lambda: self._loaded(rec, need))
        else:
            self._run(rec)

    def _loaded(self, rec, model):
        self.loaded.append(model)
        self._run(rec)

    def _run(self, rec):
        self._move(rec, "running")
        self.executions += 1
        try:
            duration, result = self.handlers[rec["task"]["type"]](rec["task"]["input"], self)
        except Exception as e:                               # a handler's failure is the task's, not the node's
            return self._finish(rec, "failed", error=f"{type(e).__name__}: {e}")
        rec["timer"] = self.net.after(duration, lambda: self._finish(rec, "completed", result=result))

    def _finish(self, rec, state, result=None, error=None):
        rec["timer"] = None
        rec["result"], rec["error"] = result, error
        if self.active is rec:
            self.active = None
        self.log("task." + state, task=rec["task"]["task_id"])
        extra = {"result": result} if state == "completed" else ({"error": error} if error else {})
        self._move(rec, state, **extra)
        self._pump()

    def _on_cancel(self, sender, data):
        rec = self.tasks.get(data.get("task_id"))
        if rec is None or rec["origin"] != sender:
            raise ProtocolError("task.cancel: no such task from this node")
        self.cancel_local(rec)

    def cancel_local(self, rec):
        tid = rec["task"]["task_id"]
        if rec["state"] in protocol.TERMINAL:                # too late: the result stands, say so again
            extra = {"result": rec["result"]} if rec["state"] == "completed" else {}
            return self._status(rec, **extra)
        if tid in self.queue:
            self.queue.remove(tid)
        if rec["timer"] is not None:
            rec["timer"].cancel()
        self._finish(rec, "cancelled")

    # ───────────── failure injection ─────────────
    def crash(self):
        """The process dies: timers stop, everything in memory is gone."""
        self.alive = False
        for rec in self.tasks.values():
            if rec["timer"] is not None:
                rec["timer"].cancel()
        self.tasks, self.by_key, self.queue, self.active, self.loaded = {}, {}, [], None, []

    def restart(self):
        self.alive = True
