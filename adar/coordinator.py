"""The coordinator: a node that also keeps a registry of the others, watches their health, and decides
where each task runs (docs/adar/SCHEDULER.md; components 2, 5, 8, 12 and 14 of ARCHITECTURE.md).

It is a `Node` first — it has an identity, pairs like any other, and runs tasks itself: "run locally" is
always a candidate, never assumed worse because another machine exists.

Three state machines live here, each with its transitions written out and enforced:

* a **node** moves `discovered → awaiting_pairing → paired → online ⇄ busy → degraded → offline`, and to
  `revoked` from anywhere. Health is heartbeats: one missed pong changes nothing (no flapping), two make a
  node `degraded` (it gets no new work), four make it `offline`; an offline node is still pinged, at an
  interval that doubles up to a cap, and one pong brings it back.
* a **task** moves through protocol.NEXT. When its node goes offline mid-task it becomes `interrupted →
  recovery_pending` and, if its retry policy allows, `queued` again — with the same idempotency key, so a
  node that did finish it and merely lost its way back answers instead of running it twice.
* every placement stores the sentence that explains it; `explain(task_id)` returns them in order.
"""
from . import capability, protocol, scheduler
from .identity import fingerprint_of
from .node import Node
from .protocol import ProtocolError

NODE_NEXT = {
    "discovered": ("awaiting_pairing", "revoked"),
    "awaiting_pairing": ("paired", "discovered", "revoked"),
    "paired": ("online", "offline", "revoked"),
    "online": ("busy", "degraded", "offline", "revoked"),
    "busy": ("online", "degraded", "offline", "revoked"),
    "degraded": ("online", "busy", "offline", "revoked"),
    "offline": ("online", "busy", "revoked"),
    "revoked": ("awaiting_pairing",),          # only a new, human-verified pairing leads back
}

HEARTBEAT_S = 1.0
DEGRADED_AFTER = 2           # consecutive missed pongs
OFFLINE_AFTER = 4
BACKOFF_MAX_S = 30.0
RESEND_S = 3.0               # a dispatched task with no word from its node is offered again (same key)
REMOTE = ("dispatched", "accepted", "loading", "running", "returning_result", "cancelling")


class Coordinator(Node):
    def __init__(self, *a, local_worker=True, **kw):
        super().__init__(*a, **kw)
        self.local_worker = local_worker
        self.nodes = {}          # fingerprint -> record (before pairing a node has no trusted id)
        self.jobs = {}           # task_id -> record
        self.keys = {}           # idempotency_key -> task_id
        self.estimates = {}      # (node_id | "local", model) -> {"tps", "load_s", "source"}
        self.conflicts = []      # adverts refused, with the reason

    # ───────────── discovery ─────────────
    def discover(self, advert):
        """An unauthenticated advertisement was seen. It earns a row in the registry and nothing else."""
        fp, addr = advert.get("fingerprint"), advert.get("addr")
        if advert.get("v") != protocol.VERSION:
            return self._conflict(advert, f"speaks protocol v{advert.get('v')!r}, this node speaks v{protocol.VERSION}")
        if not isinstance(fp, str) or not isinstance(addr, str):
            return self._conflict(advert, "malformed advertisement")
        if fp == self.identity.fingerprint:
            return self._conflict(advert, "advertises this node's own fingerprint")
        known = self.nodes.get(fp)
        if known is not None and (known["addr"] != addr or known["instance"] != advert.get("instance")):
            # the same key from two places is a clone or a replay; neither is taken on trust, and the
            # node already known is not silently replaced by the newcomer
            known["duplicate"] = True
            return self._conflict(advert, f"fingerprint {fp} is already advertised from {known['addr']}")
        if known is None:
            self.nodes[fp] = {"fingerprint": fp, "addr": addr, "instance": advert.get("instance"),
                              "state": "discovered", "node_id": None, "label": addr, "capability": None,
                              "rtt_s": None, "bps": None, "misses": 0, "ping_n": 0, "pings": {}, "heard": False,
                              "interval": HEARTBEAT_S, "timer": None, "duplicate": False, "history": ["discovered"]}
        return self.nodes[fp]

    def _conflict(self, advert, reason):
        self.conflicts.append((advert.get("addr"), reason))
        self.log("discovery.refused", reason=reason)
        return None

    def _set_state(self, rec, state):
        if rec["state"] == state:
            return
        if state not in NODE_NEXT[rec["state"]]:
            raise ProtocolError(f"illegal node transition {rec['state']} → {state}")
        rec["state"] = state
        rec["history"].append(state)
        self.log("node." + state, peer=rec["fingerprint"])

    def _by_id(self, node_id):
        return next((r for r in self.nodes.values() if r["node_id"] == node_id), None)

    # ───────────── pairing ─────────────
    def pair(self, fingerprint):
        rec = self.nodes.get(fingerprint)
        if rec is None:
            raise ProtocolError(f"no discovered node with fingerprint {fingerprint}")
        if rec["duplicate"]:
            raise ProtocolError(f"fingerprint {fingerprint} is advertised from more than one place — refusing to pair")
        self._set_state(rec, "awaiting_pairing")
        self.pair_with(rec["addr"])

    def on_paired(self, peer):
        rec = next((r for r in self.nodes.values() if r["addr"] == self.book.peers[peer]["addr"]), None)
        if rec is None:
            return
        if fingerprint_of(peer) != rec["fingerprint"]:
            self.book.revoke(peer)                 # the key that paired is not the key that was advertised
            self._set_state(rec, "discovered")
            raise ProtocolError("pairing refused: the node's key does not match its advertised fingerprint")
        rec["node_id"] = peer
        self._set_state(rec, "paired")
        self._beat(rec)

    def grant(self, fingerprint, task_types):
        self.book.grant(self.nodes[fingerprint]["node_id"], task_types)
        self.log("grant", peer=fingerprint, types=sorted(task_types))

    def revoke_node(self, fingerprint):
        rec = self.nodes[fingerprint]
        if rec["timer"] is not None:
            rec["timer"].cancel()
        if rec["node_id"]:
            self.revoke(rec["node_id"])
            self._lost(rec, "node revoked")
        self._set_state(rec, "revoked")

    # ───────────── health ─────────────
    def _beat(self, rec):
        if rec["state"] == "revoked" or not self.alive:
            return
        now = self.net.now()
        if rec["pings"] and not rec["heard"]:                # a whole interval with no answer to anything
            rec["misses"] += 1
            if rec["misses"] >= OFFLINE_AFTER and rec["state"] != "offline":
                self._set_state(rec, "offline")
                self._lost(rec, "node went offline")
            elif DEGRADED_AFTER <= rec["misses"] < OFFLINE_AFTER and rec["state"] in ("online", "busy"):
                self._set_state(rec, "degraded")
            if rec["state"] == "offline":                    # bounded exponential backoff
                rec["interval"] = min(BACKOFF_MAX_S, rec["interval"] * 2)
        rec["heard"] = False
        rec["ping_n"] += 1
        # a pong counts for as long as its ping is remembered: a link slower than the heartbeat interval
        # answers ping n after ping n+1 has gone out, and that is a slow node, not a dead one
        horizon = now - OFFLINE_AFTER * max(rec["interval"], HEARTBEAT_S)
        rec["pings"] = {n: t for n, t in rec["pings"].items() if t >= horizon}
        rec["pings"][rec["ping_n"]] = now
        self.send(rec["node_id"], "health.ping", {"n": rec["ping_n"]})
        rec["timer"] = self.net.after(rec["interval"], lambda: self._beat(rec))

    def _on_pong(self, rec, data):
        sent = rec["pings"].pop(data.get("n"), None)
        if sent is None:
            return                                           # not an answer to anything still outstanding
        rec["rtt_s"] = self.net.now() - sent
        rec["heard"], rec["misses"], rec["interval"] = True, 0, HEARTBEAT_S
        if rec["state"] in ("paired", "degraded", "offline"):
            self._set_state(rec, self._busy_state(rec))
            self._drain()

    def _busy_state(self, rec):
        doc = rec["capability"]
        busy = bool(doc and (capability.value(doc, "dynamic", "active_tasks") or
                             capability.value(doc, "dynamic", "queue_depth")))
        return "busy" if busy else "online"

    def set_link(self, fingerprint, bps):
        """A measured payload throughput for the link to this node (the profiler's result; see sim.py)."""
        self.nodes[fingerprint]["bps"] = float(bps)
        self._drain()

    def set_estimate(self, where, model, tps, load_s=0.0, source="estimated"):
        self.estimates[(where, model)] = {"tps": float(tps), "load_s": float(load_s), "source": source}
        self._drain()

    # ───────────── messages from nodes ─────────────
    def on_message(self, sender, verb, data):
        rec = self._by_id(sender)
        if rec is None:
            return super().on_message(sender, verb, data)
        if verb == "health.pong":
            return self._on_pong(rec, data)
        if verb == "capability.push":
            capability.check(data)
            if data.get("node_id") != sender:
                raise ProtocolError("capability.push: document describes a different node")
            rec["capability"] = data
            if rec["state"] in ("online", "busy"):
                self._set_state(rec, self._busy_state(rec))
            return self._drain()
        if verb in ("task.accept", "task.reject", "task.status"):
            return self._on_task_message(rec, verb, data)
        return super().on_message(sender, verb, data)

    # ───────────── tasks ─────────────
    def submit(self, task, *, deadline_s=None):
        """Take a task. Returns its id — the existing one if this idempotency key was already submitted."""
        protocol.validate_task(task)
        if task["idempotency_key"] in self.keys:
            return self.keys[task["idempotency_key"]]
        job = {"task": task, "node": None, "attempts": 0, "explain": [], "result": None, "error": None,
               "resend": None, "deadline": None, "local": None}
        self.jobs[task["task_id"]] = job
        self.keys[task["idempotency_key"]] = task["task_id"]
        protocol.advance(task, "validated")
        protocol.advance(task, "queued")
        if deadline_s is not None:
            job["deadline"] = self.net.after(deadline_s, lambda: self._deadline(job))
        self._place(job)
        return task["task_id"]

    def explain(self, task_id):
        return list(self.jobs[task_id]["explain"])

    def state(self, task_id):
        return self.jobs[task_id]["task"]["state"]

    def _views(self, model):
        out = []
        for rec in self.nodes.values():
            doc, nid = rec["capability"], rec["node_id"]
            if nid is None or doc is None or not self.book.is_paired(nid):
                continue
            est = self.estimates.get((nid, model), {})
            out.append(scheduler.View(
                node_id=nid, label=rec["label"], state=rec["state"],
                handlers=tuple(capability.value(doc, "static", "task_handlers")),
                scope=tuple(self.book.peers[nid]["scope"]),
                installed=tuple(capability.value(doc, "static", "installed_models")),
                loaded=tuple(capability.value(doc, "dynamic", "loaded_models")),
                available_gb=capability.value(doc, "dynamic", "available_memory_gb"),
                busy=(capability.value(doc, "dynamic", "active_tasks") or 0)
                + (capability.value(doc, "dynamic", "queue_depth") or 0)
                + sum(1 for j in self.jobs.values() if j["node"] == nid and j["task"]["state"] == "dispatched"),
                tps=est.get("tps"), load_s=est.get("load_s"), rtt_s=rec["rtt_s"], bps=rec["bps"]))
        return out

    def _local_view(self, model):
        if not self.local_worker:
            return None
        est = self.estimates.get(("local", model), {})
        return scheduler.View(node_id="local", label=self.label, handlers=tuple(self.handlers),
                              installed=tuple(self.models), loaded=tuple(self.loaded),
                              available_gb=self.available_gb, busy=(1 if self.active else 0) + len(self.queue),
                              tps=est.get("tps"), load_s=est.get("load_s", self.load_s), local=True)

    def _place(self, job):
        task = job["task"]
        if task["state"] in ("waiting_for_model", "waiting_for_resources"):
            protocol.advance(task, "queued")
        model = task["requirements"].get("model")
        d = scheduler.decide(task, self._local_view(model), self._views(model))
        if not job["explain"] or job["explain"][-1] != d.explanation:
            job["explain"].append(d.explanation)
        self.log("task.placed", task=task["task_id"], target=d.target or d.wait or "rejected")
        if d.target is None:
            if d.wait:
                return protocol.advance(task, d.wait)        # parked; _drain() looks again when things change
            job["error"] = d.explanation
            return protocol.advance(task, "rejected")
        protocol.advance(task, "scheduled")
        job["attempts"] += 1
        if d.target == "local":
            job["node"] = "local"
            for s in ("dispatched", "accepted"):
                protocol.advance(task, s)
            job["local"] = self.enqueue(task, None, on_change=lambda rec, extra: self._progress(job, rec["state"], extra))
            return None
        job["node"] = d.target
        protocol.advance(task, "dispatched")
        self._offer(job)
        return None

    def _offer(self, job):
        wire_task = dict(job["task"], state="dispatched")
        self.send(job["node"], "task.submit", wire_task)
        job["resend"] = self.net.after(RESEND_S, lambda: self._resend(job))

    def _resend(self, job):
        job["resend"] = None
        if job["task"]["state"] in REMOTE and job["node"] not in (None, "local"):
            rec = self._by_id(job["node"])
            if rec is not None and rec["state"] != "offline":
                self._offer(job)                             # same task, same key: the node answers, not reruns

    def _drain(self):
        """Something changed (a node came back, a capability arrived): look again at what is parked."""
        for job in list(self.jobs.values()):
            if job["task"]["state"] in ("waiting_for_model", "waiting_for_resources"):
                self._place(job)

    def _on_task_message(self, rec, verb, data):
        job = self.jobs.get(data.get("task_id"))
        if job is None or job["node"] != rec["node_id"]:
            raise ProtocolError(f"{verb}: no such task on this node")       # a node speaks only for its own tasks
        if verb == "task.accept":
            return self._progress(job, "accepted", {})
        if verb == "task.reject":
            job["explain"].append(f"Rejected by {rec['label']}: {data.get('reason')}.")
            if job["task"]["state"] == "dispatched":
                self._stop_timers(job, keep_deadline=True)
                job["error"] = str(data.get("reason"))
                protocol.advance(job["task"], "rejected")
            return None
        return self._progress(job, data.get("state"), data)

    # the order a task's remote life runs in; a status may arrive having skipped some (lost or late)
    _ORDER = ("dispatched", "accepted", "loading", "running", "returning_result", "completed")

    def _progress(self, job, state, extra):
        task = job["task"]
        if task["state"] in protocol.TERMINAL:
            return                                           # late news about a finished task changes nothing
        if state == "cancelled":
            if task["state"] != "cancelling":
                protocol.advance(task, "cancelling")
            self._stop_timers(job)
            return protocol.advance(task, "cancelled")
        if state == "failed":
            job["error"] = extra.get("error") or "the node reported a failure"
            self._stop_timers(job)
            return protocol.advance(task, "failed")
        if state == "completed":
            job["result"] = extra.get("result")
            self._stop_timers(job)
            if task["state"] == "cancelling":                # it finished before the cancel reached it
                return protocol.advance(task, "completed")
            for s in self._ORDER[self._ORDER.index(task["state"]) + 1:] if task["state"] in self._ORDER else ():
                if s == "loading":
                    continue
                protocol.advance(task, s)
            return None
        if state in self._ORDER and task["state"] in self._ORDER:
            have, want = self._ORDER.index(task["state"]), self._ORDER.index(state)
            for s in self._ORDER[have + 1:want + 1]:
                if s == "loading" and state != "loading":
                    continue
                protocol.advance(task, s)
        return None

    def _stop_timers(self, job, keep_deadline=False):
        for k in ("resend",) + (() if keep_deadline else ("deadline",)):
            if job[k] is not None:
                job[k].cancel()
                job[k] = None

    def cancel(self, task_id):
        job = self.jobs[task_id]
        task = job["task"]
        if task["state"] in protocol.TERMINAL:
            return task["state"]                             # after the fact: the outcome stands
        if task["state"] in ("queued", "waiting_for_model", "waiting_for_resources", "scheduled"):
            self._stop_timers(job)
            return protocol.advance(task, "cancelled")["state"]
        protocol.advance(task, "cancelling")
        if job["node"] == "local":
            self.cancel_local(job["local"])
        else:
            self.send(job["node"], "task.cancel", {"task_id": task_id})
        return task["state"]

    def _deadline(self, job):
        job["deadline"] = None
        task = job["task"]
        if task["state"] in protocol.TERMINAL:
            return
        self._stop_timers(job)
        # `timed_out` where the lifecycle has it (waiting on a node, or running); otherwise the nearest
        # terminal state the lifecycle allows from here
        allowed = protocol.NEXT[task["state"]]
        protocol.advance(task, next(s for s in ("timed_out", "failed", "cancelled") if s in allowed))
        job["error"] = "deadline exceeded"
        job["explain"].append("Stopped: the task's deadline passed.")
        if job["node"] == "local" and job["local"] is not None:
            job["local"]["on_change"] = None                 # its outcome no longer concerns this task
            self.cancel_local(job["local"])
        elif job["node"] not in (None, "local") and self.book.is_paired(job["node"]):
            self.send(job["node"], "task.cancel", {"task_id": task["task_id"]})

    def _lost(self, rec, why):
        """The node holding these tasks is gone. Nothing waits forever: each is recovered or failed."""
        for job in self.jobs.values():
            task = job["task"]
            if job["node"] != rec["node_id"] or task["state"] not in REMOTE:
                continue
            self._stop_timers(job, keep_deadline=True)
            if task["state"] == "cancelling":                # it was being cancelled anyway
                protocol.advance(task, "cancelled")
                continue
            protocol.advance(task, "interrupted")
            policy = task.get("retry_policy") or {}
            if policy.get("retryable") and job["attempts"] <= int(policy.get("max_retries", 0)):
                protocol.advance(task, "recovery_pending")
                job["explain"].append(f"Interrupted on {rec['label']} ({why}); recovering, attempt "
                                      f"{job['attempts'] + 1} of {int(policy.get('max_retries', 0)) + 1}.")
                job["node"] = None
                protocol.advance(task, "queued")
                self._place(job)
            else:
                job["error"] = f"{why}; no retries left"
                job["explain"].append(f"Failed: {why} on {rec['label']} and the retry policy allows no further attempt.")
                self._stop_timers(job)
                protocol.advance(task, "failed")
