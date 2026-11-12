"""The inference scheduler: who runs on the model next, and who may wait for it at all.

One loaded model serves everyone on the Mac — a chat, a coding agent, the editor, voice, the browser
assistant — and it computes one thing at a time (a merged batch counts as one). This is the single place
that decides the order, replacing a plain lock whose wake-up order was whatever the OS picked:

* admission — at most `limit` requests running or waiting (0 = no cap); over it a request is refused at
  once instead of piling up threads and prompt buffers;
* priority — a turn goes to the most urgent waiter: `interactive` (voice, editor completions: a person is
  waiting on every word) before `normal` (chat, the API default) before `background` (agents, batch jobs),
  arrival order within a class;
* aging — a waiter is promoted one class for every `aging_s` seconds it has waited, so background work is
  delayed by interactive work, never starved by it;
* memory — while the memory manager (engine/memory.py `Manager`) reports the Mac `critical`, background
  turns are held back (up to `defer_s`, then they run and the RAM guard decides), so an agent loop does
  not push a machine that is already swapping further. `rows_allowed` tells the batcher how many requests
  to merge: the merged KV cache grows with every row, so a tight machine merges fewer.

Every other use of the model — embeddings, sleep and wake, unloading for image generation — takes its
turn through the same object, so nothing reaches the model around the queue. `Scheduler` also behaves as a
lock (`with SCHED:`, `acquire`/`release`/`locked`) at `normal` priority for those call sites.
"""
import heapq
import itertools
import threading
import time

PRIORITIES = ("interactive", "normal", "background")
RANK = {p: i for i, p in enumerate(PRIORITIES)}


def parse_priority(value, default="normal"):
    """A client's stated priority (header X-Priority / body "priority"); anything else is `default`."""
    v = str(value or "").strip().lower()
    return v if v in RANK else default


class Scheduler:
    def __init__(self, limit=0, aging_s=30.0, defer_s=60.0, pressure=None, clock=time.monotonic):
        self.limit = limit
        self.aging_s, self.defer_s = aging_s, defer_s
        self.pressure = pressure or (lambda: "ok")       # () -> "ok" | "tight" | "critical"
        self.clock = clock
        self.cv = threading.Condition()
        self.admitted = 0                 # requests between admit() and leave(): running or waiting
        self.waiters = []                 # heap of [rank, seq, since, priority, label]
        self.seq = itertools.count()
        self.holder = None                # (priority, label, since) of the running turn
        self.served = {p: 0 for p in PRIORITIES}
        self.waited = {p: 0.0 for p in PRIORITIES}
        self.deferred = 0                 # turns held back by memory pressure

    # ── admission ──
    def admit(self):
        with self.cv:
            if self.limit and self.admitted >= self.limit:
                return False
            self.admitted += 1
            return True

    def leave(self):
        with self.cv:
            self.admitted = max(0, self.admitted - 1)

    @property
    def inflight(self):
        return self.admitted

    # ── turns ──
    def _rank(self, w, now):
        """The class a waiter competes in now: its own, raised one step per `aging_s` waited."""
        promoted = int((now - w[2]) // self.aging_s) if self.aging_s > 0 else 0
        return max(0, RANK[w[3]] - promoted)

    def _next(self, now):
        """The waiter that should run now, or None (memory holds every remaining one back)."""
        critical = self.pressure() == "critical"
        best = None
        for w in self.waiters:
            if critical and w[3] == "background" and now - w[2] < self.defer_s:
                continue
            key = (self._rank(w, now), w[1])
            if best is None or key < best[0]:
                best = (key, w)
        return best[1] if best else None

    def acquire_turn(self, priority="normal", label=""):
        priority = parse_priority(priority)
        with self.cv:
            now = self.clock()
            me = [RANK[priority], next(self.seq), now, priority, label]
            heapq.heappush(self.waiters, me)
            held_back = False
            while True:
                if self.holder is None:
                    nxt = self._next(self.clock())
                    if nxt is me:
                        break
                    if nxt is None and not held_back:
                        held_back = True
                        self.deferred += 1
                # woken by a release, or re-checked: aging and memory pressure change with time alone
                self.cv.wait(timeout=0.5)
            self.waiters.remove(me)
            heapq.heapify(self.waiters)
            start = self.clock()
            self.holder = (priority, label, start)
            self.served[priority] += 1
            self.waited[priority] += start - me[2]

    def release_turn(self):
        with self.cv:
            self.holder = None
            self.cv.notify_all()

    def turn(self, priority="normal", label=""):
        return _Turn(self, priority, label)

    # ── the lock interface, for call sites that just need the model to themselves ──
    def acquire(self, blocking=True):
        if not blocking:
            with self.cv:
                if self.holder is not None or self.waiters:
                    return False
                self.holder = ("normal", "", self.clock())
                return True
        self.acquire_turn("normal")
        return True

    def release(self):
        self.release_turn()

    def locked(self):
        return self.holder is not None

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *exc):
        self.release()

    # ── what the batcher and /health read ──
    def rows_allowed(self, max_rows):
        """How many requests one merged pass may take: all of them when memory is fine, two when it is
        tight, none merged when it is critical."""
        state = self.pressure()
        return max_rows if state == "ok" else (min(2, max_rows) if state == "tight" else 1)

    def snapshot(self):
        with self.cv:
            now = self.clock()
            waiting = {p: 0 for p in PRIORITIES}
            for w in self.waiters:
                waiting[w[3]] += 1
            running = None
            if self.holder:
                running = {"priority": self.holder[0], "label": self.holder[1],
                           "for_s": round(now - self.holder[2], 1)}
            return {"inflight": self.admitted, "limit": self.limit, "running": running, "waiting": waiting,
                    "served": dict(self.served),
                    "mean_wait_s": {p: round(self.waited[p] / self.served[p], 3) if self.served[p] else None
                                    for p in PRIORITIES},
                    "deferred_by_memory": self.deferred, "memory": self.pressure()}


class _Turn:
    def __init__(self, sched, priority, label):
        self.sched, self.priority, self.label = sched, priority, label

    def __enter__(self):
        self.sched.acquire_turn(self.priority, self.label)
        return self

    def __exit__(self, *exc):
        self.sched.release_turn()
