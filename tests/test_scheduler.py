"""The inference scheduler (engine/scheduler.py) and the memory manager (engine/memory.py `Manager`): the order
turns are given in, aging, deferral under memory pressure, and the order memory is given back in. No model."""
import os
import sys
import threading
import time
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import memory  # noqa: E402
from engine.scheduler import Scheduler, parse_priority  # noqa: E402


def queue_up(sched, entries, gap=0.05):
    """Hold the model, line `entries` (priority, name) up behind it in that order, release, and return the
    names in the order they got their turn."""
    order, threads = [], []
    sched.acquire_turn("normal", "holder")

    def go(priority, name):
        with sched.turn(priority, name):
            order.append(name)
    for priority, name in entries:
        t = threading.Thread(target=go, args=(priority, name))
        t.start()
        threads.append(t)
        time.sleep(gap)                       # arrival order is the order of this list
    sched.release_turn()
    for t in threads:
        t.join(10)
    return order


class TestScheduler(unittest.TestCase):
    def test_priority_parsing_falls_back_to_normal(self):
        self.assertEqual(parse_priority("Interactive"), "interactive")
        self.assertEqual(parse_priority(" background "), "background")
        self.assertEqual(parse_priority("urgent!!"), "normal")
        self.assertEqual(parse_priority(None), "normal")

    def test_more_urgent_first_and_arrival_order_within_a_class(self):
        s = Scheduler(aging_s=0)
        got = queue_up(s, [("background", "agent"), ("normal", "chat1"), ("interactive", "voice"),
                           ("normal", "chat2")])
        self.assertEqual(got, ["voice", "chat1", "chat2", "agent"])
        snap = s.snapshot()
        self.assertEqual(snap["served"], {"interactive": 1, "normal": 3, "background": 1})
        self.assertIsNone(snap["running"])

    def test_a_long_wait_promotes_a_background_request(self):
        now = [0.0]
        s = Scheduler(aging_s=30, clock=lambda: now[0])
        s.acquire_turn("normal", "holder")
        order = []

        def go(priority, name):
            with s.turn(priority, name):
                order.append(name)
        bg = threading.Thread(target=go, args=("background", "agent"))
        bg.start()
        time.sleep(0.05)
        now[0] = 61.0                          # two aging steps: background → interactive class
        chat = threading.Thread(target=go, args=("normal", "chat"))
        chat.start()
        time.sleep(0.05)
        s.release_turn()
        bg.join(10)
        chat.join(10)
        self.assertEqual(order, ["agent", "chat"])

    def test_background_waits_while_memory_is_critical_but_not_forever(self):
        state = ["critical"]
        now = [0.0]
        s = Scheduler(aging_s=0, defer_s=60, pressure=lambda: state[0], clock=lambda: now[0])
        ran = threading.Event()

        def go():
            with s.turn("background", "agent"):
                ran.set()
        t = threading.Thread(target=go)
        t.start()
        self.assertFalse(ran.wait(1.2))        # nothing else is running, still held back
        self.assertEqual(s.snapshot()["deferred_by_memory"], 1)
        with s.turn("interactive", "voice"):  # a person waiting is not held back
            pass
        now[0] = 61.0                          # held back long enough: it runs, the RAM guard decides
        self.assertTrue(ran.wait(2))
        t.join(5)

    def test_background_runs_as_soon_as_pressure_eases(self):
        state = ["critical"]
        s = Scheduler(pressure=lambda: state[0])
        ran = threading.Event()
        t = threading.Thread(target=lambda: (s.acquire_turn("background"), ran.set(), s.release_turn()))
        t.start()
        self.assertFalse(ran.wait(0.7))
        state[0] = "ok"
        self.assertTrue(ran.wait(2))
        t.join(5)

    def test_rows_allowed_follows_memory(self):
        state = ["ok"]
        s = Scheduler(pressure=lambda: state[0])
        self.assertEqual(s.rows_allowed(8), 8)
        state[0] = "tight"
        self.assertEqual(s.rows_allowed(8), 2)
        state[0] = "critical"
        self.assertEqual(s.rows_allowed(8), 1)

    def test_the_lock_interface_takes_a_normal_turn(self):
        s = Scheduler()
        with s:
            self.assertTrue(s.locked())
            self.assertFalse(s.acquire(blocking=False))
            self.assertEqual(s.snapshot()["running"]["priority"], "normal")
        self.assertFalse(s.locked())
        self.assertTrue(s.acquire(blocking=False))
        s.release()


class FakeHost:
    def __init__(self):
        self.mem_limit, self.logs, self.saved = 0.88, [], 0
        self.cache, self.cache_tokens, self.snapshots, self.store = ["kv"], [1, 2, 3], [([1], ["s"])], None

    def log(self, m):
        self.logs.append(m)

    def persist_prefix(self):
        self.saved += 1

    def memory_breakdown(self):
        return {"weights": 2 * 2**30, "kv": 2**28, "snapshots": 2**20, "mlx_cache": 2**24}


class TestMemoryManager(unittest.TestCase):
    def make(self, heads):
        """A manager whose headroom answers come from `heads`, one per call (the last one repeats)."""
        host, released = FakeHost(), []
        mgr = memory.Manager(host, lambda: released.append(1))
        it = iter(heads)
        last = [heads[-1]]

        def head(_limit):
            last[0] = next(it, last[0])
            return last[0]
        return host, mgr, released, mock.patch.object(memory, "headroom_bytes", head)

    def test_buffers_first_and_nothing_else_when_that_is_enough(self):
        host, mgr, released, patch = self.make([0, 10**9])
        with patch:
            self.assertTrue(mgr.reclaim(10**6))
        self.assertTrue(released)
        self.assertTrue(host.snapshots)                       # restore points kept
        self.assertEqual(mgr.given_up["restore_points"], 0)

    def test_restore_points_next_and_the_conversation_never_during_a_request(self):
        host, mgr, _, patch = self.make([0])
        with patch:
            self.assertFalse(mgr.reclaim(10**6))              # upto restore_points by default
        self.assertEqual(host.snapshots, [])
        self.assertEqual(host.cache, ["kv"])                  # the live conversation untouched
        self.assertEqual(mgr.given_up["conversation"], 0)

    def test_the_conversation_goes_to_the_ssd_first_between_requests(self):
        host, mgr, _, patch = self.make([0])
        with patch, mock.patch.object(memory, "used_fraction", lambda: 0.95), \
                mock.patch.object(memory, "pressure_level", lambda: 1):
            self.assertEqual(mgr.state(), "critical")
            given = mgr.relieve()
        self.assertEqual(given, ["restore_points", "conversation"])
        self.assertEqual(host.saved, 1)                       # persisted before it was dropped
        self.assertIsNone(host.cache)
        self.assertEqual(host.cache_tokens, [])

    def test_relieve_does_nothing_unless_critical(self):
        host, mgr, released, patch = self.make([0])
        with patch, mock.patch.object(memory, "used_fraction", lambda: 0.5), \
                mock.patch.object(memory, "pressure_level", lambda: 1):
            self.assertEqual(mgr.state(), "ok")
            self.assertEqual(mgr.relieve(), [])
        self.assertEqual(host.cache, ["kv"])
        self.assertFalse(released)

    def test_state_levels(self):
        _, mgr, _, _ = self.make([0])
        for used, level, want in ((0.5, 1, "ok"), (0.85, 1, "tight"), (0.5, 2, "tight"), (0.91, 1, "critical"),
                                  (0.5, 4, "critical")):
            with mock.patch.object(memory, "used_fraction", lambda u=used: u), \
                    mock.patch.object(memory, "pressure_level", lambda l=level: l):
                self.assertEqual(mgr.state(), want, (used, level))

    def test_report_reads_the_engine_breakdown(self):
        _, mgr, _, patch = self.make([2**30])
        with patch:
            r = mgr.report()
        self.assertEqual(r["resident_gb"], 2.0)
        self.assertEqual(r["reclaimable_gb"]["conversation"], 0.25)
        self.assertIsNone(r["ssd_prefix_gb"])


if __name__ == "__main__":
    unittest.main()
