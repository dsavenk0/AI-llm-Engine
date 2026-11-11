"""ADAR: the real protocol, pairing, scheduler, node and coordinator code against a simulated
network (adar/sim.py) — the areas docs/adar/TESTING.md lists.

Nothing here needs a second Mac, a socket, a model or MLX: nodes are small Python objects, time is a
number, and every network effect comes from one seeded generator, so a failure repeats exactly.

    .venv/bin/python -m unittest tests.test_adar -v
"""
import hashlib
import json
import os
import subprocess
import sys
import tracemalloc
import unittest

from adar import capability, identity, protocol, scheduler, transfer, wire
from adar.coordinator import BACKOFF_MAX_S, HEARTBEAT_S, NODE_NEXT, OFFLINE_AFTER, RESEND_S
from adar.protocol import ProtocolError
from adar.scheduler import View
from adar.sim import LABEL, Cluster, SimNet, run_transfer, task

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def cluster(seed=1, workers=("mini",), tps=None, **coordinator_kw):
    """A coordinator and paired, profiled workers, all holding the model "small"."""
    coordinator_kw.setdefault("models", ["small"])
    c = Cluster(seed=seed, **coordinator_kw)
    c.coordinator.set_estimate("local", "small", tps=40, load_s=1.0)
    for w in workers:
        c.add(w, models=["small"])
        assert c.pair(w), w
        assert c.profile(w), w
        c.coordinator.set_estimate(c.nodes[w].identity.node_id, "small", tps=(tps or {}).get(w, 40), load_s=1.0)
    return c


def refusals(node):
    return [why for _src, why in node.refused]


class TestSimulatedIdentity(unittest.TestCase):
    def setUp(self):
        self.a = identity.load_or_create("", simulated=True, seed="a")
        self.b = identity.load_or_create("", simulated=True, seed="b")

    def test_a_signature_verifies_for_its_signer_and_nothing_else(self):
        sig = self.a.sign(b"hello")
        self.assertTrue(self.b.verify(self.a.node_id, b"hello", sig))
        self.assertFalse(self.b.verify(self.a.node_id, b"hellO", sig))
        self.assertFalse(self.b.verify(self.b.node_id, b"hello", sig))

    def test_both_sides_derive_one_session_key_and_only_with_the_same_code(self):
        ea, eb = self.a.ephemeral_public(), self.b.ephemeral_public()
        self.assertEqual(self.a.exchange(eb, "123456"), self.b.exchange(ea, "123456"))
        self.assertNotEqual(self.a.exchange(eb, "123456"), self.b.exchange(ea, "123457"))

    def test_an_identity_is_stable_for_its_seed_and_marked_simulated(self):
        again = identity.load_or_create("", simulated=True, seed="a")
        self.assertEqual(again.node_id, self.a.node_id)
        self.assertTrue(self.a.node_id.startswith("sim:"))
        self.assertEqual(identity.fingerprint_of(self.a.node_id), self.a.fingerprint)

    def test_a_simulated_identity_cannot_vouch_for_a_real_one(self):
        with self.assertRaises(identity.IdentityError):
            self.a.verify("ed25519:AAAA", b"x", "AAAA")

    @unittest.skipIf(identity.crypto_available(), "cryptography is installed: the refusal without it was not covered")
    def test_a_real_node_refuses_to_start_without_real_cryptography(self):
        with self.assertRaisesRegex(identity.IdentityError, "cryptography"):
            identity.load_or_create("/nonexistent")


class TestProtocolVersioning(unittest.TestCase):
    def test_another_version_is_refused_by_name_not_parsed_on_a_guess(self):
        raw = json.dumps({"v": 2, "verb": "health.ping", "body": {}}).encode()
        with self.assertRaisesRegex(ProtocolError, r"v2.*v1"):
            protocol.decode(raw)
        with self.assertRaisesRegex(ProtocolError, "v0"):
            protocol.validate_task(dict(protocol.new_task("echo.test", {}), v=0))

    def test_oversized_malformed_and_unknown_messages_are_refused(self):
        with self.assertRaisesRegex(ProtocolError, "exceeds"):
            protocol.decode(b"{" + b" " * protocol.MAX_MESSAGE_BYTES + b"}")
        for raw in (b"not json", b"[1]", json.dumps({"v": 1, "verb": "shell.exec", "body": {}}).encode(),
                    json.dumps({"v": 1, "verb": "health.ping", "body": 3}).encode()):
            with self.assertRaises(ProtocolError):
                protocol.decode(raw)

    def test_the_task_lifecycle_allows_only_its_own_transitions(self):
        t = protocol.new_task("echo.test", {})
        with self.assertRaisesRegex(ProtocolError, "illegal"):
            protocol.advance(t, "running")
        for s in ("validated", "queued", "scheduled", "dispatched", "accepted", "running", "returning_result",
                  "completed"):
            protocol.advance(t, s)
        with self.assertRaisesRegex(ProtocolError, "terminal"):
            protocol.advance(t, "failed")

    def test_every_state_is_reachable_and_every_non_terminal_one_leads_somewhere(self):
        self.assertEqual(set(protocol.NEXT) | protocol.TERMINAL, set(protocol.STATES))
        reachable = {s for outs in protocol.NEXT.values() for s in outs} | {"created"}
        self.assertEqual(reachable, set(protocol.STATES))


class TestDiscovery(unittest.TestCase):
    def test_discovery_alone_grants_nothing(self):
        c = Cluster(seed=2, models=["small"])
        c.coordinator.set_estimate("local", "small", tps=40)
        n = c.add("mini", models=["small"])
        rec = c.coordinator.discover(n.advert())
        self.assertEqual(rec["state"], "discovered")
        self.assertFalse(c.coordinator.book.is_paired(n.identity.node_id))
        self.assertEqual(set(n.advert()), {"v", "fingerprint", "addr", "instance"})     # no models, no memory
        tid = c.coordinator.submit(task("small", tokens=5000))
        c.net.run_for(1)
        self.assertEqual(c.coordinator.jobs[tid]["node"], "local")

    def test_an_advert_of_another_protocol_version_is_refused_with_the_reason(self):
        c = Cluster(seed=2)
        n = c.add("mini")
        self.assertIsNone(c.coordinator.discover(dict(n.advert(), v=2)))
        self.assertIn("v2", c.coordinator.conflicts[-1][1])
        self.assertEqual(c.coordinator.nodes, {})

    def test_one_fingerprint_from_two_places_is_refused_not_merged(self):
        c = Cluster(seed=2)
        n = c.add("mini")
        self.assertIsNotNone(c.coordinator.discover(n.advert()))
        clone = dict(n.advert(), addr="elsewhere", instance="other")
        self.assertIsNone(c.coordinator.discover(clone))
        self.assertIn("already advertised", c.coordinator.conflicts[-1][1])
        rec = c.coordinator.nodes[n.identity.fingerprint]
        self.assertEqual(rec["addr"], "mini")                       # the known node was not replaced
        with self.assertRaisesRegex(ProtocolError, "more than one place"):
            c.coordinator.pair(n.identity.fingerprint)

    def test_the_same_advert_seen_again_is_just_the_same_node(self):
        c = Cluster(seed=2)
        n = c.add("mini")
        a = c.coordinator.discover(n.advert())
        self.assertIs(c.coordinator.discover(n.advert()), a)
        self.assertEqual(c.coordinator.conflicts, [])


class TestPairing(unittest.TestCase):
    def test_pairing_needs_the_code_a_person_read_off_the_other_screen(self):
        c = Cluster(seed=3, models=["small"])
        n = c.add("mini", models=["small"])
        seen = []
        c.net.tap = lambda s, d, raw: seen.append(raw)
        self.assertTrue(c.pair("mini"))
        code = n.codes[-1]
        self.assertRegex(code, r"^\d{6}$")
        self.assertFalse(any(code.encode() in raw for raw in seen))   # the code never crossed the network
        rec = c.coordinator.nodes[n.identity.fingerprint]
        self.assertEqual(rec["history"][:4], ["discovered", "awaiting_pairing", "paired", "online"])
        self.assertTrue(capability.value(rec["capability"], "static", "installed_models"), ["small"])

    def test_a_wrong_code_fails_closed_and_three_wrong_codes_void_the_attempt(self):
        c = Cluster(seed=3)
        n = c.add("mini")
        self.assertFalse(c.pair("mini", code="000000"))
        self.assertFalse(c.coordinator.book.is_paired(n.identity.node_id))
        self.assertFalse(n.book.is_paired(c.coordinator.identity.node_id))
        self.assertIn("pairing failed: the code does not match", refusals(n))
        n.pairing.accepting = True
        for _ in range(2):
            c.coordinator.enter_code("111111")
            c.net.run_for(0.2)
        self.assertIn("pairing failed: wrong code three times — start again", refusals(n))
        c.coordinator.enter_code(n.codes[-1])                        # even the right code is too late now
        c.net.run_for(0.2)
        self.assertFalse(n.book.is_paired(c.coordinator.identity.node_id))

    def test_a_node_that_nobody_put_into_pairing_mode_ignores_requests(self):
        c = Cluster(seed=3)
        n = c.add("mini")
        c.coordinator.discover(n.advert())
        c.coordinator.pair(n.identity.fingerprint)
        c.net.run_for(0.5)
        self.assertEqual(n.codes, [])
        self.assertIn("pair.request: this node is not in pairing mode", refusals(n))

    def test_the_human_step_cannot_be_skipped(self):
        c = Cluster(seed=3)
        n = c.add("mini")
        n.pairing.accepting = True
        me = c.coordinator
        for body in ({"step": "proof", "node_id": me.identity.node_id, "mac": "00", "sig": "AA"},
                     {"step": "accept", "node_id": me.identity.node_id, "mac": "00", "sig": "AA"}):
            c.net.send("coordinator", "mini", protocol.encode("pair.confirm", body))
        c.net.run_for(0.2)
        self.assertFalse(n.book.is_paired(me.identity.node_id))
        self.assertEqual(len(n.refused), 2)

    def test_a_recorded_pairing_proves_nothing_when_replayed(self):
        c = Cluster(seed=3)
        n = c.add("mini")
        tape = []
        c.net.tap = lambda s, d, raw: tape.append((s, d, raw))
        self.assertTrue(c.pair("mini"))
        proof = next(raw for s, d, raw in tape if b'"step": "proof"' in raw)
        n.book.revoke(c.coordinator.identity.node_id)               # as if the pairing had since been undone
        n.pairing.accepting = True
        c.net.tap = None
        before = len(n.refused)
        c.net.send("coordinator", "mini", proof)                    # the old proof, on its own
        c.net.run_for(0.2)
        self.assertFalse(n.book.is_paired(c.coordinator.identity.node_id))
        for s, d, raw in tape:                                      # the whole old conversation, replayed at it
            if d == "mini" and (b"pair.request" in raw or b"pair.confirm" in raw):
                c.net.send(s, d, raw)
                c.net.run_for(0.2)
        self.assertFalse(n.book.is_paired(c.coordinator.identity.node_id))
        self.assertGreater(len(n.refused), before)

    def test_a_key_that_does_not_match_the_advertised_fingerprint_is_not_paired(self):
        c = Cluster(seed=3)
        n = c.add("mini")
        other = identity.load_or_create("", simulated=True, seed="someone else")
        n.pairing.accepting = True
        c.coordinator.discover(dict(n.advert(), fingerprint=other.fingerprint))     # an advert for another key
        c.coordinator.pair(other.fingerprint)
        c.net.run_for(0.2)
        c.coordinator.enter_code(n.codes[-1])
        c.net.run_for(0.2)
        self.assertFalse(c.coordinator.book.is_paired(n.identity.node_id))
        self.assertIn("does not match its advertised fingerprint", " ".join(refusals(c.coordinator)))

    def test_pairing_alone_permits_no_task(self):
        c = Cluster(seed=3, models=["small"], local_worker=False)
        n = c.add("mini", models=["small"])
        self.assertTrue(c.pair("mini", grant=()))
        c.profile("mini")
        c.coordinator.set_estimate(n.identity.node_id, "small", tps=40)
        tid = c.coordinator.submit(task("small"))
        c.net.run_for(2)
        self.assertEqual(c.coordinator.state(tid), "rejected")
        self.assertIn("does not permit echo.test", c.coordinator.explain(tid)[-1])
        self.assertEqual(n.executions, 0)

    def test_a_worker_checks_the_permission_itself_whatever_the_coordinator_believes(self):
        c = cluster(seed=3, local_worker=False)
        n = c.nodes["mini"]
        n.book.grant(c.coordinator.identity.node_id, ())            # the worker's owner withdrew it
        tid = c.coordinator.submit(task("small"))
        c.net.run_for(2)
        self.assertEqual(c.coordinator.state(tid), "rejected")
        self.assertIn("Rejected by mini", c.coordinator.explain(tid)[-1])
        self.assertEqual(n.executions, 0)

    def test_a_grant_for_a_task_type_with_no_handler_is_refused(self):
        c = cluster(seed=3)
        with self.assertRaisesRegex(ProtocolError, "no handler"):
            c.coordinator.grant(c.fingerprint("mini"), ["shell.exec"])

    def test_revocation_is_immediate_and_outright(self):
        c = cluster(seed=3)
        n, me = c.nodes["mini"], c.coordinator
        me.revoke_node(c.fingerprint("mini"))
        c.net.run_for(0.5)
        self.assertEqual(me.nodes[c.fingerprint("mini")]["state"], "revoked")
        self.assertFalse(n.book.is_paired(me.identity.node_id))     # it was told
        n.book.add(me.identity.node_id, "coordinator", 0)           # even a node that still believes in it
        n.send(me.identity.node_id, "health.pong", {"n": 1})
        c.net.run_for(0.5)
        self.assertIn("health.pong: sender's pairing was revoked", refusals(me))
        tid = me.submit(task("small", tokens=100000))
        c.net.run_for(1)
        self.assertEqual(me.jobs[tid]["node"], "local")


class TestWire(unittest.TestCase):
    def setUp(self):
        self.c = cluster(seed=4)
        self.me, self.n = self.c.coordinator, self.c.nodes["mini"]
        self.tape = []
        self.c.net.tap = lambda s, d, raw: self.tape.append((s, d, raw))

    def capture(self, verb):
        self.me.send(self.n.identity.node_id, verb, {"n": 99})
        self.c.net.run_for(0.2)
        return next(raw for s, d, raw in reversed(self.tape) if d == "mini" and verb.encode() in raw)

    def test_an_unpaired_sender_is_not_even_considered(self):
        stranger = identity.load_or_create("", simulated=True, seed="stranger")
        raw = wire.seal(stranger, self.n.identity.node_id, 1, "task.submit", task("small"))
        self.c.net.send("nowhere", "mini", raw)
        self.c.net.run_for(0.2)
        self.assertIn("task.submit: sender is not a paired node", refusals(self.n))
        self.assertEqual(self.n.executions, 0)

    def test_a_message_without_an_envelope_is_refused(self):
        self.c.net.send("coordinator", "mini", protocol.encode("task.submit", task("small")))
        self.c.net.run_for(0.2)
        self.assertIn("not an authenticated envelope", refusals(self.n)[-1])

    def test_the_same_message_twice_is_a_replay(self):
        raw = self.capture("health.ping")
        before = len(self.n.refused)
        self.c.net.send("coordinator", "mini", raw)
        self.c.net.run_for(0.2)
        self.assertIn("replayed message", refusals(self.n)[-1])
        self.assertEqual(len(self.n.refused), before + 1)

    def test_a_changed_message_no_longer_matches_its_signature(self):
        raw = self.capture("health.ping")
        forged = raw.replace(b'"n": 99', b'"n": 98')
        self.assertNotEqual(forged, raw)
        self.c.net.send("coordinator", "mini", forged)
        self.c.net.run_for(0.2)
        self.assertIn("signature does not match", refusals(self.n)[-1])

    def test_a_message_for_another_node_is_not_accepted_by_this_one(self):
        other = self.c.add("other")
        self.assertTrue(self.c.pair("other"))
        raw = self.capture("health.ping")                           # addressed to mini
        self.c.net.send("coordinator", "other", raw)
        self.c.net.run_for(0.2)
        self.assertIn("addressed to another node", refusals(other)[-1])

    def test_a_late_message_is_taken_but_never_twice(self):
        book, peer = self.n.book, self.me.identity.node_id
        top = book.peers[peer]["seq_in"]
        self.assertTrue(book.accept_seq(peer, top + 5))
        self.assertTrue(book.accept_seq(peer, top + 3))             # arrived late: still new
        self.assertFalse(book.accept_seq(peer, top + 3))
        self.assertFalse(book.accept_seq(peer, top + 5))
        self.assertFalse(book.accept_seq(peer, top + 5 - 64))       # too old to tell from a repeat


class TestCapability(unittest.TestCase):
    def setUp(self):
        self.ident = identity.load_or_create("", simulated=True, seed="cap")

    def test_every_field_says_how_it_was_obtained(self):
        doc = capability.build(self.ident, models=["small"], simulated=True, memory_gb=16, available_gb=8)
        self.assertTrue(doc["simulated"])
        for section in ("static", "dynamic"):
            for name, f in doc[section].items():
                self.assertIn(f["source"], ("measured", "static", "estimated", "unavailable"), name)
        self.assertEqual(doc["static"]["chip_generation"], {"value": None, "source": "unavailable"})
        capability.check(doc)

    def test_a_simulated_node_never_reports_this_macs_memory(self):
        doc = capability.build(self.ident, simulated=True)
        self.assertEqual(doc["static"]["unified_memory_gb"], {"value": None, "source": "unavailable"})
        self.assertEqual(doc["dynamic"]["available_memory_gb"]["source"], "unavailable")

    def test_documents_that_contradict_themselves_are_refused(self):
        doc = capability.build(self.ident, models=["small"], simulated=True, loaded_models=["large"])
        with self.assertRaisesRegex(ProtocolError, "loaded models it does not list"):
            capability.check(doc)
        doc = capability.build(self.ident, handlers=["echo.test", "shell.exec"], simulated=True)
        with self.assertRaisesRegex(ProtocolError, "doesn't define"):
            capability.check(doc)
        doc = capability.build(self.ident, simulated=True)
        doc["dynamic"]["thermal_state"]["value"] = "hot"
        with self.assertRaisesRegex(ProtocolError, "unavailable"):
            capability.check(doc)

    def test_a_bad_document_from_a_node_changes_nothing_at_the_coordinator(self):
        c = cluster(seed=5)
        n, me = c.nodes["mini"], c.coordinator
        rec = me.nodes[c.fingerprint("mini")]
        good = rec["capability"]
        n.loaded = ["large"]                                        # claims a warm model it does not have
        n.send(me.identity.node_id, "capability.push", n.capability())
        c.net.run_for(0.2)
        self.assertIs(rec["capability"], good)
        self.assertIn("loaded models it does not list", refusals(me)[-1])
        n.loaded = []
        other = c.add("other", models=["small"])
        self.assertTrue(c.pair("other"))
        n.send(me.identity.node_id, "capability.push", other.capability())     # someone else's document
        c.net.run_for(0.2)
        self.assertIn("describes a different node", refusals(me)[-1])


def view(label, **kw):
    kw.setdefault("handlers", ("echo.test",))
    kw.setdefault("installed", ("small",))
    kw.setdefault("available_gb", 12.0)
    kw.setdefault("tps", 40.0)
    kw.setdefault("load_s", 2.0)
    if not kw.get("local"):
        kw.setdefault("scope", ("echo.test",))
        kw.setdefault("rtt_s", 0.002)
        kw.setdefault("bps", 1e9)
    return View(node_id=label, label=label, **kw)


class TestScheduler(unittest.TestCase):
    """Each case is small enough to do by hand; the expected cost is written next to it."""

    def test_a_small_task_stays_home_when_the_link_is_slow(self):
        t = task("small", tokens=40, size=1_000_000)                # 1 s of work, 2 MB to move
        local = view("here", local=True, loaded=("small",))         # 1.0 s
        far = view("far", loaded=("small",), tps=80.0, rtt_s=0.05, bps=8e6)   # 0.5 + 2 MB at 1 MB/s = 2.0 + 0.1
        d = scheduler.decide(t, local, [far])
        self.assertEqual(d.target, "local")
        self.assertAlmostEqual(d.costs["here"], 1.0)
        self.assertAlmostEqual(d.costs["far"], 2.6)
        self.assertIn("remote overhead on far (~2.1s", d.explanation)
        self.assertIn("compute saving (~0.5s)", d.explanation)

    def test_a_warm_faster_node_wins_over_a_cold_local_start(self):
        t = task("small", tokens=400, size=2_000)
        local = view("here", local=True)                            # cold: 2 + 10 = 12 s
        warm = view("studio", loaded=("small",), tps=160.0)         # 2.5 s + ~0
        d = scheduler.decide(t, local, [warm])
        self.assertEqual(d.target, "studio")
        self.assertAlmostEqual(d.costs["here"], 12.0)
        self.assertAlmostEqual(d.costs["studio"], 2.504, places=2)
        self.assertIn("model already loaded there", d.explanation)
        self.assertIn("faster on this model", d.explanation)
        self.assertIn("~9.5s", d.explanation)

    def test_a_busy_local_machine_sends_work_away_and_a_busy_node_does_not_get_it(self):
        t = task("small", tokens=400)
        idle_remote = view("mini", loaded=("small",))
        d = scheduler.decide(t, view("here", local=True, loaded=("small",), busy=2), [idle_remote])   # 30 vs 10
        self.assertEqual(d.target, "mini")
        self.assertIn("this machine is busy", d.explanation)
        d = scheduler.decide(t, view("here", local=True, loaded=("small",)),
                             [view("mini", loaded=("small",), busy=3)])
        self.assertEqual(d.target, "local")

    def test_equal_costs_stay_local(self):
        t = task("small", tokens=400, size=0)
        d = scheduler.decide(t, view("here", local=True, loaded=("small",)),
                             [view("mini", loaded=("small",), rtt_s=0.0)])
        self.assertEqual(d.target, "local")

    def test_a_node_that_is_merely_there_is_never_chosen(self):
        t = task("small", tokens=4000)
        local = view("here", local=True, loaded=("small",))
        cases = {"missing checkpoint small": view("a", installed=("other",), tps=1000.0),
                 "needs 1.0 GB, 0.5 GB available": view("b", available_gb=0.5, tps=1000.0),
                 "link not measured": view("c", bps=None, tps=1000.0),
                 "no runtime estimate for this model": view("d", tps=None),
                 "node is degraded": view("e", state="degraded", tps=1000.0),
                 "node is offline": view("f", state="offline", tps=1000.0),
                 "this pairing does not permit echo.test": view("g", scope=(), tps=1000.0),
                 "no echo.test handler": view("h", handlers=(), tps=1000.0),
                 "available memory unknown": view("i", available_gb=None, tps=1000.0)}
        d = scheduler.decide(t, local, list(cases.values()))
        self.assertEqual(d.target, "local")
        self.assertEqual(d.rejected, {v.label: why for why, v in cases.items()})

    def test_a_local_only_task_is_never_offered_elsewhere(self):
        t = task("small", tokens=40000, remote=False)
        d = scheduler.decide(t, view("here", local=True), [view("studio", loaded=("small",), tps=4000.0)])
        self.assertEqual(d.target, "local")
        self.assertEqual(d.explanation, "Running locally: the task is marked local-only.")
        self.assertNotIn("studio", d.costs)

    def test_what_cannot_run_anywhere_says_which_way_it_is_stuck(self):
        local = view("here", local=True)
        d = scheduler.decide(task("huge"), local, [view("mini")])
        self.assertEqual((d.target, d.wait), (None, "waiting_for_model"))
        self.assertIn("here: missing checkpoint huge; mini: missing checkpoint huge", d.explanation)
        d = scheduler.decide(task("small", memory_gb=64.0), local, [view("mini")])
        self.assertEqual((d.target, d.wait), (None, "waiting_for_resources"))
        d = scheduler.decide(task("small"), view("here", local=True, tps=None), [])
        self.assertEqual((d.target, d.wait), (None, None))
        self.assertTrue(d.explanation.startswith("Rejected: nothing can run this task"))

    def test_with_no_local_worker_the_reason_says_so(self):
        d = scheduler.decide(task("small"), None, [view("mini", loaded=("small",))])
        self.assertEqual(d.target, "mini")
        self.assertIn("this machine cannot run it", d.explanation)


class TestPlacementEndToEnd(unittest.TestCase):
    def test_remote_wins_when_the_other_node_is_faster_and_the_reason_is_kept(self):
        c = cluster(seed=6, tps={"mini": 400})
        tid = c.coordinator.submit(task("small", tokens=2000, work_s=5.0, payload="p"))
        c.net.run_for(10)
        me = c.coordinator
        self.assertEqual(me.state(tid), "completed")
        self.assertEqual(me.jobs[tid]["result"], {"echo": "p"})
        self.assertEqual((c.nodes["mini"].executions, me.executions), (1, 0))
        self.assertTrue(me.explain(tid)[0].startswith("Assigned to mini: faster on this model"))

    def test_local_wins_for_a_tiny_task_over_a_slow_link(self):
        c = Cluster(seed=6, models=["small"])
        c.coordinator.set_estimate("local", "small", tps=40, load_s=0.5)
        n = c.add("far", models=["small"])
        c.net.link("coordinator", "far", latency=0.2, bps=2e6)
        self.assertTrue(c.pair("far"))
        self.assertTrue(c.profile("far"))
        c.coordinator.set_estimate(n.identity.node_id, "small", tps=80, load_s=0.5)
        tid = c.coordinator.submit(task("small", tokens=20, work_s=0.5, size=200_000))
        c.net.run_for(5)
        self.assertEqual(c.coordinator.state(tid), "completed")
        self.assertEqual((n.executions, c.coordinator.executions), (0, 1))
        self.assertIn("Running locally: remote overhead on far", c.coordinator.explain(tid)[0])

    def test_the_link_figure_is_measured_not_the_configured_rate(self):
        c = Cluster(seed=6)
        c.add("far")
        c.net.link("coordinator", "far", latency=0.05, bps=8e6)
        self.assertTrue(c.pair("far"))
        rec = c.coordinator.nodes[c.fingerprint("far")]
        self.assertIsNone(rec["bps"])                               # paired, heard from, still not profiled
        self.assertAlmostEqual(rec["rtt_s"], 0.1, delta=0.01)       # from the heartbeat
        self.assertTrue(c.profile("far"))
        self.assertLess(rec["bps"], 8e6)                            # payload throughput is below the line rate
        self.assertGreater(rec["bps"], 4e6)

    def test_a_node_short_of_memory_is_passed_over(self):
        c = Cluster(seed=6, models=["small"], available_gb=2.0)
        c.coordinator.set_estimate("local", "small", tps=40)
        for label, free in (("tight", 3.0), ("roomy", 24.0)):
            n = c.add(label, models=["small"], available_gb=free)
            self.assertTrue(c.pair(label))
            c.profile(label)
            c.coordinator.set_estimate(n.identity.node_id, "small", tps=400 if label == "tight" else 40)
        tid = c.coordinator.submit(task("small", memory_gb=8.0, work_s=0.2))
        c.net.run_for(5)
        self.assertEqual(c.coordinator.state(tid), "completed")
        self.assertEqual((c.nodes["tight"].executions, c.nodes["roomy"].executions), (0, 1))
        self.assertIn("this machine cannot run it (needs 8.0 GB, 2.0 GB available)", c.coordinator.explain(tid)[0])

    def test_a_model_nobody_has_waits_and_runs_when_a_node_with_it_appears(self):
        c = cluster(seed=6)
        tid = c.coordinator.submit(task("large", work_s=0.2))
        c.net.run_for(3)
        self.assertEqual(c.coordinator.state(tid), "waiting_for_model")
        self.assertIn("mini: missing checkpoint large", c.coordinator.explain(tid)[0])
        n = c.add("studio", models=["large"])
        self.assertTrue(c.pair("studio"))
        c.profile("studio")
        c.coordinator.set_estimate(n.identity.node_id, "large", tps=30, load_s=1.0)
        n.send(c.coordinator.identity.node_id, "capability.push", n.capability())
        c.net.run_for(5)
        self.assertEqual(c.coordinator.state(tid), "completed")
        self.assertEqual(n.executions, 1)
        story = c.coordinator.explain(tid)                          # why it waited … then where it went
        self.assertTrue(story[0].startswith("Waiting for a model"))
        self.assertTrue(any(e.startswith("Waiting for resources") and "studio" in e for e in story[1:-1]))
        self.assertTrue(story[-1].startswith("Assigned to studio: this machine cannot run it"))

    def test_a_local_only_task_never_reaches_the_wire(self):
        c = cluster(seed=6, tps={"mini": 4000})
        seen = []
        c.net.tap = lambda s, d, raw: seen.append(raw)
        tid = c.coordinator.submit(task("small", tokens=2000, work_s=1.0, remote=False, payload="PRIVATE-PAYLOAD"))
        c.net.run_for(5)
        self.assertEqual(c.coordinator.state(tid), "completed")
        self.assertFalse(any(b"task.submit" in raw or b"PRIVATE-PAYLOAD" in raw for raw in seen))
        self.assertEqual(c.nodes["mini"].executions, 0)

    def test_a_task_for_another_node_is_local_only_at_the_worker_too(self):
        c = cluster(seed=6)
        n, me = c.nodes["mini"], c.coordinator
        me.send(n.identity.node_id, "task.submit", dict(task("small", remote=False), state="dispatched"))
        c.net.run_for(1)
        self.assertEqual(n.executions, 0)
        self.assertIn("local-only", " ".join(a.get("reason", "") for a in n.audit))


class TestTasks(unittest.TestCase):
    def setUp(self):
        self.c = cluster(seed=7, local_worker=False)
        self.me, self.n = self.c.coordinator, self.c.nodes["mini"]

    def test_the_same_key_submitted_twice_is_one_task(self):
        t = task("small", work_s=0.5, idempotency_key="k-1")
        again = dict(task("small", work_s=0.5, idempotency_key="k-1"))
        a = self.me.submit(t)
        b = self.me.submit(again)
        self.c.net.run_for(5)
        self.assertEqual(a, b)
        self.assertEqual(self.n.executions, 1)

    def test_a_lost_acknowledgment_leads_to_a_retry_and_still_one_execution(self):
        self.c.net.block("mini", "coordinator")                     # the task arrives; nothing comes back
        tid = self.me.submit(task("small", work_s=0.5, payload="once"))
        self.c.net.run_for(RESEND_S + 0.5)
        self.assertEqual(self.me.state(tid), "dispatched")
        self.assertEqual(self.n.executions, 1)                      # it ran, the coordinator just cannot know
        self.c.net.unblock("mini", "coordinator")
        self.c.net.run_for(2 * RESEND_S)
        self.assertEqual(self.me.state(tid), "completed")
        self.assertEqual(self.me.jobs[tid]["result"], {"echo": "once"})
        self.assertEqual(self.n.executions, 1)
        self.assertIn("task.duplicate", [a["action"] for a in self.n.audit])

    def test_cancel_before_it_starts(self):
        first = self.me.submit(task("small", work_s=5.0))
        second = self.me.submit(task("small", work_s=5.0))
        self.c.net.run_for(1)
        self.assertEqual(self.me.cancel(second), "cancelling")
        self.c.net.run_for(10)
        self.assertEqual((self.me.state(first), self.me.state(second)), ("completed", "cancelled"))
        self.assertEqual(self.n.executions, 1)                      # the cancelled one never ran

    def test_cancel_while_it_runs(self):
        tid = self.me.submit(task("small", work_s=5.0))
        self.c.net.run_for(2)
        self.assertEqual(self.me.state(tid), "running")
        self.me.cancel(tid)
        self.c.net.run_for(10)
        self.assertEqual(self.me.state(tid), "cancelled")
        self.assertIsNone(self.me.jobs[tid]["result"])
        self.assertIsNone(self.n.active)                            # the worker is free again
        nxt = self.me.submit(task("small", work_s=0.2))
        self.c.net.run_for(2)
        self.assertEqual(self.me.state(nxt), "completed")

    def test_cancel_after_it_finished_changes_nothing(self):
        tid = self.me.submit(task("small", work_s=0.2, payload="kept"))
        self.c.net.run_for(3)
        self.assertEqual(self.me.cancel(tid), "completed")
        self.c.net.run_for(1)
        self.assertEqual(self.me.jobs[tid]["result"], {"echo": "kept"})

    def test_a_result_already_on_its_way_survives_a_cancel(self):
        self.c.net.link("coordinator", "mini", latency=0.5)
        tid = self.me.submit(task("small", work_s=1.0, payload="in flight"))
        self.c.net.run_for(2.7)          # 0.5 there + 1 s load + 1 s work: done on the node, result not here yet
        self.assertEqual(self.n.tasks[tid]["state"], "completed")
        self.assertNotEqual(self.me.state(tid), "completed")
        self.me.cancel(tid)
        self.c.net.run_for(5)
        self.assertEqual(self.me.state(tid), "completed")
        self.assertEqual(self.me.jobs[tid]["result"], {"echo": "in flight"})

    def test_a_deadline_ends_a_task_that_overruns_and_frees_the_node(self):
        tid = self.me.submit(task("small", work_s=60.0), deadline_s=3.0)
        self.c.net.run_for(6)
        self.assertEqual(self.me.state(tid), "timed_out")
        self.assertEqual(self.me.explain(tid)[-1], "Stopped: the task's deadline passed.")
        self.assertIsNone(self.n.active)

    def test_a_handler_that_raises_fails_the_task_not_the_node(self):
        def boom(_input, _node):
            raise RuntimeError("no such luck")
        self.n.register("echo.test", boom)
        tid = self.me.submit(task("small"))
        self.c.net.run_for(3)
        self.assertEqual(self.me.state(tid), "failed")
        self.assertIn("no such luck", self.me.jobs[tid]["error"])
        self.assertEqual(self.me.nodes[self.c.fingerprint("mini")]["state"], "online")

    def test_only_allowlisted_handlers_exist(self):
        with self.assertRaisesRegex(ProtocolError, "not an allowlisted task type"):
            self.n.register("shell.exec", lambda i, n: (0, {}))
        with self.assertRaisesRegex(ProtocolError, "unknown task type"):
            protocol.new_task("shell.exec", {"cmd": "id"})
        forged = dict(task("small"), type="shell.exec", state="dispatched")
        self.me.send(self.n.identity.node_id, "task.submit", forged)
        self.c.net.run_for(1)
        self.assertEqual(self.n.executions, 0)
        self.assertIn("no handler for task type 'shell.exec'", " ".join(a.get("reason", "") for a in self.n.audit))

    def test_a_node_speaks_only_for_its_own_tasks(self):
        other = self.c.add("other", models=["small"])
        self.assertTrue(self.c.pair("other"))
        tid = self.me.submit(task("small", work_s=5.0))
        self.c.net.run_for(2)
        self.assertEqual(self.me.jobs[tid]["node"], self.n.identity.node_id)
        other.send(self.me.identity.node_id, "task.status", {"task_id": tid, "state": "completed", "result": {"x": 1}})
        self.c.net.run_for(0.5)
        self.assertEqual(self.me.state(tid), "running")
        self.assertIn("task.status: no such task on this node", refusals(self.me))

    def test_the_audit_log_holds_metadata_and_never_content(self):
        tid = self.me.submit(task("small", work_s=0.2, payload="SECRET-PAYLOAD"))
        self.c.net.run_for(3)
        self.assertEqual(self.me.state(tid), "completed")
        log = json.dumps(self.me.audit + self.n.audit)
        self.assertNotIn("SECRET-PAYLOAD", log)
        self.assertIn(tid, log)
        actions = {a["action"] for a in self.me.audit + self.n.audit}
        self.assertTrue({"paired", "grant", "task.placed", "task.accept", "task.completed"} <= actions)


class TestFailures(unittest.TestCase):
    def test_a_crashed_worker_is_noticed_and_its_task_moves_on(self):
        c = cluster(seed=8, workers=("mini", "spare"), tps={"mini": 400, "spare": 40}, local_worker=False)
        me, mini, spare = c.coordinator, c.nodes["mini"], c.nodes["spare"]
        tid = me.submit(task("small", work_s=30.0, tokens=2000, payload="survives"))
        c.net.run_for(2)
        self.assertEqual((me.jobs[tid]["node"], me.state(tid)), (mini.identity.node_id, "running"))
        mini.crash()
        t0 = c.net.now()
        c.net.run_for((OFFLINE_AFTER + 2) * HEARTBEAT_S)            # noticed within the documented bound
        rec = me.nodes[c.fingerprint("mini")]
        self.assertEqual(rec["state"], "offline")
        self.assertEqual(me.jobs[tid]["node"], spare.identity.node_id)
        self.assertLess(c.net.now() - t0, 10)
        c.net.run_for(60)
        self.assertEqual(me.state(tid), "completed")
        self.assertEqual(me.jobs[tid]["result"], {"echo": "survives"})
        self.assertEqual(me.jobs[tid]["attempts"], 2)
        self.assertTrue(any(e.startswith("Interrupted on mini (node went offline); recovering, attempt 2 of 3")
                            for e in me.explain(tid)))

    def test_with_no_retries_left_the_task_fails_rather_than_vanishing(self):
        c = cluster(seed=8, local_worker=False)
        me = c.coordinator
        tid = me.submit(task("small", work_s=30.0, retry_policy={"max_retries": 0, "retryable": True}))
        c.net.run_for(2)
        c.nodes["mini"].crash()
        c.net.run_for(10)
        self.assertEqual(me.state(tid), "failed")
        self.assertIn("no retries left", me.jobs[tid]["error"])

    def test_a_task_that_cannot_be_recovered_anywhere_waits_then_resumes(self):
        c = cluster(seed=8, local_worker=False)
        me, mini = c.coordinator, c.nodes["mini"]
        tid = me.submit(task("small", work_s=3.0, payload="again"))
        c.net.run_for(2)
        mini.crash()
        c.net.run_for(8)
        self.assertEqual(me.state(tid), "waiting_for_resources")     # nobody to give it to, and it says so
        self.assertIn("mini: node is offline", me.explain(tid)[-1])
        mini.restart()
        c.net.run_for(40)
        self.assertEqual(me.nodes[c.fingerprint("mini")]["state"], "online")
        self.assertEqual(me.state(tid), "completed")
        self.assertEqual(me.jobs[tid]["result"], {"echo": "again"})

    def test_a_partition_is_a_node_going_away_and_healing_brings_it_back(self):
        c = cluster(seed=8)
        me = c.coordinator
        rec = me.nodes[c.fingerprint("mini")]
        c.net.partition("coordinator", "mini")
        c.net.run_for(3.5)                                           # two whole intervals with no answer
        self.assertEqual(rec["state"], "degraded")
        tid = me.submit(task("small", tokens=100000, work_s=0.2))    # huge, but a degraded node gets nothing
        c.net.run_for(0.5)
        self.assertEqual(me.jobs[tid]["node"], "local")
        c.net.run_for(3)                                             # four: offline
        self.assertEqual(rec["state"], "offline")
        c.net.heal("coordinator", "mini")
        c.net.run_for(BACKOFF_MAX_S + 2)
        self.assertEqual(rec["state"], "online")
        self.assertEqual(rec["history"][-4:], ["online", "degraded", "offline", "online"])

    def test_one_lost_heartbeat_changes_nothing(self):
        c = cluster(seed=8)
        rec = c.coordinator.nodes[c.fingerprint("mini")]
        before = list(rec["history"])
        c.net.partition("coordinator", "mini")
        c.net.run_for(HEARTBEAT_S * 1.2)                             # exactly one beat goes unanswered
        c.net.heal("coordinator", "mini")
        c.net.run_for(5)
        self.assertEqual(rec["history"], before)
        self.assertEqual(rec["misses"], 0)

    def test_a_lossy_jittery_link_does_not_make_a_node_flap_offline(self):
        c = cluster(seed=8)
        c.net.link("coordinator", "mini", latency=0.05, jitter=0.1, loss=0.05)
        rec = c.coordinator.nodes[c.fingerprint("mini")]
        c.net.run_for(120)
        self.assertNotIn("offline", rec["history"])
        self.assertIn(rec["state"], ("online", "busy"))
        self.assertGreater(c.net.stats["dropped"], 0)                # the loss was real

    def test_a_link_slower_than_the_heartbeat_is_slow_not_dead(self):
        c = cluster(seed=8)
        c.net.link("coordinator", "mini", latency=0.8)               # 1.6 s round trip, 1 s heartbeat
        rec = c.coordinator.nodes[c.fingerprint("mini")]
        c.net.run_for(30)
        self.assertEqual(rec["state"], "online")
        self.assertNotIn("offline", rec["history"])
        self.assertAlmostEqual(rec["rtt_s"], 1.6, delta=0.05)

    def test_an_offline_node_is_retried_with_a_bounded_backoff(self):
        c = cluster(seed=8)
        rec = c.coordinator.nodes[c.fingerprint("mini")]
        c.nodes["mini"].crash()
        c.net.run_for(10)
        sent = c.net.stats["sent"]
        c.net.run_for(600)
        pings = c.net.stats["sent"] - sent
        self.assertEqual(rec["interval"], BACKOFF_MAX_S)
        self.assertLessEqual(pings, 600 / BACKOFF_MAX_S + 6)         # not one a second for ten minutes
        self.assertGreaterEqual(pings, 600 / BACKOFF_MAX_S - 2)      # and not given up on either

    def test_the_node_state_machine_refuses_what_it_does_not_list(self):
        c = cluster(seed=8)
        rec = c.coordinator.nodes[c.fingerprint("mini")]
        with self.assertRaisesRegex(ProtocolError, "illegal node transition"):
            c.coordinator._set_state(rec, "discovered")
        self.assertEqual(set(NODE_NEXT), {"discovered", "awaiting_pairing", "paired", "online", "busy", "degraded",
                                          "offline", "revoked"})


def source(total, seed=b"adar"):
    """A payload that exists only as a function of the offset — nothing the size of it is ever held."""
    def read(offset, n):
        out = bytearray()
        block = offset // 32
        while len(out) < n + offset % 32:
            out += hashlib.sha256(seed + block.to_bytes(8, "big")).digest()
            block += 1
        return bytes(out[offset % 32:offset % 32 + n])
    return read


class TestTransfer(unittest.TestCase):
    def setUp(self):
        self.net = SimNet(seed=9)
        self.total = 300_000
        self.read = source(self.total)
        self.want = self.read(0, self.total)
        self.sink = bytearray(self.total)

    def write(self, off, data):
        self.sink[off:off + len(data)] = data

    def test_a_clean_transfer_arrives_whole(self):
        snd, rcv, took = run_transfer(self.net, "a", "b", self.read, self.total, self.write, chunk=16384)
        self.assertTrue(rcv.complete and snd.done)
        self.assertEqual(bytes(self.sink), self.want)
        self.assertGreater(took, 0)

    def test_loss_is_recovered(self):
        self.net.link("a", "b", loss=0.15, latency=0.01)
        _s, rcv, _t = run_transfer(self.net, "a", "b", self.read, self.total, self.write, chunk=16384, rto=0.2)
        self.assertTrue(rcv.complete)
        self.assertEqual(bytes(self.sink), self.want)
        self.assertGreater(self.net.stats["dropped"], 0)

    def test_a_corrupt_chunk_is_rejected_by_its_checksum_and_asked_for_again(self):
        self.net.link("a", "b", corrupt=0.2)
        _s, rcv, _t = run_transfer(self.net, "a", "b", self.read, self.total, self.write, chunk=16384, rto=0.2)
        self.assertTrue(rcv.complete)
        self.assertGreater(rcv.rejected, 0)
        self.assertEqual(bytes(self.sink), self.want)                # none of the damage reached the sink

    def test_an_interrupted_transfer_resumes_where_it_stopped(self):
        self.net.link("a", "b", bps=1e6)
        self.net.after(0.9, lambda: self.net.partition("a", "b"))
        _s, first, _t = run_transfer(self.net, "a", "b", self.read, self.total, self.write, chunk=16384, limit_s=3)
        self.assertFalse(first.complete)
        self.assertTrue(0 < first.have < first.count)
        self.net.heal("a", "b")
        sent_before = self.net.stats["bytes"]
        prefix = hashlib.sha256(self.want[:first.have * 16384])
        _s, second, _t = run_transfer(self.net, "a", "b", self.read, self.total, self.write, chunk=16384,
                                      start=first.have, prefix_digest=prefix)
        self.assertTrue(second.complete)
        self.assertEqual(bytes(self.sink), self.want)
        self.assertLess(self.net.stats["bytes"] - sent_before, self.total)       # only the rest was sent

    def test_a_resume_from_the_wrong_place_fails_the_final_check(self):
        # the receiver believes it already holds 5 chunks, but they are not this payload's: every later
        # chunk is valid on its own, and only the checksum over the whole can tell
        k, tid, chunk = 5, b"t" * 16, 16384
        rcv = transfer.Receiver(tid, self.write, self.total, chunk=chunk, have=k,
                                prefix_digest=hashlib.sha256(b"\0" * (k * chunk)))
        for i in range(k, rcv.count):
            reply = rcv.on_frame(transfer.frame(transfer.DATA, tid, i, self.read(i * chunk, min(chunk, self.total - i * chunk))))
            self.assertEqual(transfer.parse(reply)[0], transfer.ACK)
        true_whole = hashlib.sha256(self.want).digest()
        self.assertIsNone(rcv.on_frame(transfer.frame(transfer.FIN, tid, rcv.count, true_whole)))
        self.assertFalse(rcv.complete)
        self.assertEqual(rcv.failed, "the transfer's final checksum does not match")

    def test_damage_anywhere_in_a_frame_is_caught(self):
        tid = b"t" * 16
        good = transfer.frame(transfer.DATA, tid, 0, self.read(0, 16384))
        for at in (4, 5, 22, 26, 40, 80, len(good) - 1):            # kind, id, index, length, checksum, payload…
            rcv = transfer.Receiver(tid, self.write, self.total, chunk=16384)
            bad = good[:at] + bytes([good[at] ^ 1]) + good[at + 1:]
            reply = rcv.on_frame(bad)
            self.assertEqual((rcv.have, rcv.rejected), (0, 1), at)
            self.assertEqual(transfer.parse(reply)[:3], (transfer.NACK, tid, 0), at)
        self.assertEqual(bytes(self.sink), bytes(self.total))

    def test_a_chunk_out_of_place_is_not_written(self):
        rcv = transfer.Receiver(b"t" * 16, self.write, self.total, chunk=16384)
        reply = rcv.on_frame(transfer.frame(transfer.DATA, b"t" * 16, 3, self.read(3 * 16384, 16384)))
        self.assertEqual(transfer.parse(reply)[:3], (transfer.NACK, b"t" * 16, 0))
        self.assertEqual(rcv.have, 0)
        self.assertEqual(bytes(self.sink), bytes(self.total))
        self.assertIsNone(rcv.on_frame(transfer.frame(transfer.DATA, b"x" * 16, 0, b"other transfer")))

    def test_a_slow_receiver_never_makes_the_sender_buffer_the_payload(self):
        total, chunk, window = 24 * 2**20, 64 * 1024, 4
        read = source(total)
        digest = hashlib.sha256()
        self.net.link("a", "b", bps=4e9)
        tracemalloc.start()
        base = tracemalloc.get_traced_memory()[0]
        snd, rcv, _t = run_transfer(self.net, "a", "b", read, total, lambda off, data: digest.update(data),
                                    chunk=chunk, window=window, consume_s=0.01, rto=5.0)
        peak = tracemalloc.get_traced_memory()[1] - base
        tracemalloc.stop()
        self.assertTrue(rcv.complete)
        self.assertLessEqual(snd.peak_held, window * chunk)          # never more than the window in hand
        self.assertLess(peak, 4 * 2**20)                             # 24 MB moved inside 4 MB of Python memory
        self.assertEqual(snd.held, {})

    def test_a_transfer_does_not_delay_the_control_plane(self):
        c = cluster(seed=9)
        c.net.link("coordinator", "mini", bps=2e6, latency=0.005)
        rec = c.coordinator.nodes[c.fingerprint("mini")]
        total = 4 * 2**20                                            # 16 s of line time
        _s, rcv, took = run_transfer(c.net, "coordinator", "mini", source(total), total, lambda o, d: None,
                                     chunk=64 * 1024, window=8)
        self.assertTrue(rcv.complete)
        self.assertGreater(took, 10)
        self.assertEqual(rec["state"], "online")                     # heartbeats ran on their own channel
        self.assertNotIn("degraded", rec["history"])
        self.assertLess(rec["rtt_s"], 0.1)


class TestSimulationIsLabelledAndCheap(unittest.TestCase):
    def test_every_simulated_node_says_so_everywhere(self):
        c = cluster(seed=10, workers=("mini", "studio"))
        lines = c.describe()
        self.assertEqual(len(lines), 3)
        self.assertTrue(all(line.startswith(f"[{LABEL}]") for line in lines))
        for n in c.nodes.values():
            self.assertTrue(n.capability()["simulated"])
            self.assertTrue(n.identity.node_id.startswith("sim:"))

    def test_the_demo_runs_without_loading_the_engine_or_a_model(self):
        code = ("import sys, adar.sim as s; s.main(); "
                "bad = [m for m in sys.modules if m == 'mlx' or m.startswith(('mlx.', 'engine'))]; "
                "print('LOADED', bad)")
        out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=120)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn("LOADED []", out.stdout)
        body = [line for line in out.stdout.splitlines() if line and not line.startswith("LOADED")]
        self.assertTrue(body and all(line.startswith(f"[{LABEL}]") for line in body), out.stdout)

    def test_ten_workers_cost_kilobytes_not_ten_models(self):
        tracemalloc.start()
        base = tracemalloc.get_traced_memory()[0]
        c = cluster(seed=10, workers=tuple(f"w{i}" for i in range(10)))
        ids = [c.coordinator.submit(task("small", work_s=0.5)) for _ in range(30)]
        c.net.run_for(60)
        peak = tracemalloc.get_traced_memory()[1] - base
        tracemalloc.stop()
        self.assertTrue(all(c.coordinator.state(i) == "completed" for i in ids))
        self.assertLess(peak, 16 * 2**20)

    def test_the_same_seed_gives_the_same_run(self):
        def run(seed):
            c = cluster(seed=seed, workers=("mini", "spare"), tps={"mini": 400})
            c.net.link("coordinator", "mini", latency=0.02, jitter=0.05, loss=0.1)
            ids = [c.coordinator.submit(task("small", work_s=1.0, tokens=500, idempotency_key=f"k{i}"))
                   for i in range(6)]
            c.net.after(3.0, c.nodes["mini"].crash)
            c.net.run_for(90)
            trail = [(a["t"], a["node"], a["action"]) for n in c.nodes.values() for a in n.audit]
            return ([c.coordinator.state(i) for i in ids], c.nodes["mini"].codes, c.net.stats, len(trail),
                    [a[0] for a in trail])
        self.assertEqual(run(11), run(11))
        self.assertNotEqual(run(11)[1], run(12)[1])


if __name__ == "__main__":
    unittest.main()
