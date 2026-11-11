"""Pairing: how two nodes that have only *seen* each other come to trust each other
(docs/adar/SECURITY.md, "Pairing flow"), and the book of peers that results.

The flow — the code is shown by the responder and typed at the initiator, so it never crosses the network
at all:

    A (initiator)                                   B (responder, put into pairing mode by a person)
      pair.request {node_id, eph_A, nonce_A}  ──▶
                                              ◀──   pair.confirm step=challenge {node_id, eph_B, nonce_B}
                                                    B shows a 6-digit code on its own screen
      a person reads the code off B and types it at A
      key = X25519(eph_A, eph_B) bound to the code
      pair.confirm step=proof {mac(key, transcript), sig_A(transcript)}  ──▶
                                                    B recomputes the key from the code it showed:
                                                    wrong mac → refused, and after 3 tries the attempt
                                                    is void; right mac + valid long-term signature → paired
                                              ◀──   pair.confirm step=accept {mac(key, transcript‖"ok"), sig_B}
      A checks both → paired

What each part is for: the code proves a person stood at both machines (an attacker on the LAN never
sees it, and gets three guesses in a million); the ephemeral keys make the session key new each time;
the transcript carries both nonces and both node ids, so a recorded exchange proves nothing when
replayed; the long-term signatures are what gets pinned.

Pairing grants no task permissions. `scope` starts empty and is granted separately, per task type.
"""
import hashlib
import hmac
import secrets

from .identity import IdentityError
from .protocol import HANDLERS, ProtocolError

MAX_TRIES = 3
WINDOW = 64                  # how far behind the newest sequence number a late message may still arrive
PENDING_TTL = 120.0          # seconds a started pairing stays open


class PeerBook:
    """Who this node trusts, what each may ask of it, and the message counters that stop replays."""

    def __init__(self):
        self.peers = {}          # node_id -> {"addr", "paired_at", "scope": [...], "seq_in": int, "seq_out": int}
        self.revoked = set()

    def add(self, node_id, addr, now):
        self.revoked.discard(node_id)        # a fresh, human-verified pairing is the way back
        self.peers[node_id] = {"addr": addr, "paired_at": now, "scope": [], "seq_in": 0, "seq_out": 0,
                               "seen": set()}

    def is_paired(self, node_id):
        return node_id in self.peers

    def grant(self, node_id, task_types):
        unknown = set(task_types) - set(HANDLERS)
        if unknown:
            raise ProtocolError(f"cannot grant task types no handler exists for: {sorted(unknown)}")
        self.peers[node_id]["scope"] = sorted(set(task_types))

    def revoke(self, node_id):
        self.peers.pop(node_id, None)
        self.revoked.add(node_id)

    def accept_seq(self, node_id, seq):
        """True and remembered if `seq` is new from this peer; False for a repeat, or for one so old that
        it has left the window and can no longer be told from a repeat."""
        p = self.peers[node_id]
        if seq <= p["seq_in"] - WINDOW or seq in p["seen"] or seq < 1:
            return False
        p["seen"].add(seq)
        if seq > p["seq_in"]:
            p["seq_in"] = seq
            p["seen"] = {s for s in p["seen"] if s > seq - WINDOW}
        return True

    def next_seq(self, node_id):
        self.peers[node_id]["seq_out"] += 1
        return self.peers[node_id]["seq_out"]

    def by_addr(self, addr):
        return next((n for n, p in self.peers.items() if p["addr"] == addr), None)


def _transcript(a_id, b_id, eph_a, eph_b, nonce_a, nonce_b):
    return "|".join(("adar-pair-v1", a_id, b_id, eph_a, eph_b, nonce_a, nonce_b)).encode()


def _mac(key_hex, data):
    return hmac.new(bytes.fromhex(key_hex), data, hashlib.sha256).hexdigest()


class Pairing:
    """One node's half of the flow, both roles. Holds at most one attempt at a time: the ephemeral key
    belongs to it, and is never shared between two peers."""

    def __init__(self, identity, book, *, show_code=None, rng=None):
        self.identity, self.book = identity, book
        self.show_code = show_code or (lambda code, peer: None)     # the "screen" of this node
        self.rng = rng or secrets.SystemRandom()
        self.accepting = False        # pairing mode: off unless a person turned it on
        self.out = None               # as initiator
        self.pending = None           # as responder

    # ───────────── initiator ─────────────
    def begin(self, addr, now):
        self.out = {"addr": addr, "eph": self.identity.ephemeral_public(), "nonce": secrets.token_hex(16),
                    "t": now, "peer": None}
        return {"node_id": self.identity.node_id, "eph": self.out["eph"], "nonce": self.out["nonce"]}

    def on_challenge(self, addr, body):
        o = self.out
        if o is None or o["addr"] != addr or o["peer"] is not None:
            raise ProtocolError("pair.confirm: no pairing in progress with this node")
        for f in ("node_id", "eph", "nonce"):
            if not isinstance(body.get(f), str):
                raise ProtocolError(f"pair.confirm: challenge is missing {f!r}")
        o["peer"] = {"node_id": body["node_id"], "eph": body["eph"], "nonce": body["nonce"]}

    def prove(self, code):
        """The person typed `code` (read off the other machine). Returns the proof message body."""
        o = self.out
        if o is None or o["peer"] is None:
            raise ProtocolError("no challenge received yet — nothing to confirm")
        p = o["peer"]
        o["key"] = self.identity.exchange(p["eph"], code)
        o["transcript"] = _transcript(self.identity.node_id, p["node_id"], o["eph"], p["eph"], o["nonce"], p["nonce"])
        return {"step": "proof", "node_id": self.identity.node_id, "mac": _mac(o["key"], o["transcript"]),
                "sig": self.identity.sign(o["transcript"])}

    def on_accept(self, addr, body, now):
        o = self.out
        if o is None or o["addr"] != addr or "key" not in o:
            raise ProtocolError("pair.confirm: unexpected acceptance")
        p = o["peer"]
        good = hmac.compare_digest(str(body.get("mac", "")), _mac(o["key"], o["transcript"] + b"|ok"))
        try:
            signed = self.identity.verify(p["node_id"], o["transcript"], str(body.get("sig", "")))
        except (IdentityError, ValueError):
            signed = False
        self.out = None
        if not (good and signed):
            raise ProtocolError("pairing failed: the other node's confirmation does not check out")
        self.book.add(p["node_id"], addr, now)
        return p["node_id"]

    # ───────────── responder ─────────────
    def on_request(self, addr, body, now):
        if not self.accepting:
            raise ProtocolError("pair.request: this node is not in pairing mode")
        if self.pending and now - self.pending["t"] < PENDING_TTL and self.pending["addr"] != addr:
            raise ProtocolError("pair.request: another pairing is in progress")
        for f in ("node_id", "eph", "nonce"):
            if not isinstance(body.get(f), str):
                raise ProtocolError(f"pair.request: missing {f!r}")
        if body["node_id"] == self.identity.node_id:
            raise ProtocolError("pair.request: a node cannot pair with itself")
        code = f"{self.rng.randrange(10**6):06d}"
        self.pending = {"addr": addr, "peer": body["node_id"], "eph_peer": body["eph"], "nonce_peer": body["nonce"],
                        "eph": self.identity.ephemeral_public(), "nonce": secrets.token_hex(16), "code": code,
                        "tries": 0, "t": now}
        self.show_code(code, body["node_id"])
        return {"step": "challenge", "node_id": self.identity.node_id, "eph": self.pending["eph"],
                "nonce": self.pending["nonce"]}

    def on_proof(self, addr, body, now):
        p = self.pending
        if p is None or p["addr"] != addr or body.get("node_id") != p["peer"]:
            raise ProtocolError("pair.confirm: no pairing in progress with this node")
        if now - p["t"] > PENDING_TTL:
            self.pending = None
            raise ProtocolError("pair.confirm: the pairing attempt expired")
        key = self.identity.exchange(p["eph_peer"], p["code"])
        transcript = _transcript(p["peer"], self.identity.node_id, p["eph_peer"], p["eph"], p["nonce_peer"], p["nonce"])
        good = hmac.compare_digest(str(body.get("mac", "")), _mac(key, transcript))
        try:
            signed = self.identity.verify(p["peer"], transcript, str(body.get("sig", "")))
        except (IdentityError, ValueError):
            signed = False
        if not (good and signed):
            p["tries"] += 1
            if p["tries"] >= MAX_TRIES:
                self.pending = None
                raise ProtocolError("pairing failed: wrong code three times — start again")
            raise ProtocolError("pairing failed: the code does not match")
        self.pending = None                   # one use: the same proof again finds nothing to match
        self.book.add(p["peer"], addr, now)
        return p["peer"], {"step": "accept", "node_id": self.identity.node_id, "mac": _mac(key, transcript + b"|ok"),
                           "sig": self.identity.sign(transcript)}
