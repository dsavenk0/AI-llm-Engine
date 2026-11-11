"""Authenticated control-plane envelopes between paired nodes.

The design calls for mutual TLS with pinned keys before a task message is even parsed
(docs/adar/SECURITY.md, "Not built yet"). A TLS stack needs a real socket and a real peer; what is
enforced meanwhile is the property TLS is there to give the layer above it: **every message is attributable to a paired identity,
unmodified, and not a replay.** Each message is signed by the sender's long-term key over the verb, the
body, the recipient and a per-pair sequence number; the receiver checks the signature against the key it
pinned at pairing and refuses a sequence number it has already seen (a sliding window, as in IPsec and
DTLS: a message that merely arrives late is still taken, the same message twice is not).

The only messages accepted outside an envelope are the pairing ones — by definition there is no pinned key
yet — and those are authenticated by the pairing flow itself (adar/pairing.py).
"""
import json

from . import protocol
from .identity import IdentityError
from .protocol import ProtocolError

UNSEALED = ("pair.request", "pair.confirm")        # everything else must arrive in an envelope


def _signed_bytes(verb, body, sender, recipient, seq):
    return json.dumps([protocol.VERSION, verb, sender, recipient, seq, body], sort_keys=True,
                      separators=(",", ":")).encode()


def seal(identity, recipient, seq, verb, body):
    sig = identity.sign(_signed_bytes(verb, body, identity.node_id, recipient, seq))
    return protocol.encode(verb, {"from": identity.node_id, "to": recipient, "seq": seq, "sig": sig, "data": body})


def peek(raw):
    """(verb, body) of any control message, validated structurally only. Nothing here is trusted yet."""
    return protocol.decode(raw)


def unseal(identity, verb, env, book):
    """(sender, seq, data) of an envelope `peek` returned, or ProtocolError. `book` is the node's PeerBook:
    it says who is paired and which sequence numbers it has already accepted from them."""
    for field in ("from", "to", "seq", "sig", "data"):
        if field not in env:
            raise ProtocolError(f"{verb}: not an authenticated envelope (missing {field!r})")
    sender, seq, data = env["from"], env["seq"], env["data"]
    if not isinstance(seq, int) or isinstance(seq, bool) or not isinstance(data, dict):
        raise ProtocolError(f"{verb}: malformed envelope")
    if env["to"] != identity.node_id:
        raise ProtocolError(f"{verb}: addressed to another node")
    if not book.is_paired(sender):
        raise ProtocolError(f"{verb}: sender is not a paired node")
    try:
        ok = identity.verify(sender, _signed_bytes(verb, data, sender, env["to"], seq), env["sig"])
    except (IdentityError, ValueError) as e:
        raise ProtocolError(f"{verb}: {e}") from None
    if not ok:
        raise ProtocolError(f"{verb}: signature does not match the sender's pinned key")
    if not book.accept_seq(sender, seq):
        raise ProtocolError(f"{verb}: replayed message (seq {seq})")
    return sender, seq, data
