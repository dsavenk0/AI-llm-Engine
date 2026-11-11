# ADAR — Security and pairing

**Discovery never implies trust, and nothing an unpaired Mac sends is acted on.** Every rule below exists
to enforce that sentence.

Implemented and tested in simulation: identity, the pairing flow, per-node scopes, revocation, message
authentication and the handler allowlist (`adar/identity.py`, `pairing.py`, `wire.py`, `node.py`). Not
built: TLS, Keychain storage, the on-disk ledger and audit file, connection rate limiting — listed at the
end.

## Identity

Each node has an Ed25519 key pair, generated on first use. The public key *is* the node's identity
(`node_id`, `"ed25519:<base64>"`), so it stays the same when the hostname or the IP changes. Discovery
advertises only a fingerprint of it.

The real keys come from the `cryptography` package, which is optional and imported lazily; without it a
real node refuses to start rather than fall back to something weaker that looks the same from outside. The
simulation uses a stdlib stand-in with the same interface, whose ids start with `sim:`; a simulated and a
real identity cannot verify each other.

## Pairing flow

```
A (initiator)                                    B (put into pairing mode by a person)
  pair.request {node_id, eph_A, nonce_A}   ──▶
                                           ◀──   pair.confirm challenge {node_id, eph_B, nonce_B}
                                                 B shows a 6-digit code on its own screen
  a person reads the code off B and types it at A
  key = X25519(eph_A, eph_B), bound to the code
  pair.confirm proof {mac, sig_A}          ──▶
                                                 B recomputes the key from the code it showed
                                           ◀──   pair.confirm accept {mac, sig_B}
  both sides now pin the other's long-term key
```

What each part is for:

* **The code** proves a person stood at both machines. It never crosses the network; an attacker on the
  LAN gets three guesses in a million, after which the attempt is void and has to be started again.
* **The ephemeral keys** make the session key new every time.
* **The transcript** that both sides sign and MAC carries both node ids and both nonces, so a recorded
  exchange proves nothing when replayed.
* **The long-term signatures** are what gets pinned — the same trust model as SSH's `known_hosts`: no CA,
  no shared cluster secret, revocable per node.

A node that nobody put into pairing mode ignores pairing requests. A pairing whose key does not match the
fingerprint that was advertised is undone. One fingerprint advertised from two places is refused, and
neither is paired.

## After pairing

Every message is signed, addressed and numbered (`PROTOCOL.md`, "Envelope"). That gives the layers above
what they rely on: each message is attributable to a paired identity, unmodified, meant for this node, and
not a replay.

## Authorization

Pairing grants **no** task permissions. A scope — a list of task types — is granted separately, per peer,
and starts empty. The permission is the worker's to give: it checks every task against its own peer book,
whatever the coordinator believes it was granted.

Three independent checks stand between a peer and anything running, in this order:

1. the message is from a paired identity and is not a replay;
2. this pairing was granted this task type;
3. the type is one this node ships a handler for.

There is no path by which a peer supplies a command, a script or a binary: the allowlist is the only
dispatch table, and registering a handler outside it is refused.

## Revocation

Revoking a peer removes it from the peer book and tells it. From then on its messages are refused outright,
with the reason recorded. Only a new, human-verified pairing leads back.

## Audit trail

Every pairing, grant, revocation, placement, acceptance, refusal and outcome is recorded with a timestamp
and the node. Task input and output are never recorded — only metadata (task id, type, reason).

## Refused by design

Arbitrary remote execution. Silent pairing. Task submission from an unpaired node. A shared secret for the
whole cluster. Copying a credential between nodes — each node's engine API key stays where it is.

## Not built yet

| Item | Why it waits |
|---|---|
| TLS 1.3 with the pinned key in place of CA validation | needs a real socket; message signatures cover the same properties meanwhile, except confidentiality — the simulated wire is not encrypted |
| Keychain storage of the private key and pairing records | belongs to the macOS app; reuses `app/editor/Endpoint.swift`'s `Keychain` with its own service string |
| `~/.llm/adar/peers.json` and `audit.log` on disk | the peer book and the audit trail live in memory |
| Rate limiting of connection attempts | only pairing attempts are limited today |
| The real-cryptography path | written, not exercised: `cryptography` is not installed where the tests run, and that test is skipped with the reason |
