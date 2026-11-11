"""Node identity and the key agreement the pairing flow needs (docs/adar/SECURITY.md, "Identity").

Two backends, chosen at runtime, never silently mixed:

* **real** — Ed25519 (identity, signatures) and X25519 (pairing key agreement) from the `cryptography`
  package. This is the production path. `cryptography` is *not* a hard dependency: this project ships five
  of them on purpose, and ADAR is opt-in and additive (docs/adar/ARCHITECTURE.md's standalone guarantee), so
  it installs the same way the optional mlx-lm backend already does — lazily imported, with a setup command
  that says what it's for.
* **sim** — a stdlib-only (HMAC-SHA256) stand-in with the same interface, for Phase 2's local simulation,
  where the point is to exercise the orchestration (discovery → pairing → task → result), not to ship
  cryptography. It is **structurally prevented** from being used outside simulation: `load_or_create` only
  returns a sim identity when `simulated=True` is passed explicitly, and `Identity.simulated` travels with
  every key, so a pairing record made by a simulated node is recognisable as such forever.

Never hand-roll the real thing: without `cryptography` installed, a *real* (non-simulated) node refuses to
start rather than falling back to something weaker that looks the same from the outside.
"""
import base64
import hashlib
import hmac
import json
import os
import secrets


class IdentityError(RuntimeError):
    pass


SETUP_HINT = ("real cluster identity needs the `cryptography` package (Ed25519/X25519): "
              "install it with  ./llm setup --cluster")


def _crypto():
    try:
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric import ed25519, x25519
        return ed25519, x25519, serialization
    except ImportError:
        return None


def crypto_available():
    return _crypto() is not None


def _b64(raw):
    return base64.b64encode(raw).decode()


def _unb64(text):
    return base64.b64decode(text.encode())


class Identity:
    """A node's long-term identity. `node_id` is the public key, as docs/adar/PROTOCOL.md specifies
    (`"ed25519:<base64>"`), so it stays stable when the hostname or IP changes."""

    def __init__(self, private_raw, simulated):
        self.simulated = bool(simulated)
        self._private_raw = private_raw
        if self.simulated:
            self._public_raw = hashlib.sha256(b"adar-sim-pub" + private_raw).digest()
        else:
            ed25519, _x, serialization = _crypto() or (None, None, None)
            if ed25519 is None:
                raise IdentityError(SETUP_HINT)
            key = ed25519.Ed25519PrivateKey.from_private_bytes(private_raw)
            self._public_raw = key.public_key().public_bytes(
                encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw)

    @property
    def node_id(self):
        kind = "sim" if self.simulated else "ed25519"
        return f"{kind}:{_b64(self._public_raw)}"

    @property
    def fingerprint(self):
        """What discovery advertises — a hash of the public key, not the key itself (SECURITY.md: the
        unauthenticated advertisement carries a fingerprint and nothing else)."""
        return hashlib.sha256(self._public_raw).hexdigest()[:16]

    def sign(self, data):
        if self.simulated:
            # the key a simulated verifier derives from the public key (see `verify`): signing with the
            # stored private bytes instead made every simulated signature fail its own verification
            return _b64(hmac.new(_sim_private_for(self._public_raw), data, hashlib.sha256).digest())
        ed25519, _x, _s = _crypto()
        key = ed25519.Ed25519PrivateKey.from_private_bytes(self._private_raw)
        return _b64(key.sign(data))

    def verify(self, node_id, data, signature):
        """True when `signature` over `data` really comes from `node_id`'s private key. A simulated
        identity can only verify simulated peers and vice versa — the `kind` prefix makes a mismatch
        explicit instead of a confusing verification failure."""
        kind, _, pub = node_id.partition(":")
        if (kind == "sim") != self.simulated:
            raise IdentityError(f"identity kind mismatch: this node is "
                                f"{'simulated' if self.simulated else 'real'}, peer id says {kind!r}")
        if self.simulated:
            # a simulated peer's "private key" is derivable from its public one on purpose: the simulation
            # proves the *flow*, and pretending otherwise would be security theatre in a test harness
            expect = hmac.new(_sim_private_for(_unb64(pub)), data, hashlib.sha256).digest()
            return hmac.compare_digest(_b64(expect), signature)
        ed25519, _x, _s = _crypto()
        try:
            ed25519.Ed25519PublicKey.from_public_bytes(_unb64(pub)).verify(_unb64(signature), data)
            return True
        except Exception:
            return False

    def exchange(self, peer_public_b64, code):
        """The pairing session key: HKDF-like derivation over the X25519 shared secret bound to the
        human-verified code, so the key is useless to anyone who never saw the code on the other screen
        (docs/adar/SECURITY.md, "Pairing flow")."""
        if self.simulated:
            # symmetric in the two *ephemeral* public keys, as X25519 is. (It mixed the peer's ephemeral
            # key with this node's long-term one, so the two sides never derived the same session key.)
            mine, theirs = _unb64(self._ephemeral_public), _unb64(peer_public_b64)
            shared = hashlib.sha256(b"adar-sim-ecdh" + min(mine, theirs) + max(mine, theirs)).digest()
        else:
            _e, x25519, serialization = _crypto()
            eph = x25519.X25519PrivateKey.from_private_bytes(self._ephemeral)
            shared = eph.exchange(x25519.X25519PublicKey.from_public_bytes(_unb64(peer_public_b64)))
        return hmac.new(shared, b"adar-pair-v1|" + str(code).encode(), hashlib.sha256).hexdigest()

    def ephemeral_public(self):
        """Fresh X25519 public key for one pairing attempt (never reused across peers)."""
        if self.simulated:
            self._ephemeral = secrets.token_bytes(32)
            self._ephemeral_public = _b64(hashlib.sha256(b"adar-sim-eph" + self._ephemeral).digest())
            return self._ephemeral_public
        _e, x25519, serialization = _crypto()
        self._ephemeral = x25519.X25519PrivateKey.generate().private_bytes(
            encoding=serialization.Encoding.Raw, format=serialization.PrivateFormat.Raw,
            encryption_algorithm=serialization.NoEncryption())
        key = x25519.X25519PrivateKey.from_private_bytes(self._ephemeral)
        return _b64(key.public_key().public_bytes(
            encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw))


def fingerprint_of(node_id):
    """The fingerprint a node with this id advertises — what lets a pairing be checked against the
    advertisement that led to it."""
    return hashlib.sha256(_unb64(node_id.partition(":")[2])).hexdigest()[:16]


def _sim_private_for(public_raw):
    """Simulation only: the inverse of Identity's sim public-key derivation, so a simulated verifier can
    check a simulated peer's signature without a key exchange. Deliberately trivial — see the module
    docstring on why a simulated identity is never allowed outside simulation."""
    return hashlib.sha256(b"adar-sim-priv" + public_raw).digest()


def _sim_keypair(seed):
    """A deterministic simulated private key, so a simulated node keeps its identity across restarts in a
    test without needing a keystore."""
    return hashlib.sha256(b"adar-sim-seed" + seed.encode()).digest()


def load_or_create(root, *, simulated=False, seed=None):
    """The node's own identity, created on first use. Real keys live in `<root>/identity.json` with 0600
    permissions; Keychain storage (docs/adar/SECURITY.md, "Not built yet") is the macOS app's half of this and is wired up
    when the app gains a cluster surface — the CLI's own secret files already follow this same pattern.

    `simulated=True` must be passed explicitly; nothing infers it."""
    if simulated:
        return Identity(_sim_keypair(seed or "sim"), simulated=True)
    if not crypto_available():
        raise IdentityError(SETUP_HINT)
    path = os.path.join(root, "identity.json")
    if os.path.isfile(path):
        with open(path) as f:
            saved = json.load(f)
        return Identity(_unb64(saved["private"]), simulated=False)
    ed25519, _x, serialization = _crypto()
    raw = ed25519.Ed25519PrivateKey.generate().private_bytes(
        encoding=serialization.Encoding.Raw, format=serialization.PrivateFormat.Raw,
        encryption_algorithm=serialization.NoEncryption())
    os.makedirs(root, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump({"private": _b64(raw), "created": True}, f)
    return Identity(raw, simulated=False)
