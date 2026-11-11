"""The per-layer state a model keeps between tokens, behind one contract.

Two kinds of layer keep state. An attention layer keeps every key and value it has seen (`KVCache`); a
GatedDeltaNet layer keeps a fixed-size recurrent state and a short convolution tail (`SSMCache`). Both
answer the same questions, so the code that reuses a prefix, verifies a draft, stores a conversation on
disk or accounts for memory does not ask which kind it holds:

    append(k, v)     KV only: add tokens, get back everything up to now (a recurrent layer advances itself)
    truncate(n)      forget everything after the first n tokens — or raise, if this state cannot be rewound
    snapshot()       a cheap restore point for the state as it is now
    restore(s)       go back to a snapshot
    arrays           the arrays that make up the state (to evaluate, to count)
    nbytes, dtype    its size, and the type it is stored in
    rewindable       whether truncate() works without a snapshot

There used to be two KVCache classes — this one in model.py and another in dense.py with mlx-lm's verbs
(`update_and_fetch`, `trim`) — so batching, speculation and the SSD prefix store each worked with one of
them only. This is the one class; the mlx-lm verbs are kept as methods, because the loop in compat.py
drives mlx-lm's own caches through the same calls.

What a cache holds is float16: the context budget and the RAM guard count two bytes per element. `dtype`
is there so a test can hold every engine to that (a float32 cache once doubled the memory and halved the
speed of long conversations without any test noticing).
"""
import mlx.core as mx


class KVCache:
    """Keys and values of one attention layer, in buffers grown a step at a time."""
    STEP = step = 256
    rewindable = True

    def __init__(self):
        self.keys = self.values = None
        self.offset = 0

    # ── the contract ──
    def append(self, k, v):
        prev, n = self.offset, k.shape[2]
        if self.keys is None or prev + n > self.keys.shape[2]:
            B, H, _, D = k.shape
            cap = ((prev + n + self.STEP - 1) // self.STEP) * self.STEP
            nk = mx.zeros((B, H, cap - prev, D), k.dtype)
            nv = mx.zeros((B, H, cap - prev, v.shape[3]), v.dtype)
            if self.keys is not None and prev:
                self.keys = mx.concatenate([self.keys[..., :prev, :], nk], axis=2)
                self.values = mx.concatenate([self.values[..., :prev, :], nv], axis=2)
            else:
                self.keys, self.values = nk, nv
        self.offset = prev + n
        self.keys[..., prev:self.offset, :] = k
        self.values[..., prev:self.offset, :] = v
        return self.keys[..., :self.offset, :], self.values[..., :self.offset, :]

    def truncate(self, n):
        self.offset = min(self.offset, n)

    def snapshot(self):
        return self.offset                 # rows past it are simply overwritten later

    def restore(self, snap):
        self.truncate(snap)

    @property
    def arrays(self):
        return [] if self.keys is None else [self.keys, self.values]

    @property
    def nbytes(self):
        return 0 if self.keys is None else self.keys.nbytes + self.values.nbytes

    @property
    def dtype(self):
        """The storage type, or None while empty. Keys and values must agree: a mixed cache is a defect."""
        if self.keys is None:
            return None
        if self.keys.dtype != self.values.dtype:
            raise TypeError(f"keys are {self.keys.dtype} and values {self.values.dtype}: a KV cache holds one type")
        return self.keys.dtype

    # ── speculative verification: keep the first m of the tokens fed since begin_verify ──
    def begin_verify(self):
        self.base = self.offset

    def commit(self, m):
        self.offset = self.base + m

    # ── the names older call sites and mlx-lm's loop use ──
    update = update_and_fetch = append
    trim_to = truncate
    state = arrays

    def is_trimmable(self):
        return True

    def trim(self, n):
        """Forget the newest n tokens; returns how many were forgotten."""
        n = min(n, self.offset)
        self.offset -= n
        return n


class SSMCache:
    """Conv tail + recurrent state of one GatedDeltaNet layer.

    The state is a fixed size whatever the context, and it cannot be rewound: `truncate` only accepts the
    length it is already at, and going back means `restore` to a snapshot taken earlier.

    Speculative verify: begin_verify() remembers the pre-verify state; the layer stashes the recurrence
    inputs; commit(m) keeps the first m tokens by re-running only the (cheap) recurrence on them — no
    second weight pass.
    """
    rewindable = False

    def __init__(self):
        self.conv = None
        self.state = None
        self.offset = 0
        self.verifying = False
        self.pending = None

    def truncate(self, n):
        if n != self.offset:
            raise ValueError(f"a recurrent state at {self.offset} tokens cannot be rewound to {n}: "
                             f"restore a snapshot instead")

    def snapshot(self):
        return (self.conv, self.state, self.offset)

    def restore(self, snap):
        self.conv, self.state, self.offset = snap

    @property
    def arrays(self):
        return [a for a in (self.conv, self.state) if a is not None]

    @property
    def nbytes(self):
        return sum(a.nbytes for a in self.arrays)

    @property
    def dtype(self):
        """The type of the convolution tail — what scales with the model's width; the recurrent state
        itself is float32 by design (the delta rule accumulates in it)."""
        return None if self.conv is None else self.conv.dtype

    def begin_verify(self):
        self.verifying = True
        self.before = (self.state, self.offset)
        self.pending = None

    def commit(self, m):
        from .gated_delta import gated_delta_kernel
        self.verifying = False
        if self.pending is None:
            return
        q, k, v, g, beta, conv_in, kernel = self.pending
        self.pending = None
        S = q.shape[1]
        if m < S:
            state0, off0 = self.before
            _, self.state = gated_delta_kernel(q[:, :m], k[:, :m], v[:, :m], g[:, :m], beta[:, :m], state0)
            self.conv = mx.contiguous(conv_in[:, m:m + kernel - 1])
            self.offset = off0 + m
