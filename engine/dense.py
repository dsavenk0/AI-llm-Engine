"""A dense decoder-only transformer on MLX, built from GGUF weights (no mlx-lm, no llama.cpp).

One implementation covers the Llama family as ggml names it: llama (incl. Mistral, SmolLM2, Llama 3.x),
qwen2 (incl. DeepSeek-R1-Distill-Qwen, MiMo's base), qwen3, granite, and Gemma 3 (sliding-window layers in a 5:1 pattern,
two rope bases, a norm after the attention and after the feed-forward, GeGLU). Mixture-of-experts feed-forwards
(Mixtral, Qwen3-MoE, Granite-MoE) keep the experts stacked and run only the routed ones with `gather_qmm`.

Qwen3.5 (`qwen35`, and `qwen35moe` with routed experts plus a gated shared expert) is a hybrid: three of every four
layers are GatedDeltaNet linear attention with a recurrent state (engine/cache.py SSMCache, the delta rule on
engine/gated_delta.py's Metal kernel), the fourth is gated full attention (the query projection also yields a
sigmoid gate on the output). Its GGUF follows llama.cpp's converter: A is stored as -exp(A_log), the RMSNorm
weights other than the gated one have their +1 folded in, and the value heads of every linear-attention tensor
are in ggml's *tiled* order — so the key/query heads are tiled (not repeated per group) to meet them, and no
weight has to be permuted back.
Weights stay in the quantized form of the
file (engine/gguf_tensors.py); the residual stream is float32, matmul inputs and the KV cache float16.

Everything the architectures differ in is data in `Config`, so adding a family is a few lines there plus a
verification against llama.cpp — not a new model class.
"""
import os
import re
import time
import hashlib
import struct
import tempfile
import zlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import mlx.core as mx
import numpy as np


_PACK_MAGIC = b"LLMPK01\0"
_PACK_HEADER = struct.Struct("<8sIIQI")                 # magic, expert id, bits, payload bytes, crc32
_PACK_LOCK = __import__("threading").Lock()
_PACK_USAGE = {}


def _packed_usage(root):
    total, files = 0, []
    for dp, _, names in os.walk(root):
        for name in names:
            if not name.endswith(".bin"):
                continue
            path = os.path.join(dp, name)
            try:
                st = os.stat(path)
            except OSError:
                continue
            total += st.st_size
            files.append((st.st_mtime_ns, st.st_size, path))
    return total, files


def _reserve_packed_space(root, limit, incoming):
    """Bound the shared transformed-expert cache. Files are disposable and oldest-use wins eviction."""
    with _PACK_LOCK:
        total, files = _packed_usage(root) if root not in _PACK_USAGE else (_PACK_USAGE[root], None)
        if total + incoming > limit:
            if files is None:
                total, files = _packed_usage(root)
            for _, size, path in sorted(files):
                try:
                    os.remove(path)
                    total -= size
                except OSError:
                    pass
                if total + incoming <= limit:
                    break
        allowed = total + incoming <= limit
        # Reserve immediately so concurrent writers cannot all admit themselves against the same free bytes.
        # A writer that fails before its atomic replace releases the reservation in `save`.
        _PACK_USAGE[root] = max(0, total) + (incoming if allowed else 0)
        return allowed


def _release_packed_space(root, size):
    with _PACK_LOCK:
        if root in _PACK_USAGE:
            _PACK_USAGE[root] = max(0, _PACK_USAGE[root] - size)


class PackedExpertCache:
    """Persistent, checksummed GGUF→MLX expert transforms.

    A GGUF miss is expensive mostly because its quant blocks are decoded and repacked, not because the SSD is
    slow. One file per transformed expert makes later evictions and later process starts read the exact packed
    `weight/scales/biases` bytes that MLX consumes. The directory key includes source identity and tensor layout;
    changing the GGUF cannot reuse an old transform. Files are cache data, written owner-only through atomic rename.
    """

    def __init__(self, gf, info, n_exp, rows, root, max_bytes, layout):
        self.enabled = bool(root and max_bytes > _PACK_HEADER.size)
        self.root, self.max_bytes = os.path.expanduser(root or ""), int(max_bytes)
        self.dir = ""
        self.shapes = tuple(x[0] for x in layout[1:])
        self.dtypes = tuple(x[1] for x in layout[1:])
        self.bits = layout[0]
        self.payload_bytes = sum(int(np.prod(shape)) * dtype.itemsize
                                 for shape, dtype in zip(self.shapes, self.dtypes))
        if not self.enabled or self.payload_bytes + _PACK_HEADER.size > self.max_bytes:
            self.enabled = False
            return
        try:
            st = os.stat(gf.path)
            identity = repr((os.path.realpath(gf.path), st.st_dev, st.st_ino, st.st_size,
                             st.st_mtime_ns, st.st_ctime_ns, gf.data_start,
                             info.name, info.dims, info.type, info.offset, n_exp, rows, self.bits)).encode()
            key = hashlib.sha256(identity).hexdigest()[:32]
            self.dir = os.path.join(self.root, key)
            os.makedirs(self.dir, mode=0o700, exist_ok=True)
            os.chmod(self.dir, 0o700)
        except OSError:
            self.enabled = False

    def _path(self, expert):
        return os.path.join(self.dir, f"{expert:06d}.bin")

    def load(self, expert):
        if not self.enabled:
            return None
        path = self._path(expert)
        try:
            with open(path, "rb") as f:
                data = f.read()
            magic, found, bits, nbytes, checksum = _PACK_HEADER.unpack_from(data)
            payload = memoryview(data)[_PACK_HEADER.size:]
            if (magic != _PACK_MAGIC or found != expert or bits != self.bits or nbytes != self.payload_bytes
                    or len(payload) != nbytes or zlib.crc32(payload) != checksum):
                raise ValueError("stale or corrupt packed expert")
            arrays, offset = [], 0
            for shape, dtype in zip(self.shapes, self.dtypes):
                count = int(np.prod(shape))
                arrays.append(np.frombuffer(payload, dtype=dtype, count=count, offset=offset).reshape(shape))
                offset += count * dtype.itemsize
            try:
                os.utime(path, None)
            except OSError:
                pass
            return (*arrays, self.bits), len(data)
        except (OSError, ValueError, struct.error):
            try:
                size = os.path.getsize(path)
            except OSError:
                size = 0
            try:
                os.remove(path)
            except OSError:
                pass
            else:
                _release_packed_space(self.root, size)
            return None

    def save(self, expert, arrays):
        if not self.enabled:
            return
        path = self._path(expert)
        if os.path.exists(path):
            return
        payload = b"".join(np.ascontiguousarray(a).tobytes() for a in arrays[:3])
        if len(payload) != self.payload_bytes:
            return
        blob = _PACK_HEADER.pack(_PACK_MAGIC, expert, self.bits, len(payload), zlib.crc32(payload)) + payload
        if not _reserve_packed_space(self.root, self.max_bytes, len(blob)):
            return
        try:
            fd, tmp = tempfile.mkstemp(prefix=".expert-", dir=self.dir)
        except OSError:
            _release_packed_space(self.root, len(blob))
            return
        committed = False
        try:
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "wb") as f:
                f.write(blob)
            os.replace(tmp, path)
            committed = True
        except OSError:
            try:
                os.close(fd)
            except OSError:
                pass
            try:
                os.remove(tmp)
            except OSError:
                pass
        finally:
            if not committed:
                _release_packed_space(self.root, len(blob))


@dataclass
class Config:
    arch: str
    n_layer: int
    n_embd: int
    n_head: int
    n_head_kv: int
    n_ff: int
    head_dim: int
    vocab: int
    eps: float = 1e-5
    rope_base: float = 10000.0
    rope_dims: int = 0
    rope_traditional: bool = False       # interleaved pairs (ggml "norm") vs half-split (ggml "neox")
    rope_freqs: object = None            # per-dimension factors (Llama 3.1 scaling), or None
    qkv_bias: bool = False
    qk_norm: bool = False
    tied_output: bool = False
    context: int = 4096
    attn_scale: float = 0.0              # 0 → 1/sqrt(head_dim)
    embed_scale: float = 1.0             # granite / minicpm / gemma style multipliers
    logit_scale: float = 1.0
    residual_scale: float = 1.0
    attn_softcap: float = 0.0            # Gemma 2: tanh soft-cap on the attention scores (0 = none)
    final_softcap: float = 0.0          # Gemma 2: tanh soft-cap on the output logits (0 = none)
    gelu: bool = False                   # GeGLU (tanh approximation) instead of SwiGLU
    sandwich: bool = False               # extra norms on the attention and feed-forward outputs (Gemma 2/3)
    window: int = 0                      # sliding-window size of the local layers, 0 = none
    window_pattern: int = 0              # every `pattern`-th layer is global, the rest local (Gemma 3: 6)
    rope_base_local: float = 0.0         # rope base of the local layers
    rope_scale: float = 1.0              # linear position scaling of the global layers (Gemma 3 4B+: 1/8)
    n_expert: int = 0                    # mixture of experts: experts per layer and how many a token uses
    n_expert_used: int = 0
    expert_norm: bool = True             # the chosen experts' weights are renormalised to sum to 1
    expert_scale: float = 1.0            # a multiplier on the chosen experts' weights (llama.cpp: expert_weights_scale)
    # Qwen3.5 hybrid: every `full_attn_interval`-th layer is attention, the others GatedDeltaNet
    full_attn_interval: int = 0
    attn_gated: bool = False             # the query projection yields [query | gate] per head
    ssm_conv: int = 0                    # depthwise causal convolution width
    ssm_dk: int = 0                      # key (= value) head size of the linear attention
    ssm_nk: int = 0                      # key heads
    ssm_nv: int = 0                      # value heads (a multiple of the key heads)
    # the linear-attention output projection reads value heads grouped by key head (Prism `gdn_v_grouped`: its
    # rotated columns could not be moved to ggml's tiled order), so the heads are regrouped before it
    gdn_v_grouped: bool = False
    rotated: bool = False                # Hadamard-rotated weights (engine/rotation.py)
    extra: dict = field(default_factory=dict)

    def is_local(self, i):
        return bool(self.window and self.window_pattern and (i % self.window_pattern) != self.window_pattern - 1)

    def is_recurrent(self, i):
        return bool(self.full_attn_interval) and (i + 1) % self.full_attn_interval != 0


# ggml's rope type per architecture (llama.cpp: llama_model_rope_type)
ROPE_TRADITIONAL = {"llama": True, "qwen2": False, "qwen3": False, "gemma2": False, "gemma3": False,
                    "granite": True, "granitemoe": True, "qwen3moe": False, "minicpm": True,
                    "qwen35": False, "qwen35moe": False}
HYBRID = ("qwen35", "qwen35moe")
SUPPORTED = tuple(ROPE_TRADITIONAL)


class Rotation:
    """The activation side of a Hadamard-rotated checkpoint (engine/rotation.py): a matrix stored as W·D·H
    multiplies H·(D·x) — `fwd`, blockwise normalized Walsh-Hadamard after the signs of x's width — and a rotated
    embedding row comes back with `inv`. Computed in float32 like the reference runtimes, returned in x's type.

    Projections that share an input (q/k/v, qkv/z, gate/up) are called one after another with the very same
    array, so the last transform is kept and reused: each distinct input is rotated once."""

    def __init__(self, block, signs):
        self.block, self.scale = block, block ** -0.5
        self.signs = {int(w): mx.array(v, dtype=mx.float32) for w, v in signs.items()}
        self._memo = None                                  # (input, rotated) — one tuple, swapped atomically

    def fwd(self, x):
        memo = self._memo
        if memo is not None and memo[0] is x:
            return memo[1]
        shape = x.shape
        y = x.astype(mx.float32) * self.signs[shape[-1]]
        y = mx.hadamard_transform(y.reshape(-1, self.block), scale=self.scale).reshape(shape).astype(x.dtype)
        self._memo = (x, y)
        return y

    def inv(self, x):
        shape = x.shape
        y = mx.hadamard_transform(x.astype(mx.float32).reshape(-1, self.block), scale=self.scale).reshape(shape)
        return (y * self.signs[shape[-1]]).astype(x.dtype)


class QLinear:
    """y = x @ W.T for a weight kept in MLX's affine-quantized form (or dense). `rot`: the weight is stored
    rotated, so its input is rotated first."""

    def __init__(self, w, scales=None, biases=None, bits=0, bias=None, gs=32, rot=None):
        self.w, self.scales, self.biases, self.bits, self.bias = w, scales, biases, bits, bias
        self.gs, self.rot = gs, rot

    def __call__(self, x):
        if self.rot is not None:
            x = self.rot.fwd(x)
        if self.bits:
            y = mx.quantized_matmul(x, self.w, self.scales, self.biases, transpose=True, group_size=self.gs,
                                    bits=self.bits)
        else:
            y = x @ self.w.T
        return y + self.bias if self.bias is not None else y

    @property
    def nbytes(self):
        n = self.w.nbytes + (self.scales.nbytes + self.biases.nbytes if self.bits else 0)
        return n + (self.bias.nbytes if self.bias is not None else 0)


class Embedding:
    """`rot`: the rows are stored rotated (W·D·H, like every rotated matrix) and come back with the inverse."""

    def __init__(self, w, scales=None, biases=None, bits=0, gs=32, rot=None):
        self.w, self.scales, self.biases, self.bits, self.gs, self.rot = w, scales, biases, bits, gs, rot

    def __call__(self, ids):
        if self.bits:
            x = mx.dequantize(self.w[ids], self.scales[ids], self.biases[ids], group_size=self.gs, bits=self.bits)
        else:
            x = self.w[ids]
        return self.rot.inv(x) if self.rot is not None else x

    def as_linear(self):
        # a tied head over rotated rows: (W·D·H)·(H·D·x) = W·x, i.e. the head rotates its input
        return QLinear(self.w, self.scales, self.biases, self.bits, gs=self.gs, rot=self.rot)


class Experts:
    """The stacked weights of one projection of all experts: [E, rows, packed cols] (+ scales, biases)."""

    def __init__(self, w, scales=None, biases=None, bits=0):
        self.w, self.scales, self.biases, self.bits = w, scales, biases, bits

    def __call__(self, x, idx, sorted_idx):
        if self.bits:
            return mx.gather_qmm(x, self.w, self.scales, self.biases, rhs_indices=idx, transpose=True,
                                 group_size=32, bits=self.bits, sorted_indices=sorted_idx)
        return mx.gather_mm(x, self.w.swapaxes(-1, -2), rhs_indices=idx, sorted_indices=sorted_idx)

    @property
    def nbytes(self):
        return self.w.nbytes + (self.scales.nbytes + self.biases.nbytes if self.bits else 0)


class StreamedExperts:
    """Like `Experts`, but the stacked tensor is never decoded in full. One quantized projection
    (gate/up/down) of one MoE layer, backed by the memory-mapped GGUF file: only the experts actually routed
    to are decoded, into a small, fixed-size LRU-cached pool — a global expert id always decodes to the same
    bytes, it is simply fetched on demand instead of kept resident from model load onward. This is the
    Colibri/kimi-k3-in-c idea (stream the part of a huge MoE model that is actually used, cache it, evict the
    coldest entry when the budget is full) adapted to this engine's existing GGUF reader: `gguf_tensors.py`
    already memory-maps the file and knows every tensor's exact byte offset (`TensorInfo.offset`), and experts
    are stored contiguously (`rows` quant-block rows per expert, in expert order) — so one expert's bytes are
    just an offset computation away, decoded with the very same `load_matrix` chunked-dequant code the eager
    path already uses, not a new quantization format.

    `sorted_indices` is only a performance hint to `gather_qmm` (confirmed from its own docstring: "may allow
    a faster implementation," not a correctness requirement), so remapping a global expert id to its cache
    slot after `_gather_sort` has already grouped by the *global* id costs a little of that grouping benefit
    when the remap reorders them — never correctness; `sorted_indices=False` is passed here unconditionally,
    honestly, rather than claim an ordering that no longer holds.

    One real constraint, stated plainly rather than hidden: `gather_qmm` needs every expert a *single* call
    routes to resident at once, so `expert_cache` must be at least as large as the most distinct experts any
    one forward pass can touch — for one decode step that is `n_expert_used` (small, always satisfiable); for
    a whole prompt prefilled in one pass it can be up to `min(tokens * n_expert_used, n_expert)`, which a
    small cache may not cover. A pass that needs more than the cache holds raises a clear `ValueError` naming
    how much to raise `expert_cache` by — never silently drops one and returns a wrong answer. Streaming earns
    its keep on a model with many more experts than any one chunk realistically touches at once (the
    kimi-k3-in-c case: 896 experts, 16 used per token); it does little for a small MoE model where most
    experts get touched by any prompt of ordinary length anyway.

    Each projection (gate/up/down) of a layer keeps its own independent slot assignment for the same global
    expert id — fine, because `rhs_indices` only has to be internally consistent with that projection's own
    `w`/`scales`/`biases`, never compared across projections. A layer prepares the three projections together,
    so misses decode concurrently; eviction uses the same decayed-frequency policy as the safetensors MoE path,
    with recency only breaking ties."""

    def __init__(self, gf, flat_info, n_exp, rows, cache_size, packed_cache_dir="", packed_cache_bytes=0):
        from . import gguf_tensors as T
        self.gf, self.info, self.n_exp, self.rows = gf, flat_info, n_exp, rows
        self.cache_size = max(1, min(cache_size, n_exp))
        self.resident = {}      # global expert id -> slot
        self.free = list(range(1, self.cache_size))
        self.score = np.zeros(n_exp, np.float32)
        self.tick = np.zeros(n_exp, np.int64)
        self.now = 0
        self._prepared = None
        self.hits = self.misses = 0
        self.evictions = self.decode_ops = self.source_bytes = 0
        self.decode_s = self.packed_s = 0.0
        self.packed_hits = self.packed_bytes = 0
        cols = self.info.shape[1]
        sub = T.TensorInfo(self.info.name, (cols, self.rows), self.info.type, self.info.offset)
        self.packed = PackedExpertCache(gf, self.info, n_exp, rows, packed_cache_dir, packed_cache_bytes,
                                        T.matrix_layout(sub))
        w0, s0, b0, self.bits = self._decode(T, 0)
        self.w = mx.array(np.stack([w0] * self.cache_size))
        self.scales = mx.array(np.stack([s0] * self.cache_size))
        self.biases = mx.array(np.stack([b0] * self.cache_size))
        self.resident[0] = 0

    def _decode(self, T, e):
        """(w, scales, biases, bits) for expert `e` alone — `load_matrix` unchanged, pointed at just that
        expert's row range via a `TensorInfo` whose offset is computed, not a new tensor read from disk."""
        cols = self.info.shape[1]
        _, block, size = T.TYPES[self.info.type]
        per_row = cols // block
        sub = T.TensorInfo(self.info.name, (cols, self.rows), self.info.type,
                           self.info.offset + e * self.rows * per_row * size)
        t0 = time.perf_counter()
        cached = self.packed.load(e)
        if cached is not None:
            out, nbytes = cached
            self.packed_s += time.perf_counter() - t0
            self.packed_hits += 1
            self.packed_bytes += nbytes
            return out
        t0 = time.perf_counter()
        out = T.load_matrix(self.gf, sub)
        self.decode_s += time.perf_counter() - t0
        self.decode_ops += 1
        # Logical source traffic: mmap may satisfy some pages from RAM, but these are the encoded GGUF bytes
        # the decoder had to touch. Keep the source kind in stats so it is not confused with physical SSD I/O.
        self.source_bytes += sub.nbytes
        self.packed.save(e, out)
        return out

    def stage(self, ids):
        """Plan residency and decode misses to NumPy buffers. This phase is safe on a reader thread; committing
        those buffers to MLX is deliberately separate because a worker thread has no Metal stream."""
        from . import gguf_tensors as T
        flat = np.asarray(ids).reshape(-1)
        unique, counts = np.unique(flat, return_counts=True)
        needed = [int(i) for i in unique]
        if len(needed) > self.cache_size:
            raise ValueError(f"{len(needed)} distinct experts are routed to in one pass, more than the "
                             f"expert cache holds ({self.cache_size}); set LLM_GGUF_EXPERT_CACHE to at "
                             f"least {len(needed)} (or LLM_GGUF_STREAM_EXPERTS=0 to disable streaming "
                             f"for this model).")
        self.now += 1
        self.score *= 0.9995
        for e, count in zip(needed, counts):
            self.score[e] += int(count)
        resident, free = dict(self.resident), list(self.free)
        protected = set(needed)
        staged, evictions = [], 0
        for e in needed:
            if e in resident:
                self.hits += 1
                slot = resident[e]
            else:
                self.misses += 1
                if free:
                    slot = free.pop()
                else:
                    old = min((g for g in resident if g not in protected),
                              key=lambda g: (self.score[g], self.tick[g]))
                    slot = resident.pop(old)
                    evictions += 1
                w, s_, b, _ = self._decode(T, e)
                staged.append((slot, w, s_, b))
                resident[e] = slot
        return tuple(needed), resident, free, staged, evictions

    def commit(self, staged, prepared=False):
        """Publish one staged residency update on the caller's MLX/Metal thread."""
        key, resident, free, arrays, evictions = staged
        for slot, w, scales, biases in arrays:
            self.w[slot], self.scales[slot], self.biases[slot] = \
                mx.array(w), mx.array(scales), mx.array(biases)
        self.resident, self.free = resident, free
        self.evictions += evictions
        for e in key:
            self.tick[e] = self.now
        self._prepared = key if prepared else None

    def _ensure(self, ids):
        """Make every expert id this one call needs resident, all at once — `gather_qmm` indexes them in a
        single pass, so none may be evicted to make room for another one the *same* call also needs."""
        self.commit(self.stage(ids))

    def has_all(self, ids):
        return all(int(e) in self.resident for e in np.unique(np.asarray(ids)))

    def prepare(self, ids):
        """Load a routed set now; the following call consumes it without counting the same access twice."""
        self.commit(self.stage(ids), prepared=True)

    def __call__(self, x, idx, sorted_idx):
        idx_np = np.asarray(idx)
        key = tuple(int(e) for e in np.unique(idx_np))
        if self._prepared == key:
            self._prepared = None
        else:
            self._ensure(idx_np)
        table = np.zeros(self.n_exp, np.int32)
        for g, slot in self.resident.items():
            table[g] = slot
        remap = table[idx_np]
        if sorted_idx and remap.ndim == 1 and np.any(remap[1:] < remap[:-1]):
            # A prompt arrives sorted by the experts' own numbers, and `sorted_indices=True` — the grouped
            # kernel, one matrix product per expert — needs the indices ascending *as the pool numbers them*,
            # which they no longer are. Sort by slot, run, and put the rows back. (Passing False instead ran
            # one vector product per (token, expert) pair, which is slow for a prompt and rounds differently.)
            perm = np.argsort(remap, kind="stable")
            y = self._gather(x[mx.array(perm)], mx.array(remap[perm]), True)
            return y[mx.array(np.argsort(perm))]
        return self._gather(x, mx.array(remap), bool(sorted_idx) and remap.ndim == 1)

    def _gather(self, x, local, srt):
        if self.bits:
            return mx.gather_qmm(x, self.w, self.scales, self.biases, rhs_indices=local, transpose=True,
                                 group_size=32, bits=self.bits, sorted_indices=srt)
        return mx.gather_mm(x, self.w.swapaxes(-1, -2), rhs_indices=local, sorted_indices=srt)

    @property
    def nbytes(self):
        """What is actually resident (the cache), not the full stacked tensor on disk."""
        return self.w.nbytes + self.scales.nbytes + self.biases.nbytes

    def stats(self):
        n = self.hits + self.misses
        return {"hits": self.hits, "misses": self.misses,
                "hit_rate": round(self.hits / n, 4) if n else None,
                "evictions": self.evictions, "read_ops": self.decode_ops + self.packed_hits,
                "repack_ops": self.decode_ops, "packed_hits": self.packed_hits,
                "bytes_read": self.source_bytes + self.packed_bytes,
                "gguf_bytes": self.source_bytes, "packed_bytes": self.packed_bytes,
                "read_s": self.decode_s + self.packed_s, "repack_s": self.decode_s,
                "resident": len(self.resident), "capacity": self.cache_size}


def _gather_sort(x, indices):
    """Order the (token, expert) pairs by expert so each expert's weights are read once per pass."""
    *_, m = indices.shape
    flat = indices.flatten()
    order = mx.argsort(flat)
    return x.flatten(0, -3)[order // m], flat[order], mx.argsort(order)


def _expert_safe_groups(layer, idx):
    """Split the token axis into the fewest contiguous groups whose own distinct expert ids fit every
    streamed projection's cache at once (`gather_qmm` needs a group's whole working set resident
    simultaneously — see `StreamedExperts`). Greedy, not optimal, but simple, cheap (one pass over the
    token axis, done once per MoE layer) and correct. If even one token's own k picks cannot fit,
    `StreamedExperts._ensure` still raises its own clear error — grouping only ever helps, it never
    hides a cache that is genuinely too small for `n_expert_used` alone."""
    cache_size = min(layer.gate_e.cache_size, layer.up_e.cache_size, layer.down_e.cache_size)
    idx_np = np.asarray(idx)                                        # [B, L, k]
    length = idx_np.shape[1]
    groups, start, seen = [], 0, set()
    for t in range(length):
        ids_t = set(int(i) for i in idx_np[:, t].reshape(-1))
        merged = seen | ids_t
        if len(merged) > cache_size and t > start:
            groups.append((start, t))
            start, seen = t, set(ids_t)
        else:
            seen = merged
    groups.append((start, length))
    return groups


from .cache import KVCache, SSMCache  # noqa: E402,F401  (one implementation for every engine: engine/cache.py)


def nn_softplus(x):
    return mx.logaddexp(x, 0.0)


def _rms(x, w, eps):
    return mx.fast.rms_norm(x, w, eps)


def _gelu_tanh(x):
    return 0.5 * x * (1.0 + mx.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)))


def _softcap_attention(q, k, v, scale, mask, cap):
    """Attention with Gemma 2's tanh soft-cap on the scores — mx.fast SDPA has no cap hook, so the scores are
    formed by hand (float32), capped, masked and soft-maxed. q [B,H,L,D], k/v [B,Hkv,S,D] (GQA repeated)."""
    H, Hkv = q.shape[1], k.shape[1]
    if H != Hkv:
        rep = H // Hkv
        k, v = mx.repeat(k, rep, axis=1), mx.repeat(v, rep, axis=1)
    scores = (q.astype(mx.float32) @ k.astype(mx.float32).swapaxes(-1, -2)) * scale
    scores = mx.tanh(scores / cap) * cap
    if mask is not None:
        scores = scores + mask.astype(mx.float32)
    w = mx.softmax(scores, axis=-1)
    return (w @ v.astype(mx.float32)).astype(mx.float16)


class Layer:
    def __init__(self, cfg, tensors, i):
        t = lambda name: tensors[f"blk.{i}.{name}"]
        opt = lambda name: tensors.get(f"blk.{i}.{name}")
        self.recurrent = cfg.is_recurrent(i)
        self.attn_norm = t("attn_norm.weight")
        # Qwen3.5 names the norm before its feed-forward "post_attention_norm"; Gemma uses that name for a second
        # norm on the attention output
        self.ffn_norm = t("post_attention_norm.weight") if cfg.full_attn_interval else t("ffn_norm.weight")
        self.q = self.k = self.v = self.o = self.q_norm = self.k_norm = None
        if self.recurrent:
            self.qkv, self.z = t("attn_qkv.weight"), t("attn_gate.weight")
            self.beta, self.alpha = t("ssm_beta.weight"), t("ssm_alpha.weight")
            self.dt, self.ssm_a, self.conv = t("ssm_dt.bias"), t("ssm_a"), t("ssm_conv1d.weight")
            self.ssm_norm, self.ssm_out = t("ssm_norm.weight"), t("ssm_out.weight")
        else:
            self.q, self.k, self.v, self.o = (t(f"attn_{n}.weight") for n in ("q", "k", "v", "output"))
            self.q_norm = opt("attn_q_norm.weight")
            self.k_norm = opt("attn_k_norm.weight")
        # a shared expert next to the routed ones, scaled by a sigmoid gate of its own (Qwen3.5-MoE)
        self.sh_gate, self.sh_up, self.sh_down = (opt(f"ffn_{n}_shexp.weight") for n in ("gate", "up", "down"))
        self.sh_router = opt("ffn_gate_inp_shexp.weight")
        self.router = tensors.get(f"blk.{i}.ffn_gate_inp.weight")
        if self.router is not None:                          # a mixture-of-experts layer
            self.gate_e, self.up_e, self.down_e = (t(f"ffn_{n}_exps.weight") for n in ("gate", "up", "down"))
            self.gate = self.up = self.down = None
            self.streamed = isinstance(self.gate_e, StreamedExperts)
        else:
            self.gate, self.up, self.down = t("ffn_gate.weight"), t("ffn_up.weight"), t("ffn_down.weight")
        self.post_attn_norm = None if cfg.full_attn_interval else opt("post_attention_norm.weight")
        self.post_ffn_norm = opt("post_ffw_norm.weight")

    def params(self):
        ms = (self.q, self.k, self.v, self.o, self.gate, self.up, self.down)
        if self.router is not None:
            ms = (self.q, self.k, self.v, self.o, self.router, self.gate_e, self.up_e, self.down_e)
        if self.recurrent:
            ms = ms + (self.qkv, self.z, self.beta, self.alpha, self.ssm_out, self.conv)
        ms = ms + (self.sh_gate, self.sh_up, self.sh_down)
        return [m for m in ms if m is not None]


class DenseModel:
    """Callable like the mlx-lm models: model(ids [B, L], cache=[KVCache, ...]) → logits [B, L, vocab]."""

    def __init__(self, cfg, tensors):
        self.cfg = cfg
        self.embed = tensors["token_embd.weight"]
        self.norm = tensors["output_norm.weight"]
        self.head = tensors.get("output.weight") or self.embed.as_linear()
        self.layers = [Layer(cfg, tensors, i) for i in range(cfg.n_layer)]
        self._expert_readers = ThreadPoolExecutor(max_workers=3, thread_name_prefix="gguf-expert") \
            if any(L.streamed for L in self.layers if L.router is not None) else None
        self.steer = None            # y -> y', see engine/steering.py: applied where layers write the residual
        mode = os.environ.get("LLM_GGUF_ACT", "mixed")
        self.act = mx.float32 if mode == "f32" else mx.float16      # dtype fed to the matmuls
        self.wide = mode in ("f32", "mixed")                          # residual stream + swiglu in float32
        self.args = type("Args", (), {"vocab_size": cfg.vocab})()
        self.mrope = None
        if cfg.extra.get("mrope_section"):       # an image prompt's 3-D positions (engine/model.py's M-RoPE)
            from .model import MRoPE
            self.mrope = MRoPE(cfg.rope_dims, cfg.rope_base, cfg.extra["mrope_section"])
        self.vision = None                       # the image projector, attached on the first image
        self._freqs = None
        if cfg.rope_freqs is not None:
            d = cfg.rope_dims
            base = cfg.rope_base ** (mx.arange(0, d, 2, dtype=mx.float32) / d)
            self._freqs = base * mx.array(np.asarray(cfg.rope_freqs, np.float32))
        M = self.act
        R = mx.float32 if self.wide else mx.float16
        for L in self.layers:                  # weights in the dtype they are used in, converted once
            L.attn_norm, L.ffn_norm = L.attn_norm.astype(R), L.ffn_norm.astype(R)
            if L.recurrent:
                L.conv = L.conv.astype(mx.float16).T          # [K, channels]: one row per tap
                L.ssm_a, L.dt = L.ssm_a.astype(mx.float32), L.dt.astype(mx.float32)
                L.ssm_norm = L.ssm_norm.astype(R)
            if L.sh_router is not None:
                L.sh_router = L.sh_router.astype(mx.float32).reshape(-1, 1)
            if L.q_norm is not None:
                L.q_norm, L.k_norm = L.q_norm.astype(M), L.k_norm.astype(M)
            if L.post_attn_norm is not None:
                L.post_attn_norm, L.post_ffn_norm = L.post_attn_norm.astype(R), L.post_ffn_norm.astype(R)
        self.norm = self.norm.astype(R)
        self.nbytes = (sum(m.nbytes for L in self.layers for m in L.params())
                       + self.embed.w.nbytes + (self.embed.scales.nbytes + self.embed.biases.nbytes if self.embed.bits else 0)
                       + (self.head.nbytes if "output.weight" in tensors else 0))

    def close(self):
        """Stop GGUF expert readers before the model and its mapped source are released."""
        if self._expert_readers is not None:
            self._expert_readers.shutdown(wait=True, cancel_futures=True)
            self._expert_readers = None

    def moe_stats(self):
        """The same cache/traffic view the safetensors MoE path exposes through `/health`.

        GGUF's source counter is logical mmap traffic and includes quantized decoding time; the `source` field
        keeps it distinct from moe_stream.py's explicit preadv traffic while the shared cache fields remain
        directly comparable.
        """
        layers = [L for L in self.layers if L.router is not None and L.streamed]
        if not layers:
            return None
        pools = [p for L in layers for p in (L.gate_e, L.up_e, L.down_e)]
        stats = [p.stats() for p in pools]
        hits, misses = sum(s["hits"] for s in stats), sum(s["misses"] for s in stats)
        source_bytes, source_s = sum(s["bytes_read"] for s in stats), sum(s["read_s"] for s in stats)
        packed_hits = sum(s["packed_hits"] for s in stats)
        return {"format": "gguf", "source": "packed-cache" if packed_hits else "mmap+decode", "layers": len(layers),
                "capacity": min(p.cache_size for p in pools), "experts": layers[0].gate_e.n_exp,
                "hits": hits, "misses": misses,
                "hit_rate": round(hits / (hits + misses), 4) if hits + misses else None,
                "evictions": sum(s["evictions"] for s in stats),
                "read_ops": sum(s["read_ops"] for s in stats),
                "repack_ops": sum(s["repack_ops"] for s in stats), "packed_hits": packed_hits,
                "gguf_gb_read": round(sum(s["gguf_bytes"] for s in stats) / 2**30, 3),
                "packed_gb_read": round(sum(s["packed_bytes"] for s in stats) / 2**30, 3),
                "gb_read": round(source_bytes / 2**30, 3), "read_s": round(source_s, 2),
                "read_mbps": round(source_bytes / 2**20 / source_s, 1) if source_s else None,
                "pool_gb": round(sum(p.nbytes for p in pools) / 2**30, 2)}

    def make_cache(self):
        return [SSMCache() if L.recurrent else KVCache() for L in self.layers]

    @property
    def hybrid(self):
        return bool(self.cfg.full_attn_interval)

    def _gdn(self, layer, h, cache):
        """One GatedDeltaNet layer on its normed input `h` [B, S, D]: projections, the causal depthwise
        convolution over the cached tail, the delta-rule recurrence, the gated norm and the output projection.
        Follows engine/model.py's Qwen3.5 (verified against the reference) with ggml's tensor layout."""
        from .gated_delta import gated_delta_kernel
        c = self.cfg
        F16, F32 = mx.float16, mx.float32
        B, S, _ = h.shape
        nk, nv, dk = c.ssm_nk, c.ssm_nv, c.ssm_dk
        key_dim, K = nk * dk, c.ssm_conv
        qkv = layer.qkv(h).astype(F16)                                      # [B, S, 2*key_dim + nv*dk]
        z = layer.z(h)
        beta = mx.sigmoid(layer.beta(h).astype(F32))
        # ssm_a is -exp(A_log): the decay is exp(A * softplus(alpha + dt_bias))
        g = mx.exp(layer.ssm_a * nn_softplus(layer.alpha(h).astype(F32) + layer.dt))
        conv_dim = qkv.shape[-1]
        state = cache.conv if cache.conv is not None else mx.zeros((B, K - 1, conv_dim), dtype=F16)
        conv_in = mx.concatenate([state, qkv], axis=1)
        w = layer.conv                                                      # [K, channels]
        co = conv_in[:, 0:S] * w[0]
        for j in range(1, K):
            co = co + conv_in[:, j:j + S] * w[j]
        co = (co * mx.sigmoid(co)).astype(F16)                              # silu
        inv = dk ** -0.5
        q = (inv * inv) * mx.fast.rms_norm(co[..., :key_dim].reshape(B, S, nk, dk), None, 1e-6)
        k = inv * mx.fast.rms_norm(co[..., key_dim:2 * key_dim].reshape(B, S, nk, dk), None, 1e-6)
        v = co[..., 2 * key_dim:].reshape(B, S, nv, dk)
        if nv != nk:                       # value heads are tiled [k0, k1, …, k0, k1, …]: tile the keys to match
            q, k = mx.tile(q, (1, 1, nv // nk, 1)), mx.tile(k, (1, 1, nv // nk, 1))
        out, cache.state = gated_delta_kernel(q, k, v, g.astype(F16), beta.astype(F16), cache.state)
        cache.conv = mx.contiguous(conv_in[:, -(K - 1):])
        cache.offset += S
        zz = z.reshape(B, S, nv, dk).astype(F32)
        o = _rms(out.astype(F32), layer.ssm_norm, c.eps) * (zz * mx.sigmoid(zz))
        if c.gdn_v_grouped and nv != nk:   # tiled [r][k] → grouped by key head [k][r], the order ssm_out reads
            o = o.reshape(B, S, nv // nk, nk, dk).transpose(0, 1, 3, 2, 4)
        return layer.ssm_out(o.reshape(B, S, -1).astype(self.act))

    def _ffn(self, layer, h, R):
        """The feed-forward of one layer on its normed input: routed experts (plus a gated shared expert where
        the model has one), or the plain gated MLP."""
        c = self.cfg
        if layer.router is not None:
            d = self._moe(layer, h)
            if layer.sh_up is not None:
                g = layer.sh_gate(h).astype(R)
                s = layer.sh_down(((mx.sigmoid(g) * g) * layer.sh_up(h).astype(R)).astype(self.act)).astype(R)
                d = d + s * mx.sigmoid(h.astype(mx.float32) @ layer.sh_router).astype(R)
            return d
        g = layer.gate(h).astype(R)
        act = _gelu_tanh(g) if c.gelu else mx.sigmoid(g) * g
        return layer.down((act * layer.up(h).astype(R)).astype(self.act)).astype(R)

    def _moe(self, layer, h):
        """Route each token to its top-k experts (softmax over all, optionally renormalised over the chosen),
        run only those, and add their outputs weighted. When the experts are streamed (`StreamedExperts`),
        a sequence that would route to more distinct experts than the cache holds is split into the fewest
        token groups that each fit, instead of failing — the same "shrink under a budget" idea
        `engine/generate.py`'s chunked prefill already uses for memory, scoped here to expert residency."""
        c = self.cfg
        M = self.act
        R = mx.float32 if self.wide else mx.float16
        logits = layer.router(h).astype(mx.float32)                        # [B, L, E]
        probs = mx.softmax(logits, axis=-1)
        k = c.n_expert_used
        idx = mx.argpartition(-logits, kth=k - 1, axis=-1)[..., :k]        # [B, L, k]
        w = mx.take_along_axis(probs, idx, axis=-1)
        if c.expert_norm:
            w = w / w.sum(-1, keepdims=True)
        if c.expert_scale != 1.0:
            w = w * c.expert_scale
        if layer.streamed and idx.shape[1] > 1:
            groups = _expert_safe_groups(layer, idx)
        else:
            groups = [(0, idx.shape[1])]
        if len(groups) == 1:
            lo, hi = groups[0]
            return self._moe_pass(layer, h[:, lo:hi], idx[:, lo:hi], w[:, lo:hi], c, M, R)
        return mx.concatenate([self._moe_pass(layer, h[:, lo:hi], idx[:, lo:hi], w[:, lo:hi], c, M, R)
                               for lo, hi in groups], axis=1)

    def _moe_pass(self, layer, h, idx, w, c, M, R):
        """One group's worth of tokens through the routed feed-forward — exactly today's single-pass code,
        just callable per token group so `_moe` can split a sequence when the experts are streamed."""
        x = mx.expand_dims(h, (-2, -3))                                    # [B, L, 1, 1, D]
        order = None
        do_sort = idx.size >= 64                                           # prefill: group tokens by expert
        sel = idx
        if do_sort:
            x, sel, order = _gather_sort(x, idx)
        if layer.streamed:
            # The three GGUF projections are independent source slabs. Decode/read them together on a miss,
            # like the safetensors streamer's read ring, then let each gather consume its prepared pool.
            routed = np.asarray(sel)
            pools = (layer.gate_e, layer.up_e, layer.down_e)
            if any(not p.has_all(routed) for p in pools):
                staged = list(self._expert_readers.map(lambda p: p.stage(routed), pools))
                for p, update in zip(pools, staged):
                    p.commit(update, prepared=True)
            else:
                for p in pools:
                    p.prepare(routed)
        g = layer.gate_e(x, sel, do_sort).astype(R)
        act = _gelu_tanh(g) if c.gelu else mx.sigmoid(g) * g
        inner = (act * layer.up_e(x, sel, do_sort).astype(R)).astype(M)
        d = layer.down_e(inner, sel, do_sort)
        if do_sort:
            d = mx.unflatten(d[order], 0, idx.shape)
        d = d.squeeze(-2).astype(R)                                        # [B, L, k, D]
        return (d * w[..., None].astype(R)).sum(-2)

    def _rope(self, x, offset, local=False):
        c = self.cfg
        if self._freqs is not None:
            return mx.fast.rope(x, c.rope_dims, traditional=c.rope_traditional, base=None, scale=1.0,
                                offset=offset, freqs=self._freqs)
        if local and c.rope_base_local:                 # Gemma 3 local layers: their own base, no scaling
            return mx.fast.rope(x, c.rope_dims, traditional=c.rope_traditional, base=c.rope_base_local,
                                scale=1.0, offset=offset)
        return mx.fast.rope(x, c.rope_dims, traditional=c.rope_traditional, base=c.rope_base,
                            scale=c.rope_scale, offset=offset)

    def embed_tokens(self, ids):
        """Token embeddings in the residual stream's type — what an image's features are put in place of."""
        R = mx.float32 if self.wide else mx.float16
        x = self.embed(ids).astype(R)
        return x * self.cfg.embed_scale if self.cfg.embed_scale != 1.0 else x

    def __call__(self, ids=None, cache=None, embeds=None, cos=None, sin=None, delta=0):
        """The residual stream and the SwiGLU product are float32 (as in ggml: Llama 3.2 and others carry a few
        huge activations that float16 rounds away, and greedy text drifted from llama.cpp's after a handful of
        tokens); the matmuls take float16 inputs — the weights are quantized anyway — and keys/values are
        stored in float16."""
        c = self.cfg
        M, F16, F32 = self.act, mx.float16, mx.float32
        R = F32 if self.wide else F16                    # dtype of the residual stream
        # `embeds` [B, L, D] replace the token lookup (an image prompt); `cos`/`sin` are its 3-D rotary positions;
        # `delta` shifts the text positions after an image (its patches advance the position less than its tokens)
        if embeds is not None:
            x = embeds.astype(R)
            B, L = x.shape[:2]
        else:
            B, L = ids.shape
            x = self.embed_tokens(ids)
        if cache is None:
            cache = self.make_cache()
        st = self.steer or (lambda y: y)
        x = st(x)
        off = cache[0].offset
        mask = None
        if L > 1:
            q_pos = mx.arange(off, off + L)[:, None]
            k_pos = mx.arange(off + L)[None, :]
            mask = mx.where(k_pos <= q_pos, 0.0, -mx.inf).astype(F16)
        # sliding-window layers look at the last `window` keys only: slice them (and mask what is still too old)
        w_start, w_mask = 0, None
        if c.window and off + L > c.window:
            w_start = max(0, off + 1 - c.window)
            q_pos = mx.arange(off, off + L)[:, None]
            k_pos = mx.arange(w_start, off + L)[None, :]
            w_mask = mx.where((k_pos <= q_pos) & (q_pos - k_pos < c.window), 0.0, -mx.inf).astype(F16)
        elif c.window:
            w_mask = mask                       # nothing is older than the window yet
        scale = c.attn_scale or c.head_dim ** -0.5
        for li, (layer, kv) in enumerate(zip(self.layers, cache)):
            local = c.is_local(li)
            h = _rms(x, layer.attn_norm, c.eps).astype(M)
            if layer.recurrent:
                x = x + st(self._gdn(layer, h, kv).astype(R))
                x = x + st(self._ffn(layer, _rms(x, layer.ffn_norm, c.eps).astype(M), R))
                continue
            gate = None
            if c.attn_gated:                             # [query | gate] per head
                qg = layer.q(h).reshape(B, L, c.n_head, 2 * c.head_dim)
                q, gate = qg[..., :c.head_dim], qg[..., c.head_dim:].reshape(B, L, -1)
            else:
                q = layer.q(h).reshape(B, L, c.n_head, c.head_dim)
            k = layer.k(h).reshape(B, L, c.n_head_kv, c.head_dim)
            v = layer.v(h).reshape(B, L, c.n_head_kv, c.head_dim)
            if layer.q_norm is not None:
                q = _rms(q, layer.q_norm, c.eps)
                k = _rms(k, layer.k_norm, c.eps)
            q, k, v = (a.transpose(0, 2, 1, 3) for a in (q, k, v))
            if cos is not None and self.mrope is not None:
                q, k = self.mrope.apply(q, cos, sin), self.mrope.apply(k, cos, sin)
            else:
                q, k = self._rope(q, off + delta, local), self._rope(k, off + delta, local)
            k, v = kv.update_and_fetch(k.astype(F16), v.astype(F16))
            m = mask
            if local:
                m = w_mask
                if w_start:
                    k, v = k[..., w_start:, :], v[..., w_start:, :]
            if c.attn_softcap:                           # Gemma 2: scores are tanh-capped, so attention is by hand
                a = _softcap_attention(q, k, v, scale, m, c.attn_softcap).astype(M)
            else:
                a = mx.fast.scaled_dot_product_attention(q.astype(F16), k, v, scale=scale, mask=m).astype(M)
            a = a.transpose(0, 2, 1, 3).reshape(B, L, -1)
            if gate is not None:
                a = (a.astype(F32) * mx.sigmoid(gate.astype(F32))).astype(M)
            o = layer.o(a).astype(R)
            if layer.post_attn_norm is not None:
                o = _rms(o, layer.post_attn_norm, c.eps)
            x = x + st(o) * c.residual_scale
            h = _rms(x, layer.ffn_norm, c.eps).astype(M)
            d = self._ffn(layer, h, R)
            if layer.post_ffn_norm is not None:
                d = _rms(d, layer.post_ffn_norm, c.eps)
            x = x + st(d) * c.residual_scale
        x = _rms(x, self.norm, c.eps).astype(M)
        logits = self.head(x)
        if c.final_softcap:                              # Gemma 2: tanh soft-cap on the output logits
            logits = mx.tanh(logits / c.final_softcap) * c.final_softcap
        return logits * (1.0 / c.logit_scale) if c.logit_scale != 1.0 else logits


# ───────────────────────────── building from a GGUF ─────────────────────────────
class Unsupported(ValueError):
    pass


def _linear(gf, info, T, bias_info=None):
    name = info.type_name
    bias = None
    if bias_info is not None:
        bias = mx.array(T.load_dense(gf, bias_info)).astype(mx.float16)
    if name in ("F32", "F16", "BF16"):
        return QLinear(mx.array(T.load_dense(gf, info)).astype(mx.float16), bias=bias)
    w, s, b, bits = T.load_matrix(gf, info)
    return QLinear(mx.array(w), mx.array(s), mx.array(b), bits, bias, gs=T.group_size(info))


def _experts(gf, info, T, stream=False, cache_size=64, packed_cache_dir="", packed_cache_bytes=0):
    """A stacked 3-D expert tensor [E, rows, cols] in MLX's quantized form, same layout per expert. `stream`
    (off by default — see `build(..., stream_experts=...)`) decodes experts on demand into a bounded LRU
    cache (`StreamedExperts`) instead of decoding the whole stack up front; a non-quantized (F32/F16/BF16)
    expert tensor is always loaded eagerly, same as today — streaming only applies to the quantized case,
    which is the realistic one for a MoE checkpoint too large to hold resident."""
    n_exp, rows, cols = info.shape
    name = info.type_name
    if name in ("F32", "F16", "BF16"):
        return Experts(mx.array(T.load_dense(gf, info)).astype(mx.float16))
    if T.group_size(info) != T.GROUP:      # the expert kernels and the packed cache are laid out for groups of 32
        raise Unsupported(f"{info.name}: {name} experts")
    # the loader works row by row: the stack is E*rows rows of `cols` (the bytes are contiguous in that order)
    flat = T.TensorInfo(info.name, (cols, n_exp * rows), info.type, info.offset)
    if stream:
        return StreamedExperts(gf, flat, n_exp, rows, cache_size, packed_cache_dir, packed_cache_bytes)
    w, s, b, bits = T.load_matrix(gf, flat)
    shape = lambda a: (n_exp, rows) + a.shape[1:]
    return Experts(mx.array(w).reshape(shape(w)), mx.array(s).reshape(shape(s)), mx.array(b).reshape(shape(b)), bits)


def plan_expert_cache(path, budget_bytes, reserve_bytes=2.5 * 2**30):
    """Choose decoded GGUF experts per projection from the RAM that is safe to use now.

    This is admission planning, not another memory manager: `engine.memory` supplies `budget_bytes`, and the
    resulting bounded cache remains owned by `StreamedExperts`. The estimate follows `gguf_tensors.load_matrix`
    exactly for quantized matrices (packed codes plus fp16 scale and bias per group of 32); non-expert tensors
    are conservatively counted as fp16, while `reserve_bytes` leaves room for KV, activations and allocator
    overhead. An explicit `LLM_GGUF_EXPERT_CACHE` bypasses this caller-side policy.
    """
    from . import gguf_tensors as T
    gf = T.GGUFFile(path)
    arch = gf.meta.get("general.architecture", "")
    n_exp = int(gf.meta.get(f"{arch}.expert_count", 0) or 0)
    top_k = int(gf.meta.get(f"{arch}.expert_used_count", 0) or 0)
    if not n_exp or not top_k:
        return None

    def resident_bytes(info):
        bits = T.MLX_BITS.get(info.type_name)
        if bits is None:
            return info.n_elements * 2              # dense tensors are normally converted to fp16
        # packed codes: bits/8 byte per value; fp16 scale + fp16 bias per 32 values: another 1/8 byte
        return info.n_elements * (bits + 1) // 8

    expert_infos = [v for k, v in gf.tensors.items() if "_exps.weight" in k]
    # Float experts are deliberately eager in `_experts`; no bounded cache exists to plan for them.
    if not expert_infos or any(v.type_name not in T.MLX_BITS or v.shape[0] != n_exp for v in expert_infos):
        return None
    per_expert = sum(resident_bytes(v) for v in expert_infos) // n_exp
    fixed = sum(resident_bytes(v) for k, v in gf.tensors.items() if "_exps.weight" not in k)
    room = max(0, int(budget_bytes - reserve_bytes - fixed))
    capacity = max(top_k, min(n_exp, room // max(1, per_expert)))
    resident = fixed + capacity * per_expert
    minimum = fixed + top_k * per_expert + reserve_bytes
    return {"capacity": capacity, "experts": n_exp, "top_k": top_k,
            "thrashing": capacity < n_exp, "fixed_gb": fixed / 2**30,
            "per_expert_mb": per_expert / 2**20, "resident_gb": resident / 2**30,
            "budget_gb": budget_bytes / 2**30, "fixed_bytes": fixed,
            "per_expert_bytes": per_expert, "minimum_gb": minimum / 2**30,
            "fits": minimum <= budget_bytes}


def build(path, log=print, stream_experts=None, expert_cache=None, packed_cache_dir="", packed_cache_bytes=0):
    """(DenseModel, Config, GGUFFile). Raises Unsupported for anything this decoder does not do — the caller
    then runs the file on llama.cpp instead.

    `stream_experts`: for a MoE checkpoint, decode each expert from the memory-mapped GGUF file only when a
    token is actually routed to it, keeping at most `expert_cache` decoded experts per projection resident at
    once (LRU-evicted, automatically split into smaller token groups when one pass needs more than that —
    see `StreamedExperts`/`_expert_safe_groups`) instead of decoding the whole stack of experts for every
    layer at load time. Output is unaffected — it is the same weights, fetched lazily. Direct callers default
    to streaming (`LLM_GGUF_STREAM_EXPERTS=0` turns it off); `GGUFEngine` first asks the shared memory manager
    and passes `False` when the complete decoded expert set fits safe RAM, because GGUF misses also pay a
    quant-block decode/repack cost. Pass `False` explicitly to force eager loading for a comparison or a
    benchmark.

    `expert_cache`: how many decoded experts per projection `StreamedExperts` keeps resident at once.
    `StreamedExperts._ensure`'s own error message tells the caller to "raise expert_cache" when a single pass
    routes to more distinct experts than the cache holds — `LLM_GGUF_EXPERT_CACHE` is how a caller that only
    sees that message (not this function's kwarg) actually follows that advice."""
    if stream_experts is None:
        stream_experts = os.environ.get("LLM_GGUF_STREAM_EXPERTS", "1") != "0"
    if expert_cache is None:
        expert_cache = int(os.environ.get("LLM_GGUF_EXPERT_CACHE", "64"))
    from . import gguf as G, gguf_tensors as T, rotation
    gf = T.GGUFFile(path)
    m = gf.meta
    arch = m.get("general.architecture")
    if arch not in SUPPORTED:
        raise Unsupported(f"architecture {arch!r}")
    try:
        contract = rotation.from_gguf(m, lambda key: G.read_array(path, key))
    except rotation.Unsupported as e:
        raise Unsupported(str(e))
    g = lambda key, default=None: m.get(f"{arch}.{key}", default)
    n_head = int(g("attention.head_count"))
    n_embd = int(g("embedding_length"))
    head_dim = int(g("attention.key_length", n_embd // n_head))
    if g("attention.value_length", head_dim) != head_dim:
        raise Unsupported("different key and value head sizes")
    scaling = m.get(f"{arch}.rope.scaling.type")
    longrope = scaling == "longrope" and arch == "minicpm"
    if scaling not in (None, "none", "llama3") and not (scaling == "linear" and arch == "gemma3") and not longrope:
        raise Unsupported(f"rope scaling {scaling!r}")
    if longrope:
        # llama.cpp picks the short factors while the context fits the original length (the engine's never exceeds)
        orig = g("rope.scaling.original_context_length")
        if "rope_factors_short.weight" not in gf.tensors:
            raise Unsupported("longrope scaling without its short factors")
        if orig and int(orig) < int(g("context_length", orig)):
            raise Unsupported("longrope scaling with a longer context than the original")
    if scaling == "llama3" and "rope_freqs.weight" not in gf.tensors:
        raise Unsupported("llama3 rope scaling without its frequency table")
    if g("attention.sliding_window") and arch not in ("gemma2", "gemma3"):
        raise Unsupported("sliding-window attention")
    names = gf.tensors
    hybrid = arch in HYBRID
    n_expert = int(g("expert_count", 0) or 0)
    if n_expert and "blk.0.ffn_gate_up_exps.weight" in names:
        raise Unsupported("mixture of experts with fused gate/up expert tensors")
    if n_expert and ("blk.0.ffn_gate_exps.weight" not in names or "blk.0.ffn_gate_inp.weight" not in names):
        raise Unsupported("mixture of experts stored per expert (or with a shared expert)")
    if not hybrid and any(n.startswith("blk.0.ffn_gate_shexp") or n.startswith("blk.0.ffn_up_shexp") for n in names):
        raise Unsupported("mixture of experts with a shared expert")
    vocab = names["token_embd.weight"].shape[0]
    cfg = Config(
        arch=arch, n_layer=int(g("block_count")), n_embd=n_embd, n_head=n_head,
        n_head_kv=int(g("attention.head_count_kv", n_head)), n_ff=int(g("feed_forward_length", 0) or 0),
        head_dim=head_dim,
        vocab=vocab, eps=float(g("attention.layer_norm_rms_epsilon", 1e-5)), rope_base=float(g("rope.freq_base", 10000.0)),
        rope_dims=int(g("rope.dimension_count", head_dim)), rope_traditional=ROPE_TRADITIONAL[arch],
        qkv_bias="blk.0.attn_q.bias" in names, qk_norm=any(n.endswith("attn_q_norm.weight") for n in names),
        tied_output="output.weight" not in names, context=int(g("context_length", 4096)),
    )
    if hybrid:
        cfg.full_attn_interval = int(g("full_attention_interval", 4) or 4)
        cfg.attn_gated = True
        cfg.ssm_conv, cfg.ssm_dk = int(g("ssm.conv_kernel")), int(g("ssm.state_size"))
        cfg.ssm_nk, cfg.ssm_nv = int(g("ssm.group_count")), int(g("ssm.time_step_rank"))
        if cfg.ssm_nv % cfg.ssm_nk or int(g("ssm.inner_size")) != cfg.ssm_nv * cfg.ssm_dk:
            raise Unsupported("linear-attention head sizes this decoder does not lay out")
        if cfg.n_layer % cfg.full_attn_interval == 0 and cfg.is_recurrent(cfg.n_layer - 1):
            raise Unsupported("a hybrid layout without a final attention layer")
        sections = g("rope.dimension_sections")
        if isinstance(sections, list) and len(sections) >= 3:
            cfg.extra["mrope_section"] = [int(x) for x in sections[:3]]     # 3-D positions of image prompts
    cfg.n_expert, cfg.n_expert_used = n_expert, int(g("expert_used_count", 0) or 0)
    if n_expert:
        if not 0 < cfg.n_expert_used <= n_expert:
            raise Unsupported("the number of experts used per token is missing")
        # Mixtral and Granite-MoE renormalise the chosen weights; Qwen3-MoE says so in its config; Qwen3.5-MoE too
        cfg.expert_norm = bool(g("expert_weights_norm", True)) if arch not in ("qwen3moe", "qwen35moe") else True
        scale = float(g("expert_weights_scale", 0.0) or 0.0)
        cfg.expert_scale = scale if scale not in (0.0, 1.0) else 1.0
    if g("attention.scale"):                                  # Granite family multipliers
        cfg.attn_scale = float(g("attention.scale"))
    if g("embedding_scale"):
        cfg.embed_scale = float(g("embedding_scale"))
    if g("residual_scale"):
        cfg.residual_scale = float(g("residual_scale"))
    if g("logit_scale"):
        cfg.logit_scale = float(g("logit_scale"))
    if arch == "gemma3":
        n_layer = cfg.n_layer
        cfg.gelu = cfg.sandwich = True
        cfg.embed_scale = n_embd ** 0.5
        cfg.window, cfg.window_pattern = int(g("attention.sliding_window", 0)), 6
        cfg.rope_base_local = float(g("rope.freq_base_swa", 10000.0))
        if scaling == "linear":
            cfg.rope_scale = 1.0 / float(g("rope.scaling.factor", 1.0))
        # llama.cpp: the 27B scales queries by 1/sqrt(n_embd / n_head), the others by 1/sqrt(head_dim)
        cfg.attn_scale = (n_embd / n_head) ** -0.5 if n_layer == 62 else 0.0
    if arch == "gemma2":
        cfg.gelu = cfg.sandwich = True
        cfg.embed_scale = n_embd ** 0.5
        # local sliding-window attention on every other layer (even layers local), one rope base for all
        cfg.window, cfg.window_pattern = int(g("attention.sliding_window", 0)), 2
        cfg.attn_softcap = float(g("attn_logit_softcapping", 0.0) or 0.0)
        cfg.final_softcap = float(g("final_logit_softcapping", 0.0) or 0.0)
        qps = g("attention.query_pre_attn_scalar")       # 2B/9B: == head_dim (same as the default); 27B differs
        if qps:
            cfg.attn_scale = float(qps) ** -0.5
    if "rope_freqs.weight" in names:
        cfg.rope_freqs = T.load_dense(gf, names["rope_freqs.weight"])
    elif longrope:
        cfg.rope_freqs = T.load_dense(gf, names["rope_factors_short.weight"])
    tensors = {}
    emb = names["token_embd.weight"]
    if emb.type_name in ("F32", "F16", "BF16"):
        tensors["token_embd.weight"] = Embedding(mx.array(T.load_dense(gf, emb)).astype(mx.float16))
    else:
        w, s, b, bits = T.load_matrix(gf, emb)
        tensors["token_embd.weight"] = Embedding(mx.array(w), mx.array(s), mx.array(b), bits, gs=T.group_size(emb))
    tensors["output_norm.weight"] = mx.array(T.load_dense(gf, names["output_norm.weight"]))
    if "output.weight" in names:
        tensors["output.weight"] = _linear(gf, names["output.weight"], T)
    dense_t = lambda name: mx.array(T.load_dense(gf, names[name]))
    for i in range(cfg.n_layer):
        for n in ("attn_norm", "ffn_norm"):
            if f"blk.{i}.{n}.weight" in names or not hybrid:
                tensors[f"blk.{i}.{n}.weight"] = dense_t(f"blk.{i}.{n}.weight")
        for n in ("attn_q_norm", "attn_k_norm", "post_attention_norm", "post_ffw_norm"):
            if f"blk.{i}.{n}.weight" in names:
                tensors[f"blk.{i}.{n}.weight"] = dense_t(f"blk.{i}.{n}.weight")
        moe = f"blk.{i}.ffn_gate_inp.weight" in names
        recurrent = cfg.is_recurrent(i)
        if recurrent:
            for n in ("attn_qkv", "attn_gate", "ssm_beta", "ssm_alpha", "ssm_out"):
                tensors[f"blk.{i}.{n}.weight"] = _linear(gf, names[f"blk.{i}.{n}.weight"], T)
            for n in ("ssm_dt.bias", "ssm_a", "ssm_conv1d.weight", "ssm_norm.weight"):
                tensors[f"blk.{i}.{n}"] = dense_t(f"blk.{i}.{n}")
        if f"blk.{i}.ffn_up_shexp.weight" in names:
            for n in ("gate", "up", "down"):
                tensors[f"blk.{i}.ffn_{n}_shexp.weight"] = _linear(gf, names[f"blk.{i}.ffn_{n}_shexp.weight"], T)
            tensors[f"blk.{i}.ffn_gate_inp_shexp.weight"] = dense_t(f"blk.{i}.ffn_gate_inp_shexp.weight")
        for n, bn in (("attn_q", "attn_q.bias"), ("attn_k", "attn_k.bias"), ("attn_v", "attn_v.bias"),
                      ("attn_output", None), ("ffn_gate", None), ("ffn_up", None), ("ffn_down", None)):
            if moe and n.startswith("ffn_"):
                continue
            if recurrent and n.startswith("attn_"):
                continue
            bias = names.get(f"blk.{i}.{bn}") if bn else None
            tensors[f"blk.{i}.{n}.weight"] = _linear(gf, names[f"blk.{i}.{n}.weight"], T, bias)
        if moe:
            tensors[f"blk.{i}.ffn_gate_inp.weight"] = QLinear(
                mx.array(T.load_dense(gf, names[f"blk.{i}.ffn_gate_inp.weight"])).astype(mx.float32))
            for n in ("gate", "up", "down"):
                tensors[f"blk.{i}.ffn_{n}_exps.weight"] = _experts(
                    gf, names[f"blk.{i}.ffn_{n}_exps.weight"], T, stream=stream_experts, cache_size=expert_cache,
                    packed_cache_dir=packed_cache_dir, packed_cache_bytes=packed_cache_bytes)
        if (i + 1) % 8 == 0:
            mx.eval([a for v in tensors.values() for a in (v if isinstance(v, list) else [v])
                     if isinstance(a, mx.array)])
    if contract is not None:
        _rotate(contract, cfg, tensors, names)
    model = DenseModel(cfg, tensors)
    return model, cfg, gf


def _rotate(contract, cfg, tensors, names):
    """Attach the declared Hadamard rotations: each listed weight rotates its input, the embedding its rows back.
    Only what the file lists is rotated — whatever set of matrices that is — and a listed weight this decoder
    cannot rotate is an error: running it unrotated would give fluent nonsense."""
    rot = Rotation(contract.block, contract.signs)
    for name in sorted(contract.folded | contract.inverse):
        info = names.get(name)
        if info is None:
            raise Unsupported(f"the Hadamard contract lists {name}, which the file does not have")
        t = tensors.get(name)
        if t is None:
            m = re.match(r"blk\.(\d+)\.", name)
            if m and int(m.group(1)) >= cfg.n_layer:      # a draft (MTP) layer the decoder does not load
                continue
            raise Unsupported(f"{name} is rotated, but this decoder does not use it as a matrix")
        width = info.shape[-1]
        if width not in rot.signs or width % contract.block:
            raise Unsupported(f"{name}: no {width}-wide Hadamard signs")
        if name in contract.inverse:
            if not isinstance(t, Embedding):
                raise Unsupported(f"inverse rotation on {name}")
            t.rot = rot
        elif isinstance(t, QLinear):
            t.rot = rot
        else:
            raise Unsupported(f"{name} is rotated, but is not a plain matrix here")
    cfg.rotated = True                                    # (a tied head, embed.as_linear(), carries the rotation)
    cfg.gdn_v_grouped = bool(contract.gdn_v_grouped)
