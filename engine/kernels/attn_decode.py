"""One decode step of attention over the KV cache: a single new query against every cached key.

    attn_decode(q, keys, values, n, scale) -> [1, H, 1, D] float16, or None when it declines

    q       [1, H, 1, D]      float16, positions already applied
    keys    [1, Hkv, cap, D]  float16, the cache's whole (over-allocated) buffer
    values  [1, Hkv, cap, D]  float16
    n       how many rows of the buffers are filled (the new token's row included)

Why this operation and not another. `tests/layer_profile.py` puts the whole slowdown of a long
conversation here: on a 32K context the attention layers are ~78 % of the decode step and read their
cache at ~22 GB/s, on a machine whose memory bus does ~130. The weights' matmuls sit at the bus ceiling
already; this does not — so it is where a kernel of our own can still win something.

What the Metal kernel does differently from `mx.fast.scaled_dot_product_attention`:

  * one launch instead of a chain (matmul, softmax, matmul plus the casts between them), and the
    scores never leave the GPU's threadgroup memory — nothing the size of the context is written back;
  * it takes the cache buffer whole, with the fill count as a number, so no slice of an
    over-allocated buffer is formed per step;
  * grouped-query attention reads each key row once and scores it against every query head that
    shares it (H / Hkv of them), where a broadcast computes — and reads — one copy per query head;
  * the softmax is done per chunk of rows and merged afterwards (the FlashAttention recurrence:
    a chunk reports its own maximum, its normaliser and its weighted sum; the merge rescales them to
    a common maximum), so chunks are independent threadgroups and each streams its slab of K, then of V.

Arithmetic is float32 inside (the reference works in the float16 it is given), so results agree to
float16 rounding, not bit for bit — see tests/test_kernels.py for the tolerance.
"""
import mlx.core as mx

from . import register

C = 256        # rows of the cache per threadgroup
TG = 256       # threads per threadgroup

_SOURCE = """
    const int tid = int(thread_position_in_threadgroup.x);
    const int chunk = int(threadgroup_position_in_grid.x);
    const int hk = int(thread_position_in_grid.y);
    const int nn = int(n);
    const int NC = (nn + C - 1) / C;
    const int r0 = chunk * C;
    const int rows = min(int(C), nn - r0);
    const float sc_ = float(scale);

    threadgroup float p[C * G];          // scores, then softmax numerators, of this chunk
    threadgroup float top[G];            // the chunk's maximum score per query head

    const size_t base = (size_t(hk) * CAP + size_t(r0)) * D;
    const size_t qb = size_t(hk) * G * D;

    // 1. scores: a thread owns a row of K and scores it against the G query heads sharing this kv head
    for (int r = tid; r < C; r += TG) {
        float a[G];
        for (int j = 0; j < G; ++j) a[j] = 0.0f;
        if (r < rows) {
            const size_t kr = base + size_t(r) * D;
            for (int d = 0; d < D; ++d) {
                const float x = float(k[kr + d]);
                for (int j = 0; j < G; ++j) a[j] += x * float(q[qb + j * D + d]);
            }
            for (int j = 0; j < G; ++j) p[r * G + j] = a[j] * sc_;
        } else {
            for (int j = 0; j < G; ++j) p[r * G + j] = -1.0e30f;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 2. the chunk's maximum, one thread per query head
    if (tid < G) {
        float m = -1.0e30f;
        for (int r = 0; r < rows; ++r) m = max(m, p[r * G + tid]);
        top[tid] = m;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 3. numerators exp(score - max); rows past the fill count weigh nothing
    for (int r = tid; r < C; r += TG) {
        for (int j = 0; j < G; ++j)
            p[r * G + j] = (r < rows) ? exp(p[r * G + j] - top[j]) : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const size_t ob = (size_t(hk) * NC + size_t(chunk)) * G;
    if (tid < G) {
        float s = 0.0f;
        for (int r = 0; r < rows; ++r) s += p[r * G + tid];
        den[ob + tid] = s;
        mxo[ob + tid] = top[tid];
    }

    // 4. weighted sum of V: a thread owns a column, so a simdgroup's lanes read one cache line per row
    for (int d = tid; d < D; d += TG) {
        float a[G];
        for (int j = 0; j < G; ++j) a[j] = 0.0f;
        for (int r = 0; r < rows; ++r) {
            const float x = float(v[base + size_t(r) * D + d]);
            for (int j = 0; j < G; ++j) a[j] += p[r * G + j] * x;
        }
        for (int j = 0; j < G; ++j) acc[(ob + j) * D + d] = a[j];
    }
"""

_kernel = None
_broken = False


def _build():
    global _kernel, _broken
    if _kernel is None and not _broken:
        try:
            _kernel = mx.fast.metal_kernel(
                name="engine_attn_decode", input_names=["q", "k", "v", "n", "scale"],
                output_names=["acc", "mxo", "den"], source=_SOURCE)
        except Exception:
            _broken = True
    return _kernel


def available():
    return mx.metal.is_available() and _build() is not None


def metal(q, keys, values, n, scale):
    B, H, L, D = q.shape
    if B != 1 or L != 1 or keys.dtype != mx.float16 or q.dtype != mx.float16:
        return None                      # batched rows, verify passes, other dtypes: the reference
    Hkv, cap = keys.shape[1], keys.shape[2]
    if H % Hkv or values.shape[3] != D:
        return None
    G = H // Hkv
    nc = (n + C - 1) // C
    acc, top, den = _build()(
        inputs=[q, keys, values, mx.array(n, dtype=mx.int32), mx.array(scale, dtype=mx.float32)],
        template=[("C", C), ("TG", TG), ("G", G), ("D", D), ("CAP", cap)],
        grid=(TG * nc, Hkv, 1), threadgroup=(TG, 1, 1),
        output_shapes=[(Hkv, nc, G, D), (Hkv, nc, G), (Hkv, nc, G)],
        output_dtypes=[mx.float32, mx.float32, mx.float32])
    if nc > 1:                           # merge the chunks' softmaxes onto one maximum
        w = mx.exp(top - top.max(axis=1, keepdims=True))
        out = (acc * w[..., None]).sum(axis=1) / (den * w).sum(axis=1)[..., None]
    else:
        out = acc[:, 0] / den[:, 0][..., None]
    return out.reshape(1, H, 1, D).astype(mx.float16)


def mlx(q, keys, values, n, scale):
    """The reference: MLX's own fused attention on the filled part of the cache."""
    return mx.fast.scaled_dot_product_attention(q, keys[..., :n, :], values[..., :n, :], scale=scale, mask=None)


register("attn_decode", "mlx", mlx)
register("attn_decode", "metal", metal, available=available)
