# PTQ1_0 ternary matrix-vector product (Metal) — decode path for PrismML /
# Bonsai PTQ1_0 bundles (1.75 bits per weight: 28 bytes per 128 weights).
#
# Block layout (PrismML/llama.cpp ggml type 143, as documented in mlx-kquant,
# MIT): [24 x u8 qs][2 x u8 qh][fp16 d]. qs bytes hold five base-3 trits, qh
# bytes four. Element e < 80 is trit e>>4 of qs[e & 15]; 80 <= e < 120 is trit
# (e-80)>>3 of qs[16 + ((e-80) & 7)]; e >= 120 is trit (e-120)>>1 of
# qh[(e-120) & 1]. Trit n of byte b is ((b * 3^n) mod 256) * 3 >> 8 (digit n of
# the ternary expansion of b/256 — not plain base-3 positional notation), the
# weight is (trit - 1) * d.
#
# Two ideas make this faster than MLX's own 2-bit qmm:
#
# 1. Lookup-table GEMV instead of per-weight decoding. Every row shares the same
#    x, so for each payload byte position j of a block a threadgroup builds ONE
#    256-entry table
#        tab[j][b] = sum_n (trit_n(b) - 1) * x[j, n]
#    and a row does one lookup + one add per 5 weights; the "- sum(x)" correction
#    folds into the table. The table is expanded from two tiny sub-tables (3 low
#    trits: 27 entries, 2 high trits: 9) instead of 5 FMAs per entry, x is staged
#    in threadgroup memory, K is split across threadgroups, and each of a
#    threadgroup's 256 threads walks RPT=4 rows, so one table build serves 1024
#    rows. (Per-weight decoding was ALU-bound at ~7 ops/weight: 2x slower than MLX.
#    An ablation showed the lookups themselves are free; the build was the cost.)
#
# 2. A tiled weight layout, [n_tile, NB, 1024 rows, 28 bytes] (see pack()). In
#    plain row-major each thread reads 28 bytes from its own row, 1120+ bytes
#    from its neighbour's, touching a different cache line per lane and using a
#    fraction of each; a pure-read kernel (no tables at all) took 0.47 ms on the
#    34816x5120 matrix there and 0.34 ms tiled, because a threadgroup's block then
#    is one contiguous 28 KB stream. Same bytes, ~30% less time — and once the
#    reads stream, the table build overlaps them instead of adding to them. The
#    repack is a one-time step when a PTQ1 tensor is loaded.
#
# Measured on an M5, decode (M=1), 12 distinct matrices per run (3 for the lm-head)
# so weights come from DRAM like a real decode instead of sitting in the system
# cache; medians of interleaved rounds, wall time per matvec, PTQ1 vs MLX's 2-bit qmm:
#   shape (N x K)      MLX 2-bit   this (tiled)
#   248320 x 5120       3.07 ms     2.47 ms   0.80x   (lm-head)
#   34816  x 5120       0.40 ms     0.34 ms   0.83x   (gate_up)
#   17408  x 5120       0.27 ms     0.23 ms   0.84x
#   10240  x 5120       0.14 ms     0.12 ms   0.86x
#   5120   x 17408      0.23 ms     0.18 ms   0.80x   (down)
#   5120   x 5120       0.08 ms     0.07 ms   0.90x
# i.e. 10-20% faster than MLX while reading ~22% fewer bytes (28 vs 36 per 128
# weights once MLX's scales+biases are counted): 80-115 GB/s of PTQ1 bytes. With the
# same kernel on the row-major layout it was only at parity (+1-4% time).
# In a warm-cache microbenchmark (one matrix called repeatedly) the row-major
# kernel only reached parity — that regime hides exactly the DRAM behaviour that
# the tiled layout fixes, so measure with more weights than the cache holds.
# Tried and not kept: half-precision arithmetic in the old per-weight kernel
# (slower AND ~100x less accurate), vectorized block loads, a per-trit LUT (lanes
# hit data-dependent addresses), prefetching the next block's weights (register
# pressure), more than 4 rows per thread or 512/1024-thread threadgroups (no
# gain), half-precision tables (no gain, less accurate).
#
# Still NOT wired into the engine: matvec only (M=1). Prefill and the 16-row
# speculative verify pass need a matmul path, and the loader needs PTQ1 tensors.
import mlx.core as mx

TG = 256         # threads per threadgroup; thread t owns table entry b = t
RPT = 4          # rows per thread (the table build is amortized over TG * RPT rows)
TILE = TG * RPT  # rows per tile of the packed layout
CH = 8           # blocks of x staged in threadgroup memory at a time
MAX_SPLITS = 4   # K is split across threadgroups when there are few row tiles

_SRC = """
    // w is the tiled layout [n_tile, NB, TG*RPT, 28]; rows past N are zero padding
    const uint tid = thread_position_in_threadgroup.x;
    const uint split = threadgroup_position_in_grid.y;
    constexpr int NB = K / 128;                 // blocks per row
        const int b0 = (int(split) * NB) / KS, b1 = ((int(split) + 1) * NB) / KS;

    threadgroup half xs[CH * 128];              // x of CH blocks at a time
    threadgroup float slo[26 * 27];             // per payload byte: sums over trits 0..2
    threadgroup float shi[26 * 9];              // ... and over trits 3..4
    threadgroup float tab[26 * 256];            // the 256-entry tables, one per payload byte

    // This thread owns byte value `tid` when it expands the sub-tables into table
    // entries: which of the 27 / 9 sub-table slots that byte's trits select.
    uint plo, phi;
    {
        const uint p3[5] = {1u, 3u, 9u, 27u, 81u};
        uint t[5];
        for (int n = 0; n < 5; ++n) t[n] = (((tid * p3[n]) & 0xFFu) * 3u) >> 8;
        plo = t[0] + 3u * t[1] + 9u * t[2];
        phi = t[3] + 3u * t[4];
    }

    float acc[RPT];
    const int tile = int(threadgroup_position_in_grid.x);
    const int row0 = tile * (TG * RPT) + int(tid);
    for (int i = 0; i < RPT; ++i) acc[i] = 0.0f;

    for (int b = b0; b < b1; ++b) {
        if ((b - b0) % CH == 0) {
            // every thread is past the previous block's table build before it gets here
            const int n = min(CH, b1 - b) * 128;
            for (int i = int(tid); i < n; i += TG) xs[i] = x[b * 128 + i];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        threadgroup const half* xb = xs + ((b - b0) % CH) * 128;

        // Sub-tables. Payload byte j packs trits whose x live at o0, o0+st, o0+2st, ...:
        //   qs[0..15]  (j < 16):   x[j + 16 n]            -> o0 = j,        st = 16
        //   qs[16..23] (j < 24):   x[80 + m + 8 n]        -> o0 = 80 + m,   st = 8
        //   qh[0..1]   (j >= 24):  x[120 + h + 2 n], n<4  -> o0 = 120 + h,  st = 2
        // Weights are (t - 1), so the "- sum(x)" correction needs no separate pass.
        for (uint e = tid; e < 26 * 27; e += TG) {
            const uint j = e / 27, p = e - j * 27;
            const uint o0 = j < 16 ? j : (j < 24 ? 80 + (j - 16) : 120 + (j - 24));
            const uint st = j < 16 ? 16u : (j < 24 ? 8u : 2u);
            const float a0 = float(p % 3u) - 1.0f, a1 = float((p / 3u) % 3u) - 1.0f, a2 = float(p / 9u) - 1.0f;
            slo[e] = a0 * float(xb[o0]) + a1 * float(xb[o0 + st]) + a2 * float(xb[o0 + 2 * st]);
        }
        for (uint e = tid; e < 26 * 9; e += TG) {
            const uint j = e / 9, q = e - j * 9;
            const uint o0 = j < 16 ? j : (j < 24 ? 80 + (j - 16) : 120 + (j - 24));
            const uint st = j < 16 ? 16u : (j < 24 ? 8u : 2u);
            const float a3 = float(q % 3u) - 1.0f, a4 = float(q / 3u) - 1.0f;
            // qh bytes only use 4 of the 5 trit slots: no 5th term for j >= 24
            shi[e] = a3 * float(xb[o0 + 3 * st]) + (j < 24 ? a4 * float(xb[o0 + 4 * st]) : 0.0f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int j = 0; j < 26; ++j)
            tab[j * 256 + tid] = slo[j * 27 + plo] + shi[j * 9 + phi];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (int i = 0; i < RPT; ++i) {
            const device uint* p = (const device uint*)(w + ((long(tile) * NB + b) * (TG * RPT) + i * TG + int(tid)) * 28);
            uint qw[7];                          // the block's 28 bytes (4-aligned stride)
            for (int k = 0; k < 7; ++k) qw[k] = p[k];
            float s = 0.0f;
            for (int j = 0; j < 16; ++j)
                s += tab[j * 256 + ((qw[j >> 2] >> (8 * (j & 3))) & 0xFFu)];
            for (int m = 0; m < 8; ++m)
                s += tab[(16 + m) * 256 + ((qw[4 + (m >> 2)] >> (8 * (m & 3))) & 0xFFu)];
            s += tab[24 * 256 + (qw[6] & 0xFFu)] + tab[25 * 256 + ((qw[6] >> 8) & 0xFFu)];
            acc[i] += float(as_type<half2>(qw[6]).y) * s;      // d = bytes 26..27
        }
    }
    for (int i = 0; i < RPT; ++i) {
        const int row = row0 + i * TG;
        if (row < N) y[split * N + row] = acc[i];
    }
"""

_kernel = mx.fast.metal_kernel(
    name="engine_ptq1_lut_qmv",
    input_names=["x", "w"],
    output_names=["y"],
    source=_SRC,
)


def pack(w):
    """Row-major PTQ1_0 weights [N, K/128*28] uint8 -> the tiled layout the kernel reads:
    [n_tile, K/128, TILE, 28], rows zero-padded up to a multiple of TILE. One-time cost
    when a tensor is loaded."""
    n, rb = w.shape
    assert rb % 28 == 0, w.shape
    nb = rb // 28
    n_tile = (n + TILE - 1) // TILE
    w = w.reshape(n, nb, 28)
    if n_tile * TILE != n:
        w = mx.pad(w, [(0, n_tile * TILE - n), (0, 0), (0, 0)])
    return mx.contiguous(w.reshape(n_tile, TILE, nb, 28).transpose(0, 2, 1, 3))


def ptq1_matvec(x, wt, n):
    """x: [K] or [1, K] float16/float32; wt: pack()ed weights; n: number of rows.
    Returns [n] in x's dtype."""
    n_tile, nb = wt.shape[0], wt.shape[1]
    K = nb * 128
    assert x.shape[-1] == K and wt.shape[2:] == (TILE, 28), (x.shape, wt.shape)
    # few row tiles (narrow matrices) can't fill the GPU on their own: split K as well
    splits = max(1, min(MAX_SPLITS, nb // 2, (32 + n_tile - 1) // n_tile))
    (y,) = _kernel(
        inputs=[x.reshape(-1).astype(mx.float16), wt],
        template=[("RPT", RPT), ("TG", TG), ("KS", splits), ("CH", CH), ("K", K), ("N", n)],
        grid=(n_tile * TG, splits, 1),
        threadgroup=(TG, 1, 1),
        output_shapes=[(splits * n,)],
        output_dtypes=[mx.float32],
    )
    y = y.reshape(splits, n)
    return (y[0] if splits == 1 else y.sum(axis=0)).astype(x.dtype)


def dequantize_ref(w):
    """NumPy reference decode (for tests): row-major uint8 [N, K/128*28] -> float32 [N, K]."""
    import numpy as np
    order = []
    for n in range(5):
        for m in range(16):
            order.append((m, n))
    for n in range(5):
        for m in range(8):
            order.append((16 + m, n))
    for n in range(4):
        for h in range(2):
            order.append((24 + h, n))
    order = np.array(order)
    blocks = np.asarray(w, np.uint8).reshape(-1, 28)
    pay = blocks[:, :26].astype(np.uint32)
    d = blocks[:, 26:28].copy().view(np.float16).astype(np.float32)
    pow3 = np.array([1, 3, 9, 27, 81], np.uint32)
    t = (((pay[:, order[:, 0]] * pow3[order[:, 1]]) & 0xFF) * 3) >> 8
    out = (t.astype(np.float32) - 1.0) * d
    return out.reshape(w.shape[0], -1)
