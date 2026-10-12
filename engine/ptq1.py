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
# The same 28-byte block also carries TQ1_G128 (inductiveML / "ternel" bundles, 1.75 bits too):
# 25 payload bytes of five plain base-3 digits (weight 5j+n = digit n of byte j, little end
# first), one tail byte of three digits (weights 125..127), then the fp16 scale. Only the
# trit order differs, so both formats run through the same tiled layout and kernels; `scheme`
# ("ptq1" | "tq1") picks the decode. A tq1 bundle stores [tile][group][slot][row] codes and
# raw fp16 scale bits separately; from_tq1() turns them back into row-major blocks.
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
# Wiring: the engine loads PTQ1 bundles (prism_hadamard_qwen35_ptq1) behind the
# experimental `--ptq1 on` option. Decode (one row) uses the lookup-table kernel above;
# more rows (prefill chunks) dequantize a slab of the matrix to fp16 and use MLX's
# matmul, in row chunks so the temporary stays ~100 MB. Speculative verification
# (2-16 rows) would pay that dequantization for 16 rows' worth of work, so the engine
# turns speculation off for these models.
import mlx.core as mx

TG = 256         # threads per threadgroup; thread t owns table entry b = t
RPT = 4          # rows per thread (the table build is amortized over TG * RPT rows)
TILE = TG * RPT  # rows per tile of the packed layout
CH = 8           # blocks of x staged in threadgroup memory at a time
MAX_SPLITS = 4   # K is split across threadgroups when there are few row tiles

_SRC_PTQ1 = """
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



def _tq1_lut_source(src):
    """The TQ1_G128 variant of the lookup-table kernel: only the byte -> (sub-table slots) split
    and the activation offsets of a payload byte change (plain base-3, five consecutive weights)."""
    def sub(a, b):
        nonlocal src
        assert a in src, a
        src = src.replace(a, b, 1)
    sub("""        plo = t[0] + 3u * t[1] + 9u * t[2];
        phi = t[3] + 3u * t[4];""", """        plo = tid % 27u;                       // digits 0..2 of the byte
        phi = min(tid / 27u, 8u);              // digits 3..4 (bytes > 242 never occur)""")
    sub("""            const uint o0 = j < 16 ? j : (j < 24 ? 80 + (j - 16) : 120 + (j - 24));
            const uint st = j < 16 ? 16u : (j < 24 ? 8u : 2u);
            const float a0""", """            const uint o0 = 5u * j, st = 1u;      // byte j holds weights 5j .. 5j+4
            const float a0""")
    sub("""            const uint o0 = j < 16 ? j : (j < 24 ? 80 + (j - 16) : 120 + (j - 24));
            const uint st = j < 16 ? 16u : (j < 24 ? 8u : 2u);
            const float a3""", """            const uint o0 = 5u * j, st = 1u;
            const float a3""")
    sub("""            // qh bytes only use 4 of the 5 trit slots: no 5th term for j >= 24
            shi[e] = a3 * float(xb[o0 + 3 * st]) + (j < 24 ? a4 * float(xb[o0 + 4 * st]) : 0.0f);""",
        """            // the tail byte (j = 25) holds only three digits: weights 125..127
            shi[e] = j < 25 ? a3 * float(xb[o0 + 3 * st]) + a4 * float(xb[o0 + 4 * st]) : 0.0f;""")
    return src


_SRC = {"ptq1": _SRC_PTQ1, "tq1": _tq1_lut_source(_SRC_PTQ1)}
_kernels = {name: mx.fast.metal_kernel(name=f"engine_{name}_lut_qmv", input_names=["x", "w"],
                                       output_names=["y"], source=src) for name, src in _SRC.items()}


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


def ptq1_matvec(x, wt, n, scheme="ptq1"):
    """x: [K] or [1, K] float16/float32; wt: pack()ed weights; n: number of rows.
    Returns [n] in x's dtype."""
    n_tile, nb = wt.shape[0], wt.shape[1]
    K = nb * 128
    assert x.shape[-1] == K and wt.shape[2:] == (TILE, 28), (x.shape, wt.shape)
    # few row tiles (narrow matrices) can't fill the GPU on their own: split K as well
    splits = max(1, min(MAX_SPLITS, nb // 2, (32 + n_tile - 1) // n_tile))
    (y,) = _kernels[scheme](
        inputs=[x.reshape(-1).astype(mx.float16), wt],
        template=[("RPT", RPT), ("TG", TG), ("KS", splits), ("CH", CH), ("K", K), ("N", n)],
        grid=(n_tile * TG, splits, 1),
        threadgroup=(TG, 1, 1),
        output_shapes=[(splits * n,)],
        output_dtypes=[mx.float32],
    )
    y = y.reshape(splits, n)
    return (y[0] if splits == 1 else y.sum(axis=0)).astype(x.dtype)


_DEQ_HEAD = """
    // one thread = one (row, block): 28 bytes in, 128 half weights out
    const uint tid = thread_position_in_grid.x;
    constexpr int NB = K / 128;
    const uint total = ROWS * NB;
    if (tid >= total) return;
    const uint b = tid % NB, r = tid / NB;
    long base;
    if (TILED) {
        const uint tile = r / (TG * RPT), rr = r % (TG * RPT);
        base = ((long(tile) * NB + b) * (TG * RPT) + rr) * 28;
    } else {
        base = (long(r) * NB + b) * 28;
    }
    uint qw[7];
    for (int k = 0; k < 7; ++k) {
        const device uchar* p = w + base + 4 * k;
        qw[k] = uint(p[0]) | (uint(p[1]) << 8) | (uint(p[2]) << 16) | (uint(p[3]) << 24);
    }
    const float d = float(as_type<half2>(qw[6]).y);
    device half* o = out + (long(r) * NB + b) * 128;
    const uint p3[5] = {1u, 3u, 9u, 27u, 81u};
"""
_DEQ_BODY = {
    "ptq1": """
    for (int j = 0; j < 16; ++j) {
        const uint byte = (qw[j >> 2] >> (8 * (j & 3))) & 0xFFu;
        for (int n = 0; n < 5; ++n)
            o[16 * n + j] = half((float((((byte * p3[n]) & 0xFFu) * 3u) >> 8) - 1.0f) * d);
    }
    for (int m = 0; m < 8; ++m) {
        const uint byte = (qw[4 + (m >> 2)] >> (8 * (m & 3))) & 0xFFu;
        for (int n = 0; n < 5; ++n)
            o[80 + 8 * n + m] = half((float((((byte * p3[n]) & 0xFFu) * 3u) >> 8) - 1.0f) * d);
    }
    for (int h = 0; h < 2; ++h) {
        const uint byte = (qw[6] >> (8 * h)) & 0xFFu;
        for (int n = 0; n < 4; ++n)
            o[120 + 2 * n + h] = half((float((((byte * p3[n]) & 0xFFu) * 3u) >> 8) - 1.0f) * d);
    }
""",
    "tq1": """
    for (int j = 0; j < 26; ++j) {
        const uint byte = (qw[j >> 2] >> (8 * (j & 3))) & 0xFFu;
        const int digits = j < 25 ? 5 : 3;   // the tail byte holds weights 125..127
        for (int n = 0; n < digits; ++n)
            o[5 * j + n] = half((float((byte / p3[n]) % 3u) - 1.0f) * d);
    }
""",
}
_DEQ_SRC = {name: _DEQ_HEAD + body for name, body in _DEQ_BODY.items()}
_deqs = {name: mx.fast.metal_kernel(name=f"engine_{name}_dequant", input_names=["w"], output_names=["out"],
                                    source=src) for name, src in _DEQ_SRC.items()}


def _dequant(w, rows, k, tiled, scheme="ptq1"):
    (out,) = _deqs[scheme](inputs=[w], template=[("ROWS", rows), ("K", k), ("TILED", tiled), ("TG", TG), ("RPT", RPT)],
                  grid=(((rows * (k // 128) + 255) // 256) * 256, 1, 1), threadgroup=(256, 1, 1),
                  output_shapes=[(rows, k)], output_dtypes=[mx.float16])
    return out


def dequantize(wt, n, scheme="ptq1"):
    """Packed (tiled) weights -> fp16 [n, K]. Whole matrix: use only for small ones / tests."""
    return _dequant(wt, wt.shape[0] * TILE, wt.shape[1] * 128, True, scheme)[:n]


def dequantize_rows(w, scheme="ptq1"):
    """Row-major PTQ1 rows [R, K/128*28] (e.g. embedding rows picked by token id) -> fp16 [R, K]."""
    return _dequant(w, w.shape[0], w.shape[1] // 28 * 128, False, scheme)


CHUNK_BYTES = 96 * 2**20  # fp16 weights materialized at a time in the matmul path


def ptq1_matmul(x, wt, n, scheme="ptq1"):
    """x: [..., K] with more than one row. Dequantizes the matrix a few tiles at a time and uses
    MLX's matmul, so the temporary fp16 weights stay small."""
    K = wt.shape[1] * 128
    flat = x.reshape(-1, K).astype(mx.float16)
    tiles = max(1, CHUNK_BYTES // (TILE * K * 2))   # 12 tiles at K=5120, 3 at K=17408
    outs = []
    for t0 in range(0, wt.shape[0], tiles):
        chunk = wt[t0:t0 + tiles]
        rows = min(chunk.shape[0] * TILE, n - t0 * TILE)
        outs.append(flat @ _dequant(chunk, chunk.shape[0] * TILE, K, True, scheme)[:rows].T)
    y = outs[0] if len(outs) == 1 else mx.concatenate(outs, axis=-1)
    return y.reshape(*x.shape[:-1], n).astype(x.dtype)


# --- a few rows (speculative verification, short prompts): fused decode + simdgroup_matrix ---------------
#
# 2..SMALL_MAX rows used to pay for dequantizing the whole matrix to fp16 (2 bytes per weight written and read
# back) for what is a handful of dot products per weight. Here the weights never leave registers: for an 8x8
# simdgroup_matrix a lane holds two neighbouring columns (fn, fn+1) of one row fm, so each lane decodes just
# those weights of its row straight from the 28-byte block into the A fragment, and the activations of the
# tokens come from threadgroup memory as the B fragment. Weight order inside a block does not matter for a dot
# product as long as the activations are staged in the same order, so the k-steps are chosen to make each lane's
# bytes fixed (8 bytes per row and block) and the activations are permuted while they are staged.
# Measured on an M5 (34816 x 5120, 16 rows, weights from DRAM): dequantize+matmul 8.4 ms, this 2.3 ms, the
# one-row lookup-table kernel 0.35 ms. The multiply-accumulates are what it costs now; decode is almost free.
SMALL_MAX = 32          # more rows than this: dequantize + matmul amortizes better
_MM_SG = 4              # simdgroups per threadgroup; each owns 32 rows (4 fragments of 8)
_MM_HEADER = "#include <metal_simdgroup_matrix>\nusing namespace metal;\n"

_MM_SRC = """
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint tid = thread_position_in_threadgroup.x;
    constexpr int NB = K / 128;
    constexpr int THREADS = SG * 32;
    constexpr int tile_rows = TG * RPT;
    constexpr int tg_per_tile = tile_rows / (SG * 32);
    const int tile = int(threadgroup_position_in_grid.x) / tg_per_tile;
    const int sg_row0 = (int(threadgroup_position_in_grid.x) % tg_per_tile) * (SG * 32) + int(sg) * 32;
    const uint q = lane / 4;
    const uint fm = (q & 4u) + ((lane / 2u) % 4u);              // row of this lane's two elements
    const uint fn = (q & 2u) * 2u + (lane % 2u) * 2u;           // their first column

    threadgroup half xs[MT * 8 * 128];
    simdgroup_matrix<float, 8, 8> acc[4][MT];
    for (int r = 0; r < 4; ++r) for (int t = 0; t < MT; ++t) acc[r][t] = simdgroup_matrix<float, 8, 8>(0.0f);
    const uint P3[5] = {1u, 3u, 9u, 27u, 81u};

    for (int b = 0; b < NB; ++b) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int i = int(tid); i < MT * 8 * 128; i += THREADS) {
            const int tok = i / 128, kk = i % 128;
            STAGE
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint by[4][8];                  // the payload bytes this lane's columns live in, per row fragment
        half dh[4];
        for (int r = 0; r < 4; ++r) {
            const device uchar* pp = w + ((long(tile) * NB + b) * tile_rows + sg_row0 + r * 8 + int(fm)) * 28;
            by[r][0] = pp[fn];      by[r][1] = pp[fn + 1];
            by[r][2] = pp[8 + fn];  by[r][3] = pp[9 + fn];
            by[r][4] = pp[16 + fn]; by[r][5] = pp[17 + fn];
            by[r][6] = pp[24];      by[r][7] = pp[25];
            dh[r] = as_type<half>(ushort(uint(pp[26]) | (uint(pp[27]) << 8)));
        }
        #pragma unroll
        for (int s = 0; s < 16; ++s) {
            simdgroup_matrix<half, 8, 8> bm[MT];
            for (int t = 0; t < MT; ++t)
                simdgroup_load(bm[t], xs + t * 8 * 128 + s * 8, 128, ulong2(0, 0), true);
            #pragma unroll
            for (int r = 0; r < 4; ++r) {
                half w0, w1;
                STEP
                simdgroup_matrix<half, 8, 8> am;
                am.thread_elements()[0] = w0;
                am.thread_elements()[1] = w1;
                for (int t = 0; t < MT; ++t)
                    simdgroup_multiply_accumulate(acc[r][t], am, bm[t], acc[r][t]);
            }
        }
    }
    const int yrow0 = tile * tile_rows + sg_row0;
    if (yrow0 < NP) {
        for (int r = 0; r < 4; ++r)
            for (int t = 0; t < MT; ++t) {
                const int rr = yrow0 + r * 8;
                if (rr < NP)
                    simdgroup_store(acc[r][t], y + (t * 8) * NP + rr, NP, ulong2(0, 0), true);
            }
    }
"""
_MM_PARTS = {
    # PTQ1_0: k-step s is one trit position over 8 consecutive payload bytes (steps 0-9: qs[0..15], 10-14: qs[16..23],
    # 15: the qh pair) — contiguous in natural order, so the activations are staged unchanged
    "ptq1": ("xs[i] = x[tok * K + b * 128 + kk];", """
                uint b0v, b1v; int n;
                if (s < 10)      { n = s >> 1;  b0v = by[r][(s & 1) * 2]; b1v = by[r][(s & 1) * 2 + 1]; }
                else if (s < 15) { n = s - 10;  b0v = by[r][4];           b1v = by[r][5]; }
                else             { n = int(fn >> 1); b0v = by[r][6];      b1v = by[r][7]; }
                const uint p = (s == 15) ? P3[min(n, 4)] : P3[n];
                w0 = (half(((b0v * p) & 0xFFu) * 3u >> 8) - 1.0h) * dh[r];
                w1 = (half(((b1v * p) & 0xFFu) * 3u >> 8) - 1.0h) * dh[r];
"""),
    # TQ1_G128: byte j holds weights 5j..5j+4 (plain base 3). Step s = 5g + n (g = byte group of 8, n = digit)
    # takes column c from byte 8g + c, digit n: weight 5(8g + c) + n; step 15 is the last two bytes (24: five
    # digits, 25: three) = weights 120..127 in order. The activations are staged in that order.
    "tq1": ("const int e = (kk < 120) ? 5 * (8 * (kk / 40) + (kk % 8)) + ((kk / 8) % 5) : kk;\n"
            "            xs[i] = x[tok * K + b * 128 + e];", """
                half d0, d1;
                if (s < 15) {
                    const uint pw = P3[s % 5];
                    d0 = half((by[r][(s / 5) * 2] / pw) % 3u);
                    d1 = half((by[r][(s / 5) * 2 + 1] / pw) % 3u);
                } else {
                    const uint c0 = fn, c1 = fn + 1;
                    const uint ba = c0 < 5 ? by[r][6] : by[r][7], bb2 = c1 < 5 ? by[r][6] : by[r][7];
                    d0 = half((ba / P3[c0 < 5 ? c0 : c0 - 5]) % 3u);
                    d1 = half((bb2 / P3[c1 < 5 ? c1 : c1 - 5]) % 3u);
                }
                w0 = (d0 - 1.0h) * dh[r];
                w1 = (d1 - 1.0h) * dh[r];
"""),
}
_mm_kernels = {}


def _mm_kernel(scheme):
    if scheme not in _mm_kernels:
        stage, step = _MM_PARTS[scheme]
        src = _MM_SRC.replace("STAGE", stage).replace("STEP", step)
        _mm_kernels[scheme] = mx.fast.metal_kernel(name=f"engine_{scheme}_mm_small", input_names=["x", "w"],
                                                   output_names=["y"], source=src, header=_MM_HEADER)
    return _mm_kernels[scheme]


def ptq1_matmul_small(x, wt, n, scheme="ptq1"):
    """x: [..., K] with 2..SMALL_MAX rows. Fused decode + 8x8 matrix multiply, 16 rows at a time."""
    K = wt.shape[1] * 128
    flat = x.reshape(-1, K).astype(mx.float16)
    npad = (n + 7) // 8 * 8
    n_tile = wt.shape[0]
    outs = []
    for r0 in range(0, flat.shape[0], 16):
        part = flat[r0:r0 + 16]
        rows = part.shape[0]
        mt = 1 if rows <= 8 else 2
        if rows < mt * 8:
            part = mx.pad(part, [(0, mt * 8 - rows), (0, 0)])
        (y,) = _mm_kernel(scheme)(
            inputs=[part, wt],
            template=[("K", K), ("NP", npad), ("MT", mt), ("SG", _MM_SG), ("TG", TG), ("RPT", RPT)],
            grid=(n_tile * (TILE // (_MM_SG * 32)) * _MM_SG * 32, 1, 1), threadgroup=(_MM_SG * 32, 1, 1),
            output_shapes=[(mt * 8, npad)], output_dtypes=[mx.float32])
        outs.append(y[:rows, :n])
    y = outs[0] if len(outs) == 1 else mx.concatenate(outs, axis=0)
    return y.reshape(*x.shape[:-1], n).astype(x.dtype)


def linear(x, wt, n, scheme="ptq1"):
    """y = x @ W.T for packed weights `wt` and `n` output rows: the lookup-table kernel for a
    single row, a fused decode + matrix-multiply kernel for a few (speculative verification), dequantize
    + matmul for many."""
    rows = x.size // x.shape[-1]
    if rows == 1:
        return ptq1_matvec(x, wt, n, scheme).reshape(*x.shape[:-1], n)
    if rows <= SMALL_MAX:
        return ptq1_matmul_small(x, wt, n, scheme)
    return ptq1_matmul(x, wt, n, scheme)


def dequantize_ref(w, scheme="ptq1"):
    """NumPy reference decode (for tests): row-major uint8 [N, K/128*28] -> float32 [N, K]."""
    import numpy as np
    if scheme == "tq1":
        blocks = np.asarray(w, np.uint8).reshape(-1, 28)
        pay = blocks[:, :26].astype(np.uint32)
        d = blocks[:, 26:28].copy().view(np.float16).astype(np.float32)
        digits = np.stack([(pay // 3 ** n) % 3 for n in range(5)], axis=-1)      # [B, 26, 5]
        wts = digits.reshape(-1, 130)[:, :128].astype(np.float32) - 1.0         # 25*5 + 3 = 128 columns
        return (wts * d).reshape(w.shape[0], -1)
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


TQ1_SLOTS = 26    # payload bytes per group: 25 of five digits + one tail byte of three


def _check_digits(full, tail):
    if mx.max(full).item() > 242 or mx.max(tail).item() > 26:
        raise ValueError("TQ1 codes contain bytes outside the base-3 range (corrupt or not TQ1_G128)")


def from_packed26(weight, scales):
    """JANG `ternary_packed_26b` storage -> row-major 28-byte blocks (the TQ1_G128 digit order, as
    the bundle's own bytes show). weight: uint8 [rows, groups*26]; scales: float16 [rows, groups]."""
    if weight.ndim != 2 or weight.dtype != mx.uint8 or weight.shape[1] % TQ1_SLOTS:
        raise ValueError(f"packed ternary weight must be uint8 [rows, groups*{TQ1_SLOTS}], got {weight.dtype} {tuple(weight.shape)}")
    rows, g = weight.shape[0], weight.shape[1] // TQ1_SLOTS
    if tuple(scales.shape) != (rows, g) or scales.dtype != mx.float16:
        raise ValueError(f"packed ternary scales must be float16 {(rows, g)}, got {scales.dtype} {tuple(scales.shape)}")
    pay = weight.reshape(rows, g, TQ1_SLOTS)
    _check_digits(pay[:, :, :TQ1_SLOTS - 1], pay[:, :, TQ1_SLOTS - 1])
    sc = mx.contiguous(scales).view(mx.uint8).reshape(rows, g, 2)
    return mx.concatenate([pay, sc], axis=-1).reshape(rows, g * 28)


def is_packed26(weight_shape, weight_dtype, scales_shape, scales_dtype):
    """Is (weight, scales) a JANG packed ternary pair? uint8 weights are never an MLX affine/mx
    quantization (those are uint32), so the shapes decide."""
    return (weight_dtype == mx.uint8 and scales_dtype == mx.float16 and len(weight_shape) == 2
            and len(scales_shape) == 2 and weight_shape[0] == scales_shape[0]
            and weight_shape[1] == scales_shape[1] * TQ1_SLOTS)


def from_tq1(codes, scales):
    """TQ1_G128 storage -> row-major 28-byte blocks [N, K/128*28] uint8 (what pack() takes).

    codes:  uint8  [tiles, groups, 26, tile]  (a group's payload bytes, one row per lane)
    scales: uint16 [tiles, groups, tile]      (raw fp16 bits)
    Rejects code bytes no packer emits (> 242, tail > 26): they would decode as silent noise."""
    if codes.ndim != 4 or codes.shape[2] != TQ1_SLOTS or codes.dtype != mx.uint8:
        raise ValueError(f"TQ1 codes must be uint8 [tiles, groups, {TQ1_SLOTS}, tile], got {codes.dtype} {tuple(codes.shape)}")
    t, g, _, tile = codes.shape
    if tuple(scales.shape) != (t, g, tile) or scales.dtype not in (mx.uint16, mx.float16):
        raise ValueError(f"TQ1 scales must be uint16 {(t, g, tile)}, got {scales.dtype} {tuple(scales.shape)}")
    _check_digits(codes[:, :, :TQ1_SLOTS - 1], codes[:, :, TQ1_SLOTS - 1])
    pay = codes.transpose(0, 3, 1, 2)                                  # [t, tile, g, 26]
    sc = scales.transpose(0, 2, 1)                                     # [t, tile, g]
    sc = mx.contiguous(sc).view(mx.uint8).reshape(t, tile, g, 2)       # little-endian fp16 bytes
    return mx.concatenate([pay, sc], axis=-1).reshape(t * tile, g * 28)
