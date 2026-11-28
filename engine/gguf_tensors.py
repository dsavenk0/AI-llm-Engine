"""GGUF tensors → MLX arrays, without llama.cpp or the `gguf` package (numpy only).

A GGUF stores each weight matrix in one of ggml's block formats. MLX multiplies 2–8-bit *affine* groups
(value = scale·code + bias, group of 32/64/128), and nearly all popular ggml formats are exactly that:

    Q4_0 Q4_1 Q4_K   →  4-bit, group 32 (the codes are exact; scales and biases are rounded to fp16)
    Q8_0             →  8-bit, group 32, exact
    Q5_0 Q5_1 Q5_K   →  5-bit, group 32 (as above)
    Q6_K             →  8-bit, group 32: ggml's scale per 16 weights does not fit MLX's per 32, so the values are
                        re-quantized to 8 bits per group — finer than the 6-bit step except where the two
                        16-wide halves differ in scale by more than ~4x
    Q2_K Q3_K        →  scales (and mins) per 16 weights: decoded exactly, then re-quantized per 32 to 4 / 5 bits
                        (the grid is finer than the 2 / 3-bit steps they started from)
    IQ4_NL IQ4_XS    →  16 non-uniform levels (a lookup table): decoded exactly, re-quantized per 32 to 6 bits
    IQ2_XXS IQ2_XS IQ2_S IQ1_S IQ1_M
                     →  8-value codebook entries (ggml's lookup grids) with sign bits: decoded exactly,
                        re-quantized per 32 to 4 bits
    IQ3_XXS IQ3_S    →  4-value codebook entries with sign bits: decoded exactly, re-quantized per 32 to 5 bits

so the weights are repacked once at load, never expanded to fp16: memory and decode speed stay those of the
quantized file (the re-quantized ones spend a few bits more than the file does — for the 1–3-bit i-quants about
twice the file's size, the price of MLX having no codebook matmul). The grids are ggml's tables (ggml-common.h,
MIT, the ggml authors), kept as text in engine/iq_grids.py.
"""
import struct

import numpy as np

GROUP = 32
# Q6_K has a scale per 16 weights, MLX one per 32: stored in 8 bits (6 bits cost ~5x the error vs llama.cpp for ~8% speed)
Q6K_BITS = 8
# ggml_type ids → (name, block elements, bytes per block)
TYPES = {0: ("F32", 1, 4), 1: ("F16", 1, 2), 2: ("Q4_0", 32, 18), 3: ("Q4_1", 32, 20), 6: ("Q5_0", 32, 22),
         7: ("Q5_1", 32, 24), 8: ("Q8_0", 32, 34), 12: ("Q4_K", 256, 144), 13: ("Q5_K", 256, 176),
         14: ("Q6_K", 256, 210), 30: ("BF16", 1, 2), 10: ("Q2_K", 256, 84), 11: ("Q3_K", 256, 110),
         20: ("IQ4_NL", 32, 18), 23: ("IQ4_XS", 256, 136),
         16: ("IQ2_XXS", 256, 66), 17: ("IQ2_XS", 256, 74), 22: ("IQ2_S", 256, 82), 18: ("IQ3_XXS", 256, 98),
         21: ("IQ3_S", 256, 110), 19: ("IQ1_S", 256, 50), 29: ("IQ1_M", 256, 56),
         42: ("Q2_0", 64, 18), 142: ("PQ2_0", 128, 34), 143: ("PTQ1_0", 128, 28)}
OTHER = {15: "Q8_K"}
# Ternary formats (Prism Bonsai): every weight is (code - 1) * d with code 0..2 (Q2_0 also allows 3), one fp16 d per
# block. That is MLX's affine 2-bit form with scale d and bias -d over the block, so they load exactly, with MLX's
# group set to the block — no re-quantization, the file's own bits.
#   Q2_0    upstream ggml (42): [fp16 d][16 B: 64 codes, 2 bits each, little end first]
#   PQ2_0   Prism (142):        [fp16 d][32 B: 128 codes, same packing]
#   PTQ1_0  Prism (143):        [24 B qs][2 B qh][fp16 d]: 128 trits, five per qs byte and four per qh byte
#                               (the trit order is documented in engine/ptq1.py)
TERNARY = {"Q2_0": 64, "PQ2_0": 128, "PTQ1_0": 128}
# bits the exactly decoded values are re-quantized to (per group of 32)
REQUANT_BITS = {"Q6_K": 8, "Q2_K": 4, "Q3_K": 5, "IQ4_NL": 6, "IQ4_XS": 6,
                "IQ2_XXS": 4, "IQ2_XS": 4, "IQ2_S": 4, "IQ1_S": 4, "IQ1_M": 4, "IQ3_XXS": 5, "IQ3_S": 5}
# The representation produced for MLX. Kept in one place so the memory planner and persistent expert cache do
# not guess a layout independently from `load_matrix`.
MLX_BITS = {"Q4_0": 4, "Q4_1": 4, "Q5_0": 5, "Q5_1": 5, "Q8_0": 8,
            "Q4_K": 4, "Q5_K": 5, **REQUANT_BITS, **{t: 2 for t in TERNARY}}
# ggml's kvalues_iq4nl: the 16 levels of IQ4_NL / IQ4_XS
KVALUES_IQ4NL = np.array([-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113], np.float32)


class UnsupportedQuant(ValueError):
    pass


class TensorInfo:
    def __init__(self, name, dims, ggml_type, offset):
        self.name, self.dims, self.type, self.offset = name, tuple(dims), ggml_type, offset

    @property
    def shape(self):                        # ggml lists the fastest dimension first
        return tuple(reversed(self.dims))

    @property
    def type_name(self):
        return TYPES.get(self.type, (OTHER.get(self.type, f"type {self.type}"),))[0]

    @property
    def n_elements(self):
        n = 1
        for d in self.dims:
            n *= d
        return n

    @property
    def nbytes(self):
        name, block, size = TYPES[self.type]
        return self.n_elements // block * size


class GGUFFile:
    """Tensor directory + memory-mapped data of one GGUF file (split files: open each part)."""

    def __init__(self, path):
        from . import gguf as meta
        self.path = path
        self.meta = meta.read_header_full(path)
        self.tensors = {}
        with open(path, "rb") as f:
            f.seek(self.meta["_tensor_info_offset"])
            for _ in range(self.meta["_tensor_count"]):
                n = struct.unpack("<Q", f.read(8))[0]
                name = f.read(n).decode("utf-8", "replace")
                nd = struct.unpack("<I", f.read(4))[0]
                dims = struct.unpack(f"<{nd}Q", f.read(8 * nd))
                gt, off = struct.unpack("<IQ", f.read(12))
                self.tensors[name] = TensorInfo(name, dims, gt, off)
            end = f.tell()
        align = int(self.meta.get("general.alignment", 32))
        self.data_start = (end + align - 1) // align * align
        self._map = None

    def raw(self, info):
        if self._map is None:
            self._map = np.memmap(self.path, dtype=np.uint8, mode="r")
        start = self.data_start + info.offset
        return self._map[start:start + info.nbytes]


# ───────────────────────────── block decoders: → (codes, scale, bias, bits) ─────────────────────────────
def _scale_min_k4(sc):
    """The 8 (scale, min) pairs packed in 12 bytes of a Q4_K/Q5_K block. sc: [n, 12] uint8 → ([n,8], [n,8])."""
    sc = sc.astype(np.uint16)
    d = np.empty((sc.shape[0], 8), np.uint16)
    m = np.empty_like(d)
    d[:, :4] = sc[:, 0:4] & 63
    m[:, :4] = sc[:, 4:8] & 63
    d[:, 4:] = (sc[:, 8:12] & 0xF) | ((sc[:, 0:4] >> 6) << 4)
    m[:, 4:] = (sc[:, 8:12] >> 4) | ((sc[:, 4:8] >> 6) << 4)
    return d.astype(np.float32), m.astype(np.float32)


def _f16(b):
    """[n, 2] uint8 → float32 [n]"""
    return np.ascontiguousarray(b).view(np.float16).reshape(-1).astype(np.float32)


def decode_blocks(name, blocks):
    """blocks: [n, bytes] uint8. → (codes uint8 [n, elems], scale f32 [n, elems/32], bias f32 [n, elems/32], bits)."""
    n = blocks.shape[0]
    if name == "Q4_0":
        d = _f16(blocks[:, 0:2])
        qs = blocks[:, 2:18]
        codes = np.concatenate([qs & 0xF, qs >> 4], axis=1)
        return codes, d[:, None], (-8 * d)[:, None], 4
    if name == "Q4_1":
        d, m = _f16(blocks[:, 0:2]), _f16(blocks[:, 2:4])
        qs = blocks[:, 4:20]
        return np.concatenate([qs & 0xF, qs >> 4], axis=1), d[:, None], m[:, None], 4
    if name == "Q8_0":
        d = _f16(blocks[:, 0:2])
        codes = (blocks[:, 2:34].view(np.int8).astype(np.int16) + 128).astype(np.uint8)
        return codes, d[:, None], (-128 * d)[:, None], 8
    if name in ("Q5_0", "Q5_1"):
        off = 2 if name == "Q5_0" else 4
        d = _f16(blocks[:, 0:2])
        m = _f16(blocks[:, 2:4]) if name == "Q5_1" else (-16 * d)
        qh = np.ascontiguousarray(blocks[:, off:off + 4]).view("<u4").reshape(n)
        qs = blocks[:, off + 4:off + 20]
        j = np.arange(16, dtype=np.uint32)
        hi0 = ((qh[:, None] >> j) & 1) << 4
        hi1 = ((qh[:, None] >> (j + 16)) & 1) << 4
        codes = np.concatenate([(qs & 0xF) | hi0.astype(np.uint8), (qs >> 4) | hi1.astype(np.uint8)], axis=1)
        return codes, d[:, None], m[:, None], 5
    if name in ("Q4_K", "Q5_K"):
        d, dmin = _f16(blocks[:, 0:2]), _f16(blocks[:, 2:4])
        sc, mn = _scale_min_k4(blocks[:, 4:16])
        scale = d[:, None] * sc                                  # [n, 8]
        bias = -dmin[:, None] * mn
        if name == "Q4_K":
            qs = blocks[:, 16:144].reshape(n, 4, 32)
            parts = []
            for p in range(4):
                parts += [qs[:, p] & 0xF, qs[:, p] >> 4]
            return np.concatenate(parts, axis=1), scale, bias, 4
        qh = blocks[:, 16:48]
        qs = blocks[:, 48:176].reshape(n, 4, 32)
        parts = []
        for p in range(4):
            parts.append((qs[:, p] & 0xF) | (((qh >> (2 * p)) & 1) << 4))
            parts.append((qs[:, p] >> 4) | (((qh >> (2 * p + 1)) & 1) << 4))
        return np.concatenate(parts, axis=1).astype(np.uint8), scale, bias, 5
    if name == "Q6_K":
        return _requantize(q6k_values(blocks), Q6K_BITS)
    if name in EXACT:
        return _requantize(EXACT[name](blocks), REQUANT_BITS[name])
    raise UnsupportedQuant(f"no decoder for {name}")


def q6k_values(blocks):
    """Q6_K blocks [n, 210] → exact float32 values [n, 256] (16 sub-blocks of 16, each with its own scale)."""
    n = blocks.shape[0]
    ql, qh = blocks[:, 0:128], blocks[:, 128:192]
    sc = blocks[:, 192:208].view(np.int8).astype(np.float32)
    d = _f16(blocks[:, 208:210])
    out = np.empty((n, 256), np.float32)
    isub = np.repeat(np.arange(2), 16)                        # element l of a 32-run uses sub-scale l // 16
    for half in range(2):
        l = ql[:, 64 * half:64 * half + 64]
        h = qh[:, 32 * half:32 * half + 32]
        s = sc[:, 8 * half:8 * half + 8]
        q1 = ((l[:, :32] & 0xF) | (((h >> 0) & 3) << 4)).astype(np.int16) - 32
        q2 = ((l[:, 32:] & 0xF) | (((h >> 2) & 3) << 4)).astype(np.int16) - 32
        q3 = ((l[:, :32] >> 4) | (((h >> 4) & 3) << 4)).astype(np.int16) - 32
        q4 = ((l[:, 32:] >> 4) | (((h >> 6) & 3) << 4)).astype(np.int16) - 32
        for k, q in enumerate((q1, q2, q3, q4)):
            out[:, 128 * half + 32 * k:128 * half + 32 * (k + 1)] = q * s[:, isub + 2 * k]
    out *= d[:, None]
    return out


def q2k_values(blocks):
    """Q2_K blocks [n, 84] → float32 [n, 256]: 16 sub-blocks of 16, each with a 4-bit scale and a 4-bit min."""
    n = blocks.shape[0]
    sc = blocks[:, 0:16]
    q = blocks[:, 16:80]
    d, dmin = _f16(blocks[:, 80:82]), _f16(blocks[:, 82:84])
    out = np.empty((n, 256), np.float32)
    for half in range(2):                                      # two runs of 128 elements, 32 quant bytes each
        qq = q[:, 32 * half:32 * half + 32]
        for j in range(4):                                     # four 2-bit planes
            plane = ((qq >> (2 * j)) & 3).astype(np.float32)   # [n, 32]: elements 0-15 then 16-31 of this plane
            for k in range(2):
                i = 8 * half + 2 * j + k
                dl = d * (sc[:, i] & 0xF).astype(np.float32)
                ml = dmin * (sc[:, i] >> 4).astype(np.float32)
                out[:, 128 * half + 32 * j + 16 * k:128 * half + 32 * j + 16 * k + 16] = \
                    dl[:, None] * plane[:, 16 * k:16 * k + 16] - ml[:, None]
    return out


def q3k_values(blocks):
    """Q3_K blocks [n, 110] → float32 [n, 256]: 3-bit codes (2 bits + a high-bit mask), 6-bit scale per 16."""
    n = blocks.shape[0]
    hmask, q = blocks[:, 0:32], blocks[:, 32:96]
    raw = np.ascontiguousarray(blocks[:, 96:108])
    d = _f16(blocks[:, 108:110])
    aux = raw.view("<u4").reshape(n, 3).astype(np.uint32)
    kmask1, kmask2 = np.uint32(0x03030303), np.uint32(0x0F0F0F0F)
    tmp = aux[:, 2]
    a = np.empty((n, 4), np.uint32)
    a[:, 2] = ((aux[:, 0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4)
    a[:, 3] = ((aux[:, 1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4)
    a[:, 0] = (aux[:, 0] & kmask2) | (((tmp >> 0) & kmask1) << 4)
    a[:, 1] = (aux[:, 1] & kmask2) | (((tmp >> 2) & kmask1) << 4)
    scales = np.ascontiguousarray(a).view(np.uint8).reshape(n, 16).astype(np.float32) - 32.0
    out = np.empty((n, 256), np.float32)
    for half in range(2):
        qq = q[:, 32 * half:32 * half + 32]
        for j in range(4):
            m = 4 * half + j                                   # bit of the high-bit mask used by this plane
            low = ((qq >> (2 * j)) & 3).astype(np.int16)
            high = ((hmask >> m) & 1).astype(np.int16)         # 1 → no -4 offset
            code = (low - 4 * (1 - high)).astype(np.float32)   # [n, 32]
            for k in range(2):
                i = 8 * half + 2 * j + k
                out[:, 128 * half + 32 * j + 16 * k:128 * half + 32 * j + 16 * k + 16] = \
                    (d * scales[:, i])[:, None] * code[:, 16 * k:16 * k + 16]
    return out


def iq4nl_values(blocks):
    """IQ4_NL blocks [n, 18] → float32 [n, 32]: d * kvalues[nibble]; low nibbles first, then high."""
    d = _f16(blocks[:, 0:2])
    qs = blocks[:, 2:18]
    vals = np.concatenate([KVALUES_IQ4NL[qs & 0xF], KVALUES_IQ4NL[qs >> 4]], axis=1)
    return vals * d[:, None]


def iq4xs_values(blocks):
    """IQ4_XS blocks [n, 136] → float32 [n, 256]: eight 32-element groups, each a 6-bit scale and 16 bytes of nibbles."""
    n = blocks.shape[0]
    d = _f16(blocks[:, 0:2])
    sh = np.ascontiguousarray(blocks[:, 2:4]).view("<u2").reshape(n).astype(np.uint32)
    sl = blocks[:, 4:8]
    qs = blocks[:, 8:136].reshape(n, 8, 16)
    out = np.empty((n, 256), np.float32)
    for ib in range(8):
        ls = ((sl[:, ib // 2] >> (4 * (ib % 2))) & 0xF).astype(np.int32) | (((sh >> (2 * ib)) & 3).astype(np.int32) << 4)
        dl = d * (ls - 32).astype(np.float32)
        q = qs[:, ib]
        vals = np.concatenate([KVALUES_IQ4NL[q & 0xF], KVALUES_IQ4NL[q >> 4]], axis=1)
        out[:, 32 * ib:32 * ib + 32] = vals * dl[:, None]
    return out


# ───────────────────────────── the codebook i-quants (ggml-quants.c dequantize_row_iq*) ─────────────────────────────
_GRIDS = None
IQ1S_DELTA = 0.125


def _grids():
    """ggml's lookup grids: entry → its 8 (IQ2, IQ1) or 4 (IQ3) values, and the 7-bit → 8-bit sign table."""
    global _GRIDS
    if _GRIDS is None:
        from .iq_grids import load
        _GRIDS = load()
        _GRIDS["f2xxs"] = _GRIDS["iq2xxs_grid"].astype(np.float32)
        _GRIDS["f2xs"] = _GRIDS["iq2xs_grid"].astype(np.float32)
        _GRIDS["f2s"] = _GRIDS["iq2s_grid"].astype(np.float32)
        _GRIDS["f3xxs"] = _GRIDS["iq3xxs_grid"].astype(np.float32)
        _GRIDS["f3s"] = _GRIDS["iq3s_grid"].astype(np.float32)
        _GRIDS["f1s"] = _GRIDS["iq1s_grid"].astype(np.float32)
    return _GRIDS


_BIT = np.arange(8, dtype=np.uint8)


def _signed(grid_vals, sign_bytes):
    """grid values [..., 8] times ±1 by the bits of sign_bytes [...] (bit j → element j)."""
    neg = (sign_bytes[..., None] >> _BIT) & 1
    return grid_vals * (1.0 - 2.0 * neg.astype(np.float32))


def _u32(b):
    return np.ascontiguousarray(b).view("<u4")


def _u16(b):
    return np.ascontiguousarray(b).view("<u2")


def iq2xxs_values(blocks):
    G = _grids()
    n = blocks.shape[0]
    d = _f16(blocks[:, 0:2])
    q = blocks[:, 2:66].reshape(n, 8, 8)
    aux = _u32(q[:, :, 4:8]).reshape(n, 8)
    db = d[:, None] * (0.5 + (aux >> 28).astype(np.float32)) * 0.25
    sidx = (aux[:, :, None] >> (7 * np.arange(4, dtype=np.uint32))) & 127
    vals = _signed(G["f2xxs"][q[:, :, 0:4]], G["ksigns_iq2xs"][sidx])          # [n, 8, 4, 8]
    return (vals * db[:, :, None, None]).reshape(n, 256)


def _iq2_scales(d, sc):
    db0 = d[:, None] * (0.5 + (sc & 0xF).astype(np.float32)) * 0.25
    db1 = d[:, None] * (0.5 + (sc >> 4).astype(np.float32)) * 0.25
    return np.stack([db0, db0, db1, db1], axis=2)                              # [n, 8, 4]: l // 2 picks the half


def iq2xs_values(blocks):
    G = _grids()
    n = blocks.shape[0]
    d = _f16(blocks[:, 0:2])
    qs = _u16(blocks[:, 2:66]).reshape(n, 8, 4)
    vals = _signed(G["f2xs"][qs & 511], G["ksigns_iq2xs"][qs >> 9])
    return (vals * _iq2_scales(d, blocks[:, 66:74])[..., None]).reshape(n, 256)


def iq2s_values(blocks):
    G = _grids()
    n = blocks.shape[0]
    d = _f16(blocks[:, 0:2])
    qs = blocks[:, 2:34].reshape(n, 8, 4).astype(np.uint16)
    signs = blocks[:, 34:66].reshape(n, 8, 4)
    qh = blocks[:, 66:74].astype(np.uint16)
    hi = (qh[:, :, None] << np.array([8, 6, 4, 2], np.uint16)) & 0x300
    vals = _signed(G["f2s"][qs | hi], signs)
    return (vals * _iq2_scales(d, blocks[:, 74:82])[..., None]).reshape(n, 256)


def iq3xxs_values(blocks):
    G = _grids()
    n = blocks.shape[0]
    d = _f16(blocks[:, 0:2])
    qs = blocks[:, 2:66].reshape(n, 8, 8)
    aux = _u32(blocks[:, 66:98]).reshape(n, 8)
    db = d[:, None] * (0.5 + (aux >> 28).astype(np.float32)) * 0.5
    sidx = (aux[:, :, None] >> (7 * np.arange(4, dtype=np.uint32))) & 127
    grid = G["f3xxs"][qs].reshape(n, 8, 4, 8)                                   # two 4-value entries per 8
    vals = _signed(grid, G["ksigns_iq2xs"][sidx])
    return (vals * db[:, :, None, None]).reshape(n, 256)


def iq3s_values(blocks):
    G = _grids()
    n = blocks.shape[0]
    d = _f16(blocks[:, 0:2])
    qs = blocks[:, 2:66].reshape(n, 8, 4, 2).astype(np.uint16)
    qh = blocks[:, 66:74].astype(np.uint16)
    signs = blocks[:, 74:106].reshape(n, 8, 4)
    sc = blocks[:, 106:110]
    nib = np.stack([sc & 0xF, sc >> 4], axis=2).reshape(n, 8).astype(np.float32)   # ib32 even: low, odd: high
    db = d[:, None] * (1 + 2 * nib)
    shifts = np.array([[8, 7], [6, 5], [4, 3], [2, 1]], np.uint16)
    hi = (qh[:, :, None, None] << shifts) & 256
    grid = G["f3s"][qs | hi].reshape(n, 8, 4, 8)
    vals = _signed(grid, signs)
    return (vals * db[:, :, None, None]).reshape(n, 256)


def iq1s_values(blocks):
    G = _grids()
    n = blocks.shape[0]
    d = _f16(blocks[:, 0:2])
    qs = blocks[:, 2:34].reshape(n, 8, 4).astype(np.uint16)
    qh = _u16(blocks[:, 34:50]).reshape(n, 8)
    dl = d[:, None] * (2 * ((qh >> 12) & 7).astype(np.float32) + 1)
    delta = np.where(qh & 0x8000, -IQ1S_DELTA, IQ1S_DELTA).astype(np.float32)
    idx = qs | (((qh[:, :, None] >> (3 * np.arange(4, dtype=np.uint16))) & 7) << 8)
    vals = (G["f1s"][idx] + delta[:, :, None, None]) * dl[:, :, None, None]
    return vals.reshape(n, 256)


def iq1m_values(blocks):
    G = _grids()
    n = blocks.shape[0]
    qs = blocks[:, 0:32].reshape(n, 8, 4).astype(np.uint16)
    qh = blocks[:, 32:48].reshape(n, 8, 2).astype(np.uint16)
    sc = _u16(blocks[:, 48:56]).reshape(n, 4)
    u = (sc[:, 0] >> 12) | ((sc[:, 1] >> 8) & 0x00F0) | ((sc[:, 2] >> 4) & 0x0F00) | (sc[:, 3] & 0xF000)
    d = np.ascontiguousarray(u.astype("<u2")).view(np.float16).astype(np.float32)
    s = sc[:, np.arange(8) // 2]                                                # [n, 8]: the word of each ib
    sh = (6 * (np.arange(8) % 2)).astype(np.uint16)
    dl1 = d[:, None] * (2 * ((s >> sh) & 7).astype(np.float32) + 1)
    dl2 = d[:, None] * (2 * ((s >> (sh + 3)) & 7).astype(np.float32) + 1)
    h0, h1 = qh[:, :, 0], qh[:, :, 1]
    idx = np.stack([qs[:, :, 0] | ((h0 << 8) & 0x700), qs[:, :, 1] | ((h0 << 4) & 0x700),
                    qs[:, :, 2] | ((h1 << 8) & 0x700), qs[:, :, 3] | ((h1 << 4) & 0x700)], axis=2)
    delta = np.stack([h0 & 0x08, h0 & 0x80, h1 & 0x08, h1 & 0x80], axis=2)
    delta = np.where(delta, -IQ1S_DELTA, IQ1S_DELTA).astype(np.float32)
    dl = np.stack([dl1, dl1, dl2, dl2], axis=2)
    vals = (G["f1s"][idx] + delta[..., None]) * dl[..., None]
    return vals.reshape(n, 256)


EXACT = {"Q2_K": q2k_values, "Q3_K": q3k_values, "IQ4_NL": iq4nl_values, "IQ4_XS": iq4xs_values,
         "IQ2_XXS": iq2xxs_values, "IQ2_XS": iq2xs_values, "IQ2_S": iq2s_values, "IQ3_XXS": iq3xxs_values,
         "IQ3_S": iq3s_values, "IQ1_S": iq1s_values, "IQ1_M": iq1m_values}


def _requantize(w, bits):
    """float32 [n, elems] → affine codes per group of 32 (nearest level)."""
    n, e = w.shape
    g = w.reshape(n, e // GROUP, GROUP)
    lo, hi = g.min(axis=2), g.max(axis=2)
    levels = (1 << bits) - 1
    scale = (hi - lo) / levels
    safe = np.where(scale == 0, 1.0, scale)
    codes = np.clip(np.rint((g - lo[:, :, None]) / safe[:, :, None]), 0, levels).astype(np.uint8)
    return codes.reshape(n, e), scale.astype(np.float32), lo.astype(np.float32), bits


# ───────────────────────────── packing for MLX ─────────────────────────────
def pack_codes(codes, bits):
    """uint8 codes [rows, cols] (< 2**bits) → uint32 [rows, cols*bits/32] in MLX's layout: a little-endian
    bit stream, code k of a row at bits [k*bits, (k+1)*bits)."""
    rows, cols = codes.shape
    if (cols * bits) % 32:
        raise ValueError(f"{cols} columns of {bits} bits do not fill whole words")
    shifts = np.arange(bits, dtype=np.uint8)
    bitmat = ((codes[:, :, None] >> shifts) & 1).astype(np.uint8).reshape(rows, cols * bits)
    return np.packbits(bitmat, axis=1, bitorder="little").view("<u4").reshape(rows, cols * bits // 32)


def group_size(info):
    """MLX's group for this tensor once loaded: the block of a ternary format, else 32."""
    return TERNARY.get(info.type_name, GROUP)


_POW3 = np.array([1, 3, 9, 27, 81], dtype=np.uint16)


def _trits(b, n):
    """Trits 0..n-1 of each byte [..., m] → [..., n, m]: trit k of b is ((b * 3^k) mod 256) * 3 >> 8."""
    x = b.astype(np.uint16)[..., None, :] * _POW3[:n, None]
    return (((x & 255) * 3) >> 8).astype(np.uint8)


def ptq1_codes(blk):
    """PTQ1_0 blocks [n, 28] → codes [n, 128] (0..2): elements 0-79 are trits of qs[0:16], 80-119 of qs[16:24],
    120-127 of qh[0:2], each trit-major."""
    n = blk.shape[0]
    return np.concatenate([_trits(blk[:, 0:16], 5).reshape(n, 80), _trits(blk[:, 16:24], 5).reshape(n, 40),
                           _trits(blk[:, 24:26], 4).reshape(n, 8)], axis=1)


def _ptq1_words_mlx(blk):
    """ptq1_codes packed into MLX's 2-bit words, on the GPU: the same arithmetic, ~7x faster than NumPy's gathers
    (a 27B model's PTQ1_0 file: ~12 s instead of ~90 s at load)."""
    import mlx.core as mx
    n = blk.shape[0]
    t = mx.array(blk[:, :26]).astype(mx.uint16)[:, :, None] * mx.array(_POW3)
    t = (((t & 255) * 3) >> 8).astype(mx.uint8)                                  # [n, 26 bytes, 5 trits]
    c = mx.concatenate([t[:, 0:16].transpose(0, 2, 1).reshape(n, 80), t[:, 16:24].transpose(0, 2, 1).reshape(n, 40),
                        t[:, 24:26, :4].transpose(0, 2, 1).reshape(n, 8)], axis=1).reshape(n, 32, 4)
    packed = c[..., 0] | (c[..., 1] << 2) | (c[..., 2] << 4) | (c[..., 3] << 6)  # 4 codes per byte, low first
    return np.array(packed).view("<u4")


def ternary_blocks(name, blk):
    """Blocks [n, bytes] of a ternary format → (words uint32 [n, block/16], scale f16 [n], bias f16 [n])."""
    if name == "PTQ1_0":
        d = np.ascontiguousarray(blk[:, 26:28]).view("<f2")[:, 0]
        try:
            words = _ptq1_words_mlx(blk)
        except ImportError:
            codes = ptq1_codes(blk).astype(np.uint32).reshape(-1, 8, 16)
            words = np.bitwise_or.reduce(codes << (2 * np.arange(16, dtype=np.uint32)), axis=-1).astype("<u4")
    else:                                   # Q2_0 / PQ2_0: the codes are already MLX's 2-bit little-endian stream
        d = np.ascontiguousarray(blk[:, 0:2]).view("<f2")[:, 0]
        words = np.ascontiguousarray(blk[:, 2:]).view("<u4")
    if not np.isfinite(d).all():
        raise UnsupportedQuant(f"{name}: a block scale is not finite")
    return words, d, -d


def _load_ternary(gf, info, chunk_rows):
    name = info.type_name
    rows, cols = info.shape
    _, block, size = TYPES[info.type]
    if cols % block:
        raise UnsupportedQuant(f"{info.name}: {cols} columns are not a multiple of the {name} block ({block})")
    raw = gf.raw(info).reshape(rows, cols // block, size)
    ws, ss, bs = [], [], []
    for r0 in range(0, rows, chunk_rows):
        blk = np.ascontiguousarray(raw[r0:r0 + chunk_rows]).reshape(-1, size)
        words, d, bias = ternary_blocks(name, blk)
        n = min(chunk_rows, rows - r0)
        ws.append(words.reshape(n, cols // 16))
        ss.append(d.reshape(n, cols // block))
        bs.append(bias.reshape(n, cols // block))
    return np.concatenate(ws), np.concatenate(ss), np.concatenate(bs), 2


def load_matrix(gf, info, chunk_rows=2048):
    """A 2-D tensor as (w uint32, scales f16, biases f16, bits) numpy arrays ready for mx.quantized_matmul, with
    the group `group_size(info)`."""
    name = info.type_name
    rows, cols = info.shape
    if name in TERNARY:
        return _load_ternary(gf, info, chunk_rows)
    if name not in ("Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_0", "Q4_K", "Q5_K", "Q6_K") and name not in EXACT:
        raise UnsupportedQuant(f"{info.name}: {name}")
    _, block, size = TYPES[info.type]
    if cols % block:
        raise UnsupportedQuant(f"{info.name}: {cols} columns are not a multiple of the {name} block ({block})")
    per_row = cols // block
    raw = gf.raw(info).reshape(rows, per_row, size)
    ws, ss, bs, bits = [], [], [], None
    for r0 in range(0, rows, chunk_rows):
        blk = np.ascontiguousarray(raw[r0:r0 + chunk_rows]).reshape(-1, size)
        codes, scale, bias, bits = decode_blocks(name, blk)
        codes = codes.reshape(-1, cols)
        scale = scale.reshape(-1, cols // GROUP)
        bias = bias.reshape(-1, cols // GROUP)
        ws.append(pack_codes(codes, bits))
        ss.append(scale.astype(np.float16))
        bs.append(bias.astype(np.float16))
    return np.concatenate(ws), np.concatenate(ss), np.concatenate(bs), bits


def matrix_layout(info):
    """The exact NumPy shapes/dtypes `load_matrix` returns, without decoding the tensor."""
    rows, cols = info.shape
    bits, gs = MLX_BITS.get(info.type_name), group_size(info)
    if bits is None or (cols * bits) % 32 or cols % gs:
        raise UnsupportedQuant(f"{info.name}: no MLX packed layout for {info.type_name}")
    return bits, ((rows, cols * bits // 32), np.dtype("<u4")), \
        ((rows, cols // gs), np.dtype("<f2")), ((rows, cols // gs), np.dtype("<f2"))


def load_dense(gf, info):
    """F32/F16/BF16 tensors as float32 numpy (norms, small matrices, embeddings kept unquantized)."""
    name = info.type_name
    raw = gf.raw(info)
    if name == "F32":
        a = np.ascontiguousarray(raw).view("<f4")
    elif name == "F16":
        a = np.ascontiguousarray(raw).view("<f2").astype(np.float32)
    elif name == "BF16":
        u = np.ascontiguousarray(raw).view("<u2").astype(np.uint32) << 16
        a = u.view(np.float32)
    else:
        raise UnsupportedQuant(f"{info.name}: {name}")
    return a.reshape(info.shape)
