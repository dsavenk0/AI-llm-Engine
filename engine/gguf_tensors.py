"""GGUF tensors → MLX arrays, without llama.cpp or the `gguf` package (numpy only).

A GGUF stores each weight matrix in one of ggml's block formats. MLX multiplies 2–8-bit *affine* groups
(value = scale·code + bias, group of 32/64/128), and nearly all popular ggml formats are exactly that:

    Q4_0 Q4_1 Q4_K   →  4-bit, group 32 (the codes are exact; scales and biases are rounded to fp16)
    Q8_0             →  8-bit, group 32, exact
    Q5_0 Q5_1 Q5_K   →  5-bit, group 32 (as above)
    Q6_K             →  8-bit, group 32: ggml's scale per 16 weights does not fit MLX's per 32, so the values are
                        re-quantized to 8 bits per group — finer than the 6-bit step except where the two
                        16-wide halves differ in scale by more than ~4x

so the weights are repacked once at load, never expanded to fp16: memory and decode speed stay those of the
quantized file. Formats that need lookup tables or odd group shapes (IQ*, Q2_K, Q3_K) are refused with a
clear error so the caller can fall back to llama.cpp.
"""
import struct

import numpy as np

GROUP = 32
# Q6_K has a scale per 16 weights, MLX one per 32: stored in 8 bits (6 bits cost ~5x the error vs llama.cpp for ~8% speed)
Q6K_BITS = 8
# ggml_type ids → (name, block elements, bytes per block)
TYPES = {0: ("F32", 1, 4), 1: ("F16", 1, 2), 2: ("Q4_0", 32, 18), 3: ("Q4_1", 32, 20), 6: ("Q5_0", 32, 22),
         7: ("Q5_1", 32, 24), 8: ("Q8_0", 32, 34), 12: ("Q4_K", 256, 144), 13: ("Q5_K", 256, 176),
         14: ("Q6_K", 256, 210), 30: ("BF16", 1, 2)}
OTHER = {10: "Q2_K", 11: "Q3_K", 15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S", 20: "IQ4_NL",
         21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS", 29: "IQ1_M"}


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


def load_matrix(gf, info, chunk_rows=2048):
    """A 2-D tensor as (w uint32, scales f16, biases f16, bits) numpy arrays ready for mx.quantized_matmul."""
    name = info.type_name
    rows, cols = info.shape
    if name not in ("Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_0", "Q4_K", "Q5_K", "Q6_K"):
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
