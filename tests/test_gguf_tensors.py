"""Block decoders and the repacking to MLX affine groups, against dequantizers written straight from ggml's
reference code (ggml-quants.c) — plain Python loops, one element at a time, sharing nothing with the engine."""
import os
import struct
import tempfile
import unittest

import numpy as np

from tests import gguf_synth as S

F16 = lambda x: np.float16(x).tobytes()


def _scale_min(j, q):
    if j < 4:
        return q[j] & 63, q[j + 4] & 63
    return (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4), (q[j + 4] >> 4) | ((q[j] >> 6) << 4)


def ref_block(name, raw):
    """One block's float values (list), from the ggml reference algorithm."""
    r = bytes(raw)
    f = lambda o: float(np.frombuffer(r[o:o + 2], np.float16)[0])
    out = []
    if name == "Q4_0":
        d, qs = f(0), r[2:18]
        out = [((qs[j] & 0xF) - 8) * d for j in range(16)] + [((qs[j] >> 4) - 8) * d for j in range(16)]
    elif name == "Q4_1":
        d, m, qs = f(0), f(2), r[4:20]
        out = [(qs[j] & 0xF) * d + m for j in range(16)] + [(qs[j] >> 4) * d + m for j in range(16)]
    elif name in ("Q5_0", "Q5_1"):
        o = 2 if name == "Q5_0" else 4
        d = f(0)
        m = f(2) if name == "Q5_1" else None
        qh = struct.unpack("<I", r[o:o + 4])[0]
        qs = r[o + 4:o + 20]
        lo, hi = [], []
        for j in range(16):
            x0 = (qs[j] & 0xF) | (((qh >> j) << 4) & 0x10)
            x1 = (qs[j] >> 4) | ((qh >> (j + 12)) & 0x10)
            lo.append(x0 * d + m if m is not None else (x0 - 16) * d)
            hi.append(x1 * d + m if m is not None else (x1 - 16) * d)
        out = lo + hi
    elif name == "Q8_0":
        d = f(0)
        out = [struct.unpack("b", r[2 + j:3 + j])[0] * d for j in range(32)]
    elif name == "Q4_K":
        d, dmin, sc, qs = f(0), f(2), r[4:16], r[16:144]
        for g in range(4):
            q = qs[32 * g:32 * g + 32]
            s1, m1 = _scale_min(2 * g, sc)
            s2, m2 = _scale_min(2 * g + 1, sc)
            out += [d * s1 * (q[l] & 0xF) - dmin * m1 for l in range(32)]
            out += [d * s2 * (q[l] >> 4) - dmin * m2 for l in range(32)]
    elif name == "Q5_K":
        d, dmin, sc, qh, qs = f(0), f(2), r[4:16], r[16:48], r[48:176]
        u1, u2 = 1, 2
        for g in range(4):
            q = qs[32 * g:32 * g + 32]
            s1, m1 = _scale_min(2 * g, sc)
            s2, m2 = _scale_min(2 * g + 1, sc)
            out += [d * s1 * ((q[l] & 0xF) + (16 if qh[l] & u1 else 0)) - dmin * m1 for l in range(32)]
            out += [d * s2 * ((q[l] >> 4) + (16 if qh[l] & u2 else 0)) - dmin * m2 for l in range(32)]
            u1 <<= 2
            u2 <<= 2
    elif name == "Q6_K":
        ql, qh = r[0:128], r[128:192]
        sc = struct.unpack("16b", r[192:208])
        d = f(208)
        out = [0.0] * 256
        for half in range(2):
            L, H, SC = ql[64 * half:], qh[32 * half:], sc[8 * half:]
            for l in range(32):
                i = l // 16
                q1 = ((L[l] & 0xF) | (((H[l] >> 0) & 3) << 4)) - 32
                q2 = ((L[l + 32] & 0xF) | (((H[l] >> 2) & 3) << 4)) - 32
                q3 = ((L[l] >> 4) | (((H[l] >> 4) & 3) << 4)) - 32
                q4 = ((L[l + 32] >> 4) | (((H[l] >> 6) & 3) << 4)) - 32
                base = 128 * half
                out[base + l] = d * SC[i] * q1
                out[base + l + 32] = d * SC[i + 2] * q2
                out[base + l + 64] = d * SC[i + 4] * q3
                out[base + l + 96] = d * SC[i + 6] * q4
    return np.array(out, np.float32)


def random_blocks(name, n, rng):
    size = BLOCK_BYTES[name]                                  # ggml's sizes, not the engine's table
    raw = rng.integers(0, 256, (n, size), dtype=np.uint8)
    fields = {"Q4_0": [0], "Q8_0": [0], "Q4_1": [0, 2], "Q5_0": [0], "Q5_1": [0, 2], "Q4_K": [0, 2], "Q5_K": [0, 2],
              "Q6_K": [208]}[name]
    for i in range(n):
        for o in fields:                                     # small finite fp16 scales (random bytes could be NaN)
            raw[i, o:o + 2] = np.frombuffer(F16(rng.uniform(0.002, 0.03)), np.uint8)
    return raw


def write_file(path, tensors):
    """tensors: [(name, dims fastest-first, ggml type, raw bytes)] → a minimal GGUF the reader accepts."""
    meta = {"general.architecture": "llama"}
    head = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(meta)) + b"".join(S._kv(k, v) for k, v in meta.items())
    infos, off, blobs = b"", 0, []
    for name, dims, gt, raw in tensors:
        nb = name.encode()
        infos += struct.pack("<Q", len(nb)) + nb + struct.pack("<I", len(dims)) + struct.pack(f"<{len(dims)}Q", *dims)
        infos += struct.pack("<IQ", gt, off)
        pad = (-len(raw)) % 32
        blobs.append(raw + b"\0" * pad)
        off += len(raw) + pad
    pre = head + infos
    pre += b"\0" * ((-len(pre)) % 32)
    with open(path, "wb") as f:
        f.write(pre + b"".join(blobs))


BLOCK_BYTES = {"Q4_0": 18, "Q4_1": 20, "Q5_0": 22, "Q5_1": 24, "Q8_0": 34, "Q4_K": 144, "Q5_K": 176, "Q6_K": 210}
GGML_ID = {"F32": 0, "F16": 1, "Q4_0": 2, "Q4_1": 3, "Q5_0": 6, "Q5_1": 7, "Q8_0": 8, "Q4_K": 12, "Q5_K": 13,
           "Q6_K": 14, "BF16": 30}


class TestBlockDecoders(unittest.TestCase):
    def test_every_format_matches_the_reference_dequantizer(self):
        from engine import gguf_tensors as T
        rng = np.random.default_rng(7)
        for name, (elems, bits) in {"Q4_0": (32, 4), "Q4_1": (32, 4), "Q5_0": (32, 5), "Q5_1": (32, 5), "Q8_0": (32, 8),
                                    "Q4_K": (256, 4), "Q5_K": (256, 5)}.items():
            raw = random_blocks(name, 5, rng)
            codes, scale, bias, got_bits = T.decode_blocks(name, raw)
            self.assertEqual(got_bits, bits, name)
            self.assertEqual(codes.shape, (5, elems), name)
            self.assertTrue(int(codes.max()) < (1 << bits), name)
            groups = elems // 32
            sc = np.broadcast_to(scale, (5, groups))
            bi = np.broadcast_to(bias, (5, groups))
            got = codes.reshape(5, groups, 32) * sc[:, :, None] + bi[:, :, None]
            want = np.stack([ref_block(name, raw[i]) for i in range(5)]).reshape(5, groups, 32)
            np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-4, err_msg=name)

    def test_q6_k_values_are_exact_and_requantized_to_eight_bits_stay_close(self):
        from engine import gguf_tensors as T
        rng = np.random.default_rng(3)
        raw = random_blocks("Q6_K", 4, rng)
        want = np.stack([ref_block("Q6_K", raw[i]) for i in range(4)])
        np.testing.assert_allclose(T.q6k_values(raw), want, rtol=1e-5, atol=1e-5)
        codes, scale, bias, bits = T.decode_blocks("Q6_K", raw)
        self.assertEqual(bits, T.Q6K_BITS)
        self.assertEqual(bits, 8)
        self.assertEqual((codes.shape, scale.shape, bias.shape), ((4, 256), (4, 8), (4, 8)))
        got = codes.reshape(4, 8, 32) * scale[:, :, None] + bias[:, :, None]
        step = scale[:, :, None]                              # half a level is the most a rounding can cost
        self.assertTrue(np.all(np.abs(got - want.reshape(4, 8, 32)) <= step / 2 + 1e-5))

    def test_requantize_hits_the_group_extremes_and_handles_flat_groups(self):
        from engine import gguf_tensors as T
        w = np.zeros((1, 64), np.float32)
        w[0, :32] = np.linspace(-3, 5, 32)
        w[0, 32:] = 0.5                                       # a flat group: scale 0 must not divide by zero
        codes, scale, bias, bits = T.decode_blocks.__globals__["_requantize"](w, 4)
        self.assertEqual((int(codes[0, :32].min()), int(codes[0, :32].max())), (0, 15))
        self.assertAlmostEqual(float(bias[0, 0]), -3.0)
        self.assertAlmostEqual(float(scale[0, 0]), 8.0 / 15)
        self.assertEqual(float(scale[0, 1]), 0.0)
        self.assertTrue((codes[0, 32:] == 0).all())
        self.assertEqual(float(bias[0, 1]), 0.5)

    def test_unknown_formats_are_refused(self):
        from engine import gguf_tensors as T
        with self.assertRaises(T.UnsupportedQuant):
            T.decode_blocks("Q2_K", np.zeros((1, 84), np.uint8))


class TestPacking(unittest.TestCase):
    def test_codes_pack_as_a_little_endian_bit_stream_like_mlx(self):
        import mlx.core as mx
        from engine import gguf_tensors as T
        rng = np.random.default_rng(1)
        for bits in (4, 5, 8):
            codes = rng.integers(0, 1 << bits, (3, 64), dtype=np.uint8)
            packed = T.pack_codes(codes, bits)
            self.assertEqual(packed.shape, (3, 64 * bits // 32))
            self.assertEqual(packed.dtype, np.uint32)
            scales = mx.ones((3, 2), mx.float16)
            deq = mx.dequantize(mx.array(packed), scales, mx.zeros((3, 2), mx.float16), group_size=32, bits=bits)
            np.testing.assert_array_equal(np.asarray(deq.astype(mx.float32)), codes.astype(np.float32), err_msg=str(bits))
        with self.assertRaises(ValueError):
            T.pack_codes(np.zeros((1, 6), np.uint8), 5)


class TestFile(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.mkdtemp()

    def test_tensor_directory_offsets_and_sizes(self):
        from engine import gguf_tensors as T
        rng = np.random.default_rng(5)
        a = random_blocks("Q4_K", 2 * 3, rng)                  # a [3 rows, 512 cols] tensor
        b = random_blocks("Q8_0", 4 * 2, rng)                  # [2 rows, 128 cols]
        p = os.path.join(self.dir, "t.gguf")
        write_file(p, [("a", (512, 3), 12, a.tobytes()), ("b", (128, 2), 8, b.tobytes())])
        gf = T.GGUFFile(p)
        ia, ib = gf.tensors["a"], gf.tensors["b"]
        self.assertEqual((ia.shape, ia.type_name, ia.n_elements, ia.nbytes), ((3, 512), "Q4_K", 1536, 6 * 144))
        for tname, bytes_per_block, elems in (("Q4_0", 18, 32), ("Q4_1", 20, 32), ("Q5_0", 22, 32), ("Q5_1", 24, 32),
                                              ("Q8_0", 34, 32), ("Q4_K", 144, 256), ("Q5_K", 176, 256),
                                              ("Q6_K", 210, 256), ("F32", 4, 1), ("F16", 2, 1), ("BF16", 2, 1)):
            info = T.TensorInfo("x", (2 * elems,), GGML_ID[tname], 0)
            self.assertEqual((info.type_name, info.nbytes), (tname, 2 * bytes_per_block), tname)
        self.assertEqual((ib.shape, ib.type_name, ib.nbytes), ((2, 128), "Q8_0", 8 * 34))
        self.assertEqual(gf.data_start % 32, 0)
        np.testing.assert_array_equal(np.asarray(gf.raw(ia)), a.reshape(-1))
        np.testing.assert_array_equal(np.asarray(gf.raw(ib)), b.reshape(-1))

    def test_data_starts_on_the_alignment_whatever_the_directory_length(self):
        from engine import gguf_tensors as T
        payload = np.arange(64, dtype=np.float32)
        for pad in range(0, 34):                                  # every residue of the directory's end modulo 32
            p = os.path.join(self.dir, f"align{pad}.gguf")
            write_file(p, [("t" + "x" * pad, (64,), 0, payload.tobytes())])
            gf = T.GGUFFile(p)
            self.assertEqual(gf.data_start % 32, 0, pad)
            np.testing.assert_array_equal(T.load_dense(gf, gf.tensors["t" + "x" * pad]), payload, err_msg=str(pad))

    def test_type_names_cover_known_unsupported_and_unknown_ids(self):
        from engine import gguf_tensors as T
        for tid, name in ((10, "Q2_K"), (11, "Q3_K"), (15, "Q8_K"), (16, "IQ2_XXS"), (17, "IQ2_XS"), (18, "IQ3_XXS"),
                          (19, "IQ1_S"), (20, "IQ4_NL"), (21, "IQ3_S"), (22, "IQ2_S"), (23, "IQ4_XS"), (29, "IQ1_M"),
                          (99, "type 99")):
            self.assertEqual(T.TensorInfo("x", (32,), tid, 0).type_name, name)

    def test_load_matrix_dequantizes_to_the_reference_for_every_format(self):
        import mlx.core as mx
        from engine import gguf_tensors as T
        rng = np.random.default_rng(11)
        rows = 3
        for name, cols, block in (("Q4_0", 64, 32), ("Q4_1", 64, 32), ("Q5_0", 64, 32), ("Q5_1", 64, 32),
                                  ("Q8_0", 64, 32), ("Q4_K", 512, 256), ("Q5_K", 512, 256), ("Q6_K", 512, 256)):
            raw = random_blocks(name, rows * cols // block, rng)
            p = os.path.join(self.dir, f"{name}.gguf")
            write_file(p, [("w", (cols, rows), GGML_ID[name], raw.tobytes())])
            gf = T.GGUFFile(p)
            w, s, b, bits = T.load_matrix(gf, gf.tensors["w"], chunk_rows=2)      # chunking must not matter
            self.assertEqual(w.dtype, np.uint32)
            self.assertEqual(s.dtype, np.float16)
            self.assertEqual((s.shape, b.shape), ((rows, cols // 32), (rows, cols // 32)))
            deq = np.asarray(mx.dequantize(mx.array(w), mx.array(s), mx.array(b), group_size=32, bits=bits)
                             .astype(mx.float32))
            want = np.stack([ref_block(name, raw[i]) for i in range(raw.shape[0])]).reshape(rows, cols)
            tol = float(np.abs(want).max()) * (1 / 255 + 2e-3) if name == "Q6_K" else 0.0   # 8-bit levels + fp16 scales
            np.testing.assert_allclose(deq, want, rtol=3e-3, atol=3e-2 + tol, err_msg=name)

    def test_load_matrix_refuses_what_it_cannot_do(self):
        from engine import gguf_tensors as T
        p = os.path.join(self.dir, "bad.gguf")
        write_file(p, [("odd", (48, 1), 8, bytes(34 * 2)), ("k2", (256, 1), 10, bytes(84)),
                       ("f", (32, 1), 0, bytes(128))])
        gf = T.GGUFFile(p)
        for name in ("odd", "k2", "f"):
            with self.assertRaises(T.UnsupportedQuant, msg=name):
                T.load_matrix(gf, gf.tensors[name])

    def test_dense_formats_and_shapes(self):
        from engine import gguf_tensors as T
        f32 = np.arange(6, dtype=np.float32) - 2.5
        f16 = f32.astype(np.float16)
        bf = (f32.view(np.uint32) >> 16).astype(np.uint16)
        p = os.path.join(self.dir, "d.gguf")
        write_file(p, [("f32", (3, 2), 0, f32.tobytes()), ("f16", (3, 2), 1, f16.tobytes()),
                       ("bf16", (3, 2), 30, bf.tobytes()), ("q", (32, 1), 8, bytes(34))])
        gf = T.GGUFFile(p)
        for n in ("f32", "f16", "bf16"):
            got = T.load_dense(gf, gf.tensors[n])
            self.assertEqual(got.shape, (2, 3), n)
            np.testing.assert_array_equal(got.reshape(-1), f32, err_msg=n)
        with self.assertRaises(T.UnsupportedQuant):
            T.load_dense(gf, gf.tensors["q"])


if __name__ == "__main__":
    unittest.main()
