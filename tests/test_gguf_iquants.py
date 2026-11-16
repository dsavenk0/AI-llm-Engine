"""The 1–3-bit codebook i-quants (IQ1_S IQ1_M IQ2_XXS IQ2_XS IQ2_S IQ3_XXS IQ3_S) in engine/gguf_tensors.py.

The expected values were produced by llama.cpp's own numpy dequantizer (the `gguf` package, 0.19.0) from the same
random blocks, and the decoders matched it exactly on 64 random blocks of each type when they were written; the
checksums below keep them that way without needing that package here.

    .venv/bin/python -m unittest tests.test_gguf_iquants -v
"""
import unittest

import numpy as np

from engine import gguf_tensors as T

# name → (sum, sum of squares, the first six values of the second block), from gguf.quants.dequantize
REFERENCE = {'IQ1_M': [1450559.248334, 20202245390809.26, [-178640.0, 25520.0, -178640.0, 25520.0, -178640.0, 25520.0]],
 'IQ1_S': [-1.330284, 4.538814, [0.0087519, -0.0012503, 0.0087519, 0.0087519, -0.0012503, -0.0012503]],
 'IQ2_S': [-4.040863, 290.270383, [-0.070015, -0.2187967, -0.070015, -0.2187967, -0.2187967, 0.3763304]],
 'IQ2_XS': [-10.433478, 324.068905, [0.2300491, -0.7189035, 0.2300491, -0.2300491, 0.2300491, 0.2300491]],
 'IQ2_XXS': [-26.255608, 274.343847, [-0.0300064, 0.0300064, 0.09377, 0.09377, 0.0300064, -0.0300064]],
 'IQ3_S': [-25.365417, 1926.764539, [-0.9502029, -0.9502029, 0.1900406, -1.3302841, -1.7103653, 0.9502029]],
 'IQ3_XXS': [60.452911, 1966.370705, [-1.0852318, 0.4901047, 0.3500748, -0.070015, 0.7701645, 1.0852318]]}


def blocks(name):
    tid = next(k for k, v in T.TYPES.items() if v[0] == name)
    rng = np.random.default_rng(7)
    raw = rng.integers(0, 256, (4, T.TYPES[tid][2]), dtype=np.uint8)
    if name != "IQ1_M":                          # a sane fp16 super-block scale (IQ1_M spreads its own over the block)
        raw[:, 0:2] = np.frombuffer(np.full(4, 0.01, np.float16).tobytes(), np.uint8).reshape(4, 2)
    return raw


class TestCodebookQuants(unittest.TestCase):
    def test_each_decoder_gives_llama_cpps_values(self):
        for name, (total, squares, first) in REFERENCE.items():
            v = T.EXACT[name](blocks(name)).astype(np.float64)
            self.assertEqual(v.shape, (4, 256), name)
            np.testing.assert_allclose(v.sum(), total, rtol=1e-6, atol=1e-5, err_msg=name)
            np.testing.assert_allclose((v ** 2).sum(), squares, rtol=1e-6, err_msg=name)
            np.testing.assert_allclose(v[1, :6], first, rtol=1e-5, atol=1e-6, err_msg=name)

    def test_they_load_as_mlx_affine_groups_close_to_the_exact_values(self):
        class File:                               # load_matrix needs only raw bytes and the tensor's shape
            def __init__(self, raw):
                self._raw = raw

            def raw(self, info):
                return self._raw
        for name in REFERENCE:
            tid = next(k for k, v in T.TYPES.items() if v[0] == name)
            raw = blocks(name)
            if name == "IQ1_M":                   # its scale is spread over the top nibbles of 4 words: make it 0.01
                u = int(np.float16(0.01).view(np.uint16))
                sc = raw[:, 48:56].copy().view("<u2")
                for i in range(4):
                    sc[:, i] = (sc[:, i] & 0x0FFF) | (((u >> (4 * i)) & 0xF) << 12)
                raw[:, 48:56] = sc.view(np.uint8)
            info = T.TensorInfo("w", (512, 2), tid, 0)          # 2 rows of 512 columns = 4 blocks
            w, s, b, bits = T.load_matrix(File(raw.reshape(-1)), info)
            self.assertEqual(bits, T.REQUANT_BITS[name], name)
            codes = np.unpackbits(w.view(np.uint8), bitorder="little").reshape(2, -1, bits)
            codes = (codes * (1 << np.arange(bits))).sum(-1)
            got = codes.reshape(2, -1, 32) * s.astype(np.float32)[:, :, None] + b.astype(np.float32)[:, :, None]
            want = T.EXACT[name](raw).reshape(2, -1, 32)
            step = (want.max(-1) - want.min(-1)) / ((1 << bits) - 1)
            # nearest level of the re-quantized grid: within half a step (plus fp16 rounding of scale and bias)
            self.assertTrue(np.all(np.abs(got - want) <= 0.5 * step[..., None] * 1.02 + 1e-3 * np.abs(want).max()), name)

    def test_the_native_route_accepts_them(self):
        from engine import gguf
        self.assertTrue({16, 17, 18, 19, 21, 22, 29} <= gguf.NATIVE_TYPES)


if __name__ == "__main__":
    unittest.main()
