"""engine/kernels: the selection rules, and every replacement kernel against its MLX reference."""
import os
import unittest

import mlx.core as mx

from engine import kernels
from engine.kernels import attn_decode


def _case(h, hkv, d, n, cap, seed=0):
    mx.random.seed(seed)
    q = mx.random.normal((1, h, 1, d)).astype(mx.float16)
    k = (mx.random.normal((1, hkv, cap, d)) * 0.5).astype(mx.float16)
    v = mx.random.normal((1, hkv, cap, d)).astype(mx.float16)
    return q, k, v, n, d ** -0.5


class Selection(unittest.TestCase):
    def setUp(self):
        self.env = os.environ.get("LLM_KERNELS")

    def tearDown(self):
        if self.env is None:
            os.environ.pop("LLM_KERNELS", None)
        else:
            os.environ["LLM_KERNELS"] = self.env
        kernels.reset()

    def pick(self, spec):
        os.environ["LLM_KERNELS"] = spec
        kernels.reset()
        return kernels.resolve("attn_decode")[0]

    def test_reference_is_the_default(self):
        # a replacement becomes the default only with a benchmark behind it; none has one yet
        self.assertEqual(self.pick("auto"), "mlx")
        self.assertEqual(self.pick(""), "mlx")

    def test_one_name_applies_to_every_operation(self):
        self.assertEqual(self.pick("mlx"), "mlx")

    def test_an_operation_can_be_pinned(self):
        want = "metal" if kernels.implementations("attn_decode")["metal"] else "mlx"
        self.assertEqual(self.pick("attn_decode=metal"), want)
        self.assertEqual(self.pick("other_op=metal"), "mlx")

    def test_an_unknown_or_unavailable_choice_falls_back(self):
        self.assertEqual(self.pick("attn_decode=nonesuch"), "mlx")
        kernels.register("attn_decode", "absent", lambda *a: None, available=lambda: False)
        try:
            self.assertEqual(self.pick("attn_decode=absent"), "mlx")
        finally:
            del kernels._ops["attn_decode"]["absent"]


@unittest.skipUnless(attn_decode.available(), "no Metal device: the Metal attention kernel was not covered")
class AttnDecodeMetal(unittest.TestCase):
    def check(self, h, hkv, d, n, cap=None):
        q, k, v, n, scale = _case(h, hkv, d, n, cap or ((n + 255) // 256) * 256)
        ref = attn_decode.mlx(q, k, v, n, scale).astype(mx.float32)
        out = attn_decode.metal(q, k, v, n, scale)
        self.assertEqual(out.shape, (1, h, 1, d))
        self.assertEqual(out.dtype, mx.float16)
        # float16 outputs of magnitude <= 1: half a unit in the last place is 2.4e-4
        self.assertLess(mx.abs(out.astype(mx.float32) - ref).max().item(), 5e-4)

    def test_one_chunk(self):
        self.check(8, 2, 256, 200)

    def test_a_single_row(self):
        self.check(8, 2, 256, 1)

    def test_several_chunks_and_a_partial_last_one(self):
        self.check(8, 2, 256, 1000)

    def test_exactly_on_a_chunk_boundary(self):
        self.check(8, 2, 256, 512)

    def test_rows_past_the_fill_count_are_ignored(self):
        q, k, v, n, scale = _case(8, 2, 256, 300, 1024)
        a = attn_decode.metal(q, k, v, n, scale)
        k[..., n:, :] = 40.0                       # garbage where the cache is not filled yet
        v[..., n:, :] = -40.0
        b = attn_decode.metal(q, k, v, n, scale)
        self.assertTrue(mx.array_equal(a, b).item())

    def test_no_grouping_and_other_head_sizes(self):
        self.check(4, 4, 64, 700)                  # every query head has its own keys; D < threads
        self.check(16, 8, 128, 700)
        self.check(2, 1, 512, 300)                 # D > threads per threadgroup

    def test_a_peaked_distribution_survives_the_chunk_merge(self):
        q, k, v, n, scale = _case(8, 2, 256, 900, 1024)
        k[..., 700, :] = q[0, 0, 0] * 4            # one row far ahead of the rest, in the third chunk
        ref = attn_decode.mlx(q, k, v, n, scale).astype(mx.float32)
        out = attn_decode.metal(q, k, v, n, scale).astype(mx.float32)
        self.assertLess(mx.abs(out - ref).max().item(), 5e-4)

    def test_it_declines_what_it_does_not_cover(self):
        q, k, v, n, scale = _case(8, 2, 256, 100, 256)
        self.assertIsNone(attn_decode.metal(mx.concatenate([q, q], 2), k, v, n, scale))    # a verify pass
        self.assertIsNone(attn_decode.metal(mx.concatenate([q, q], 0), k, v, n, scale))    # batched rows
        self.assertIsNone(attn_decode.metal(q, k.astype(mx.float32), v, n, scale))


if __name__ == "__main__":
    unittest.main()
