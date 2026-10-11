"""engine/ptq1.py tests (stdlib + MLX only — compiles a Metal kernel, kept out of
test_engine.py so that cost isn't paid by every other test run).

    .venv/bin/python -m unittest tests.test_ptq1 -v

engine/ptq1.py is not wired into the engine yet (matvec only — see its module
docstring); these tests pin down its lookup-table kernel and tiled weight layout so
they stay correct while they're tuned or integrated: tile/split/staging-chunk
boundaries, the byte decode, the repack, dtypes.
"""
import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import mlx.core as mx  # noqa: E402
from engine.ptq1 import TILE, dequantize_ref, pack, ptq1_matvec  # noqa: E402


def make_weights(n, k, seed=0):
    """Random PTQ1_0-shaped weights, row-major [n, k/128*28] uint8; the last 2 bytes of
    each block are a plausible fp16 scale."""
    rng = np.random.default_rng(seed)
    nb = k // 128
    w = rng.integers(0, 256, size=(n, nb, 28), dtype=np.uint8)
    d = (rng.random((n, nb)).astype(np.float32) * 0.1 + 0.01).astype(np.float16)
    w[:, :, 26:28] = d.view(np.uint8).reshape(n, nb, 2)
    return w.reshape(n, nb * 28)


def matvec(x, w_np):
    return ptq1_matvec(x, pack(mx.array(w_np)), w_np.shape[0])


class TestPack(unittest.TestCase):
    def test_layout_is_tile_block_row_byte(self):
        n, nb = 1500, 3  # 2 tiles, the second one partly padding
        w = make_weights(n, 128 * nb, seed=11)
        wt = np.asarray(pack(mx.array(w)))
        self.assertEqual(wt.shape, (2, nb, TILE, 28))
        rows = w.reshape(n, nb, 28)
        for tile, b, r in ((0, 0, 0), (0, 2, 5), (0, 1, TILE - 1), (1, 0, 0), (1, 2, n - TILE - 1)):
            np.testing.assert_array_equal(wt[tile, b, r], rows[tile * TILE + r, b])
        # padding rows are zeros, and nothing was dropped
        self.assertTrue((wt[1, :, n - TILE:] == 0).all())

    def test_exact_multiple_of_the_tile_needs_no_padding(self):
        w = make_weights(TILE, 128, seed=12)
        self.assertEqual(tuple(pack(mx.array(w)).shape), (1, 1, TILE, 28))


class TestPTQ1Matvec(unittest.TestCase):
    def assert_matches(self, got, ref, msg=""):
        """The output is rounded to fp16 (<= 2^-11 relative), and a row whose terms nearly
        cancel carries float32 accumulation noise on top — so allow 1e-3 relative plus an
        absolute floor scaled to a typical output, instead of dividing by a near-zero |ref|.
        A mis-decoded trit moves a row by >= d*|x| ~ 1e-2, far outside this."""
        tol = 1e-3 * np.abs(ref) + 1e-4 * np.abs(ref).mean() + 1e-5
        bad = np.abs(got - ref) > tol
        self.assertFalse(bad.any(), f"{msg}: {int(bad.sum())} rows off, worst abs error "
                                    f"{float(np.abs(got - ref).max()):.3g}")

    def check(self, n, k, seed=0):
        w_np = make_weights(n, k, seed)
        x = mx.random.normal((k,)).astype(mx.float16)
        y = matvec(x, w_np)
        mx.eval(y)
        self.assertEqual(y.shape, (n,))
        ref = dequantize_ref(w_np) @ np.asarray(x.astype(mx.float32))
        self.assert_matches(np.asarray(y.astype(mx.float32)), ref, f"n={n} k={k}")

    def test_single_block(self):
        self.check(n=4, k=128)

    def test_multi_block(self):
        self.check(n=8, k=512)

    def test_row_count_not_a_multiple_of_the_tile(self):
        # a threadgroup covers one tile of 1024 rows; none of these are multiples of it
        # (or of 256), exercising the zero-padded tail rows and the final-row guard.
        for n in (1, 3, 5, 13, 255, 257, 1025, 2100):
            self.check(n=n, k=256, seed=n)

    def test_block_counts_around_split_and_chunk_boundaries(self):
        """K is split across threadgroups (min(4, NB // 2, ceil(32 / tiles)) ways, so a
        block count that doesn't divide evenly gives uneven splits) and x is staged
        CH=8 blocks at a time; walk block counts across every one of those edges."""
        for nb in (1, 2, 3, 5, 7, 8, 9, 15, 16, 17, 31, 33, 40):
            self.check(n=300, k=128 * nb, seed=nb)

    def test_splits_longer_than_one_staging_chunk(self):
        """A split only restages x mid-way when it holds more than CH=8 blocks: with a
        single row tile that is NB > 32 (4 splits), e.g. a 17408-wide down projection;
        with 32+ tiles splitting stops (1 split) and it's any NB > 8."""
        for nb in (33, 36, 37, 64, 136, 137, 161):
            self.check(n=130, k=128 * nb, seed=nb)
        self.check(n=33 * TILE + 5, k=128 * 12, seed=5)   # 34 tiles -> one split of 12 blocks

    def test_dtype_follows_the_input(self):
        w_np = make_weights(64, 512, seed=3)
        x32 = mx.random.normal((512,))  # float32
        y = matvec(x32, w_np)
        mx.eval(y)
        self.assertEqual(y.dtype, mx.float32)
        # the kernel reads x as fp16, so compare against the reference on the fp16-rounded x
        ref = dequantize_ref(w_np) @ np.asarray(x32.astype(mx.float16).astype(mx.float32))
        self.assert_matches(np.asarray(y), ref, "float32 input")

    def test_accepts_a_row_vector(self):
        w_np = make_weights(16, 256, seed=4)
        x = mx.random.normal((1, 256)).astype(mx.float16)
        y = matvec(x, w_np)
        mx.eval(y)
        self.assertEqual(y.shape, (16,))

    def test_every_byte_value_in_every_payload_position(self):
        """256 rows, one per possible byte value (0..255) at a given payload byte
        index, with every other byte fixed — exercises every (byte value, trit
        position) combination the three ranges (qs low, qs high, qh) use, instead of
        trusting random sampling to eventually cover the full byte space. Checked
        against dequantize_ref, the same reference the kernel is meant to match (not
        against a hand-derived "expected" value — this format's byte encoding isn't
        plain base-3 positional notation, so deriving one by hand is a trap, not a
        stronger check)."""
        k = 128
        base = np.random.default_rng(7).integers(0, 256, size=28, dtype=np.uint8)
        base[26:28] = np.array([0.37], np.float16).view(np.uint8)
        for byte_index in (0, 16, 24):  # one representative of each of the 3 groups
            w = np.tile(base, (256, 1))
            w[:, byte_index] = np.arange(256, dtype=np.uint8)
            x = mx.random.normal((k,), dtype=mx.float32).astype(mx.float16)
            y = matvec(x, w)
            mx.eval(y)
            ref = dequantize_ref(w) @ np.asarray(x.astype(mx.float32))
            self.assert_matches(np.asarray(y.astype(mx.float32)), ref, f"byte_index={byte_index}")


if __name__ == "__main__":
    unittest.main()
