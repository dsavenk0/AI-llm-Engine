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
from engine.ptq1 import (TILE, dequantize, dequantize_ref, from_packed26, from_tq1, is_packed26,  # noqa: E402
                         linear, pack, ptq1_matvec)


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


def tq1_blocks(n, k, seed=0, digits=None):
    """Random TQ1_G128 blocks, row-major [n, k/128*28]: 25 payload bytes of five base-3 digits, a
    tail byte of three, an fp16 scale. Returns (blocks, digits [n, k/128, 128])."""
    rng = np.random.default_rng(seed)
    nb = k // 128
    if digits is None:
        digits = rng.integers(0, 3, size=(n, nb, 128))
    pad = np.concatenate([digits, np.zeros((n, nb, 2), digits.dtype)], axis=-1).reshape(n, nb, 26, 5)
    pay = (pad * np.array([1, 3, 9, 27, 81])).sum(-1).astype(np.uint8)
    d = (rng.random((n, nb)).astype(np.float32) * 0.1 + 0.01).astype(np.float16)
    return np.concatenate([pay, d.view(np.uint8).reshape(n, nb, 2)], axis=-1).reshape(n, nb * 28), digits


def tq1_truth(blocks, digits):
    """What the format says, spelled out: weight (row, 128*g + c) = (digit c - 1) * scale."""
    n, nb, _ = digits.shape
    d = blocks.reshape(n, nb, 28)[:, :, 26:28].copy().view(np.float16).astype(np.float32)
    return ((digits.astype(np.float32) - 1.0) * d).reshape(n, nb * 128)


class TestTQ1Format(unittest.TestCase):
    """TQ1_G128 (inductiveML): same 28-byte block as PTQ1_0, plain base-3 payload."""

    def test_a_hand_built_block_decodes_to_the_digits_written_in_it(self):
        blk = np.zeros((1, 28), np.uint8)
        blk[0, 0] = 1 + 2 * 3 + 0 * 9 + 1 * 27 + 2 * 81   # digits (1, 2, 0, 1, 2) -> weights 0, +1, -1, 0, +1
        blk[0, 25] = 2 + 1 * 3 + 0 * 9                      # tail: digits (2, 1, 0) -> columns 125..127
        blk[0, 26:28] = np.array([0.5], np.float16).view(np.uint8)
        w = dequantize_ref(blk, "tq1")[0]
        np.testing.assert_array_equal(w[:5], [0, 0.5, -0.5, 0, 0.5])
        np.testing.assert_array_equal(w[125:], [0.5, 0, -0.5])
        # every payload byte that is 0 means digits all 0 -> weight -scale, not 0
        self.assertEqual(float(w[5]), -0.5)

    def test_the_reference_matches_the_spelled_out_truth(self):
        blocks, digits = tq1_blocks(37, 512, seed=2)
        np.testing.assert_array_equal(dequantize_ref(blocks, "tq1"), tq1_truth(blocks, digits))

    def test_the_two_formats_read_the_same_bytes_differently(self):
        blocks, _ = tq1_blocks(8, 256, seed=3)
        self.assertGreater(np.abs(dequantize_ref(blocks, "tq1") - dequantize_ref(blocks, "ptq1")).max(), 0.01)
        x = mx.random.normal((256,)).astype(mx.float16)
        a = np.asarray(ptq1_matvec(x, pack(mx.array(blocks)), 8, "tq1").astype(mx.float32))
        b = np.asarray(ptq1_matvec(x, pack(mx.array(blocks)), 8, "ptq1").astype(mx.float32))
        self.assertGreater(np.abs(a - b).max(), 0.01)


class TestTQ1Kernels(unittest.TestCase):
    def assert_close(self, got, ref, msg=""):
        tol = 1e-3 * np.abs(ref) + 1e-4 * np.abs(ref).mean() + 1e-5
        bad = np.abs(got - ref) > tol
        self.assertFalse(bad.any(), f"{msg}: {int(bad.sum())} off, worst {float(np.abs(got - ref).max()):.3g}")

    def check(self, n, k, seed=0):
        blocks, digits = tq1_blocks(n, k, seed)
        truth = tq1_truth(blocks, digits)
        wt = pack(mx.array(blocks))
        x = mx.random.normal((k,)).astype(mx.float16)
        y = ptq1_matvec(x, wt, n, "tq1")
        mx.eval(y)
        self.assert_close(np.asarray(y.astype(mx.float32)), truth @ np.asarray(x.astype(mx.float32)), f"matvec {n}x{k}")
        # the dequantizing path (prefill) reproduces the weights exactly, and linear() routes by rows
        np.testing.assert_allclose(np.asarray(dequantize(wt, n, "tq1").astype(mx.float32)), truth, rtol=1e-3, atol=1e-6)
        X = mx.random.normal((3, k)).astype(mx.float16)
        Y = np.asarray(linear(X, wt, n, "tq1").astype(mx.float32))
        self.assert_close(Y, np.asarray(X.astype(mx.float32)) @ truth.T, f"matmul {n}x{k}")

    def test_shapes_around_the_tile_and_block_edges(self):
        for n, k in ((1, 128), (4, 256), (255, 384), (1025, 640), (300, 128 * 33), (130, 128 * 37)):
            self.check(n, k, seed=n)

    def test_every_byte_value_in_every_kind_of_payload_position(self):
        """256 rows, one per byte value, with the byte under test moved through a first slot, a
        middle one, the last full slot and the tail slot. Legal values only go to 242 (tail 26);
        the rest are clamped by the loader, but the kernel must not read out of its tables."""
        base = np.random.default_rng(9).integers(0, 3, size=(1, 1, 128))
        blocks, _ = tq1_blocks(1, 128, digits=base)
        for slot in (0, 7, 24, 25):
            w = np.tile(blocks, (243, 1))
            hi = 27 if slot == 25 else 243
            w[:, slot] = np.arange(243, dtype=np.uint8) % hi
            x = mx.random.normal((128,)).astype(mx.float16)
            y = ptq1_matvec(x, pack(mx.array(w)), 243, "tq1")
            mx.eval(y)
            ref = dequantize_ref(w, "tq1") @ np.asarray(x.astype(mx.float32))
            self.assert_close(np.asarray(y.astype(mx.float32)), ref, f"slot {slot}")

    def test_from_tq1_restores_row_major_blocks_for_both_tile_shapes(self):
        for n, tile in ((512, 256), (384, 384), (128, 128)):
            blocks, _ = tq1_blocks(n, 384, seed=n)
            nb = 3
            b = blocks.reshape(n, nb, 28)
            tiles = n // tile
            codes = np.ascontiguousarray(b[:, :, :26].reshape(tiles, tile, nb, 26).transpose(0, 2, 3, 1))
            scales = np.ascontiguousarray(b[:, :, 26:].copy().view(np.uint16).reshape(tiles, tile, nb).transpose(0, 2, 1))
            out = from_tq1(mx.array(codes), mx.array(scales))
            np.testing.assert_array_equal(np.asarray(out), blocks)

    def test_from_tq1_refuses_bytes_no_packer_writes(self):
        codes = np.zeros((1, 1, 26, 8), np.uint8)
        scales = np.zeros((1, 1, 8), np.uint16)
        from_tq1(mx.array(codes), mx.array(scales))        # all-zero is legal
        for slot, bad in ((0, 243), (24, 255), (25, 27)):
            c = codes.copy()
            c[0, 0, slot, 3] = bad
            with self.assertRaises(ValueError, msg=f"slot {slot} byte {bad}"):
                from_tq1(mx.array(c), mx.array(scales))
        with self.assertRaises(ValueError):                # wrong slot count (a PTQ1_0 `qs` array)
            from_tq1(mx.array(np.zeros((1, 1, 24, 8), np.uint8)), mx.array(scales))


class TestJangPacked(unittest.TestCase):
    """JANG ternary_packed_26b: uint8 [rows, groups*26] + float16 [rows, groups]."""

    def test_detection_is_by_dtype_and_shape(self):
        u8, f16, u32 = mx.uint8, mx.float16, mx.uint32
        self.assertTrue(is_packed26((64, 40 * 26), u8, (64, 40), f16))
        self.assertFalse(is_packed26((64, 40 * 26 + 1), u8, (64, 40), f16))     # not exactly 26 bytes per group
        self.assertFalse(is_packed26((64, 40 * 27), u8, (64, 40), f16))
        self.assertFalse(is_packed26((64, 40 * 26), u8, (65, 40), f16))         # rows disagree
        self.assertFalse(is_packed26((64, 40 * 26), u32, (64, 40), f16))        # an MLX affine weight
        self.assertFalse(is_packed26((64, 40 * 26), u8, (64, 40), u8))          # an mx-mode (mxfp4) scale
        self.assertFalse(is_packed26((64, 40, 26), u8, (64, 40), f16))

    def test_blocks_match_the_tile_layout_of_the_same_weights(self):
        blocks, digits = tq1_blocks(96, 384, seed=4)
        b = blocks.reshape(96, 3, 28)
        w = mx.array(np.ascontiguousarray(b[:, :, :26].reshape(96, 78)))
        s = mx.array(np.ascontiguousarray(b[:, :, 26:].copy().view(np.float16).reshape(96, 3)))
        out = from_packed26(w, s)
        np.testing.assert_array_equal(np.asarray(out), blocks)
        np.testing.assert_array_equal(dequantize_ref(np.asarray(out), "tq1"), tq1_truth(blocks, digits))

    def test_refuses_what_is_not_a_packed_pair(self):
        w = mx.zeros((8, 52), mx.uint8)
        s = mx.zeros((8, 2), mx.float16)
        from_packed26(w, s)
        with self.assertRaises(ValueError):
            from_packed26(mx.zeros((8, 53), mx.uint8), s)
        with self.assertRaises(ValueError):
            from_packed26(w, mx.zeros((8, 3), mx.float16))
        with self.assertRaises(ValueError):
            from_packed26(w, mx.zeros((8, 2), mx.float32))
        bad = np.zeros((8, 52), np.uint8)
        bad[2, 25] = 27                                   # a tail byte holds three digits: 26 at most
        with self.assertRaises(ValueError):
            from_packed26(mx.array(bad), s)


if __name__ == "__main__":
    unittest.main()
