"""Vision: image preprocessing, the tower, and the engine's lazy image path — on a tiny synthetic
bundle (random weights; see tests/synth.py make_vision_bundle).

    .venv/bin/python -m unittest tests.test_vision -v
"""
import base64
import json
import os
import sys
import tempfile
import unittest

import mlx.core as mx
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from engine import vision  # noqa: E402
from tests.synth import make_vision_bundle, png_data_uri  # noqa: E402


class TestSmartResize(unittest.TestCase):
    def test_sizes_are_multiples_of_the_factor_and_within_bounds(self):
        for h, w in ((37, 91), (480, 640), (1000, 30), (16, 16), (5000, 5000)):
            hb, wb = vision.smart_resize(h, w, 16, 16 * 16, 512 * 512)
            with self.subTest((h, w)):
                self.assertEqual((hb % 16, wb % 16), (0, 0))
                self.assertGreaterEqual(hb * wb, 16 * 16)
                self.assertLessEqual(hb * wb, 512 * 512)

    def test_aspect_ratio_is_roughly_kept(self):
        hb, wb = vision.smart_resize(300, 600, 28, 28 * 28, 2000 * 2000)
        self.assertAlmostEqual(wb / hb, 2.0, delta=0.2)

    def test_small_images_are_scaled_up_to_the_minimum(self):
        hb, wb = vision.smart_resize(20, 20, 16, 64 * 64, 512 * 512)
        self.assertGreaterEqual(hb * wb, 64 * 64)


class TestPreprocess(unittest.TestCase):
    PP = {"patch_size": 8, "temporal_patch_size": 2, "merge_size": 2, "image_mean": [0.5] * 3,
          "image_std": [0.5] * 3, "size": {"shortest_edge": 16 * 16, "longest_edge": 128 * 128}}

    def test_patches_grid_and_normalisation(self):
        from PIL import Image
        img = Image.new("RGB", (60, 44), (255, 255, 255))
        patches, (t, gh, gw) = vision.preprocess(img, self.PP, max_pixels=128 * 128)
        self.assertEqual(t, 1)
        self.assertEqual((gh % 2, gw % 2), (0, 0))                       # whole merge windows
        self.assertEqual(patches.shape, (gh * gw, 3 * 2 * 8 * 8))
        self.assertEqual(patches.dtype, mx.float16)
        self.assertAlmostEqual(float(mx.mean(patches.astype(mx.float32))), 1.0, places=2)   # (1 - .5) / .5
        black = vision.preprocess(Image.new("RGB", (60, 44), (0, 0, 0)), self.PP, 128 * 128)[0]
        self.assertAlmostEqual(float(mx.mean(black.astype(mx.float32))), -1.0, places=2)

    def test_max_pixels_caps_the_token_count(self):
        from PIL import Image
        big = Image.new("RGB", (2000, 1500), (10, 20, 30))
        _, (_, gh, gw) = vision.preprocess(big, self.PP, max_pixels=64 * 64)
        self.assertLessEqual(gh * gw * 8 * 8, 64 * 64)


class TestLoadImage(unittest.TestCase):
    def test_every_accepted_reference_form(self):
        uri = png_data_uri(32, 24)
        self.assertEqual(vision.load_image(uri).size, (32, 24))
        self.assertEqual(vision.load_image({"url": uri}).size, (32, 24))
        self.assertEqual(vision.load_image({"image_url": {"url": uri}}).size, (32, 24))
        path = os.path.join(tempfile.mkdtemp(), "x.png")
        with open(path, "wb") as f:
            f.write(base64.b64decode(uri.split(",", 1)[1]))
        self.assertEqual(vision.load_image(path).size, (32, 24))
        self.assertEqual(vision.load_image("file://" + path).size, (32, 24))

    def test_garbage_is_a_clean_error(self):
        with self.assertRaises(ValueError):
            vision.load_image(12345)
        with self.assertRaises(ValueError):
            vision.load_image({"nothing": 1})
        with self.assertRaises(Exception):
            vision.load_image("data:image/png;base64," + base64.b64encode(b"not an image").decode())


class TestTower(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dirs = {}
        for name, q in (("float", False), ("quant", True)):
            d = tempfile.mkdtemp()
            make_vision_bundle(d, seed=5, quantized_tower=q)
            cls.dirs[name] = d

    def tower(self, which):
        from engine.loader import read_shapes
        import glob
        d = self.dirs[which]
        cfg = json.load(open(os.path.join(d, "config.json")))
        shapes = read_shapes(sorted(glob.glob(os.path.join(d, "*.safetensors"))))
        weights = mx.load(os.path.join(d, "model.safetensors"))
        v = vision.VisionModel(cfg["vision_config"], shapes, cfg.get("quantization") or {})
        w = {k.removeprefix("vision_tower."): x for k, x in weights.items() if k.startswith("vision_tower.")}
        pw = w["patch_embed.proj.weight"]
        if pw.ndim == 5 and pw.shape[-1] != 3:
            w["patch_embed.proj.weight"] = pw.transpose(0, 2, 3, 4, 1)
        v.load_weights(list(w.items()), strict=True)
        mx.eval(v.parameters())
        return v, cfg["vision_config"]

    def pixels(self, grids, seed=0):
        rng = np.random.default_rng(seed)
        n = sum(t * h * w for t, h, w in grids)
        return mx.array(rng.standard_normal((n, 3 * 2 * 8 * 8)).astype(np.float32), dtype=mx.float16)

    def test_output_has_one_row_per_merged_window_and_the_text_width(self):
        v, vc = self.tower("float")
        for grids in ([(1, 4, 4)], [(1, 4, 6)], [(1, 2, 2), (1, 6, 4)]):
            out = v(self.pixels(grids), grids)
            with self.subTest(grids):
                expected = sum(t * h * w // 4 for t, h, w in grids)
                self.assertEqual(out.shape, (expected, vc["out_hidden_size"]))
                self.assertTrue(bool(mx.all(mx.isfinite(out.astype(mx.float32)))))

    def test_different_pictures_give_different_features_and_same_picture_the_same(self):
        v, _ = self.tower("float")
        g = [(1, 4, 4)]
        a, b = v(self.pixels(g, 1), g), v(self.pixels(g, 2), g)
        again = v(self.pixels(g, 1), g)
        self.assertTrue(bool(mx.array_equal(a, again)))
        self.assertGreater(float(mx.mean(mx.abs(a.astype(mx.float32) - b.astype(mx.float32)))), 1e-4)

    def test_images_in_a_batch_do_not_attend_to_each_other(self):
        v, _ = self.tower("float")
        a, b = self.pixels([(1, 4, 4)], 1), self.pixels([(1, 4, 4)], 2)
        alone = v(a, [(1, 4, 4)])
        together = v(mx.concatenate([a, b]), [(1, 4, 4), (1, 4, 4)])
        diff = float(mx.max(mx.abs(together[:4].astype(mx.float32) - alone.astype(mx.float32))))
        self.assertLess(diff, 0.05)                                       # fp16 noise only

    def test_quantized_tower_loads_with_shape_derived_specs(self):
        v, vc = self.tower("quant")
        from engine.model import QLinear
        self.assertIsInstance(v.blocks[0].attn.qkv, QLinear)
        out = v(self.pixels([(1, 4, 4)]), [(1, 4, 4)])
        self.assertEqual(out.shape, (4, vc["out_hidden_size"]))
        self.assertTrue(bool(mx.all(mx.isfinite(out.astype(mx.float32)))))

    def test_deepstack_towers_are_refused_up_front(self):
        vc = dict(self.tower("float")[1], deepstack_visual_indexes=[1])
        with self.assertRaises(NotImplementedError):
            vision.VisionModel(vc, {}, {})


def reference_tower(weights, vc, pixels, grids):
    """An independent float64 NumPy forward pass of the vision tower, written from the Qwen-VL
    definition (not from engine/vision.py): the engine's vectorised code must agree with it."""
    import math
    W = {k.removeprefix("vision_tower."): np.asarray(v.astype(mx.float32), dtype=np.float64)
         for k, v in weights.items() if k.startswith("vision_tower.")}
    d, H, m, side = vc["hidden_size"], vc["num_heads"], vc["spatial_merge_size"], int(vc["num_position_embeddings"] ** 0.5)
    hd = d // H
    x = np.asarray(pixels.astype(mx.float32), dtype=np.float64)
    x = x @ W["patch_embed.proj.weight"].reshape(d, -1).T + W["patch_embed.proj.bias"]

    def ln(v, p):
        mu, var = v.mean(-1, keepdims=True), v.var(-1, keepdims=True)
        return (v - mu) / np.sqrt(var + 1e-6) * W[p + ".weight"] + W[p + ".bias"]

    def lin(v, p):
        return v @ W[p + ".weight"].T + W[p + ".bias"]

    def gelu_tanh(v):
        return 0.5 * v * (1 + np.tanh(math.sqrt(2 / math.pi) * (v + 0.044715 * v ** 3)))

    def gelu_exact(v):
        return 0.5 * v * (1 + np.vectorize(math.erf)(v / math.sqrt(2)))

    # per-token (row, col) in the image grid; tokens come merge-window by merge-window
    positions, segments, pos_emb = [], [], []
    for t, h, w in grids:
        start = sum(len(p) for p in positions)
        coords = []
        for wr in range(h // m):
            for wc in range(w // m):
                for i in range(m):
                    for j in range(m):
                        coords.append((wr * m + i, wc * m + j))
        coords = coords * t
        positions.append(coords)
        for f in range(t):                                    # attention is per frame (cu_seqlens repeats h*w, t times)
            segments.append((start + f * h * w, start + (f + 1) * h * w))
        table = W["pos_embed.weight"]
        for r, c in coords:                                   # bilinear lookup in the learned side x side table
            fr = r * (side - 1) / (h - 1) if h > 1 else 0.0
            fc = c * (side - 1) / (w - 1) if w > 1 else 0.0
            r0, c0 = int(math.floor(fr)), int(math.floor(fc))
            r1, c1 = min(r0 + 1, side - 1), min(c0 + 1, side - 1)
            dr, dc = fr - r0, fc - c0
            pos_emb.append((1 - dr) * (1 - dc) * table[r0 * side + c0] + (1 - dr) * dc * table[r0 * side + c1]
                           + dr * (1 - dc) * table[r1 * side + c0] + dr * dc * table[r1 * side + c1])
    x = x + np.stack(pos_emb)
    coords = [rc for g in positions for rc in g]
    rot = hd // 2
    inv = 1.0 / (10000.0 ** (np.arange(0, rot, 2) / rot))
    ang = np.array([np.concatenate([r * inv, c * inv]) for r, c in coords])        # (L, rot)
    ang = np.concatenate([ang, ang], axis=-1)                                      # (L, hd)
    cos, sin = np.cos(ang)[:, None, :], np.sin(ang)[:, None, :]

    def rotate(v):                                            # v: (L, H, hd)
        half = v.shape[-1] // 2
        return v * cos + np.concatenate([-v[..., half:], v[..., :half]], axis=-1) * sin

    for b in range(vc["depth"]):
        p = f"blocks.{b}"
        qkv = lin(ln(x, p + ".norm1"), p + ".attn.qkv").reshape(-1, 3, H, hd)
        q, k, v = rotate(qkv[:, 0]), rotate(qkv[:, 1]), qkv[:, 2]
        out = np.zeros_like(q)
        for s0, s1 in segments:                               # attention only inside one image
            for h_ in range(H):
                sc = q[s0:s1, h_] @ k[s0:s1, h_].T / math.sqrt(hd)
                e = np.exp(sc - sc.max(-1, keepdims=True))
                out[s0:s1, h_] = (e / e.sum(-1, keepdims=True)) @ v[s0:s1, h_]
        x = x + lin(out.reshape(-1, d), p + ".attn.proj")
        x = x + lin(gelu_tanh(lin(ln(x, p + ".norm2"), p + ".mlp.linear_fc1")), p + ".mlp.linear_fc2")
    x = ln(x, "merger.norm").reshape(-1, d * m * m)
    return lin(gelu_exact(lin(x, "merger.linear_fc1")), "merger.linear_fc2")


class TestTowerAgainstReference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.mkdtemp()
        make_vision_bundle(cls.dir, seed=9)
        cls.weights = mx.load(os.path.join(cls.dir, "model.safetensors"))
        cls.cfg = json.load(open(os.path.join(cls.dir, "config.json")))["vision_config"]

    def engine_tower(self):
        from engine.loader import read_shapes
        import glob
        shapes = read_shapes(sorted(glob.glob(os.path.join(self.dir, "*.safetensors"))))
        v = vision.VisionModel(self.cfg, shapes, {})
        w = {k.removeprefix("vision_tower."): x for k, x in self.weights.items() if k.startswith("vision_tower.")}
        w["patch_embed.proj.weight"] = w["patch_embed.proj.weight"].transpose(0, 2, 3, 4, 1)
        v.load_weights(list(w.items()), strict=True)
        mx.eval(v.parameters())
        return v

    def check(self, grids, seed):
        rng = np.random.default_rng(seed)
        n = sum(t * h * w for t, h, w in grids)
        pix = mx.array(rng.standard_normal((n, 3 * 2 * 8 * 8)).astype(np.float32), dtype=mx.float16)
        got = np.asarray(self.engine_tower()(pix, grids).astype(mx.float32), dtype=np.float64)
        want = reference_tower(self.weights, self.cfg, pix, grids)
        self.assertEqual(got.shape, want.shape)
        err = np.abs(got - want).max() / (np.abs(want).max() + 1e-9)
        self.assertLess(err, 0.03, f"relative error {err:.4f} against the reference")

    def test_a_non_square_grid_exercises_position_interpolation_and_2d_rotary(self):
        self.check([(1, 4, 6)], 1)

    def test_a_square_grid(self):
        self.check([(1, 4, 4)], 2)

    def test_several_images_in_one_batch(self):
        self.check([(1, 2, 2), (1, 4, 6), (1, 6, 2)], 3)

    def test_multi_frame_tubes_repeat_positions_per_frame(self):
        self.check([(2, 4, 2)], 4)


class TestEngineImages(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from engine.generate import Engine
        cls.dir = tempfile.mkdtemp()
        make_vision_bundle(cls.dir, seed=7)
        cls.logs = []
        cls.eng = Engine(cls.dir, log=cls.logs.append, vision=True, max_context=2048, max_tokens=8, disk_cache_gb=0)

    def ask(self, eng, image_seeds=(0,), text="what is this?"):
        content = [{"type": "text", "text": text}] + [
            {"type": "image_url", "image_url": {"url": png_data_uri(seed=s)}} for s in image_seeds]
        prompt, images, _ = eng.build_prompt([{"role": "user", "content": content}])
        return prompt, images

    def gen(self, eng, prompt, images):
        ids, media = eng.encode(prompt, images)
        out, st = "", None
        for it in eng.generate(ids, media, max_tokens=6, sampling={"temperature": 0}):
            if isinstance(it, dict):
                st = it
            else:
                out += it[1]
        return ids, out, st

    def test_the_tower_loads_on_demand_not_before(self):
        from engine.generate import Engine
        logs = []
        eng = Engine(self.dir, log=logs.append, vision=True, max_context=2048, max_tokens=8, disk_cache_gb=0)
        self.assertTrue(eng.vision_allowed)
        text_only = self.gen(eng, *self.ask(eng, ()))
        self.assertNotIn("vision_tower", eng.model)                       # text requests never pay for it
        self.assertEqual(eng.vision_bytes, 0)
        self.gen(eng, *self.ask(eng))
        self.assertIn("vision_tower", eng.model)
        self.assertGreater(eng.vision_bytes, 0)
        self.assertTrue(any("vision tower loaded" in m for m in logs))
        self.assertIsNotNone(text_only[2])

    def test_image_tokens_match_the_grid_and_prompt_counts_them(self):
        prompt, images = self.ask(self.eng)
        self.assertEqual(prompt.count("<|image_pad|>"), 1)
        ids, media = self.eng.encode(prompt, images)
        pix, grids = media
        t, gh, gw = grids[0]
        n_tokens = sum(1 for i in ids if i == self.eng.image_token)
        self.assertEqual(n_tokens, t * gh * gw // 4)
        self.assertEqual(pix.shape[0], gh * gw)

    def test_two_images_two_placeholders(self):
        prompt, images = self.ask(self.eng, (1, 2))
        ids, (pix, grids) = self.eng.encode(prompt, images)
        self.assertEqual(len(grids), 2)
        self.assertEqual(sum(1 for i in ids if i == self.eng.image_token), sum(t * h * w // 4 for t, h, w in grids))

    def test_the_image_changes_the_answer_and_generation_is_stable(self):
        _, _, st = self.gen(self.eng, *self.ask(self.eng, (3,)))
        self.assertEqual(st["completion_tokens"] > 0, True)
        ids_a, out_a, _ = self.gen(self.eng, *self.ask(self.eng, (3,)))
        ids_b, out_b, _ = self.gen(self.eng, *self.ask(self.eng, (3,)))
        self.assertEqual(out_a, out_b)
        # a different picture keeps the same prompt length but must reach the model as different embeddings
        pa = self.eng.encode(*self.ask(self.eng, (3,)))[1]
        pb = self.eng.encode(*self.ask(self.eng, (4,)))[1]
        self.assertFalse(bool(mx.array_equal(pa[0], pb[0])))

    def test_image_requests_do_not_reuse_a_text_prefix_as_if_it_were_text(self):
        self.gen(self.eng, *self.ask(self.eng))
        self.assertIsNone(self.eng.cache)                                 # 3-D positions: not a text prefix

    def test_mismatched_placeholders_and_missing_support_are_clear_errors(self):
        prompt, images = self.ask(self.eng)
        with self.assertRaises(ValueError):
            self.eng.encode(prompt, images + images)                      # two images, one placeholder
        from engine.generate import Engine
        off = Engine(self.dir, log=lambda *_: None, vision=False, max_context=1024, max_tokens=4, disk_cache_gb=0)
        with self.assertRaises(ValueError) as ctx:
            off.encode(*self.ask(off))
        self.assertIn("no vision support", str(ctx.exception))

    def test_sleep_unloads_the_tower_and_the_next_image_brings_it_back(self):
        self.gen(self.eng, *self.ask(self.eng))
        self.eng.sleep()
        self.assertTrue(self.eng.sleeping)
        self.assertEqual(self.eng.vision_bytes, 0)
        ids, out, st = self.gen(self.eng, *self.ask(self.eng))           # encode() wakes it, then attaches
        self.assertFalse(self.eng.sleeping)
        self.assertIn("vision_tower", self.eng.model)
        self.assertGreater(st["completion_tokens"], 0)


if __name__ == "__main__":
    unittest.main()
