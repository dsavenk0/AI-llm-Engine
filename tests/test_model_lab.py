"""Model Lab, first increment: tensor inspection (model_lab.inspection) and activation capture
(model_lab.capture). No weight editing yet — see docs/model-lab/compatibility.md.

    .venv/bin/python -m unittest tests.test_model_lab -v
"""
import os
import tempfile
import unittest

import mlx.core as mx
import numpy as np

from model_lab import ablation, capture, export, inspection, patch
from tests import gguf_synth as G
from tests.synth import HIDDEN, LAYERS, make_bundle


class TestInspection(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()

    def test_safetensors_tensors_are_named_shaped_and_sized(self):
        make_bundle(self.dir, seed=1)
        tensors = inspection.inspect_tensors(self.dir, "native")
        name = "language_model.model.layers.0.mlp.down_proj.weight"
        self.assertIn(name, tensors)
        info = tensors[name]
        self.assertEqual(len(info["shape"]), 2)
        self.assertIsNotNone(info["dtype"])
        self.assertIsNotNone(info["nbytes"])
        count, total = inspection.summarize(tensors)
        self.assertEqual(count, len(tensors))
        self.assertEqual(total, sum(t["nbytes"] for t in tensors.values()))

    def test_no_safetensors_files_is_a_clear_error(self):
        with self.assertRaises(ValueError):
            inspection.inspect_tensors(self.dir, "native")

    def test_gguf_tensors_match_what_was_written(self):
        path = os.path.join(self.dir, "llama-Q8_0.gguf")
        G.write(path, seed=3, quant="Q8_0", arch="llama", moe=False)
        tensors = inspection.inspect_tensors(path, "gguf")
        self.assertGreater(len(tensors), 0)
        for t in tensors.values():
            self.assertIsInstance(t["shape"], list)
            self.assertGreater(t["nbytes"], 0)
        self.assertEqual(tensors["token_embd.weight"]["dtype"], "Q8_0")   # matrices: quantized
        self.assertEqual(tensors["blk.0.attn_norm.weight"]["dtype"], "F32")  # norm vectors: always F32


class TestCapture(unittest.TestCase):
    """Correctness is the point here, not coverage: a capture hook that silently records the wrong
    thing (see model_lab/capture.py's module docstring on why `set_steer`-style injection would) is
    worse than no hook at all."""

    @classmethod
    def setUpClass(cls):
        from engine.loader import load
        cls.bundle = tempfile.mkdtemp()
        make_bundle(cls.bundle, seed=7)
        cls.model, _cfg, _n = load(cls.bundle, vision=False, log=lambda *a: None)
        cls.ids = mx.array([[3, 7, 11, 5, 9, 2, 8]])

    def test_captures_every_layer_by_default_with_the_right_shapes(self):
        captured, final_hidden = capture.capture_activations(self.model, self.ids)
        self.assertEqual(sorted(captured), list(range(LAYERS)))
        B, L = self.ids.shape
        for layer in captured.values():
            for key in ("pre_residual", "post_attn_residual", "post_mlp_residual"):
                self.assertEqual(layer[key].shape, (B, L, HIDDEN))
                self.assertEqual(layer[key].dtype, np.float32)
        self.assertEqual(final_hidden.shape, (B, L, HIDDEN))

    def test_layers_filter_returns_only_what_was_asked(self):
        captured, _ = capture.capture_activations(self.model, self.ids, layers=[0, 2])
        self.assertEqual(sorted(captured), [0, 2])

    def test_an_out_of_range_layer_is_refused_clearly(self):
        with self.assertRaisesRegex(ValueError, "no such layer"):
            capture.capture_activations(self.model, self.ids, layers=[LAYERS + 5])

    def test_residual_stream_is_continuous_across_layers(self):
        """Layer N's post_mlp_residual is literally layer N+1's pre_residual — the same tensor the
        real Model.__call__ passes along; if the capture orchestration ever diverged from
        DecoderLayer.__call__'s real sequence, this would be the first thing to break."""
        captured, _ = capture.capture_activations(self.model, self.ids)
        for i in range(LAYERS - 1):
            np.testing.assert_array_equal(captured[i]["post_mlp_residual"], captured[i + 1]["pre_residual"])

    def test_capturing_does_not_change_the_models_own_output(self):
        """The whole point of not reusing `set_steer` for this (see capture.py's docstring): capture
        must be read-only. Confirmed against a real forward pass, not just "it didn't crash"."""
        captured, final_hidden = capture.capture_activations(self.model, self.ids)
        last_captured = final_hidden[:, -1:, :]
        real = self.model(self.ids, self.model.make_cache(), hidden=True)
        mx.eval(real)
        real = np.asarray(real.astype(mx.float32))
        np.testing.assert_allclose(last_captured, real, atol=1e-4)

    def test_save_capture_writes_arrays_and_meta(self):
        captured, final_hidden = capture.capture_activations(self.model, self.ids, layers=[0])
        root = tempfile.mkdtemp()
        exp_id, out_dir = capture.save_capture(root, "test-model", "hello", captured, final_hidden)
        self.assertTrue(os.path.isfile(os.path.join(out_dir, "layer_000.npz")))
        self.assertTrue(os.path.isfile(os.path.join(out_dir, "meta.json")))
        loaded = np.load(os.path.join(out_dir, "layer_000.npz"))
        np.testing.assert_array_equal(loaded["pre_residual"], captured[0]["pre_residual"])


class TestPatchNative(unittest.TestCase):
    """Reversible weight edits on the native (safetensors) route. The point under test: a patch really
    changes output, and a *fresh reload* (never-patched — the checkpoint file is never touched) always
    reproduces the exact pre-patch output, which is what "rollback" actually relies on."""

    @classmethod
    def setUpClass(cls):
        from engine.loader import load
        cls.load = staticmethod(load)
        cls.bundle = tempfile.mkdtemp()
        make_bundle(cls.bundle, seed=9)      # LAYERS=4, attn_interval=4: layer 3 is attention, 0-2 are GDN
        cls.ids = mx.array([[3, 7, 11, 5, 9, 2, 8]])

    def logits(self, model):
        out = model(self.ids, model.make_cache())
        mx.eval(out)
        return np.asarray(out.astype(mx.float32))

    def test_a_patch_on_a_non_fused_tensor_changes_output_and_reload_restores_it_exactly(self):
        name = "language_model.model.layers.0.mlp.down_proj.weight"
        model, _, _ = self.load(self.bundle, vision=False, log=lambda *a: None)
        before = self.logits(model)
        leaf = patch.locate(model, "native", name)
        base, _spec = patch.read_logical(leaf)
        delta = np.random.default_rng(0).standard_normal(base.shape).astype(np.float32) * 0.05
        patch.apply_delta(model, "native", name, delta)
        after = self.logits(model)
        self.assertGreater(float(np.abs(after - before).max()), 1e-3)
        fresh, _, _ = self.load(self.bundle, vision=False, log=lambda *a: None)
        np.testing.assert_array_equal(self.logits(fresh), before)

    def test_fused_attention_and_gdn_tensors_resolve_by_their_fused_name(self):
        model, _, _ = self.load(self.bundle, vision=False, log=lambda *a: None)
        qkv = patch.locate(model, "native", "language_model.model.layers.3.self_attn.qkv_proj.weight")
        gdn = patch.locate(model, "native", "language_model.model.layers.0.linear_attn.in_proj_qkvz.weight")
        self.assertEqual(qkv.bits, 4)
        self.assertEqual(gdn.bits, 4)

    def test_an_unfused_per_head_name_is_refused_not_guessed_at(self):
        model, _, _ = self.load(self.bundle, vision=False, log=lambda *a: None)
        with self.assertRaisesRegex(patch.PatchError, "fuses"):
            patch.locate(model, "native", "language_model.model.layers.3.self_attn.q_proj.weight")

    def test_a_wrongly_shaped_delta_is_refused(self):
        model, _, _ = self.load(self.bundle, vision=False, log=lambda *a: None)
        with self.assertRaises(patch.PatchError):
            patch.apply_delta(model, "native", "language_model.model.layers.0.mlp.down_proj.weight",
                              np.zeros((3, 3), np.float32))


class TestPatchGGUF(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()

    def test_a_patch_on_a_dense_layer_changes_output_and_reload_restores_it_exactly(self):
        from engine.dense import build
        path = os.path.join(self.dir, "llama-Q8_0.gguf")
        G.write(path, seed=5, quant="Q8_0", arch="llama", moe=False)
        model, _cfg, _gf = build(path)
        ids = mx.array([[1, 2, 3, 4, 5]])

        def logits(m):
            out = m(ids, m.make_cache())
            mx.eval(out)
            return np.asarray(out.astype(mx.float32))

        before = logits(model)
        name = "blk.0.ffn_down.weight"
        leaf = patch.locate(model, "gguf", name)
        base, _spec = patch.read_logical(leaf)
        delta = np.random.default_rng(1).standard_normal(base.shape).astype(np.float32) * 0.05
        patch.apply_delta(model, "gguf", name, delta)
        after = logits(model)
        self.assertGreater(float(np.abs(after - before).max()), 1e-3)
        fresh, _, _ = build(path)
        np.testing.assert_array_equal(logits(fresh), before)

    def test_moe_experts_are_refused_not_silently_skipped(self):
        from engine.dense import build
        path = os.path.join(self.dir, "llama-moe.gguf")
        G.write(path, seed=5, quant="Q8_0", arch="llama", moe=True)
        model, _, _ = build(path)
        with self.assertRaisesRegex(patch.PatchError, "MoE"):
            patch.locate(model, "gguf", "blk.0.ffn_down.weight")

    def test_a_tied_output_embedding_is_refused_not_silently_disconnected(self):
        """Gemma ties the output projection to the input embedding (dense.py: `self.head =
        tensors.get("output.weight") or self.embed.as_linear()`) — patching "output.weight" as if it
        were its own tensor would silently edit a disconnected copy that the model never reads from."""
        from engine.dense import build
        path = os.path.join(self.dir, "gemma3.gguf")
        G.write(path, seed=2, quant="Q8_0", arch="gemma3")
        model, _, _ = build(path)
        with self.assertRaisesRegex(patch.PatchError, "tied"):
            patch.locate(model, "gguf", "output.weight")


class TestAblation(unittest.TestCase):
    def test_mean_diff_direction_is_unit_and_points_from_negative_to_positive(self):
        rng = np.random.default_rng(0)
        base = rng.standard_normal(16).astype(np.float32)
        negative = rng.standard_normal((5, 16)).astype(np.float32) * 0.1
        positive = negative + base          # shifted by `base` in every row
        d = ablation.mean_diff_direction(positive, negative)
        self.assertAlmostEqual(float(np.linalg.norm(d)), 1.0, places=5)
        cos = float(d @ (base / np.linalg.norm(base)))
        self.assertGreater(cos, 0.99)        # points the same way as the true shift

    def test_identical_means_refuse_rather_than_divide_by_zero(self):
        same = np.ones((3, 8), np.float32)
        with self.assertRaises(ValueError):
            ablation.mean_diff_direction(same, same.copy())

    def test_projection_delta_zeroes_the_direction_component_for_every_input(self):
        """W' = W - alpha·d(d^T W): at alpha=1, d^T W' is exactly zero, which means y'=W'x has zero
        component along `d` for *any* x — the actual guarantee "ablation" is supposed to give."""
        rng = np.random.default_rng(1)
        hidden, cols = 32, 20
        base = rng.standard_normal((hidden, cols)).astype(np.float32)
        d = rng.standard_normal(hidden).astype(np.float32)
        d /= np.linalg.norm(d)
        delta = ablation.projection_delta(base, d, alpha=1.0)
        after = base + delta
        np.testing.assert_allclose(d @ after, 0, atol=1e-4)
        self.assertGreater(float(np.abs(d @ base).max()), 1e-2)   # it actually had a component before

    def test_projection_delta_rejects_a_direction_of_the_wrong_length(self):
        with self.assertRaises(ValueError):
            ablation.projection_delta(np.zeros((4, 4), np.float32), np.zeros(5, np.float32))

    def test_ablation_deltas_zero_the_direction_on_the_real_patched_tensor(self):
        """End to end on the synthetic native bundle: compute deltas from two prompt groups, apply them,
        and confirm the *actual* patched down_proj at each layer satisfies the same zero-component
        guarantee the unit-tested math promises in isolation."""
        from engine.loader import load
        from engine.weightpatch import locate, read_logical

        from tests.synth import uniform
        bundle = tempfile.mkdtemp()
        # down_proj stored plain (not quantized): isolates the ablation wiring from 4-bit quantization
        # noise, which otherwise dominates a projection this small on random weights
        scheme = lambda name: None if name == "mlp.down_proj" else uniform()(name)
        make_bundle(bundle, seed=13, scheme=scheme)
        model, _, _ = load(bundle, vision=False, log=lambda *a: None)
        safe_ids = [mx.array([[1, 2, 3, 4]]), mx.array([[5, 6, 7, 8, 9]])]
        harmful_ids = [mx.array([[10, 11, 12]]), mx.array([[13, 14, 15, 16]])]
        layers = [0, 3]
        deltas = ablation.ablation_deltas(model, safe_ids, harmful_ids, layers, alpha=1.0)
        self.assertEqual(sorted(deltas), sorted(f"language_model.model.layers.{i}.mlp.down_proj.weight"
                                                 for i in layers))
        # snapshot every layer's direction/base *before* patching any of them — patching layer 0 changes
        # what a forward pass computes at layer 3 (its activations run through layer 0 first), so
        # recomputing a later layer's direction after an earlier one is already patched would compare
        # against the wrong, post-patch direction instead of the one the delta was actually built from
        before_state = {}
        for name in deltas:
            i = int(name.split(".")[3])
            leaf = locate(model, "native", name)
            base, spec = read_logical(leaf)
            direction = ablation.mean_diff_direction(
                ablation.collect_last_token_residual(model, harmful_ids, i),
                ablation.collect_last_token_residual(model, safe_ids, i))
            before_state[name] = (leaf, base, spec, direction)
            self.assertGreater(float(np.abs(direction @ base).max()), 1e-4)   # it had a component before
        from engine.weightpatch import write_logical
        for name, delta in deltas.items():
            leaf, base, spec, direction = before_state[name]
            write_logical(leaf, base + delta, spec)
            patched, _ = read_logical(leaf)
            self.assertLess(float(np.abs(direction @ patched).max()), 1e-4)


class TestExport(unittest.TestCase):
    """Baking a patch into a standalone model folder (model_lab/export.py) — the alternative to activating
    it at load time. Caught by this test the first time it ran, for real reasons worth keeping in mind:
    fused quantized tensors (qkv_proj, gate_up_proj, in_proj_qkvz/ba) must be split back to their
    checkpoint's original per-part names or the reload misjudges them as unquantized (wrong shape, crash);
    and `jang_norms_pre_shifted` has an inverted-sounding meaning in engine/loader.py (`false` means "not
    shifted yet, shift it now", not "no shift needed") — getting it backwards silently double-shifted
    every RMSNorm weight by +1."""

    def setUp(self):
        self.bundle = tempfile.mkdtemp()
        make_bundle(self.bundle, seed=17)     # default scheme quantizes everything, incl. in_proj_b/a
        self.out = tempfile.mkdtemp()
        os.rmdir(self.out)        # export_model creates it; must not exist with content beforehand

    def test_export_round_trips_bit_exact_with_no_patch(self):
        from engine.loader import load
        model, _, _ = load(self.bundle, vision=False, log=lambda *a: None)
        export.export_model(model, self.bundle, self.out)
        reloaded, _, _ = load(self.out, vision=False, log=lambda *a: None)
        ids = mx.array([[3, 7, 11, 5, 9, 2, 8]])
        a = np.asarray(model(ids, model.make_cache()).astype(mx.float32))
        mx.eval(model(ids, model.make_cache()))
        b = np.asarray(reloaded(ids, reloaded.make_cache()).astype(mx.float32))
        np.testing.assert_array_equal(a, b)

    def test_export_round_trips_bit_exact_with_a_patch_applied(self):
        from engine.loader import load
        from engine.weightpatch import apply_delta
        model, _, _ = load(self.bundle, vision=False, log=lambda *a: None)
        name = "language_model.model.layers.0.mlp.down_proj.weight"
        leaf = patch.locate(model, "native", name)
        base, _ = patch.read_logical(leaf)
        delta = np.random.default_rng(2).standard_normal(base.shape).astype(np.float32) * 0.05
        apply_delta(model, "native", name, delta)
        export.export_model(model, self.bundle, self.out)
        reloaded, _, _ = load(self.out, vision=False, log=lambda *a: None)
        ids = mx.array([[3, 7, 11, 5, 9, 2, 8]])
        a = np.asarray(model(ids, model.make_cache()).astype(mx.float32))
        b = np.asarray(reloaded(ids, reloaded.make_cache()).astype(mx.float32))
        np.testing.assert_array_equal(a, b)

    def test_refuses_to_export_into_a_nonempty_folder(self):
        from engine.loader import load
        model, _, _ = load(self.bundle, vision=False, log=lambda *a: None)
        os.makedirs(self.out, exist_ok=True)
        with open(os.path.join(self.out, "x"), "w") as f:
            f.write("x")
        with self.assertRaises(ValueError):
            export.export_model(model, self.bundle, self.out)

    def test_refuses_to_export_onto_the_source_folder(self):
        from engine.loader import load
        model, _, _ = load(self.bundle, vision=False, log=lambda *a: None)
        with self.assertRaises(ValueError):
            export.export_model(model, self.bundle, self.bundle)


class TestPatchArtifacts(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()

    def test_create_list_and_load_round_trip(self):
        delta = np.ones((4, 4), np.float32) * 0.1
        exp_id, out_dir = patch.create_patch(self.root, "demo-model", "native", {"a.weight": delta}, "note!")
        self.assertTrue(os.path.isfile(os.path.join(out_dir, "manifest.json")))
        self.assertTrue(os.path.isfile(os.path.join(out_dir, "delta.safetensors")))
        manifest, loaded_deltas, _ = patch.load_patch(self.root, exp_id)
        self.assertEqual(manifest["tensors"], ["a.weight"])
        self.assertEqual(manifest["note"], "note!")
        np.testing.assert_array_equal(loaded_deltas["a.weight"], delta)
        patches = patch.list_patches(self.root, "demo-model")
        self.assertEqual(len(patches), 1)
        self.assertEqual(patch.list_patches(self.root, "some-other-model"), [])

    def test_a_patch_can_cover_more_than_one_tensor(self):
        deltas = {"a.weight": np.ones((2, 2), np.float32), "b.weight": np.ones((3, 3), np.float32) * 2}
        exp_id, _ = patch.create_patch(self.root, "m", "native", deltas, "multi")
        manifest, loaded, _ = patch.load_patch(self.root, exp_id)
        self.assertEqual(sorted(manifest["tensors"]), ["a.weight", "b.weight"])
        np.testing.assert_array_equal(loaded["b.weight"], deltas["b.weight"])

    def test_evaluation_round_trip(self):
        exp_id, _ = patch.create_patch(self.root, "m", "native", {"a.weight": np.zeros((2, 2), np.float32)})
        self.assertIsNone(patch.load_evaluation(self.root, exp_id))
        patch.save_evaluation(self.root, exp_id, {"prompt": "hi", "baseline": [1], "patched": [2]})
        self.assertEqual(patch.load_evaluation(self.root, exp_id)["prompt"], "hi")


if __name__ == "__main__":
    unittest.main()
