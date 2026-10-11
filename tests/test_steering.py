"""Activation steering (engine/steering.py): an edit shipped next to a base model — a direction and a strength."""
import json
import os
import tempfile
import unittest

import mlx.core as mx
import numpy as np

from engine import steering
from tests.synth import HIDDEN, make_bundle


def unit(seed=0, n=HIDDEN):
    v = np.random.default_rng(seed).standard_normal(n).astype(np.float32)
    return v / np.linalg.norm(v)


def write_direction(folder, v, name="refusal_dir.safetensors"):
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, name)
    mx.save_safetensors(path, {"direction": mx.array(v)})
    return path


class TestProjector(unittest.TestCase):
    def setUp(self):
        self.d = mx.array(unit(1))
        self.y = mx.array(np.random.default_rng(2).standard_normal((2, 5, HIDDEN)).astype(np.float32)).astype(mx.float16)

    def component(self, y):
        return (y.astype(mx.float32) * self.d).sum(axis=-1)

    def test_alpha_one_removes_the_component_and_keeps_the_rest(self):
        out = steering.projector(self.d, 1.0)(self.y)
        self.assertEqual(out.dtype, mx.float16)
        self.assertLess(float(mx.abs(self.component(out)).max()), 0.02)       # fp16 rounding only
        rest = self.y.astype(mx.float32) - self.component(self.y)[..., None] * self.d
        self.assertLess(float(mx.abs(out.astype(mx.float32) - rest).max()), 0.02)

    def test_alpha_zero_is_the_unmodified_model_and_two_overshoots(self):
        same = steering.projector(self.d, 0.0)(self.y)
        self.assertTrue(bool(mx.all(same == self.y)))
        flipped = steering.projector(self.d, 2.0)(self.y)
        np.testing.assert_allclose(np.asarray(self.component(flipped)), -np.asarray(self.component(self.y)),
                                   atol=0.05)


class TestLoading(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.v = unit(3) * 7.5                     # not unit length on purpose

    def test_every_format_normalises_to_a_unit_vector(self):
        st = write_direction(self.dir, self.v)
        npy = os.path.join(self.dir, "d.npy")
        np.save(npy, self.v)
        binf = os.path.join(self.dir, "d.bin")
        self.v.astype("<f4").tofile(binf)
        for p in (st, npy, binf):
            d = steering.load_direction(p, HIDDEN)
            self.assertAlmostEqual(float(mx.linalg.norm(d)), 1.0, places=5)
            np.testing.assert_allclose(np.asarray(d), self.v / np.linalg.norm(self.v), atol=1e-5)

    def test_wrong_size_zero_nan_and_unknown_extension_are_refused(self):
        st = write_direction(self.dir, unit(4, HIDDEN * 2), "big.safetensors")
        with self.assertRaisesRegex(steering.SteerError, "different model"):
            steering.load_direction(st, HIDDEN)
        with self.assertRaises(steering.SteerError):
            steering.load_direction(write_direction(self.dir, np.zeros(HIDDEN, np.float32), "z.safetensors"), HIDDEN)
        bad = np.full(HIDDEN, np.nan, np.float32)
        with self.assertRaises(steering.SteerError):
            steering.load_direction(write_direction(self.dir, bad, "n.safetensors"), HIDDEN)
        with self.assertRaises(steering.SteerError):
            steering._read_vector(os.path.join(self.dir, "x.txt"))

    def test_safetensors_picks_the_tensor_named_direction_or_the_only_one(self):
        v = unit(8)
        other = np.ones(HIDDEN, np.float32)
        both = os.path.join(self.dir, "both.safetensors")
        mx.save_safetensors(both, {"alpha": mx.array(other), "direction": mx.array(v)})
        np.testing.assert_allclose(np.asarray(steering.load_direction(both, HIDDEN)), v, atol=1e-6)   # not "alpha"
        only = os.path.join(self.dir, "only.safetensors")
        mx.save_safetensors(only, {"refusal": mx.array(v)})
        np.testing.assert_allclose(np.asarray(steering.load_direction(only, HIDDEN)), v, atol=1e-6)  # a lone tensor
        many = os.path.join(self.dir, "many.safetensors")
        mx.save_safetensors(many, {"a": mx.array(v), "b": mx.array(other)})
        with self.assertRaisesRegex(steering.SteerError, "expected one tensor named"):                # ambiguous
            steering.load_direction(many, HIDDEN)

    def test_a_folder_gives_its_direction_and_the_recommended_alpha(self):
        write_direction(self.dir, self.v)
        os.makedirs(os.path.join(self.dir, "runtime-adapter"))
        with open(os.path.join(self.dir, "runtime-adapter", "adapter.json"), "w") as f:
            json.dump({"alpha": 1.5, "modules": []}, f)
        path, alpha = steering.resolve(self.dir)
        self.assertTrue(path.endswith("refusal_dir.safetensors"))
        self.assertEqual(alpha, 1.5)
        with self.assertRaises(steering.SteerError):
            steering.resolve(tempfile.mkdtemp())                              # an empty folder
        self.assertEqual(steering.resolve(path), (path, None))               # a file is used as it is


class TestInModel(unittest.TestCase):
    """Through the real loader and decoder on a synthetic bundle (random weights: only the plumbing counts)."""

    @classmethod
    def setUpClass(cls):
        from engine.loader import load
        cls.load = staticmethod(load)
        cls.bundle = tempfile.mkdtemp()
        make_bundle(cls.bundle, seed=11)
        cls.dir = tempfile.mkdtemp()
        cls.file = write_direction(cls.dir, unit(5))
        cls.ids = mx.array([[3, 7, 11, 5, 9, 2, 8]])

    def logits(self, **kw):
        model, _cfg, _n = self.load(self.bundle, vision=False, log=lambda *a: None, **kw)
        out = model(self.ids, model.make_cache(), last_only=False)
        mx.eval(out)
        return model, np.asarray(out.astype(mx.float32))

    def test_off_zero_and_on_behave_as_they_should(self):
        _, base = self.logits()
        _, zero = self.logits(steer=(self.file, 0.0))
        m, on = self.logits(steer=(self.file, 1.0))
        np.testing.assert_allclose(zero, base, atol=1e-3)                     # alpha 0 is the same model
        self.assertGreater(float(np.abs(on - base).max()), 1e-2)              # alpha 1 changes it
        # the token embedding reaching the first layer has no component along the direction
        d = steering.load_direction(self.file, HIDDEN)
        emb = m.embed(mx.array([[1, 2, 3]]))
        self.assertLess(float(mx.abs((emb.astype(mx.float32) * d).sum(axis=-1)).max()), 0.02)

    def test_a_direction_for_another_model_stops_the_load(self):
        wrong = write_direction(self.dir, unit(6, HIDDEN + 64), "wrong.safetensors")
        with self.assertRaisesRegex(steering.SteerError, "hidden size"):
            self.load(self.bundle, vision=False, log=lambda *a: None, steer=(wrong, None))


if __name__ == "__main__":
    unittest.main()
