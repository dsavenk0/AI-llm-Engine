"""Batched decoding (engine/batch.py): several requests in one pass must match the single-sequence result.

    .venv/bin/python -m unittest tests.test_batch -v

Uses a tiny synthetic attention-only native bundle (full_attention_interval=1 -> no SSM layers).
"""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx  # noqa: E402

from engine import batch  # noqa: E402
from engine.model import Positions  # noqa: E402
from tests.synth import make_bundle  # noqa: E402


def greedy_single(model, ids, max_new, eos):
    """The reference: one request, its own cache, plain greedy — what batching must reproduce exactly."""
    cache = model.make_cache()
    logits = model(mx.array([list(ids)]), cache, Positions(delta=0))
    tok = int(mx.argmax(logits[0, -1]).item())
    out = []
    for _ in range(max_new):
        if tok in eos:
            break
        out.append(tok)
        logits = model(mx.array([[tok]]), cache, Positions(delta=0))
        tok = int(mx.argmax(logits[0, -1]).item())
    return out


class TestBatchedDecode(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from engine.generate import Engine
        cls.dir = tempfile.mkdtemp()
        make_bundle(cls.dir, seed=5, attn_interval=1)       # attention-only: every layer is full attention
        cls.eng = Engine(cls.dir, vision=False, max_context=1024, max_tokens=64, log=lambda *_: None)
        cls.model = cls.eng.model
        cls.eos = set(cls.eng.eos)

    def test_batched_matches_single_for_each_request(self):
        prompts = [[20, 21, 22, 23, 24, 25], [30, 31, 32], [40, 41, 42, 43, 44]]   # ragged on purpose
        max_new = 10
        want = [greedy_single(self.model, p, max_new, self.eos) for p in prompts]
        got = batch.batched_generate(self.model, prompts, max_new=max_new, eos=self.eos)
        self.assertEqual(got, want)

    def test_order_is_preserved_and_streaming_fires_per_row(self):
        prompts = [[12, 13, 14], [50, 51, 52, 53]]
        seen = []
        got = batch.batched_generate(self.model, prompts, max_new=6, eos=self.eos,
                                     on_token=lambda i, t: seen.append((i, t)))
        # every streamed token belongs to its row, in that row's order
        for i, _ in enumerate(prompts):
            streamed = [t for (r, t) in seen if r == i]
            self.assertEqual(streamed, got[i])

    def test_single_request_batch_equals_the_reference(self):
        p = [7, 8, 9, 10]
        want = greedy_single(self.model, p, 8, self.eos)
        got = batch.batched_generate(self.model, [p], max_new=8, eos=self.eos)
        self.assertEqual(got, [want])

    def test_an_ssm_model_is_refused_clearly(self):
        other = tempfile.mkdtemp()
        make_bundle(other, seed=6, attn_interval=2)         # every 2nd layer is a GatedDeltaNet (SSM)
        from engine.generate import Engine
        eng = Engine(other, vision=False, max_context=512, max_tokens=32, log=lambda *_: None)
        with self.assertRaises(batch.Unbatchable):
            batch.batched_generate(eng.model, [[1, 2, 3], [4, 5, 6]], max_new=4, eos=set())


if __name__ == "__main__":
    unittest.main()
