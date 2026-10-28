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


def single_next_logits(model, ids):
    """Prefill `ids` alone, feed the greedy first token, return (next-step logits row, that first token)."""
    cache = model.make_cache()
    logits = model(mx.array([list(ids)]), cache, Positions(delta=0))
    tok = int(mx.argmax(logits[0, -1]).item())
    logits2 = model(mx.array([[tok]]), cache, Positions(delta=0))
    return logits2[0, -1], tok


def batched_next_logits(model, prompts):
    """The first batched decode step, built exactly as engine/batch.py does, returning [B, V] logits."""
    per, first = [], []
    for p in prompts:
        c = model.make_cache()
        lg = model(mx.array([list(p)]), c, Positions(delta=0))
        per.append(c)
        first.append(int(mx.argmax(lg[0, -1]).item()))
    lens = [len(p) for p in prompts]
    lmax = max(lens)
    cache = [batch._assemble(per, i, lmax) for i in range(len(per[0]))]
    pad = mx.array([lmax - n for n in lens])
    delta = mx.array([n - lmax for n in lens])
    lk = cache[0].offset + 1
    cols = mx.arange(lk)
    mask = mx.where(cols[None, None, None, :] < pad[:, None, None, None], batch.NEG, 0.0).astype(mx.float16)
    logits = model(mx.array([[t] for t in first]), cache, Positions(delta=delta), mask=mask)
    return logits[:, -1, :], first


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

    def test_per_request_sampler_is_used(self):
        # row 0 greedy, row 1 forced to always emit token 77 — each row follows its own sampler
        prompts = [[20, 21, 22], [30, 31, 32]]
        samplers = [batch._greedy, lambda _row: 77]
        got = batch.batched_generate(self.model, prompts, max_new=5, eos=set(), samplers=samplers)
        self.assertEqual(got[1], [77, 77, 77, 77, 77])
        self.assertNotEqual(got[0], [77, 77, 77, 77, 77])

    def test_should_stop_ends_a_row_early(self):
        prompts = [[20, 21, 22], [30, 31, 32]]
        calls = {"n": 0}

        def stop_row1():
            calls["n"] += 1
            return calls["n"] > 2                      # let a couple of tokens through, then stop row 1
        got = batch.batched_generate(self.model, prompts, max_new=10, eos=set(),
                                     should_stop=[None, stop_row1])
        self.assertLess(len(got[1]), len(got[0]))
        self.assertGreaterEqual(len(got[1]), 1)

    def test_single_request_batch_equals_the_reference(self):
        p = [7, 8, 9, 10]
        want = greedy_single(self.model, p, 8, self.eos)
        got = batch.batched_generate(self.model, [p], max_new=8, eos=self.eos)
        self.assertEqual(got, [want])

    def test_batched_forward_matches_single_for_a_hybrid_ssm_model(self):
        # A hybrid model (GatedDeltaNet/SSM layers): the batched step must reproduce each request's
        # single-sequence logits. Greedy tokens can diverge on a random model when fp16 noise flips a
        # near-tie, so this checks the logits agree within fp16 — the real losslessness guarantee.
        other = tempfile.mkdtemp()
        make_bundle(other, seed=6, attn_interval=2)         # every 2nd layer is a GatedDeltaNet (SSM)
        from engine.generate import Engine
        eng = Engine(other, vision=False, max_context=512, max_tokens=32, log=lambda *_: None)
        prompts = [[20, 21, 22, 23, 24], [30, 31, 32], [40, 41, 42, 43]]
        bl, _ = batched_next_logits(eng.model, prompts)
        for i, p in enumerate(prompts):
            sl, _ = single_next_logits(eng.model, p)
            self.assertEqual(int(mx.argmax(bl[i]).item()), int(mx.argmax(sl).item()))
            self.assertLess(mx.max(mx.abs(bl[i].astype(mx.float32) - sl.astype(mx.float32))).item(), 0.2)

    def test_batched_forward_matches_single_for_attention_logits(self):
        # the attention-only path is bit-exact; here we also assert the logits agree very tightly
        prompts = [[20, 21, 22, 23, 24, 25], [30, 31, 32], [40, 41, 42, 43, 44]]
        bl, _ = batched_next_logits(self.model, prompts)
        for i, p in enumerate(prompts):
            sl, _ = single_next_logits(self.model, p)
            self.assertLess(mx.max(mx.abs(bl[i].astype(mx.float32) - sl.astype(mx.float32))).item(), 0.05)


def serial(eng, ids, max_tokens, stop=None):
    """The reference: the engine's ordinary one-request path, with speculation and the prefix cache off so
    each run is independent — exactly what a batch row does."""
    eng.speculative = False
    eng.cache = None
    pieces = []
    for item in eng.generate(ids, max_tokens=max_tokens, sampling={"temperature": 0}, stop=stop):
        if isinstance(item, dict):
            return "".join(pieces), item
        pieces.append(item[1])
    return "".join(pieces), {}


class TestEngineGenerateBatch(unittest.TestCase):
    """Engine.generate_batch must reproduce the ordinary path's text, token count and finish reason."""

    @classmethod
    def setUpClass(cls):
        from engine.generate import Engine
        cls.dir = tempfile.mkdtemp()
        make_bundle(cls.dir, seed=5, attn_interval=1)
        cls.eng = Engine(cls.dir, vision=False, max_context=1024, max_tokens=64, log=lambda *_: None)

    def run_batch(self, reqs):
        got = {i: [] for i in range(len(reqs))}
        stats = self.eng.generate_batch(reqs, on_piece=lambda i, t: got[i].append(t))
        return ["".join(got[i]) for i in range(len(reqs))], stats

    def test_matches_the_single_request_path(self):
        prompts = [[20, 21, 22, 23, 24, 25], [30, 31, 32], [40, 41, 42, 43, 44]]   # ragged on purpose
        want = [serial(self.eng, p, 10) for p in prompts]
        texts, stats = self.run_batch([{"ids": p, "max_tokens": 10, "sampling": {"temperature": 0}}
                                       for p in prompts])
        for i, (wtext, wstat) in enumerate(want):
            self.assertEqual(texts[i], wtext, f"row {i}")
            self.assertEqual(stats[i]["completion_tokens"], wstat["completion_tokens"], f"row {i}")
            self.assertEqual(stats[i]["finish_reason"], wstat["finish_reason"], f"row {i}")

    def test_each_row_keeps_its_own_max_tokens(self):
        prompts = [[20, 21, 22], [30, 31, 32]]
        texts, stats = self.run_batch([{"ids": prompts[0], "max_tokens": 3, "sampling": {"temperature": 0}},
                                       {"ids": prompts[1], "max_tokens": 9, "sampling": {"temperature": 0}}])
        self.assertEqual(stats[0]["completion_tokens"], 3)
        self.assertLessEqual(stats[1]["completion_tokens"], 9)
        self.assertGreater(stats[1]["completion_tokens"], 3)
        self.assertEqual(stats[0]["finish_reason"], "length")

    def test_a_row_can_be_cancelled_without_touching_the_others(self):
        prompts = [[20, 21, 22], [30, 31, 32]]
        texts, stats = self.run_batch([
            {"ids": prompts[0], "max_tokens": 8, "sampling": {"temperature": 0}, "should_stop": lambda: True},
            {"ids": prompts[1], "max_tokens": 8, "sampling": {"temperature": 0}}])
        self.assertEqual(stats[0]["finish_reason"], "cancelled")
        self.assertEqual(stats[1]["completion_tokens"], 8)

    def test_stats_are_marked_as_batched(self):
        _, stats = self.run_batch([{"ids": [20, 21, 22], "max_tokens": 4, "sampling": {"temperature": 0}}])
        self.assertTrue(stats[0]["batched"])
        self.assertEqual(stats[0]["prompt_tokens"], 3)

    def test_an_empty_batch_is_fine(self):
        self.assertEqual(self.eng.generate_batch([]), [])


if __name__ == "__main__":
    unittest.main()
