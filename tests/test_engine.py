"""Engine tests (stdlib unittest).

    .venv/bin/python -m unittest discover -s tests -v

Unit tests need only MLX. Integration tests load a real model and run when
ENGINE_TEST_MODEL points to a model directory (they are skipped otherwise):

    ENGINE_TEST_MODEL=~/.cache/huggingface/hub/dealignai/Bonsai-2-27B-CRACK-Ternary-JANG \
        .venv/bin/python -m unittest discover -s tests -v
"""
import json
import math
import os
import sys
import unittest

import mlx.core as mx
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.generate import Drafter, Sampler, is_repeating, mrope_positions  # noqa: E402
from engine.model import F16, Hadamard, KVCache, MRoPE, QLinear, SSMCache  # noqa: E402
from engine.server import OutputParser, normalize_messages, parse_tool_calls, resolve_options  # noqa: E402

MODEL = os.path.expanduser(os.environ.get("ENGINE_TEST_MODEL", ""))


def hadamard_matrix(n):
    h = np.array([[1.0]])
    while h.shape[0] < n:
        h = np.block([[h, h], [h, -h]])
    return h / math.sqrt(n)


class TestHadamard(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.block = 64
        self.signs = {256: mx.array(rng.choice([-1.0, 1.0], 256).astype(np.float32))}
        self.h = Hadamard(self.block, self.signs)
        self.x = mx.array(rng.standard_normal((3, 256)).astype(np.float16))

    def test_forward_matches_explicit_matrix(self):
        y = np.array(self.h.fwd(self.x).astype(mx.float32))
        x = np.array(self.x.astype(mx.float32)) * np.array(self.signs[256])
        H = hadamard_matrix(self.block)
        ref = (x.reshape(-1, self.block) @ H.T).reshape(3, 256)
        np.testing.assert_allclose(y, ref, atol=2e-2)

    def test_inverse_undoes_forward(self):
        back = self.h.inv(self.h.fwd(self.x))
        np.testing.assert_allclose(np.array(back.astype(mx.float32)),
                                   np.array(self.x.astype(mx.float32)), atol=2e-2)

    def test_rotated_matmul_equals_plain_matmul(self):
        # W stored in rotated basis: W_rot = W @ diag(s) @ H^T  =>  W_rot @ H(s*x) == W @ x
        rng = np.random.default_rng(1)
        W = rng.standard_normal((32, 256)).astype(np.float32)
        s = np.array(self.signs[256])
        Hb = np.kron(np.eye(256 // self.block), hadamard_matrix(self.block))
        W_rot = W @ np.diag(s) @ Hb.T
        x = np.array(self.x.astype(mx.float32))
        got = np.array(self.h.fwd(self.x).astype(mx.float32)) @ W_rot.T
        np.testing.assert_allclose(got, x @ W.T, rtol=5e-2, atol=5e-2)


class TestQLinear(unittest.TestCase):
    def test_matches_dequantized_matmul(self):
        w = mx.random.normal((64, 256))
        qw, sc, bi = mx.quantize(w, group_size=64, bits=4)
        lin = QLinear(256, 64, 4, 64)
        lin.update({"weight": qw, "scales": sc.astype(F16), "biases": bi.astype(F16)})
        x = mx.random.normal((2, 256)).astype(F16)
        ref = x.astype(mx.float32) @ mx.dequantize(qw, sc, bi, group_size=64, bits=4).T
        np.testing.assert_allclose(np.array(lin(x).astype(mx.float32)), np.array(ref), rtol=5e-2, atol=5e-1)

    def test_fused_projection_equals_separate(self):
        a, b = mx.random.normal((32, 128)), mx.random.normal((48, 128))
        qa, qb = mx.quantize(a, 64, 4), mx.quantize(b, 64, 4)
        fused = QLinear(128, 80, 4, 64)
        fused.update({k: mx.concatenate([qa[i], qb[i]]).astype(F16 if i else mx.uint32)
                      for i, k in enumerate(("weight", "scales", "biases"))})
        x = mx.random.normal((1, 128)).astype(F16)
        sep = [mx.quantized_matmul(x, *q[:1], q[1].astype(F16), q[2].astype(F16), transpose=True,
                                   group_size=64, bits=4) for q in (qa, qb)]
        np.testing.assert_allclose(np.array(fused(x)), np.array(mx.concatenate(sep, -1)), atol=1e-2)


class TestCaches(unittest.TestCase):
    def test_kv_cache_grows_and_trims(self):
        c = KVCache()
        for n in (3, 300, 1):
            k, v = c.update(mx.ones((1, 2, n, 4)), mx.ones((1, 2, n, 4)))
        self.assertEqual(c.offset, 304)
        self.assertEqual(k.shape[2], 304)
        c.trim_to(10)
        k, _ = c.update(mx.zeros((1, 2, 1, 4)), mx.zeros((1, 2, 1, 4)))
        self.assertEqual(k.shape[2], 11)

    def test_ssm_snapshot_restore(self):
        c = SSMCache()
        c.conv, c.state, c.offset = mx.ones((1, 3, 4)), mx.zeros((1, 2, 2, 2)), 7
        snap = c.snapshot()
        c.conv, c.state, c.offset = mx.zeros((1, 3, 4)), mx.ones((1, 2, 2, 2)), 9
        c.restore(snap)
        self.assertEqual(c.offset, 7)
        self.assertEqual(c.conv.sum().item(), 12)


class TestRope(unittest.TestCase):
    def test_mrope_equals_1d_rope_for_text_positions(self):
        dims, base = 64, 1e7
        r = MRoPE(dims, base, [11, 11, 10])
        x = mx.random.normal((1, 2, 5, 128)).astype(F16)
        pos = mx.broadcast_to(mx.arange(3, 8)[None], (3, 5))
        cos, sin = r.cos_sin(pos)
        got = r.apply(x, cos, sin)
        ref = mx.fast.rope(x, dims, traditional=False, base=base, scale=1.0, offset=3)
        np.testing.assert_allclose(np.array(got.astype(mx.float32)), np.array(ref.astype(mx.float32)), atol=2e-2)

    def test_mrope_positions_for_image(self):
        IMG = 9
        ids = [1, 2, IMG, IMG, IMG, IMG, IMG, IMG, 3]  # grid (1, 4, 6) merge 2 -> 2x3 = 6 tokens
        pos, delta = mrope_positions(ids, [(1, 4, 6)], IMG, 2)
        self.assertEqual(pos[:, :2].tolist(), [[0, 1]] * 3)
        self.assertEqual(pos[0, 2:8].tolist(), [2] * 6)           # t
        self.assertEqual(pos[1, 2:8].tolist(), [2, 2, 2, 3, 3, 3])  # h
        self.assertEqual(pos[2, 2:8].tolist(), [2, 3, 4, 2, 3, 4])  # w
        self.assertEqual(pos[:, 8].tolist(), [5, 5, 5])
        self.assertEqual(delta, 6 - len(ids))


class TestSampler(unittest.TestCase):
    def test_greedy(self):
        logits = mx.array([[[0.1, 3.0, 0.2, 0.0]]])
        self.assertEqual(Sampler(4, temperature=0)(logits).item(), 1)

    def test_top_k_one_is_greedy(self):
        logits = mx.array([[[0.1, 0.2, 5.0, 0.0]]])
        s = Sampler(4, temperature=1.0, top_k=1, seed=0)
        self.assertTrue(all(s(logits).item() == 2 for _ in range(20)))

    def test_top_p_excludes_tail(self):
        logits = mx.log(mx.array([[[0.7, 0.25, 0.04, 0.01]]]))
        s = Sampler(4, temperature=1.0, top_p=0.8, seed=1)
        self.assertTrue({s(logits).item() for _ in range(200)} <= {0, 1})

    def test_presence_penalty_discourages_repeats(self):
        logits = mx.array([[[2.0, 1.9]]])
        s = Sampler(2, temperature=0, presence_penalty=1.0)
        self.assertEqual([s(logits).item() for _ in range(2)], [0, 1])


class TestDrafter(unittest.TestCase):
    def test_continues_most_recent_match(self):
        d = Drafter([1, 2, 3, 4, 9, 1, 2, 3, 5, 6, 7, 1, 2, 3])
        self.assertEqual(d.propose(3), [5, 6, 7])  # latest earlier "1 2 3" is followed by 5 6 7

    def test_prefers_longer_ngram(self):
        d = Drafter([7, 2, 3, 8, 1, 2, 3, 9, 1, 2, 3])
        self.assertEqual(d.propose(1), [9])        # 4-gram "x 1 2 3"? no: 3-gram "1 2 3" -> 9

    def test_no_match(self):
        self.assertEqual(Drafter([1, 2, 3, 4]).propose(4), [])

    def test_incremental_add(self):
        d = Drafter([10, 11, 12, 13])
        for t in (10, 11):
            d.add(t)
        self.assertEqual(d.propose(2), [12, 13])


class TestSpeculativeSampling(unittest.TestCase):
    def test_greedy_verify(self):
        s = Sampler(4, temperature=0)
        row = mx.array([0.1, 3.0, 0.2, 0.0])
        self.assertEqual(s.verify(row, 1), (True, 1))
        self.assertEqual(s.verify(row, 2), (False, 1))

    def test_acceptance_preserves_distribution(self):
        # target p = softmax(logits); deterministic draft d=0 accepted w.p. p(0),
        # otherwise resampled from p without d: the marginal must equal p.
        logits = mx.log(mx.array([0.5, 0.3, 0.2]))
        s = Sampler(3, temperature=1.0, seed=3)
        counts = [0, 0, 0]
        for _ in range(3000):
            _, t = s.verify(logits, 0)
            counts[t] += 1
        freq = [c / 3000 for c in counts]
        for got, want in zip(freq, (0.5, 0.3, 0.2)):
            self.assertAlmostEqual(got, want, delta=0.04)


class TestSSMRollback(unittest.TestCase):
    def test_commit_equals_sequential(self):
        from engine.gated_delta import gated_delta_kernel
        rng = np.random.default_rng(0)
        T, Hk, Hv, D = 5, 2, 4, 32
        mk = lambda *s: mx.array(rng.standard_normal(s).astype(np.float32) * 0.3).astype(F16)
        q, k, v = mk(1, T, Hk, D), mk(1, T, Hk, D), mk(1, T, Hv, D)
        g = mx.array(rng.uniform(0.8, 1.0, (1, T, Hv)).astype(np.float32))
        b = mx.array(rng.uniform(0.1, 0.9, (1, T, Hv)).astype(np.float32))
        c = SSMCache()
        c.state, c.offset, c.conv = mx.zeros((1, Hv, D, D)), 10, mx.zeros((1, 3, 8))
        c.begin_verify()
        _, c.state = gated_delta_kernel(q, k, v, g, b, c.state)
        c.pending = (q, k, v, g, b, mx.ones((1, 3 + T, 8)), 4)
        c.commit(2)
        _, ref = gated_delta_kernel(q[:, :2], k[:, :2], v[:, :2], g[:, :2], b[:, :2], mx.zeros((1, Hv, D, D)))
        np.testing.assert_allclose(np.array(c.state), np.array(ref), atol=1e-6)
        self.assertEqual(c.offset, 12)
        self.assertEqual(c.conv.shape, (1, 3, 8))

    def test_kv_commit(self):
        c = KVCache()
        c.update(mx.ones((1, 1, 7, 4)), mx.ones((1, 1, 7, 4)))
        c.begin_verify()
        c.update(mx.ones((1, 1, 5, 4)), mx.ones((1, 1, 5, 4)))
        c.commit(3)
        self.assertEqual(c.offset, 10)


class TestMemoryWatchdog(unittest.TestCase):
    def test_readings_are_sane(self):
        from engine import memory
        used = memory.used_fraction()
        self.assertTrue(0.0 < used < 1.0)
        self.assertGreater(memory.TOTAL, 1 << 30)
        self.assertIn(memory.pressure_level(), (1, 2, 4))
        self.assertGreater(memory.headroom_bytes(0.99), memory.headroom_bytes(0.5))

    def _fake_engine(self, mem_limit=0.88):
        from types import SimpleNamespace
        return SimpleNamespace(prefill_step=1024, base_work=64 * 2**20, act_per_token=900 * 1024,
                               kv_per_token=4096, mem_limit=mem_limit, snapshots=[], log=lambda *_: None)

    def test_reserve_allows_a_tiny_prompt_right_at_the_limit(self):
        from engine import generate as gen_mod
        eng = self._fake_engine()
        orig = gen_mod.memory.headroom_bytes
        # idle usage already sits exactly at mem_limit (common with a big model resident) —
        # a few-token prompt must still fit inside the 2% grace band, not be refused.
        gen_mod.memory.headroom_bytes = lambda limit: int((limit - 0.88) * gen_mod.memory.TOTAL - 50 * 2**20)
        try:
            gen_mod.Engine._reserve(eng, 4)
        finally:
            gen_mod.memory.headroom_bytes = orig
        self.assertEqual(eng.step_now, 128)

    def test_reserve_still_refuses_a_real_deficit(self):
        from engine import generate as gen_mod
        eng = self._fake_engine()
        orig = gen_mod.memory.headroom_bytes
        gen_mod.memory.headroom_bytes = lambda limit: int((limit - 0.88) * gen_mod.memory.TOTAL - 2 * 2**30)
        try:
            with self.assertRaises(gen_mod.MemoryLimitError):
                gen_mod.Engine._reserve(eng, 4)
        finally:
            gen_mod.memory.headroom_bytes = orig


class TestRepetitionDetector(unittest.TestCase):
    def test_detects_a_verbatim_loop(self):
        chunk = "Okay, now let's process the user's message again. "
        self.assertTrue(is_repeating(chunk * 3))

    def test_detects_a_two_phrase_cycle(self):
        # not one repeated sentence but a cycle of two — "A, then B, then A, then B..."
        cycle = "Let me reconsider the question. Wait, actually let's re-examine this. "
        self.assertTrue(is_repeating(cycle * 3))

    def test_ignores_normal_prose(self):
        self.assertFalse(is_repeating("The quick brown fox jumps over the lazy dog. " * 3 + "but not quite the same."))

    def test_short_tail_is_not_a_false_positive(self):
        self.assertFalse(is_repeating("hi"))


class TestOutputParser(unittest.TestCase):
    def run_parser(self, chunks, thinking=True, tools=False):
        p, out = OutputParser(thinking, tools), []
        for c in chunks:
            out += p.feed(c)
        out += p.finish()
        r = "".join(s for k, s in out if k == "reasoning")
        c = "".join(s for k, s in out if k == "content")
        return r, c, p.tool_text

    def test_split_think_across_chunks(self):
        r, c, _ = self.run_parser(["Let me ", "think</th", "ink>\n\nHello", " world"])
        self.assertEqual((r, c), ("Let me think", "Hello world"))

    def test_no_thinking(self):
        r, c, _ = self.run_parser(["\n\nAnswer: 4"], thinking=False)
        self.assertEqual((r, c), ("", "Answer: 4"))

    def test_tool_call_is_held_back(self):
        r, c, t = self.run_parser(["ok</think>Let me check.<tool", "_call>\n<function=f>\n</function>\n</tool_call>"],
                                  tools=True)
        self.assertEqual(c, "Let me check.")
        self.assertTrue(t.startswith("<tool_call>"))

    def test_tool_tag_ignored_without_tools(self):
        _, c, t = self.run_parser(["<tool_call>x"], thinking=False, tools=False)
        self.assertEqual((c, t), ("<tool_call>x", None))


class TestToolCalls(unittest.TestCase):
    TOOLS = [{"type": "function", "function": {"name": "search", "parameters": {"type": "object", "properties": {
        "query": {"type": "string"}, "limit": {"type": "integer"}, "filters": {"type": "object"}}}}}]

    def test_parse_typed_arguments(self):
        text = ("<tool_call>\n<function=search>\n<parameter=query>\n42 cats\n</parameter>\n"
                "<parameter=limit>\n5\n</parameter>\n<parameter=filters>\n{\"a\": [1]}\n</parameter>\n"
                "</function>\n</tool_call>")
        calls = parse_tool_calls(text, self.TOOLS)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["function"]["name"], "search")
        self.assertEqual(json.loads(calls[0]["function"]["arguments"]),
                         {"query": "42 cats", "limit": 5, "filters": {"a": [1]}})

    def test_multiple_and_multiline(self):
        text = ("<tool_call>\n<function=a>\n<parameter=x>\nline1\nline2\n</parameter>\n</function>\n</tool_call>\n"
                "<tool_call>\n<function=b>\n</function>\n</tool_call>")
        calls = parse_tool_calls(text, [])
        self.assertEqual([c["function"]["name"] for c in calls], ["a", "b"])
        self.assertEqual(json.loads(calls[0]["function"]["arguments"]), {"x": "line1\nline2"})

    def test_normalize_assistant_tool_calls(self):
        msgs = normalize_messages([{"role": "assistant", "content": None, "tool_calls": [
            {"id": "1", "type": "function", "function": {"name": "f", "arguments": "{\"k\": 1}"}}]},
            {"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}])
        self.assertEqual(msgs[0]["tool_calls"][0]["function"]["arguments"], {"k": 1})
        self.assertEqual(msgs[1]["content"], "ab")

    def test_resolve_options(self):
        self.assertEqual(resolve_options({"reasoning_effort": "high"})[:2], (None, "xhigh"))
        self.assertEqual(resolve_options({"chat_template_kwargs": {"enable_thinking": False}})[0], False)
        self.assertEqual(resolve_options({"reasoning_effort": "none"})[:2], (False, None))


@unittest.skipUnless(MODEL and os.path.isdir(MODEL), "set ENGINE_TEST_MODEL to run integration tests")
class TestIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from engine.generate import Engine
        cls.eng = Engine(MODEL, vision=False, max_context=4096, max_tokens=64, log=lambda *_: None)

    def gen(self, messages, speculative=True, **kw):
        self.eng.speculative = speculative
        text, _, _ = self.eng.build_prompt(messages, thinking=False)
        ids, _ = self.eng.encode(text, [])
        toks, out, st = [], "", None
        for item in self.eng.generate(ids, sampling={"temperature": 0}, **kw):
            if isinstance(item, dict):
                st = item
            else:
                out += item[1]
        return ids, out, st

    def test_answers_correctly(self):
        _, out, st = self.gen([{"role": "user", "content": "What is the capital of France? One word."}])
        self.assertIn("Paris", out)
        self.assertEqual(st["finish_reason"], "stop")

    def test_stream_decoder_roundtrip(self):
        from engine.tokenizer import StreamDecoder
        text = "Привет, 世界! émoji 🎉 done"
        ids = self.eng.tok.encode(text)
        d = StreamDecoder(self.eng.tok)
        self.assertEqual("".join(d.add(t) for t in ids) + d.flush(), text)

    def test_prefix_cache_reuse_is_exact(self):
        conv = [{"role": "user", "content": "Count from 1 to 5."}]
        ids1, out1, _ = self.gen(conv)
        conv2 = conv + [{"role": "assistant", "content": out1}, {"role": "user", "content": "Now from 6 to 8."}]
        ids2, out2, st2 = self.gen(conv2)
        self.assertGreater(st2["cached_tokens"], len(ids1) // 2)
        self.eng.cache = None  # same prompt, cold cache must give the same greedy answer
        _, out3, st3 = self.gen(conv2)
        self.assertEqual(st3["cached_tokens"], 0)
        self.assertEqual(out2, out3)

    def test_speculative_greedy_matches_plain_on_copy_task(self):
        src = "\n".join(f"def f{i}(x):\n    return x * {i} + {i * 7}\n" for i in range(12))
        msgs = [{"role": "user", "content": "Repeat this code exactly, renaming f3 to g3. Output only code.\n\n" + src}]
        self.eng.cache = None
        _, plain, _ = self.gen(msgs, speculative=False, max_tokens=64)
        self.eng.cache = None
        _, spec, st = self.gen(msgs, speculative=True, max_tokens=64)
        self.assertEqual(plain, spec)
        self.assertGreater(st["verifies"], 0)
        self.assertGreater(st["draft_accept"], 0.5)

    def test_think_budget_forces_an_answer(self):
        if not self.eng.supports_thinking:
            self.skipTest("model has no thinking mode")
        old_frac = self.eng.think_frac
        self.eng.think_frac = 0.2  # tiny budget: the model can't possibly finish reasoning in time
        try:
            self.eng.cache = None
            text, _, thinking_open = self.eng.build_prompt([{"role": "user", "content": "What is 2+2?"}],
                                                            thinking=True)
            ids, _ = self.eng.encode(text, [])
            out, st = "", None
            for item in self.eng.generate(ids, sampling={"temperature": 0}, max_tokens=64,
                                          thinking_open=thinking_open):
                if isinstance(item, dict):
                    st = item
                else:
                    out += item[1]
            self.assertIn("</think>", out)  # forced closed, not just cut off mid-thought
            self.assertEqual(st["finish_reason"], "stop")
        finally:
            self.eng.think_frac = old_frac

    def test_memory_limit_refuses_instead_of_swapping(self):
        from engine.generate import MemoryLimitError
        old = self.eng.mem_limit
        self.eng.mem_limit = 0.05  # far below what is already in use
        try:
            with self.assertRaises(MemoryLimitError):
                self.eng.cache = None
                self.gen([{"role": "user", "content": "hi"}])
        finally:
            self.eng.mem_limit = old

    def test_ssd_prefix_cache_survives_restart(self):
        import tempfile
        from engine.prefix_store import PrefixStore
        old = self.eng.store
        self.eng.store = PrefixStore(tempfile.mkdtemp(), 2 << 30, min_tokens=256, log=lambda *a: None)
        try:
            doc = " ".join(f"Fact {i}: the value of item {i} is {i * 7}." for i in range(120))
            conv = [{"role": "system", "content": doc}, {"role": "user", "content": "What is the value of item 5?"}]
            self.eng.cache = None
            self.gen(conv, max_tokens=16)
            self.eng.persist_prefix()
            self.assertGreaterEqual(len(self.eng.store.entries), 1)  # system-prompt and full-prompt points
            conv2 = [conv[0], {"role": "user", "content": "What is the value of item 9?"}]
            self.eng.cache = None  # "restart": nothing in memory
            _, warm, st = self.gen(conv2, max_tokens=16)
            self.assertGreater(st["cached_tokens"], 256)
            self.eng.cache, self.eng.snapshots = None, []
            store, self.eng.store = self.eng.store, None
            _, cold, st2 = self.gen(conv2, max_tokens=16)
            self.eng.store = store
            self.assertEqual(st2["cached_tokens"], 0)
            self.assertEqual(warm, cold)
        finally:
            self.eng.store = old

    def test_sleep_and_wake(self):
        import mlx.core as mx
        self.eng.sleep()
        self.assertTrue(self.eng.sleeping)
        self.assertLess(mx.get_active_memory(), 1 << 30)
        _, out, _ = self.gen([{"role": "user", "content": "What is the capital of France? One word."}])
        self.assertFalse(self.eng.sleeping)
        self.assertIn("Paris", out)

    def test_max_tokens_clamped(self):
        _, _, st = self.gen([{"role": "user", "content": "Write a long story."}], max_tokens=10**6)
        self.assertLessEqual(st["completion_tokens"], self.eng.max_tokens)


if __name__ == "__main__":
    unittest.main()


class TestSamplingPresets(unittest.TestCase):
    def test_jang_modes_win(self):
        from engine.generate import sampling_presets
        jang = {"chat": {"sampling_modes": {"thinking": {"temperature": 0.3, "top_k": 5},
                                            "instruct_nothinking": {"temperature": 0.1}}}}
        p, src = sampling_presets(jang, {"temperature": 9.0})
        self.assertEqual((p[True]["temperature"], p[False]["temperature"], src), (0.3, 0.1, "jang_config"))

    def test_generation_config_when_no_modes(self):
        from engine.generate import sampling_presets
        p, src = sampling_presets({}, {"temperature": 0.6, "top_p": 0.9, "top_k": 40, "eos_token_id": 1})
        self.assertEqual(p[True], {"temperature": 0.6, "top_p": 0.9, "top_k": 40})
        self.assertEqual((p[True], src), (p[False], "generation_config"))

    def test_family_defaults_when_the_bundle_ships_nothing(self):
        # a bundle with no presets used to fall through to the Sampler's own defaults:
        # temperature 1.0 over the whole vocabulary, no top-k/top-p
        from engine.generate import sampling_presets
        p, src = sampling_presets({}, {})
        self.assertEqual(src, "qwen3.5 defaults")
        for thinking in (True, False):
            self.assertGreater(p[thinking]["top_k"], 0)
            self.assertLess(p[thinking]["top_p"], 1.0)
        self.assertEqual(p[True]["temperature"], 1.0)
        self.assertEqual(p[False]["presence_penalty"], 1.5)

    def test_presets_are_independent_copies(self):
        from engine.generate import sampling_presets
        a, _ = sampling_presets({}, {})
        a[True]["temperature"] = 99
        b, _ = sampling_presets({}, {})
        self.assertEqual(b[True]["temperature"], 1.0)


class TestThinkingDefault(unittest.TestCase):
    """The reasoning mode a model gets when nobody asks is its template's own default."""
    # the two conventions found in the Qwen3.5 family's chat templates
    ON_UNLESS_DISABLED = ("{{ '<|im_start|>assistant\\n' }}"
                          "{%- if enable_thinking is defined and enable_thinking is false %}"
                          "{{ '<think>\\n\\n</think>\\n\\n' }}{%- else %}{{ '<think>\\n' }}{%- endif %}")
    OFF_UNLESS_ENABLED = ("{{ '<|im_start|>assistant\\n' }}"
                          "{%- if enable_thinking is defined and enable_thinking is true %}"
                          "{{ '<think>\\n' }}{%- else %}{{ '<think>\\n\\n</think>\\n\\n' }}{%- endif %}")

    def decide(self, template, **kw):
        import jinja2
        from engine.generate import template_thinks_by_default
        return template_thinks_by_default(lambda: jinja2.Template(template).render(**kw))

    def test_template_that_reasons_unless_disabled(self):
        self.assertTrue(self.decide(self.ON_UNLESS_DISABLED))

    def test_template_that_only_reasons_when_enabled(self):
        self.assertFalse(self.decide(self.OFF_UNLESS_ENABLED))

    def test_explicit_kwarg_flips_both(self):
        self.assertTrue(self.decide(self.OFF_UNLESS_ENABLED, enable_thinking=True))
        self.assertFalse(self.decide(self.ON_UNLESS_DISABLED, enable_thinking=False))

    def test_a_template_that_cannot_render_keeps_the_old_default(self):
        from engine.generate import template_thinks_by_default
        def boom():
            raise ValueError("needs tools")
        self.assertTrue(template_thinks_by_default(boom))


class TestModelsWithoutReasoningOrVision(unittest.TestCase):
    """Not every model reasons, and not every model sees: a synthetic bundle (random weights,
    see tests/synth.py) stands in for the shapes of model nobody has lying around."""

    @classmethod
    def setUpClass(cls):
        import tempfile
        from engine.generate import Engine
        from tests.synth import CHATML, CHATML_THINKING_OFF_BY_DEFAULT, make_bundle
        cls.logs = []
        cls.plain_dir, cls.small_dir, cls.cfg_dir = (tempfile.mkdtemp() for _ in range(3))
        make_bundle(cls.plain_dir, CHATML, seed=1)  # no enable_thinking anywhere in the template
        make_bundle(cls.small_dir, CHATML_THINKING_OFF_BY_DEFAULT, seed=2)
        make_bundle(cls.cfg_dir, CHATML, seed=3, generation_config={"temperature": 0.5, "top_p": 0.7, "top_k": 7})
        kw = dict(vision=True, max_context=1024, max_tokens=16, disk_cache_gb=0)
        cls.plain = Engine(cls.plain_dir, log=cls.logs.append, **kw)
        cls.small = Engine(cls.small_dir, log=lambda *_: None, **kw)
        cls.cfg = Engine(cls.cfg_dir, log=lambda *_: None, **kw)

    def run_gen(self, eng, text, **kw):
        ids, _ = eng.encode(text, [])
        out, st = "", None
        for it in eng.generate(ids, sampling={"temperature": 0}, max_tokens=6, **kw):
            if isinstance(it, dict):
                st = it
            else:
                out += it[1]
        return out, st

    def test_capabilities_are_reported_honestly(self):
        e = self.plain
        self.assertFalse(e.supports_thinking)
        self.assertFalse(e.thinking_default)
        self.assertFalse(e.thinking)
        self.assertFalse(e.vision_allowed)

    def test_a_reasoning_request_is_ignored_not_an_error(self):
        e = self.plain
        text, _, opened = e.build_prompt([{"role": "user", "content": "hi"}], thinking=True, effort="high")
        self.assertNotIn("<think>", text)
        self.assertFalse(opened)
        _, st = self.run_gen(e, text, thinking_open=opened)
        self.assertIsNone(st["think_forced"])  # no reasoning phase -> nothing to budget or force-close

    def test_an_image_gets_a_clear_error(self):
        text, _, _ = self.plain.build_prompt([{"role": "user", "content": "hi"}])
        with self.assertRaises(ValueError) as ctx:
            self.plain.encode(text, ["data:image/png;base64,AAAA"])
        self.assertIn("no vision support", str(ctx.exception))

    def test_a_bundle_with_no_presets_gets_the_family_defaults(self):
        self.assertTrue(any("Qwen3.5 family defaults" in m for m in self.logs))
        self.assertEqual(self.plain.sampling[True]["top_k"], 20)
        self.assertLess(self.plain.sampling[True]["top_p"], 1.0)

    def test_generation_config_beats_the_family_defaults(self):
        self.assertEqual(self.cfg.sampling[True], {"temperature": 0.5, "top_p": 0.7, "top_k": 7})

    def test_small_model_convention_reasoning_is_opt_in(self):
        e = self.small
        self.assertTrue(e.supports_thinking)
        self.assertFalse(e.thinking_default)
        self.assertFalse(e.thinking)  # nobody asked -> the template's own default
        _, _, opened = e.build_prompt([{"role": "user", "content": "hi"}])
        self.assertFalse(opened)
        text, _, opened = e.build_prompt([{"role": "user", "content": "hi"}], thinking=True)
        self.assertTrue(opened)
        self.assertTrue(text.rstrip().endswith("<think>"))

    def test_an_explicit_choice_overrides_the_template_default(self):
        from engine.generate import Engine
        forced_on = Engine(self.small_dir, thinking=True, vision=False, max_context=1024, max_tokens=16,
                           disk_cache_gb=0, log=lambda *_: None)
        self.assertTrue(forced_on.thinking)
        self.assertFalse(forced_on.thinking_default)  # the default itself is unchanged
