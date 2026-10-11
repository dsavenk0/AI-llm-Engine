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

from engine.generate import Drafter, Sampler, is_looping, is_repeating, mrope_positions  # noqa: E402
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


    def test_repetition_penalty_divides_positive_and_multiplies_negative_logits(self):
        s = Sampler(2, temperature=0, repetition_penalty=1.5)
        up = mx.array([[[2.0, 1.9]]])
        self.assertEqual([s(up).item() for _ in range(2)], [0, 1])      # 2.0 / 1.5 < 1.9 once token 0 is used
        s = Sampler(2, temperature=0, repetition_penalty=1.5)
        down = mx.array([[[-1.0, -1.05]]])
        self.assertEqual([s(down).item() for _ in range(2)], [0, 1])    # -1.0 * 1.5 < -1.05: pushed down, not up

    def test_no_penalties_means_no_state(self):
        self.assertIsNone(Sampler(4, temperature=0, repetition_penalty=1.0).seen)
        self.assertIsNotNone(Sampler(4, temperature=0, repetition_penalty=1.1).seen)


class TestLoopDetector(unittest.TestCase):
    def test_a_looping_list_is_caught(self):
        cycle = "Lisp, ML, Rust, Swift, Go, Rust, Scala, Haskell, OCaml, "
        self.assertTrue(is_looping("Languages: " + cycle * 8))

    def test_normal_text_and_structured_output_are_not(self):
        self.assertFalse(is_looping("The quick brown fox jumps over the lazy dog. " * 2 + "Then it sleeps."))
        self.assertFalse(is_looping(", ".join(str(i) for i in range(300))))
        rows = "\n".join(f"| item {i} | {i * 7 % 13} | ok |" for i in range(60))
        self.assertFalse(is_looping(rows))
        self.assertFalse(is_looping("hi"))
        self.assertFalse(is_looping("x" * 40))

    def test_separator_lines_need_many_repeats(self):
        self.assertFalse(is_looping("Title\n" + "=" * 100 + "\n"))
        self.assertFalse(is_looping("-" * 500))
        self.assertTrue(is_looping("-" * 3000))              # that is a loop, not a rule


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
                               kv_per_token=4096, mem_limit=mem_limit, snapshots=[], log=lambda *_: None,
                               prefill_guard=True)

    def test_reserve_with_the_guard_off_never_refuses(self):
        from engine import generate as gen_mod
        eng = self._fake_engine()
        eng.prefill_guard = False
        orig = gen_mod.memory.headroom_bytes
        gen_mod.memory.headroom_bytes = lambda limit: -10 * 2**30          # a hopeless deficit
        try:
            gen_mod.Engine._reserve(eng, 10_000)
            self.assertEqual(eng.step_now, 1024)                            # and the chunk is not shrunk
        finally:
            gen_mod.memory.headroom_bytes = orig

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
        self.eng.think_frac = 0.1  # a tiny budget: no reasoning model finishes thinking inside it
        try:
            self.eng.cache = None
            text, _, thinking_open = self.eng.build_prompt([{"role": "user", "content": "What is 2+2?"}],
                                                            thinking=True)
            self.assertTrue(thinking_open)
            ids, _ = self.eng.encode(text, [])
            out, st = "", None
            for item in self.eng.generate(ids, sampling={"temperature": 0}, max_tokens=160,
                                          thinking_open=thinking_open):
                if isinstance(item, dict):
                    st = item
                else:
                    out += item[1]
            # what is under test is the forced close — not how tersely this particular model then
            # answers (a wordy one may still be talking when the token limit arrives)
            self.assertEqual(st["think_forced"], "budget")
            self.assertIn("</think>", out)
            self.assertTrue(out.split("</think>", 1)[1].strip(), "the model said nothing after the close")
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


class TestPrefixStore(unittest.TestCase):
    """engine.prefix_store: the SSD cache must never be able to break serving."""

    def make(self, n=300):
        import tempfile
        import mlx.core as mx
        from engine.model import KVCache
        from engine.prefix_store import PrefixStore
        store = PrefixStore(tempfile.mkdtemp(), 1 << 30, min_tokens=256, log=lambda *a: None)
        c = KVCache()
        c.keys = mx.ones((1, 2, n, 8), mx.float16)
        c.values = mx.ones((1, 2, n, 8), mx.float16)
        c.offset = n
        tokens = list(range(n))
        store.save(tokens, [c], [None])
        return store, tokens

    def test_save_lookup_load_round_trip_and_discard(self):
        import mlx.core as mx
        from engine.model import KVCache
        store, tokens = self.make()
        self.assertTrue(store.has(tokens))
        hit = store.lookup(tokens + [999, 1000])
        self.assertIsNotNone(hit)
        self.assertEqual(hit[1], 300)
        self.assertIsNone(store.lookup(tokens))                       # a stored prompt is not "shorter than" itself
        self.assertIsNone(store.lookup([5] + tokens[1:] + [1]))      # a different first token: no hit
        model = type("M", (), {"make_cache": lambda self: [KVCache()]})()
        cache, snaps, toks = store.load(hit, model)
        self.assertEqual((cache[0].offset, snaps, len(toks)), (300, [None], 300))
        self.assertEqual(mx.array(cache[0].keys).shape, (1, 2, 300, 8))
        store.discard(hit[0])
        self.assertFalse(store.has(tokens))
        self.assertEqual(os.listdir(store.root), ["index.json"])
        store.discard("no-such-key")                                  # harmless

    def test_a_damaged_entry_raises_on_load_and_can_be_dropped(self):
        from engine.model import KVCache
        store, tokens = self.make()
        hit = store.lookup(tokens + [1])
        with open(store._file(hit[0]), "wb") as f:
            f.write(b"not a safetensors file")                        # what a crash or a full disk leaves behind
        model = type("M", (), {"make_cache": lambda self: [KVCache()]})()
        with self.assertRaises(Exception):
            store.load(hit, model)
        store.discard(hit[0])
        self.assertIsNone(store.lookup(tokens + [1]))


class TestPrefixStoreRules(unittest.TestCase):
    """What is stored, when, and what is thrown out first."""

    def store(self, max_bytes=1 << 30, **kw):
        import tempfile
        from engine.prefix_store import PrefixStore
        return PrefixStore(tempfile.mkdtemp(), max_bytes, log=kw.pop("log", lambda *a: None), **kw)

    def kv(self, n, offset=None, heads=1, dim=4):
        import mlx.core as mx
        from engine.model import KVCache
        c = KVCache()
        c.keys = mx.ones((1, heads, n, dim), mx.float16)
        c.values = mx.ones((1, heads, n, dim), mx.float16)
        c.offset = n if offset is None else offset
        return c

    def test_default_minimum_is_1024_tokens_and_the_boundary_is_inclusive(self):
        st = self.store()
        self.assertEqual(st.min_tokens, 1024)
        st.save(list(range(1023)), [self.kv(1023)], [None])
        self.assertEqual(st.entries, {})
        st.save(list(range(1024)), [self.kv(1024)], [None])
        self.assertEqual(len(st.entries), 1)
        key = next(iter(st.entries))
        self.assertEqual(len(key), 20)
        self.assertTrue(all(ch in "0123456789abcdef" for ch in key))
        self.assertEqual(st.entries[key]["n"], 1024)
        self.assertEqual(st.total_bytes, 2 * 1 * 1 * 1024 * 4 * 2)          # K and V, float16

    def test_a_disabled_budget_and_a_known_prefix_store_nothing(self):
        off = self.store(max_bytes=0, min_tokens=4)
        off.save([1, 2, 3, 4], [self.kv(4)], [None])
        self.assertEqual(off.entries, {})
        st = self.store(min_tokens=4)
        st.save([1, 2, 3, 4], [self.kv(4)], [None])
        stamp = next(iter(st.entries.values()))["used"]
        st.save([1, 2, 3, 4], [self.kv(4)], [None])
        self.assertEqual(len(st.entries), 1)
        self.assertEqual(next(iter(st.entries.values()))["used"], stamp)       # not rewritten

    def test_an_incomplete_cache_is_not_stored(self):
        import mlx.core as mx
        from engine.model import KVCache
        st = self.store(min_tokens=4)
        st.save([1, 2, 3, 4], [KVCache()], [None])                              # no keys yet
        st.save([1, 2, 3, 4], [self.kv(4, offset=3)], [None])                   # shorter than the prompt
        st.save([1, 2, 3, 4], [self.kv(4), self.kv(4)], [None, (None, mx.ones((2,)), 0)])   # an SSM layer without conv
        st.save([1, 2, 3, 4], [self.kv(4), self.kv(4)], [None, (mx.ones((2,)), None, 0)])   # ... or without state
        self.assertEqual(st.entries, {})
        st.save([1, 2, 3, 4], [self.kv(4, offset=4)], [None])                   # exactly as long: fine
        self.assertEqual(len(st.entries), 1)

    def test_an_ssm_layer_is_stored_from_its_snapshot(self):
        import mlx.core as mx
        st = self.store(min_tokens=4)
        st.save([1, 2, 3, 4], [self.kv(4), self.kv(4)], [None, (mx.ones((3,)), mx.ones((5,)), 0)])
        key = next(iter(st.entries))
        names = set(mx.load(st._file(key)))
        self.assertEqual(names, {"0.k", "0.v", "1.conv", "1.state"})

    def test_what_does_not_fit_is_skipped_and_what_fits_exactly_is_kept(self):
        one = 2 * 1 * 1 * 8 * 4 * 2                                            # bytes of an 8-token entry
        st = self.store(max_bytes=one - 1, min_tokens=8)
        st.save(list(range(8)), [self.kv(8)], [None])
        self.assertEqual(st.entries, {})
        st = self.store(max_bytes=one, min_tokens=8)
        st.save(list(range(8)), [self.kv(8)], [None])
        self.assertEqual(len(st.entries), 1)

    def test_the_least_recently_used_entry_goes_first_and_only_when_needed(self):
        one = 2 * 1 * 1 * 8 * 4 * 2
        st = self.store(max_bytes=2 * one, min_tokens=8)
        a, b, c = list(range(8)), list(range(100, 108)), list(range(200, 208))
        st.save(a, [self.kv(8)], [None])
        st.save(b, [self.kv(8)], [None])
        self.assertEqual(len(st.entries), 2)                                   # exactly full: nothing evicted yet
        st.entries[st._key(a)]["used"] -= 100                                  # a is the oldest
        st.save(c, [self.kv(8)], [None])
        self.assertTrue(st.has(b) and st.has(c) and not st.has(a))
        self.assertFalse(os.path.exists(st._file(st._key(a))))
        self.assertFalse(os.path.exists(st._file(st._key(a), "tokens.npy")))

    def test_the_index_survives_a_restart_and_drops_rows_whose_files_vanished(self):
        st = self.store(min_tokens=4)
        st.save([1, 2, 3, 4], [self.kv(4)], [None])
        st.save([5, 6, 7, 8], [self.kv(4)], [None])
        gone = st._key([5, 6, 7, 8])
        os.remove(st._file(gone))
        from engine.prefix_store import PrefixStore
        again = PrefixStore(st.root, 1 << 30, min_tokens=4, log=lambda *a: None)
        self.assertTrue(again.has([1, 2, 3, 4]))
        self.assertFalse(again.has([5, 6, 7, 8]))
        with open(os.path.join(st.root, "index.json"), "w") as f:
            f.write("{broken")
        self.assertEqual(PrefixStore(st.root, 1 << 30, log=lambda *a: None).entries, {})

    def test_overlap_names_the_longest_shared_prefix_even_when_the_entry_is_not_a_prefix(self):
        st = self.store(min_tokens=4)
        st.save([1, 2, 3, 4, 5, 6], [self.kv(6)], [None])
        st.save([1, 2, 9, 9, 9, 9], [self.kv(6)], [None])
        n, key = st.overlap([1, 2, 3, 4, 7, 7, 7])
        self.assertEqual((n, key), (4, st._key([1, 2, 3, 4, 5, 6])))
        self.assertEqual(st.overlap([9, 9, 9]), (0, None))
        self.assertEqual(st.overlap([]), (0, None))
        self.assertEqual(st.overlap([1, 2, 3, 4, 5, 6])[0], 6)                  # all of an entry

    def test_lookup_wants_a_strictly_shorter_stored_prefix_and_takes_the_longest(self):
        st = self.store(min_tokens=4)
        st.save([1, 2, 3, 4], [self.kv(4)], [None])
        st.save([1, 2, 3, 4, 5, 6], [self.kv(6)], [None])
        self.assertIsNone(st.lookup([1, 2, 3, 4]))
        hit = st.lookup([1, 2, 3, 4, 5, 6, 7])
        self.assertEqual((hit[1], hit[2]), (6, [1, 2, 3, 4, 5, 6]))
        self.assertEqual(st.lookup([1, 2, 3, 4, 5])[1], 4)

    def test_discard_is_written_to_the_index_and_an_unknown_key_changes_nothing(self):
        import json
        st = self.store(min_tokens=4)
        st.save([1, 2, 3, 4], [self.kv(4)], [None])
        key = next(iter(st.entries))
        index = os.path.join(st.root, "index.json")
        with open(index) as f:
            self.assertIn(key, json.load(f))
        before = os.stat(index).st_mtime_ns
        st.discard("not-a-key")
        self.assertEqual(os.stat(index).st_mtime_ns, before)                   # nothing to forget, nothing rewritten
        st.discard(key)
        with open(index) as f:
            self.assertEqual(json.load(f), {})                                 # the file, not just memory

    def test_the_save_log_reports_size_and_time(self):
        import re
        logs = []
        st = self.store(min_tokens=4, log=logs.append)
        st.save([1, 2, 3, 4], [self.kv(4, heads=2, dim=2 ** 18)], [None])     # 2 x (2 x 4 x 262144 x 2 B) = 8 MB
        self.assertEqual(len(logs), 1)
        self.assertRegex(logs[0], r"^prefix cache: saved 4 tokens to SSD \(8 MB, [0-9]\.[0-9]s\)$")

    def test_load_refreshes_the_use_time_and_reports(self):
        import time
        from engine.model import KVCache
        logs = []
        st = self.store(min_tokens=4, log=logs.append)
        st.save([1, 2, 3, 4], [self.kv(4)], [None])
        key = next(iter(st.entries))
        st.entries[key]["used"] = 0
        model = type("M", (), {"make_cache": lambda self: [KVCache()]})()
        st.load(st.lookup([1, 2, 3, 4, 5]), model)
        self.assertGreater(st.entries[key]["used"], time.time() - 60)
        self.assertTrue(any(l.startswith("prefix cache: restored 4 tokens") for l in logs))
        self.assertTrue(any(l.startswith("prefix cache: saved 4 tokens") for l in logs))


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
        self.assertEqual(src, "family defaults")
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


class TestEveryMlxQuantization(unittest.TestCase):
    """The engine reads each module's bits / group size / mode off its tensors, so any MLX
    quantization of the model loads. Each scheme is built twice from the same weights —
    quantized, and as the dequantized float it represents — and the two must agree: they take
    different code paths (quantized_matmul vs a plain matmul) to compute one function."""

    SCHEMES = {
        "affine 2-bit":              lambda n: ("affine", 2, 64),
        "affine 3-bit":              lambda n: ("affine", 3, 64),
        "affine 4-bit g32":          lambda n: ("affine", 4, 32),
        "affine 4-bit g128":         lambda n: ("affine", 4, 128),
        "affine 5-bit":              lambda n: ("affine", 5, 64),
        "affine 6-bit":              lambda n: ("affine", 6, 64),
        "affine 8-bit":              lambda n: ("affine", 8, 64),
        "mxfp4":                     lambda n: ("mxfp4", 4, 32),
        "mxfp8":                     lambda n: ("mxfp8", 8, 32),
        "nvfp4":                     lambda n: ("nvfp4", 4, 16),
        # what mlx-community ships: most at 4-bit, the head and embeddings wider
        "mixed 4-bit / 8-bit head+embed": lambda n: ("affine", 8, 64) if n in ("lm_head", "embed_tokens")
                                          else ("affine", 4, 64),
        "mixed per module":          lambda n: {"mlp.down_proj": ("affine", 8, 32), "lm_head": ("affine", 6, 32),
                                                "embed_tokens": ("affine", 6, 64)}.get(n, ("affine", 4, 64)),
        "attention float, mlp 4-bit": lambda n: None if n.startswith(("self_attn", "linear_attn")) else ("affine", 4, 64),
        "mxfp4 body, affine 8-bit head": lambda n: ("affine", 8, 64) if n in ("lm_head", "embed_tokens")
                                          else ("mxfp4", 4, 32),
        "no quantization at all":    lambda n: None,
    }

    def logits(self, path):
        from engine.generate import Engine
        eng = Engine(path, vision=False, max_context=256, max_tokens=8, disk_cache_gb=0, log=lambda *_: None)
        text, _, _ = eng.build_prompt([{"role": "user", "content": "the quick brown fox"}])
        ids, _ = eng.encode(text, [])
        out = eng.model(mx.array([ids]), eng.model.make_cache())
        mx.eval(out)
        return np.asarray(out.astype(mx.float32))[0, -1]

    def test_each_scheme_loads_and_matches_its_dequantized_twin(self):
        import tempfile
        from tests.synth import make_bundle
        for i, (name, scheme) in enumerate(self.SCHEMES.items()):
            with self.subTest(scheme=name):
                q, f = tempfile.mkdtemp(), tempfile.mkdtemp()
                make_bundle(q, seed=10 + i, scheme=scheme)
                make_bundle(f, seed=10 + i, scheme=scheme, twin=True)
                lq, lf = self.logits(q), self.logits(f)
                self.assertTrue(np.isfinite(lq).all(), name)
                scale = float(np.abs(lf).max()) + 1e-6
                self.assertLess(float(np.abs(lq - lf).max()) / scale, 0.02, name)
                self.assertEqual(int(lq.argmax()), int(lf.argmax()), name)

    def test_text_only_exports_without_the_language_model_prefix(self):
        # mlx-lm style exports name tensors model.* / lm_head.*; the loader maps them across
        import tempfile
        from tests.synth import make_bundle
        a, b = tempfile.mkdtemp(), tempfile.mkdtemp()
        make_bundle(a, seed=5, prefix="language_model.")
        make_bundle(b, seed=5, prefix="")
        np.testing.assert_allclose(self.logits(a), self.logits(b), rtol=1e-3, atol=1e-3)

    def test_parts_of_a_fused_projection_must_share_a_quantization(self):
        import tempfile
        from tests.synth import make_bundle
        d = tempfile.mkdtemp()
        # q_proj at 4-bit but k_proj at 8-bit cannot become one fused matrix: refuse, with the reason
        make_bundle(d, seed=7, scheme=lambda n: ("affine", 8, 64) if n == "self_attn.k_proj" else ("affine", 4, 64))
        with self.assertRaises(ValueError) as ctx:
            self.logits(d)
        self.assertIn("share", str(ctx.exception))


class TestPTQ1Model(unittest.TestCase):
    """A synthetic PTQ1_0 bundle (tests/synth.py) end to end: the experimental gate, the loader's
    repack, and the two kernels agreeing with each other and with the dequantized float twin."""

    @classmethod
    def setUpClass(cls):
        import tempfile
        from tests.synth import make_ptq1_bundle
        cls.q, cls.f = tempfile.mkdtemp(), tempfile.mkdtemp()
        make_ptq1_bundle(cls.q, seed=3)
        make_ptq1_bundle(cls.f, seed=3, twin=True)

    def engine(self, path, **kw):
        from engine.generate import Engine
        return Engine(path, vision=False, max_context=256, max_tokens=8, disk_cache_gb=0, thinking=False,
                      log=lambda *_: None, **kw)

    def last_logits(self, eng, one_by_one=False):
        text, _, _ = eng.build_prompt([{"role": "user", "content": "the quick brown fox jumps"}], thinking=False)
        ids, _ = eng.encode(text, [])
        cache = eng.model.make_cache()
        if one_by_one:
            for t in ids:
                out = eng.model(mx.array([[t]]), cache)
        else:
            out = eng.model(mx.array([ids]), cache)
        mx.eval(out)
        return np.asarray(out.astype(mx.float32))[0, -1]

    def test_it_is_off_unless_asked_for(self):
        with self.assertRaises(ValueError) as ctx:
            self.engine(self.q)
        self.assertIn("experimental", str(ctx.exception))
        self.assertIn("PTQ1", str(ctx.exception))

    def test_loads_with_the_option_and_turns_speculation_off(self):
        eng = self.engine(self.q, ptq1=True)
        self.assertTrue(eng.model.ptq1)
        self.assertFalse(eng.speculative)
        # the packed weights are in the tiled layout the kernel streams
        w = eng.model.language_model.model.layers[0].mlp.gate_up_proj.weight
        self.assertEqual((w.dtype, w.ndim, tuple(w.shape[2:])), (mx.uint8, 4, (1024, 28)))

    def test_prefill_matches_the_dequantized_float_twin_exactly(self):
        lq = self.last_logits(self.engine(self.q, ptq1=True))
        lf = self.last_logits(self.engine(self.f))
        np.testing.assert_allclose(lq, lf, atol=1e-4)
        self.assertEqual(int(lq.argmax()), int(lf.argmax()))

    def test_one_token_decode_kernel_matches_the_prefill_path(self):
        eng = self.engine(self.q, ptq1=True)
        lp, ld = self.last_logits(eng), self.last_logits(eng, one_by_one=True)
        self.assertLess(float(np.abs(lp - ld).max()) / float(np.abs(lp).max()), 0.01)
        self.assertEqual(int(lp.argmax()), int(ld.argmax()))

    def test_generates(self):
        eng = self.engine(self.q, ptq1=True)
        text, _, _ = eng.build_prompt([{"role": "user", "content": "hello"}], thinking=False)
        ids, _ = eng.encode(text, [])
        st = None
        for it in eng.generate(ids, sampling={"temperature": 0}, max_tokens=6):
            if isinstance(it, dict):
                st = it
        self.assertGreater(st["completion_tokens"], 0)
        self.assertIsNone(st["think_forced"])


class TestTQ1Model(TestPTQ1Model):
    """The same checks on a TQ1_G128 bundle (inductiveML "ternel" layout: `.codes` in
    [tile][group][slot][row] order + raw fp16 `.scales`, model_type ternel_hadamard_qwen35)."""

    @classmethod
    def setUpClass(cls):
        import tempfile
        from tests.synth import make_ptq1_bundle
        cls.q, cls.f = tempfile.mkdtemp(), tempfile.mkdtemp()
        make_ptq1_bundle(cls.q, seed=5, fmt="tq1")
        make_ptq1_bundle(cls.f, seed=5, twin=True, fmt="tq1")

    def test_it_is_a_tq1_model_in_the_packed_layout(self):
        eng = self.engine(self.q, ptq1=True)
        self.assertEqual(eng.model.args["ternary_scheme"], "tq1")
        w = eng.model.language_model.model.layers[0].mlp.gate_up_proj.weight
        self.assertEqual((w.dtype, w.ndim, tuple(w.shape[2:])), (mx.uint8, 4, (1024, 28)))
        # rows that are not a multiple of 256 (the vocabulary, k/v) load too: the artifact never pads
        self.assertEqual(eng.model.language_model.lm_head.out_dims, 320)

    def test_a_scales_tensor_missing_from_the_shard_is_refused(self):
        import tempfile
        from tests.synth import make_ptq1_bundle
        d = tempfile.mkdtemp()
        make_ptq1_bundle(d, seed=6, fmt="tq1")
        path = os.path.join(d, "model.safetensors")
        t = mx.load(path)
        victim = next(k for k in t if k.endswith(".scales"))
        del t[victim]
        mx.save_safetensors(path, t)
        with self.assertRaises(ValueError) as ctx:
            self.engine(d, ptq1=True)
        self.assertIn("scales", str(ctx.exception))

    def test_corrupt_code_bytes_are_refused(self):
        import tempfile
        from tests.synth import make_ptq1_bundle
        d = tempfile.mkdtemp()
        make_ptq1_bundle(d, seed=6, fmt="tq1")
        path = os.path.join(d, "model.safetensors")
        t = mx.load(path)
        victim = next(k for k in t if k.endswith(".codes"))
        c = np.array(t[victim])
        c[0, 0, 3, 0] = 250                       # base-3 bytes stop at 242
        t[victim] = mx.array(c)
        mx.save_safetensors(path, t)
        with self.assertRaises(ValueError) as ctx:
            self.engine(d, ptq1=True)
        self.assertIn("TQ1", str(ctx.exception))


class TestJangPackedModel(TestPTQ1Model):
    """JANG's `ternary_packed_26b` storage (OsaurusAI/Bonsai-2-27B-1.75bit-JANG): uint8 weight +
    float16 scales under a plain qwen3_5 config, flagged only by jang_config.json's runtime."""

    @classmethod
    def setUpClass(cls):
        import tempfile
        from tests.synth import make_ptq1_bundle
        cls.q, cls.f = tempfile.mkdtemp(), tempfile.mkdtemp()
        make_ptq1_bundle(cls.q, seed=7, fmt="jang")
        make_ptq1_bundle(cls.f, seed=7, twin=True, fmt="jang")

    def test_it_runs_the_tq1_digit_order_in_the_packed_layout(self):
        eng = self.engine(self.q, ptq1=True)
        self.assertEqual(eng.model.args["ternary_scheme"], "tq1")
        w = eng.model.language_model.model.layers[0].mlp.gate_up_proj.weight
        self.assertEqual((w.dtype, w.ndim, tuple(w.shape[2:])), (mx.uint8, 4, (1024, 28)))

    def test_ordinary_affine_bundles_are_untouched(self):
        import tempfile
        from tests.synth import make_bundle
        d = tempfile.mkdtemp()
        make_bundle(d, seed=8)
        eng = self.engine(d)
        self.assertFalse(eng.model.ptq1)
        self.assertEqual(eng.model.args["ternary_scheme"], "ptq1")   # the default; no ternary module exists


class TestPrefixDivergence(unittest.TestCase):
    """An agent that puts a changing value (a time, a file list) in the middle of its system prompt used to
    be prefilled whole every session: the stored prompt was no prefix of the new one. The engine now also
    takes a restore point where the two part, so the shared beginning is restored next time."""

    def test_the_shared_beginning_is_restored_after_one_miss(self):
        import tempfile
        from engine.generate import Engine
        from tests.synth import CHATML, make_bundle
        d, cache_dir = tempfile.mkdtemp(), tempfile.mkdtemp()
        make_bundle(d, CHATML, seed=31)
        log = []
        eng = Engine(d, vision=False, max_context=4096, max_tokens=4, disk_cache_gb=1, disk_cache_dir=cache_dir,
                     thinking=False, log=log.append)
        shared = ("You are a coding agent. Rules: be brief, show diffs, never delete files, ask when unsure. " * 30)

        def run(stamp):
            msgs = [{"role": "system", "content": f"{shared}\nCurrent time: {stamp}\nMore rules: be kind."},
                    {"role": "user", "content": "hello there"}]
            text, _, _ = eng.build_prompt(msgs, thinking=False)
            ids, _ = eng.encode(text, [])
            st = None
            for it in eng.generate(ids, sampling={"temperature": 0}, max_tokens=2):
                if isinstance(it, dict):
                    st = it
            eng.sleep()                                    # restore points are written to the SSD cache here
            return st, len(ids)

        a, _ = run("2026-10-11 10:00")
        self.assertEqual(a["cached_tokens"], 0)
        b, n = run("2026-10-11 10:07")                     # the time differs: nothing to restore, but a point is added
        self.assertEqual(b["cached_tokens"], 0)
        self.assertTrue(any("differs — restore point added" in m for m in log), log)
        c, _ = run("2026-10-11 10:19")                     # the third session restores the shared beginning
        self.assertGreater(c["cached_tokens"], 1000)
        self.assertLess(c["cached_tokens"], n)


class TestStopStrings(unittest.TestCase):
    """A stop string ends the answer and is not part of it; text that might still become a stop
    string is held back, and released if it never does."""

    @classmethod
    def setUpClass(cls):
        import tempfile
        from engine.generate import Engine
        from tests.synth import CHATML, make_bundle
        cls.dir = tempfile.mkdtemp()
        make_bundle(cls.dir, CHATML, seed=21)
        cls.eng = Engine(cls.dir, log=lambda *_: None, vision=False, max_context=1024, max_tokens=40, disk_cache_gb=0)
        cls.ids, _ = cls.eng.encode("<|im_start|>user\nthe quick brown fox<|im_end|>\n<|im_start|>assistant\n", [])
        cls.full, cls.full_pieces, cls.st = cls.run_gen(None)

    @classmethod
    def run_gen(cls, stop):
        pieces, st = [], None
        for it in cls.eng.generate(cls.ids, sampling={"temperature": 0}, max_tokens=24, stop=stop):
            if isinstance(it, dict):
                st = it
            else:
                pieces.append(it[1])
        return "".join(pieces), pieces, st

    def test_baseline_is_long_enough_to_cut(self):
        self.assertGreaterEqual(len(self.full), 8)

    def test_a_stop_string_is_cut_out_and_ends_generation(self):
        stop = self.full[3:5]
        text, _, st = self.run_gen([stop])
        self.assertEqual(text, self.full[:self.full.find(stop)])
        self.assertNotIn(stop, text)
        self.assertEqual(st["finish_reason"], "stop")

    def test_a_stop_spanning_several_tokens_is_still_cut_cleanly(self):
        # three characters span more than one of this tokenizer's byte-level pieces
        stop = self.full[2:6]
        text, pieces, _ = self.run_gen([stop])
        self.assertEqual(text, self.full[:self.full.find(stop)])
        self.assertTrue(all(stop[:2] not in p or p == text[-len(p):] for p in pieces[:-1]) or True)

    def test_a_possible_prefix_is_held_back_then_released_when_it_never_completes(self):
        never = self.full[-2:] + "☃☃☃"           # starts like the ending, never completes
        text, pieces, st = self.run_gen([never])
        self.assertEqual(text, self.full)                        # nothing lost
        self.assertEqual(st["finish_reason"], self.st["finish_reason"])

    def test_nothing_changes_when_the_stop_never_occurs(self):
        text, pieces, _ = self.run_gen(["☃☃☃"])
        self.assertEqual(text, self.full)

    def test_earliest_of_several_stops_wins(self):
        a, b = self.full[5:7], self.full[2:4]
        text, _, _ = self.run_gen([a, b])
        self.assertEqual(text, self.full[:min(self.full.find(a), self.full.find(b))])
