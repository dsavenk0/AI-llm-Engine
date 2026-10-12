"""The engine's own GGUF path: reader → repacker → MLX decoder → tokenizer → engine (no llama.cpp)."""
import os
import tempfile
import unittest

import numpy as np

from tests import gguf_synth as S


class Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.mkdtemp()
        cls.f16 = os.path.join(cls.dir, "tiny-f16.gguf")
        cls.q8 = os.path.join(cls.dir, "tiny-q8.gguf")
        cls.v16 = S.write(cls.f16, seed=1, quant="F16")
        cls.v8 = S.write(cls.q8, seed=1, quant="Q8_0")


class TestSupportProbe(Base):
    def test_a_plain_llama_file_qualifies_and_the_reasons_for_refusal_are_named(self):
        from engine import gguf
        self.assertEqual(gguf.native_support(self.f16), (True, ""))
        self.assertEqual(gguf.native_support(self.q8), (True, ""))
        moe = os.path.join(self.dir, "moe.gguf")
        S.write(moe, extra_meta={"llama.expert_count": 8})
        ok, why = gguf.native_support(moe)
        self.assertFalse(ok)
        self.assertIn("experts", why)
        odd = os.path.join(self.dir, "odd.gguf")
        S.write(odd, arch="gemma3")
        self.assertIn("architecture", gguf.native_support(odd)[1])
        iq = os.path.join(self.dir, "iq.gguf")
        S.write(iq, tensor_type_override=("blk.0.ffn_up.weight", 20))           # IQ4_NL
        self.assertIn("quantization", gguf.native_support(iq)[1])
        spm = os.path.join(self.dir, "spm.gguf")
        S.write(spm, extra_meta={"tokenizer.ggml.model": "llama"})
        self.assertIn("tokenizer", gguf.native_support(spm)[1])
        self.assertFalse(gguf.native_support(os.path.join(self.dir, "missing.gguf"))[0])
        for key, value, word in (("llama.attention.sliding_window", 4096, "sliding"),
                                 ("llama.rope.scaling.type", "linear", "rope"),
                                 ("llama.rope.scaling.type", "yarn", "rope"),
                                 ("llama.rope.scaling.type", "llama3", "frequency")):   # llama3 needs rope_freqs.weight
            path = os.path.join(self.dir, f"{word}-{value}.gguf")
            S.write(path, extra_meta={key: value})
            ok, why = gguf.native_support(path)
            self.assertFalse(ok, (key, value))
            self.assertIn(word, why)
            from engine import dense
            with self.assertRaises(dense.Unsupported):                                   # the builder agrees
                dense.build(path)

    def test_the_probe_and_the_tokenizer_agree_on_the_known_pre_tokenizers(self):
        from engine import gguf, gguf_tokenizer
        self.assertEqual(set(gguf.NATIVE_PRE), set(gguf_tokenizer.PRE))


class TestDecoder(Base):
    def logits(self, path, ids):
        import mlx.core as mx
        from engine import dense
        model, cfg, gf = dense.build(path)
        out = model(mx.array([ids]), model.make_cache())
        return model, np.asarray(out[0].astype(mx.float32))

    def test_f16_file_matches_an_independent_numpy_forward(self):
        ids = [5, 40, 7, 257, 12, 99, 3, 260]
        _, got = self.logits(self.f16, ids)
        want = S.reference_logits(self.v16, ids)
        np.testing.assert_allclose(got, want, atol=0.08, rtol=0)
        self.assertTrue((got.argmax(-1) == want.argmax(-1)).mean() > 0.85)

    def test_qwen2_biases_and_qwen3_qk_norm_with_neox_rope_match_the_reference(self):
        ids = [5, 40, 7, 257, 12, 99, 3, 260, 14, 15]
        for arch in ("qwen2", "qwen3"):
            path = os.path.join(self.dir, f"{arch}.gguf")
            values = S.write(path, seed=9, quant="F16", arch=arch)
            _, got = self.logits(path, ids)
            want = S.reference_logits(values, ids, arch=arch)
            np.testing.assert_allclose(got, want, atol=0.08, rtol=0, err_msg=arch)
            self.assertTrue((got.argmax(-1) == want.argmax(-1)).mean() > 0.8, arch)
        # the variants really differ (so the matches above are not a coincidence of shared code): the same
        # Qwen3 weights read with llama's rope and no q/k norm give other logits
        qwen3 = S.write(os.path.join(self.dir, "q3.gguf"), seed=9, quant="F16", arch="qwen3")
        self.assertGreater(float(np.abs(S.reference_logits(qwen3, ids, "qwen3")
                                        - S.reference_logits(qwen3, ids, "llama")).max()), 0.05)

    def test_q8_0_repacks_to_8_bit_and_matches_the_reference(self):
        ids = [9, 8, 7, 6, 5, 4]
        _, got = self.logits(self.q8, ids)
        want = S.reference_logits(self.v8, ids)
        np.testing.assert_allclose(got, want, atol=0.08, rtol=0)

    def test_prefill_then_one_token_steps_equal_one_full_pass(self):
        import mlx.core as mx
        from engine import dense
        model, _, _ = dense.build(self.f16)
        ids = list(range(10, 34))
        full = np.asarray(model(mx.array([ids]), model.make_cache())[0, -1].astype(mx.float32))
        cache = model.make_cache()
        model(mx.array([ids[:12]]), cache)
        for t in ids[12:-1]:
            model(mx.array([[t]]), cache)
        step = np.asarray(model(mx.array([[ids[-1]]]), cache)[0, -1].astype(mx.float32))
        np.testing.assert_allclose(step, full, atol=0.05)

    def test_trimming_the_cache_forgets_the_newest_tokens(self):
        import mlx.core as mx
        from engine import dense
        model, _, _ = dense.build(self.f16)
        ids = list(range(10, 30))
        cache = model.make_cache()
        model(mx.array([ids]), cache)
        for c in cache:
            c.trim(5)
        again = np.asarray(model(mx.array([ids[-5:]]), cache)[0, -1].astype(mx.float32))
        full = np.asarray(model(mx.array([ids]), model.make_cache())[0, -1].astype(mx.float32))
        np.testing.assert_allclose(again, full, atol=0.05)

    def test_a_steering_direction_changes_logits_and_alpha_zero_does_not(self):
        import mlx.core as mx
        from engine import dense, steering
        model, _, _ = dense.build(self.f16)
        ids = [4, 5, 6, 7, 8]
        base = np.asarray(model(mx.array([ids]), model.make_cache())[0].astype(mx.float32))
        d = mx.array(np.random.default_rng(0).standard_normal(S.HIDDEN).astype(np.float32))
        d = d / mx.linalg.norm(d)
        model.steer = steering.projector(d, 0.0)
        zero = np.asarray(model(mx.array([ids]), model.make_cache())[0].astype(mx.float32))
        model.steer = steering.projector(d, 1.0)
        on = np.asarray(model(mx.array([ids]), model.make_cache())[0].astype(mx.float32))
        np.testing.assert_allclose(zero, base, atol=1e-3)
        self.assertGreater(float(np.abs(on - base).max()), 1e-2)


class TestTokenizer(Base):
    def setUp(self):
        from engine import gguf, gguf_tokenizer
        self.tok = gguf_tokenizer.GGUFTokenizer(self.f16, gguf.read_header_full(self.f16))

    def test_round_trip_specials_and_template(self):
        text = "the thin cat sat in a hat, привет"
        ids = self.tok.encode(text)
        self.assertEqual(self.tok.decode(ids), text)
        ids = self.tok.encode("<|im_start|>user\nhi<|im_end|>")
        self.assertEqual(ids[0], self.tok.encode("<|im_start|>")[0])         # a special is one token
        self.assertEqual(len(self.tok.encode("<|im_start|>")), 1)
        prompt = self.tok.apply_chat_template([{"role": "user", "content": "hi"}])
        self.assertEqual(prompt, "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n")
        self.assertIn(self.tok.eos_id, self.tok.eos_token_ids)

    def test_the_pre_tokenizer_regex_decides_what_may_merge(self):
        # llama-bpe keeps "!a" together (punctuation + letters is one pre-token), so the merge "! a" applies; a second,
        # GPT-2-style split after it would cut it into "!" and "a" and the merge could never happen
        ids = self.tok.encode("x!a")
        self.assertEqual(len(ids), 2)
        self.assertEqual(self.tok.decode(ids), "x!a")

    def test_no_phantom_space_is_added_in_front(self):
        self.assertEqual(self.tok.decode(self.tok.encode("hi")), "hi")
        self.assertNotEqual(self.tok.encode(" hi"), self.tok.encode("hi"))

    def test_merges_apply_like_bpe(self):
        base = self.tok.encode("t")
        self.assertEqual(len(self.tok.encode(" the")), 1)                       # Ġ t h e → Ġthe via the merges
        self.assertEqual(len(base), 1)


class TestTokenizerFeatures(Base):
    def make(self, **meta):
        from engine import gguf, gguf_tokenizer
        path = os.path.join(self.dir, f"tok{len(os.listdir(self.dir))}.gguf")
        S.write(path, extra_meta=meta)
        return gguf_tokenizer.GGUFTokenizer(path, gguf.read_header_full(path))

    def test_template_helpers_tojson_raise_exception_and_time(self):
        tok = self.make(**{"tokenizer.chat_template":
                           "{{ {'a': 'привет'} | tojson }}|{{ bos_token }}|{{ eos_token }}|{{ strftime_now('%Y') }}"})
        out = tok.apply_chat_template([{"role": "user", "content": "x"}], add_generation_prompt=False)
        self.assertIn('"привет"', out)                      # not \u-escaped: ensure_ascii is off
        self.assertIn("|<|bos|>|<|im_end|>|", out)
        self.assertRegex(out, r"\|20\d\d$")
        bad = self.make(**{"tokenizer.chat_template": "{{ raise_exception('no system role') }}"})
        with self.assertRaisesRegex(ValueError, "no system role"):
            bad.apply_chat_template([{"role": "user", "content": "x"}])

    def test_no_template_and_tokenize_flag(self):
        tok = self.make(**{"tokenizer.chat_template": ""})
        with self.assertRaisesRegex(ValueError, "no chat template"):
            tok.apply_chat_template([{"role": "user", "content": "x"}])
        with_t = self.make()
        ids = with_t.apply_chat_template([{"role": "user", "content": "hi"}], tokenize=True)
        self.assertEqual(with_t.decode(ids), with_t.apply_chat_template([{"role": "user", "content": "hi"}]))

    def test_tool_call_markers_follow_the_template(self):
        plain = self.make()
        self.assertFalse(plain.has_tool_calling)
        self.assertIsNone(plain.tool_call_start)
        tools = self.make(**{"tokenizer.chat_template": "{{ '<tool_call>' }}x{{ '</tool_call>' }}"})
        self.assertTrue(tools.has_tool_calling)
        self.assertEqual((tools.tool_call_start, tools.tool_call_end), ("<tool_call>", "</tool_call>"))
        self.assertEqual(tools.tool_parser('{"name": "f", "arguments": {"a": 1}}'), [{"name": "f", "arguments": {"a": 1}}])
        self.assertEqual(tools.tool_parser('[{"name": "f"}, {"name": "g"}]'), [{"name": "f"}, {"name": "g"}])

    def test_end_tokens_bos_eos_and_bad_ids(self):
        tok = self.make()
        end = tok.encode("<|im_end|>")[0]
        self.assertEqual((tok.bos_token, tok.eos_token), ("<|bos|>", "<|im_end|>"))
        self.assertEqual(tok.eos_token_ids, [end])
        start = tok.encode("<|im_start|>")[0]
        extra = self.make(**{"tokenizer.ggml.eot_token_id": start})       # a model that names a separate end-of-turn
        self.assertEqual(extra.eos_token_ids, sorted({end, start}))
        out_of_range = self.make(**{"tokenizer.ggml.bos_token_id": 10 ** 6})
        self.assertEqual(out_of_range.bos_token, "")                     # an id past the vocabulary is no token
        self.assertEqual(self.make().vocab_size, len(S.vocab()[0]))

    def test_unknown_tokenizers_are_refused_not_guessed(self):
        from engine import gguf_tokenizer
        for key, value in (("tokenizer.ggml.model", "llama"), ("tokenizer.ggml.pre", "some-new-pre")):
            with self.assertRaises(gguf_tokenizer.Unsupported):
                self.make(**{key: value})

    def test_only_control_and_user_defined_tokens_are_special(self):
        tokens, types, _ = S.vocab()
        for ty, expect_one in ((3, True), (4, True), (1, False), (2, False)):
            tok = self.make(**{"tokenizer.ggml.token_type": [ty if t.startswith("<|") else y
                                                              for t, y in zip(tokens, types)]})
            self.assertEqual(len(tok.encode("<|im_start|>")) == 1, expect_one, ty)

    def test_a_longer_type_table_than_vocabulary_is_ignored_past_the_end(self):
        tokens, types, _ = S.vocab()
        tok = self.make(**{"tokenizer.ggml.token_type": list(types) + [3] * 5})
        self.assertEqual(tok.vocab_size, len(tokens))
        self.assertEqual(len(tok.encode("<|im_start|>")), 1)

    def test_a_file_without_merges_or_tokens_is_refused(self):
        from engine import gguf_tokenizer
        for key in ("tokenizer.ggml.merges", "tokenizer.ggml.tokens"):
            with self.assertRaises(gguf_tokenizer.Unsupported, msg=key):
                self.make(**{key: []})

    def test_eos_id_at_the_edge_of_the_vocabulary_names_no_token(self):
        n = len(S.vocab()[0])
        edge = self.make(**{"tokenizer.ggml.eos_token_id": n})
        self.assertEqual(edge.eos_token, "")
        last = self.make(**{"tokenizer.ggml.eos_token_id": n - 1})
        self.assertEqual(last.eos_token, S.vocab()[0][n - 1])
        self.assertEqual(self.make(**{"tokenizer.ggml.bos_token_id": n}).bos_token, "")

    def test_tools_reach_the_template_and_an_empty_list_is_none(self):
        tok = self.make(**{"tokenizer.chat_template":
                           "{% if tools %}T{{ tools | length }}{% else %}none{% endif %}"})
        msgs = [{"role": "user", "content": "x"}]
        self.assertEqual(tok.apply_chat_template(msgs, tools=[{"name": "a"}, {"name": "b"}]), "T2")
        self.assertEqual(tok.apply_chat_template(msgs, tools=[]), "none")
        self.assertEqual(tok.apply_chat_template(msgs), "none")

    def test_template_whitespace_follows_jinja_trim_and_lstrip(self):
        tok = self.make(**{"tokenizer.chat_template": "A\n  {% if true %}\nB\n  {% endif %}\nC"})
        self.assertEqual(tok.apply_chat_template([{"role": "user", "content": "x"}]), "A\nB\nC")

    def test_extra_template_variables_and_loop_control(self):
        tok = self.make(**{"tokenizer.chat_template":
                           "{% for m in messages %}{% if loop.index > 1 %}{% break %}{% endif %}{{ m.content }}"
                           "{% endfor %}{{ extra }}"})
        out = tok.apply_chat_template([{"role": "user", "content": "a"}, {"role": "user", "content": "b"}], extra="!")
        self.assertEqual(out, "a!")

    def test_tojson_keeps_unicode_and_takes_indent_and_sort_keys(self):
        tok = self.make(**{"tokenizer.chat_template":
                           "{{ 'é' | tojson }}{{ {'b': 1, 'a': 2} | tojson(sort_keys=true) }}"
                           "{{ [1] | tojson(indent=1) }}"})
        out = tok.apply_chat_template([{"role": "user", "content": "x"}])
        self.assertEqual(out, '"é"{"a": 2, "b": 1}[\n 1\n]')

    def test_decode_can_hide_special_tokens(self):
        tok = self.make()
        ids = tok.encode("<|im_start|>hi<|im_end|>")
        self.assertEqual(tok.decode(ids), "<|im_start|>hi<|im_end|>")
        self.assertEqual(tok.decode(ids, skip_special_tokens=True), "hi")


class TestEngine(Base):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        from engine.gguf_engine import GGUFEngine
        cls.engine = GGUFEngine(cls.q8, max_context=512, max_tokens=64, log=lambda *a: None)

    def run_gen(self, ids, **kw):
        out = []
        for item in self.engine.generate(ids, **kw):
            if isinstance(item, dict):
                return out, item
            out.append(item)
        return out, {}

    def test_generates_with_the_shared_loop_and_reuses_the_prefix(self):
        ids = self.engine.tok.encode("<|im_start|>user\nthe cat<|im_end|>\n<|im_start|>assistant\n")
        out, st = self.run_gen(ids, max_tokens=12, sampling={"temperature": 0.0})
        self.assertGreater(st["completion_tokens"], 0)
        self.assertEqual(st["cached_tokens"], 0)
        out2, st2 = self.run_gen(ids + [10, 11], max_tokens=4, sampling={"temperature": 0.0})
        self.assertGreaterEqual(st2["cached_tokens"], len(ids) - 1)             # the shared prefix came from the cache
        # the same prompt again is deterministic at temperature 0
        a, _ = self.run_gen(ids, max_tokens=8, sampling={"temperature": 0.0})
        b, _ = self.run_gen(ids, max_tokens=8, sampling={"temperature": 0.0})
        self.assertEqual([t for t, _ in a], [t for t, _ in b])

    def test_speculation_never_changes_the_greedy_output_and_drafts_on_repeats(self):
        from engine.gguf_engine import GGUFEngine
        plain = GGUFEngine(self.q8, max_context=1024, max_tokens=96, speculative=False, log=lambda *a: None)
        spec = GGUFEngine(self.q8, max_context=1024, max_tokens=96, speculative=True, log=lambda *a: None)
        unit = spec.tok.encode("the cat in the hat sat in a hat. ")
        for ids in (unit * 6, unit[:5] * 9 + unit):                    # texts that repeat, so lookups find drafts
            greedy = {"temperature": 0.0}
            a = [t for t, _ in list(plain.generate(ids, max_tokens=60, sampling=greedy))[:-1] if t is not None]
            out = list(spec.generate(ids, max_tokens=60, sampling=greedy))
            b = [t for t, _ in out[:-1] if t is not None]
            self.assertEqual(a, b)
        st = out[-1]
        self.assertGreater(st["completion_tokens"], 0)
        if st["verifies"]:                                             # drafts were verified at least once
            self.assertIsNotNone(st["draft_accept"])
        # the cache stays usable after a verify with rejections: a follow-up request still matches a fresh run
        again = [t for t, _ in list(spec.generate(unit * 6 + [9], max_tokens=10, sampling={"temperature": 0.0}))[:-1]
                 if t is not None]
        fresh = [t for t, _ in list(plain.generate(unit * 6 + [9], max_tokens=10, sampling={"temperature": 0.0}))[:-1]
                 if t is not None]
        self.assertEqual(again, fresh)

    def test_a_stop_inside_an_accepted_draft_leaves_the_cache_exactly_as_long_as_the_fed_tokens(self):
        from engine.gguf_engine import GGUFEngine
        sharp = os.path.join(self.dir, "sharp.gguf")
        S.write(sharp, seed=4, quant="Q8_0", sharp=True)
        plain = GGUFEngine(sharp, max_context=1024, max_tokens=96, speculative=False, log=lambda *a: None)
        spec = GGUFEngine(sharp, max_context=1024, max_tokens=96, speculative=True, log=lambda *a: None)
        unit = spec.tok.encode("the cat in the hat sat in a hat. ")
        ids = unit * 6
        list(plain.generate(ids, max_tokens=40, sampling={"temperature": 0.0}))
        full = plain.cache_tokens[len(ids):]           # every token that was fed, including ones with no text yet
        self.assertGreater(len(full), 12)
        from unittest import mock

        class Oracle:
            """A drafter that proposes the model's own continuation, so every draft is accepted."""
            def __init__(self, tokens, nmin=2, nmax=4):
                self.n, self.toks = len(tokens), list(tokens)

            def add(self, t):
                self.toks.append(t)

            def propose(self, k):
                g = len(self.toks) - self.n
                return full[g:g + k]
        with mock.patch("engine.compat.Drafter", Oracle):
            for i in range(1, 6):                      # the stop is the i-th reply token, inside the first draft block
                stop = spec.tok.decode([full[i]])
                out = list(spec.generate(ids, max_tokens=60, sampling={"temperature": 0.0}, stop=[stop]))
                self.assertEqual(out[-1]["finish_reason"], "stop")
                self.assertGreater(out[-1]["verifies"], 0)                # the speculative path really ran
                self.assertGreater(out[-1]["draft_accept"] or 0, 0.5)     # …and drafts were accepted
                self.assertEqual(spec.cache[0].offset, len(spec.cache_tokens), f"stop at reply token {i}")
                # and the next request is still right: same output as a fresh engine
                list(spec.generate(ids, max_tokens=12, sampling={"temperature": 0.0}))
                self.assertEqual(spec.cache_tokens[len(ids):][:12], full[:12])

    def test_steering_is_applied_when_asked_for(self):
        import mlx.core as mx
        import tempfile
        from engine.gguf_engine import GGUFEngine
        d = np.random.default_rng(1).standard_normal(S.HIDDEN).astype(np.float32)
        p = os.path.join(tempfile.mkdtemp(), "dir.safetensors")
        mx.save_safetensors(p, {"direction": mx.array(d)})
        eng = GGUFEngine(self.q8, max_context=512, max_tokens=64, steer=(p, 1.0), log=lambda *a: None)
        self.assertIsNotNone(eng.model.steer)
        self.assertIsNone(self.engine.model.steer)


class TestDecoderDetails(Base):
    """The corners the plain llama files above do not reach: rope frequency factors, the multipliers other
    families use, the byte accounting and the growth of the key/value cache."""

    def build(self, path):
        from engine import dense
        return dense.build(path)[0]

    def fwd(self, model, ids, cache=None):
        import mlx.core as mx
        return np.asarray(model(mx.array([ids]), cache or model.make_cache())[0].astype(mx.float32))

    def test_llama3_frequency_factors_rotate_like_the_reference_and_hold_across_steps(self):
        import mlx.core as mx
        ids = [5, 40, 7, 257, 12, 99, 3, 260, 14, 15]
        factors = np.linspace(1.0, 8.0, S.HEAD_DIM // 2)
        path = os.path.join(self.dir, "l3.gguf")
        values = S.write(path, seed=4, quant="F16", rope_factors=factors)
        model = self.build(path)
        got = self.fwd(model, ids)
        np.testing.assert_allclose(got, S.reference_logits(values, ids), atol=0.08, rtol=0)
        plain = {k: v for k, v in values.items() if k != "rope_freqs.weight"}
        self.assertGreater(float(np.abs(S.reference_logits(plain, ids) - got).max()), 0.02)   # factors matter
        cache = model.make_cache()                                  # an offset must reach the factor path too
        model(mx.array([ids[:6]]), cache)
        step = self.fwd(model, ids[6:], cache)
        np.testing.assert_allclose(step[-1], got[-1], atol=0.08)

    def test_llama3_scaling_without_its_table_is_refused(self):
        from engine import dense
        path = os.path.join(self.dir, "l3-notable.gguf")
        S.write(path, seed=4, quant="F16", extra_meta={"llama.rope.scaling.type": "llama3"})
        with self.assertRaises(dense.Unsupported):
            dense.build(path)
        path = os.path.join(self.dir, "yarn.gguf")
        S.write(path, seed=4, quant="F16", extra_meta={"llama.rope.scaling.type": "yarn"})
        with self.assertRaises(dense.Unsupported):
            dense.build(path)

    def test_embedding_and_logit_multipliers(self):
        model = self.build(self.f16)
        ids = [4, 5, 6, 7]
        base = self.fwd(model, ids)
        model.cfg.logit_scale = 2.0
        np.testing.assert_allclose(self.fwd(model, ids), base / 2.0, atol=0.02)
        model.cfg.logit_scale = 1.0
        model.cfg.embed_scale = 3.0
        self.assertGreater(float(np.abs(self.fwd(model, ids) - base).max()), 0.05)

    def test_byte_accounting_counts_weights_scales_and_the_embedding(self):
        import mlx.core as mx
        for path, bits in ((self.f16, 0), (self.q8, 8)):
            model = self.build(path)
            want = sum(m.nbytes for L in model.layers for m in L.params())
            e = model.embed
            want += e.w.nbytes + (e.scales.nbytes + e.biases.nbytes if e.bits else 0) + model.head.nbytes
            self.assertEqual(model.nbytes, want)
            self.assertEqual(model.embed.bits, bits)
            q = model.layers[0].q
            self.assertEqual(q.nbytes, q.w.nbytes + (q.scales.nbytes + q.biases.nbytes if q.bits else 0))
        self.assertGreater(q.biases.nbytes, 0)                     # the 8-bit file really has scales and biases

    def test_a_qwen2_file_counts_the_bias_bytes_and_reports_its_flags(self):
        from engine import dense
        for arch, flags in (("llama", (False, False)), ("qwen2", (True, False)), ("qwen3", (False, True))):
            path = os.path.join(self.dir, f"flags-{arch}.gguf")
            S.write(path, seed=2, quant="F16", arch=arch)
            model, cfg, _ = dense.build(path)
            self.assertEqual((cfg.qkv_bias, cfg.qk_norm), flags, arch)
            self.assertEqual((cfg.vocab, cfg.context, cfg.n_layer, cfg.n_embd), (len(S.vocab()[0]), 256, S.LAYERS, S.HIDDEN))
            q = model.layers[0].q
            self.assertEqual(q.bias is not None, arch == "qwen2")
            self.assertEqual(q.nbytes, q.w.nbytes + (q.bias.nbytes if q.bias is not None else 0))
        self.assertFalse(cfg.tied_output)                         # the synthetic files carry output.weight

    def test_the_embedding_multiplier_scales_the_input_exactly(self):
        ids = [5, 40, 7, 257, 12]
        path = os.path.join(self.dir, "emb2.gguf")
        values = S.write(path, seed=6, quant="F16")
        model = self.build(path)
        model.cfg.embed_scale = 2.0
        doubled = dict(values, **{"token_embd.weight": values["token_embd.weight"] * 2.0})
        np.testing.assert_allclose(self.fwd(model, ids), S.reference_logits(doubled, ids), atol=0.1, rtol=0)

    def test_cache_grows_in_steps_and_keeps_what_it_holds(self):
        import mlx.core as mx
        from engine.dense import KVCache
        c = KVCache()
        self.assertEqual((c.nbytes, c.state, c.offset), (0, [], 0))
        k = mx.arange(2 * 1 * 3 * 4, dtype=mx.float16).reshape(2, 1, 3, 4)
        ck, cv = c.update_and_fetch(k, k + 100)
        self.assertEqual(ck.shape, (2, 1, 3, 4))
        self.assertEqual(c.keys.shape[2], KVCache.STEP)             # one step of room, not just 3 slots
        self.assertEqual(len(c.state), 2)
        self.assertEqual(c.nbytes, c.keys.nbytes + c.values.nbytes)
        big = mx.ones((2, 1, KVCache.STEP, 4), mx.float16)
        ck, cv = c.update_and_fetch(big, big * 2)
        self.assertEqual(c.offset, 3 + KVCache.STEP)
        self.assertEqual(c.keys.shape[2], 2 * KVCache.STEP)         # (3 + STEP) rounds up to two steps exactly
        np.testing.assert_array_equal(np.asarray(ck[:, :, :3].astype(mx.float32)), np.asarray(k.astype(mx.float32)))
        np.testing.assert_array_equal(np.asarray(cv[:, :, :3].astype(mx.float32)), np.asarray((k + 100).astype(mx.float32)))
        self.assertEqual(float(ck[0, 0, 3, 0]), 1.0)
        self.assertEqual(float(cv[0, 0, -1, 0]), 2.0)
        c.update_and_fetch(mx.ones((2, 1, 1, 4), mx.float16), mx.ones((2, 1, 1, 4), mx.float16))
        self.assertEqual(c.keys.shape[2], 2 * KVCache.STEP)         # still fits: no regrowth
        exact = KVCache()                                            # a request that fills a step exactly gets just it
        exact.update_and_fetch(mx.zeros((1, 1, KVCache.STEP, 2), mx.float16), mx.zeros((1, 1, KVCache.STEP, 2), mx.float16))
        self.assertEqual(exact.keys.shape[2], KVCache.STEP)
        exact.update_and_fetch(mx.zeros((1, 1, 1, 2), mx.float16), mx.zeros((1, 1, 1, 2), mx.float16))
        self.assertEqual(exact.keys.shape[2], 2 * KVCache.STEP)     # one token past it takes the next step
        self.assertEqual(c.trim(5), 5)
        self.assertEqual(c.offset, 3 + KVCache.STEP + 1 - 5)


class TestGenerationLoop(Base):
    """The shared generate() loop of engine/compat.py, watched from outside: how the prompt is cut into chunks,
    what is checked and how often, and the numbers speculation reports."""

    def engine(self, **kw):
        from engine.gguf_engine import GGUFEngine
        logs = []
        kw.setdefault("max_context", 1024)
        kw.setdefault("max_tokens", 200)
        e = GGUFEngine(self.q8, log=lambda *a: logs.append(" ".join(map(str, a))), **kw)
        e._logs = logs
        return e

    def run_gen(self, e, ids, **kw):
        out = []
        for item in e.generate(ids, **kw):
            if isinstance(item, dict):
                return out, item
            out.append(item)
        return out, {}

    def spy(self, e):
        """Record the width of every model call (and the engine's progress at that moment)."""
        calls, real = [], e.model

        class Spy:
            def __getattr__(inner, name):
                return getattr(real, name)

            def __call__(inner, ids, cache=None):
                calls.append((ids.shape[1], dict(e.progress) if e.progress else None))
                return real(ids, cache)
        e.model = Spy()
        return calls

    GREEDY = {"temperature": 0.0}

    def test_the_context_limit_is_exclusive_and_max_tokens_is_clamped_to_the_room(self):
        from engine.generate import ContextError
        e = self.engine(max_context=40, max_tokens=64)
        ids = list(range(10, 50))
        with self.assertRaises(ContextError):
            self.run_gen(e, ids[:40], sampling=self.GREEDY)                      # exactly the limit: refused
        _, st = self.run_gen(e, ids[:35], max_tokens=50, sampling=self.GREEDY)   # 5 tokens of room
        self.assertLessEqual(st["completion_tokens"], 5)

    def test_a_long_prompt_is_prefilled_in_chunks_of_at_least_64_and_progress_reports_them(self):
        e = self.engine(prefill_step=1)                                          # asks for 1: the floor is 64
        calls = self.spy(e)
        ids = [(7 * i) % 200 + 3 for i in range(150)]
        _, st = self.run_gen(e, ids, max_tokens=2, sampling=self.GREEDY)
        widths = [w for w, _ in calls]
        self.assertEqual(widths[:3], [64, 64, 21])                              # 149 = 64 + 64 + 21, then the last token
        self.assertEqual([p["done"] for _, p in calls[:3]], [0, 64, 128])
        self.assertEqual({p["phase"] for _, p in calls[:3]}, {"prefill"})
        self.assertEqual(calls[3][0], 1)                                        # the last prompt token, alone
        self.assertEqual(st["prompt_tokens"], 150)
        self.assertIsNone(e.progress)                                           # cleared when prefill is over

    def test_the_cache_holds_exactly_what_was_fed(self):
        e = self.engine()
        ids = [(5 * i) % 150 + 3 for i in range(30)]
        out, st = self.run_gen(e, ids, max_tokens=7, sampling=self.GREEDY)
        n_out = st["completion_tokens"]
        self.assertEqual(e.cache_tokens[:30], ids)
        self.assertEqual(len(e.cache_tokens), 30 + n_out)                       # every sampled token went through
        self.assertEqual(e.cache[0].offset, len(e.cache_tokens))

    def test_memory_is_checked_every_32_tokens_and_a_refusal_ends_the_answer(self):
        e = self.engine(max_tokens=100)
        seen = []
        real = e._check_memory
        e._check_memory = lambda stage: (seen.append(stage), True)[1]
        ids = [(3 * i) % 100 + 3 for i in range(12)]
        _, st = self.run_gen(e, ids, max_tokens=70, sampling=self.GREEDY, stop=None)
        self.assertEqual(st["completion_tokens"], 70)
        self.assertEqual(seen.count("decode"), 2)                                # at 32 and 64
        self.assertEqual(seen.count("prefill"), 1)                               # a 12-token prompt is one chunk
        e2 = self.engine(max_tokens=100)
        e2._check_memory = lambda stage: stage != "decode"                      # decode checks fail
        _, st2 = self.run_gen(e2, ids, max_tokens=90, sampling=self.GREEDY)
        self.assertEqual((st2["completion_tokens"], st2["finish_reason"]), (32, "length"))
        self.assertTrue(any("memory: stopped generation" in l for l in e2._logs))

    def test_a_looping_answer_is_cut_at_the_first_check_after_16_tokens(self):
        import engine.compat as compat
        from unittest import mock
        e = self.engine(max_tokens=100)
        ids = [(3 * i) % 100 + 3 for i in range(12)]
        with mock.patch.object(compat, "is_looping", lambda text: True):
            _, st = self.run_gen(e, ids, max_tokens=60, sampling=self.GREEDY)
        self.assertEqual((st["finish_reason"], st["loop_stopped"], st["completion_tokens"]), ("stop", True, 16))
        self.assertTrue(any("repetition: the answer was looping; stopped after 16 tokens" in l for l in e._logs))

    def test_speculation_reports_exactly_what_the_model_calls_show(self):
        e = self.engine(max_context=2048, max_tokens=120)
        calls = self.spy(e)
        unit = e.tok.encode("the cat in the hat sat in a hat. ")
        ids = unit * 8
        out, st = self.run_gen(e, ids, max_tokens=60, sampling=self.GREEDY)
        widths = [w for w, _ in calls]
        decode_widths = widths[len(ids) // 64 + 1:] if False else [w for w, p in calls if p is None]
        verifies = [w for w in decode_widths if w > 1]
        self.assertEqual(st["verifies"], len(verifies))
        self.assertGreater(st["verifies"], 0)                                   # the repeating prompt gave drafts
        drafted = sum(w - 1 for w in verifies)
        self.assertEqual(e.totals["drafted"], drafted)
        self.assertLessEqual(e.totals["accepted"], drafted)
        self.assertEqual(st["draft_accept"], round(e.totals["accepted"] / drafted, 2))
        self.assertEqual(e.totals["requests"], 1)
        self.assertEqual(e.totals["prompt_tokens"], len(ids))
        self.assertEqual(e.totals["completion_tokens"], st["completion_tokens"])
        self.assertEqual(e.history[-1]["out"], st["completion_tokens"])
        self.assertEqual(e.stats["last"], st)
        # drafts never exceed the configured longest level
        self.assertLessEqual(max(w - 1 for w in verifies), max(e.draft_levels))

    def test_without_speculation_nothing_is_drafted_and_the_rate_is_none(self):
        e = self.engine(speculative=False)
        ids = e.tok.encode("the cat in the hat sat in a hat. ") * 4
        _, st = self.run_gen(e, ids, max_tokens=20, sampling=self.GREEDY)
        self.assertEqual((st["verifies"], st["draft_accept"]), (0, None))
        self.assertEqual((e.totals["drafted"], e.totals["accepted"]), (0, 0))


class TestEngineSettings(Base):
    def make(self, **kw):
        from engine.gguf_engine import GGUFEngine
        logs = []
        return GGUFEngine(self.f16, max_context=256, max_tokens=8, log=lambda *a: logs.append(" ".join(map(str, a))),
                          **kw), logs

    def test_draft_and_ngram_limits_are_clamped(self):
        e, _ = self.make(draft_max=3, ngram_min=9)
        self.assertTrue(e.draft_levels and max(e.draft_levels) == 3)
        self.assertEqual(e.ngram_min, 4)
        e, _ = self.make(draft_max=100, ngram_min=0)
        self.assertEqual(max(e.draft_levels), 15)
        self.assertEqual(e.ngram_min, 1)
        e, _ = self.make(draft_max=0, ngram_min=3)
        self.assertEqual(e.draft_levels, (1,))
        self.assertEqual(e.ngram_min, 3)
        default, _ = self.make()                                   # what a plain start gets
        self.assertEqual((max(default.draft_levels), default.ngram_min), (15, 2))
        self.assertTrue(self.make()[0].speculative)
        self.assertFalse(self.make(speculative=False)[0].speculative)

    def test_the_load_log_reports_size_and_time(self):
        import re
        _, logs = self.make()
        line = next(l for l in logs if l.startswith("loaded "))
        self.assertRegex(line, r"tiny-f16\.gguf \(llama, \d+ layers\) on the MLX decoder: 0\.00 GiB, [0-9]\.[0-9]s$")

    def test_trim_helper_reports_the_first_layer_and_zero_without_a_cache(self):
        from engine.gguf_engine import _trim
        class C:
            def __init__(self, r): self.r = r
            def trim(self, n): return self.r
        self.assertEqual(_trim([C(7), C(3)], 2), 7)
        self.assertEqual(_trim([], 2), 0)

    def test_the_configs_come_from_the_header_with_documented_defaults(self):
        from unittest import mock
        from engine import gguf
        from engine.gguf_engine import GGUFEngine
        e, _ = self.make()
        full = {"general.architecture": "llama", "llama.block_count": 5, "llama.embedding_length": 64,
                "llama.attention.head_count": 8, "llama.attention.head_count_kv": 2,
                "llama.attention.key_length": 16, "llama.context_length": 999}
        with mock.patch.object(gguf, "read_header_full", return_value=full):
            cfg, gen = GGUFEngine._read_configs(e, "x.gguf")
        self.assertEqual(cfg, {"model_type": "llama", "num_hidden_layers": 5, "hidden_size": 64,
                               "num_attention_heads": 8, "num_key_value_heads": 2, "head_dim": 16,
                               "max_position_embeddings": 999, "vocab_size": 0})
        self.assertEqual(gen, {})
        bare = {"general.architecture": "llama", "llama.embedding_length": 128, "llama.attention.head_count": 4}
        with mock.patch.object(gguf, "read_header_full", return_value=bare):
            cfg, _ = GGUFEngine._read_configs(e, "x.gguf")
        self.assertEqual((cfg["num_hidden_layers"], cfg["num_key_value_heads"], cfg["head_dim"],
                          cfg["max_position_embeddings"]), (32, 4, 32, 4096))
        with mock.patch.object(gguf, "read_header_full", return_value={"general.architecture": "llama"}):
            cfg, _ = GGUFEngine._read_configs(e, "x.gguf")
        self.assertEqual((cfg["num_attention_heads"], cfg["hidden_size"], cfg["head_dim"]), (32, 4096, 128))

    def test_steering_strength_prefers_the_request_then_the_recorded_value_then_one(self):
        import tempfile
        from unittest import mock
        import mlx.core as mx
        from engine import steering
        p = os.path.join(tempfile.mkdtemp(), "dir.safetensors")
        mx.save_safetensors(p, {"direction": mx.array(np.ones(S.HIDDEN, np.float32))})
        for asked, recorded, want in ((0.5, 2.5, "0.5"), (None, 2.5, "2.5"), (None, None, "1"), (0.0, 2.5, "0")):
            with mock.patch.object(steering, "resolve", return_value=(p, recorded)):
                _, logs = self.make(steer=(p, asked))
            self.assertTrue(any(l.endswith(f"alpha {want}") for l in logs), (asked, recorded, logs))
            self.assertTrue(any("dir.safetensors" in l for l in logs))


if __name__ == "__main__":
    unittest.main()
