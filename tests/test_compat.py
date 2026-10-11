"""The shared stop-string filter, backend selection, and the mlx-lm backend in-process on synthetic
bundles (needs mlx-lm; those tests skip without it — the pure ones always run).

    .venv/bin/python -m unittest tests.test_compat -v
"""
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from engine import compat  # noqa: E402
from engine.stopfilter import StopFilter  # noqa: E402
from tests.synth import CHATML, CHATML_THINKING_OFF_BY_DEFAULT, make_bundle, make_qwen3_bundle  # noqa: E402

ON_BY_DEFAULT = (
    "{%- for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{%- endfor %}"
    "{%- if add_generation_prompt %}<|im_start|>assistant\n"
    "{%- if enable_thinking is undefined or enable_thinking is true %}<think>\n"
    "{%- else %}<think>\n\n</think>\n\n{%- endif %}{%- endif %}"
)
# Qwen3's convention: the template never writes an opening <think>; the model does, unless told not to
QWEN3_STYLE = (
    "{%- for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{%- endfor %}"
    "{%- if add_generation_prompt %}<|im_start|>assistant\n"
    "{%- if enable_thinking is defined and enable_thinking is false %}<think>\n\n</think>\n\n{%- endif %}"
    "{%- endif %}"
)


def drive(f, pieces):
    out, hit = "", False
    for p in pieces:
        h, text = f.feed(p)
        out += text
        if h:
            hit = True
            break
    if not hit:
        out += f.flush()
    return out, hit


class TestStopFilter(unittest.TestCase):
    def test_no_stop_passes_everything_through(self):
        self.assertEqual(drive(StopFilter(None), ["ab", "cd"]), ("abcd", False))
        self.assertEqual(drive(StopFilter([]), ["ab"]), ("ab", False))
        self.assertEqual(drive(StopFilter([""]), ["ab"]), ("ab", False))      # an empty stop never matches

    def test_cut_inside_one_piece(self):
        self.assertEqual(drive(StopFilter(["END"]), ["hello EN", "D and more"]), ("hello ", True))
        self.assertEqual(drive(StopFilter(["lo"]), ["hello world"]), ("hel", True))

    def test_stop_split_across_many_pieces(self):
        self.assertEqual(drive(StopFilter(["STOP"]), ["a S", "T", "O", "P tail"]), ("a ", True))

    def test_a_partial_prefix_is_held_back_until_decided(self):
        f = StopFilter(["STOP"])
        released = [f.feed(p)[1] for p in ["x ST", "OX"]]
        self.assertEqual(released[0], "x ")           # "ST" might become STOP: not released yet
        self.assertEqual(released[1], "STOX")         # it did not: now released together with the rest
        self.assertEqual(f.flush(), "")

    def test_text_held_at_the_end_is_flushed(self):
        out, hit = drive(StopFilter(["STOP"]), ["abc ST"])
        self.assertEqual((out, hit), ("abc ST", False))

    def test_earliest_of_several_stops_wins(self):
        self.assertEqual(drive(StopFilter(["cc", "bb"]), ["aabbcc"]), ("aa", True))

    def test_overlapping_candidates(self):
        self.assertEqual(drive(StopFilter(["aab"]), ["aaa", "ab"]), ("aa", True))     # "aaaab": the stop starts at index 2

    def test_nothing_is_ever_lost_or_duplicated(self):
        import random
        rng = random.Random(7)
        alphabet = "ab"
        for _ in range(300):
            text = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 25)))
            stop = ["".join(rng.choice(alphabet) for _ in range(rng.randint(1, 4)))]
            cuts = sorted(rng.sample(range(len(text) + 1), min(len(text) + 1, rng.randint(0, 5))))
            pieces = [text[a:b] for a, b in zip([0] + cuts, cuts + [len(text)])]
            out, hit = drive(StopFilter(stop), pieces)
            idx = text.find(stop[0])
            self.assertEqual((out, hit), (text[:idx], True) if idx >= 0 else (text, False), (text, stop, pieces))


class TestBackendSelection(unittest.TestCase):
    def cfg(self, **kw):
        d = tempfile.mkdtemp()
        with open(os.path.join(d, "config.json"), "w") as f:
            json.dump(kw, f)
        return d

    def test_native_family_goes_native_everything_else_to_mlx_lm(self):
        pick = compat.pick_backend
        self.assertEqual(pick(self.cfg(model_type="qwen3_5", text_config={"model_type": "qwen3_5_text"})), "native")
        self.assertEqual(pick(self.cfg(schema="prism_hadamard_qwen35_ptq1_v1")), "native")
        self.assertEqual(pick(self.cfg(model_type="qwen3")), "mlxlm")
        self.assertEqual(pick(self.cfg(model_type="glm4_moe")), "mlxlm")
        self.assertEqual(pick(self.cfg(model_type="qwen3_5", text_config={"num_experts": 8})), "mlxlm")  # MoE
        self.assertEqual(pick(self.cfg(model_type="qwen3"), "native"), "native")                       # explicit wins
        self.assertEqual(pick(self.cfg(model_type="qwen3_5"), "mlxlm"), "mlxlm")

    def test_kv_cost_estimate(self):
        cfg = {"text_config": {"num_hidden_layers": 10, "num_attention_heads": 8, "num_key_value_heads": 2,
                               "hidden_size": 512}}
        self.assertEqual(compat.kv_bytes_per_token(cfg), 10 * 2 * 2 * 64 * 2)       # layers · K+V · kv heads · head dim · fp16
        self.assertGreater(compat.kv_bytes_per_token({}), 0)                         # an empty config still gives a figure

    def test_without_mlx_lm_the_error_says_how_to_install_it(self):
        d = self.cfg(model_type="qwen3")
        with mock.patch.object(compat, "mlx_lm_available", lambda: False):
            with self.assertRaises(RuntimeError) as ctx:
                compat.CompatEngine(d)
        self.assertIn("setup --compat", str(ctx.exception))


@unittest.skipUnless(compat.mlx_lm_available(), "mlx-lm is not installed")
class TestCompatEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engines, cls.dirs = {}, {}
        for name, tpl in (("plain", CHATML), ("off", CHATML_THINKING_OFF_BY_DEFAULT),
                          ("on", ON_BY_DEFAULT), ("qwen3", QWEN3_STYLE)):
            d = tempfile.mkdtemp()
            make_bundle(d, tpl, seed=31)
            cls.dirs[name] = d
            cls.engines[name] = compat.CompatEngine(d, max_context=1024, max_tokens=24, log=lambda *_: None)

    def gen(self, eng, text, **kw):
        ids, _ = eng.encode(text, [])
        out, st = "", None
        for it in eng.generate(ids, sampling={"temperature": 0}, **kw):
            if isinstance(it, dict):
                st = it
            else:
                out += it[1]
        return out, st

    def msgs(self, text="what is two plus two"):
        return [{"role": "user", "content": text}]

    def test_the_default_comes_from_the_template_not_from_mlx_lms_wrapper(self):
        caps = {n: (e.supports_thinking, e.thinking_default) for n, e in self.engines.items()}
        self.assertEqual(caps, {"plain": (False, False), "off": (True, False),
                                "on": (True, True), "qwen3": (True, True)})

    def test_prompts_open_or_close_reasoning_as_asked(self):
        off, q3, plain = self.engines["off"], self.engines["qwen3"], self.engines["plain"]
        text, _, opened = off.build_prompt(self.msgs(), thinking=True)
        self.assertTrue(text.rstrip().endswith("<think>") and opened)
        text, _, opened = off.build_prompt(self.msgs(), thinking=False)
        self.assertTrue(text.rstrip().endswith("</think>") and not opened)
        # Qwen3 style: the template says nothing, so the backend writes the opening tag itself
        text, _, opened = q3.build_prompt(self.msgs(), thinking=True)
        self.assertTrue(text.endswith("<think>\n") and opened)
        text, _, opened = q3.build_prompt(self.msgs(), thinking=False)
        self.assertTrue(text.rstrip().endswith("</think>") and not opened)
        text, _, opened = plain.build_prompt(self.msgs(), thinking=True)
        self.assertNotIn("<think>", text)
        self.assertFalse(opened)

    def test_parser_markers(self):
        self.assertEqual(self.engines["qwen3"].parser_markers()["think_end"], "</think>")
        self.assertIsNone(self.engines["plain"].parser_markers()["think_end"])

    def test_greedy_decoding_is_deterministic_and_respects_max_tokens(self):
        e = self.engines["plain"]
        text, _, _ = e.build_prompt(self.msgs())
        a, st = self.gen(e, text, max_tokens=10)
        b, _ = self.gen(e, text, max_tokens=10)
        self.assertEqual(a, b)
        self.assertLessEqual(st["completion_tokens"], 10)
        self.assertIn(st["finish_reason"], ("stop", "length"))

    def test_stop_strings_cut_the_answer_like_the_native_engine(self):
        e = self.engines["plain"]
        text, _, _ = e.build_prompt(self.msgs("tell me about foxes"))
        full, _ = self.gen(e, text, max_tokens=20)
        if len(full) < 6:
            self.skipTest("the random model answered too briefly to cut")
        stop = full[3:5]
        cut, st = self.gen(e, text, max_tokens=20, stop=[stop])
        self.assertEqual(cut, full[:full.find(stop)])
        self.assertEqual(st["finish_reason"], "stop")
        never = full[-2:] + "☃☃"
        held, _ = self.gen(e, text, max_tokens=20, stop=[never])
        self.assertEqual(held, full)                              # the held-back tail is released, nothing lost

    def test_a_runaway_reasoning_phase_is_force_closed(self):
        e = self.engines["qwen3"]
        old = e.think_frac
        e.think_frac = 0.1
        try:
            text, _, opened = e.build_prompt(self.msgs(), thinking=True)
            self.assertTrue(opened)
            ids, _ = e.encode(text, [])
            out, st = "", None
            for it in e.generate(ids, sampling={"temperature": 0}, max_tokens=20, thinking_open=True):
                if isinstance(it, dict):
                    st = it
                else:
                    out += it[1]
            self.assertEqual(st["think_forced"], "budget")
            self.assertIn("</think>", out)
        finally:
            e.think_frac = old

    def run_ids(self, eng, ids, max_tokens=4):
        out_ids, st = [], None
        for it in eng.generate(ids, sampling={"temperature": 0}, max_tokens=max_tokens):
            if isinstance(it, dict):
                st = it
            elif it[0] is not None:
                out_ids.append(it[0])
        return out_ids, st

    def test_an_exact_prefix_is_reused_even_when_the_cache_cannot_be_trimmed(self):
        e = self.engines["plain"]                                  # hybrid: its recurrent state is not trimmable
        ids, _ = e.encode(e.build_prompt(self.msgs("the quick brown fox jumps over the lazy dog " * 3))[0], [])
        self.run_ids(e, ids)
        held = list(e.cache_tokens)                                # prompt + everything generated, all in the cache
        self.assertEqual(e.cache[0].offset if hasattr(e.cache[0], "offset") else len(held), len(held)) \
            if hasattr(e.cache[0], "offset") else None
        follow = held + e.encode(" and then some more words", [])[0]
        _, st = self.run_ids(e, follow)
        self.assertEqual(st["cached_tokens"], len(held))           # nothing re-done, and nothing to trim

    def test_a_diverging_prompt_is_trimmed_back_on_an_attention_only_model(self):
        d = tempfile.mkdtemp()
        make_qwen3_bundle(d, seed=32)                               # plain Qwen3: full attention, trimmable KV cache
        e = compat.CompatEngine(d, max_context=1024, max_tokens=24, log=lambda *_: None)
        ids, _ = e.encode(e.build_prompt(self.msgs("the quick brown fox jumps over the lazy dog " * 3))[0], [])
        self.run_ids(e, ids, 6)
        other = ids[:len(ids) - 5] + e.encode(" a different ending entirely", [])[0]
        a, st = self.run_ids(e, other)
        self.assertEqual(st["cached_tokens"], len(ids) - 5)         # exactly the shared part
        # the cache holds this prompt plus what was generated — nothing of the abandoned continuation
        self.assertEqual(e.cache[0].offset, len(other) + st["completion_tokens"])
        self.assertEqual(len(e.cache_tokens), e.cache[0].offset)    # the bookkeeping matches the real cache
        e2 = compat.CompatEngine(d, max_context=1024, max_tokens=24, log=lambda *_: None)
        b, st2 = self.run_ids(e2, other)                            # from scratch
        self.assertEqual(st2["cached_tokens"], 0)
        self.assertTrue(a)                                          # not a vacuous comparison of two empty answers
        self.assertEqual(a, b)                                      # a trimmed cache gives the same answer

    def test_an_older_qwen_architecture_is_routed_to_the_backend_and_answers(self):
        d = tempfile.mkdtemp()
        make_qwen3_bundle(d, QWEN3_STYLE, seed=33)
        self.assertEqual(compat.pick_backend(d), "mlxlm")
        e = compat.CompatEngine(d, max_context=512, max_tokens=12, log=lambda *_: None)
        self.assertEqual((e.supports_thinking, e.thinking_default), (True, True))
        text, _, opened = e.build_prompt(self.msgs())
        self.assertTrue(opened)
        out, st = self.gen(e, text, max_tokens=6)
        self.assertGreater(st["completion_tokens"], 0)

    def test_an_unrelated_prompt_starts_from_scratch(self):
        e = self.engines["plain"]
        self.gen(e, e.build_prompt(self.msgs("the quick brown fox " * 5))[0], max_tokens=4)
        _, st = self.gen(e, e.build_prompt(self.msgs("something entirely different"))[0], max_tokens=4)
        self.assertEqual(st["cached_tokens"], 0)

    def test_sleep_unloads_the_model_and_the_next_request_wakes_it(self):
        e = self.engines["off"]
        e.sleep()
        self.assertTrue(e.sleeping)
        self.assertIsNone(e.model)
        text, _, _ = e.build_prompt(self.msgs(), thinking=False)
        out, st = self.gen(e, text, max_tokens=4)
        self.assertFalse(e.sleeping)
        self.assertGreater(st["completion_tokens"], 0)

    def test_a_prompt_beyond_the_context_is_a_context_error(self):
        e = self.engines["plain"]
        from engine.generate import ContextError
        with self.assertRaises(ContextError):
            list(e.generate([1] * 2000, sampling={"temperature": 0}))


if __name__ == "__main__":
    unittest.main()
