"""Streaming the experts of an MLX Mixture-of-Experts model from disk (engine/moe_stream.py), against the same
model with every expert in memory. A tiny quantized Qwen3-MoE is built here; no download, no tokenizer.

    .venv/bin/python -m unittest tests.test_moe_stream -v
"""
import json
import os
import tempfile
import unittest
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten

from engine import moe_stream as M

CFG = {"model_type": "qwen3_moe", "hidden_size": 128, "num_hidden_layers": 3, "intermediate_size": 256,
       "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 32, "num_experts": 16,
       "num_experts_per_tok": 2, "decoder_sparse_step": 1, "mlp_only_layers": [], "moe_intermediate_size": 64,
       "rms_norm_eps": 1e-6, "vocab_size": 96, "rope_theta": 10000.0, "tie_word_embeddings": False,
       "max_position_embeddings": 512, "norm_topk_prob": True}


def build(folder, bits=4, quantize=True, seed=3, cfg=None):
    # the tiny model is built with mlx-lm's Qwen3-MoE, an optional install (not in requirements.txt): without it
    # these tests are skipped and say so, instead of erroring (CI installs only requirements.txt)
    try:
        from mlx_lm.models import qwen3_moe
    except ImportError:
        raise unittest.SkipTest("mlx-lm is not installed (pip install mlx-lm): MoE streaming was not covered")
    cfg = dict(CFG, **(cfg or {}))
    mx.random.seed(seed)
    model = qwen3_moe.Model(qwen3_moe.ModelArgs.from_dict(cfg))
    if quantize:
        nn.quantize(model, group_size=64, bits=bits)
        cfg["quantization"] = {"group_size": 64, "bits": bits}
    mx.eval(model.parameters())
    os.makedirs(folder, exist_ok=True)
    mx.save_safetensors(os.path.join(folder, "model.safetensors"), dict(tree_flatten(model.parameters())))
    with open(os.path.join(folder, "config.json"), "w") as f:
        json.dump(cfg, f)
    return cfg


def load_lazy(folder):
    from mlx_lm.utils import load_model
    model, _ = load_model(Path(folder), lazy=True)
    return model


def plan(capacity, folder=None, experts=CFG["num_experts"]):
    return {"capacity": capacity, "experts": experts, "top_k": 2, "layers": 3, "total_gb": 0, "experts_gb": 0,
            "resident_gb": 0}


def logits(model, ids):
    from mlx_lm.models.cache import make_prompt_cache
    out = model(mx.array([ids]), cache=make_prompt_cache(model))
    mx.eval(out)
    return np.asarray(out.astype(mx.float32))


class Base(unittest.TestCase):
    bits = 4
    quantize = True

    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.mkdtemp()
        cls.cfg = build(cls.dir, cls.bits, cls.quantize)
        cls.eager = load_lazy(cls.dir)
        mx.eval(cls.eager.parameters())

    def streamed(self, capacity):
        m = load_lazy(self.dir)
        st = M.stream_experts(m, self.dir, plan(capacity))
        self.addCleanup(st.close)
        mx.eval(m.parameters())
        return m, st


class TestEquivalence(Base):
    def test_every_expert_resident_is_identical_to_the_original(self):
        m, st = self.streamed(16)
        ids = [3, 9, 27, 81, 5, 44, 7]
        np.testing.assert_array_equal(logits(m, ids), logits(self.eager, ids))
        self.assertEqual(len(st.pools), 3)

    def test_a_pool_smaller_than_the_set_of_experts_still_gives_the_same_answers(self):
        m, st = self.streamed(4)                                         # 4 of 16 experts, 2 used per token
        for ids in ([3, 9, 27, 81], [5, 44, 7, 12, 90, 1, 2, 3, 4, 5], [50, 51]):
            np.testing.assert_array_equal(logits(m, ids), logits(self.eager, ids), str(ids))
        s = st.stats()
        self.assertGreater(s["misses"], 0)                               # experts really were read from the file
        self.assertGreater(s["gb_read"], 0)

    def test_a_long_prompt_is_identical_while_each_group_is_as_big_as_the_original_runs_it(self):
        ids = list(range(1, 80))                                         # 158 (token, expert) pairs: the sorted path
        for capacity in (16, 8):                                         # one group, then two
            m, _ = self.streamed(capacity)
            np.testing.assert_array_equal(logits(m, ids), logits(self.eager, ids), f"capacity {capacity}")

    def test_a_long_prompt_needing_more_experts_than_the_pool_holds_is_split_not_refused(self):
        m, st = self.streamed(3)                                         # a pool of 3, a prompt that wants all 16
        ids = list(range(1, 80))
        got, want = logits(m, ids), logits(self.eager, ids)
        # the groups are smaller, so MLX picks another tile for the quantized product: rounding, not a different
        # computation. Everything below is the same to well under the spacing of the logits themselves.
        self.assertLess(float(np.abs(got - want).max()), 0.01 * float(np.ptp(want)))
        np.testing.assert_array_equal(got.argmax(-1), want.argmax(-1))
        self.assertEqual(len(next(iter(st.pools.values())).resident), 3)

    def test_decoding_token_by_token_matches(self):
        from mlx_lm.models.cache import make_prompt_cache
        m, st = self.streamed(5)
        ca, cb = make_prompt_cache(m), make_prompt_cache(self.eager)
        toks = [3, 9, 27]
        mx.eval(m(mx.array([toks]), cache=ca), self.eager(mx.array([toks]), cache=cb))
        for t in (40, 41, 42, 7, 8, 9, 10):
            a = m(mx.array([[t]]), cache=ca)
            b = self.eager(mx.array([[t]]), cache=cb)
            mx.eval(a, b)
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b), f"token {t}")

    def test_the_streamed_model_holds_far_less_than_the_whole(self):
        m, st = self.streamed(4)
        held = sum(v.nbytes for _, v in tree_flatten(m.parameters()))
        whole = sum(v.nbytes for _, v in tree_flatten(self.eager.parameters()))
        pools = sum(p.nbytes for p in st.pools.values())
        self.assertLess(held, 0.35 * whole)                              # the experts were dropped from the tree
        self.assertLess(pools, 0.35 * (whole - held))                    # and 4/16 of them are resident


class TestUnquantized(Base):
    quantize = False

    def test_plain_float_experts_stream_too(self):
        m, st = self.streamed(4)
        ids = [3, 9, 27, 81, 5, 44, 7]
        got, want = logits(m, ids), logits(self.eager, ids)
        self.assertLess(float(np.abs(got - want).max()), 0.01 * float(np.ptp(want)))   # grouped products: see above
        np.testing.assert_array_equal(got.argmax(-1), want.argmax(-1))
        self.assertGreater(st.stats()["misses"], 0)


class TestEight(Base):
    bits = 8

    def test_eight_bit_experts(self):
        m, _ = self.streamed(4)
        ids = [11, 12, 13, 14, 15]
        np.testing.assert_array_equal(logits(m, ids), logits(self.eager, ids))


class TestPool(unittest.TestCase):
    def pool(self, capacity=3, n=8):
        d = tempfile.mkdtemp()
        w = mx.arange(n * 4 * 2, dtype=mx.uint32).reshape(n, 4, 2)
        mx.save_safetensors(os.path.join(d, "model.safetensors"), {"x.switch_mlp.up_proj.weight": w})
        idx = M.index(d)
        files = M.Files()
        from concurrent.futures import ThreadPoolExecutor
        pool = M.LayerPool(files, {"up_proj": {"weight": idx["x.switch_mlp.up_proj.weight"]}}, ("up_proj",), n,
                           capacity, ThreadPoolExecutor(2))
        self.addCleanup(files.close)
        return pool, np.asarray(w)

    def test_slots_hold_the_bytes_of_their_expert(self):
        pool, w = self.pool()
        slots = pool.ensure([5, 1, 7])
        got = np.asarray(pool.arrays["up_proj", "weight"])
        for e, s in slots.items():
            np.testing.assert_array_equal(got[s], w[e])

    def test_the_often_used_survive_and_what_was_just_asked_for_is_never_evicted(self):
        pool, _ = self.pool(capacity=3)
        pool.ensure([0, 1, 2])
        for _ in range(20):
            pool.use({0: 3, 1: 3})                                       # 0 and 1 are hot, 2 is not
        pool.ensure([5])
        self.assertEqual(sorted(pool.resident), [0, 1, 5])               # 2 went, not a hot one
        pool.ensure([6, 7, 0])
        self.assertEqual(sorted(pool.resident), [0, 6, 7])               # all three just asked for are kept
        with self.assertRaisesRegex(M.StreamError, "pool holds 3"):
            pool.ensure([1, 2, 3, 4])

    def test_the_score_follows_the_conversation(self):
        pool, _ = self.pool()
        pool.use({0: 100})
        for _ in range(20000):
            pool.use({1: 1})
        self.assertGreater(pool.score[1], pool.score[0])                 # old routing faded

    def test_hits_and_misses_are_counted(self):
        pool, _ = self.pool()
        pool.ensure([0, 1])
        pool.ensure([0, 1, 2])
        s = pool.stats()
        self.assertEqual((s["hits"], s["misses"]), (2, 3))
        self.assertAlmostEqual(s["hit_rate"], 0.4)

    def test_a_truncated_file_is_an_error_not_garbage(self):
        pool, _ = self.pool()
        path = next(iter(pool.slab.values()))[4](0)[0]
        with open(path, "r+b") as f:
            f.truncate(os.path.getsize(path) - 20)
        with self.assertRaisesRegex(M.StreamError, "truncated"):
            pool.ensure([7])


class TestPlan(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.cfg = build(self.dir)
        self.env = dict(os.environ)
        self.addCleanup(lambda: (os.environ.clear(), os.environ.update(self.env)))

    def test_a_model_that_fits_is_not_streamed(self):
        os.environ.pop("LLM_MOE_STREAM", None)
        self.assertIsNone(M.plan(self.dir, self.cfg, 0.88, 16 * 2**30))

    def test_a_model_that_does_not_fit_is_streamed_with_what_room_there_is(self):
        os.environ.pop("LLM_MOE_STREAM", None)
        experts, rest = M.expert_bytes(self.dir)
        ram = (rest + experts * 0.5) / 0.88                              # half the experts fit
        p = M.plan(self.dir, self.cfg, 0.88, ram, headroom_gb=0)
        self.assertEqual((p["experts"], p["layers"], p["top_k"]), (16, 3, 2))
        self.assertTrue(4 <= p["capacity"] <= 9, p)
        self.assertLess(p["resident_gb"], p["total_gb"])

    def test_off_and_on_and_the_forced_capacity(self):
        os.environ["LLM_MOE_STREAM"] = "off"
        self.assertIsNone(M.plan(self.dir, self.cfg, 0.88, 1 * 2**30))
        os.environ["LLM_MOE_STREAM"] = "on"
        os.environ["LLM_MOE_CACHE"] = "6"
        self.assertEqual(M.plan(self.dir, self.cfg, 0.88, 64 * 2**30)["capacity"], 6)

    def test_a_forced_capacity_is_honoured_even_when_small_and_flagged(self):
        os.environ["LLM_MOE_STREAM"] = "on"
        os.environ["LLM_MOE_CACHE"] = "1"                                # one expert per layer, two used per token
        p = M.plan(self.dir, self.cfg, 0.88, 64 * 2**30)
        self.assertEqual((p["capacity"], p["thrashing"]), (1, True))

    def test_too_little_memory_even_for_streaming_says_how_much_is_needed(self):
        os.environ["LLM_MOE_STREAM"] = "on"
        os.environ.pop("LLM_MOE_CACHE", None)
        experts, rest = M.expert_bytes(self.dir)
        with self.assertRaisesRegex(M.StreamError, r"even streamed, this model needs about"):
            M.plan(self.dir, self.cfg, 0.88, (rest + experts * 0.1) / 0.88, headroom_gb=0)

    def test_a_dense_model_is_never_streamed(self):
        os.environ["LLM_MOE_STREAM"] = "on"
        d = tempfile.mkdtemp()
        mx.save_safetensors(os.path.join(d, "model.safetensors"), {"model.layers.0.w": mx.ones((4, 4))})
        self.assertIsNone(M.plan(d, {"hidden_size": 4}, 0.88, 1 * 2**30))


class TestUsage(Base):
    def test_the_hot_set_is_saved_and_the_next_load_starts_warm(self):
        os.environ["LLM_MOE_USAGE_DIR"] = tempfile.mkdtemp()
        self.addCleanup(os.environ.pop, "LLM_MOE_USAGE_DIR")
        m, st = self.streamed(4)
        for ids in ([3, 9, 27, 81, 5, 44, 7, 12], [3, 9, 27, 81]):
            logits(m, ids)
        hot = {name: set(np.argsort(-p.score)[:4].tolist()) for name, p in st.pools.items()}
        st.save_usage()
        m2 = load_lazy(self.dir)
        st2 = M.stream_experts(m2, self.dir, plan(4))
        self.addCleanup(st2.close)
        for name, pool in st2.pools.items():
            self.assertEqual(set(pool.resident), hot[name], name)        # filled before the first token
            self.assertEqual(pool.misses, 0)                             # warming is not counted as missing

    def test_a_hot_set_from_a_different_model_is_ignored(self):
        os.environ["LLM_MOE_USAGE_DIR"] = tempfile.mkdtemp()
        self.addCleanup(os.environ.pop, "LLM_MOE_USAGE_DIR")
        os.makedirs(os.environ["LLM_MOE_USAGE_DIR"], exist_ok=True)
        with open(M.usage_file(self.dir), "w") as f:
            json.dump({"experts": 64, "layers": {"x": [1.0] * 64}}, f)
        _, st = self.streamed(4)
        self.assertTrue(all(not p.resident for p in st.pools.values()))


class TestEngine(unittest.TestCase):
    """The mlx-lm engine class on a MoE model: the same tokens whether the experts are resident or streamed."""

    @classmethod
    def setUpClass(cls):
        try:
            import mlx_lm  # noqa: F401
            import transformers  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("mlx-lm / transformers are not installed: expert streaming was not run "
                                    "through the engine")
        from tests.synth import CHATML, build_tokenizer
        cls.dir = tempfile.mkdtemp()
        vocab = build_tokenizer(cls.dir)
        build(cls.dir, cfg={"vocab_size": vocab, "eos_token_id": 2})
        with open(os.path.join(cls.dir, "chat_template.jinja"), "w") as f:
            f.write(CHATML)
        with open(os.path.join(cls.dir, "tokenizer_config.json"), "w") as f:
            json.dump({"tokenizer_class": "PreTrainedTokenizerFast", "eos_token": "<|im_end|>",
                       "chat_template": CHATML}, f)

    def engine(self, **env):
        from engine.compat import CompatEngine
        old = {k: os.environ.get(k) for k in env}
        os.environ.update(env)
        self.addCleanup(lambda: [os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
                                 for k, v in old.items()])
        logs = []
        e = CompatEngine(self.dir, max_context=1024, max_tokens=24, log=lambda *a: logs.append(" ".join(map(str, a))))
        e._logs = logs
        return e

    def generate(self, e, ids, n=20):
        toks = []
        for item in e.generate(ids, max_tokens=n, sampling={"temperature": 0}):
            if isinstance(item, dict):
                return toks, item
            toks.append(item[0])

    def test_streamed_and_resident_generate_the_same_tokens(self):
        ids = [3, 9, 27, 81, 5, 44, 7, 12, 30]
        whole = self.engine(LLM_MOE_STREAM="off")
        self.assertIsNone(whole.moe_stats())
        want, _ = self.generate(whole, ids)
        streamed = self.engine(LLM_MOE_STREAM="on", LLM_MOE_CACHE="5", LLM_MOE_USAGE_DIR=tempfile.mkdtemp())
        got, st = self.generate(streamed, ids)
        self.assertEqual(got, want)
        s = streamed.moe_stats()
        self.assertEqual((s["capacity"], s["layers"]), (5, 3))
        self.assertGreater(s["misses"], 0)
        self.assertTrue(any("experts streamed from disk: 5 of 16 per layer resident" in l for l in streamed._logs))
        self.assertLess(streamed.model_bytes, whole.model_bytes)

    def test_a_second_turn_hits_the_pool_more_than_the_first_misses(self):
        e = self.engine(LLM_MOE_STREAM="on", LLM_MOE_CACHE="8", LLM_MOE_USAGE_DIR=tempfile.mkdtemp())
        ids = [3, 9, 27, 81, 5, 44, 7, 12, 30]
        self.generate(e, ids)
        first = e.moe_stats()
        self.generate(e, ids + [50, 51])
        second = e.moe_stats()
        self.assertGreater(second["hit_rate"], first["hit_rate"])

    def test_sleep_saves_the_hot_set_and_wake_starts_warm(self):
        usage = tempfile.mkdtemp()
        e = self.engine(LLM_MOE_STREAM="on", LLM_MOE_CACHE="6", LLM_MOE_USAGE_DIR=usage)
        self.generate(e, [3, 9, 27, 81, 5, 44, 7, 12, 30])
        e.sleep()
        self.assertTrue(os.path.exists(M.usage_file(self.dir)))
        e.ensure_awake()
        pool = next(iter(e.moe.pools.values()))
        self.assertEqual(len(pool.resident), 6)                          # filled from the saved counts at load
        self.assertEqual(e.moe_stats()["misses"], 0)

    def test_health_reports_it_and_the_protocol_knows_the_capability(self):
        from engine import protocol
        e = self.engine(LLM_MOE_STREAM="on", LLM_MOE_CACHE="6", LLM_MOE_USAGE_DIR=tempfile.mkdtemp())
        self.assertEqual(protocol.missing(e), [])
        self.assertTrue(protocol.can(e, "moe_stats"))

    def test_a_model_that_cannot_fit_even_streamed_is_refused_with_the_reason(self):
        from engine import memory
        from engine.generate import MemoryLimitError
        real = memory.TOTAL
        memory.TOTAL = 3 * 2**20                                         # a Mac with 3 MB: not even the router fits
        self.addCleanup(setattr, memory, "TOTAL", real)
        with self.assertRaisesRegex(MemoryLimitError, "even streamed"):
            self.engine(LLM_MOE_STREAM="on", LLM_MOE_CACHE="")

    def test_a_pool_of_one_still_gives_the_right_tokens(self):
        ids = [3, 9, 27, 81, 5, 44, 7, 12, 30]
        want, _ = self.generate(self.engine(LLM_MOE_STREAM="off"), ids)
        got, _ = self.generate(self.engine(LLM_MOE_STREAM="on", LLM_MOE_CACHE="1",
                                           LLM_MOE_USAGE_DIR=tempfile.mkdtemp()), ids)
        self.assertEqual(got, want)


def split_per_expert(folder, style="qwen"):
    """Rewrite the checkpoint's stacked experts as one tensor per expert, the way many conversions ship them
    (mlx-lm stacks them again on load): mlp.experts.N.gate_proj.* — or Mixtral's block_sparse_moe.experts.N.w1.*"""
    path = os.path.join(folder, "model.safetensors")
    out = {}
    for k, v in mx.load(path).items():
        if ".switch_mlp." in k:
            layer, rest = k.split(".switch_mlp.")
            proj, part = rest.split(".")
            for e in range(v.shape[0]):
                name = {"gate_proj": "w1", "down_proj": "w2", "up_proj": "w3"}[proj] if style == "mixtral" else proj
                out[f"{layer}.experts.{e}.{name}.{part}"] = v[e]
        else:
            out[k] = v
    mx.save_safetensors(path, out)


class TestPerExpertCheckpoint(unittest.TestCase):
    """Many MLX conversions ship each expert as its own tensor; the experts must be found there too."""

    @classmethod
    def setUpClass(cls):
        # float experts: Qwen3-MoE's sanitize in mlx-lm stacks only ".weight", so a quantized per-expert file
        # for it cannot be loaded at all; the quantized case is run on a real OLMoE checkpoint (bench_moe.py)
        cls.dir = tempfile.mkdtemp()
        cls.cfg = build(cls.dir, quantize=False)
        cls.eager = load_lazy(cls.dir)
        mx.eval(cls.eager.parameters())
        split_per_expert(cls.dir)

    def test_the_experts_are_found_and_the_model_is_the_same(self):
        m = load_lazy(self.dir)
        st = M.stream_experts(m, self.dir, plan(4))
        self.addCleanup(st.close)
        mx.eval(m.parameters())
        for ids in ([3, 9, 27, 81, 5, 44, 7], [50, 51, 52]):
            got, want = logits(m, ids), logits(self.eager, ids)
            self.assertLess(float(np.abs(got - want).max()), 0.01 * float(np.ptp(want)), str(ids))   # grouped products
            np.testing.assert_array_equal(got.argmax(-1), want.argmax(-1))
        self.assertGreater(st.stats()["misses"], 0)

    def test_the_plan_counts_them_as_experts_and_the_layers_correctly(self):
        os.environ["LLM_MOE_STREAM"] = "on"
        os.environ["LLM_MOE_CACHE"] = "6"
        self.addCleanup(lambda: [os.environ.pop(k, None) for k in ("LLM_MOE_STREAM", "LLM_MOE_CACHE")])
        p = M.plan(self.dir, self.cfg, 0.88, 64 * 2**30)
        self.assertEqual((p["layers"], p["experts"], p["capacity"]), (3, 16, 6))
        experts, rest = M.expert_bytes(self.dir)
        self.assertGreater(experts, rest)                                # most of the file is experts

    def test_mixtral_names_are_understood(self):
        d = tempfile.mkdtemp()
        build(d)
        split_per_expert(d, "mixtral")
        idx = M.index(d)
        parts = M.find_tensors(idx, "model.layers.1.mlp.switch_mlp", "up_proj", 16)
        self.assertEqual(sorted(parts), ["biases", "scales", "weight"])
        self.assertEqual(len(parts["weight"]), 16)
        self.assertTrue(parts["weight"][3][0].endswith("model.safetensors"))
        self.assertIn(".experts.3.w3.weight", [k for k in idx if ".experts.3.w3.weight" in k][0])
        self.assertEqual(M.find_tensors(idx, "model.layers.1.mlp.switch_mlp", "nothing_proj", 16), {})

    def test_experts_that_differ_in_shape_are_refused(self):
        d = tempfile.mkdtemp()
        build(d)
        split_per_expert(d)
        w = dict(mx.load(os.path.join(d, "model.safetensors")))
        w["model.layers.0.mlp.experts.5.up_proj.weight"] = mx.zeros((3, 3), dtype=mx.uint32)
        mx.save_safetensors(os.path.join(d, "model.safetensors"), w)
        with self.assertRaisesRegex(M.StreamError, "differ in type or shape"):
            M.find_tensors(M.index(d), "model.layers.0.mlp.switch_mlp", "up_proj", 16)


REAL = os.path.expanduser(os.environ.get("MOE_TEST_MODEL",
                                         "~/.cache/huggingface/hub/mlx-community/OLMoE-1B-7B-0125-Instruct-4bit"))


@unittest.skipUnless(os.path.isdir(REAL), "no real MoE model (MOE_TEST_MODEL, default OLMoE-1B-7B-4bit): the "
                                          "quantized per-expert layout was not run on real weights")
class TestRealModel(unittest.TestCase):
    """A real quantized checkpoint that ships one tensor per expert: greedy tokens, resident against streamed."""

    def tokens(self, **env):
        from engine.compat import CompatEngine
        old = {k: os.environ.get(k) for k in env}
        os.environ.update(env)
        try:
            e = CompatEngine(REAL, max_context=1024, max_tokens=40, mem_limit=0.95, log=lambda *a: None)
            text, _, _ = e.build_prompt([{"role": "user", "content": "Name three rivers and say where they flow."}],
                                        thinking=False)
            ids, _ = e.encode(text, [])
            out = [t for t, _ in list(e.generate(ids, max_tokens=40, sampling={"temperature": 0}))[:-1] if t is not None]
            stats = e.moe_stats()
            if e.moe:
                e.moe.close()
            return out, stats
        finally:
            for k, v in old.items():
                os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)

    def test_streamed_tokens_are_the_resident_tokens(self):
        want, none = self.tokens(LLM_MOE_STREAM="off")
        self.assertIsNone(none)
        for capacity in ("32", "12"):
            got, s = self.tokens(LLM_MOE_STREAM="on", LLM_MOE_CACHE=capacity, LLM_MOE_USAGE_DIR=tempfile.mkdtemp())
            self.assertEqual(got, want, f"capacity {capacity}")
            self.assertGreater(s["misses"], 0)
            self.assertEqual((s["layers"], s["experts"]), (16, 64))


class TestRefusals(unittest.TestCase):
    def test_a_layout_that_is_not_stacked_is_refused_with_the_tensor_named(self):
        d = tempfile.mkdtemp()
        build(d)
        weights = dict(mx.load(os.path.join(d, "model.safetensors")))
        gone = "model.layers.1.mlp.switch_mlp.up_proj.weight"
        del weights[gone]
        mx.save_safetensors(os.path.join(d, "model.safetensors"), weights)
        m = load_lazy(d) if False else None
        from mlx_lm.utils import load_model
        try:
            m, _ = load_model(Path(d), lazy=True, strict=False)
        except Exception:
            self.skipTest("mlx-lm refuses to build the model without that tensor")
        with self.assertRaisesRegex(M.StreamError, "switch_mlp.up_proj.weight is in neither layout"):
            M.stream_experts(m, d, plan(4))

    def test_a_model_with_no_expert_layers_is_refused(self):
        class Dense(nn.Module):
            def __init__(self):
                super().__init__()
                self.l = nn.Linear(4, 4)
        d = tempfile.mkdtemp()
        mx.save_safetensors(os.path.join(d, "model.safetensors"), {"l.weight": mx.ones((4, 4))})
        with self.assertRaisesRegex(M.StreamError, "nothing to stream"):
            M.stream_experts(Dense(), d, plan(4))


if __name__ == "__main__":
    unittest.main()
