"""Inkling 2 on MLX (engine/inkling2, ported from mlx-lm PR #2086): a tiny random model — forward, the cache
against the whole forward (looped too), the Metal kernels against the reference maths, renaming an original
checkpoint, images, and a checkpoint in mlx-community's format loaded through mlx-lm 0.32 via register().
The PR's own tests are the source of most of these. No real model is needed.

    .venv/bin/python -m unittest tests.test_inkling2 -v
"""
import json
import os
import re
import sys
import tempfile
import unittest
import unittest.mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402
from mlx.utils import tree_flatten, tree_unflatten  # noqa: E402

from engine import inkling2  # noqa: E402
from engine.inkling2 import kernels, layers, switch  # noqa: E402

CONFIG = {
    "model_type": "inkling2",
    "image_token_id": 127,
    "text_config": {
        "hidden_size": 64, "num_hidden_layers": 8, "vocab_size": 128, "unpadded_vocab_size": 120,
        "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16, "swa_num_attention_heads": 4,
        "swa_num_key_value_heads": 2, "swa_head_dim": 16, "sliding_window_size": 8,
        "local_layer_ids": [0, 1, 2, 3, 4, 6], "d_rel": 4, "rel_extent": 8, "log_scaling_n_floor": 4,
        "log_scaling_alpha": 0.1, "rms_norm_eps": 1e-6, "qk_l2_clip_threshold": 3.0, "rel_l2_clip_threshold": 2.0,
        "sconv_kernel_size": 4, "n_layers_per_attn_res_block": 3, "n_routed_experts": 4, "num_experts_per_tok": 2,
        "intermediate_size": 64, "mlp_gate_softcap": 4.0, "mlp_up_softcap": 25.0, "route_scale": 2.0,
        "final_logit_softcapping": 30.0,
    },
}
VISION_CONFIG = {"hidden_size": 32, "num_hidden_layers": 2, "num_attention_heads": 2, "native_patch_size": 2,
                 "adapter_merge_size": 2, "posemb_grid_size": 4, "projection_hidden_dim": 48}
LOOP_CONFIG = {**CONFIG, "model_type": "inkling2_loop",
               "text_config": {**CONFIG["text_config"],
                               "loop_config": {"core_start": 3, "core_end": 6, "num_passes": 3, "gate_rank": 8,
                                               "is_varloop": False}}}
UNPADDED_VOCAB = CONFIG["text_config"]["unpadded_vocab_size"]


def create_model(vision=False, config=CONFIG):
    mx.random.seed(0)
    config = {**config, "vision_config": VISION_CONFIG} if vision else config
    model = inkling2.Model(inkling2.ModelArgs.from_dict(config))
    weights = []
    for k, v in tree_flatten(model.parameters()):
        if k.endswith(("norm.weight", "l2_clip.scale", "lambda_1", "lambda_2")):
            v = 1 + 0.1 * mx.random.normal(v.shape)
        else:
            fan_in = v.shape[-1] if v.ndim > 1 else 16
            v = mx.random.normal(v.shape) * fan_in ** -0.5
        weights.append((k, v))
    model.update(tree_unflatten(weights))
    mx.eval(model.parameters())
    return model


def quantize_experts(model, mxfp8_only=False):
    """nvfp4 experts with per-expert scales, mxfp8 in the last layer — as the released Inkling-2-Mini."""
    for i, layer in enumerate(model.model.language_model.layers):
        ex = layer.mlp.experts
        for name in ("gate_up_proj", "down_proj"):
            lin = getattr(ex, name)
            if mxfp8_only or i == 7:
                setattr(ex, name, switch.quantize(lin, 32, 8, "mxfp8"))
            else:
                setattr(ex, name, switch.quantize(lin, 16, 4, "nvfp4", global_scale=True))
    mx.eval(model.parameters())


class TestModel(unittest.TestCase):
    """On the CPU, where float32 is exact enough to compare a chunked run with a whole one: an M5 GPU's float32
    matmul rounds like TF32 (about 2e-2 on a 64-wide product), which a random model amplifies (the GPU paths are
    checked below: the kernels against float64 and against the plain path)."""

    def setUp(self):
        self._device = mx.default_device()
        mx.set_default_device(mx.cpu)
        self.addCleanup(mx.set_default_device, self._device)

    def test_forward(self):
        model = create_model()
        for dtype in (mx.float32, mx.bfloat16):
            model.update(tree_unflatten([(k, v.astype(dtype)) for k, v in tree_flatten(model.parameters())]))
            out = model(mx.array([[0, 1]]))
            self.assertEqual((out.shape, out.dtype), ((1, 2, UNPADDED_VOCAB), mx.float32))

    def test_cached_matches_forward(self):
        model = create_model()
        tokens = mx.random.randint(0, UNPADDED_VOCAB, (2, 20))
        full = model(tokens)
        cache = model.make_cache()
        steps = [model(tokens[:, :6], cache=cache), model(tokens[:, 6:12], cache=cache)]
        steps += [model(tokens[:, i:i + 1], cache=cache) for i in range(12, 20)]
        self.assertTrue(mx.allclose(mx.concatenate(steps, axis=1), full, rtol=1e-4, atol=1e-4))

    def test_loop(self):
        model = create_model(config=LOOP_CONFIG)
        quantize_experts(model, mxfp8_only=True)        # nvfp4 with a global scale runs on the GPU only
        tokens = mx.random.randint(0, UNPADDED_VOCAB, (2, 20))
        full = model(tokens)
        cache = model.make_cache()
        self.assertEqual(len(cache), 3 + 3 * 3 + 2)                     # the core layers are visited three times
        steps = [model(tokens[:, :6], cache=cache), model(tokens[:, 6:12], cache=cache)]
        steps += [model(tokens[:, i:i + 1], cache=cache) for i in range(12, 20)]
        self.assertTrue(mx.allclose(mx.concatenate(steps, axis=1), full, rtol=1e-4, atol=1e-4))

    def test_the_loop_matters(self):
        """A pass more or less changes the output: the loop is really run, not just its first pass."""
        model = create_model(config=LOOP_CONFIG)
        tokens = mx.array([[3, 4, 5, 6]])
        out = model(tokens)
        model.args.text.loop_config["num_passes"] = 2
        self.assertFalse(mx.allclose(model(tokens), out, atol=1e-3))

    def test_embed_inputs(self):
        model = create_model(vision=True)
        vision = model.args.vision
        side = vision.native_patch_size * vision.adapter_merge_size
        h, w = 8, 12
        image = mx.random.uniform(-1, 1, shape=(3, h, w))
        inputs = mx.array([1, 2] + [model.args.image_token_id] * ((h // side) * (w // side)) + [3, 4])
        rows = model.encode_image(image)
        self.assertEqual(rows.shape, (6, CONFIG["text_config"]["hidden_size"]))
        emb = model.embed_inputs(inputs, pixel_values=[image])
        text = model.model.language_model.embed(inputs)
        self.assertTrue(mx.array_equal(emb[2:8], rows))
        self.assertTrue(mx.array_equal(emb[:2], text[:2]) and mx.array_equal(emb[8:], text[8:]))
        out = model(inputs[None], pixel_values=[image])
        self.assertTrue(mx.allclose(out, model(inputs[None], input_embeddings=emb[None])))


@unittest.skipUnless(mx.metal.is_available(), "needs Metal")
class TestKernels(unittest.TestCase):
    def test_relative_attention_matches_the_reference(self):
        mx.random.seed(1)
        B, Hq, Hkv, L, S, D, R = 2, 4, 2, 37, 50, 64, 16
        q = (mx.random.normal((B, Hq, L, D)) * 0.3).astype(mx.bfloat16)
        k = (mx.random.normal((B, Hkv, S, D)) * 0.3).astype(mx.bfloat16)
        v = mx.random.normal((B, Hkv, S, D)).astype(mx.bfloat16)
        rel = mx.random.normal((B, L, Hq, R)).astype(mx.bfloat16)
        sinks = mx.random.normal((B, L, Hq)).astype(mx.bfloat16)
        for window in (0, 12):
            got = kernels.relative_attention(q, k, v, rel, sinks, scale=D ** -0.5, window=window,
                                             key_start=mx.zeros((B,), mx.int32))
            qp = mx.arange(S - L, S)
            want = layers._attention_reference(q, k, v, rel, sinks, scale=D ** -0.5, window=window,
                                               query_positions=qp, key_positions=mx.arange(S))
            self.assertLess(float(mx.max(mx.abs(got.astype(mx.float32) - want.astype(mx.float32)))), 0.05, window)

    def test_depth_attention_matches_float64(self):
        import numpy as np
        mx.random.seed(2)
        for k, q in ((3, 3), (1, 1), (0, 4)):
            blocks, partial = mx.random.normal((k, 2, 5, 96)), mx.random.normal((2, 5, 96))
            queries = mx.random.normal((q, 96))
            got = np.array(kernels.depth_attention(blocks, partial, queries, scale=96 ** -0.5))
            b, p, qq = (np.array(a, np.float64) for a in (blocks, partial, queries))
            st = np.moveaxis(np.concatenate([b, p[None]]), 0, -2)
            lg = st @ qq.T * 96 ** -0.5
            w = np.exp(lg - lg.max(-2, keepdims=True))
            w /= w.sum(-2, keepdims=True)
            self.assertLess(np.abs(got - np.swapaxes(w, -1, -2) @ st).max(), 1e-5, (k, q))

    def test_the_kernels_agree_with_the_plain_path(self):
        """The whole model on the GPU with the depth-attention kernel against the same model without it, in
        float32 (in bf16 a random model differs from itself in float32 by about 1; the relative-attention kernel,
        used for bf16 prompts, is checked on its own above)."""
        for dtype, tol in ((mx.float32, 1e-2),):
            model = create_model()
            model.update(tree_unflatten([(k, v.astype(dtype)) for k, v in tree_flatten(model.parameters())]))
            tokens = mx.random.randint(0, UNPADDED_VOCAB, (1, 40))
            fused = model(tokens)
            with unittest.mock.patch.object(layers, "_use_kernels", return_value=False):
                plain = model(tokens)
            self.assertLess(float(mx.max(mx.abs(fused - plain))), tol, dtype)

    def test_one_token_convolutions_match_the_plain_ops(self):
        """The decode-step kernels against the chain of ops they replace: the same numbers, bit for bit."""
        mx.random.seed(4)
        for dt in (mx.bfloat16, mx.float32):
            x, st = mx.random.normal((2, 1, 64)).astype(dt), mx.random.normal((2, 3, 64)).astype(dt)
            sc = layers.ShortConvolution(64, 4)
            sc.conv1d.weight = mx.random.normal(sc.conv1d.weight.shape).astype(dt)
            dc = layers.DynamicShortConvolution(64, 32, 4, 4)
            dc.static_weight = mx.random.normal((4, 64)).astype(dt)
            dc.weight_proj.weight = dc.weight_proj.weight.astype(dt)
            h = mx.random.normal((2, 1, 32)).astype(dt)
            for conv, args in ((sc, (x, st, None)), (dc, (x, h, st, None))):
                y, state = conv(*args)
                with unittest.mock.patch.object(layers, "_use_kernels", return_value=False):
                    y0, window = conv(*args)
                self.assertLess(float(mx.max(mx.abs(y.astype(mx.float32) - y0.astype(mx.float32)))), 1e-6)
                self.assertTrue(mx.array_equal(state, layers._conv_state(window, None, 3)))

    def test_decode_mask_matches_the_plain_ops(self):
        """One query over a growing cache and over a sliding one that wraps around."""
        from mlx_lm.models.cache import KVCache, RotatingKVCache
        mx.random.seed(5)
        for window, cache in ((0, KVCache()), (8, RotatingKVCache(max_size=8))):
            k = mx.random.normal((1, 2, 5, 16))
            cache.update_and_fetch(k, k)
            for step in range(15):
                k = mx.random.normal((1, 2, 1, 16))
                keys, _ = cache.update_and_fetch(k, k)
                S, end = keys.shape[2], cache.offset
                rel = mx.random.normal((4, 8)).astype(mx.bfloat16)
                rot = cache._idx if isinstance(cache, RotatingKVCache) else -1
                got = kernels.decode_bias_mask(rel, S, end, rot, window=window, dtype=mx.bfloat16)
                d = layers._positions(end - 1, 1).reshape(1) - layers._key_positions(cache, S, 1, end).reshape(S)
                bias = mx.where(d < 8, mx.take(rel, mx.clip(d, 0, 7), axis=-1), 0)
                visible = (d >= 0) & (end - 1 - d >= 0) & ((d < window) if window else True)
                want = mx.where(visible, bias, -mx.inf).astype(mx.bfloat16)[None, :, None, :]
                self.assertTrue(mx.array_equal(got, want), (window, step))

    def test_compiled_step_is_the_same_step(self):
        """One-token steps run each layer's compiled pieces; in bfloat16 they give what the uncompiled path does."""
        model = create_model(config=LOOP_CONFIG)
        model.update(tree_unflatten([(k, v.astype(mx.bfloat16)) for k, v in tree_flatten(model.parameters())]))
        tokens = mx.random.randint(0, UNPADDED_VOCAB, (1, 24))
        taken = []
        step = layers.DecoderLayer._step

        def run():
            cache = model.make_cache()
            out = [model(tokens[:, :5], cache=cache)]
            out += [model(tokens[:, i:i + 1], cache=cache) for i in range(5, 24)]
            return mx.concatenate(out, axis=1)

        with unittest.mock.patch.object(layers.DecoderLayer, "_step",
                                        lambda self, key: taken.append(key) or step(self, key)):
            compiled = run()
        with unittest.mock.patch.object(layers.DecoderLayer, "_step", lambda self, key: None):
            plain = run()
        self.assertEqual(len(taken), 19 * len(model.make_cache()))
        self.assertTrue(mx.array_equal(compiled, plain))

    def test_nvfp4_experts_with_their_own_scales(self):
        mx.random.seed(3)
        lin = switch.SwitchLinear(64, 32, 4)
        lin.weight = lin.weight * mx.array([0.01, 1.0, 30.0, 500.0])[:, None, None]   # very different ranges
        q = switch.quantize(lin, 16, 4, "nvfp4", global_scale=True)
        x = mx.random.normal((4, 1, 1, 64))
        idx = mx.array([[3], [0], [2], [1]])
        got = q(x, idx).reshape(4, -1)
        want = mx.stack([x[i, 0, 0] @ lin.weight[int(idx[i, 0])].T for i in range(4)])
        rel = mx.max(mx.abs(got - want), axis=-1) / mx.max(mx.abs(want), axis=-1)
        self.assertLess(float(rel.max()), 0.2)          # 4-bit; without its own scale an expert is off by orders


class TestCheckpoints(unittest.TestCase):
    def test_original_names_are_renamed(self):
        """sanitize() maps Thinking Machines' tensor names (here: made from ours) back to ours."""
        model = create_model(vision=True, config=LOOP_CONFIG)
        weights = dict(tree_flatten(model.parameters()))
        ckpt = {}
        for k, v in weights.items():
            if k.endswith(".message_input.query"):
                ckpt["model.llm.middle_cycle_cross_pass_queries"] = mx.concatenate([v[:, :1], v[:, 3:]], 1).flatten(0, 1)
                ckpt["model.llm.middle_cycle_cross_pass_kv_queries"] = v[:, 1:3].flatten(0, 1)
                continue
            if m := re.match(r".+\.layers\.(\d+)\.attention_input\.query$", k):
                ckpt[f"model.llm.attn_res.attn_queries.{int(m.group(1)) - 1}"] = v
                continue
            name = (k.replace("model.language_model.embed_tokens.", "model.llm.embed.")
                    .replace("model.language_model.", "model.llm.").replace("lm_head.", "model.llm.unembed."))
            name = (name.replace(".self_attn.q_proj.", ".attn.wq_du.").replace(".mlp_input.query", "#")
                    .replace(".post_attention_layernorm.", ".mlp_norm."))
            if "#" in name:
                layer = re.search(r"layers\.(\d+)", name).group(1)
                name = f"model.llm.attn_res.mlp_queries.{layer}"
            ckpt[name] = v
        back = model.sanitize(ckpt)
        for k in ("model.language_model.layers.2.self_attn.q_proj.weight", "lm_head.weight",
                  "model.language_model.layers.4.attention_input.query", "model.language_model.layers.5.mlp_input.query",
                  "model.language_model.message_input.query", "model.language_model.layers.1.post_attention_layernorm.weight"):
            self.assertTrue(mx.array_equal(back[k], weights[k]), k)
        same = model.sanitize(dict(weights))                             # our names pass through
        self.assertTrue(all(mx.array_equal(same[k], v) for k, v in weights.items()))

    def test_an_mlx_checkpoint_loads_through_mlx_lm(self):
        """What mlx-community ships: affine 4-bit everywhere but the experts (nvfp4 with per-expert scales, mxfp8
        in the last layers), the vision tower and the loop gates — loaded by mlx-lm 0.32 once registered."""
        from mlx_lm.utils import load_model
        model = create_model(vision=True, config=LOOP_CONFIG)
        quantize_experts(model)
        nn.quantize(model, 64, 4, class_predicate=lambda p, m: isinstance(m, (nn.Linear, nn.Embedding))
                    and not p.startswith("model.vision_tower.") and "_message_gate." not in p
                    and m.weight.shape[-1] % 64 == 0)
        mx.eval(model.parameters())
        tokens = mx.array([[5, 6, 7, 8, 9]])
        want = model(tokens)
        inkling2.register()
        with tempfile.TemporaryDirectory() as d:
            mx.save_safetensors(os.path.join(d, "model.safetensors"), dict(tree_flatten(model.parameters())))
            cfg = {**LOOP_CONFIG, "vision_config": VISION_CONFIG,
                   "quantization": {"group_size": 64, "bits": 4, "mode": "affine"}}
            with open(os.path.join(d, "config.json"), "w") as f:
                json.dump(cfg, f)
            from pathlib import Path
            loaded, _ = load_model(Path(d))
        modes = {getattr(loaded.model.language_model.layers[i].mlp.experts.gate_up_proj, "mode", None) for i in (0, 7)}
        self.assertEqual(modes, {"nvfp4", "mxfp8"})
        self.assertTrue(mx.allclose(loaded(tokens), want, rtol=1e-4, atol=1e-4))


class TestStreamedExperts(unittest.TestCase):
    """Experts read from the checkpoint as the router asks (engine/moe_stream.py) give exactly what the model with
    every expert resident gives — nvfp4 experts with their own scales, mxfp8 ones, the gate/up rows interleaved,
    the loop visiting the core layers' experts three times."""

    def test_streamed_matches_resident(self):
        from pathlib import Path

        from mlx_lm.utils import load_model

        from engine import moe_stream as M
        model = create_model(config=LOOP_CONFIG)
        quantize_experts(model)
        tokens = mx.random.randint(0, UNPADDED_VOCAB, (1, 12))
        inkling2.register()
        with tempfile.TemporaryDirectory() as d:
            mx.save_safetensors(os.path.join(d, "model.safetensors"), dict(tree_flatten(model.parameters())))
            with open(os.path.join(d, "config.json"), "w") as f:
                json.dump({**LOOP_CONFIG, "quantization": {"group_size": 64, "bits": 4, "mode": "affine"}}, f)
            resident, _ = load_model(Path(d))
            want_prompt = resident(tokens)
            cache = resident.make_cache()
            want_steps = [resident(tokens[:, :8], cache=cache)] + [resident(tokens[:, i:i + 1], cache=cache)
                                                                   for i in range(8, 12)]
            idx = M.index(d)
            experts, rest = M.expert_bytes(d, idx)
            self.assertGreater(experts, 0)                     # the stacked mlp.experts tensors are seen as experts
            # every expert resident: the prompt and the steps after it are bit for bit the same
            streamed, _ = load_model(Path(d), lazy=True)
            st = M.stream_experts(streamed, d, {"capacity": 4, "experts": 4, "top_k": 2})
            try:
                self.assertEqual(len(st.pools), 8)
                self.assertTrue(mx.array_equal(streamed(tokens), want_prompt))
                cache = streamed.make_cache()
                got = [streamed(tokens[:, :8], cache=cache)] + [streamed(tokens[:, i:i + 1], cache=cache)
                                                                for i in range(8, 12)]
                self.assertTrue(all(mx.array_equal(a, b) for a, b in zip(got, want_steps)))
            finally:
                st.close()
            # half of them: token by token (the direct path), experts evicted and read again, still bit for bit
            # (a prompt split into groups rounds differently, which a random looped model amplifies)
            cache = resident.make_cache()
            want_one = [resident(tokens[:, i:i + 1], cache=cache) for i in range(12)]
            streamed, _ = load_model(Path(d), lazy=True)
            st = M.stream_experts(streamed, d, {"capacity": 2, "experts": 4, "top_k": 2})
            try:
                cache = streamed.make_cache()
                got = [streamed(tokens[:, i:i + 1], cache=cache) for i in range(12)]
                self.assertTrue(all(mx.array_equal(a, b) for a, b in zip(got, want_one)))
                self.assertGreater(st.stats()["misses"], 8)
            finally:
                st.close()
            with unittest.mock.patch.dict(os.environ, {"LLM_MOE_STREAM": "on", "LLM_MOE_CACHE": "2"}):
                plan = M.plan(d, LOOP_CONFIG, 0.9, 16 * 2**30, headroom_gb=0, idx=idx)
            self.assertEqual((plan["layers"], plan["experts"], plan["top_k"], plan["capacity"]), (8, 4, 2, 2))


if __name__ == "__main__":
    unittest.main()


class TestChatFormat(unittest.TestCase):
    """Inkling 2's reply segments rewritten for the server's OutputParser (engine/inkling2/chat.py)."""

    def run_stream(self, reply, thinking, tools=True, pieces=None):
        from engine.inkling2 import chat
        from engine.server import OutputParser
        f = chat.StreamFilter(thinking)
        p = OutputParser(thinking, tools, think_end="</think>", tool_start="<tool_call>", tool_end="</tool_call>")
        out = {"reasoning": "", "content": ""}
        chunks = pieces or [reply[i:i + 3] for i in range(0, len(reply), 3)]       # markers cut anywhere
        for c in chunks:
            for kind, s in p.feed(f.feed(c)):
                out[kind] += s
        for kind, s in p.feed(f.finish()) + p.finish():
            out[kind] += s
        return out, p.tool_text

    def test_reasoning_then_answer(self):
        reply = ("<|message_model|><|content_thinking|>The user greets me.<|end_message|>"
                 "<|message_model|><|content_text|>Hello! How can I help?<|end_message|><|content_model_end_sampling|>")
        for pieces in (None, [reply]):
            out, tool = self.run_stream(reply, True, pieces=pieces)
            self.assertEqual(out, {"reasoning": "The user greets me.", "content": "Hello! How can I help?"})
            self.assertIsNone(tool)

    def test_no_reasoning_when_thinking_is_off_or_skipped(self):
        reply = "<|message_model|><|content_text|>Paris.<|end_message|>"
        self.assertEqual(self.run_stream(reply, False)[0], {"reasoning": "", "content": "Paris."})
        self.assertEqual(self.run_stream(reply, True)[0], {"reasoning": "", "content": "Paris."})
        stray = "<|message_model|><|content_thinking|>hmm<|end_message|><|message_model|><|content_text|>Paris.<|end_message|>"
        self.assertEqual(self.run_stream(stray, False)[0], {"reasoning": "", "content": "Paris."})

    def test_a_tool_call(self):
        from engine.inkling2 import chat
        reply = ('<|message_model|><|content_thinking|>I need the weather.<|end_message|>'
                 '<|message_model|>get_weather<|content_invoke_tool_json|>{"name":"get_weather","args":{"city":"Oslo"}}'
                 '<|end_message|><|content_model_end_sampling|>')
        out, tool = self.run_stream(reply, True)
        self.assertEqual(out, {"reasoning": "I need the weather.", "content": ""})
        calls = chat.parse_tool_calls(tool, [])
        self.assertEqual([(c["function"]["name"], json.loads(c["function"]["arguments"])) for c in calls],
                         [("get_weather", {"city": "Oslo"})])

    def test_effort(self):
        from engine.inkling2 import chat
        self.assertEqual(chat.effort_for(False, "high"), "none")
        self.assertEqual(chat.effort_for(True, "low"), "low")
        self.assertEqual(chat.effort_for(True, None), "high")

    def test_the_template_renders_with_the_effort(self):
        """The model's own chat template (as shipped) with what build_prompt passes."""
        path = os.path.join(os.path.dirname(__file__), "fixtures", "inkling2_chat_template.jinja")
        if not os.path.exists(path):
            self.skipTest("template fixture missing")
        try:
            from transformers.utils.chat_template_utils import _compile_jinja_template
        except ImportError:
            self.skipTest("transformers is not installed")
        tpl = _compile_jinja_template(open(path).read())                 # what apply_chat_template uses
        msgs = [{"role": "user", "content": "hi"},
                {"role": "assistant", "content": "", "tool_calls": [{"id": "1", "function": {"name": "f", "arguments": {"a": 1}}}]},
                {"role": "tool", "tool_call_id": "1", "content": "ok"}]
        off = tpl.render(messages=msgs, reasoning_effort="none")
        self.assertIn("Thinking effort level: 0<|end_message|>", off)
        self.assertIn('<|message_model|>f<|content_invoke_tool_json|>{"name":"f","args":{"a":1}}', off)
        self.assertIn("<|message_tool|>f<|content_text|>ok", off)
        self.assertIn("Thinking effort level: 0.9", tpl.render(messages=msgs[:1], reasoning_effort="high"))


class TestImages(unittest.TestCase):
    def test_preprocess_pads_to_blocks_and_scales_big_images_down(self):
        from PIL import Image
        from engine.inkling2.engine import preprocess
        px, n = preprocess(Image.new("RGB", (100, 50), (255, 0, 0)), 40, 0)
        self.assertEqual((px.shape, n), ((3, 80, 120), 6))
        self.assertAlmostEqual(float(px[0, 0, 0]), 1.0)                    # red, normalised to [-1, 1]
        self.assertLess(float(px[0, 79, 119]), -1.0)                        # padding is -1 on the 0–255 scale
        _, n = preprocess(Image.new("RGB", (4000, 3000)), 40, 1024 * 1024)
        self.assertLessEqual(n * 1600, 1024 * 1024 * 1.2)

    def test_an_image_prompt_runs_through_the_runtime(self):
        from engine.inkling2.engine import InklingRuntime
        model = create_model(vision=True)
        image = mx.random.uniform(-1, 1, shape=(3, 8, 12))
        ids = [1, 2] + [model.args.image_token_id] * 6 + [3, 4]
        rt = InklingRuntime(model, (lambda m: m.make_cache(), lambda c: False, None))
        emb, pos3, delta = rt.embed_media(ids, [image])
        self.assertEqual((emb.shape[:2], pos3, delta), ((1, 10), None, 0))
        cache = rt.make_cache()
        rt.prefill(cache, ids, emb, None, 0, 0, 6)
        got = rt.prefill(cache, ids, emb, None, 0, 6, 10)
        want = model(mx.array([ids]), pixel_values=[image])
        self.assertTrue(mx.allclose(got, want[:, 6:], atol=1e-3))


class TestCliSeesIt(unittest.TestCase):
    def test_the_model_list_shows_reasoning_tools_and_images_on_mlx_lm(self):
        import shutil
        import llm
        d = tempfile.mkdtemp()
        with open(os.path.join(d, "config.json"), "w") as f:
            json.dump({**LOOP_CONFIG, "vision_config": VISION_CONFIG}, f)
        shutil.copy(os.path.join(os.path.dirname(__file__), "fixtures", "inkling2_chat_template.jinja"),
                    os.path.join(d, "chat_template.jinja"))
        with open(os.path.join(d, "model.safetensors"), "wb") as f:
            f.write((2).to_bytes(8, "little") + b"{}")
        m = llm.inspect_model(d)
        self.assertEqual((m["route"], m["thinking"], m["thinking_default"]), ("mlxlm", True, True))
        self.assertEqual(llm.route_features({}, m), ("mlx-lm", ["vision", "thinking", "tools"]))
