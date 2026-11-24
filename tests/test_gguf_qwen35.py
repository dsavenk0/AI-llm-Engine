"""Qwen3.5 GGUF (`qwen35moe`, `qwen35`) on the engine's own MLX decoder (engine/dense.py), against mlx-lm's Qwen3.5
implementation with the same weights.

A tiny model is built with mlx-lm (random weights), then written as a GGUF the way llama.cpp's converter writes
one (convert_hf_to_gguf.py, Qwen3NextModel / _LinearAttentionVReorderBase): A as -exp(A_log), the conv kernel
without its unit axis, and the value heads of every linear-attention tensor moved to ggml's tiled order. The
decoder must give mlx-lm's logits — for a prompt and token by token through the recurrent state.

    .venv/bin/python -m unittest tests.test_gguf_qwen35 -v
"""
import os
import struct
import tempfile
import unittest

import numpy as np

try:
    import mlx.core as mx
    from mlx.utils import tree_flatten
except ImportError:                                       # pragma: no cover
    mx = None

from tests import gguf_synth as S

D, LAYERS, HEADS, KV, HD = 64, 4, 4, 2, 32
NK, NV, DK, CONV = 2, 4, 32, 4          # the Metal delta-rule kernel wants heads of 32+ (real models: 128)
E, TOPK, FF_E, FF_SH, FF = 4, 2, 32, 48, 96


def text_config(moe=True):
    tc = dict(model_type="qwen3_5_moe_text" if moe else "qwen3_5_text", hidden_size=D, num_hidden_layers=LAYERS,
              num_attention_heads=HEADS, num_key_value_heads=KV, head_dim=HD, rms_norm_eps=1e-6, vocab_size=0,
              linear_num_value_heads=NV, linear_num_key_heads=NK, linear_key_head_dim=DK, linear_value_head_dim=DK,
              linear_conv_kernel_dim=CONV, full_attention_interval=4, intermediate_size=FF,
              rope_parameters={"rope_theta": 10000.0, "partial_rotary_factor": 0.25, "type": "default",
                               "mrope_section": [2, 1, 1]},
              tie_word_embeddings=False, max_position_embeddings=512)
    if moe:
        tc.update(num_experts=E, num_experts_per_tok=TOPK, moe_intermediate_size=FF_E,
                  shared_expert_intermediate_size=FF_SH, decoder_sparse_step=1, mlp_only_layers=[], norm_topk_prob=True)
    return tc


def build_reference(vocab, moe=True, seed=0):
    """An mlx-lm Qwen3.5 (MoE) model with random float16-exact weights; returns (model, {name: np.float32})."""
    try:
        from mlx_lm.models import qwen3_5, qwen3_5_moe
    except ImportError:
        raise unittest.SkipTest("mlx-lm is not installed: the Qwen3.5 GGUF port was not compared with it")
    tc = text_config(moe)
    tc["vocab_size"] = vocab
    mod = qwen3_5_moe if moe else qwen3_5
    model = mod.Model(mod.ModelArgs.from_dict({"model_type": "qwen3_5_moe" if moe else "qwen3_5", "text_config": tc}))
    rng = np.random.default_rng(seed)
    new = {}
    for name, v in tree_flatten(model.parameters()):
        shape = v.shape
        if name.endswith("norm.weight"):
            a = 1.0 + rng.standard_normal(shape) * 0.1
        elif name.endswith("A_log"):
            a = rng.standard_normal(shape) * 0.5
        elif name.endswith("dt_bias"):
            a = 1.0 + rng.standard_normal(shape) * 0.2
        elif name.endswith("embed_tokens.weight"):
            a = rng.standard_normal(shape) * 0.5
        elif name.endswith("mlp.gate.weight"):
            a = rng.standard_normal(shape) * 0.5
        else:
            a = rng.standard_normal(shape) * 0.08
        new[name] = a.astype(np.float16).astype(np.float32)        # what an F16 GGUF holds exactly
        if ".switch_mlp." in name:                                  # experts go in as Q8_0 (only those stream)
            new[name] = S.q8_0(new[name].reshape(-1, shape[-1]))[1].reshape(shape)
    model.load_weights([(k, mx.array(v)) for k, v in new.items()])
    mx.eval(model.parameters())
    return model, new


def _reorder_v(t, dim, head_dim):
    """llama.cpp's _reorder_v_heads: value heads grouped by key head → ggml's tiled order."""
    shape = list(t.shape)
    dim = dim % len(shape)
    t = t.reshape(shape[:dim] + [NK, NV // NK, head_dim] + shape[dim + 1:])
    t = np.swapaxes(t, dim, dim + 1)
    return np.ascontiguousarray(t).reshape(shape)


def to_gguf_tensors(p, moe=True):
    """{gguf name: float32 array in numpy (row-major) shape} as llama.cpp's converter would write them."""
    out = {}
    pre = "language_model.model."
    out["token_embd.weight"] = p[pre + "embed_tokens.weight"]
    out["output_norm.weight"] = p[pre + "norm.weight"]
    out["output.weight"] = p["language_model.lm_head.weight"]
    key_dim = NK * DK
    for i in range(LAYERS):
        L = f"{pre}layers.{i}."
        b = f"blk.{i}."
        out[b + "attn_norm.weight"] = p[L + "input_layernorm.weight"]
        out[b + "post_attention_norm.weight"] = p[L + "post_attention_layernorm.weight"]
        if (i + 1) % 4:
            a = L + "linear_attn."
            qkv = p[a + "in_proj_qkv.weight"]
            out[b + "attn_qkv.weight"] = np.concatenate([qkv[:2 * key_dim], _reorder_v(qkv[2 * key_dim:], 0, DK)])
            out[b + "attn_gate.weight"] = _reorder_v(p[a + "in_proj_z.weight"], 0, DK)
            out[b + "ssm_beta.weight"] = _reorder_v(p[a + "in_proj_b.weight"], 0, 1)
            out[b + "ssm_alpha.weight"] = _reorder_v(p[a + "in_proj_a.weight"], 0, 1)
            out[b + "ssm_dt.bias"] = _reorder_v(p[a + "dt_bias"][:, None], 0, 1)[:, 0]
            out[b + "ssm_a"] = -np.exp(_reorder_v(p[a + "A_log"][:, None], 0, 1)[:, 0])
            conv = p[a + "conv1d.weight"][:, :, 0]                                   # [channels, K]
            out[b + "ssm_conv1d.weight"] = np.concatenate([conv[:2 * key_dim], _reorder_v(conv[2 * key_dim:], 0, DK)])
            out[b + "ssm_norm.weight"] = p[a + "norm.weight"]
            out[b + "ssm_out.weight"] = _reorder_v(p[a + "out_proj.weight"], 1, DK)
        else:
            a = L + "self_attn."
            for src, dst in (("q_proj", "attn_q"), ("k_proj", "attn_k"), ("v_proj", "attn_v"), ("o_proj", "attn_output"),
                             ("q_norm", "attn_q_norm"), ("k_norm", "attn_k_norm")):
                out[b + dst + ".weight"] = p[a + src + ".weight"]
        m = L + "mlp."
        if moe:
            out[b + "ffn_gate_inp.weight"] = p[m + "gate.weight"]
            for n in ("gate", "up", "down"):
                out[b + f"ffn_{n}_exps.weight"] = p[m + f"switch_mlp.{n}_proj.weight"]
                out[b + f"ffn_{n}_shexp.weight"] = p[m + f"shared_expert.{n}_proj.weight"]
            out[b + "ffn_gate_inp_shexp.weight"] = p[m + "shared_expert_gate.weight"][0]
        else:
            for n in ("gate", "up", "down"):
                out[b + f"ffn_{n}.weight"] = p[m + f"{n}_proj.weight"]
    return out


def write_gguf(path, tensors, moe=True):
    arch = "qwen35moe" if moe else "qwen35"
    tokens, types, merges = S.vocab()
    meta = {
        "general.architecture": arch, "general.name": "tiny-qwen35", "general.file_type": 1,
        f"{arch}.context_length": 512, f"{arch}.embedding_length": D, f"{arch}.block_count": LAYERS,
        f"{arch}.attention.head_count": HEADS, f"{arch}.attention.head_count_kv": KV,
        f"{arch}.attention.key_length": HD, f"{arch}.attention.value_length": HD,
        f"{arch}.attention.layer_norm_rms_epsilon": 1e-6, f"{arch}.rope.freq_base": 10000.0,
        f"{arch}.rope.dimension_count": HD // 4, f"{arch}.rope.dimension_sections": [2, 1, 1, 0],
        f"{arch}.ssm.conv_kernel": CONV, f"{arch}.ssm.state_size": DK, f"{arch}.ssm.group_count": NK,
        f"{arch}.ssm.time_step_rank": NV, f"{arch}.ssm.inner_size": NV * DK, f"{arch}.full_attention_interval": 4,
        "tokenizer.ggml.model": "gpt2", "tokenizer.ggml.pre": "qwen35", "tokenizer.ggml.tokens": tokens,
        "tokenizer.ggml.token_type": types, "tokenizer.ggml.merges": merges,
        "tokenizer.ggml.bos_token_id": tokens.index("<|bos|>"), "tokenizer.ggml.eos_token_id": tokens.index("<|im_end|>"),
        "tokenizer.chat_template": S.CHATML,
    }
    if moe:
        meta.update({f"{arch}.expert_count": E, f"{arch}.expert_used_count": TOPK,
                     f"{arch}.expert_feed_forward_length": FF_E, f"{arch}.expert_shared_feed_forward_length": FF_SH})
    else:
        meta[f"{arch}.feed_forward_length"] = FF
    entries = []
    for name, a in tensors.items():
        dims = tuple(reversed(a.shape))                     # ggml lists the fastest dimension first
        if a.ndim == 1:
            entries.append((name, dims, 0, a.astype(np.float32).tobytes()))
        elif "_exps." in name:
            entries.append((name, dims, 8, S.q8_0(a.reshape(-1, a.shape[-1]))[0]))
        else:
            entries.append((name, dims, 1, a.astype(np.float16).tobytes()))
    head = b"GGUF" + struct.pack("<IQQ", 3, len(entries), len(meta)) + b"".join(S._kv(k, v) for k, v in meta.items())
    infos, offset, blobs = b"", 0, []
    for name, dims, gtype, raw in entries:
        nb = name.encode()
        infos += struct.pack("<Q", len(nb)) + nb + struct.pack("<I", len(dims)) + struct.pack(f"<{len(dims)}Q", *dims)
        infos += struct.pack("<IQ", gtype, offset)
        pad = (-len(raw)) % 32
        blobs.append(raw + b"\0" * pad)
        offset += len(raw) + pad
    pre = head + infos
    pre += b"\0" * ((-len(pre)) % 32)
    with open(path, "wb") as f:
        f.write(pre + b"".join(blobs))


@unittest.skipIf(mx is None, "MLX is not installed")
class _Base(unittest.TestCase):
    moe = True

    @classmethod
    def setUpClass(cls):
        tokens, _, _ = S.vocab()
        cls.ref, params = build_reference(len(tokens), cls.moe)
        cls.dir = tempfile.mkdtemp()
        cls.path = os.path.join(cls.dir, "tiny-qwen35.gguf")
        write_gguf(cls.path, to_gguf_tensors(params, cls.moe), cls.moe)

    def ours(self, **kw):
        from engine import dense
        model, cfg, _ = dense.build(self.path, log=lambda *a: None, **kw)
        self.addCleanup(model.close)
        return model, cfg

    def run_both(self, model, prompt, steps):
        """Logits of a prompt then of each following token, from ours and from mlx-lm, side by side."""
        ca, cb = model.make_cache(), self.ref.make_cache()
        pairs = [(model(mx.array([prompt]), cache=ca), self.ref(mx.array([prompt]), cache=cb))]
        for t in steps:
            pairs.append((model(mx.array([[t]]), cache=ca), self.ref(mx.array([[t]]), cache=cb)))
        out = []
        for a, b in pairs:
            mx.eval(a, b)
            out.append((np.asarray(a.astype(mx.float32))[0], np.asarray(b.astype(mx.float32))[0]))
        return out

    def assertClose(self, got, want, what):
        # the decoder feeds float16 into its matmuls (like ggml), mlx-lm runs these float32 weights in float32:
        # rounding, well under the spread of the logits, and the same choice of next token
        spread = float(np.ptp(want))
        self.assertLess(float(np.abs(got - want).max()), 0.02 * spread, what)
        np.testing.assert_array_equal(got.argmax(-1), want.argmax(-1), what)


class TestQwen35MoE(_Base):
    def test_the_file_is_recognised_as_a_hybrid_moe(self):
        from engine import gguf
        self.assertEqual(gguf.native_support(self.path), (True, ""))
        model, cfg = self.ours(stream_experts=False)
        self.assertTrue(model.hybrid)
        self.assertEqual([L.recurrent for L in model.layers], [True, True, True, False])
        self.assertEqual((cfg.ssm_nk, cfg.ssm_nv, cfg.n_expert, cfg.n_expert_used), (NK, NV, E, TOPK))

    def test_prompt_and_decoding_match_mlx_lm(self):
        model, _ = self.ours(stream_experts=False)
        for i, (got, want) in enumerate(self.run_both(model, [5, 17, 99, 3, 250, 42, 7], [8, 61, 200, 4, 33])):
            self.assertClose(got, want, f"step {i}")

    def test_streamed_experts_give_the_same_logits(self):
        resident, _ = self.ours(stream_experts=False)
        streamed, _ = self.ours(stream_experts=True, expert_cache=2)      # 2 of 4 experts, 2 used per token
        prompt, steps = [5, 17, 99, 3, 250, 42, 7, 11, 12], [8, 61, 200]
        ca, cb = resident.make_cache(), streamed.make_cache()
        for ids in [prompt] + [[t] for t in steps]:
            a, b = resident(mx.array([ids]), cache=ca), streamed(mx.array([ids]), cache=cb)
            mx.eval(a, b)
            np.testing.assert_allclose(np.asarray(b), np.asarray(a), atol=2e-3, rtol=0)
        self.assertGreater(streamed.moe_stats()["misses"], 0)

    def test_the_cache_is_recurrent_where_the_layer_is(self):
        from engine.cache import KVCache, SSMCache
        model, _ = self.ours(stream_experts=False)
        cache = model.make_cache()
        self.assertEqual([type(c) for c in cache], [SSMCache, SSMCache, SSMCache, KVCache])
        model(mx.array([[1, 2, 3]]), cache=cache)
        self.assertEqual([c.offset for c in cache], [3, 3, 3, 3])
        self.assertFalse(cache[0].is_trimmable())


class TestQwen35Dense(_Base):
    moe = False

    def test_prompt_and_decoding_match_mlx_lm(self):
        from engine import gguf
        self.assertEqual(gguf.native_support(self.path), (True, ""))
        model, _ = self.ours()
        for i, (got, want) in enumerate(self.run_both(model, [5, 17, 99, 3, 250], [8, 61, 200])):
            self.assertClose(got, want, f"step {i}")


def write_projector(path, kind):
    """A header-only mmproj GGUF that declares its projector type."""
    meta = {"general.architecture": "clip", "clip.projector_type": kind}
    head = b"GGUF" + struct.pack("<IQQ", 3, 0, len(meta)) + b"".join(S._kv(k, v) for k, v in meta.items())
    with open(path, "wb") as f:
        f.write(head)


class TestProjectorRoute(unittest.TestCase):
    def test_the_qwen3vl_projector_is_ours_another_goes_to_llama_cpp(self):
        from engine import gguf
        d = tempfile.mkdtemp()
        model = os.path.join(d, "m.gguf")
        open(model, "wb").close()
        self.assertFalse(gguf.native_vision(model))                      # no projector at all
        write_projector(os.path.join(d, "mmproj-F16.gguf"), "qwen3vl_merger")
        self.assertTrue(gguf.native_vision(model))
        write_projector(os.path.join(d, "mmproj-F16.gguf"), "gemma3")
        self.assertFalse(gguf.native_vision(model))


REAL = os.path.expanduser("~/.cache/huggingface/hub/unsloth/Qwen3.5-0.8B-GGUF")


@unittest.skipUnless(mx is not None and os.path.isfile(os.path.join(REAL, "Qwen3.5-0.8B-Q8_0.gguf"))
                     and os.path.isfile(os.path.join(REAL, "mmproj-F16.gguf")),
                     "Qwen3.5-0.8B Q8_0 GGUF + mmproj-F16 not downloaded: native GGUF vision was not covered")
class TestRealVision(unittest.TestCase):
    """The real file and its projector: the answer llama.cpp's mtmd gives too ("A red square and a blue circle.")."""

    def test_an_image_is_read_without_restarting_and_text_works_after(self):
        from PIL import Image, ImageDraw
        from engine.gguf_engine import GGUFEngine
        d = tempfile.mkdtemp()
        img = Image.new("RGB", (448, 320), "white")
        draw = ImageDraw.Draw(img)
        draw.rectangle([40, 60, 180, 200], fill=(220, 20, 20))
        draw.ellipse([250, 80, 400, 230], fill=(20, 60, 220))
        img.save(os.path.join(d, "shapes.png"))
        e = GGUFEngine(os.path.join(REAL, "Qwen3.5-0.8B-Q8_0.gguf"), max_context=4096, max_tokens=60, mem_limit=0.95,
                       disk_cache_dir=d, log=lambda *a: None)
        self.assertTrue(e.vision_allowed)
        q = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": os.path.join(d, "shapes.png")}},
                                          {"type": "text", "text": "What shapes and colors are in this image? "
                                                                   "Answer in one sentence."}]}]
        text, images, _ = e.build_prompt(q, thinking=False)
        ids, media = e.encode(text, images)
        *pieces, _ = list(e.generate(ids, media=media, sampling={"temperature": 0}, max_tokens=40))
        answer = "".join(p for _, p in pieces).lower()
        self.assertIn("red", answer)
        self.assertIn("blue", answer)
        text, images, _ = e.build_prompt([{"role": "user", "content": "Say hello in one word."}], thinking=False)
        ids, media = e.encode(text, images)
        *pieces, _ = list(e.generate(ids, media=media, sampling={"temperature": 0}, max_tokens=8))
        self.assertIn("hello", "".join(p for _, p in pieces).lower())
        e.sleep()


if __name__ == "__main__":
    unittest.main()
