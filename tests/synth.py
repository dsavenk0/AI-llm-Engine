"""A tiny synthetic Qwen3.5-shaped bundle for model-free end-to-end tests.

Random weights, a byte-level BPE trained on a few lines, a ChatML template. It exists so
the loader -> Engine -> generate path can be exercised for shapes of model nobody has
lying around (no vision tower, no `enable_thinking` in the template, no generation
config) — outputs are noise, only the plumbing is under test.
"""
import json
import os

import mlx.core as mx
import numpy as np

# hidden/intermediate are multiples of the quantization group (64) and of 128 (PTQ1 blocks)
HIDDEN, INTER, LAYERS = 256, 512, 4
NH, NKV, HD = 2, 1, 128             # attention: nh*hd == hidden
LNV, LNK, LDK, LDV = 2, 1, 128, 128  # GatedDeltaNet
CONV_K = 4
BITS, GROUP = 4, 64

CHATML = (
    "{%- for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{%- endfor %}"
    "{%- if add_generation_prompt %}<|im_start|>assistant\n{%- endif %}"
)
# the same template with the small Qwen3.5 models' convention: reasoning only on request
CHATML_THINKING_OFF_BY_DEFAULT = (
    "{%- for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{%- endfor %}"
    "{%- if add_generation_prompt %}<|im_start|>assistant\n"
    "{%- if enable_thinking is defined and enable_thinking is true %}<think>\n"
    "{%- else %}<think>\n\n</think>\n\n{%- endif %}{%- endif %}"
)


def build_tokenizer(path):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=320, show_progress=False, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<think>", "</think>"])
    tok.train_from_iterator(["the quick brown fox jumps over the lazy dog 0123456789 " * 4,
                             "user assistant system hello world what is two plus two"], trainer)
    tok.save(os.path.join(path, "tokenizer.json"))
    return tok.get_vocab_size()


def uniform(mode="affine", bits=BITS, group=GROUP):
    return lambda name: (mode, bits, group)


def make_bundle(path, template=CHATML, seed=0, generation_config=None, scheme=None, twin=False,
                prefix="language_model."):
    """Write a text-only bundle into `path` and return its config dict.

    scheme(name) -> None (store that module as float) or (mode, bits, group_size); `name` is the
    module's path inside its layer ("mlp.gate_proj", "self_attn.q_proj", ...) or "embed_tokens" /
    "lm_head". Default: everything affine 4-bit / group 64. With twin=True every quantized module is
    stored as its own dequantization in fp16 instead — the same function, computed by a different
    code path, which the quantized bundle must agree with.
    prefix: "language_model." (VLM conversions) or "" (text-only exports)."""
    scheme = scheme or uniform()
    os.makedirs(path, exist_ok=True)
    vocab = build_tokenizer(path)
    with open(os.path.join(path, "chat_template.jinja"), "w") as f:
        f.write(template)
    cfg = {
        "model_type": "qwen3_5",
        "quantization": {"bits": BITS, "group_size": GROUP},   # informational: tensors are what's read
        "text_config": {
            "model_type": "qwen3_5_text", "hidden_size": HIDDEN, "intermediate_size": INTER,
            "num_hidden_layers": LAYERS, "num_attention_heads": NH, "num_key_value_heads": NKV,
            "head_dim": HD, "rms_norm_eps": 1e-6, "vocab_size": vocab, "max_position_embeddings": 4096,
            "linear_num_value_heads": LNV, "linear_num_key_heads": LNK, "linear_key_head_dim": LDK,
            "linear_value_head_dim": LDV, "linear_conv_kernel_dim": CONV_K, "full_attention_interval": 4,
            "tie_word_embeddings": False,
            # rope_dims = HD * 0.25 = 32 -> 16 frequencies, which the sections must sum to
            "rope_parameters": {"rope_theta": 10000.0, "partial_rotary_factor": 0.25,
                                "mrope_section": [6, 5, 5]},
        },
        "eos_token_id": [1, 2],
    }
    with open(os.path.join(path, "config.json"), "w") as f:
        json.dump(cfg, f)
    if generation_config is not None:
        with open(os.path.join(path, "generation_config.json"), "w") as f:
            json.dump(generation_config, f)

    rng = np.random.default_rng(seed)
    t = {}

    def dense(name, out, inp, std=0.05):
        w = mx.array(rng.standard_normal((out, inp)).astype(np.float32) * std).astype(mx.float16)
        rel = name.split("layers.", 1)[-1].split(".", 1)[-1] if "layers." in name else name.split(".", 1)[-1]
        spec = scheme(rel.removeprefix("model."))
        if spec is None:
            t[f"{name}.weight"] = w
            return
        mode, bits, gs = spec
        parts = mx.quantize(w, group_size=gs, bits=bits, mode=mode)
        if twin:  # same weights, stored as the float the quantized module represents
            t[f"{name}.weight"] = mx.dequantize(*parts[:3] if len(parts) > 2 else (parts[0], parts[1], None),
                                                group_size=gs, bits=bits, mode=mode).astype(mx.float16)
            return
        t[f"{name}.weight"], t[f"{name}.scales"] = parts[0], parts[1]
        if len(parts) > 2 and parts[2] is not None:
            t[f"{name}.biases"] = parts[2]

    def vec(name, n, value=1.0):
        t[name] = mx.full((n,), value, dtype=mx.float32)

    L = prefix + "model."
    dense(L + "embed_tokens", vocab, HIDDEN)
    for i in range(LAYERS):
        p = f"{L}layers.{i}."
        vec(p + "input_layernorm.weight", HIDDEN)
        vec(p + "post_attention_layernorm.weight", HIDDEN)
        dense(p + "mlp.gate_proj", INTER, HIDDEN)
        dense(p + "mlp.up_proj", INTER, HIDDEN)
        dense(p + "mlp.down_proj", HIDDEN, INTER)
        if (i + 1) % 4:  # GatedDeltaNet layer
            conv_dim = 2 * LDK * LNK + LDV * LNV
            dense(p + "linear_attn.in_proj_qkv", conv_dim, HIDDEN)
            dense(p + "linear_attn.in_proj_z", LDV * LNV, HIDDEN)
            for n in ("in_proj_b", "in_proj_a"):
                t[p + f"linear_attn.{n}.weight"] = mx.array(
                    rng.standard_normal((LNV, HIDDEN)).astype(np.float32) * 0.05)
            t[p + "linear_attn.conv1d.weight"] = mx.array(
                rng.standard_normal((conv_dim, CONV_K, 1)).astype(np.float32) * 0.1)
            vec(p + "linear_attn.A_log", LNV, 0.0)
            vec(p + "linear_attn.dt_bias", LNV, 1.0)
            vec(p + "linear_attn.norm.weight", LDV)
            dense(p + "linear_attn.out_proj", HIDDEN, LDV * LNV)
        else:
            dense(p + "self_attn.q_proj", NH * HD * 2, HIDDEN)
            dense(p + "self_attn.k_proj", NKV * HD, HIDDEN)
            dense(p + "self_attn.v_proj", NKV * HD, HIDDEN)
            dense(p + "self_attn.o_proj", HIDDEN, NH * HD)
            vec(p + "self_attn.q_norm.weight", HD)
            vec(p + "self_attn.k_norm.weight", HD)
    vec(L + "norm.weight", HIDDEN)
    dense(prefix + "lm_head", vocab, HIDDEN)
    mx.save_safetensors(os.path.join(path, "model.safetensors"), t)
    return cfg


# ───────────────────────── PTQ1_0 bundles (the Nurymanau/Bonsai layout) ─────────────────────────
def _legal_bytes(n_trits):
    """The byte values a PTQ1_0 packer can emit: ceil(v * 256 / 3^n) for every v < 3^n."""
    mod = 3 ** n_trits
    return np.array(sorted({-(-v * 256 // mod) for v in range(mod)}), dtype=np.uint8)


def make_ptq1_bundle(path, seed=0, twin=False, block=128, template=CHATML):
    """A tiny prism_hadamard_qwen35_ptq1_v1 checkpoint: every large matrix PTQ1_0 (or, with
    twin=True, its exact dequantization in fp16 under an ordinary Hadamard config), per-module
    +-1 sign vectors, text-only, `model.*` / `lm_head.*` key names like the real repo."""
    from engine.ptq1 import dequantize_ref
    os.makedirs(path, exist_ok=True)
    vocab = build_tokenizer(path)
    with open(os.path.join(path, "chat_template.jinja"), "w") as f:
        f.write(template)
    rng = np.random.default_rng(seed)
    qs_legal, qh_legal = _legal_bytes(5), _legal_bytes(4)
    t, modules = {}, []
    signs_by_width = {w: rng.choice([-1.0, 1.0], size=w).astype(np.float32) for w in (HIDDEN, INTER, NH * HD)}

    def ptq(name, out, inp, embedding=False):
        nb = inp // 128
        w = np.zeros((out, nb, 28), np.uint8)
        w[:, :, :24] = rng.choice(qs_legal, size=(out, nb, 24))
        w[:, :, 24:26] = rng.choice(qh_legal, size=(out, nb, 2))
        d = (rng.random((out, nb)) * 0.04 + 0.005).astype(np.float16)
        w[:, :, 26:28] = d.view(np.uint8).reshape(out, nb, 2)
        w = w.reshape(out, nb * 28)
        if twin:
            t[f"{name}.weight"] = mx.array(dequantize_ref(w)).astype(mx.float16)
        else:
            t[f"{name}.weight"] = mx.array(w)
            modules.append({"path": name, "block": block, "embedding": embedding, "dtype": "float16"})
        t[f"{name}.signs"] = mx.array(signs_by_width[inp])

    def vec(name, n, value=1.0):
        t[name] = mx.full((n,), value, dtype=mx.float32)

    ptq("model.embed_tokens", vocab, HIDDEN, embedding=True)
    for i in range(LAYERS):
        p = f"model.layers.{i}."
        vec(p + "input_layernorm.weight", HIDDEN, 1.0)
        vec(p + "post_attention_layernorm.weight", HIDDEN, 1.0)
        ptq(p + "mlp.gate_proj", INTER, HIDDEN)
        ptq(p + "mlp.up_proj", INTER, HIDDEN)
        ptq(p + "mlp.down_proj", HIDDEN, INTER)
        if (i + 1) % 4:
            conv_dim = 2 * LDK * LNK + LDV * LNV
            ptq(p + "linear_attn.in_proj_qkv", conv_dim, HIDDEN)
            ptq(p + "linear_attn.in_proj_z", LDV * LNV, HIDDEN)
            for n in ("in_proj_b", "in_proj_a"):
                t[p + f"linear_attn.{n}.weight"] = mx.array(
                    rng.standard_normal((LNV, HIDDEN)).astype(np.float32) * 0.05)
            t[p + "linear_attn.conv1d.weight"] = mx.array(
                rng.standard_normal((conv_dim, CONV_K, 1)).astype(np.float32) * 0.1)
            vec(p + "linear_attn.A_log", LNV, 0.0)
            vec(p + "linear_attn.dt_bias", LNV, 1.0)
            vec(p + "linear_attn.norm.weight", LDV)
            ptq(p + "linear_attn.out_proj", HIDDEN, LDV * LNV)
        else:
            ptq(p + "self_attn.q_proj", NH * HD * 2, HIDDEN)
            ptq(p + "self_attn.k_proj", NKV * HD, HIDDEN)
            ptq(p + "self_attn.v_proj", NKV * HD, HIDDEN)
            ptq(p + "self_attn.o_proj", HIDDEN, NH * HD)
            vec(p + "self_attn.q_norm.weight", HD, 1.0)
            vec(p + "self_attn.k_norm.weight", HD, 1.0)
    vec("model.norm.weight", HIDDEN, 2.0)
    ptq("lm_head", vocab, HIDDEN)

    text_config = {
        "model_type": "qwen3_5_text", "hidden_size": HIDDEN, "intermediate_size": INTER,
        "num_hidden_layers": LAYERS, "num_attention_heads": NH, "num_key_value_heads": NKV, "head_dim": HD,
        "rms_norm_eps": 1e-6, "vocab_size": vocab, "max_position_embeddings": 4096,
        "linear_num_value_heads": LNV, "linear_num_key_heads": LNK, "linear_key_head_dim": LDK,
        "linear_value_head_dim": LDV, "linear_conv_kernel_dim": CONV_K, "full_attention_interval": 4,
        "tie_word_embeddings": False,
        "rope_parameters": {"rope_theta": 10000.0, "partial_rotary_factor": 0.25, "mrope_section": [6, 5, 5]},
    }
    if twin:   # the same function through the ordinary float path: float weights + a Hadamard contract
        cfg = {"model_type": "qwen3_5", "text_config": text_config, "hadamard": {"block_size": block},
               "eos_token_id": [1, 2]}
        # float twin has no PTQ norm policy: PTQ1 norms are final-form, so declare them unshifted
        cfg["jang_norms_pre_shifted"] = True
    else:
        cfg = {"schema": "prism_hadamard_qwen35_ptq1_v1", "model_type": "prism_hadamard_qwen35_ptq1",
               "text_config": text_config, "eos_token_id": [1, 2],
               "quantization": {"format": "PTQ1_0", "group_size": 128, "bytes_per_group": 28},
               "modules": modules}
    with open(os.path.join(path, "config.json"), "w") as f:
        json.dump(cfg, f)
    mx.save_safetensors(os.path.join(path, "model.safetensors"), t)
    return cfg
