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


def make_bundle(path, template=CHATML, seed=0, generation_config=None):
    """Write an affine-4-bit text-only bundle into `path`. Returns the config dict."""
    os.makedirs(path, exist_ok=True)
    vocab = build_tokenizer(path)
    with open(os.path.join(path, "chat_template.jinja"), "w") as f:
        f.write(template)
    cfg = {
        "model_type": "qwen3_5",
        "quantization": {"bits": BITS, "group_size": GROUP},
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
        wq, sc, bi = mx.quantize(w, group_size=GROUP, bits=BITS)
        t[f"{name}.weight"], t[f"{name}.scales"], t[f"{name}.biases"] = wq, sc, bi

    def vec(name, n, value=1.0):
        t[name] = mx.full((n,), value, dtype=mx.float32)

    L = "language_model.model."
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
    dense("language_model.lm_head", vocab, HIDDEN)
    mx.save_safetensors(os.path.join(path, "model.safetensors"), t)
    return cfg
