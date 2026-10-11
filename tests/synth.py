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


def build_tokenizer(path, extra_special=()):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=320, show_progress=False, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<think>", "</think>", *extra_special])
    tok.train_from_iterator(["the quick brown fox jumps over the lazy dog 0123456789 " * 4,
                             "user assistant system hello world what is two plus two"], trainer)
    tok.save(os.path.join(path, "tokenizer.json"))
    return tok.get_vocab_size()


def uniform(mode="affine", bits=BITS, group=GROUP):
    return lambda name: (mode, bits, group)


def make_bundle(path, template=CHATML, seed=0, generation_config=None, scheme=None, twin=False,
                prefix="language_model.", extra_special=(), attn_interval=4):
    """Write a text-only bundle into `path` and return its config dict.

    scheme(name) -> None (store that module as float) or (mode, bits, group_size); `name` is the
    module's path inside its layer ("mlp.gate_proj", "self_attn.q_proj", ...) or "embed_tokens" /
    "lm_head". Default: everything affine 4-bit / group 64. With twin=True every quantized module is
    stored as its own dequantization in fp16 instead — the same function, computed by a different
    code path, which the quantized bundle must agree with.
    prefix: "language_model." (VLM conversions) or "" (text-only exports)."""
    scheme = scheme or uniform()
    os.makedirs(path, exist_ok=True)
    vocab = build_tokenizer(path, extra_special)
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
            "linear_value_head_dim": LDV, "linear_conv_kernel_dim": CONV_K, "full_attention_interval": attn_interval,
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
        if (i + 1) % attn_interval:  # GatedDeltaNet layer
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


def make_ptq1_bundle(path, seed=0, twin=False, block=128, template=CHATML, fmt="ptq1"):
    """A tiny prism_hadamard_qwen35_ptq1_v1 checkpoint: every large matrix PTQ1_0 (or, with
    twin=True, its exact dequantization in fp16 under an ordinary Hadamard config), per-module
    +-1 sign vectors, text-only, `model.*` / `lm_head.*` key names like the real repo.

    fmt="tq1" writes the TQ1_G128 layout of the inductiveML "ternel" bundles instead: plain base-3
    payload bytes stored as `.codes` [tile][group][slot][row] plus raw fp16 `.scales`, with the
    `language_model.` key prefix and model_type ternel_hadamard_qwen35."""
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
        if fmt == "tq1":
            digits = rng.integers(0, 3, size=(out, nb, 26, 5))
            digits[:, :, 25, 3:] = 0                      # the tail byte holds weights 125..127 only
            w[:, :, :26] = (digits * np.array([1, 3, 9, 27, 81])).sum(-1).astype(np.uint8)
        else:
            w[:, :, :24] = rng.choice(qs_legal, size=(out, nb, 24))
            w[:, :, 24:26] = rng.choice(qh_legal, size=(out, nb, 2))
        d = (rng.random((out, nb)) * 0.04 + 0.005).astype(np.float16)
        w[:, :, 26:28] = d.view(np.uint8).reshape(out, nb, 2)
        rows_blocks = w.reshape(out, nb * 28)
        key = f"language_model.{name}" if fmt == "tq1" else name
        if twin:
            t[f"{name}.weight"] = mx.array(dequantize_ref(rows_blocks, fmt)).astype(mx.float16)
        elif fmt == "tq1":
            tile = 256 if out % 256 == 0 else out        # the artifact never pads rows
            tiles = out // tile
            t[f"{key}.codes"] = mx.array(np.ascontiguousarray(
                w[:, :, :26].reshape(tiles, tile, nb, 26).transpose(0, 2, 3, 1)))
            t[f"{key}.scales"] = mx.array(np.ascontiguousarray(
                d.view(np.uint16).reshape(tiles, tile, nb).transpose(0, 2, 1)))
            modules.append({"path": name, "block": block, "embedding": embedding, "dtype": "float16"})
        else:
            t[f"{name}.weight"] = mx.array(rows_blocks)
            modules.append({"path": name, "block": block, "embedding": embedding, "dtype": "float16"})
        t[f"{key}.signs" if (fmt == "tq1" and not twin) else f"{name}.signs"] = mx.array(signs_by_width[inp])

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
    elif fmt == "tq1":
        cfg = {"model_type": "ternel_hadamard_qwen35", "base_model_type": "qwen3_5", "schema_version": 2,
               "text_config": text_config, "eos_token_id": [1, 2], "gdn_activation_layout": "grouped",
               "tensor_namespace": "mlx-vlm-qwen3_5", "modules": modules}
    else:
        cfg = {"schema": "prism_hadamard_qwen35_ptq1_v1", "model_type": "prism_hadamard_qwen35_ptq1",
               "text_config": text_config, "eos_token_id": [1, 2],
               "quantization": {"format": "PTQ1_0", "group_size": 128, "bytes_per_group": 28},
               "modules": modules}
    with open(os.path.join(path, "config.json"), "w") as f:
        json.dump(cfg, f)
    mx.save_safetensors(os.path.join(path, "model.safetensors"), t)
    return cfg


# ───────────────────────── vision bundles ─────────────────────────
VISION_TEMPLATE = (
    "{%- for m in messages %}<|im_start|>{{ m.role }}\n"
    "{%- if m.content is string %}{{ m.content }}"
    "{%- else %}{%- for it in m.content %}"
    "{%- if it.type == 'text' %}{{ it.text }}"
    "{%- elif it.type == 'image' or it.type == 'image_url' or it.image_url is defined %}"
    "<|vision_start|><|image_pad|><|vision_end|>{%- endif %}"
    "{%- endfor %}{%- endif %}<|im_end|>\n{%- endfor %}"
    "{%- if add_generation_prompt %}<|im_start|>assistant\n{%- endif %}"
)
VISION = {"depth": 2, "hidden_size": 64, "num_heads": 2, "intermediate_size": 128, "patch_size": 8,
          "temporal_patch_size": 2, "in_channels": 3, "spatial_merge_size": 2, "num_position_embeddings": 16}


def make_vision_bundle(path, seed=0, quantized_tower=False, template=VISION_TEMPLATE):
    """A text bundle plus a tiny vision tower (random weights) laid out the way converted Qwen3.5
    checkpoints are: vision_tower.* tensors in the same shard, an index, preprocessor_config.json."""
    cfg = make_bundle(path, template, seed=seed, extra_special=["<|image_pad|>", "<|vision_start|>", "<|vision_end|>"])
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(path, "tokenizer.json"))
    vc = dict(VISION, out_hidden_size=HIDDEN)
    cfg["vision_config"] = vc
    cfg["image_token_id"] = tok.token_to_id("<|image_pad|>")
    cfg["vision_start_token_id"] = tok.token_to_id("<|vision_start|>")
    cfg["vision_end_token_id"] = tok.token_to_id("<|vision_end|>")
    with open(os.path.join(path, "config.json"), "w") as f:
        json.dump(cfg, f)
    with open(os.path.join(path, "preprocessor_config.json"), "w") as f:
        json.dump({"patch_size": vc["patch_size"], "temporal_patch_size": vc["temporal_patch_size"],
                   "merge_size": vc["spatial_merge_size"], "image_mean": [0.5] * 3, "image_std": [0.5] * 3,
                   "size": {"shortest_edge": 16 * 16, "longest_edge": 128 * 128}}, f)

    rng = np.random.default_rng(seed + 1000)
    d, m, hid = vc["hidden_size"], vc["spatial_merge_size"], vc["intermediate_size"]
    t = dict(mx.load(os.path.join(path, "model.safetensors")))

    def w(shape, std=0.05):
        return mx.array(rng.standard_normal(shape).astype(np.float32) * std).astype(mx.float16)

    def linear(name, out, inp, std=0.05):
        wt = w((out, inp), std)
        if quantized_tower:
            parts = mx.quantize(wt, group_size=GROUP, bits=BITS)
            t[f"{name}.weight"], t[f"{name}.scales"], t[f"{name}.biases"] = parts
        else:
            t[f"{name}.weight"] = wt
        t[f"{name}.bias"] = w((out,), 0.01)

    def norm(name, n):
        t[f"{name}.weight"] = mx.ones((n,), dtype=mx.float16)
        t[f"{name}.bias"] = mx.zeros((n,), dtype=mx.float16)

    pe = vc["patch_size"]
    t["vision_tower.patch_embed.proj.weight"] = w((d, vc["in_channels"], vc["temporal_patch_size"], pe, pe), 0.1)
    t["vision_tower.patch_embed.proj.bias"] = w((d,), 0.01)
    t["vision_tower.pos_embed.weight"] = w((vc["num_position_embeddings"], d), 0.1)
    for i in range(vc["depth"]):
        b = f"vision_tower.blocks.{i}"
        norm(f"{b}.norm1", d)
        norm(f"{b}.norm2", d)
        # strong q/k weights: attention must be peaked, or positions (rotary, interpolation) cannot matter
        linear(f"{b}.attn.qkv", 3 * d, d, std=0.5)
        linear(f"{b}.attn.proj", d, d)
        linear(f"{b}.mlp.linear_fc1", hid, d)
        linear(f"{b}.mlp.linear_fc2", d, hid)
    norm("vision_tower.merger.norm", d)
    linear("vision_tower.merger.linear_fc1", d * m * m, d * m * m)
    linear("vision_tower.merger.linear_fc2", HIDDEN, d * m * m)
    mx.save_safetensors(os.path.join(path, "model.safetensors"), t)
    with open(os.path.join(path, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {}, "weight_map": {k: "model.safetensors" for k in t}}, f)
    return cfg


def png_data_uri(width=48, height=40, seed=0):
    """A small deterministic RGB test image as a data: URI."""
    import base64
    import io
    from PIL import Image
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 255, (height, width, 3), dtype=np.uint8)
    arr[:, : width // 2, 0] = 255          # a coarse structure, so different seeds give different pictures
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def make_qwen3_bundle(path, template=CHATML, seed=0, layers=2):
    """A plain Qwen3 (full attention everywhere, float weights) — an architecture the native engine does
    not run and mlx-lm does: the stand-in for 'older Qwen' / Llama-style models on the compat backend."""
    os.makedirs(path, exist_ok=True)
    vocab = build_tokenizer(path)
    with open(os.path.join(path, "chat_template.jinja"), "w") as f:
        f.write(template)
    hid, inter, nh, nkv, hd = 64, 128, 4, 2, 16
    cfg = {"model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"], "hidden_size": hid,
           "intermediate_size": inter, "num_hidden_layers": layers, "num_attention_heads": nh,
           "num_key_value_heads": nkv, "head_dim": hd, "rms_norm_eps": 1e-6, "vocab_size": vocab,
           "max_position_embeddings": 2048, "rope_theta": 10000.0, "tie_word_embeddings": False,
           "eos_token_id": [1, 2]}
    with open(os.path.join(path, "config.json"), "w") as f:
        json.dump(cfg, f)
    rng = np.random.default_rng(seed)
    t = {}

    def w(name, *shape, std=0.15):     # strong enough that the argmax has a margin fp16 noise cannot flip
        t[name] = mx.array(rng.standard_normal(shape).astype(np.float32) * std).astype(mx.float16)

    def ones(name, n):
        t[name] = mx.ones((n,), dtype=mx.float16)

    w("model.embed_tokens.weight", vocab, hid, std=0.6)
    for i in range(layers):
        b = f"model.layers.{i}."
        w(b + "self_attn.q_proj.weight", nh * hd, hid)
        w(b + "self_attn.k_proj.weight", nkv * hd, hid)
        w(b + "self_attn.v_proj.weight", nkv * hd, hid)
        w(b + "self_attn.o_proj.weight", hid, nh * hd)
        ones(b + "self_attn.q_norm.weight", hd)
        ones(b + "self_attn.k_norm.weight", hd)
        w(b + "mlp.gate_proj.weight", inter, hid)
        w(b + "mlp.up_proj.weight", inter, hid)
        w(b + "mlp.down_proj.weight", hid, inter)
        ones(b + "input_layernorm.weight", hid)
        ones(b + "post_attention_layernorm.weight", hid)
    ones("model.norm.weight", hid)
    w("lm_head.weight", vocab, hid, std=0.6)
    mx.save_safetensors(os.path.join(path, "model.safetensors"), t, metadata={"format": "mlx"})
    return cfg
