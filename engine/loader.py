# Checkpoint loading for Qwen3.5 MLX / JANG bundles.
#
# Weight surgery done here once, at load time:
#   * the per-module Hadamard sign vectors collapse to one vector per width
#     (verified identical; a bundle that differs is rejected);
#   * zero-centered RMSNorm weights get their +1 (JANG layout stamp, or the
#     value heuristic used by vMLX for unstamped bundles);
#   * projections that share an input are concatenated into one matrix.
import glob
import json
import os
import time

import mlx.core as mx
import mlx.nn as nn

from .model import Hadamard, Model

SHIFTED_NORMS = (".input_layernorm.weight", ".post_attention_layernorm.weight",
                 "model.norm.weight", ".q_norm.weight", ".k_norm.weight")
# fused name -> parts, in output order
FUSE = {
    "self_attn.qkv_proj": ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"),
    "mlp.gate_up_proj": ("mlp.gate_proj", "mlp.up_proj"),
    "linear_attn.in_proj_qkvz": ("linear_attn.in_proj_qkv", "linear_attn.in_proj_z"),
    "linear_attn.in_proj_ba": ("linear_attn.in_proj_b", "linear_attn.in_proj_a"),
}
PART_TO_FUSED = {p: (f, i, len(ps)) for f, ps in FUSE.items() for i, p in enumerate(ps)}


def _json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _split_key(k):
    """'language_model.model.layers.3.self_attn.q_proj.scales' ->
    ('language_model.model.layers.3.', 'self_attn.q_proj', 'scales')"""
    head, _, leaf = k.rpartition(".")
    marker = ".layers."
    i = head.find(marker)
    if i < 0:
        return None
    j = head.find(".", i + len(marker))
    return head[:j + 1], head[j + 1:], leaf


def load(path, vision=True, log=print):
    t0 = time.time()
    config = _json(os.path.join(path, "config.json"))
    jang = _json(os.path.join(path, "jang_config.json"))
    if config.get("model_type") not in ("qwen3_5",) and config.get("text_config", {}).get("model_type") not in ("qwen3_5_text", "qwen3_5"):
        raise ValueError(f"unsupported model_type {config.get('model_type')!r} (engine supports qwen3_5)")
    if "text_config" not in config:
        config = {"text_config": config, **{k: v for k, v in config.items() if k != "text_config"}}
    tc = config["text_config"]
    if tc.get("num_experts", 0):
        raise ValueError("MoE Qwen3.5 variants are not supported by this engine")
    qcfg = config.get("quantization") or config.get("quantization_config")
    if not qcfg or "bits" not in qcfg:
        raise ValueError("engine needs an MLX-quantized checkpoint (config.quantization)")

    files = sorted(glob.glob(os.path.join(path, "*.safetensors")))
    index = _json(os.path.join(path, "model.safetensors.index.json")).get("weight_map")
    keys = set(index) if index else {k for f in files for k in mx.load(f).keys()}

    vis = None
    if vision and config.get("vision_config") and any(k.startswith("vision_tower.") for k in keys):
        from .vision import VisionModel
        vis = VisionModel(config["vision_config"], keys, qcfg)
    model = Model(config, qcfg, vis, keys)
    from mlx.utils import tree_flatten
    expected_shapes = {k: v.shape for k, v in tree_flatten(model.parameters())}

    had_cfg = config.get("hadamard") or jang.get("hadamard")
    layout = jang.get("layout", {})
    shift = layout.get("language_norms") == "zero-centered-runtime-plus-one" or config.get("jang_norms_pre_shifted") is False
    shift_decided = bool(layout.get("language_norms")) or "jang_norms_pre_shifted" in config

    signs, pending, loaded = {}, {}, set()
    for f in files:
        shard = mx.load(f)
        out = {}
        for k in list(shard):
            v = shard.pop(k)  # drop the shard's reference so fused parts can be freed early
            if k.startswith("vision_tower.") and vis is None:
                continue
            if "mtp." in k or k.startswith("model.visual"):
                continue
            if k.endswith(".signs"):
                w = v.shape[0]
                if w not in signs:
                    signs[w] = v.astype(mx.float32)
                elif not mx.array_equal(signs[w], v).item():
                    raise ValueError(f"{k}: sign vectors differ within width {w}; unsupported layout")
                continue
            if k.endswith(SHIFTED_NORMS) and v.ndim == 1 and k.startswith("language_model"):
                if not shift_decided:
                    mean = v[:1024].astype(mx.float32).mean().item()
                    shift = mean < (1.5 if k.endswith("model.norm.weight") else 0.5)
                    shift_decided = True
                if shift:
                    v = v + 1.0
            if k.endswith("conv1d.weight") and v.ndim == 3 and v.shape[-1] != 1:
                v = v.moveaxis(2, 1)
            if k.endswith("patch_embed.proj.weight") and v.ndim == 5 and v.shape[-1] != 3:
                v = v.transpose(0, 2, 3, 4, 1)
            sp = _split_key(k)
            if sp and sp[0].startswith("language_model.") and sp[1] in PART_TO_FUSED:
                fused, i, n = PART_TO_FUSED[sp[1]]
                slot = pending.setdefault((sp[0], fused, sp[2]), [None] * n)
                slot[i] = v
                if all(x is not None for x in slot):
                    # materialize now: parts + result live together only for this one tensor
                    fused_arr = mx.concatenate(slot, axis=0)
                    mx.eval(fused_arr)
                    out[f"{sp[0]}{fused}.{sp[2]}"] = fused_arr
                    del pending[(sp[0], fused, sp[2])]
                continue
            out[k] = v
        bad = [k for k, v in out.items() if k in expected_shapes and tuple(v.shape) != tuple(expected_shapes[k])]
        if bad:
            k = bad[0]
            raise ValueError(f"unexpected tensor shape for {k}: {tuple(out[k].shape)} vs {tuple(expected_shapes[k])} "
                             f"(per-module bit widths are not supported yet)")
        model.load_weights(list(out.items()), strict=False)
        loaded.update(out)
        # evaluate only this shard's tensors: model.parameters() would also
        # materialize the zero placeholders of layers not loaded yet
        mx.eval(list(out.values()))
        del shard, out
        mx.clear_cache()

    if pending:
        raise ValueError(f"incomplete fused projections: {list(pending)[:3]}")
    from mlx.utils import tree_flatten
    expected = {k for k, _ in tree_flatten(model.parameters())}
    missing = sorted(expected - loaded)
    if missing:
        raise ValueError(f"{len(missing)} parameters missing from checkpoint, e.g. {missing[:5]}")

    if had_cfg:
        block = int(had_cfg.get("block_size", 1024))
        need = {tc["hidden_size"], tc["intermediate_size"],
                tc["num_attention_heads"] * tc.get("head_dim", tc["hidden_size"] // tc["num_attention_heads"]),
                tc["linear_num_value_heads"] * tc["linear_value_head_dim"]}
        if not need <= set(signs):
            raise ValueError(f"Hadamard bundle is missing sign vectors for widths {sorted(need - set(signs))}")
        if not all(mx.all(mx.abs(s) == 1).item() for s in signs.values()):
            raise ValueError("Hadamard sign vectors are not all +-1")
        model.set_hadamard(Hadamard(block, signs))
    elif signs:
        raise ValueError("bundle ships .signs tensors but declares no Hadamard contract")

    model.eval()
    nbytes = sum(v.nbytes for _, v in tree_flatten(model.parameters()))
    log(f"loaded {os.path.basename(path.rstrip('/'))}: {nbytes / 2**30:.2f} GiB, "
        f"hadamard={'yes' if had_cfg else 'no'}, norm_shift={shift}, vision={'yes' if vis else 'no'}, "
        f"{time.time() - t0:.1f}s")
    return model, config, nbytes


def has_vision(path):
    config = _json(os.path.join(path, "config.json"))
    index = _json(os.path.join(path, "model.safetensors.index.json")).get("weight_map") or {}
    return bool(config.get("vision_config")) and any(k.startswith("vision_tower.") for k in index)


def attach_vision(model, path, log=print):
    """Load the vision tower into an already loaded text model (lazy vision, ~0.5 GB)."""
    from .vision import VisionModel
    t0 = time.time()
    config = model.config
    qcfg = config.get("quantization") or config.get("quantization_config")
    index = _json(os.path.join(path, "model.safetensors.index.json")).get("weight_map") or {}
    keys = {k for k in index if k.startswith("vision_tower.")}
    vis = VisionModel(config["vision_config"], set(index), qcfg)
    weights = {}
    for f in sorted({index[k] for k in keys}):
        shard = mx.load(os.path.join(path, f))
        for k in [k for k in shard if k.startswith("vision_tower.")]:
            v = shard[k]
            if k.endswith("patch_embed.proj.weight") and v.ndim == 5 and v.shape[-1] != 3:
                v = v.transpose(0, 2, 3, 4, 1)
            weights[k.removeprefix("vision_tower.")] = v
    vis.load_weights(list(weights.items()), strict=True)
    mx.eval(vis.parameters())
    vis.eval()
    model.vision_tower = vis
    log(f"vision tower loaded on demand: {sum(v.nbytes for v in weights.values()) / 2**30:.2f} GiB, "
        f"{time.time() - t0:.1f}s")
