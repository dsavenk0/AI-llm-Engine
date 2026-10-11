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

from . import family
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


_ST_DTYPES = {"F32": mx.float32, "F16": mx.float16, "BF16": mx.bfloat16, "U8": mx.uint8, "U32": mx.uint32,
              "I32": mx.int32, "I8": mx.int8, "BOOL": mx.bool_, "F64": mx.float32, "U16": mx.uint16}


def canonical(k):
    """Text-only exports name the language model `model.*` / `lm_head.*`; VLM conversions put it
    under `language_model.`. The engine's modules use the latter, so map the former onto it."""
    if k.startswith(("model.", "lm_head.")) and not k.startswith("model.visual"):
        return "language_model." + k
    return k


def read_shapes(files):
    """{canonical tensor name: (shape, dtype)} from the safetensors headers alone — nothing is
    read or mapped, so this is cheap even for a multi-GB checkpoint."""
    import struct
    shapes = {}
    for f in files:
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            header = json.loads(fh.read(n))
        for k, t in header.items():
            if k != "__metadata__":
                shapes[canonical(k)] = (tuple(t["shape"]), _ST_DTYPES.get(t["dtype"]))
    return shapes


TQ1_MODEL_TYPE = "ternel_hadamard_qwen35"


def _join_tq1(shard):
    """TQ1_G128 stores each matrix as `X.codes` (+ `X.scales`, `X.signs`); JANG's packed ternary as
    uint8 `X.weight` + float16 `X.scales`. Replace every such pair in this shard by one row-major
    `X.weight` of 28-byte blocks, the form PTQ1_0 bundles use, so fusing, the repack and the
    kernels need no second path."""
    from .ptq1 import from_packed26, from_tq1, is_packed26
    for raw in [k for k in shard if k.endswith(".weight")]:
        base = raw[:-len(".weight")]
        s = shard.get(base + ".scales")
        w = shard[raw]
        if s is not None and is_packed26(w.shape, w.dtype, s.shape, s.dtype):
            joined = from_packed26(shard.pop(raw), shard.pop(base + ".scales"))
            mx.eval(joined)
            shard[raw] = joined
    for raw in [k for k in shard if k.endswith(".codes")]:
        base = raw[:-len(".codes")]
        if base + ".scales" not in shard:
            raise ValueError(f"{base}: TQ1 codes without their scales in the same shard")
        weight = from_tq1(shard.pop(raw), shard.pop(base + ".scales"))
        mx.eval(weight)
        shard[base + ".weight"] = weight


def load(path, vision=True, log=print, ptq1=False, steer=None):
    """`steer`: (direction file or folder, alpha or None) — an activation edit, see engine/steering.py."""
    t0 = time.time()
    config = _json(os.path.join(path, "config.json"))
    jang = _json(os.path.join(path, "jang_config.json"))
    if not family.native_capable(config) and config.get("model_type") != TQ1_MODEL_TYPE:
        raise ValueError(f"unsupported model_type {config.get('model_type')!r}: the native engine runs the Qwen3.5 "
                         f"hybrid architecture (GatedDeltaNet + attention, dense); other families need the mlx-lm backend")
    if "text_config" not in config:
        config = {"text_config": config, **{k: v for k, v in config.items() if k != "text_config"}}
    tc = config["text_config"]
    if tc.get("num_experts", 0):
        raise ValueError("MoE Qwen3.5 variants are not supported by this engine")
    qcfg = config.get("quantization") or config.get("quantization_config") or {}
    is_tq1 = config.get("model_type") == TQ1_MODEL_TYPE
    jang_packed = bool((jang.get("runtime") or {}).get("requires_jang_ternary_packed_expansion"))
    is_ptq1 = (qcfg.get("format") == "PTQ1_0" or str(config.get("schema", "")).startswith("prism_hadamard_qwen35_ptq1")
               or is_tq1 or jang_packed)
    if is_ptq1 and not ptq1:
        raise ValueError("this is a 1.75-bit ternary bundle (PTQ1_0 / TQ1_G128 / JANG packed) — supported as an experimental option: "
                         "enable Settings → Advanced → \"Experimental: PTQ1 models\" (or pass --ptq1 on)")

    files = sorted(glob.glob(os.path.join(path, "*.safetensors")))
    if not files:
        raise ValueError("no .safetensors files in the model folder")
    shapes = read_shapes(files)
    keys = set(shapes)

    vis = None
    if vision and config.get("vision_config") and any(k.startswith("vision_tower.") for k in keys):
        from .vision import VisionModel
        vis = VisionModel(config["vision_config"], shapes, qcfg)
    model = Model(config, qcfg, vis, shapes)
    from mlx.utils import tree_flatten
    expected_shapes = {k: v.shape for k, v in tree_flatten(model.parameters())}

    had_cfg = config.get("hadamard") or jang.get("hadamard")
    layout = jang.get("layout", {})
    shift = layout.get("language_norms") == "zero-centered-runtime-plus-one" or config.get("jang_norms_pre_shifted") is False
    shift_decided = bool(layout.get("language_norms")) or "jang_norms_pre_shifted" in config
    if is_ptq1:
        # GGUF-derived: norm weights are already in final form (means 0.8-1.9 in the real file), and
        # the Hadamard block size is listed per module in config["modules"]. A TQ1_G128 bundle is a
        # repack of a Prism MLX checkpoint: its norms keep that source's convention, so they go
        # through the same value heuristic as any unstamped Prism bundle.
        if not (is_tq1 or jang_packed):
            shift, shift_decided = False, True
        blocks = {m.get("block") for m in config.get("modules", []) if m.get("block")}
        if len(blocks) == 1:
            had_cfg = {"block_size": blocks.pop()}
        elif blocks:
            raise ValueError(f"mixed Hadamard block sizes {sorted(blocks)} are not supported")

    signs, pending, loaded = {}, {}, set()
    for f in files:
        shard = mx.load(f)
        if is_tq1 or jang_packed:
            _join_tq1(shard)
        out = {}
        for raw in list(shard):
            v = shard.pop(raw)  # drop the shard's reference so fused parts can be freed early
            k = canonical(raw)
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
            if (k.endswith(".weight") and v.ndim == 2 and v.dtype in (mx.bfloat16, mx.float32)
                    and k.startswith("language_model.") and f"{k[:-7]}.scales" not in keys
                    and not k.endswith(("in_proj_a.weight", "in_proj_b.weight"))):
                v = v.astype(mx.float16)  # an unquantized linear / embedding: keep the pipeline fp16
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
                    if len({tuple(x.shape[1:]) for x in slot}) > 1 or len({x.dtype for x in slot}) > 1:
                        raise ValueError(
                            f"{sp[0]}{fused}.{sp[2]}: the parts of one fused projection must share their bit "
                            f"width, group size and mode, but have shapes {[tuple(x.shape) for x in slot]} — "
                            f"this mixed quantization isn't supported (e.g. q_proj 4-bit with k_proj 8-bit)")
                    # materialize now: parts + result live together only for this one tensor
                    fused_arr = mx.concatenate(slot, axis=0)
                    mx.eval(fused_arr)
                    out[f"{sp[0]}{fused}.{sp[2]}"] = fused_arr
                    del pending[(sp[0], fused, sp[2])]
                continue
            out[k] = v
        for k in list(out):
            v = out[k]
            if v.dtype == mx.uint8 and v.ndim == 2 and len(expected_shapes.get(k, ())) == 4:
                from .ptq1 import pack
                out[k] = pack(v)  # row-major PTQ1 blocks -> tiled layout, once, at load
        bad = [k for k, v in out.items() if k in expected_shapes and tuple(v.shape) != tuple(expected_shapes[k])]
        if bad:
            k = bad[0]
            raise ValueError(f"unexpected tensor shape for {k}: {tuple(out[k].shape)} vs {tuple(expected_shapes[k])} "
                             f"(the parts of one fused projection — q/k/v, gate/up, qkv/z — must share "
                             f"their bit width, group size and mode)")
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

    if steer:
        from . import steering
        file, rec_alpha = steering.resolve(steer[0])
        alpha = steer[1] if steer[1] is not None else (rec_alpha if rec_alpha is not None else 1.0)
        d = steering.load_direction(file, tc["hidden_size"])
        model.set_steer(steering.projector(d, alpha))
        log(f"activation steering on: {os.path.basename(file)}, alpha {alpha:g}")

    model.ptq1 = is_ptq1
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
    qcfg = config.get("quantization") or config.get("quantization_config") or {}
    index = _json(os.path.join(path, "model.safetensors.index.json")).get("weight_map") or {}
    keys = {k for k in index if k.startswith("vision_tower.")}
    shapes = read_shapes(sorted(glob.glob(os.path.join(path, "*.safetensors"))))
    vis = VisionModel(config["vision_config"], shapes, qcfg)
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
