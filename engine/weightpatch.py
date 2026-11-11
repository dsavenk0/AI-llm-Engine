"""Direct, reversible weight edits: W_effective = W_base + ΔW for one or more tensors, applied once at
model-load time — the same way `engine/steering.py`'s activation edit is: an optional, opt-in file next to
a base model, never a rewrite of the checkpoint itself. A patch's steady-state memory cost is zero (the
patched tensor is dequantized, edited and requantized back to the exact same bits/group_size/shape it had);
the only extra cost is a brief, one-tensor-at-a-time transient float32 copy while that happens.

Native (`engine.model.Model`) and GGUF (`engine.dense.DenseModel`) routes only — mlx-lm-backed models use
a forward path with no equivalent tensor structure to patch. See docs/model-lab/compatibility.md for what
is and isn't covered: MoE experts and ternary-packed (PTQ1_0/TQ1_G128) tensors are refused, never silently
misapplied; the native route's fused input projections (qkv_proj, gate_up_proj, in_proj_qkvz/ba) are
patchable only by their fused name, not by a per-head/part name a raw checkpoint listing would show; only
`affine`-mode MLX quantization round-trips (the `mx` modes — mxfp4/mxfp8/nvfp4 — are refused).

`model_lab/patch.py` (the CLI/research side: computing a delta, saving an experiment artifact, comparing
before/after) builds on this module rather than duplicating it.
"""
import json
import os
import re

import mlx.core as mx
import numpy as np


class PatchError(ValueError):
    pass


class _Leaf:
    """A uniform read/write view over one patchable tensor, whichever route it came from. Native modules
    (`QLinear`/`FLinear`/`QEmbedding`/`FEmbedding`) and GGUF's plain `dense.QLinear`/`dense.Embedding`
    objects use the same attribute names for everything except the raw weight itself (`.weight` vs `.w`),
    and GGUF's decoder always uses group size 32 and affine mode without storing either as an attribute —
    both are exactly what the `getattr(..., default)` calls below fall back to."""

    def __init__(self, obj, weight_attr):
        self.obj, self.weight_attr = obj, weight_attr

    @property
    def weight(self):
        return getattr(self.obj, self.weight_attr)

    @weight.setter
    def weight(self, v):
        setattr(self.obj, self.weight_attr, v)

    @property
    def scales(self):
        return getattr(self.obj, "scales", None)

    @property
    def biases(self):
        return getattr(self.obj, "biases", None)

    @property
    def bits(self):
        return getattr(self.obj, "bits", 0)

    @property
    def group_size(self):
        return getattr(self.obj, "group_size", 32)

    @property
    def mode(self):
        return getattr(self.obj, "mode", "affine")


def _native_leaf(model, dotted_name):
    from .model import PTQ1Embedding, PTQ1Linear
    parts = dotted_name.split(".")
    if len(parts) < 2 or parts[-1] != "weight":
        raise PatchError(f"not a patchable tensor: {dotted_name!r} (name a *.weight leaf, e.g. "
                         f"'language_model.model.layers.0.mlp.down_proj.weight')")
    obj = model
    try:
        for p in parts[:-1]:
            obj = obj[int(p)] if p.lstrip("-").isdigit() else getattr(obj, p)
        if not hasattr(obj, "weight"):
            raise AttributeError(parts[-1])
    except (AttributeError, IndexError, TypeError) as e:
        raise PatchError(f"no such tensor on this model: {dotted_name!r} ({e}) — note that the native "
                         f"engine fuses q/k/v, gate/up and GatedDeltaNet's qkv/z and b/a projections into "
                         f"one module at load time, so only the fused tensor's name resolves here") from None
    if isinstance(obj, (PTQ1Linear, PTQ1Embedding)):
        raise PatchError(f"{dotted_name}: ternary-packed (PTQ1_0/TQ1_G128) tensors are not supported in "
                         f"this increment — see docs/model-lab/compatibility.md")
    return _Leaf(obj, "weight")


_GGUF_LAYER_ATTR = {"attn_q": "q", "attn_k": "k", "attn_v": "v", "attn_output": "o",
                    "ffn_gate": "gate", "ffn_up": "up", "ffn_down": "down"}


def _gguf_leaf(model, raw_name):
    """`token_embd.weight`/`output.weight`/`blk.N.attn_*|ffn_*.weight` resolve to GGUF's own plain
    `QLinear`/`Embedding` objects (`.w`/`.scales`/`.biases`/`.bits`, group size always 32) — one level
    below the attribute that names them (`model.embed`, `model.layers[i].down`, ...), since that attribute
    holds the wrapper object itself, not an array. `output_norm.weight` is the one exception: `model.norm`
    is a plain array directly, with no wrapper."""
    if raw_name == "output_norm.weight":
        return _Leaf(model, "norm")
    if raw_name == "token_embd.weight":
        return _Leaf(model.embed, "w")
    if raw_name == "output.weight":
        if model.head.w is model.embed.w:
            raise PatchError("output.weight is tied to token_embd.weight on this model (no separate "
                             "tensor exists) — patch token_embd.weight instead")
        return _Leaf(model.head, "w")
    m = re.match(r"^blk\.(\d+)\.([a-z_]+)\.weight$", raw_name)
    if not m or m.group(2) not in _GGUF_LAYER_ATTR:
        raise PatchError(f"not a patchable tensor: {raw_name!r} (MoE experts/routers and norms other than "
                         f"output_norm.weight are not supported in this increment)")
    i, attr = int(m.group(1)), _GGUF_LAYER_ATTR[m.group(2)]
    if i >= len(model.layers):
        raise PatchError(f"no such layer: {i} (model has {len(model.layers)})")
    held = getattr(model.layers[i], attr)
    if held is None:
        raise PatchError(f"{raw_name}: this layer has no such tensor (likely a MoE layer)")
    return _Leaf(held, "w")


def locate(model, route, tensor_name):
    return (_gguf_leaf if route == "gguf" else _native_leaf)(model, tensor_name)


# ───────────────────────── dequantize → add → requantize ─────────────────────────

def read_logical(leaf):
    """(float32 ndarray at the tensor's logical shape, opaque spec to pass back to write_logical)."""
    if not leaf.bits:
        return np.asarray(leaf.weight.astype(mx.float32)), {"kind": "plain", "dtype": leaf.weight.dtype}
    if leaf.mode != "affine":
        raise PatchError(f"{leaf.mode!r}-mode quantization is not supported in this increment (affine only)")
    dq = mx.dequantize(leaf.weight, leaf.scales, leaf.biases, group_size=leaf.group_size, bits=leaf.bits,
                       mode=leaf.mode)
    return np.asarray(dq.astype(mx.float32)), {
        "kind": "affine", "bits": leaf.bits, "group_size": leaf.group_size,
        "wdtype": leaf.weight.dtype, "sdtype": leaf.scales.dtype,
        "bdtype": leaf.biases.dtype if leaf.biases is not None else None}


def write_logical(leaf, new_float, spec):
    if spec["kind"] == "plain":
        leaf.weight = mx.array(new_float).astype(spec["dtype"])
        return
    nw, ns, nb = mx.quantize(mx.array(new_float), group_size=spec["group_size"], bits=spec["bits"])
    leaf.weight = nw.astype(spec["wdtype"])
    leaf.obj.scales = ns.astype(spec["sdtype"])
    if spec["bdtype"] is not None:
        leaf.obj.biases = nb.astype(spec["bdtype"])


def apply_delta(model, route, tensor_name, delta, alpha=1.0):
    """In-memory only. `delta`: float32 ndarray, the tensor's logical (dequantized) shape. Mutates `model`
    in place and returns the tensor's shape (for a caller to sanity-check against the manifest)."""
    leaf = locate(model, route, tensor_name)
    base, spec = read_logical(leaf)
    if tuple(delta.shape) != base.shape:
        raise PatchError(f"delta shape {tuple(delta.shape)} does not match {tensor_name}'s logical shape "
                         f"{base.shape}")
    write_logical(leaf, base + alpha * delta.astype(np.float32), spec)
    return base.shape


def apply_patch_dir(model, route, patch_dir, alpha=1.0):
    """Reads `<patch_dir>/{manifest.json, delta.safetensors}` (the format `model_lab.patch.create_patch`
    writes: one or more tensors, each a key in delta.safetensors matching its name in manifest["tensors"])
    and applies every tensor's delta. `alpha` scales every delta uniformly, the same role `--steer-alpha`
    plays for activation steering — 1.0 is what the patch was created at, 0 turns it off without needing a
    different checkpoint or process."""
    with open(os.path.join(patch_dir, "manifest.json")) as f:
        manifest = json.load(f)
    if manifest.get("route") != route:
        raise PatchError(f"this patch was created for the {manifest.get('route')!r} route, not {route!r}")
    deltas = mx.load(os.path.join(patch_dir, "delta.safetensors"))
    if alpha == 0:
        return manifest
    for name in manifest["tensors"]:
        apply_delta(model, route, name, np.asarray(deltas[name].astype(mx.float32)), alpha)
    return manifest
