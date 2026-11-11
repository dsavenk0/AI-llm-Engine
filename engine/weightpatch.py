"""Direct, reversible weight edits: W_effective = W_base + ΔW for one or more tensors, applied once at
model-load time — the same way `engine/steering.py`'s activation edit is: an optional, opt-in file next to
a base model, never a rewrite of the checkpoint itself.

How the sum is stored decides what the patch really does, so it is explicit (`storage`):

* ``"quantized"`` (default): the tensor keeps its bits and its memory. Each group keeps its own scale and
  bias — its grid — and only the codes of the weights that moved are re-rounded onto it; a group whose
  values leave its range gets a new range. A weight whose delta is zero keeps its code exactly, so a zero
  delta is the identity. What this cannot do is hold a delta smaller than the grid: at 4 bits a step is a
  third of a typical weight, and a change of a tenth of a step mostly rounds away. `apply_delta` returns
  how much of the request survived (`realized`), and that number is the thing to look at.
* ``"float16"``: the patched tensor is kept dequantized, so the sum is exact to float16. Costs the memory
  and the read bandwidth of an unquantized tensor (4x a 4-bit one), for the patched tensors only. This is
  what a low-rank edit such as an ablation needs on a quantized model.

(Before this, a quantized tensor was re-quantized from scratch with `mx.quantize`. That refits every
group's range, and the refit of an already quantized tensor is not the identity: measured on
Qwen3.5-0.8B-4bit, writing a tensor back with a zero delta changed 15 % of its packed words, by up to three
quantization steps, and an ablation delta came out with more unrequested change than requested.)

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

    def __init__(self, obj, weight_attr, parent=None, key=None, mirrors=(), owner=None):
        self.obj, self.weight_attr = obj, weight_attr
        self.parent, self.key = parent, key      # where `obj` hangs, for replacing it by a float module
        self.mirrors = list(mirrors)             # objects that share this tensor (a tied output head)
        self.owner = owner                       # the GGUF model, whose byte count follows the storage

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
    obj, parent, key = model, None, None
    try:
        for p in parts[:-1]:
            parent, key = obj, (int(p) if p.lstrip("-").isdigit() else p)
            obj = obj[key] if isinstance(key, int) else getattr(obj, p)
        if not hasattr(obj, "weight"):
            raise AttributeError(parts[-1])
    except (AttributeError, IndexError, TypeError) as e:
        raise PatchError(f"no such tensor on this model: {dotted_name!r} ({e}) — note that the native "
                         f"engine fuses q/k/v, gate/up and GatedDeltaNet's qkv/z and b/a projections into "
                         f"one module at load time, so only the fused tensor's name resolves here") from None
    if isinstance(obj, (PTQ1Linear, PTQ1Embedding)):
        raise PatchError(f"{dotted_name}: ternary-packed (PTQ1_0/TQ1_G128) tensors are not supported in "
                         f"this increment — see docs/model-lab/compatibility.md")
    return _Leaf(obj, "weight", parent, key)


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
        # a tied output head is a second object holding the same arrays (`embed.as_linear()`): it has to
        # follow, or the patch edits what the model reads tokens from and not what it predicts them with
        tied = model.head.w is model.embed.w
        return _Leaf(model.embed, "w", mirrors=[model.head] if tied else (), owner=model)
    if raw_name == "output.weight":
        if model.head.w is model.embed.w:
            raise PatchError("output.weight is tied to token_embd.weight on this model (no separate "
                             "tensor exists) — patch token_embd.weight instead")
        return _Leaf(model.head, "w", owner=model)
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
    return _Leaf(held, "w", owner=model)


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


STORAGES = ("quantized", "float16")


def _pack(q, bits):
    """Codes [rows, cols] -> MLX's packed uint32 words: each row is one little-endian bit stream, `bits`
    per code (checked against mx.quantize for every width it supports, 2..8)."""
    rows, cols = q.shape
    b = ((q[..., None] >> np.arange(bits, dtype=np.uint32)) & 1).astype(np.uint8).reshape(rows, cols * bits)
    return np.ascontiguousarray(np.packbits(b, axis=1, bitorder="little")).view("<u4")


def _on_grid(leaf, base, new_float, spec):
    """(packed codes, scales, biases, groups refit) for `new_float`, keeping each group's scale and bias
    wherever the new values still fit between its end points. Float32 arithmetic throughout."""
    bits, gs = spec["bits"], spec["group_size"]
    top = float(2 ** bits - 1)
    shape = new_float.shape
    w = new_float.reshape(-1, shape[-1] // gs, gs).astype(np.float32)
    old = base.reshape(w.shape)
    s = np.asarray(leaf.scales.astype(mx.float32)).reshape(-1, w.shape[1], 1).copy()
    b = np.asarray(leaf.biases.astype(mx.float32)).reshape(-1, w.shape[1], 1).copy()
    safe = np.where(s == 0, 1.0, s)
    q = np.rint((w - b) / safe)
    out = ((q < 0) | (q > top)).any(axis=-1, keepdims=True)        # the group's range no longer holds it
    if out.any():
        lo, hi = w.min(axis=-1, keepdims=True), w.max(axis=-1, keepdims=True)
        ns = np.maximum((hi - lo) / top, 1e-7)
        s, b = np.where(out, ns, s), np.where(out, lo, b)
        safe = np.where(s == 0, 1.0, s)
        q = np.rint((w - b) / safe)
    # a weight that was not asked to move keeps the code it had (in a group that kept its grid)
    q_old = np.rint((old - b) / safe)
    q = np.where((w == old) & ~out, q_old, q)
    q = np.clip(q, 0, top).astype(np.uint32)
    packed = _pack(q.reshape(-1, shape[-1]), bits).reshape(*shape[:-1], -1)
    return packed, s.reshape(*shape[:-1], -1), b.reshape(*shape[:-1], -1), int(out.sum())


def _float_module(leaf, arr):
    """Replace a quantized native module by its float equivalent (same call, plain matmul / lookup)."""
    from .model import FEmbedding, FLinear, QEmbedding
    old = leaf.obj
    new = FEmbedding(*arr.shape) if isinstance(old, QEmbedding) else FLinear(arr.shape[1], arr.shape[0])
    new.weight = arr
    if "bias" in old:
        new.bias = old.bias
    if leaf.parent is None:
        raise PatchError("this tensor cannot be kept in float16 (no parent module to hold it)")
    if isinstance(leaf.key, int):
        leaf.parent[leaf.key] = new
    else:
        setattr(leaf.parent, leaf.key, new)
    leaf.obj = new


def write_logical(leaf, new_float, spec, storage="quantized", base=None):
    """Store `new_float` as the tensor. Returns {"storage", "groups_refit"}."""
    if storage not in STORAGES:
        raise PatchError(f"unknown storage {storage!r} (one of: {', '.join(STORAGES)})")
    if spec["kind"] == "plain":
        leaf.weight = mx.array(new_float).astype(spec["dtype"])
        _sync(leaf)
        return {"storage": "plain", "groups_refit": 0}
    if storage == "float16":
        arr = mx.array(new_float).astype(mx.float16)
        before = leaf.weight.nbytes + leaf.scales.nbytes + (leaf.biases.nbytes if leaf.biases is not None else 0)
        if leaf.weight_attr == "w":                      # GGUF: the same object, switched to dense
            leaf.obj.w, leaf.obj.scales, leaf.obj.biases, leaf.obj.bits = arr, None, None, 0
            if leaf.owner is not None and hasattr(leaf.owner, "nbytes"):
                leaf.owner.nbytes += arr.nbytes - before
        else:
            _float_module(leaf, arr)
        _sync(leaf)
        return {"storage": "float16", "groups_refit": 0}
    if base is None:
        base, _ = read_logical(leaf)
    packed, ns, nb, refit = _on_grid(leaf, base, new_float, spec)
    leaf.weight = mx.array(packed).astype(spec["wdtype"])
    leaf.obj.scales = mx.array(ns).astype(spec["sdtype"])
    if spec["bdtype"] is not None:
        leaf.obj.biases = mx.array(nb).astype(spec["bdtype"])
    _sync(leaf)
    return {"storage": "quantized", "groups_refit": refit}


def _sync(leaf):
    for m in leaf.mirrors:
        m.w, m.scales, m.biases, m.bits = leaf.obj.w, leaf.obj.scales, leaf.obj.biases, leaf.obj.bits


def fidelity(base, requested, realized_tensor):
    """How much of a requested change is really in the tensor now. `realized` is the share of the request
    that arrived (its projection onto the request: 1.0 = all of it, 0 = none); `unrequested` is the size of
    whatever else changed, relative to the request (0 = nothing else moved)."""
    real = realized_tensor.astype(np.float32) - base
    want = requested.astype(np.float32)
    n = float(np.linalg.norm(want))
    if n == 0:
        return {"realized": 1.0, "unrequested": float(np.linalg.norm(real)), "requested_norm": 0.0}
    along = float((real * want).sum()) / (n * n)
    off = float(np.linalg.norm(real - along * want)) / n
    return {"realized": round(along, 4), "unrequested": round(off, 4), "requested_norm": n}


def apply_delta(model, route, tensor_name, delta, alpha=1.0, storage="quantized"):
    """In-memory only. `delta`: float32 ndarray, the tensor's logical (dequantized) shape. Mutates `model`
    in place and returns a report: the tensor's shape, how it is stored now, and how much of the request
    survived that storage (see `fidelity`) — check `realized` before believing a patch did what it says."""
    leaf = locate(model, route, tensor_name)
    base, spec = read_logical(leaf)
    if tuple(delta.shape) != base.shape:
        raise PatchError(f"delta shape {tuple(delta.shape)} does not match {tensor_name}'s logical shape "
                         f"{base.shape}")
    want = alpha * delta.astype(np.float32)
    report = write_logical(leaf, base + want, spec, storage, base)
    after, _ = read_logical(leaf)
    report.update(fidelity(base, want, after), shape=base.shape)
    return report


def apply_patch_dir(model, route, patch_dir, alpha=1.0):
    """Reads `<patch_dir>/{manifest.json, delta.safetensors}` (the format `model_lab.patch.create_patch`
    writes: one or more tensors, each a key in delta.safetensors matching its name in manifest["tensors"])
    and applies every tensor's delta, stored the way the manifest asks (`"storage"`, default quantized).
    `alpha` scales every delta uniformly, the same role `--steer-alpha` plays for activation steering —
    1.0 is what the patch was created at, 0 turns it off without needing a different checkpoint or
    process. The returned manifest carries `"applied"`: one fidelity report per tensor."""
    with open(os.path.join(patch_dir, "manifest.json")) as f:
        manifest = json.load(f)
    if manifest.get("route") != route:
        raise PatchError(f"this patch was created for the {manifest.get('route')!r} route, not {route!r}")
    manifest["applied"] = {}
    if alpha == 0:
        return manifest
    deltas = mx.load(os.path.join(patch_dir, "delta.safetensors"))
    storage = manifest.get("storage", "quantized")
    for name in manifest["tensors"]:
        if name not in deltas:
            raise PatchError(f"{patch_dir}: delta.safetensors has no tensor {name!r} that the manifest lists")
        manifest["applied"][name] = apply_delta(model, route, name, np.asarray(deltas[name].astype(mx.float32)),
                                                alpha, storage)
    return manifest


def describe(applied):
    """One line for a log: what share of the requested change the weights now hold, worst tensor first."""
    if not applied:
        return "nothing applied"
    worst = min(applied.values(), key=lambda r: r["realized"])
    kinds = sorted({r["storage"] for r in applied.values()})
    text = f"{len(applied)} tensor(s), stored {'/'.join(kinds)}, {worst['realized']:.0%} of the requested change realized"
    if len(applied) > 1:
        text += " in the worst tensor"
    if worst["realized"] < 0.9 and "quantized" in kinds:
        text += (" — the delta is finer than the quantization grid; recreate the patch with float16 storage "
                 "if it has to be exact")
    return text
