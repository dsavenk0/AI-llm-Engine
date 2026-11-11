"""Export a patched (or unpatched) native-route model as a complete, standalone copy — the alternative
to activating a patch at load time (engine/weightpatch.py): instead of a small delta re-applied every
startup, this bakes W_base + ΔW into a full new checkpoint folder once. The source model is never
modified; the export is a new, separate folder the loader can open on its own, with no patch involved.

Confirmed by direct testing, not assumed: the live, already-fused parameter tree
(`mlx.utils.tree_flatten(model.parameters())`) is **not** directly re-loadable when a fused module is
quantized. `canonical()` renaming `model.*`/`lm_head.*` to `language_model.*` has nothing to do with it —
the real issue is that `Model.__init__` decides each module's quantization (`engine/quant.py`'s
`module_spec`) by reading `scales`/`biases` off the checkpoint's header *under the original per-part hint
name* (`self_attn.q_proj`, `mlp.gate_proj`, `linear_attn.in_proj_qkv`/`in_proj_b`, ...) — the names
`engine/loader.py`'s own `FUSE` table lists — never the fused attribute name the live module is actually
stored under. A checkpoint that only has the fused name's `scales`/`biases` makes `module_spec` report
"not quantized" (it doesn't find `mlp.gate_proj.scales`), so the freshly constructed module is plain
`FLinear` at the *wrong* (unpacked) shape, while the actual exported tensor is still packed — the reload
then fails on a shape mismatch. `gate_up_proj` exists in every layer of every model, so this is not an
edge case; it reproduces on every quantized native model.

The fix: un-fuse each quantized fused tensor back into its real per-part tensors before writing, slicing
`weight`/`scales`/`biases` along axis 0 (output rows) — exactly the inverse of `loader.py`'s own
`mx.concatenate(slot, axis=0)`, and just as exact (a pure slice, no dequantize/requantize, so no precision
loss beyond what fusing already didn't introduce). The four fused tensors and their row-split sizes, each
computed from `model.args` (the same config fields the modules themselves were built from) — not
hardcoded, so this tracks the model's own real head counts/dims:

    self_attn.qkv_proj     -> q_proj, k_proj, v_proj     split [nh*hd*2, nkv*hd, nkv*hd]
    mlp.gate_up_proj       -> gate_proj, up_proj          split [inter, inter]
    linear_attn.in_proj_qkvz -> in_proj_qkv, in_proj_z    split [conv_dim, dv*nv]
    linear_attn.in_proj_ba   -> in_proj_b, in_proj_a      split [nv, nv]   (only if this one is quantized —
                                                           it is plain `nn.Linear` on most checkpoints)

An *unquantized* fused tensor also needs splitting, for a second, independent reason: `loader.py`'s rule
for which float32/bfloat16 tensors get downcast to float16 explicitly exempts the checkpoint's own
`in_proj_a.weight`/`in_proj_b.weight` names (kept at full precision on purpose) but has no matching
exemption for the fused `in_proj_ba.weight` — so leaving that one fused, even though its *shape* alone
would reload without error, silently loses precision that was never meant to be lost. Always splitting,
regardless of quantization, sidesteps needing to track which naming exemptions exist where.

One more subtlety handled explicitly: shifted RMSNorm weights (JANG's "+1" convention) are applied once,
in memory, at load time, based on `jang_norms_pre_shifted`/a value heuristic — confirmed by direct testing
to matter: `loader.py`'s condition is `shift = ... or config.get("jang_norms_pre_shifted") is False`, so
`jang_norms_pre_shifted: false` actually means "not shifted yet, shift it now" and **caused** a double
shift the first time this was tried (every norm off by exactly +1 from being shifted twice). The exported
tensors already carry the shift, so the exported `config.json` pins `jang_norms_pre_shifted: true`
("already shifted — skip it") instead.
"""
import glob
import json
import os
import shutil

import mlx.core as mx
from mlx.utils import tree_flatten

# (fused name, [(part name, row count fn)]) — row counts are computed from `a` (model.args) at export time
_FUSE_SPLITS = {
    "self_attn.qkv_proj": [
        ("self_attn.q_proj", lambda a: a["num_attention_heads"] * a["head_dim"] * 2),
        ("self_attn.k_proj", lambda a: a["num_key_value_heads"] * a["head_dim"]),
        ("self_attn.v_proj", lambda a: a["num_key_value_heads"] * a["head_dim"]),
    ],
    "mlp.gate_up_proj": [
        ("mlp.gate_proj", lambda a: a["intermediate_size"]),
        ("mlp.up_proj", lambda a: a["intermediate_size"]),
    ],
    "linear_attn.in_proj_qkvz": [
        ("linear_attn.in_proj_qkv", lambda a: 2 * a["linear_key_head_dim"] * a["linear_num_key_heads"]
                                             + a["linear_value_head_dim"] * a["linear_num_value_heads"]),
        ("linear_attn.in_proj_z", lambda a: a["linear_value_head_dim"] * a["linear_num_value_heads"]),
    ],
    "linear_attn.in_proj_ba": [
        ("linear_attn.in_proj_b", lambda a: a["linear_num_value_heads"]),
        ("linear_attn.in_proj_a", lambda a: a["linear_num_value_heads"]),
    ],
}


def _unfuse(flat, prefix, fused_name, a):
    """Replaces `{prefix}{fused_name}.{weight,scales,biases}` in `flat` (in place) with the per-part
    tensors `loader.py`'s own FUSE table says it came from. Always splits, quantized or not — a plain
    fused tensor still needs it: `loader.py`'s float16-downcast rule explicitly exempts the *unfused*
    `in_proj_a.weight`/`in_proj_b.weight` (kept float32 on purpose) but has no such exemption for the
    fused `in_proj_ba.weight`, so leaving it fused — even though its *shape* alone would reload fine —
    silently downcasts it and was caught only by an exact-output comparison, not a shape check."""
    base = f"{prefix}{fused_name}"
    if f"{base}.weight" not in flat:
        return
    parts = _FUSE_SPLITS[fused_name]
    sizes = [size_fn(a) for _, size_fn in parts]
    w = flat.pop(f"{base}.weight")
    s = flat.pop(f"{base}.scales", None)
    b = flat.pop(f"{base}.biases", None)
    if sum(sizes) != w.shape[0]:
        raise ValueError(f"{base}: expected output rows {sum(sizes)} from config, tensor has {w.shape[0]} "
                         f"— this model's shape doesn't match what export.py assumes for this fusion")
    if s is not None and (s.shape[0] != w.shape[0] or (b is not None and b.shape[0] != w.shape[0])):
        raise ValueError(f"{base}: scales/biases don't have one row per weight row — "
                         f"affine quantization is assumed, not an mx mode, for a fused tensor")
    off = 0
    for part_name, n in zip((p for p, _ in parts), sizes):
        flat[f"{prefix}{part_name}.weight"] = w[off:off + n]
        if s is not None:
            flat[f"{prefix}{part_name}.scales"] = s[off:off + n]
        if b is not None:
            flat[f"{prefix}{part_name}.biases"] = b[off:off + n]
        off += n


def _exportable_weights(model):
    flat = dict(tree_flatten(model.parameters()))
    a = model.args
    n_layers = len(model.layers)
    for i in range(n_layers):
        prefix = f"language_model.model.layers.{i}."
        if model.layers[i].is_linear:
            _unfuse(flat, prefix, "linear_attn.in_proj_qkvz", a)
            _unfuse(flat, prefix, "linear_attn.in_proj_ba", a)
        else:
            _unfuse(flat, prefix, "self_attn.qkv_proj", a)
        _unfuse(flat, prefix, "mlp.gate_up_proj", a)
    return flat


def export_model(model, source_path, out_dir):
    """Writes a complete, standalone copy of `source_path` to `out_dir`, with `model`'s current (live,
    possibly weight-patched) parameters as the weights, fused tensors split back to the checkpoint's
    original per-part layout where that matters (see module docstring) — config/tokenizer/chat-template
    files are copied from `source_path` unchanged except for the norm-shift pin. Refuses to export into a
    non-empty folder other than `source_path` itself, so this can never overwrite the source."""
    source_path = os.path.realpath(source_path)
    out_dir = os.path.realpath(out_dir)
    if out_dir == source_path:
        raise ValueError("the export folder must not be the source model's own folder")
    if os.path.isdir(out_dir) and os.listdir(out_dir):
        raise ValueError(f"{out_dir}: not empty — refusing to export into it")
    os.makedirs(out_dir, exist_ok=True)

    for f in os.listdir(source_path):
        src = os.path.join(source_path, f)
        if os.path.isfile(src) and not f.endswith((".safetensors", ".safetensors.index.json")):
            shutil.copy2(src, os.path.join(out_dir, f))

    config_path = os.path.join(out_dir, "config.json")
    if os.path.isfile(config_path):
        with open(config_path) as f:
            config = json.load(f)
        config["jang_norms_pre_shifted"] = True
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

    weights = _exportable_weights(model)
    mx.save_safetensors(os.path.join(out_dir, "model.safetensors"), weights)
    for stale in glob.glob(os.path.join(out_dir, "model-*-of-*.safetensors")):
        os.remove(stale)      # the source may have been sharded; the export is always one file
    stale_index = os.path.join(out_dir, "model.safetensors.index.json")
    if os.path.isfile(stale_index):
        os.remove(stale_index)
    return out_dir
