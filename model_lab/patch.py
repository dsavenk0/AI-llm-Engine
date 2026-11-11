"""Experiment-side helpers for direct, reversible weight edits: computing/saving a candidate patch,
listing what exists, and recording an apply/rollback evaluation. The actual tensor-locating and
dequantize/add/requantize mechanics live in `engine.weightpatch` — the engine's own serving path (loading
with `--weight-patch`, mirroring `--steer`) uses that module directly, so this one builds on it rather than
duplicating it, keeping exactly one implementation of "what does applying a delta to this tensor mean."

The original checkpoint file is never touched or rewritten — a patch only ever modifies the already-loaded,
in-memory model object. "Rollback" is therefore not undoing a write: it is loading the same checkpoint again
without applying the patch, which is provably the original model because nothing on disk ever changed. See
docs/model-lab/compatibility.md for exactly what is and isn't covered (MoE experts, ternary/PTQ1 tensors,
the native route's fused input projections, `mx`-mode quantization) — all enforced by `engine.weightpatch`.
"""
import json
import os
import time

import mlx.core as mx
import numpy as np

from engine.weightpatch import PatchError, apply_delta, locate, read_logical, write_logical  # noqa: F401

__all__ = ["PatchError", "apply_delta", "locate", "read_logical", "write_logical",
          "create_patch", "load_patch", "list_patches", "save_evaluation", "load_evaluation"]


def create_patch(root, model_name, route, deltas, note=""):
    """`deltas`: {tensor_name: float32 ndarray}, one or more. Writes
    `<root>/experiments/<id>/candidate.patch/{manifest.json, delta.safetensors}` — the spec's own
    artifact layout, scoped down for this increment (no lineage/conflict-checking machinery yet)."""
    if not deltas:
        raise PatchError("a patch needs at least one tensor")
    exp_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + f"-{abs(hash((model_name, tuple(deltas)))) % 10000:04d}"
    patch_dir = os.path.join(root, "experiments", exp_id, "candidate.patch")
    os.makedirs(patch_dir, exist_ok=True)
    mx.save_safetensors(os.path.join(patch_dir, "delta.safetensors"),
                        {name: mx.array(d) for name, d in deltas.items()})
    manifest = {"model": model_name, "route": route, "tensors": list(deltas),
               "shapes": {name: list(d.shape) for name, d in deltas.items()},
               "max_abs_delta": {name: float(np.max(np.abs(d))) if d.size else 0.0 for name, d in deltas.items()},
               "note": note, "created": exp_id}
    with open(os.path.join(patch_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    return exp_id, patch_dir


def load_patch(root, exp_id):
    """(manifest, {tensor_name: float32 ndarray}, patch_dir)."""
    patch_dir = os.path.join(root, "experiments", exp_id, "candidate.patch")
    with open(os.path.join(patch_dir, "manifest.json")) as f:
        manifest = json.load(f)
    loaded = mx.load(os.path.join(patch_dir, "delta.safetensors"))
    deltas = {name: np.asarray(loaded[name].astype(mx.float32)) for name in manifest["tensors"]}
    return manifest, deltas, patch_dir


def list_patches(root, model_name=None):
    exp_root = os.path.join(root, "experiments")
    out = []
    if not os.path.isdir(exp_root):
        return out
    for exp_id in sorted(os.listdir(exp_root)):
        p = os.path.join(exp_root, exp_id, "candidate.patch", "manifest.json")
        if not os.path.isfile(p):
            continue
        with open(p) as f:
            manifest = json.load(f)
        if model_name is None or manifest.get("model") == model_name:
            out.append(manifest)
    return out


def save_evaluation(root, exp_id, evaluation):
    patch_dir = os.path.join(root, "experiments", exp_id, "candidate.patch")
    with open(os.path.join(patch_dir, "evaluation.json"), "w") as f:
        json.dump(evaluation, f, indent=2)


def load_evaluation(root, exp_id):
    p = os.path.join(root, "experiments", exp_id, "candidate.patch", "evaluation.json")
    if not os.path.isfile(p):
        return None
    with open(p) as f:
        return json.load(f)
