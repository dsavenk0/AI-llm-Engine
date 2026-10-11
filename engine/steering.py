"""Activation steering — one direction removed from what every layer writes into the residual stream.

The "edits" that ship next to a base model instead of new weights (a refusal direction, a style
direction…) are a single unit vector `d` in the model's hidden space and a strength `alpha`.
At inference, the output `y` of every module that writes the residual (the attention output
projections, the MLP down projection, the token embedding) becomes

    y  -=  alpha * dot(y, d) * d          (float32 accumulation)

The quantized weights stay exactly as they are, so nothing is requantized and nothing is added to
the quantization error; alpha 0 is the unmodified model.

An optional module: nothing here is imported unless the engine is started with --steer.
"""
import json
import os

import numpy as np

import mlx.core as mx


class SteerError(ValueError):
    pass


def _read_vector(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".safetensors":
        t = mx.load(path)
        if "direction" in t:
            v = t["direction"]
        elif len(t) == 1:
            v = next(iter(t.values()))
        else:
            raise SteerError(f"{path}: expected one tensor named 'direction', found {sorted(t)}")
        return np.asarray(v.astype(mx.float32))
    if ext == ".npy":
        return np.load(path).astype(np.float32)
    if ext == ".bin":
        return np.fromfile(path, dtype="<f4")
    raise SteerError(f"{path}: use .safetensors, .npy or .bin (float32)")


def resolve(path):
    """(direction file, alpha or None) for a file, or for a folder holding an edit: the folder's own
    refusal_dir.safetensors / direction file, and the strength the author recommends (adapter.json)."""
    if os.path.isfile(path):
        return path, None
    if not os.path.isdir(path):
        raise SteerError(f"{path}: no such file or folder")
    alpha = None
    for meta in ("runtime-adapter/adapter.json", "adapter.json"):
        try:
            with open(os.path.join(path, meta), encoding="utf-8") as f:
                alpha = float(json.load(f)["alpha"])
            break
        except (OSError, ValueError, KeyError, TypeError):
            continue
    for name in ("refusal_dir.safetensors", "direction.safetensors", "runtime-adapter/refusal_dir.npy",
                 "refusal_dir.npy", "refusal_dir_fp32.bin"):
        p = os.path.join(path, name)
        if os.path.isfile(p):
            return p, alpha
    raise SteerError(f"{path}: no direction file (refusal_dir.safetensors / .npy / .bin) in it")


def load_direction(path, hidden_size):
    """The unit-length float32 direction as an mx.array of `hidden_size`. Raises SteerError on a mismatch."""
    v = _read_vector(path).reshape(-1)
    if v.shape[0] != hidden_size:
        raise SteerError(f"the direction has {v.shape[0]} values but this model's hidden size is {hidden_size} "
                         f"— it belongs to a different model")
    if not np.all(np.isfinite(v)):
        raise SteerError("the direction contains NaN or infinity")
    n = float(np.linalg.norm(v))
    if n < 1e-8:
        raise SteerError("the direction is a zero vector")
    return mx.array(v / n)


def projector(d, alpha):
    """y → y - alpha * (y·d) d, for activations of any leading shape (…, hidden). Keeps y's dtype."""
    def steer(y):
        yf = y.astype(mx.float32)
        c = (yf * d).sum(axis=-1, keepdims=True)
        return (yf - alpha * c * d).astype(y.dtype)
    return steer
