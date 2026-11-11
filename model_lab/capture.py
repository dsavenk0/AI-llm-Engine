"""Activation capture for the native engine's decoder stack.

`engine/model.py`'s `set_steer` hook cannot be reused for this: `steer` is called *inside* the
`mx.compile`'d `Ops.mlp`/`gdn_post`/`attn_post` closures (model.py `Ops.__init__`, the
`for name, fn in locals().items(): ... mx.compile(fn)` loop at the end of that class). That is correct for
steering, where `steer(y)` is a pure array transform folded into the traced dataflow and re-evaluated on
every call — but a capture hook needs a Python-level *side effect* (stash a copy of the activation), and
`mx.compile` only traces Python code once per input shape, caching the result; a side effect placed inside
that trace would fire once, at trace time, not on every real call with a new prompt. Using `steer`-style
injection for capture would silently record only the first call's activations.

Instead, this module mirrors `DecoderLayer.__call__`'s own orchestration one level down (model.py ~460-470):
that function itself is plain, uncompiled Python, calling the *same* real sub-modules
(`layer.self_attn`/`layer.linear_attn`, then `ops.mlp`) in the same order — nothing here reimplements any
of the actual math, it only inserts capture points between two calls `DecoderLayer.__call__` already makes
separately. Every array captured is `mx.eval`'d before being handed back, so it reflects the real forward
pass, not a lazily-deferred graph that might still change.
"""
import json
import os
import time

import mlx.core as mx
import numpy as np

from engine.model import Positions, _w


def _layer_forward(layer, x, ops, cache, pos):
    """The exact two sub-calls `DecoderLayer.__call__` makes, split so the value between them (the
    attention/GatedDeltaNet block's output, pre-MLP) is observable. Returns (after_attn, after_mlp)."""
    n, nr = ops.norm_had(x, layer.input_layernorm.weight)
    if layer.is_linear:
        after_attn = layer.linear_attn(x, n, nr, ops, cache)
    else:
        after_attn = layer.self_attn(x, nr, ops, cache, pos)
    after_mlp = ops.mlp(after_attn, layer.post_attention_layernorm.weight,
                        *_w(layer.mlp.gate_up_proj), *_w(layer.mlp.down_proj))
    return after_attn, after_mlp


def capture_activations(model, ids, layers=None):
    """ids: an int array, shape [1, L] (one short prompt, prefill only — this is a bounded diagnostic
    call, not a decode-loop primitive). `layers`: iterable of layer indices, default all.

    Returns ({layer_idx: {"pre_residual": np.ndarray, "post_attn_residual": np.ndarray,
    "post_mlp_residual": np.ndarray}}, final_hidden: np.ndarray) — `final_hidden` is the model's own
    normed last-token state (`model(ids, cache, hidden=True)`'s definition), included so a caller can
    verify this capture pass produced the same output as an ordinary forward call (see
    tests/test_model_lab.py's no-observer-effect check)."""
    cache = model.make_cache()
    ops = model.ops
    pos = Positions()
    h = model.embed(ids)
    n_layers = len(model.layers)
    wanted = set(range(n_layers)) if layers is None else set(layers)
    bad = wanted - set(range(n_layers))
    if bad:
        raise ValueError(f"no such layer index: {sorted(bad)} (model has {n_layers} layers)")
    out = {}
    for idx, (layer, c) in enumerate(zip(model.layers, cache)):
        x = h
        after_attn, h = _layer_forward(layer, x, ops, c, pos)
        if idx in wanted:
            mx.eval(x, after_attn, h)
            out[idx] = {
                "pre_residual": np.asarray(x.astype(mx.float32)),
                "post_attn_residual": np.asarray(after_attn.astype(mx.float32)),
                "post_mlp_residual": np.asarray(h.astype(mx.float32)),
            }
    final_hidden = ops.rms(h, model.language_model.model.norm.weight)
    mx.eval(final_hidden)
    return out, np.asarray(final_hidden.astype(mx.float32))


def save_capture(root, model_name, prompt, captured, final_hidden):
    """Writes one capture to `<root>/experiments/<id>/captures/` — the spec's own artifact layout, scoped
    down for this increment: just the arrays and a small meta.json, no manifest/lineage machinery yet
    (that's part of the weight-patch increment, queued separately)."""
    exp_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + f"-{abs(hash((model_name, prompt))) % 10000:04d}"
    captures_dir = os.path.join(root, "experiments", exp_id, "captures")
    os.makedirs(captures_dir, exist_ok=True)
    for idx, tensors in captured.items():
        np.savez(os.path.join(captures_dir, f"layer_{idx:03d}.npz"), **tensors)
    meta = {"model": model_name, "prompt": prompt, "layers": sorted(captured),
            "final_hidden_shape": list(final_hidden.shape), "created": exp_id}
    with open(os.path.join(captures_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return exp_id, captures_dir
