"""Computing a weight-space ablation direction from two sets of captured activations (mean-diff), and the
delta that removes — or partially removes, or over-removes for alpha>1 — its component from an output
projection's rows. The weight-level analog of `engine/steering.py`'s activation-level projector, built on
`model_lab.capture`/`engine.weightpatch`.

Native engine only, and only each chosen layer's MLP output projection (`mlp.down_proj`, or GGUF's
`ffn_down` when that route is added) — the best-known, most commonly targeted ablation point, and the one
every layer (attention or GatedDeltaNet) has in common without the native route's input-projection fusing
complicating anything (output projections are never fused). Extending to attention/GatedDeltaNet output
projections (`self_attn.o_proj`/`linear_attn.out_proj`) is a small, symmetric follow-up, not attempted here.

This computes directions from whatever activations are handed to it — it does not itself judge whether a
prompt pair is a good or representative probe for a given behavior; per the project's own scientific
caution (docs/model-lab/compatibility.md), a direction extracted this way is a candidate to evaluate on
held-out prompts, not a proven fix."""
import numpy as np

from . import capture


def mean_diff_direction(positive, negative):
    """positive/negative: [N, hidden] float32 arrays (e.g. several prompts' last-token activations) -> a
    unit vector pointing from `negative`'s mean toward `positive`'s."""
    d = positive.mean(axis=0) - negative.mean(axis=0)
    norm = np.linalg.norm(d)
    if norm == 0:
        raise ValueError("the two activation sets have an identical mean — no direction to extract")
    return (d / norm).astype(np.float32)


def projection_delta(base_weight, direction, alpha=1.0):
    """ΔW such that `(W + ΔW)`'s rows have `alpha` less of their component along `direction` — alpha=1.0
    fully zeroes it (the standard "ablation" operation: W' = W - alpha·d(d^T W)), alpha<1 partially
    removes it, alpha<0 pushes further along it. `base_weight`: [rows, cols] float32; `rows` must equal
    `direction`'s length — an output projection's rows are the hidden-size dimension it writes into the
    residual stream."""
    if base_weight.shape[0] != direction.shape[0]:
        raise ValueError(f"direction has {direction.shape[0]} dims, the tensor has {base_weight.shape[0]} "
                         f"output rows — direction must match the tensor's output (row) dimension")
    coeff = direction @ base_weight          # [cols]
    return (-alpha * np.outer(direction, coeff)).astype(np.float32)


def collect_last_token_residual(model, ids_list, layer):
    """ids_list: a list of already-tokenized [1, L] mx arrays, one per prompt. Returns [N, hidden] — the
    post-MLP residual (this layer's own output) at each prompt's last token."""
    rows = []
    for ids in ids_list:
        captured, _ = capture.capture_activations(model, ids, layers=[layer])
        rows.append(captured[layer]["post_mlp_residual"][0, -1])
    return np.stack(rows)


def ablation_deltas(model, safe_ids, harmful_ids, layers, alpha=1.0):
    """{tensor_name: delta} for the native route's `mlp.down_proj` at each layer in `layers`, each from its
    own `direction = unit(mean(harmful) - mean(safe))` at that layer's post_mlp_residual."""
    from engine.weightpatch import locate, read_logical
    out = {}
    for i in layers:
        safe_acts = collect_last_token_residual(model, safe_ids, i)
        harmful_acts = collect_last_token_residual(model, harmful_ids, i)
        direction = mean_diff_direction(harmful_acts, safe_acts)
        name = f"language_model.model.layers.{i}.mlp.down_proj.weight"
        base, _spec = read_logical(locate(model, "native", name))
        out[name] = projection_delta(base, direction, alpha)
    return out
