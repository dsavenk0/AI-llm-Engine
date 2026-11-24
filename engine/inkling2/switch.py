# Copyright © 2026 Apple Inc. — the parts of mlx-lm's main branch (mlx_lm/layers/moe.py, layers/common.py) that
# Inkling 2 uses and mlx-lm 0.32 does not have: experts with a per-expert nvfp4 global scale. MIT License, see NOTICE.
"""Stacked experts that may carry one nvfp4 tensor scale each (`global_scale`, shape (experts,)).

mlx-lm 0.32's loader quantizes a module by calling `to_quantized(group_size, bits, mode)` with what it reads off
the saved scales (nvfp4, mxfp8 or affine); an nvfp4 expert stack of Inkling 2 also has its per-expert scale in the
checkpoint, so the quantized module made for nvfp4 holds one — `mx.gather_qmm` picks each row's by its expert."""
import math

import mlx.core as mx
from mlx import nn


def _gather_sort(x, indices):
    *_, M = indices.shape
    indices = indices.flatten()
    order = mx.argsort(indices)
    inv_order = mx.argsort(order)
    return x.flatten(0, -3)[order // M], indices[order], inv_order


def _scatter_unsort(x, inv_order, shape=None):
    x = x[inv_order]
    if shape is not None:
        x = mx.unflatten(x, 0, shape)
    return x


def splice(embeddings, inputs, *, rows, token_id):
    """The embeddings with the rows, in order, at the token_id positions of inputs."""
    if rows.shape[0] == 0:
        return embeddings
    mask = inputs == token_id
    idx = mx.cumsum(mask.flatten()).reshape(mask.shape) - 1
    idx = mx.clip(idx, 0, rows.shape[0] - 1)
    selected = rows[idx].astype(embeddings.dtype)
    return mx.where(mask[..., None], selected, embeddings)


class QuantizedSwitchLinear(nn.Module):
    def __init__(self, input_dims, output_dims, num_experts, group_size=64, bits=4, mode="affine"):
        super().__init__()
        self.group_size, self.bits, self.mode = group_size, bits, mode
        # placeholders of the right shapes: the checkpoint replaces them
        self.weight = mx.zeros((num_experts, output_dims, input_dims * bits // 32), mx.uint32)
        if mode == "affine":
            self.scales = mx.zeros((num_experts, output_dims, input_dims // group_size), mx.float16)
            self.biases = mx.zeros((num_experts, output_dims, input_dims // group_size), mx.float16)
        else:
            self.scales = mx.zeros((num_experts, output_dims, input_dims // group_size), mx.uint8)
        if mode == "nvfp4":
            self.global_scale = mx.ones((num_experts,), mx.float32)
        self.freeze()

    @property
    def num_experts(self):
        return self.weight.shape[0]

    def __call__(self, x, indices, sorted_indices=False):
        return mx.gather_qmm(
            x, self["weight"], self["scales"], self.get("biases"), rhs_indices=indices, transpose=True,
            group_size=self.group_size, bits=self.bits, mode=self.mode, global_scale=self.get("global_scale"),
            sorted_indices=sorted_indices)


class SwitchLinear(nn.Module):
    def __init__(self, input_dims, output_dims, num_experts, bias=False):
        super().__init__()
        scale = math.sqrt(1 / input_dims)
        self.weight = mx.random.uniform(low=-scale, high=scale, shape=(num_experts, output_dims, input_dims))

    @property
    def num_experts(self):
        return self.weight.shape[0]

    def __call__(self, x, indices, sorted_indices=False):
        return mx.gather_mm(x, self["weight"].swapaxes(-1, -2), rhs_indices=indices, sorted_indices=sorted_indices)

    def to_quantized(self, group_size=64, bits=4, mode="affine", **_):
        num_experts, output_dims, input_dims = self.weight.shape
        return QuantizedSwitchLinear(input_dims, output_dims, num_experts, group_size, bits, mode)


def quantize(linear, group_size, bits, mode, global_scale=False):
    """A QuantizedSwitchLinear holding `linear`'s weights (each expert with its own nvfp4 tensor scale when
    `global_scale`) — what a converter does; loading a quantized checkpoint does not need it."""
    w = linear.weight
    q = QuantizedSwitchLinear(w.shape[2], w.shape[1], w.shape[0], group_size, bits, mode)
    if global_scale:
        if mode != "nvfp4":
            raise ValueError(f"a global scale needs nvfp4, not {mode}")
        gs = mx.max(mx.abs(w), axis=(-2, -1)).astype(mx.float32)
        parts = [mx.quantize(w[e], group_size, bits, mode=mode, global_scale=gs[e]) for e in range(w.shape[0])]
        q.weight, q.scales = mx.stack([p[0] for p in parts]), mx.stack([p[1] for p in parts])
        q.global_scale = gs
    else:
        q.weight, q.scales, *biases = mx.quantize(w, group_size, bits, mode=mode)
        if biases:
            q.biases = biases[0]
        if "global_scale" in q:
            del q["global_scale"]
    return q
