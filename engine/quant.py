# What a module's quantization is, read off its tensors instead of one global number.
#
# An MLX checkpoint says `quantization: {bits: 4, group_size: 64}` once, but real ones
# aren't uniform: mlx-community mixes widths (embeddings / lm_head / some layers at 6-8 bit,
# the rest at 4), some leave modules in float, and the mx* modes (mxfp4 / mxfp8 / nvfp4) have
# no biases at all. The tensors themselves are unambiguous, so that's what we read:
#
#   weight  (out, in * bits / 32)  uint32      -> bits
#   scales  (out, in / group_size)             -> group_size
#   biases  present -> "affine"; absent (scales are uint8) -> mxfp4 / mxfp8 / nvfp4,
#           told apart by the (bits, group_size) pair they use.
MX_MODES = {(4, 32): "mxfp4", (8, 32): "mxfp8", (4, 16): "nvfp4"}
BITS = (2, 3, 4, 5, 6, 8)


def spec_from_shapes(in_dims, weight_cols, scales_cols, has_biases, mode=None):
    """(bits, group_size, mode) for one quantized module; raises ValueError when the shapes
    don't describe any MLX quantization of an `in_dims`-wide input."""
    if (weight_cols * 32) % in_dims or (weight_cols * 32) // in_dims not in BITS:
        raise ValueError(f"weight of {weight_cols} words doesn't pack a {in_dims}-wide input at any "
                         f"supported bit width {BITS}")
    bits = weight_cols * 32 // in_dims
    if scales_cols <= 0 or in_dims % scales_cols:
        raise ValueError(f"{scales_cols} scale groups don't divide a {in_dims}-wide input")
    gs = in_dims // scales_cols
    if mode is None or mode == "affine" and not has_biases:
        mode = "affine" if has_biases else MX_MODES.get((bits, gs))
    if mode is None:
        raise ValueError(f"no biases, so not affine, but {bits}-bit/group {gs} matches no mx mode")
    if mode == "affine" and not has_biases:
        raise ValueError("affine quantization without biases")
    return bits, gs, mode


def module_spec(shapes, name, in_dims, mode_hint=None):
    """Quantization of module `name` from {key: (shape, dtype)}; None for a float module,
    or when the module isn't in the checkpoint at all."""
    w = shapes.get(f"{name}.weight")
    if w is None or f"{name}.scales" not in shapes:
        return None
    return spec_from_shapes(in_dims, w[0][-1], shapes[f"{name}.scales"][0][-1],
                            f"{name}.biases" in shapes, mode_hint)
