"""Tensor inspection without dequantizing: name, shape, dtype/quant type, byte size.

Reuses the two readers the engine already has for this — `engine.loader.read_shapes` (safetensors header
only) and `engine.gguf_tensors.GGUFFile` (GGUF header + memory map) — rather than loading a model to look
at its tensors.
"""
import glob
import os

_DTYPE_BYTES = {"float32": 4, "float16": 2, "bfloat16": 2, "uint8": 1, "uint32": 4, "int32": 4,
                "int8": 1, "bool": 1, "uint16": 2}


def _dtype_name(dtype):
    if dtype is None:
        return None
    return str(dtype).rsplit(".", 1)[-1]


def inspect_tensors(path, route):
    """{tensor name: {"shape": [...], "dtype": str, "nbytes": int}}, sorted by name.

    `route`: "gguf" for a single .gguf file path, anything else for a safetensors model directory
    (matches the `route` field `scan_models()`/`resolve_model()` already put on a model entry).
    """
    if route == "gguf":
        from engine.gguf_tensors import GGUFFile
        gf = GGUFFile(path)
        out = {name: {"shape": list(info.shape), "dtype": info.type_name, "nbytes": info.nbytes}
               for name, info in gf.tensors.items()}
    else:
        from engine.loader import read_shapes
        files = sorted(glob.glob(os.path.join(path, "*.safetensors")))
        if not files:
            raise ValueError(f"no .safetensors files in {path!r}")
        shapes = read_shapes(files)
        out = {}
        for name, (shape, dtype) in shapes.items():
            dn = _dtype_name(dtype)
            n = 1
            for d in shape:
                n *= d
            nbytes = n * _DTYPE_BYTES[dn] if dn in _DTYPE_BYTES else None
            out[name] = {"shape": list(shape), "dtype": dn, "nbytes": nbytes}
    return dict(sorted(out.items()))


def summarize(tensors):
    """(tensor_count, total_bytes_or_None) — None when any tensor's size couldn't be determined."""
    total = 0
    for info in tensors.values():
        if info["nbytes"] is None:
            return len(tensors), None
        total += info["nbytes"]
    return len(tensors), total
