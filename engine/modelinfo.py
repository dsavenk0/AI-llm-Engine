"""What is inside a model, without loading it: `llm inspect <model>`.

Reads only headers — the safetensors JSON headers of an MLX folder, or a GGUF's metadata and tensor directory —
so a 27B model is described in a blink and with no MLX or numpy (stdlib only, like the manager).

  inspect(path) → {kind, name, architecture, params, size_bytes, tensors, dtypes, quantization, config,
                   groups: [{pattern, count, dtype, shape, bytes}], notes}

Tensors are grouped by name with layer numbers replaced by N, so "model.layers.0.self_attn.q_proj.weight" …
"model.layers.63.self_attn.q_proj.weight" are one row: how many, which storage type, how many bytes.
"""
import json
import math
import os
import re
import struct

from . import gguf

GGML_NAMES = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 9: "Q8_1", 10: "Q2_K",
              11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS",
              19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS", 24: "I8", 25: "I16", 26: "I32",
              27: "I64", 28: "F64", 29: "IQ1_M", 30: "BF16"}
ST_BYTES = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2, "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "U16": 2,
            "U32": 4, "U64": 8, "BOOL": 1, "F8_E4M3": 1, "F8_E5M2": 1}


class InspectError(ValueError):
    pass


def group_name(name):
    """Layer / expert numbers replaced by N: the same weight of every layer is one group."""
    return re.sub(r"(?<=[._])\d+(?=[._]|$)", "N", name)


def _groups(rows):
    """rows: [(name, dtype, shape, nbytes)] → groups sorted by size, largest first."""
    out = {}
    for name, dtype, shape, nbytes in rows:
        g = out.setdefault((group_name(name), dtype), {"pattern": group_name(name), "count": 0, "dtype": dtype,
                                                      "shape": list(shape), "bytes": 0})
        g["count"] += 1
        g["bytes"] += nbytes
    return sorted(out.values(), key=lambda g: (-g["bytes"], g["pattern"]))


def _dtypes(rows):
    out = {}
    for _, dtype, _, nbytes in rows:
        d = out.setdefault(dtype, {"tensors": 0, "bytes": 0})
        d["tensors"] += 1
        d["bytes"] += nbytes
    return out


def _read_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            v = json.load(f)
        return v if isinstance(v, dict) else {}
    except (OSError, ValueError):
        return {}


# ───────────────────────────── MLX / safetensors ─────────────────────────────
def safetensors_header(path):
    with open(path, "rb") as f:
        raw = f.read(8)
        if len(raw) != 8:
            raise InspectError(f"{os.path.basename(path)}: not a safetensors file")
        (n,) = struct.unpack("<Q", raw)
        if n > 100 << 20:
            raise InspectError(f"{os.path.basename(path)}: header of {n} bytes is not plausible")
        try:
            return json.loads(f.read(n))
        except ValueError:
            raise InspectError(f"{os.path.basename(path)}: damaged safetensors header")


def _mlx(path):
    files = sorted(f for f in os.listdir(path) if f.endswith(".safetensors"))
    if not files:
        raise InspectError("no .safetensors files in this folder")
    rows, shapes = [], {}
    for fn in files:
        for name, t in safetensors_header(os.path.join(path, fn)).items():
            if name == "__metadata__" or not isinstance(t, dict):
                continue
            a, b = t.get("data_offsets", [0, 0])
            rows.append((name, t.get("dtype", "?"), t.get("shape", []), b - a))
            shapes[name] = t.get("shape", [])
    cfg = _read_json(os.path.join(path, "config.json"))
    text = cfg.get("text_config") if isinstance(cfg.get("text_config"), dict) else cfg
    q = cfg.get("quantization") or cfg.get("quantization_config") or {}
    group = int(q.get("group_size", 64)) if isinstance(q, dict) else 64
    ptq1 = (q.get("format") == "PTQ1_0" if isinstance(q, dict) else False) or \
        str(cfg.get("schema", "")).startswith("prism_hadamard_qwen35_ptq1")
    params, bits_used = 0, {}
    for name, dtype, shape, _ in rows:
        scales = shapes.get(name[:-len(".weight")] + ".scales") if name.endswith(".weight") else None
        if ptq1 and dtype == "U8" and not scales and name.endswith(".weight") and shape and shape[-1] % 28 == 0:
            n = math.prod(shape[:-1]) * (shape[-1] // 28) * 128          # 28 bytes carry 128 ternary weights
            bits_used["1.75"] = bits_used.get("1.75", 0) + n
            params += n
        elif scales and shape and dtype in ("U32", "U8", "I32"):
            cols = scales[-1] * group                       # the real width of a quantized row
            n = math.prod(shape[:-1]) * cols
            bits = round(shape[-1] * 32 / cols) if dtype == "U32" and cols else 0
            if bits:
                bits_used[str(bits)] = bits_used.get(str(bits), 0) + n
            params += n
        elif not name.endswith((".scales", ".biases")):
            params += math.prod(shape) if shape else 1
    layers = text.get("num_hidden_layers") or text.get("n_layer")
    info = {"kind": "mlx", "architecture": text.get("model_type") or cfg.get("model_type", "?"),
            "layers": layers, "hidden": text.get("hidden_size"), "heads": text.get("num_attention_heads"),
            "kv_heads": text.get("num_key_value_heads"), "vocab": text.get("vocab_size"),
            "context": text.get("max_position_embeddings"),
            "experts": text.get("num_experts") or text.get("num_local_experts") or 0,
            "quantization": {"group_size": group, "bits": q.get("bits")} if q else None,
            "weights_by_bits": dict(sorted(bits_used.items(), key=lambda kv: float(kv[0]))),
            "files": files, "rows": rows, "params": params}
    notes = []
    if cfg.get("schema") or (_read_json(os.path.join(path, "jang_config.json"))):
        notes.append("JANG / Prism bundle")
    if not os.path.exists(os.path.join(path, "tokenizer.json")) and not os.path.exists(os.path.join(path, "tokenizer.model")):
        notes.append("no tokenizer file in the folder")
    info["notes"] = notes
    return info


# ───────────────────────────── GGUF ─────────────────────────────
def gguf_tensors(path):
    """[(name, ggml type id, dims fastest-first, offset)] and where the data starts."""
    meta = gguf.read_header_full(path)
    out = []
    with open(path, "rb") as f:
        f.seek(meta["_tensor_info_offset"])
        for _ in range(meta["_tensor_count"]):
            (n,) = struct.unpack("<Q", f.read(8))
            name = f.read(n).decode("utf-8", "replace")
            (nd,) = struct.unpack("<I", f.read(4))
            dims = struct.unpack(f"<{nd}Q", f.read(8 * nd))
            gt, off = struct.unpack("<IQ", f.read(12))
            out.append((name, gt, dims, off))
        end = f.tell()
    align = int(meta.get("general.alignment", 32))
    return meta, out, (end + align - 1) // align * align


def _gguf(path):
    meta, tensors, data_start = gguf_tensors(path)
    total = os.path.getsize(path)
    by_offset = sorted(tensors, key=lambda t: t[3])
    sizes = {}
    for i, t in enumerate(by_offset):                       # a tensor lasts until the next one starts
        end = by_offset[i + 1][3] if i + 1 < len(by_offset) else total - data_start
        sizes[t[0]] = max(0, end - t[3])
    rows = [(n, GGML_NAMES.get(gt, f"type {gt}"), tuple(reversed(dims)), sizes[n]) for n, gt, dims, _ in tensors]
    arch = meta.get("general.architecture", "?")
    g = lambda k, d=None: meta.get(f"{arch}.{k}", d)
    params = sum(math.prod(r[2]) if r[2] else 1 for r in rows)
    ok, why = gguf.native_support(path)
    notes = []
    for k in ("general.name", "general.size_label", "general.finetune", "general.license"):
        if meta.get(k):
            notes.append(f"{k.split('.')[-1]}: {meta[k]}")
    return {"kind": "gguf", "architecture": arch, "layers": g("block_count"), "hidden": g("embedding_length"),
            "heads": g("attention.head_count"), "kv_heads": g("attention.head_count_kv"),
            "vocab": len(gguf.read_array(path, "tokenizer.ggml.tokens") or []) or g("vocab_size"),
            "context": g("context_length"), "experts": g("expert_count") or 0, "quantization": None,
            "file_type": meta.get("general.file_type"), "tokenizer": meta.get("tokenizer.ggml.model"),
            "own_decoder": ok, "own_decoder_why": why, "rows": rows, "params": params,
            "files": [os.path.basename(path)], "notes": notes}


def inspect(path):
    """Describe a model folder (MLX) or file (GGUF)."""
    path = os.path.expanduser(path)
    if os.path.isfile(path) and path.lower().endswith(".gguf"):
        info = _gguf(path)
    elif os.path.isdir(path):
        info = _mlx(path)
    else:
        raise InspectError(f"not a model folder or a .gguf file: {path}")
    rows = info.pop("rows")
    info["path"] = path
    info["name"] = os.path.basename(path.rstrip("/"))
    info["tensors"] = len(rows)
    info["size_bytes"] = sum(r[3] for r in rows)
    info["dtypes"] = _dtypes(rows)
    info["groups"] = _groups(rows)
    return info


def human(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1000 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1000
    return f"{n:.1f} GB"


def render(info, top=14):
    """Plain text for a terminal."""
    L = []
    L.append(f"{info['name']}  ·  {info['kind'].upper()}  ·  {info['architecture']}")
    facts = [("layers", info.get("layers")), ("hidden", info.get("hidden")), ("heads", info.get("heads")),
             ("kv heads", info.get("kv_heads")), ("vocab", info.get("vocab")), ("context", info.get("context")),
             ("experts", info.get("experts") or None)]
    L.append("  " + "  ".join(f"{k} {v}" for k, v in facts if v))
    L.append(f"  {info['tensors']} tensors · {info['params'] / 1e9:.2f} B parameters · {human(info['size_bytes'])} on disk"
             + (f" · {8 * info['size_bytes'] / info['params']:.2f} bits per weight" if info["params"] else ""))
    if info.get("quantization"):
        q = info["quantization"]
        L.append(f"  quantization: group {q.get('group_size')}" + (f", {q['bits']} bits" if q.get("bits") else ""))
    if info.get("weights_by_bits"):
        tot = sum(info["weights_by_bits"].values()) or 1
        L.append("  weights by bits: " + ", ".join(f"{b}-bit {100 * n / tot:.0f}%" for b, n in info["weights_by_bits"].items()))
    if info.get("tokenizer"):
        L.append(f"  tokenizer: {info['tokenizer']}")
    if info["kind"] == "gguf":
        L.append("  own decoder: " + ("yes" if info["own_decoder"] else f"no — {info['own_decoder_why']} (llama.cpp)"))
    L.append("  storage: " + ", ".join(f"{d} {v['tensors']} ({human(v['bytes'])})"
                                      for d, v in sorted(info["dtypes"].items(), key=lambda kv: -kv[1]["bytes"])))
    L.append("")
    L.append("  largest groups (layer numbers shown as N):")
    for g in info["groups"][:top]:
        shape = "×".join(str(x) for x in g["shape"])
        L.append(f"    {human(g['bytes']):>9}  {g['count']:>4} ×  {g['dtype']:<8} {shape:<18} {g['pattern']}")
    if len(info["groups"]) > top:
        rest = info["groups"][top:]
        L.append(f"    {human(sum(g['bytes'] for g in rest)):>9}  … {len(rest)} more groups")
    for n in info.get("notes", []):
        L.append(f"  note: {n}")
    return "\n".join(L)
