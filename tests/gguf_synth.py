"""A tiny llama-architecture GGUF written without the `gguf` package, plus a numpy reference forward pass.

Random weights and a byte-level BPE with a handful of merges: it exists so the reader, the repacker, the
decoder, the tokenizer and the server can be tested end to end without a downloaded model.
"""
import struct

import numpy as np

HIDDEN, FF, LAYERS, HEADS, KV_HEADS, HEAD_DIM = 64, 128, 2, 4, 2, 16
EPS, ROPE_BASE = 1e-5, 10000.0
CHATML = ("{% for m in messages %}{{ '<|im_start|>' + m.role + '\n' + m.content + '<|im_end|>\n' }}{% endfor %}"
          "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}")


def bytes_to_unicode():
    """GPT-2's byte → printable character table (the BPE vocabulary is written in it)."""
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


def vocab():
    table = bytes_to_unicode()
    tokens = [table[b] for b in range(256)]
    merges = [("Ġ", "t"), ("h", "e"), ("Ġt", "he"), ("i", "n"), ("Ġ", "a"), ("!", "a")]
    for a, b in merges:
        tokens.append(a + b)
    specials = ["<|bos|>", "<|im_start|>", "<|im_end|>"]
    types = [1] * len(tokens) + [3] * len(specials)
    return tokens + specials, types, [f"{a} {b}" for a, b in merges]


def _kv(key, value):
    def s(x):
        b = x.encode()
        return struct.pack("<Q", len(b)) + b
    out = s(key)
    if isinstance(value, str):
        return out + struct.pack("<I", 8) + s(value)
    if isinstance(value, bool):
        return out + struct.pack("<IB", 7, int(value))
    if isinstance(value, int):
        return out + struct.pack("<II", 4, value)
    if isinstance(value, float):
        return out + struct.pack("<If", 6, value)
    if isinstance(value, list) and value and isinstance(value[0], str):
        return out + struct.pack("<IIQ", 9, 8, len(value)) + b"".join(s(v) for v in value)
    if isinstance(value, list):
        return out + struct.pack("<IIQ", 9, 5, len(value)) + struct.pack(f"<{len(value)}i", *value)
    raise TypeError(key)


def q8_0(w):
    """float32 [rows, cols] → ggml Q8_0 bytes (blocks of 32: fp16 scale + 32 int8)."""
    blocks = w.reshape(-1, 32)
    d = np.abs(blocks).max(axis=1) / 127.0
    d16 = d.astype(np.float16)
    safe = np.where(d16 == 0, 1, d16.astype(np.float32))
    q = np.clip(np.rint(blocks / safe[:, None]), -127, 127).astype(np.int8)
    out = np.empty((blocks.shape[0], 34), np.uint8)
    out[:, :2] = d16.view(np.uint8).reshape(-1, 2)
    out[:, 2:] = q.view(np.uint8)
    return out.tobytes(), (q.astype(np.float32) * d16.astype(np.float32)[:, None]).reshape(w.shape)


def write(path, seed=0, quant="F16", arch="llama", extra_meta=None, tensor_type_override=None, sharp=False,
          rope_factors=None):
    """Write the file; returns {tensor name: the float32 values the file actually represents}."""
    rng = np.random.default_rng(seed)
    tokens, types, merges = vocab()
    V = len(tokens)
    meta = {
        "general.architecture": arch, "general.name": "tiny", "general.file_type": 1 if quant == "F16" else 7,
        f"{arch}.context_length": 256, f"{arch}.embedding_length": HIDDEN, f"{arch}.block_count": LAYERS,
        f"{arch}.feed_forward_length": FF, f"{arch}.attention.head_count": HEADS,
        f"{arch}.attention.head_count_kv": KV_HEADS, f"{arch}.attention.layer_norm_rms_epsilon": EPS,
        f"{arch}.rope.freq_base": ROPE_BASE, f"{arch}.rope.dimension_count": HEAD_DIM,
        "tokenizer.ggml.model": "gpt2", "tokenizer.ggml.pre": "llama-bpe", "tokenizer.ggml.tokens": tokens,
        "tokenizer.ggml.token_type": types, "tokenizer.ggml.merges": merges,
        "tokenizer.ggml.bos_token_id": tokens.index("<|bos|>"), "tokenizer.ggml.eos_token_id": tokens.index("<|im_end|>"),
        "tokenizer.chat_template": CHATML,
    }
    meta.update(extra_meta or {})
    if rope_factors is not None:
        meta[f"{arch}.rope.scaling.type"] = "llama3"
    values, tensors = {}, []

    def matrix(name, rows, cols, std=0.08):
        w = (rng.standard_normal((rows, cols)) * std).astype(np.float32)
        if quant == "Q8_0":
            raw, w = q8_0(w)
            tensors.append((name, (cols, rows), 8, raw))
        else:
            w = w.astype(np.float16).astype(np.float32)
            tensors.append((name, (cols, rows), 1, w.astype(np.float16).tobytes()))
        values[name] = w

    def vector(name, n, base=1.0):
        w = (base + rng.standard_normal(n) * 0.05).astype(np.float32)
        tensors.append((name, (n,), 0, w.tobytes()))
        values[name] = w

    matrix("token_embd.weight", V, HIDDEN, 0.3)
    vector("output_norm.weight", HIDDEN)
    matrix("output.weight", V, HIDDEN, 2.0 if sharp else 0.08)   # sharp: peaked logits, so argmax is robust
    for i in range(LAYERS):
        vector(f"blk.{i}.attn_norm.weight", HIDDEN)
        vector(f"blk.{i}.ffn_norm.weight", HIDDEN)
        matrix(f"blk.{i}.attn_q.weight", HEADS * HEAD_DIM, HIDDEN)
        matrix(f"blk.{i}.attn_k.weight", KV_HEADS * HEAD_DIM, HIDDEN)
        matrix(f"blk.{i}.attn_v.weight", KV_HEADS * HEAD_DIM, HIDDEN)
        if arch == "qwen2":                                    # Qwen2 has biases on q, k and v
            for n, rows in (("q", HEADS * HEAD_DIM), ("k", KV_HEADS * HEAD_DIM), ("v", KV_HEADS * HEAD_DIM)):
                vector(f"blk.{i}.attn_{n}.bias", rows, base=0.0)
        if arch == "qwen3":                                    # Qwen3 normalises q and k per head
            vector(f"blk.{i}.attn_q_norm.weight", HEAD_DIM)
            vector(f"blk.{i}.attn_k_norm.weight", HEAD_DIM)
        matrix(f"blk.{i}.attn_output.weight", HIDDEN, HEADS * HEAD_DIM)
        matrix(f"blk.{i}.ffn_gate.weight", FF, HIDDEN)
        matrix(f"blk.{i}.ffn_up.weight", FF, HIDDEN)
        matrix(f"blk.{i}.ffn_down.weight", HIDDEN, FF)
    if rope_factors is not None:                              # per-dimension divisors of the rotation frequencies
        w = np.asarray(rope_factors, np.float32)
        tensors.append(("rope_freqs.weight", (len(w),), 0, w.tobytes()))
        values["rope_freqs.weight"] = w
    if tensor_type_override:
        n, t = tensor_type_override
        tensors = [(a, b, t if a == n else c, d) for a, b, c, d in tensors]

    head = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(meta)) + b"".join(_kv(k, v) for k, v in meta.items())
    infos, offset, blobs = b"", 0, []
    for name, dims, gtype, raw in tensors:
        nb = name.encode()
        infos += struct.pack("<Q", len(nb)) + nb + struct.pack("<I", len(dims)) + struct.pack(f"<{len(dims)}Q", *dims)
        infos += struct.pack("<IQ", gtype, offset)
        pad = (-len(raw)) % 32
        blobs.append(raw + b"\0" * pad)
        offset += len(raw) + pad
    pre = head + infos
    pre += b"\0" * ((-len(pre)) % 32)
    with open(path, "wb") as f:
        f.write(pre + b"".join(blobs))
    return values


# ───────────────────────────── a plain numpy llama forward, the independent reference ─────────────────────────────
def reference_logits(values, ids, arch="llama"):
    """float32 logits [len(ids), V]. llama: ggml's NORM rope (adjacent pairs rotate); qwen2/qwen3: NEOX rope
    (first half with second half), qwen2 adds q/k/v biases, qwen3 an RMSNorm of q and k per head."""
    def rms(x, w):
        return x / np.sqrt((x * x).mean(-1, keepdims=True) + EPS) * w

    def rope(x, pos):                                     # x [T, heads, D]
        D = x.shape[-1]
        inv = ROPE_BASE ** (-np.arange(0, D, 2) / D)
        if "rope_freqs.weight" in values:
            inv = inv / values["rope_freqs.weight"]
        ang = pos[:, None] * inv[None, :]
        c, s = np.cos(ang)[:, None, :], np.sin(ang)[:, None, :]
        if arch == "llama":
            a, b = x[..., 0::2], x[..., 1::2]
            out = np.empty_like(x)
            out[..., 0::2], out[..., 1::2] = a * c - b * s, a * s + b * c
            return out
        h = D // 2
        a, b = x[..., :h], x[..., h:]
        return np.concatenate([a * c - b * s, b * c + a * s], axis=-1)

    T = len(ids)
    pos = np.arange(T)
    x = values["token_embd.weight"][ids]
    for i in range(LAYERS):
        p = f"blk.{i}."
        h = rms(x, values[p + "attn_norm.weight"])
        q = h @ values[p + "attn_q.weight"].T
        k = h @ values[p + "attn_k.weight"].T
        v = h @ values[p + "attn_v.weight"].T
        if arch == "qwen2":
            q, k, v = q + values[p + "attn_q.bias"], k + values[p + "attn_k.bias"], v + values[p + "attn_v.bias"]
        q = q.reshape(T, HEADS, HEAD_DIM)
        k = k.reshape(T, KV_HEADS, HEAD_DIM)
        v = v.reshape(T, KV_HEADS, HEAD_DIM)
        if arch == "qwen3":
            q, k = rms(q, values[p + "attn_q_norm.weight"]), rms(k, values[p + "attn_k_norm.weight"])
        q, k = rope(q, pos), rope(k, pos)
        rep = HEADS // KV_HEADS
        k, v = np.repeat(k, rep, axis=1), np.repeat(v, rep, axis=1)
        att = np.einsum("thd,shd->hts", q, k) / np.sqrt(HEAD_DIM)
        att = att + np.triu(np.full((T, T), -np.inf), 1)[None]
        att = np.exp(att - att.max(-1, keepdims=True))
        att /= att.sum(-1, keepdims=True)
        o = np.einsum("hts,shd->thd", att, v).reshape(T, HEADS * HEAD_DIM)
        x = x + o @ values[p + "attn_output.weight"].T
        h = rms(x, values[p + "ffn_norm.weight"])
        g, u = h @ values[p + "ffn_gate.weight"].T, h @ values[p + "ffn_up.weight"].T
        x = x + (g / (1 + np.exp(-g)) * u) @ values[p + "ffn_down.weight"].T
    return rms(x, values["output_norm.weight"]) @ values["output.weight"].T
