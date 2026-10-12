"""A tiny llama-architecture GGUF written without the `gguf` package, plus a numpy reference forward pass.

Random weights and a byte-level BPE with a handful of merges: it exists so the reader, the repacker, the
decoder, the tokenizer and the server can be tested end to end without a downloaded model.
"""
import struct

import numpy as np

HIDDEN, FF, LAYERS, HEADS, KV_HEADS, HEAD_DIM = 64, 128, 2, 4, 2, 16
EPS, ROPE_BASE = 1e-5, 10000.0
GEMMA_LAYERS = 6                 # five sliding-window layers and the first global one (Gemma 3's 5:1 pattern)
GEMMA_WINDOW = 4
GEMMA_GLOBAL_BASE = 100000.0     # the local layers keep the default 10000
N_EXPERT, N_USED = 4, 2          # mixture-of-experts files: experts per layer, experts per token
GRANITE_SCALES = {"embedding_scale": 3.0, "residual_scale": 0.5, "logit_scale": 2.0, "attention.scale": 0.2}
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


def spm_vocab():
    """A SentencePiece vocabulary in Gemma's shape: control tokens, 256 byte pieces, every printable ASCII
    character (space is ▁), and a few merged pieces whose scores decide the merge order."""
    tokens = ["<pad>", "<eos>", "<bos>", "<unk>"]
    types = [3, 3, 3, 2]
    scores = [0.0, 0.0, 0.0, 0.0]
    for b in range(256):
        tokens.append(f"<0x{b:02X}>")
        types.append(6)
        scores.append(0.0)
    for i, c in enumerate(range(32, 127)):
        tokens.append("\u2581" if c == 32 else chr(c))
        types.append(1)
        scores.append(-100.0 - i)
    for piece, score in (("\u2581t", -1.0), ("he", -2.0), ("\u2581the", -3.0), ("in", -4.0), ("\u2581a", -5.0),
                         ("ing", -6.0), ("\u2581in", -7.0)):
        tokens.append(piece)
        types.append(1)
        scores.append(score)
    for piece in ("<start_of_turn>", "<end_of_turn>"):
        tokens.append(piece)
        types.append(3)
        scores.append(0.0)
    return tokens, types, scores


GEMMA_TEMPLATE = ("{{ bos_token }}{% for m in messages %}<start_of_turn>{{ m.role }}\n{{ m.content }}<end_of_turn>\n"
                  "{% endfor %}{% if add_generation_prompt %}<start_of_turn>model\n{% endif %}")


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
    if isinstance(value, list) and value and isinstance(value[0], float):
        return out + struct.pack("<IIQ", 9, 6, len(value)) + struct.pack(f"<{len(value)}f", *value)
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
          rope_factors=None, moe=False):
    """Write the file; returns {tensor name: the float32 values the file actually represents}."""
    rng = np.random.default_rng(seed)
    tokens, types, merges = vocab()
    gemma = arch == "gemma3"
    layers = GEMMA_LAYERS if gemma else LAYERS
    if gemma:
        tokens, types, scores = spm_vocab()
    V = len(tokens)
    meta = {
        "general.architecture": arch, "general.name": "tiny", "general.file_type": 1 if quant == "F16" else 7,
        f"{arch}.context_length": 256, f"{arch}.embedding_length": HIDDEN, f"{arch}.block_count": layers,
        f"{arch}.feed_forward_length": FF, f"{arch}.attention.head_count": HEADS,
        f"{arch}.attention.head_count_kv": KV_HEADS, f"{arch}.attention.layer_norm_rms_epsilon": EPS,
        f"{arch}.rope.freq_base": ROPE_BASE, f"{arch}.rope.dimension_count": HEAD_DIM,
        "tokenizer.ggml.model": "gpt2", "tokenizer.ggml.pre": "llama-bpe", "tokenizer.ggml.tokens": tokens,
        "tokenizer.ggml.token_type": types, "tokenizer.ggml.merges": merges,
        "tokenizer.ggml.bos_token_id": 2 if gemma else tokens.index("<|bos|>"),
        "tokenizer.ggml.eos_token_id": 1 if gemma else tokens.index("<|im_end|>"),
        "tokenizer.chat_template": CHATML,
    }
    if gemma:
        meta.update({
            "tokenizer.ggml.model": "llama", "tokenizer.ggml.scores": scores, "tokenizer.ggml.add_space_prefix": False,
            "tokenizer.ggml.bos_token_id": 2, "tokenizer.ggml.eos_token_id": 1, "tokenizer.ggml.unknown_token_id": 3,
            "tokenizer.chat_template": GEMMA_TEMPLATE, "gemma3.attention.sliding_window": GEMMA_WINDOW,
            "gemma3.attention.key_length": HEAD_DIM, "gemma3.rope.freq_base": GEMMA_GLOBAL_BASE})
        for k in ("tokenizer.ggml.pre", "tokenizer.ggml.merges"):
            meta.pop(k, None)
    if moe:
        meta.update({f"{arch}.expert_count": N_EXPERT, f"{arch}.expert_used_count": N_USED})
    if arch in ("granite", "granitemoe"):
        meta.update({f"{arch}.{k}": v for k, v in GRANITE_SCALES.items()})
    if arch == "minicpm":                                       # the same multipliers, no attention scale
        meta.update({f"{arch}.{k}": v for k, v in GRANITE_SCALES.items() if k != "attention.scale"})
    meta.update(extra_meta or {})
    if rope_factors is not None:
        meta[f"{arch}.rope.scaling.type"] = "longrope" if arch == "minicpm" else "llama3"
    values, tensors = {}, []

    def matrix(name, rows, cols, std=0.08, experts=0):
        shape = (experts, rows, cols) if experts else (rows, cols)
        w = (rng.standard_normal(shape) * std).astype(np.float32)
        dims = (cols, rows, experts) if experts else (cols, rows)
        if quant == "Q8_0":
            raw, w = q8_0(w.reshape(-1, cols))
            tensors.append((name, dims, 8, raw))
            w = w.reshape(shape)
        else:
            w = w.astype(np.float16).astype(np.float32)
            tensors.append((name, dims, 1, w.astype(np.float16).tobytes()))
        values[name] = w

    def vector(name, n, base=1.0):
        w = (base + rng.standard_normal(n) * 0.05).astype(np.float32)
        tensors.append((name, (n,), 0, w.tobytes()))
        values[name] = w

    matrix("token_embd.weight", V, HIDDEN, 0.3)
    vector("output_norm.weight", HIDDEN)
    if not gemma:                                              # Gemma ties the output to the embedding
        matrix("output.weight", V, HIDDEN, 2.0 if sharp else 0.08)   # sharp: peaked logits, so argmax is robust
    for i in range(layers):
        vector(f"blk.{i}.attn_norm.weight", HIDDEN)
        vector(f"blk.{i}.ffn_norm.weight", HIDDEN)
        matrix(f"blk.{i}.attn_q.weight", HEADS * HEAD_DIM, HIDDEN)
        matrix(f"blk.{i}.attn_k.weight", KV_HEADS * HEAD_DIM, HIDDEN)
        matrix(f"blk.{i}.attn_v.weight", KV_HEADS * HEAD_DIM, HIDDEN)
        if arch == "qwen2":                                    # Qwen2 has biases on q, k and v
            for n, rows in (("q", HEADS * HEAD_DIM), ("k", KV_HEADS * HEAD_DIM), ("v", KV_HEADS * HEAD_DIM)):
                vector(f"blk.{i}.attn_{n}.bias", rows, base=0.0)
        if arch in ("qwen3", "gemma3", "qwen3moe"):            # q and k are normalised per head
            vector(f"blk.{i}.attn_q_norm.weight", HEAD_DIM)
            vector(f"blk.{i}.attn_k_norm.weight", HEAD_DIM)
        if gemma:                                              # norms on the attention and feed-forward outputs
            vector(f"blk.{i}.post_attention_norm.weight", HIDDEN)
            vector(f"blk.{i}.post_ffw_norm.weight", HIDDEN)
        matrix(f"blk.{i}.attn_output.weight", HIDDEN, HEADS * HEAD_DIM)
        if moe:
            router = (rng.standard_normal((N_EXPERT, HIDDEN)) * 0.5).astype(np.float32)
            tensors.append((f"blk.{i}.ffn_gate_inp.weight", (HIDDEN, N_EXPERT), 0, router.tobytes()))
            values[f"blk.{i}.ffn_gate_inp.weight"] = router
            matrix(f"blk.{i}.ffn_gate_exps.weight", FF, HIDDEN, experts=N_EXPERT)
            matrix(f"blk.{i}.ffn_up_exps.weight", FF, HIDDEN, experts=N_EXPERT)
            matrix(f"blk.{i}.ffn_down_exps.weight", HIDDEN, FF, experts=N_EXPERT)
        else:
            matrix(f"blk.{i}.ffn_gate.weight", FF, HIDDEN)
            matrix(f"blk.{i}.ffn_up.weight", FF, HIDDEN)
            matrix(f"blk.{i}.ffn_down.weight", HIDDEN, FF)
    if rope_factors is not None and arch == "minicpm":        # LongRoPE: a short and a long table, short is read
        w = np.asarray(rope_factors, np.float32)
        for name, table in (("rope_factors_short.weight", w), ("rope_factors_long.weight", w * 3.0)):
            tensors.append((name, (len(w),), 0, table.tobytes()))
            values[name] = table
        values["rope_freqs.weight"] = w
    elif rope_factors is not None:                            # per-dimension divisors of the rotation frequencies
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
def reference_logits(values, ids, arch="llama", rope_scale=1.0, moe=False, norm_topk=True):
    """float32 logits [len(ids), V]. llama: ggml's NORM rope (adjacent pairs rotate); qwen2/qwen3: NEOX rope
    (first half with second half), qwen2 adds q/k/v biases, qwen3 an RMSNorm of q and k per head. gemma3: the
    qwen3 layout plus an embedding scaled by sqrt(hidden), norms after attention and feed-forward, GeGLU
    (tanh), tied output, sliding-window attention in layers 0-4 of every six (own rope base, no scaling) and
    full attention with the global base (positions scaled by `rope_scale`) in the sixth."""
    gemma = arch == "gemma3"
    n_layers = GEMMA_LAYERS if gemma else LAYERS

    def rms(x, w):
        return x / np.sqrt((x * x).mean(-1, keepdims=True) + EPS) * w

    def rope(x, pos, base=ROPE_BASE, scale=1.0):          # x [T, heads, D]
        D = x.shape[-1]
        pos = pos * scale
        inv = base ** (-np.arange(0, D, 2) / D)
        if "rope_freqs.weight" in values:
            inv = inv / values["rope_freqs.weight"]
        ang = pos[:, None] * inv[None, :]
        c, s = np.cos(ang)[:, None, :], np.sin(ang)[:, None, :]
        if arch in ("llama", "granite", "granitemoe", "minicpm"):
            a, b = x[..., 0::2], x[..., 1::2]
            out = np.empty_like(x)
            out[..., 0::2], out[..., 1::2] = a * c - b * s, a * s + b * c
            return out
        h = D // 2
        a, b = x[..., :h], x[..., h:]
        return np.concatenate([a * c - b * s, b * c + a * s], axis=-1)

    T = len(ids)
    pos = np.arange(T)
    granite = arch in ("granite", "granitemoe", "minicpm")
    x = values["token_embd.weight"][ids]
    if gemma:
        x = x * np.sqrt(HIDDEN)
    if granite:
        x = x * GRANITE_SCALES["embedding_scale"]
    res = GRANITE_SCALES["residual_scale"] if granite else 1.0
    att_scale = GRANITE_SCALES["attention.scale"] if arch in ("granite", "granitemoe") else 1.0 / np.sqrt(HEAD_DIM)
    for i in range(n_layers):
        p = f"blk.{i}."
        local = gemma and i % 6 != 5
        h = rms(x, values[p + "attn_norm.weight"])
        q = h @ values[p + "attn_q.weight"].T
        k = h @ values[p + "attn_k.weight"].T
        v = h @ values[p + "attn_v.weight"].T
        if arch == "qwen2":
            q, k, v = q + values[p + "attn_q.bias"], k + values[p + "attn_k.bias"], v + values[p + "attn_v.bias"]
        q = q.reshape(T, HEADS, HEAD_DIM)
        k = k.reshape(T, KV_HEADS, HEAD_DIM)
        v = v.reshape(T, KV_HEADS, HEAD_DIM)
        if arch in ("qwen3", "gemma3", "qwen3moe"):
            q, k = rms(q, values[p + "attn_q_norm.weight"]), rms(k, values[p + "attn_k_norm.weight"])
        if gemma:
            base, scale = (ROPE_BASE, 1.0) if local else (GEMMA_GLOBAL_BASE, rope_scale)
            q, k = rope(q, pos, base, scale), rope(k, pos, base, scale)
        else:
            q, k = rope(q, pos), rope(k, pos)
        rep = HEADS // KV_HEADS
        k, v = np.repeat(k, rep, axis=1), np.repeat(v, rep, axis=1)
        att = np.einsum("thd,shd->hts", q, k) * att_scale
        att = att + np.triu(np.full((T, T), -np.inf), 1)[None]
        if local:                                          # only the last GEMMA_WINDOW positions (itself included)
            att = att + np.tril(np.full((T, T), -np.inf), -GEMMA_WINDOW)[None]
        att = np.exp(att - att.max(-1, keepdims=True))
        att /= att.sum(-1, keepdims=True)
        o = np.einsum("hts,shd->thd", att, v).reshape(T, HEADS * HEAD_DIM)
        o = o @ values[p + "attn_output.weight"].T
        if gemma:
            o = rms(o, values[p + "post_attention_norm.weight"])
        x = x + o * res
        h = rms(x, values[p + "ffn_norm.weight"])
        if moe:
            logits = h @ values[p + "ffn_gate_inp.weight"].T                       # [T, E]
            probs = np.exp(logits - logits.max(-1, keepdims=True))
            probs /= probs.sum(-1, keepdims=True)
            d = np.zeros_like(h)
            for t in range(T):
                top = np.argsort(-probs[t])[:N_USED]
                wts = probs[t, top] / (probs[t, top].sum() if norm_topk else 1.0)
                for e, wt in zip(top, wts):
                    g = values[p + "ffn_gate_exps.weight"][e] @ h[t]
                    u = values[p + "ffn_up_exps.weight"][e] @ h[t]
                    d[t] += wt * (values[p + "ffn_down_exps.weight"][e] @ (g / (1 + np.exp(-g)) * u))
        else:
            g, u = h @ values[p + "ffn_gate.weight"].T, h @ values[p + "ffn_up.weight"].T
            act = (0.5 * g * (1 + np.tanh(np.sqrt(2 / np.pi) * (g + 0.044715 * g ** 3)))) if gemma else g / (1 + np.exp(-g))
            d = (act * u) @ values[p + "ffn_down.weight"].T
        if gemma:
            d = rms(d, values[p + "post_ffw_norm.weight"])
        x = x + d * res
    head = values["token_embd.weight"] if gemma else values["output.weight"]
    out = rms(x, values["output_norm.weight"]) @ head.T
    return out / GRANITE_SCALES["logit_scale"] if granite else out
