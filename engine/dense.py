"""A dense decoder-only transformer on MLX, built from GGUF weights (no mlx-lm, no llama.cpp).

One implementation covers the Llama family as ggml names it: llama (incl. Mistral, SmolLM2, Llama 3.x),
qwen2 (incl. DeepSeek-R1-Distill-Qwen, MiMo's base), qwen3, granite, and Gemma 3 (sliding-window layers in a 5:1 pattern,
two rope bases, a norm after the attention and after the feed-forward, GeGLU). Mixture-of-experts feed-forwards
(Mixtral, Qwen3-MoE, Granite-MoE) keep the experts stacked and run only the routed ones with `gather_qmm`.
Weights stay in the quantized form of the
file (engine/gguf_tensors.py); the residual stream is float32, matmul inputs and the KV cache float16.

Everything the architectures differ in is data in `Config`, so adding a family is a few lines there plus a
verification against llama.cpp — not a new model class.
"""
import os
from dataclasses import dataclass, field

import mlx.core as mx
import numpy as np


@dataclass
class Config:
    arch: str
    n_layer: int
    n_embd: int
    n_head: int
    n_head_kv: int
    n_ff: int
    head_dim: int
    vocab: int
    eps: float = 1e-5
    rope_base: float = 10000.0
    rope_dims: int = 0
    rope_traditional: bool = False       # interleaved pairs (ggml "norm") vs half-split (ggml "neox")
    rope_freqs: object = None            # per-dimension factors (Llama 3.1 scaling), or None
    qkv_bias: bool = False
    qk_norm: bool = False
    tied_output: bool = False
    context: int = 4096
    attn_scale: float = 0.0              # 0 → 1/sqrt(head_dim)
    embed_scale: float = 1.0             # granite / minicpm / gemma style multipliers
    logit_scale: float = 1.0
    residual_scale: float = 1.0
    gelu: bool = False                   # GeGLU (tanh approximation) instead of SwiGLU
    sandwich: bool = False               # extra norms on the attention and feed-forward outputs (Gemma 2/3)
    window: int = 0                      # sliding-window size of the local layers, 0 = none
    window_pattern: int = 0              # every `pattern`-th layer is global, the rest local (Gemma 3: 6)
    rope_base_local: float = 0.0         # rope base of the local layers
    rope_scale: float = 1.0              # linear position scaling of the global layers (Gemma 3 4B+: 1/8)
    n_expert: int = 0                    # mixture of experts: experts per layer and how many a token uses
    n_expert_used: int = 0
    expert_norm: bool = True             # the chosen experts' weights are renormalised to sum to 1
    extra: dict = field(default_factory=dict)

    def is_local(self, i):
        return bool(self.window and self.window_pattern and (i % self.window_pattern) != self.window_pattern - 1)


# ggml's rope type per architecture (llama.cpp: llama_model_rope_type)
ROPE_TRADITIONAL = {"llama": True, "qwen2": False, "qwen3": False, "gemma3": False,
                    "granite": True, "granitemoe": True, "qwen3moe": False, "minicpm": True}
SUPPORTED = tuple(ROPE_TRADITIONAL)


class QLinear:
    """y = x @ W.T for a weight kept in MLX's affine-quantized form (or dense)."""

    def __init__(self, w, scales=None, biases=None, bits=0, bias=None):
        self.w, self.scales, self.biases, self.bits, self.bias = w, scales, biases, bits, bias

    def __call__(self, x):
        if self.bits:
            y = mx.quantized_matmul(x, self.w, self.scales, self.biases, transpose=True, group_size=32, bits=self.bits)
        else:
            y = x @ self.w.T
        return y + self.bias if self.bias is not None else y

    @property
    def nbytes(self):
        n = self.w.nbytes + (self.scales.nbytes + self.biases.nbytes if self.bits else 0)
        return n + (self.bias.nbytes if self.bias is not None else 0)


class Embedding:
    def __init__(self, w, scales=None, biases=None, bits=0):
        self.w, self.scales, self.biases, self.bits = w, scales, biases, bits

    def __call__(self, ids):
        if self.bits:
            return mx.dequantize(self.w[ids], self.scales[ids], self.biases[ids], group_size=32, bits=self.bits)
        return self.w[ids]

    def as_linear(self):
        return QLinear(self.w, self.scales, self.biases, self.bits)


class Experts:
    """The stacked weights of one projection of all experts: [E, rows, packed cols] (+ scales, biases)."""

    def __init__(self, w, scales=None, biases=None, bits=0):
        self.w, self.scales, self.biases, self.bits = w, scales, biases, bits

    def __call__(self, x, idx, sorted_idx):
        if self.bits:
            return mx.gather_qmm(x, self.w, self.scales, self.biases, rhs_indices=idx, transpose=True,
                                 group_size=32, bits=self.bits, sorted_indices=sorted_idx)
        return mx.gather_mm(x, self.w.swapaxes(-1, -2), rhs_indices=idx, sorted_indices=sorted_idx)

    @property
    def nbytes(self):
        return self.w.nbytes + (self.scales.nbytes + self.biases.nbytes if self.bits else 0)


def _gather_sort(x, indices):
    """Order the (token, expert) pairs by expert so each expert's weights are read once per pass."""
    *_, m = indices.shape
    flat = indices.flatten()
    order = mx.argsort(flat)
    return x.flatten(0, -3)[order // m], flat[order], mx.argsort(order)


class KVCache:
    """Keys and values of one layer, grown in steps; `trim` forgets the newest tokens (speculation, reuse)."""
    STEP = 256

    def __init__(self):
        self.keys = self.values = None
        self.offset = 0

    def update_and_fetch(self, k, v):
        n = k.shape[2]
        if self.keys is None or self.offset + n > self.keys.shape[2]:
            B, H, _, D = k.shape
            cap = ((self.offset + n + self.STEP - 1) // self.STEP) * self.STEP
            nk = mx.zeros((B, H, cap, D), k.dtype)
            nv = mx.zeros((B, H, cap, v.shape[3]), v.dtype)
            if self.keys is not None:
                nk[..., :self.offset, :] = self.keys[..., :self.offset, :]
                nv[..., :self.offset, :] = self.values[..., :self.offset, :]
            self.keys, self.values = nk, nv
        self.keys[..., self.offset:self.offset + n, :] = k
        self.values[..., self.offset:self.offset + n, :] = v
        self.offset += n
        return self.keys[..., :self.offset, :], self.values[..., :self.offset, :]

    @property
    def state(self):
        return [] if self.keys is None else [self.keys, self.values]

    @property
    def nbytes(self):
        return 0 if self.keys is None else self.keys.nbytes + self.values.nbytes

    def is_trimmable(self):
        return True

    def trim(self, n):
        n = min(n, self.offset)
        self.offset -= n
        return n


def _rms(x, w, eps):
    return mx.fast.rms_norm(x, w, eps)


def _gelu_tanh(x):
    return 0.5 * x * (1.0 + mx.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)))


class Layer:
    def __init__(self, cfg, tensors, i):
        t = lambda name: tensors[f"blk.{i}.{name}"]
        self.attn_norm, self.ffn_norm = t("attn_norm.weight"), t("ffn_norm.weight")
        self.q, self.k, self.v, self.o = (t(f"attn_{n}.weight") for n in ("q", "k", "v", "output"))
        self.q_norm = tensors.get(f"blk.{i}.attn_q_norm.weight")
        self.k_norm = tensors.get(f"blk.{i}.attn_k_norm.weight")
        self.router = tensors.get(f"blk.{i}.ffn_gate_inp.weight")
        if self.router is not None:                          # a mixture-of-experts layer
            self.gate_e, self.up_e, self.down_e = (t(f"ffn_{n}_exps.weight") for n in ("gate", "up", "down"))
            self.gate = self.up = self.down = None
        else:
            self.gate, self.up, self.down = t("ffn_gate.weight"), t("ffn_up.weight"), t("ffn_down.weight")
        self.post_attn_norm = tensors.get(f"blk.{i}.post_attention_norm.weight")
        self.post_ffn_norm = tensors.get(f"blk.{i}.post_ffw_norm.weight")

    def params(self):
        ms = (self.q, self.k, self.v, self.o, self.gate, self.up, self.down)
        if self.router is not None:
            ms = (self.q, self.k, self.v, self.o, self.router, self.gate_e, self.up_e, self.down_e)
        return [m for m in ms]


class DenseModel:
    """Callable like the mlx-lm models: model(ids [B, L], cache=[KVCache, ...]) → logits [B, L, vocab]."""

    def __init__(self, cfg, tensors):
        self.cfg = cfg
        self.embed = tensors["token_embd.weight"]
        self.norm = tensors["output_norm.weight"]
        self.head = tensors.get("output.weight") or self.embed.as_linear()
        self.layers = [Layer(cfg, tensors, i) for i in range(cfg.n_layer)]
        self.steer = None            # y -> y', see engine/steering.py: applied where layers write the residual
        mode = os.environ.get("LLM_GGUF_ACT", "mixed")
        self.act = mx.float32 if mode == "f32" else mx.float16      # dtype fed to the matmuls
        self.wide = mode in ("f32", "mixed")                          # residual stream + swiglu in float32
        self.args = type("Args", (), {"vocab_size": cfg.vocab})()
        self._freqs = None
        if cfg.rope_freqs is not None:
            d = cfg.rope_dims
            base = cfg.rope_base ** (mx.arange(0, d, 2, dtype=mx.float32) / d)
            self._freqs = base * mx.array(np.asarray(cfg.rope_freqs, np.float32))
        M = self.act
        R = mx.float32 if self.wide else mx.float16
        for L in self.layers:                  # weights in the dtype they are used in, converted once
            L.attn_norm, L.ffn_norm = L.attn_norm.astype(R), L.ffn_norm.astype(R)
            if L.q_norm is not None:
                L.q_norm, L.k_norm = L.q_norm.astype(M), L.k_norm.astype(M)
            if L.post_attn_norm is not None:
                L.post_attn_norm, L.post_ffn_norm = L.post_attn_norm.astype(R), L.post_ffn_norm.astype(R)
        self.norm = self.norm.astype(R)
        self.nbytes = (sum(m.nbytes for L in self.layers for m in L.params())
                       + self.embed.w.nbytes + (self.embed.scales.nbytes + self.embed.biases.nbytes if self.embed.bits else 0)
                       + (self.head.nbytes if "output.weight" in tensors else 0))

    def make_cache(self):
        return [KVCache() for _ in self.layers]

    def _moe(self, layer, h):
        """Route each token to its top-k experts (softmax over all, optionally renormalised over the chosen),
        run only those, and add their outputs weighted."""
        c = self.cfg
        M = self.act
        R = mx.float32 if self.wide else mx.float16
        logits = layer.router(h).astype(mx.float32)                        # [B, L, E]
        probs = mx.softmax(logits, axis=-1)
        k = c.n_expert_used
        idx = mx.argpartition(-logits, kth=k - 1, axis=-1)[..., :k]        # [B, L, k]
        w = mx.take_along_axis(probs, idx, axis=-1)
        if c.expert_norm:
            w = w / w.sum(-1, keepdims=True)
        x = mx.expand_dims(h, (-2, -3))                                    # [B, L, 1, 1, D]
        order = None
        do_sort = idx.size >= 64                                           # prefill: group tokens by expert
        sel = idx
        if do_sort:
            x, sel, order = _gather_sort(x, idx)
        g = layer.gate_e(x, sel, do_sort).astype(R)
        act = _gelu_tanh(g) if c.gelu else mx.sigmoid(g) * g
        inner = (act * layer.up_e(x, sel, do_sort).astype(R)).astype(M)
        d = layer.down_e(inner, sel, do_sort)
        if do_sort:
            d = mx.unflatten(d[order], 0, idx.shape)
        d = d.squeeze(-2).astype(R)                                        # [B, L, k, D]
        return (d * w[..., None].astype(R)).sum(-2)

    def _rope(self, x, offset, local=False):
        c = self.cfg
        if self._freqs is not None:
            return mx.fast.rope(x, c.rope_dims, traditional=c.rope_traditional, base=None, scale=1.0,
                                offset=offset, freqs=self._freqs)
        if local and c.rope_base_local:                 # Gemma 3 local layers: their own base, no scaling
            return mx.fast.rope(x, c.rope_dims, traditional=c.rope_traditional, base=c.rope_base_local,
                                scale=1.0, offset=offset)
        return mx.fast.rope(x, c.rope_dims, traditional=c.rope_traditional, base=c.rope_base,
                            scale=c.rope_scale, offset=offset)

    def __call__(self, ids, cache=None):
        """The residual stream and the SwiGLU product are float32 (as in ggml: Llama 3.2 and others carry a few
        huge activations that float16 rounds away, and greedy text drifted from llama.cpp's after a handful of
        tokens); the matmuls take float16 inputs — the weights are quantized anyway — and keys/values are
        stored in float16."""
        c = self.cfg
        M, F16, F32 = self.act, mx.float16, mx.float32
        R = F32 if self.wide else F16                    # dtype of the residual stream
        B, L = ids.shape
        if cache is None:
            cache = self.make_cache()
        x = self.embed(ids).astype(R)
        if c.embed_scale != 1.0:
            x = x * c.embed_scale
        st = self.steer or (lambda y: y)
        x = st(x)
        off = cache[0].offset
        mask = None
        if L > 1:
            q_pos = mx.arange(off, off + L)[:, None]
            k_pos = mx.arange(off + L)[None, :]
            mask = mx.where(k_pos <= q_pos, 0.0, -mx.inf).astype(F16)
        # sliding-window layers look at the last `window` keys only: slice them (and mask what is still too old)
        w_start, w_mask = 0, None
        if c.window and off + L > c.window:
            w_start = max(0, off + 1 - c.window)
            q_pos = mx.arange(off, off + L)[:, None]
            k_pos = mx.arange(w_start, off + L)[None, :]
            w_mask = mx.where((k_pos <= q_pos) & (q_pos - k_pos < c.window), 0.0, -mx.inf).astype(F16)
        elif c.window:
            w_mask = mask                       # nothing is older than the window yet
        scale = c.attn_scale or c.head_dim ** -0.5
        for li, (layer, kv) in enumerate(zip(self.layers, cache)):
            local = c.is_local(li)
            h = _rms(x, layer.attn_norm, c.eps).astype(M)
            q = layer.q(h).reshape(B, L, c.n_head, c.head_dim)
            k = layer.k(h).reshape(B, L, c.n_head_kv, c.head_dim)
            v = layer.v(h).reshape(B, L, c.n_head_kv, c.head_dim)
            if layer.q_norm is not None:
                q = _rms(q, layer.q_norm, c.eps)
                k = _rms(k, layer.k_norm, c.eps)
            q, k, v = (a.transpose(0, 2, 1, 3) for a in (q, k, v))
            q, k = self._rope(q, off, local), self._rope(k, off, local)
            k, v = kv.update_and_fetch(k.astype(F16), v.astype(F16))
            m = mask
            if local:
                m = w_mask
                if w_start:
                    k, v = k[..., w_start:, :], v[..., w_start:, :]
            a = mx.fast.scaled_dot_product_attention(q.astype(F16), k, v, scale=scale, mask=m).astype(M)
            o = layer.o(a.transpose(0, 2, 1, 3).reshape(B, L, -1)).astype(R)
            if layer.post_attn_norm is not None:
                o = _rms(o, layer.post_attn_norm, c.eps)
            x = x + st(o) * c.residual_scale
            h = _rms(x, layer.ffn_norm, c.eps).astype(M)
            if layer.router is not None:
                d = self._moe(layer, h)
            else:
                g = layer.gate(h).astype(R)
                act = _gelu_tanh(g) if c.gelu else mx.sigmoid(g) * g
                inner = (act * layer.up(h).astype(R)).astype(M)
                d = layer.down(inner).astype(R)
            if layer.post_ffn_norm is not None:
                d = _rms(d, layer.post_ffn_norm, c.eps)
            x = x + st(d) * c.residual_scale
        x = _rms(x, self.norm, c.eps).astype(M)
        logits = self.head(x)
        return logits * (1.0 / c.logit_scale) if c.logit_scale != 1.0 else logits


# ───────────────────────────── building from a GGUF ─────────────────────────────
class Unsupported(ValueError):
    pass


def _linear(gf, info, T, bias_info=None):
    name = info.type_name
    bias = None
    if bias_info is not None:
        bias = mx.array(T.load_dense(gf, bias_info)).astype(mx.float16)
    if name in ("F32", "F16", "BF16"):
        return QLinear(mx.array(T.load_dense(gf, info)).astype(mx.float16), bias=bias)
    w, s, b, bits = T.load_matrix(gf, info)
    return QLinear(mx.array(w), mx.array(s), mx.array(b), bits, bias)


def _experts(gf, info, T):
    """A stacked 3-D expert tensor [E, rows, cols] in MLX's quantized form, same layout per expert."""
    n_exp, rows, cols = info.shape
    name = info.type_name
    if name in ("F32", "F16", "BF16"):
        return Experts(mx.array(T.load_dense(gf, info)).astype(mx.float16))
    # the loader works row by row: the stack is E*rows rows of `cols` (the bytes are contiguous in that order)
    flat = T.TensorInfo(info.name, (cols, n_exp * rows), info.type, info.offset)
    w, s, b, bits = T.load_matrix(gf, flat)
    shape = lambda a: (n_exp, rows) + a.shape[1:]
    return Experts(mx.array(w).reshape(shape(w)), mx.array(s).reshape(shape(s)), mx.array(b).reshape(shape(b)), bits)


def build(path, log=print):
    """(DenseModel, Config, GGUFFile). Raises Unsupported for anything this decoder does not do — the caller
    then runs the file on llama.cpp instead."""
    from . import gguf_tensors as T
    gf = T.GGUFFile(path)
    m = gf.meta
    arch = m.get("general.architecture")
    if arch not in SUPPORTED:
        raise Unsupported(f"architecture {arch!r}")
    g = lambda key, default=None: m.get(f"{arch}.{key}", default)
    n_head = int(g("attention.head_count"))
    n_embd = int(g("embedding_length"))
    head_dim = int(g("attention.key_length", n_embd // n_head))
    if g("attention.value_length", head_dim) != head_dim:
        raise Unsupported("different key and value head sizes")
    scaling = m.get(f"{arch}.rope.scaling.type")
    longrope = scaling == "longrope" and arch == "minicpm"
    if scaling not in (None, "none", "llama3") and not (scaling == "linear" and arch == "gemma3") and not longrope:
        raise Unsupported(f"rope scaling {scaling!r}")
    if longrope:
        # llama.cpp picks the short factors while the context fits the original length (the engine's never exceeds)
        orig = g("rope.scaling.original_context_length")
        if "rope_factors_short.weight" not in gf.tensors:
            raise Unsupported("longrope scaling without its short factors")
        if orig and int(orig) < int(g("context_length", orig)):
            raise Unsupported("longrope scaling with a longer context than the original")
    if scaling == "llama3" and "rope_freqs.weight" not in gf.tensors:
        raise Unsupported("llama3 rope scaling without its frequency table")
    if g("attention.sliding_window") and arch != "gemma3":
        raise Unsupported("sliding-window attention")
    names = gf.tensors
    n_expert = int(g("expert_count", 0) or 0)
    if n_expert and ("blk.0.ffn_gate_exps.weight" not in names or "blk.0.ffn_gate_inp.weight" not in names):
        raise Unsupported("mixture of experts stored per expert (or with a shared expert)")
    if any(n.startswith("blk.0.ffn_gate_shexp") or n.startswith("blk.0.ffn_up_shexp") for n in names):
        raise Unsupported("mixture of experts with a shared expert")
    vocab = names["token_embd.weight"].shape[0]
    cfg = Config(
        arch=arch, n_layer=int(g("block_count")), n_embd=n_embd, n_head=n_head,
        n_head_kv=int(g("attention.head_count_kv", n_head)), n_ff=int(g("feed_forward_length")), head_dim=head_dim,
        vocab=vocab, eps=float(g("attention.layer_norm_rms_epsilon", 1e-5)), rope_base=float(g("rope.freq_base", 10000.0)),
        rope_dims=int(g("rope.dimension_count", head_dim)), rope_traditional=ROPE_TRADITIONAL[arch],
        qkv_bias="blk.0.attn_q.bias" in names, qk_norm="blk.0.attn_q_norm.weight" in names,
        tied_output="output.weight" not in names, context=int(g("context_length", 4096)),
    )
    cfg.n_expert, cfg.n_expert_used = n_expert, int(g("expert_used_count", 0) or 0)
    if n_expert:
        if not 0 < cfg.n_expert_used <= n_expert:
            raise Unsupported("the number of experts used per token is missing")
        # Mixtral and Granite-MoE renormalise the chosen weights; Qwen3-MoE says so in its config
        cfg.expert_norm = bool(g("expert_weights_norm", True)) if arch != "qwen3moe" else True
    if g("attention.scale"):                                  # Granite family multipliers
        cfg.attn_scale = float(g("attention.scale"))
    if g("embedding_scale"):
        cfg.embed_scale = float(g("embedding_scale"))
    if g("residual_scale"):
        cfg.residual_scale = float(g("residual_scale"))
    if g("logit_scale"):
        cfg.logit_scale = float(g("logit_scale"))
    if arch == "gemma3":
        n_layer = cfg.n_layer
        cfg.gelu = cfg.sandwich = True
        cfg.embed_scale = n_embd ** 0.5
        cfg.window, cfg.window_pattern = int(g("attention.sliding_window", 0)), 6
        cfg.rope_base_local = float(g("rope.freq_base_swa", 10000.0))
        if scaling == "linear":
            cfg.rope_scale = 1.0 / float(g("rope.scaling.factor", 1.0))
        # llama.cpp: the 27B scales queries by 1/sqrt(n_embd / n_head), the others by 1/sqrt(head_dim)
        cfg.attn_scale = (n_embd / n_head) ** -0.5 if n_layer == 62 else 0.0
    if "rope_freqs.weight" in names:
        cfg.rope_freqs = T.load_dense(gf, names["rope_freqs.weight"])
    elif longrope:
        cfg.rope_freqs = T.load_dense(gf, names["rope_factors_short.weight"])
    tensors = {}
    emb = names["token_embd.weight"]
    if emb.type_name in ("F32", "F16", "BF16"):
        tensors["token_embd.weight"] = Embedding(mx.array(T.load_dense(gf, emb)).astype(mx.float16))
    else:
        w, s, b, bits = T.load_matrix(gf, emb)
        tensors["token_embd.weight"] = Embedding(mx.array(w), mx.array(s), mx.array(b), bits)
    tensors["output_norm.weight"] = mx.array(T.load_dense(gf, names["output_norm.weight"]))
    if "output.weight" in names:
        tensors["output.weight"] = _linear(gf, names["output.weight"], T)
    for i in range(cfg.n_layer):
        for n in ("attn_norm", "ffn_norm"):
            tensors[f"blk.{i}.{n}.weight"] = mx.array(T.load_dense(gf, names[f"blk.{i}.{n}.weight"]))
        for n in ("attn_q_norm", "attn_k_norm", "post_attention_norm", "post_ffw_norm"):
            if f"blk.{i}.{n}.weight" in names:
                tensors[f"blk.{i}.{n}.weight"] = mx.array(T.load_dense(gf, names[f"blk.{i}.{n}.weight"]))
        moe = f"blk.{i}.ffn_gate_inp.weight" in names
        for n, bn in (("attn_q", "attn_q.bias"), ("attn_k", "attn_k.bias"), ("attn_v", "attn_v.bias"),
                      ("attn_output", None), ("ffn_gate", None), ("ffn_up", None), ("ffn_down", None)):
            if moe and n.startswith("ffn_"):
                continue
            bias = names.get(f"blk.{i}.{bn}") if bn else None
            tensors[f"blk.{i}.{n}.weight"] = _linear(gf, names[f"blk.{i}.{n}.weight"], T, bias)
        if moe:
            tensors[f"blk.{i}.ffn_gate_inp.weight"] = QLinear(
                mx.array(T.load_dense(gf, names[f"blk.{i}.ffn_gate_inp.weight"])).astype(mx.float32))
            for n in ("gate", "up", "down"):
                tensors[f"blk.{i}.ffn_{n}_exps.weight"] = _experts(gf, names[f"blk.{i}.ffn_{n}_exps.weight"], T)
        if (i + 1) % 8 == 0:
            mx.eval([a for v in tensors.values() for a in (v if isinstance(v, list) else [v])
                     if isinstance(a, mx.array)])
    model = DenseModel(cfg, tensors)
    return model, cfg, gf
