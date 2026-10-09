# Qwen3.5 hybrid (GatedDeltaNet + gated full attention) for MLX, with the
# JANG / Prism Hadamard-rotated affine storage contract.
#
# Derived from mlx-lm (Apple, MIT), mlx-vlm (MIT) and vMLX (Jinho Jang,
# Apache-2.0) — see engine/NOTICE. Rewritten for a single model family and
# for decode speed:
#   * every Hadamard-rotated input is transformed ONCE per distinct input and
#     the projections that share it run as one fused quantized matmul
#     (q|k|v, gate|up, qkv|z) — the bundle stores one sign vector per width;
#   * text-only positions use mx.fast.rope; 3-D M-RoPE only for image prompts;
#   * prefill computes logits for the last position only.
import math

import mlx.core as mx
import mlx.nn as nn

F16, F32 = mx.float16, mx.float32
PAD_MIN = 6  # see Ops.qmm


# ───────────────────────────── primitives ─────────────────────────────
class QLinear(nn.Module):
    """Affine-quantized linear layer with packed uint32 weights."""

    def __init__(self, in_dims, out_dims, bits, group_size, bias=False):
        super().__init__()
        self.bits, self.group_size = bits, group_size
        # Lazy placeholders with the checkpoint's shapes; replaced by load_weights.
        self.weight = mx.zeros((out_dims, in_dims * bits // 32), dtype=mx.uint32)
        self.scales = mx.zeros((out_dims, in_dims // group_size), dtype=F16)
        self.biases = mx.zeros((out_dims, in_dims // group_size), dtype=F16)
        if bias:
            self.bias = mx.zeros((out_dims,), dtype=F16)

    def __call__(self, x):
        y = mx.quantized_matmul(x, self.weight, self.scales, self.biases,
                                transpose=True, group_size=self.group_size, bits=self.bits)
        if "bias" in self:
            y = y + self.bias
        return y


class Hadamard:
    """Blockwise normalized Walsh-Hadamard on activations (prism.hadamard.v1).

    forward:  H(signs * x)    inverse: signs * H(x)
    Computed in float32 like the reference runtime, returned as float16.
    """

    def __init__(self, block, signs):
        self.block = block
        self.scale = 1.0 / math.sqrt(block)
        self.signs = signs  # {width: float32 (+-1) vector}

    def fwd(self, x):
        shape = x.shape
        y = x.astype(F32) * self.signs[shape[-1]]
        y = mx.hadamard_transform(y.reshape(-1, self.block), scale=self.scale)
        return y.reshape(shape).astype(F16)

    def inv(self, x):
        shape = x.shape
        y = mx.hadamard_transform(x.astype(F32).reshape(-1, self.block), scale=self.scale)
        return (y.reshape(shape) * self.signs[shape[-1]]).astype(F16)


class Identity:
    def fwd(self, x): return x
    def inv(self, x): return x


class RMSNorm(nn.Module):
    def __init__(self, dims, eps):
        super().__init__()
        self.eps = eps
        self.weight = mx.ones((dims,))

    def __call__(self, x):
        return mx.fast.rms_norm(x, self.weight, self.eps).astype(F16)


def swiglu(gate, x):
    return nn.silu(gate) * x


# ───────────────────────────── cache ─────────────────────────────
class KVCache:
    step = 256

    def __init__(self):
        self.keys = self.values = None
        self.offset = 0

    def update(self, k, v):
        prev = self.offset
        n = k.shape[2]
        if self.keys is None or prev + n > self.keys.shape[2]:
            B, H, _, D = k.shape
            grow = ((n + self.step - 1) // self.step) * self.step
            nk = mx.zeros((B, H, grow, D), k.dtype)
            nv = mx.zeros((B, H, grow, v.shape[3]), v.dtype)
            if self.keys is not None:
                self.keys = mx.concatenate([self.keys[..., :prev, :], nk], axis=2)
                self.values = mx.concatenate([self.values[..., :prev, :], nv], axis=2)
            else:
                self.keys, self.values = nk, nv
        self.offset += n
        self.keys[..., prev:self.offset, :] = k
        self.values[..., prev:self.offset, :] = v
        return self.keys[..., :self.offset, :], self.values[..., :self.offset, :]

    def trim_to(self, n):
        self.offset = min(self.offset, n)

    def begin_verify(self):
        self.base = self.offset

    def commit(self, m):
        self.offset = self.base + m

    @property
    def state(self):
        return [] if self.keys is None else [self.keys, self.values]

    @property
    def nbytes(self):
        return 0 if self.keys is None else self.keys.nbytes + self.values.nbytes


class SSMCache:
    """Conv tail + recurrent state of one GatedDeltaNet layer.

    Speculative verify: begin_verify() remembers the pre-verify state; the
    layer stashes the recurrence inputs; commit(m) keeps the first m tokens by
    re-running only the (cheap) recurrence on them — no second weight pass.
    """

    def __init__(self):
        self.conv = None
        self.state = None
        self.offset = 0
        self.verifying = False
        self.pending = None

    def begin_verify(self):
        self.verifying = True
        self.before = (self.state, self.offset)
        self.pending = None

    def commit(self, m):
        from .gated_delta import gated_delta_kernel
        self.verifying = False
        if self.pending is None:
            return
        q, k, v, g, beta, conv_in, kernel = self.pending
        self.pending = None
        S = q.shape[1]
        if m < S:
            state0, off0 = self.before
            _, self.state = gated_delta_kernel(q[:, :m], k[:, :m], v[:, :m], g[:, :m], beta[:, :m], state0)
            self.conv = mx.contiguous(conv_in[:, m:m + kernel - 1])
            self.offset = off0 + m

    def snapshot(self):
        return (self.conv, self.state, self.offset)

    def restore(self, snap):
        self.conv, self.state, self.offset = snap

    @property
    def arrays(self):
        return [a for a in (self.conv, self.state) if a is not None]


# ───────────────────────────── rope ─────────────────────────────
class MRoPE:
    """Partial interleaved multimodal RoPE (Qwen3.5)."""

    def __init__(self, dims, base, mrope_section):
        self.dims, self.base = dims, base
        self.inv_freq = 1.0 / (base ** (mx.arange(0, dims, 2, dtype=F32) / dims))
        # For each of the dims/2 frequencies: which position axis (t/h/w) drives it.
        axis = [0] * (dims // 2)
        for a, off in ((1, 1), (2, 2)):
            for i in range(off, mrope_section[a] * 3, 3):
                axis[i] = a
        self.axis = mx.array(axis)

    def cos_sin(self, position_ids):
        # position_ids: (3, L) int -> cos, sin: (L, dims)
        pos = position_ids.astype(F32)                                   # (3, L)
        freqs = self._select(pos[:, :, None] * self.inv_freq[None, None])  # (L, dims/2)
        emb = mx.concatenate([freqs, freqs], axis=-1)
        return mx.cos(emb), mx.sin(emb)

    def _select(self, freqs):
        # freqs: (3, L, F) -> (L, F) choosing axis per frequency
        onehot = (mx.arange(3)[:, None] == self.axis[None, :]).astype(F32)  # (3, F)
        return (freqs * onehot[:, None, :]).sum(axis=0)

    def apply(self, x, cos, sin):
        # x: (B, H, L, D); rotate the first `dims` channels (rotate_half style)
        d = self.dims
        xr, xp = x[..., :d], x[..., d:]
        h = d // 2
        rot = mx.concatenate([-xr[..., h:], xr[..., :h]], axis=-1)
        y = xr.astype(F32) * cos + rot.astype(F32) * sin
        return mx.concatenate([y.astype(x.dtype), xp], axis=-1)


# ───────────────────────────── fused ops ─────────────────────────────
class Ops:
    """mx.compile'd building blocks shared by all layers.

    Decode is latency-bound by thousands of small dependent kernels per token,
    not by weight bandwidth. Each block below fuses its elementwise chain
    (norm casts, sign flips, gates, swiglu, residual adds, depthwise conv) into
    a few kernels. Weights are passed as arguments, so every layer reuses the
    same compiled graph.
    """

    def __init__(self, a, had, bits, gs):
        eps = a["rms_norm_eps"]
        nk, nv = a["linear_num_key_heads"], a["linear_num_value_heads"]
        dk, dv = a["linear_key_head_dim"], a["linear_value_head_dim"]
        key_dim, conv_dim, K = dk * nk, 2 * dk * nk + dv * nv, a["linear_conv_kernel_dim"]
        nh, nkv, hd = a["num_attention_heads"], a["num_key_value_heads"], a["head_dim"]
        inv = dk ** -0.5
        self.had = had

        def qmm(x, w, s, b):
            # MLX's small-M kernel re-streams the weights per row (~linear in M),
            # its tiled kernel kicks in at 16 rows and costs about the same for
            # 16..32 rows. Pad 6..15-row inputs (speculative verify) up to 16.
            rows = x.size // x.shape[-1]
            if PAD_MIN <= rows < 16:
                flat = x.reshape(rows, x.shape[-1])
                flat = mx.concatenate([flat, mx.zeros((16 - rows, x.shape[-1]), x.dtype)])
                y = mx.quantized_matmul(flat, w, s, b, transpose=True, group_size=gs, bits=bits)
                return y[:rows].reshape(*x.shape[:-1], -1)
            return mx.quantized_matmul(x, w, s, b, transpose=True, group_size=gs, bits=bits)

        def rms(x, w):
            return mx.fast.rms_norm(x, w, eps).astype(F16)

        def norm_had(x, w):
            n = rms(x, w)
            return n, had.fwd(n)

        def mlp(h, nw, gw, gsc, gb, dw, dsc, db):
            gu = qmm(had.fwd(rms(h, nw)), gw, gsc, gb)
            i = gu.shape[-1] // 2
            return h + qmm(had.fwd(swiglu(gu[..., :i], gu[..., i:])), dw, dsc, db)

        def gdn_pre(n, nr, conv_state, w, s, b, ba, convw, A_log, dt_bias):
            B, S, _ = n.shape
            qkvz = qmm(nr, w, s, b)
            qkv, z = qkvz[..., :conv_dim], qkvz[..., conv_dim:]
            beta = mx.sigmoid(ba[..., :nv])
            g = mx.exp(-mx.exp(A_log.astype(F32)) * nn.softplus(ba[..., nv:] + dt_bias))
            conv_in = mx.concatenate([conv_state, qkv], axis=1)
            # depthwise causal conv as a sum of K shifted products. mx.conv1d with
            # groups=channels materializes ~1 GB of temporaries per layer call.
            w = convw[:, :, 0].T  # (K, C)
            if S == 1:
                co = (conv_in * w).sum(axis=1, keepdims=True)
            else:
                co = conv_in[:, 0:S] * w[0]
                for j in range(1, K):
                    co = co + conv_in[:, j:j + S] * w[j]
            co = nn.silu(co).astype(F16)
            q = co[..., :key_dim].reshape(B, S, nk, dk)
            k = co[..., key_dim:2 * key_dim].reshape(B, S, nk, dk)
            v = co[..., 2 * key_dim:].reshape(B, S, nv, dv)
            q = (inv * inv) * mx.fast.rms_norm(q, None, 1e-6)
            k = inv * mx.fast.rms_norm(k, None, 1e-6)
            return q, k, v, g, beta, z, conv_in

        def gdn_post(out, z, normw, x, w, s, b):
            B, S = x.shape[:2]
            o = swiglu(z.reshape(B, S, nv, dv), rms(out, normw)).astype(F16)
            return x + qmm(had.fwd(o.reshape(B, S, -1)), w, s, b)

        def attn_pre(nr, w, s, b, qnw, knw):
            B, L, _ = nr.shape
            qkv = qmm(nr, w, s, b)
            qd, kvd = nh * hd * 2, nkv * hd
            q = qkv[..., :qd].reshape(B, L, nh, 2 * hd)
            gate = q[..., hd:].reshape(B, L, -1)
            q = rms(q[..., :hd], qnw).transpose(0, 2, 1, 3)
            k = rms(qkv[..., qd:qd + kvd].reshape(B, L, nkv, hd), knw).transpose(0, 2, 1, 3)
            v = qkv[..., qd + kvd:].reshape(B, L, nkv, hd).transpose(0, 2, 1, 3)
            return q, k, v, gate

        def attn_post(o, gate, x, w, s, b):
            B, _, L, _ = o.shape
            o = o.transpose(0, 2, 1, 3).reshape(B, L, -1)
            return x + qmm(had.fwd(o * mx.sigmoid(gate)), w, s, b)

        def head(h, nw, w, s, b):
            return qmm(had.fwd(rms(h, nw)), w, s, b)

        for name, fn in list(locals().items()):
            if name in ("norm_had", "mlp", "gdn_pre", "gdn_post", "attn_pre", "attn_post", "head"):
                setattr(self, name, mx.compile(fn))


def _w(m):
    return m.weight, m.scales, m.biases


# ───────────────────────────── layers ─────────────────────────────
class Attention(nn.Module):
    def __init__(self, a, q):
        super().__init__()
        self.nh, self.nkv, self.hd = a["num_attention_heads"], a["num_key_value_heads"], a["head_dim"]
        self.scale = self.hd ** -0.5
        hid = a["hidden_size"]
        self.qkv_proj = q(hid, self.nh * self.hd * 2 + 2 * self.nkv * self.hd)
        self.o_proj = q(self.nh * self.hd, hid)
        self.q_norm = RMSNorm(self.hd, a["rms_norm_eps"])
        self.k_norm = RMSNorm(self.hd, a["rms_norm_eps"])
        rp = a["rope_parameters"]
        self.rope_dims = int(self.hd * rp.get("partial_rotary_factor", 0.25))
        self.rope_base = float(rp.get("rope_theta", 1e7))
        self.mrope = MRoPE(self.rope_dims, self.rope_base, rp.get("mrope_section", [11, 11, 10]))

    def __call__(self, x, nr, ops, cache, pos):
        L = nr.shape[1]
        q, k, v, gate = ops.attn_pre(nr, *_w(self.qkv_proj), self.q_norm.weight, self.k_norm.weight)
        if pos.cos is not None:  # image prompt: explicit 3-D positions
            q = self.mrope.apply(q, pos.cos, pos.sin)
            k = self.mrope.apply(k, pos.cos, pos.sin)
        else:
            off = cache.offset + pos.delta
            q = mx.fast.rope(q, self.rope_dims, traditional=False, base=self.rope_base, scale=1.0, offset=off)
            k = mx.fast.rope(k, self.rope_dims, traditional=False, base=self.rope_base, scale=1.0, offset=off)
        k, v = cache.update(k, v)
        o = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask="causal" if L > 1 else None)
        return ops.attn_post(o, gate, x, *_w(self.o_proj))


class GatedDeltaNet(nn.Module):
    def __init__(self, a, q, ba_quantized=False):
        super().__init__()
        hid = a["hidden_size"]
        self.nv = a["linear_num_value_heads"]
        dk, dv, nk = a["linear_key_head_dim"], a["linear_value_head_dim"], a["linear_num_key_heads"]
        self.conv_dim = 2 * dk * nk + dv * self.nv
        self.kernel = a["linear_conv_kernel_dim"]
        self.in_proj_qkvz = q(hid, self.conv_dim + dv * self.nv)
        # b/a gate projections: float32 in JANG bundles, quantized in mlx-community conversions
        self.in_proj_ba = q(hid, 2 * self.nv) if ba_quantized else nn.Linear(hid, 2 * self.nv, bias=False)
        self.conv1d = nn.Conv1d(self.conv_dim, self.conv_dim, self.kernel, groups=self.conv_dim, bias=False)
        self.dt_bias = mx.ones((self.nv,))
        self.A_log = mx.zeros((self.nv,))
        self.norm = RMSNorm(dv, a["rms_norm_eps"])
        self.out_proj = q(dv * self.nv, hid)

    def __call__(self, x, n, nr, ops, cache):
        from .gated_delta import gated_delta_kernel
        B = x.shape[0]
        conv_state = cache.conv if cache.conv is not None else \
            mx.zeros((B, self.kernel - 1, self.conv_dim), dtype=F16)
        q, k, v, g, beta, z, conv_in = ops.gdn_pre(
            n, nr, conv_state, *_w(self.in_proj_qkvz), self.in_proj_ba(n),
            self.conv1d.weight, self.A_log, self.dt_bias)
        cache.conv = mx.contiguous(conv_in[:, -(self.kernel - 1):])
        if cache.verifying and x.shape[1] > 1:
            cache.pending = (q, k, v, g, beta, conv_in, self.kernel)
        out, cache.state = gated_delta_kernel(q, k, v, g, beta, cache.state)
        cache.offset += x.shape[1]
        return ops.gdn_post(out, z, self.norm.weight, x, *_w(self.out_proj))


class MLP(nn.Module):
    def __init__(self, a, q):
        super().__init__()
        self.gate_up_proj = q(a["hidden_size"], 2 * a["intermediate_size"])
        self.down_proj = q(a["intermediate_size"], a["hidden_size"])


class DecoderLayer(nn.Module):
    def __init__(self, a, q, idx, ba_quantized=False):
        super().__init__()
        self.is_linear = (idx + 1) % a["full_attention_interval"] != 0
        if self.is_linear:
            self.linear_attn = GatedDeltaNet(a, q, ba_quantized)
        else:
            self.self_attn = Attention(a, q)
        self.input_layernorm = RMSNorm(a["hidden_size"], a["rms_norm_eps"])
        self.post_attention_layernorm = RMSNorm(a["hidden_size"], a["rms_norm_eps"])
        self.mlp = MLP(a, q)

    def __call__(self, x, ops, cache, pos):
        n, nr = ops.norm_had(x, self.input_layernorm.weight)
        if self.is_linear:
            h = self.linear_attn(x, n, nr, ops, cache)
        else:
            h = self.self_attn(x, nr, ops, cache, pos)
        return ops.mlp(h, self.post_attention_layernorm.weight,
                       *_w(self.mlp.gate_up_proj), *_w(self.mlp.down_proj))


class Positions:
    """cos/sin for 3-D image positions, or a scalar delta for 1-D text rope."""

    def __init__(self, cos=None, sin=None, delta=0):
        self.cos, self.sin, self.delta = cos, sin, delta


class QEmbedding(nn.Module):
    def __init__(self, num, dims, bits, group_size):
        super().__init__()
        self.bits, self.group_size = bits, group_size
        self.weight = mx.zeros((num, dims * bits // 32), dtype=mx.uint32)
        self.scales = mx.zeros((num, dims // group_size), dtype=F16)
        self.biases = mx.zeros((num, dims // group_size), dtype=F16)

    def __call__(self, ids):
        return mx.dequantize(self.weight[ids], self.scales[ids], self.biases[ids],
                             group_size=self.group_size, bits=self.bits).astype(F16)


class TextModel(nn.Module):
    def __init__(self, a, q, qe):
        super().__init__()
        self.embed_tokens = qe(a["vocab_size"], a["hidden_size"])
        self.layers = [DecoderLayer(a, q, i, a.get("ba_quantized", False)) for i in range(a["num_hidden_layers"])]
        self.norm = RMSNorm(a["hidden_size"], a["rms_norm_eps"])


class LanguageModel(nn.Module):
    def __init__(self, a, q, qe):
        super().__init__()
        self.model = TextModel(a, q, qe)
        if not a.get("tie_word_embeddings", False):
            self.lm_head = q(a["hidden_size"], a["vocab_size"])


class Model(nn.Module):
    """Language model (+ optional vision tower) with Hadamard-aware forward."""

    def __init__(self, config, qcfg, vision=None, keys=()):
        super().__init__()
        self.args = a = dict(config["text_config"])
        a["ba_quantized"] = any(k.endswith("linear_attn.in_proj_b.scales") for k in keys)
        a.setdefault("head_dim", a["hidden_size"] // a["num_attention_heads"])
        self.config = config
        bits, gs = qcfg["bits"], qcfg["group_size"]
        q = lambda i, o: QLinear(i, o, bits, gs)
        qe = lambda n, d: QEmbedding(n, d, bits, gs)
        self.language_model = LanguageModel(a, q, qe)
        if vision is not None:
            self.vision_tower = vision
        self.qbits, self.qgs = bits, gs
        self.set_hadamard(Identity())

    def set_hadamard(self, had):
        self.had = had
        self.ops = Ops(self.args, had, self.qbits, self.qgs)

    @property
    def layers(self):
        return self.language_model.model.layers

    def make_cache(self):
        return [SSMCache() if l.is_linear else KVCache() for l in self.layers]

    def embed(self, ids):
        return self.had.inv(self.language_model.model.embed_tokens(ids))

    def commit(self, cache, m):
        """After a verify pass: keep the first m fed tokens in every cache."""
        for c in cache:
            c.commit(m)

    def __call__(self, ids=None, cache=None, pos=None, embeds=None, last_only=True, verify=False):
        lm = self.language_model
        if verify:
            for c in cache:
                c.begin_verify()
        h = self.embed(ids) if embeds is None else embeds
        pos = pos or Positions()
        # Prefill: materialize each layer's output and cache state right away. A lazy
        # cache state keeps all of its inputs (q/k/v, gates, conv window for every
        # token of the chunk) alive until the end — ~2 GB for 1024 tokens × 48 layers.
        step_eval = h.shape[1] > 64
        for layer, c in zip(lm.model.layers, cache):
            h = layer(h, self.ops, c, pos)
            if step_eval:
                mx.eval(h, c.arrays if hasattr(c, "arrays") else c.state)
        if last_only:
            h = h[:, -1:, :]
        # tied embeddings: the quantized embedding matrix doubles as the output head
        head = lm.lm_head if "lm_head" in lm else lm.model.embed_tokens
        return self.ops.head(h, lm.model.norm.weight, *_w(head))
