# Qwen3-VL / Qwen3.5 vision tower + image preprocessing.
# Derived from mlx-vlm (MIT) and the Qwen2-VL image processor (Apache-2.0) —
# see engine/NOTICE. Images only (no video), no torch / transformers.
import base64
import io
import math
import urllib.request
from itertools import accumulate

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .model import F16, QLinear
from .quant import module_spec


def make_linear(keys, qcfg, prefix, i, o, bias=True):
    """QLinear if the checkpoint stores this module quantized, else nn.Linear."""
    if f"{prefix}.scales" in keys:
        if isinstance(keys, dict):  # {name: (shape, dtype)}: read the module's own quantization
            spec = module_spec(keys, prefix, i, qcfg.get("mode") if qcfg else None)
            return QLinear(i, o, spec[0], spec[1], bias=bias, mode=spec[2])
        m = qcfg.get(prefix, qcfg)
        return QLinear(i, o, m["bits"], m["group_size"], bias=bias)
    return nn.Linear(i, o, bias=bias)


def _rot_half(x):
    h = x.shape[-1] // 2
    return mx.concatenate([-x[..., h:], x[..., :h]], axis=-1)


class PatchEmbed(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.p, self.tp, self.cin, self.hid = c["patch_size"], c["temporal_patch_size"], c["in_channels"], c["hidden_size"]
        k = [self.tp, self.p, self.p]
        self.proj = nn.Conv3d(self.cin, self.hid, kernel_size=k, stride=k, bias=True)

    def __call__(self, x):
        x = x.reshape(-1, self.cin, self.tp, self.p, self.p).moveaxis(1, 4)
        return self.proj(x).reshape(-1, self.hid)


class Attention(nn.Module):
    def __init__(self, keys, qcfg, prefix, dim, heads):
        super().__init__()
        self.heads, self.scale = heads, (dim // heads) ** -0.5
        self.qkv = make_linear(keys, qcfg, f"{prefix}.qkv", dim, 3 * dim)
        self.proj = make_linear(keys, qcfg, f"{prefix}.proj", dim, dim)

    def __call__(self, x, cu, cos, sin):
        L = x.shape[0]
        q, k, v = mx.split(self.qkv(x).reshape(L, 3, self.heads, -1).transpose(1, 0, 2, 3), 3)
        q, k, v = q[0], k[0], v[0]                       # (L, H, D)
        q = (q * cos + _rot_half(q) * sin).astype(x.dtype)
        k = (k * cos + _rot_half(k) * sin).astype(x.dtype)
        q, k, v = (t.transpose(1, 0, 2)[None] for t in (q, k, v))  # (1, H, L, D)
        outs = []
        for s, e in zip(cu[:-1], cu[1:]):
            outs.append(mx.fast.scaled_dot_product_attention(
                q[:, :, s:e], k[:, :, s:e], v[:, :, s:e], scale=self.scale))
        o = mx.concatenate(outs, axis=2)[0].transpose(1, 0, 2).reshape(L, -1)
        return self.proj(o)


class MLP(nn.Module):
    def __init__(self, keys, qcfg, prefix, dim, hid):
        super().__init__()
        self.linear_fc1 = make_linear(keys, qcfg, f"{prefix}.linear_fc1", dim, hid)
        self.linear_fc2 = make_linear(keys, qcfg, f"{prefix}.linear_fc2", hid, dim)

    def __call__(self, x):
        return self.linear_fc2(nn.gelu_approx(self.linear_fc1(x)))


class Block(nn.Module):
    def __init__(self, keys, qcfg, prefix, c):
        super().__init__()
        d = c["hidden_size"]
        self.norm1 = nn.LayerNorm(d, eps=1e-6)
        self.norm2 = nn.LayerNorm(d, eps=1e-6)
        self.attn = Attention(keys, qcfg, f"{prefix}.attn", d, c["num_heads"])
        self.mlp = MLP(keys, qcfg, f"{prefix}.mlp", d, c["intermediate_size"])

    def __call__(self, x, cu, cos, sin):
        x = x + self.attn(self.norm1(x), cu, cos, sin)
        return x + self.mlp(self.norm2(x))


class Merger(nn.Module):
    def __init__(self, keys, qcfg, prefix, c):
        super().__init__()
        self.hid = c["hidden_size"] * c["spatial_merge_size"] ** 2
        self.norm = nn.LayerNorm(c["hidden_size"], eps=1e-6)
        self.linear_fc1 = make_linear(keys, qcfg, f"{prefix}.linear_fc1", self.hid, self.hid)
        self.linear_fc2 = make_linear(keys, qcfg, f"{prefix}.linear_fc2", self.hid, c["out_hidden_size"])

    def __call__(self, x):
        x = self.norm(x).reshape(-1, self.hid)
        return self.linear_fc2(nn.gelu(self.linear_fc1(x)))


class VisionModel(nn.Module):
    def __init__(self, c, keys, qcfg):
        super().__init__()
        if c.get("deepstack_visual_indexes"):
            raise NotImplementedError("deepstack vision features are not supported")
        self.c = c
        self.merge = c["spatial_merge_size"]
        self.patch_embed = PatchEmbed(c)
        self.pos_embed = nn.Embedding(c["num_position_embeddings"], c["hidden_size"])
        self.side = int(c["num_position_embeddings"] ** 0.5)
        hd = c["hidden_size"] // c["num_heads"]
        self.rot_dim = hd // 2
        self.blocks = [Block(keys, qcfg, f"vision_tower.blocks.{i}", c) for i in range(c["depth"])]
        self.merger = Merger(keys, qcfg, "vision_tower.merger", c)

    # rotary 2-D positions inside each merge window
    def _rot(self, grids):
        inv = 1.0 / (10000.0 ** (mx.arange(0, self.rot_dim, 2, dtype=mx.float32) / self.rot_dim))
        m = self.merge
        rows, cols = [], []
        for t, h, w in grids:
            r = np.broadcast_to(np.arange(h).reshape(h // m, 1, m, 1), (h // m, w // m, m, m))
            cc = np.arange(w).reshape(1, w // m, 1, m)
            cc = np.broadcast_to(cc, (h // m, w // m, m, m))
            rows.append(np.tile(r.reshape(-1), t))
            cols.append(np.tile(cc.reshape(-1), t))
        rows = mx.array(np.concatenate(rows))
        cols = mx.array(np.concatenate(cols))
        f = mx.concatenate([rows[:, None] * inv[None], cols[:, None] * inv[None]], axis=-1)
        emb = mx.concatenate([f, f], axis=-1)[:, None, :]  # (L, 1, D)
        return mx.cos(emb), mx.sin(emb)

    # bilinear interpolation of the learned 2-D position table
    def _pos(self, grids):
        n, m, out = self.side, self.merge, []
        for t, h, w in grids:
            hi, wi = np.linspace(0, n - 1, h), np.linspace(0, n - 1, w)
            hf, wf = hi.astype(np.int32), wi.astype(np.int32)
            hc, wc = np.minimum(hf + 1, n - 1), np.minimum(wf + 1, n - 1)
            dh, dw = (hi - hf)[:, None], (wi - wf)[None, :]
            idx = [hf[:, None] * n + wf[None], hf[:, None] * n + wc[None],
                   hc[:, None] * n + wf[None], hc[:, None] * n + wc[None]]
            wts = [(1 - dh) * (1 - dw), (1 - dh) * dw, dh * (1 - dw), dh * dw]
            pe = sum(self.pos_embed(mx.array(i.reshape(-1))) * mx.array(wt.reshape(-1, 1), dtype=F16)
                     for i, wt in zip(idx, wts))
            d = pe.shape[-1]
            pe = mx.tile(pe, (t, 1)).reshape(t, h // m, m, w // m, m, d).transpose(0, 1, 3, 2, 4, 5)
            out.append(pe.reshape(-1, d))
        return mx.concatenate(out)

    def __call__(self, pixels, grids):
        x = self.patch_embed(pixels.astype(F16))
        x = x + self._pos(grids)
        cos, sin = self._rot(grids)
        cu = [0] + list(accumulate(h * w for t, h, w in grids for _ in range(t)))
        for blk in self.blocks:
            x = blk(x, cu, cos, sin)
        return self.merger(x)


# ───────────────────────────── preprocessing ─────────────────────────────
def load_image(src):
    from PIL import Image
    if isinstance(src, dict):
        src = src.get("url") or src.get("image_url") or src.get("image")
        if isinstance(src, dict):
            src = src.get("url")
    if not isinstance(src, str):
        raise ValueError("unsupported image reference")
    if src.startswith("data:"):
        data = base64.b64decode(src.split(",", 1)[1])
    elif src.startswith(("http://", "https://")):
        with urllib.request.urlopen(src, timeout=30) as r:
            data = r.read()
    else:
        with open(src.removeprefix("file://"), "rb") as f:
            data = f.read()
    return Image.open(io.BytesIO(data)).convert("RGB")


def smart_resize(h, w, factor, min_px, max_px):
    hb, wb = max(factor, round(h / factor) * factor), max(factor, round(w / factor) * factor)
    if hb * wb > max_px:
        beta = math.sqrt(h * w / max_px)
        hb, wb = max(factor, math.floor(h / beta / factor) * factor), max(factor, math.floor(w / beta / factor) * factor)
    elif hb * wb < min_px:
        beta = math.sqrt(min_px / (h * w))
        hb, wb = math.ceil(h * beta / factor) * factor, math.ceil(w * beta / factor) * factor
    return hb, wb


def preprocess(img, pp, max_pixels):
    """PIL image -> (patches (N, C*T*p*p) float16, (t, gh, gw))."""
    from PIL import Image
    p, tp, m = pp["patch_size"], pp["temporal_patch_size"], pp["merge_size"]
    size = pp.get("size", {})
    min_px = size.get("shortest_edge", 56 * 56)
    max_px = min(size.get("longest_edge", max_pixels), max_pixels)
    h, w = smart_resize(img.height, img.width, p * m, min_px, max_px)
    img = img.resize((w, h), Image.BICUBIC)
    mean = np.array(pp.get("image_mean", [0.5] * 3), np.float32)
    std = np.array(pp.get("image_std", [0.5] * 3), np.float32)
    x = ((np.asarray(img, np.float32) / 255.0 - mean) / std).transpose(2, 0, 1)  # (C, H, W)
    x = np.repeat(x[None], tp, axis=0)                                             # (T, C, H, W)
    c, gh, gw = x.shape[1], h // p, w // p
    x = x.reshape(1, tp, c, gh // m, m, p, gw // m, m, p)
    x = x.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8).reshape(gh * gw, c * tp * p * p)
    return mx.array(x, dtype=F16), (1, gh, gw)
