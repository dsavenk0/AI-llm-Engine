"""The MLX runtime of the native model: everything the decode loop (engine/session.py) asks for, computed
with MLX on engine/model.py.

This is the Runtime seam of architecture/BACKEND_INTERFACE_SPEC.md. The loop above it never touches an
array; a second runtime — own Metal kernels behind the same methods, a native library — implements this
class's methods and the loop does not change. Sampling lives here because it runs on the device: pulling
248 320 logits to the CPU each token would cost more than the sampler itself. The *policy* it applies
(temperature, top-k/p, penalties, the format constraint) is passed in by the loop.
"""
import mlx.core as mx
import numpy as np

from .model import KVCache, Positions, SSMCache
from .session import FormatError


class Sampler:
    def __init__(self, vocab, temperature=1.0, top_p=1.0, top_k=0, min_p=0.0, presence_penalty=0.0, seed=None,
                 repetition_penalty=1.0):
        self.temp, self.top_p, self.top_k, self.min_p = temperature, top_p, top_k, min_p
        self.pp = presence_penalty
        self.rp = repetition_penalty or 1.0      # >1 discourages every token already generated (HF/llama.cpp rule)
        self.vocab = vocab
        self.seen = mx.zeros((vocab,), dtype=mx.bool_) if (presence_penalty or self.rp != 1.0) else None
        self.key = mx.random.key(seed) if seed is not None else None
        self.allow = None         # set for constrained decoding: allow(token_id) -> bool (see structured.py)
        self.on_pick = None       # called with the token actually chosen, so the constraint can advance

    @property
    def greedy(self):
        return self.temp <= 0 and self.seen is None

    def _rand(self):
        if self.key is None:
            return None
        self.key, sub = mx.random.split(self.key)
        return sub

    def _dist(self, row):
        """Candidate token ids and their log-weights after penalty/temp/top-k/min-p/top-p."""
        row = self._penalize(row.astype(mx.float32))
        if self.top_k and self.top_k < self.vocab:
            idx = mx.argpartition(-row, self.top_k - 1)[:self.top_k]
        else:
            idx = mx.arange(self.vocab)
        vals = row[idx] / self.temp
        order = mx.argsort(-vals)
        idx, vals = idx[order], vals[order]
        if self.min_p > 0:
            vals = mx.where(vals < vals[0] + float(np.log(self.min_p)), -mx.inf, vals)
        if self.top_p < 1.0:
            probs = mx.softmax(vals)
            vals = mx.where(mx.cumsum(probs) - probs > self.top_p, -mx.inf, vals)
        return idx, vals

    def _penalize(self, r):
        """Presence penalty (subtract) and repetition penalty (divide a positive logit, multiply a negative
        one) for the tokens generated so far."""
        if self.seen is None:
            return r
        if self.pp:
            r = r - self.pp * self.seen.astype(mx.float32)
        if self.rp != 1.0:
            r = mx.where(self.seen, mx.where(r > 0, r / self.rp, r * self.rp), r)
        return r

    def _mark(self, tok):
        if self.seen is not None:
            self.seen = mx.logical_or(self.seen, mx.arange(self.vocab) == tok)

    def __call__(self, logits):
        """Sample from the last position of [1, L, V] logits; returns a lazy scalar."""
        row = logits[0, -1]
        if self.allow is not None:
            return mx.array(self._pick_allowed(row))
        if self.temp <= 0:
            tok = mx.argmax(self._penalize(row.astype(mx.float32)))
        else:
            idx, vals = self._dist(row)
            tok = idx[mx.random.categorical(vals, key=self._rand())]
        self._mark(tok)
        return tok

    def _pick_allowed(self, row):
        """The token the model ranks highest among those the constraint still permits.

        Candidates are tested in the order the model prefers them and the window is widened only when a
        whole slice is forbidden, so the usual cost is one test per token — the model's first choice is
        normally already valid. Sampling renormalizes over the permitted candidates, which is the same
        restriction top-k and top-p already apply.
        """
        r = self._penalize(row.astype(mx.float32))
        for k in (24, 256, 4096, self.vocab):
            k = min(k, self.vocab)
            if k < self.vocab:
                idx = mx.argpartition(-r, k - 1)[:k]
                vals = r[idx]
                order = mx.argsort(-vals)
                idx, vals = idx[order], vals[order]
            else:
                idx = mx.argsort(-r)
                vals = r[idx]
            ids = idx.tolist()
            if self.temp <= 0:
                for t in ids:
                    if self.allow(t):
                        self._mark(t)
                        if self.on_pick:
                            self.on_pick(t)
                        return t
                continue
            keep = [i for i, t in enumerate(ids) if self.allow(t)]
            if not keep:
                continue
            sel = mx.array(keep)
            kvals = vals[sel] / self.temp
            if self.min_p > 0:
                kvals = mx.where(kvals < kvals[0] + float(np.log(self.min_p)), -mx.inf, kvals)
            if self.top_p < 1.0:
                probs = mx.softmax(kvals)
                kvals = mx.where(mx.cumsum(probs) - probs > self.top_p, -mx.inf, kvals)
            tok = ids[keep[mx.random.categorical(kvals, key=self._rand()).item()]]
            self._mark(tok)
            if self.on_pick:
                self.on_pick(tok)
            return tok
        raise FormatError("no token can continue the requested format")

    def verify(self, row, draft):
        """Speculative step for a deterministic draft token.

        Accept with probability p(draft); otherwise sample from p with the draft
        removed. This keeps the output distribution exactly that of sampling.
        Returns (accepted, token).
        """
        if self.temp <= 0:
            tok = self(row[None, None]).item()
            return tok == draft, tok
        idx, vals = self._dist(row)
        probs = mx.softmax(vals)
        is_d = idx == draft
        p_d = mx.sum(mx.where(is_d, probs, 0.0)).item()
        u = mx.random.uniform(key=self._rand()).item()
        if u < p_d:
            self._mark(mx.array(draft))
            return True, draft
        rest = mx.where(is_d, -mx.inf, vals)
        tok = idx[mx.random.categorical(rest, key=self._rand())]
        self._mark(tok)
        return False, tok.item()


def mrope_positions(ids, grids, image_token, merge):
    """3-D (t, h, w) positions for a prompt containing image placeholder runs."""
    n = len(ids)
    pos = np.zeros((3, n), np.int32)
    cur = i = img = 0
    while i < n:
        if ids[i] == image_token and img < len(grids):
            t, h, w = grids[img]
            img += 1
            gh, gw = h // merge, w // merge
            cnt = t * gh * gw
            pos[0, i:i + cnt] = np.repeat(np.arange(t), gh * gw) + cur
            pos[1, i:i + cnt] = np.tile(np.repeat(np.arange(gh), gw), t) + cur
            pos[2, i:i + cnt] = np.tile(np.arange(gw), t * gh) + cur
            cur = int(pos[:, i:i + cnt].max()) + 1
            i += cnt
        else:
            pos[:, i] = cur
            cur += 1
            i += 1
    return pos, cur - n


NEG = -30000.0          # additive mask value that zeroes a key in float16 softmax


def _assemble(caches, layer_idx, lmax):
    """Combine one layer's per-request caches into a batched one. Attention: left-pad KV to `lmax` and stack.
    SSM: stack the (position-independent) conv tail and recurrent state over the batch."""
    sample = caches[0][layer_idx]
    if isinstance(sample, KVCache):
        ks, vs = [], []
        for c in caches:
            kv = c[layer_idx]
            k, v = kv.keys[..., :kv.offset, :], kv.values[..., :kv.offset, :]
            pad = lmax - k.shape[2]
            if pad:
                kz = mx.zeros((1, k.shape[1], pad, k.shape[3]), k.dtype)
                vz = mx.zeros((1, v.shape[1], pad, v.shape[3]), v.dtype)
                k, v = mx.concatenate([kz, k], axis=2), mx.concatenate([vz, v], axis=2)
            ks.append(k)
            vs.append(v)
        merged = KVCache()
        merged.keys, merged.values = mx.concatenate(ks, 0), mx.concatenate(vs, 0)
        merged.offset = lmax
        return merged
    merged = SSMCache()
    merged.conv = mx.concatenate([c[layer_idx].conv for c in caches], 0) if sample.conv is not None else None
    merged.state = mx.concatenate([c[layer_idx].state for c in caches], 0) if sample.state is not None else None
    merged.offset = lmax
    return merged


class BatchState:
    """A lockstep batch: the joined cache and, per row, the padding in front of it and its rope offset."""

    def __init__(self, cache, pad, delta):
        self.cache, self.pad, self.delta = cache, pad, delta


class MlxHybridRuntime:
    name = "mlx"

    def __init__(self, model, config=None, image_token=None):
        self.model = model
        self.config = config or {}
        self.image_token = image_token
        self._attn = next((l.self_attn for l in model.layers if not l.is_linear), None)

    # ── state ──
    def make_cache(self):
        return self.model.make_cache()

    def settle(self, cache):
        """Finish computing everything the caches hold (they are lazy until something reads them)."""
        mx.eval([a for c in cache for a in c.arrays])

    def release(self):
        """Hand reusable buffers back to the system."""
        mx.clear_cache()

    def peak_gb(self):
        return round(mx.get_peak_memory() / 2**30, 2)

    def commit(self, cache, m):
        self.model.commit(cache, m)

    def can_verify(self, cache):
        return True             # the native model undoes a verify pass itself, recurrent layers included

    def rewind(self, cache, n):
        return False            # a recurrent state cannot be cut back: the engine drops the cache instead

    # ── sampling ──
    def sampler(self, vocab, seed, sampling):
        return Sampler(vocab, seed=seed, **sampling)

    def sample(self, sampler, logits):
        return sampler(logits)

    def flush(self, *handles):
        mx.async_eval(*handles)

    def read_token(self, handle):
        return handle if isinstance(handle, int) else handle.item()

    # ── forward passes ──
    def prefill(self, cache, ids, embeds, pos3, delta, s, e, hidden=False):
        """Positions s..e of the prompt, evaluated together with the cache state they leave behind."""
        if pos3 is not None:
            cos, sin = self._attn.mrope.cos_sin(mx.array(pos3[:, s:e]))
            pos = Positions(cos.astype(mx.float16), sin.astype(mx.float16))
        else:
            pos = Positions(delta=delta)
        if embeds is not None:
            logits = self.model(cache=cache, pos=pos, embeds=embeds[:, s:e], hidden=hidden)
        else:
            logits = self.model(mx.array([ids[s:e]]), cache, pos, hidden=hidden)
        mx.eval(logits, [a for c in cache for a in c.arrays])
        return logits

    def step(self, token, cache, delta):
        """One token — a handle from `sample`, or a plain id."""
        ids = mx.array([[token]]) if isinstance(token, int) else token.reshape(1, 1)
        return self.model(ids, cache, Positions(delta=delta))

    def verify(self, token, draft, cache, delta):
        return self.model(mx.array([[token] + draft]), cache, Positions(delta=delta), last_only=False, verify=True)

    def greedy_tokens(self, logits):
        return mx.argmax(logits[0], axis=-1).tolist()

    def verify_token(self, sampler, logits, i, draft_token):
        return sampler.verify(logits[0, i], draft_token)

    def sample_at(self, sampler, logits, i):
        return sampler(logits[:, i:i + 1]).item()

    # ── batched decode (engine/batch.py) ──
    def batch_prefill(self, ids):
        """One request alone, from position 0: (its cache, the last position's logits row)."""
        cache = self.make_cache()
        logits = self.model(mx.array([list(ids)]), cache, Positions(delta=0))
        return cache, logits[0, -1]

    def batch_join(self, caches, lens):
        """The per-request caches, left-padded to the longest and stacked into one batch."""
        lmax = max(lens)
        cache = [_assemble(caches, li, lmax) for li in range(len(caches[0]))]
        return BatchState(cache, mx.array([lmax - n for n in lens]), mx.array([n - lmax for n in lens]))

    def batch_step(self, state, tokens):
        """One token per row, every row at its own position, the padding masked: the [B, V] logits rows."""
        cols = mx.arange(state.cache[0].offset + 1)            # keys after this step's update
        mask = mx.where(cols[None, None, None, :] < state.pad[:, None, None, None], NEG, 0.0).astype(mx.float16)
        logits = self.model(mx.array([[t] for t in tokens]), state.cache, Positions(delta=state.delta), mask=mask)
        return logits[:, -1, :]

    def greedy_row(self, row):
        return int(mx.argmax(row).item())

    def sample_row(self, sampler, row):
        """A Sampler takes [1, L, V] and returns a lazy scalar; the batch loop needs a real id."""
        return int(sampler(row[None, None, :]).item())

    # ── images ──
    def embed_media(self, ids, media):
        """(embeddings with the image features in place, 3-D positions, the text position offset)."""
        pixels, grids = media
        feats = self.model.vision_tower(pixels, grids).astype(mx.float16)
        emb = self.model.embed(mx.array([ids]))
        idx = mx.array([i for i, t in enumerate(ids) if t == self.image_token])
        emb[0, idx] = feats
        pos3, delta = mrope_positions(ids, grids, self.image_token, self.config["vision_config"]["spatial_merge_size"])
        return emb, pos3, delta
