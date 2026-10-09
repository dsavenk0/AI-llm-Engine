# Generation runtime: prompt building, prefix reuse, chunked prefill,
# GPU-side sampling and a pipelined decode loop.
import gc
import hashlib
import json
import os
import threading
import time
from collections import deque

import mlx.core as mx
import numpy as np

from . import memory
from .loader import attach_vision, has_vision, load
from .model import Positions
from .tokenizer import StreamDecoder, Tokenizer

GiB = 2 ** 30


def _json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


class ContextError(ValueError):
    pass


class MemoryLimitError(RuntimeError):
    """Not enough free RAM to serve the request without crossing the system limit."""


class Sampler:
    def __init__(self, vocab, temperature=1.0, top_p=1.0, top_k=0, min_p=0.0, presence_penalty=0.0, seed=None):
        self.temp, self.top_p, self.top_k, self.min_p = temperature, top_p, top_k, min_p
        self.pp = presence_penalty
        self.vocab = vocab
        self.seen = mx.zeros((vocab,), dtype=mx.bool_) if presence_penalty else None
        self.key = mx.random.key(seed) if seed is not None else None

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
        row = row.astype(mx.float32)
        if self.seen is not None:
            row = row - self.pp * self.seen.astype(mx.float32)
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

    def _mark(self, tok):
        if self.seen is not None:
            self.seen = mx.logical_or(self.seen, mx.arange(self.vocab) == tok)

    def __call__(self, logits):
        """Sample from the last position of [1, L, V] logits; returns a lazy scalar."""
        row = logits[0, -1]
        if self.temp <= 0:
            r = row.astype(mx.float32)
            tok = mx.argmax(r - self.pp * self.seen.astype(mx.float32) if self.seen is not None else r)
        else:
            idx, vals = self._dist(row)
            tok = idx[mx.random.categorical(vals, key=self._rand())]
        self._mark(tok)
        return tok

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


class Drafter:
    """Prompt-lookup drafts: continue the most recent earlier occurrence of the
    current n-gram suffix (n = nmax..nmin) found in prompt + output."""

    def __init__(self, tokens, nmin=2, nmax=4):
        self.nmin, self.nmax = nmin, nmax
        self.toks = []
        self.index = {n: {} for n in range(nmin, nmax + 1)}
        for t in tokens:
            self.add(t)

    def add(self, t):
        L = len(self.toks)  # t lands at position L: every n-gram ending at L-1 now has a continuation
        for n in range(self.nmin, self.nmax + 1):
            if L >= n:
                self.index[n][tuple(self.toks[L - n:L])] = L
        self.toks.append(t)

    def propose(self, k):
        T = self.toks
        for n in range(self.nmax, self.nmin - 1, -1):
            if len(T) >= n:
                p = self.index[n].get(tuple(T[-n:]))
                if p is not None and p < len(T):
                    return T[p:p + k]
        return []


# Draft lengths to try. Verifying 1..4 extra tokens costs ~1.4..2.3 decode
# steps; 5..15 all cost the same (~5 steps, tiled matmul) — so jump to 15.
DRAFT_LEVELS = (1, 2, 3, 4, 15)


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


class Engine:
    def __init__(self, path, vision=True, max_context=32768, max_tokens=8192, prefill_step=1024,
                 max_pixels=1024 * 1024, thinking=True, effort="medium", speculative=True,
                 mem_limit=0.88, sampling_override=None, draft_max=15, ngram_min=2, cache_limit_mb=128,
                 disk_cache_gb=8.0, log=print):
        self.path, self.log = path, log
        self.max_context, self.max_tokens, self.prefill_step = max_context, max_tokens, prefill_step  # 0 = auto
        self.max_pixels = max_pixels
        self.thinking, self.effort = thinking, effort
        self.speculative = speculative
        self.mem_limit = mem_limit  # max fraction of system RAM in use (whole machine)

        self.sleeping, self.last_used = False, time.time()
        self._load_weights()
        nbytes = self.model_bytes
        self.vision_allowed = vision and has_vision(path)
        mx.set_cache_limit(int(cache_limit_mb) * 2**20)
        self.draft_levels = tuple(sorted({min(l, draft_max) for l in DRAFT_LEVELS}))
        self.ngram_min = max(1, min(4, ngram_min))
        self.tok = Tokenizer(path)
        jang = _json(os.path.join(path, "jang_config.json"))
        gen = _json(os.path.join(path, "generation_config.json"))
        self.pp_cfg = _json(os.path.join(path, "preprocessor_config.json"))
        modes = jang.get("chat", {}).get("sampling_modes", {})
        base = {k: gen[k] for k in ("temperature", "top_p", "top_k") if k in gen}
        self.sampling = {True: dict(modes.get("thinking") or base), False: dict(modes.get("instruct_nothinking") or base)}
        for mode in self.sampling.values():  # manual overrides win over the model's presets
            mode.update({k: v for k, v in (sampling_override or {}).items() if v is not None})
        eos = gen.get("eos_token_id", self.config.get("eos_token_id", []))
        self.eos = set(eos if isinstance(eos, list) else [eos])
        im_end = self.tok.token_id("<|im_end|>")
        if im_end is not None:
            self.eos.add(im_end)
        self.im_start = self.tok.token_id("<|im_start|>")
        self.image_token = self.config.get("image_token_id")
        self.efforts = jang.get("reasoning", {}).get("supported_reasoning_efforts") or []
        self.supports_thinking = "enable_thinking" in self.tok.template_source
        self.vocab = self.config["text_config"]["vocab_size"]
        tc = self.config["text_config"]
        n_full = sum(1 for l in self.model.layers if not l.is_linear)
        self.kv_per_token = n_full * 2 * tc["num_key_value_heads"] * tc.get("head_dim", 256) * 2
        # transient activations per prefill token (projections, gate/up, logits row) — measured ~0.3 MB
        self.act_per_token = 900 * 1024  # measured: ~1 GB peak for a 1024-token chunk incl. cache churn
        self.step_now = prefill_step
        self.base_work = 64 * 2**20
        if not self.max_context:  # auto: what fits next to the weights under the RAM limit on this Mac
            native = tc.get("max_position_embeddings", 32768)
            budget = memory.TOTAL * self.mem_limit - nbytes - 4 * GiB  # ~4 GB for macOS and apps
            self.max_context = int(max(4096, min(native, budget // self.kv_per_token // 4096 * 4096)))
            log(f"context: auto → {self.max_context} tokens "
                f"({memory.TOTAL / GiB:.0f} GB RAM, {self.kv_per_token // 1024} KB KV per token)")
        self.max_tokens = min(self.max_tokens, self.max_context // 2)

        self.lock = threading.Lock()
        self.cache = None
        self.cache_tokens = []    # tokens whose effects are in self.cache
        self.snapshots = []       # [(token prefix, per-layer SSM snapshots)] — cheap restore points
        self.stats = {"requests": 0, "last": {}}
        self.totals = {"requests": 0, "prompt_tokens": 0, "cached_tokens": 0, "completion_tokens": 0,
                       "prefill_s": 0.0, "decode_s": 0.0, "drafted": 0, "accepted": 0, "since": time.time()}
        self.history = deque(maxlen=60)
        self.store = None
        if disk_cache_gb > 0:
            from .prefix_store import PrefixStore
            with open(os.path.join(path, "config.json"), "rb") as f:
                mid = os.path.basename(path.rstrip("/")) + "-" + hashlib.sha1(f.read()).hexdigest()[:8]
            self.store = PrefixStore(os.path.join(os.path.expanduser("~/.cache/ai-llm-engine/prefix"), mid),
                                     int(disk_cache_gb * GiB), log=log)
        self._warmup()

    # ───────────────────────── sleep / wake ─────────────────────────
    def _load_weights(self):
        # vision tower is loaded on the first image request (saves ~0.5 GB for text/code work)
        self.model, self.config, self.model_bytes = load(self.path, vision=False, log=self.log)
        # Keep the weights resident: without a wired limit macOS may compress or
        # page them out under pressure and decode speed collapses.
        rec = mx.device_info().get("max_recommended_working_set_size", int(self.model_bytes * 1.3))
        mx.set_wired_limit(int(min(rec, self.model_bytes + 0.5 * GiB)))

    def sleep(self):
        """Free the GPU/unified memory held by the weights; the next request wakes the model."""
        if self.sleeping:
            return
        self.persist_prefix()
        self.model = None
        self.cache, self.cache_tokens, self.snapshots = None, [], []
        gc.collect()
        mx.clear_cache()
        mx.set_wired_limit(0)
        self.sleeping = True
        self.log("sleep: model unloaded (idle); it wakes on the next request")

    def ensure_awake(self):
        self.last_used = time.time()
        if not self.sleeping:
            return
        t0 = time.time()
        self._load_weights()
        self._warmup()
        self.sleeping = False
        self.log(f"wake: model reloaded in {time.time() - t0:.1f}s")

    def persist_prefix(self):
        """Write the current snapshot prefix to the SSD cache (called between requests)."""
        if self.store and self.cache is not None:
            try:
                for toks, snaps in self.snapshots:
                    self.store.save(toks, self.cache, snaps)
            except Exception as e:  # the SSD cache must never break serving
                self.log(f"prefix cache: save failed: {e!r}")

    def _warmup(self):
        cache = self.model.make_cache()
        mx.eval(self.model(mx.array([[self.im_start or 0] * 4]), cache))
        mx.eval(self.model(mx.array([[0]]), cache))
        mx.clear_cache()

    # ───────────────────────── prompt ─────────────────────────
    def build_prompt(self, messages, tools=None, thinking=None, effort=None, extra_kwargs=None):
        thinking = self.thinking if thinking is None else thinking
        kw = dict(extra_kwargs or {})
        if self.supports_thinking:
            kw["enable_thinking"] = bool(thinking)
            if thinking and (effort or self.effort) and (not self.efforts or (effort or self.effort) in self.efforts):
                kw["reasoning_effort"] = effort or self.effort
        images = []
        for m in messages:
            if isinstance(m.get("content"), list):
                for it in m["content"]:
                    if isinstance(it, dict) and ("image_url" in it or "image" in it or it.get("type") == "image"):
                        images.append(it.get("image_url") or it.get("image") or it)
        text = self.tok.render(messages, tools=tools, **kw)
        thinking_open = text.rstrip().endswith("<think>")
        return text, images, thinking_open

    def encode(self, text, images):
        self.ensure_awake()
        if not images:
            return self.tok.encode(text), None
        if "vision_tower" not in self.model:
            if not self.vision_allowed:
                raise ValueError("this model/server has no vision support")
            self._drop_prefix_cache()
            self._reserve(4096)  # ~0.5 GB tower + image activations
            attach_vision(self.model, self.path, log=self.log)
        from .vision import load_image, preprocess
        pix, grids = [], []
        for ref in images:
            p, g = preprocess(load_image(ref), self.pp_cfg, self.max_pixels)
            pix.append(p)
            grids.append(g)
        merge = self.config["vision_config"]["spatial_merge_size"]
        parts = text.split("<|image_pad|>")
        if len(parts) - 1 != len(grids):
            raise ValueError("image placeholders do not match images")
        ids = []
        for i, part in enumerate(parts):
            ids += self.tok.encode(part)
            if i < len(grids):
                t, h, w = grids[i]
                ids += [self.image_token] * (t * h * w // merge ** 2)
        return ids, (mx.concatenate(pix), grids)

    # ───────────────────────── prefill ─────────────────────────
    def _prefill(self, cache, ids, embeds=None, pos3=None, delta=0, start=0):
        """Run ids[start:] through the model in chunks; return last-position logits.

        Near the RAM limit the chunk shrinks (1024 → 256 → 128): peak activation
        memory scales with it, so long prompts still go through instead of failing.
        """
        logits, n, s, step = None, len(ids), start, self.step_now
        attn = self.model.layers[self._attention_layer()].self_attn
        while s < n:
            e = min(n, s + step)
            if pos3 is not None:
                cos, sin = attn.mrope.cos_sin(mx.array(pos3[:, s:e]))
                pos = Positions(cos.astype(mx.float16), sin.astype(mx.float16))
            else:
                pos = Positions(delta=delta)
            if embeds is not None:
                logits = self.model(cache=cache, pos=pos, embeds=embeds[:, s:e])
            else:
                logits = self.model(mx.array([ids[s:e]]), cache, pos)
            mx.eval(logits, [a for c in cache for a in (c.arrays if hasattr(c, "arrays") else c.state)])
            s = e
            if s < n:
                mx.clear_cache()
                used = memory.used_fraction()
                if used > self.mem_limit and step > 128:
                    step = max(128, step // 4)
                    self.log(f"memory: {used:.0%} RAM in use, prefill chunk reduced to {step}")
                elif used > self.mem_limit:
                    self._check_memory("prefill")
        return logits

    def _attention_layer(self):
        return next(i for i, l in enumerate(self.model.layers) if not l.is_linear)

    def _restore_or_reset(self, ids):
        """Reuse the cache when the new prompt extends a known prefix."""
        if self.cache is not None:
            full = self.cache_tokens
            if len(ids) > len(full) and ids[:len(full)] == full:
                return len(full)
            for toks, snaps in sorted(self.snapshots, key=lambda x: -len(x[0])):
                if len(ids) > len(toks) and ids[:len(toks)] == toks:
                    for c, s in zip(self.cache, snaps):
                        if s is None:
                            c.trim_to(len(toks))
                        else:
                            c.restore(s)
                    # longer snapshots referred to KV positions that will now be overwritten
                    self.snapshots = [x for x in self.snapshots if len(x[0]) <= len(toks)]
                    return len(toks)
        if self.store:
            hit = self.store.lookup(ids)
            if hit:
                self.cache, snaps, toks = self.store.load(hit, self.model)
                self.snapshots, self.cache_tokens = [(toks, snaps)], list(toks)
                return len(toks)
        self.cache = self.model.make_cache()
        self.cache_tokens, self.snapshots = [], []
        return 0

    # ───────────────────────── memory ─────────────────────────
    def _drop_prefix_cache(self):
        self.cache, self.cache_tokens, self.snapshots = None, [], []
        mx.clear_cache()

    def _reserve(self, new_tokens):
        """Make sure prefill of `new_tokens` fits under the system RAM limit.

        Picks the largest prefill chunk (1024 → 512 → 256 → 128) whose peak fits
        the current headroom, so tight memory means smaller chunks, not a refusal.
        """
        def need(step):
            return new_tokens * self.kv_per_token + self.base_work + min(new_tokens, step) * self.act_per_token

        steps = [s_ for s_ in (self.prefill_step, 512, 256, 128) if s_ <= self.prefill_step]
        for attempt in range(3):
            head = memory.headroom_bytes(self.mem_limit)
            fit = next((s_ for s_ in steps if need(s_) <= head), None)
            if fit:
                self.step_now = fit
                if fit < self.prefill_step:
                    self.log(f"memory: {memory.used_fraction():.0%} RAM in use, prefill chunk {fit}")
                return
            mx.clear_cache()
            if attempt == 1 and self.snapshots:
                self.snapshots = []  # snapshot states are the cheapest thing to give up
        need = need(steps[-1])
        if head < need:
            raise MemoryLimitError(
                f"not enough free RAM: {memory.used_fraction():.0%} of system memory in use, limit "
                f"{self.mem_limit:.0%}; the prompt needs ~{need / GiB:.1f} GB more. Close other apps or "
                f"shorten the conversation.")

    def _fit_output(self, max_tokens):
        """Clamp the output length to what still fits under the RAM limit."""
        head = memory.headroom_bytes(self.mem_limit) - self.base_work
        fit = max(16, head // self.kv_per_token)
        if fit < max_tokens:
            self.log(f"memory: output capped at {fit} tokens ({memory.used_fraction():.0%} RAM in use)")
        return int(min(max_tokens, fit))

    def _check_memory(self, stage):
        """Called during long work; frees caches, raises when the limit cannot be kept."""
        if memory.used_fraction() <= self.mem_limit:
            return True
        mx.clear_cache()
        if memory.used_fraction() <= self.mem_limit + 0.02:
            return True
        if stage == "prefill":
            raise MemoryLimitError(f"system memory reached {memory.used_fraction():.0%} during prefill "
                                   f"(limit {self.mem_limit:.0%}); request stopped to keep the Mac responsive")
        return False

    # ───────────────────────── generate ─────────────────────────
    def generate(self, ids, media=None, max_tokens=None, sampling=None, stop=None, should_stop=None, seed=None):
        """Yield (token_id, text) pairs; final item is a dict with stats."""
        self.ensure_awake()
        if len(ids) >= self.max_context:
            raise ContextError(f"prompt is {len(ids)} tokens, context limit is {self.max_context}")
        max_tokens = min(max_tokens or self.max_tokens, self.max_tokens, self.max_context - len(ids))
        sampler = Sampler(self.vocab, seed=seed, **(sampling or {}))
        t0 = time.time()
        delta = 0
        if media is not None:
            self._drop_prefix_cache()
            self._reserve(len(ids))
            pixels, grids = media
            feats = self.model.vision_tower(pixels, grids).astype(mx.float16)
            emb = self.model.embed(mx.array([ids]))
            idx = mx.array([i for i, t in enumerate(ids) if t == self.image_token])
            emb[0, idx] = feats
            pos3, delta = mrope_positions(ids, grids, self.image_token, self.config["vision_config"]["spatial_merge_size"])
            self.cache = self.model.make_cache()
            self.cache_tokens, self.snapshots = [], []
            cache, reused = self.cache, 0
            logits = self._prefill(cache, ids, embeds=emb, pos3=pos3)
        else:
            reused = self._restore_or_reset(ids)
            self._reserve(len(ids) - reused)
            cache = self.cache
            # Restore points: the end of the first message (usually the system
            # prompt — shared by every conversation of an agent) and the final
            # <|im_start|> (the generation prompt — shared by the next turn).
            starts = [i for i, t in enumerate(ids) if t == self.im_start]
            cuts = sorted({c for c in (starts[1] if len(starts) > 1 else None, starts[-1] if starts else None)
                           if c is not None and reused < c < len(ids)})
            pos = reused
            for cut in cuts:
                self._prefill(cache, ids[:cut], start=pos)
                self.snapshots.append((ids[:cut], [c.snapshot() if hasattr(c, "snapshot") else None for c in cache]))
                pos = cut
            self.snapshots = self.snapshots[-3:]
            logits = self._prefill(cache, ids, start=pos)
        self.cache_tokens = list(ids)
        t_prefill = time.time() - t0
        max_tokens = self._fit_output(max_tokens)

        dec = StreamDecoder(self.tok)
        out_tokens, text, finish = [], "", "length"
        spec = {"verifies": 0, "drafted": 0, "accepted": 0}
        t1 = time.time()

        def emit(tok):
            """Returns (stop, piece). Handles eos / stop strings / cancellation."""
            nonlocal text, finish
            out_tokens.append(tok)
            if tok in self.eos:
                finish = "stop"
                return True, None
            piece = dec.add(tok)
            text += piece
            if stop and any(s_ in text for s_ in stop):
                finish = "stop"
                return True, piece
            if should_stop and should_stop():
                finish = "cancelled"
                return True, piece
            if len(out_tokens) % 32 == 0 and not self._check_memory("decode"):
                finish = "length"
                self.log(f"memory: stopped generation at {memory.used_fraction():.0%} RAM")
                return True, piece
            return len(out_tokens) >= max_tokens, piece

        if not (self.speculative and media is None):
            # plain pipelined decode: graph for token t+1 is built while t runs
            y = sampler(logits)
            mx.async_eval(y)
            fed = list(ids)
            for _ in range(max_tokens):
                nxt = sampler(self.model(y.reshape(1, 1), cache, Positions(delta=delta)))
                mx.async_eval(nxt)
                tok = y.item()
                fed.append(tok)
                done, piece = emit(tok)
                if piece:
                    yield tok, piece
                if done:
                    break
                y = nxt
        else:
            # Prompt-lookup speculative decode with exact acceptance. Plain steps
            # stay pipelined; verification kicks in only while the output copies
            # text that already exists (code edits, quotes, tool arguments).
            drafter = Drafter(ids, nmin=self.ngram_min, nmax=max(4, self.ngram_min))
            fed = list(ids)
            y = sampler(logits)
            mx.async_eval(y)
            lvl, spec_mode, done = 1, False, False
            while not done:
                if not spec_mode:
                    nxt = sampler(self.model(y.reshape(1, 1), cache, Positions(delta=delta)))
                    mx.async_eval(nxt)
                    tok = y.item()
                    drafter.add(tok)
                    fed.append(tok)
                    done, piece = emit(tok)
                    if piece:
                        yield tok, piece
                    y = nxt
                    spec_mode = bool(drafter.propose(1))  # current suffix seen before → likely copying
                    continue

                tok = y if isinstance(y, int) else y.item()
                if tok in self.eos:
                    out_tokens.append(tok)
                    finish = "stop"
                    break
                drafter.add(tok)
                room = max_tokens - len(out_tokens) - 1
                draft = drafter.propose(self.draft_levels[lvl])[:max(0, room)]
                if not draft:  # back to pipelined plain decode
                    y = sampler(self.model(mx.array([[tok]]), cache, Positions(delta=delta)))
                    mx.async_eval(y)
                    fed.append(tok)
                    done, piece = emit(tok)
                    if piece:
                        yield tok, piece
                    spec_mode = False
                    continue
                logits = self.model(mx.array([[tok] + draft]), cache, Positions(delta=delta),
                                    last_only=False, verify=True)
                acc, nxt = 0, None
                if sampler.greedy:
                    best = mx.argmax(logits[0], axis=-1).tolist()
                    while acc < len(draft) and best[acc] == draft[acc]:
                        acc += 1
                    nxt = best[acc]
                else:
                    for i, d in enumerate(draft):
                        ok, t_ = sampler.verify(logits[0, i], d)
                        if not ok:
                            nxt = t_
                            break
                        acc += 1
                    if nxt is None:
                        nxt = sampler(logits[:, len(draft):len(draft) + 1]).item()
                self.model.commit(cache, 1 + acc)
                spec["verifies"] += 1
                spec["drafted"] += len(draft)
                spec["accepted"] += acc
                full = acc == len(draft) == self.draft_levels[lvl]
                lvl = min(lvl + 1, len(self.draft_levels) - 1) if full else max(0, min(lvl, acc) - (acc == 0))
                for j, t_ in enumerate([tok] + draft[:acc]):
                    if j:
                        drafter.add(t_)
                    fed.append(t_)
                    done, piece = emit(t_)
                    if piece:
                        yield t_, piece
                    if done:
                        break
                y = nxt
        tail = dec.flush()
        if tail and finish == "length":
            yield None, tail
        mx.eval([a for c in cache for a in (c.arrays if hasattr(c, "arrays") else c.state)])
        self.cache_tokens = fed
        if media is not None:
            self.cache = None  # 3-D positions: not reusable as a text prefix
        t_dec = time.time() - t1
        n_out = len(out_tokens)
        st = {
            "prompt_tokens": len(ids), "cached_tokens": reused, "completion_tokens": n_out,
            "prefill_s": round(t_prefill, 3), "decode_s": round(t_dec, 3),
            "prefill_tps": round((len(ids) - reused) / t_prefill, 1) if t_prefill > 0 else 0,
            "decode_tps": round(n_out / t_dec, 2) if t_dec > 0 else 0,
            "finish_reason": finish, "peak_gb": round(mx.get_peak_memory() / GiB, 2),
            "draft_accept": round(spec["accepted"] / spec["drafted"], 2) if spec["drafted"] else None,
            "verifies": spec["verifies"],
            "ram_used": round(memory.used_fraction(), 3),
        }
        self.stats["requests"] += 1
        self.stats["last"] = st
        T_ = self.totals
        T_["requests"] += 1
        T_["prompt_tokens"] += st["prompt_tokens"]
        T_["cached_tokens"] += st["cached_tokens"]
        T_["completion_tokens"] += st["completion_tokens"]
        T_["prefill_s"] += st["prefill_s"]
        T_["decode_s"] += st["decode_s"]
        T_["drafted"] += spec["drafted"]
        T_["accepted"] += spec["accepted"]
        self.history.append({"t": round(time.time()), "prompt": st["prompt_tokens"], "cached": st["cached_tokens"],
                             "out": st["completion_tokens"], "prefill_tps": st["prefill_tps"],
                             "decode_tps": st["decode_tps"], "accept": st["draft_accept"]})
        mx.clear_cache()
        yield st


def summarize(t):
    """Derived numbers for a totals dict (session or all-time)."""
    p, c = t.get("prompt_tokens", 0), t.get("cached_tokens", 0)
    return {**t,
            "cache_hit": round(c / p, 3) if p else None,
            "avg_prefill_tps": round((p - c) / t["prefill_s"], 1) if t.get("prefill_s") else None,
            "avg_decode_tps": round(t.get("completion_tokens", 0) / t["decode_s"], 2) if t.get("decode_s") else None,
            "draft_accept": round(t["accepted"] / t["drafted"], 2) if t.get("drafted") else None}
