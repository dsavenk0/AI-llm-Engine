# Generation runtime: prompt building, prefix reuse, chunked prefill,
# GPU-side sampling and a pipelined decode loop.
import gc
import hashlib
import json
import os
import shutil
import threading
import time
from collections import deque

import mlx.core as mx
import numpy as np

from . import memory
from .loader import attach_vision, has_vision, load
from .runtime import MlxHybridRuntime, Sampler, mrope_positions  # noqa: F401  (re-exported)
from .session import (DRAFT_LEVELS, ContextError, DecodeLoop, Drafter, FormatError,  # noqa: F401
                      MemoryLimitError, is_looping, is_repeating)
from .tokenizer import StreamDecoder, Tokenizer
from .stopfilter import StopFilter

GiB = 2 ** 30


def _json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


# Qwen3.5 family sampling presets, used only when a bundle ships neither jang_config
# sampling_modes nor a generation_config.json (some mlx-community conversions don't —
# e.g. Qwen3.5-0.8B-4bit). Values are the ones the Bonsai authors wrote into their own
# bundle's sampling_modes. Without them the Sampler's defaults apply: temperature 1.0
# over the WHOLE vocabulary (no top-k/top-p), which sends a small model rambling.
QWEN35_SAMPLING = {
    True: {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 0.0},
    False: {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5},
}


def template_thinks_by_default(render):
    """Whether the chat template opens a reasoning block when no `enable_thinking` is passed:
    `render()` is the generation prompt of a one-message chat without that kwarg. Authors
    differ — Bonsai-27B's template reasons unless told not to, while the small Qwen3.5
    models (0.8B..) ship with reasoning OFF and only think on an explicit enable_thinking —
    and forcing it on a model its authors turned off is how a 0.8B ends up writing
    900 tokens of "Thinking Process:" before answering 7 + 5."""
    try:
        return render().rstrip().endswith("<think>")
    except Exception:  # an exotic template: keep the old behaviour
        return True


def sampling_presets(jang, gen, fallback=None):
    """{thinking: bool -> sampling kwargs}: the bundle's own presets, else its
    generation_config, else `fallback` (a {bool: kwargs} pair, or one kwargs dict for both
    modes; default: the Qwen3.5 family presets). Returns (presets, source)."""
    modes = jang.get("chat", {}).get("sampling_modes") or {}
    base = {k: gen[k] for k in ("temperature", "top_p", "top_k", "min_p") if k in gen}
    fb = QWEN35_SAMPLING if fallback is None else (fallback if True in fallback else {True: fallback, False: fallback})
    if modes.get("thinking") or modes.get("instruct_nothinking"):
        return ({True: dict(modes.get("thinking") or base or fb[True]),
                 False: dict(modes.get("instruct_nothinking") or base or fb[False])}, "jang_config")
    if base:
        return {True: dict(base), False: dict(base)}, "generation_config"
    return {True: dict(fb[True]), False: dict(fb[False])}, "family defaults"


class Engine(DecodeLoop):
    backend = "native"

    def __init__(self, path, vision=True, max_context=32768, max_tokens=8192, prefill_step=1024,
                 max_pixels=1024 * 1024, thinking=None, effort="medium", speculative=True,
                 mem_limit=0.88, sampling_override=None, draft_max=15, ngram_min=2, cache_limit_mb=128,
                 disk_cache_gb=8.0, disk_cache_dir="", think_frac=0.6, ptq1=False, prefill_guard=True, log=print,
                 steer=None, weight_patch=None):
        self.path, self.log = path, log
        self.prefill_guard = prefill_guard   # refuse a prompt that would not fit in free memory
        self.max_context, self.max_tokens, self.prefill_step = max_context, max_tokens, prefill_step  # 0 = auto
        self.max_pixels = max_pixels
        self.thinking, self.effort = thinking, effort
        self.speculative = speculative
        self.think_frac = think_frac  # share of a reply's token budget reasoning may use before forced to close
        self.ptq1_enabled = ptq1      # experimental: PTQ1_0 (1.75-bit) bundles
        self.steer = steer            # (direction file or folder, alpha or None): an activation edit, see steering.py
        self.weight_patch = weight_patch  # (candidate.patch folder, alpha or None): see engine/weightpatch.py
        self.mem_limit = mem_limit  # max fraction of system RAM in use (whole machine)

        self.progress = None           # {"phase", "done", "total"} while a long prompt is being prefilled
        self.sleeping, self.last_used = False, time.time()
        self._load_weights()
        nbytes = self.model_bytes
        self.vision_capable = has_vision(path)           # the checkpoint ships a vision tower
        self.vision_allowed = vision and self.vision_capable
        self.vision_bytes = 0          # bytes of the lazily loaded tower (0 until the first image)
        mx.set_cache_limit(int(cache_limit_mb) * 2**20)
        self.draft_levels = tuple(sorted({min(l, draft_max) for l in DRAFT_LEVELS}))
        self.ngram_min = max(1, min(4, ngram_min))
        self.tok = Tokenizer(path)
        jang = _json(os.path.join(path, "jang_config.json"))
        gen = _json(os.path.join(path, "generation_config.json"))
        self.pp_cfg = _json(os.path.join(path, "preprocessor_config.json"))
        self.sampling, source = sampling_presets(jang, gen)
        if source == "family defaults":
            log("sampling: the bundle ships no presets — using the Qwen3.5 family defaults "
                "(temperature 1.0/0.7, top-p 0.95/0.8, top-k 20)")
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
        # thinking=None means "what the model's own template does"; True/False is the user's choice
        self.thinking_default = bool(self.supports_thinking and template_thinks_by_default(
            lambda: self.tok.render([{"role": "user", "content": "hi"}])))
        self.thinking = self.thinking_default if thinking is None else bool(thinking)
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
        root = os.path.expanduser(disk_cache_dir) if disk_cache_dir else \
            os.path.expanduser("~/.cache/ai-llm-engine/prefix")
        gb = disk_cache_gb
        if isinstance(gb, str) and gb.strip().lower() == "auto":
            os.makedirs(root, exist_ok=True)
            free = shutil.disk_usage(root).free
            gb = max(1.0, free * 0.10 / GiB)  # 10% of free disk space, floor 1 GB
            log(f"disk cache: auto → {gb:.1f} GB (10% of {free / GiB:.0f} GB free on {root})")
        elif isinstance(gb, str):
            gb = float(gb)
        if gb > 0:
            from .prefix_store import PrefixStore
            with open(os.path.join(path, "config.json"), "rb") as f:
                key = f.read()
            if self.steer:      # a steered model computes different activations: its cache is its own
                from . import steering
                sf, ra = steering.resolve(self.steer[0])
                with open(sf, "rb") as g:
                    key += g.read() + repr(self.steer[1] if self.steer[1] is not None else ra).encode()
            if self.weight_patch and self.weight_patch[1] != 0:
                # patched weights compute different activations too: without this a patched model restored
                # prefixes the unpatched one had stored (and the reverse), mixing two models in one reply.
                # The manifest names the patch; size and mtime of the deltas catch a file replaced in place
                # without reading a delta that can be hundreds of megabytes.
                pdir, palpha = self.weight_patch
                with open(os.path.join(pdir, "manifest.json"), "rb") as g:
                    key += g.read()
                st = os.stat(os.path.join(pdir, "delta.safetensors"))
                key += repr((st.st_size, st.st_mtime_ns, 1.0 if palpha is None else palpha)).encode()
            mid = os.path.basename(path.rstrip("/")) + "-" + hashlib.sha1(key).hexdigest()[:8]
            self.store = PrefixStore(os.path.join(root, mid), int(gb * GiB), log=log)
        self._warmup()

    # ───────────────────────── sleep / wake ─────────────────────────
    def _load_weights(self):
        # vision tower is loaded on the first image request (saves ~0.5 GB for text/code work)
        self.model, self.config, self.model_bytes = load(self.path, vision=False, log=self.log,
                                                         ptq1=self.ptq1_enabled, steer=self.steer,
                                                         weight_patch=self.weight_patch)
        # the decode loop (engine/session.py) reaches the model only through this
        self.rt = MlxHybridRuntime(self.model, self.config, self.config.get("image_token_id"))
        if getattr(self.model, "ptq1", False) and self.speculative and os.environ.get("LLM_PTQ1_SPECULATIVE") != "1":
            # a verification pass of 2-16 rows is still several one-row steps of work on a ternary model
            # (the fused kernel is compute-bound: ~6 steps for 16 rows); LLM_PTQ1_SPECULATIVE=1 to measure
            self.speculative = False
            self.log("PTQ1 model: speculative decoding turned off (a multi-row pass costs several decode steps)")
        # Keep the weights resident: without a wired limit macOS may compress or
        # page them out under pressure and decode speed collapses.
        need = self.model_bytes + 0.5 * GiB
        rec = mx.device_info().get("max_recommended_working_set_size", int(self.model_bytes * 1.3))
        mx.set_wired_limit(int(min(rec, need)))
        # macOS caps `rec` from the system's iogpu.wired_limit_mb; below `need` the engine
        # is silently given less GPU memory than this model wants, which costs decode speed.
        # We only suggest the sysctl (it needs sudo and survives only until reboot) — never run it.
        if need > rec:
            self.wired_limit_advice_mb = int((need + GiB) // 2**20)  # +1 GB headroom for other apps
            self.log(f"memory: Metal's recommended working set is {rec / GiB:.1f} GB, this model wants "
                     f"~{need / GiB:.1f} GB — raise it with: sudo sysctl iogpu.wired_limit_mb="
                     f"{self.wired_limit_advice_mb} (resets on reboot)")
        else:
            self.wired_limit_advice_mb = None

    def sleep(self):
        """Free the GPU/unified memory held by the weights; the next request wakes the model."""
        if self.sleeping:
            return
        self.persist_prefix()
        self.model = self.rt = None
        self.cache, self.cache_tokens, self.snapshots = None, [], []
        self.vision_bytes = 0
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
            from mlx.utils import tree_flatten
            self.vision_bytes = sum(v.nbytes for _, v in tree_flatten(self.model.vision_tower.parameters()))
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

    def memory_breakdown(self):
        """Where this engine's memory goes, in bytes (safe to call from another thread)."""
        from .model import KVCache
        br = {"weights": 0 if self.sleeping else self.model_bytes, "vision": getattr(self, "vision_bytes", 0),
              "kv": 0, "ssm": 0, "snapshots": 0}
        seen = set()

        def add(key, arr):
            if arr is not None and id(arr) not in seen:
                seen.add(id(arr))
                br[key] += arr.nbytes
        try:
            for c in list(self.cache or []):
                if isinstance(c, KVCache):
                    add("kv", c.keys)
                    add("kv", c.values)
                else:
                    add("ssm", c.conv)
                    add("ssm", c.state)
            for _, snaps in list(self.snapshots):
                for snap in snaps:
                    if isinstance(snap, tuple):          # a recurrent layer's restore point holds arrays
                        add("snapshots", snap[0])
                        add("snapshots", snap[1])
        except Exception:  # the worker thread may be swapping caches right now
            pass
        active, cached = mx.get_active_memory(), mx.get_cache_memory()
        known = br["weights"] + br["vision"] + br["kv"] + br["ssm"] + br["snapshots"]
        br["activations"] = max(0, active - known)   # transient tensors of a running request
        br["mlx_cache"] = cached                       # freed buffers kept for reuse
        fp = memory.process_footprint()
        br["process"] = max(0, fp - active - cached) if fp else None  # Python, tokenizer, HTTP
        br["footprint"] = fp
        return br

    # ───────────────────────── memory ─────────────────────────
    def _drop_prefix_cache(self):
        self.cache, self.cache_tokens, self.snapshots = None, [], []
        mx.clear_cache()

    def _reserve(self, new_tokens):
        """Make sure prefill of `new_tokens` fits under the system RAM limit; sets the chunk size."""
        memory.reserve(self, new_tokens, mx.clear_cache)

    def _fit_output(self, max_tokens):
        """Clamp the output length to what still fits under the RAM limit."""
        return memory.fit_output(self, max_tokens)

    def _check_memory(self, stage):
        """Called during long work; frees caches, raises when the limit cannot be kept."""
        return memory.check(self, stage, mx.clear_cache)

    # ───────────────────────────── embeddings ─────────────────────────────
    def embed(self, texts, normalize=True):
        """One vector per text: the model's final hidden state at the last token, L2-normalized.

        What this is, and is not. The vector is read where the output head reads — after the last layer and
        the final norm — so it is whatever the loaded model already knows about the text. Unless the loaded
        model was trained for retrieval (Qwen3-Embedding and friends, which pool exactly here), these vectors
        are useful for comparing texts *this same model* embedded and should not be expected to match a
        dedicated embedding model. The dimension is the model's hidden size, not a number this picks.

        Each text is prefilled on its own cache, which is thrown away afterwards, so the conversation's
        prefix cache and recurrent state are untouched. Texts are done one at a time: prefill is already
        compute-bound (see tests/perf_baseline.json), so there is nothing for a batch to save here.
        """
        self.ensure_awake()
        if isinstance(texts, str):
            texts = [texts]
        out, total = [], 0
        for text in texts:
            ids = self.tok.encode(text) if isinstance(text, str) else list(text)
            if not ids:
                raise ValueError("cannot embed an empty input")
            if len(ids) >= self.max_context:
                raise ContextError(f"input is {len(ids)} tokens, context limit is {self.max_context}")
            self._reserve(len(ids))
            h = self._prefill(self.model.make_cache(), ids, hidden=True)
            v = h[0, -1].astype(mx.float32)
            if normalize:
                v = v / mx.maximum(mx.sqrt(mx.sum(v * v)), 1e-12)
            out.append([float(x) for x in v.tolist()])
            total += len(ids)
        mx.clear_cache()
        self.last_used = time.time()
        return out, total

    # ───────────────────────── several requests in one pass ─────────────────────────
    def generate_batch(self, reqs, on_piece=None):
        """Generate for several requests in one lockstep batch: the weights are read once per step for the
        whole batch instead of once per request. Measured motivation: with requests served one at a time the
        aggregate throughput is flat (four clients buy 1.01x), because ~68 % of a token is reading the weights.

        `reqs`: a list of dicts {ids, max_tokens, sampling, stop, should_stop, seed}. `on_piece(i, text)` is
        called as text appears for row i. Returns a list of per-row stats, in the same order.

        Deliberately not used inside a batch (both are single-request optimisations that a batch replaces):
        the prefix cache — every row prefills on its own — and speculative decoding. The thinking budget is
        not applied either; a reasoning model still stops on its own markers, max_tokens and the loop guard.
        """
        from .batch import batched_generate
        if not reqs:
            return []
        self.ensure_awake()
        for r in reqs:
            if len(r["ids"]) >= self.max_context:
                raise ContextError(f"prompt is {len(r['ids'])} tokens, context limit is {self.max_context}")

        rows = [_BatchRow(self, r) for r in reqs]
        max_new = max(r.max_tokens for r in rows)

        samplers = [(lambda row, s=r.sampler: self.rt.sample_row(s, row)) for r in rows]
        stoppers = [(lambda i=i: rows[i].stopped()) for i in range(len(rows))]

        def token(i, tok):
            rows[i].feed(tok, on_piece, i)

        t0 = time.time()
        outputs = batched_generate(self.rt, [list(r["ids"]) for r in reqs], max_new=max_new, eos=self.eos,
                                   samplers=samplers, should_stop=stoppers, on_token=token)
        dt = time.time() - t0
        for i, row in enumerate(rows):
            row.settle(outputs[i])          # the finish reason first: the tail is only released on "length"
            row.flush(on_piece, i)
        self.rt.release()
        return [r.stats(dt, len(reqs[i]["ids"])) for i, r in enumerate(rows)]


class _BatchRow:
    """One request inside a batch: its own detokeniser, stop strings, limits and finish reason."""

    def __init__(self, eng, req):
        self.eng = eng
        self.max_tokens = min(int(req.get("max_tokens") or eng.max_tokens), eng.max_tokens,
                              eng.max_context - len(req["ids"]))
        self.sampler = Sampler(eng.vocab, seed=req.get("seed"), **(req.get("sampling") or {}))
        self.dec = StreamDecoder(eng.tok)
        self.filter = StopFilter(req.get("stop"))
        self.should_stop = req.get("should_stop")
        self.out, self.text, self.finish, self.done, self.looped = [], "", "length", False, False

    def stopped(self):
        """Checked by the batch loop before a token is accepted."""
        if not self.done and self.should_stop and self.should_stop():
            self.finish, self.done = "cancelled", True
        return self.done

    def feed(self, tok, on_piece, i):
        self.out.append(tok)
        piece = self.dec.add(tok)
        self.text += piece
        hit, piece = self.filter.feed(piece)
        if piece and on_piece:
            on_piece(i, piece)
        if hit:
            self.finish, self.done = "stop", True
            return
        if len(self.out) % 32 == 0 and not self.eng._check_memory("decode"):
            self.finish, self.done = "length", True
            return
        if len(self.out) % 16 == 0 and is_looping(self.text):
            self.finish, self.done, self.looped = "stop", True, True
            return
        if len(self.out) >= self.max_tokens:
            self.finish, self.done = "length", True

    def settle(self, produced):
        """A row the batch ended on its own (never marked done here) stopped on an eos token."""
        if not self.done and len(produced) < self.max_tokens:
            self.finish = "stop"

    def flush(self, on_piece, i):
        """Release what the stop filter held back, and the decoder's tail — the latter only when the answer
        ran to its length limit, exactly as the single-request path does."""
        if not on_piece:
            return
        rest = self.filter.flush()
        if rest:
            on_piece(i, rest)
        tail = self.dec.flush()
        if tail and self.finish == "length":
            on_piece(i, tail)

    def stats(self, dt, prompt_tokens):
        n = len(self.out)
        return {"prompt_tokens": prompt_tokens, "cached_tokens": 0, "completion_tokens": n,
                "prefill_s": 0.0, "decode_s": round(dt, 3),
                "prefill_tps": 0.0, "decode_tps": round(n / dt, 2) if dt else 0.0,
                "finish_reason": self.finish, "loop_stopped": self.looped, "think_forced": None,
                "peak_gb": round(mx.get_peak_memory() / GiB, 2), "batched": True,
                "ram_used": memory.used_fraction(), "verifies": 0, "drafted": 0, "accepted": 0,
                "draft_accept": None}


def summarize(t):
    """Derived numbers for a totals dict (session or all-time)."""
    p, c = t.get("prompt_tokens", 0), t.get("cached_tokens", 0)
    return {**t,
            "cache_hit": round(c / p, 3) if p else None,
            "avg_prefill_tps": round((p - c) / t["prefill_s"], 1) if t.get("prefill_s") else None,
            "avg_decode_tps": round(t.get("completion_tokens", 0) / t["decode_s"], 2) if t.get("decode_s") else None,
            "draft_accept": round(t["accepted"] / t["drafted"], 2) if t.get("drafted") else None}
