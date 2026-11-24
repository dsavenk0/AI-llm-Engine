"""A GGUF model run by this engine's own MLX decoder (engine/dense.py) — no llama.cpp involved.

It plugs into the same server, prompt cache, stop handling, reasoning budget and memory watchdog as the
mlx-lm backend (engine/compat.py) by replacing only what that class takes from mlx-lm: the model, the
tokenizer and the cache. Which files qualify is decided by `engine.gguf.native_support`; everything
else keeps running on llama.cpp.
"""
import os
import shutil
import time

from . import gguf
from .compat import CompatEngine, MlxLmRuntime, _Decoder
from .generate import DRAFT_LEVELS, GiB, MemoryLimitError

IMAGE_PAD = "<|image_pad|>"
IMAGE_SLOT = "<|vision_start|>" + IMAGE_PAD + "<|vision_end|>"


class GGUFRuntime(MlxLmRuntime):
    """MlxLmRuntime plus what an image prompt needs on the built-in decoder: embeddings with the image features in
    place, 3-D rotary positions for the prompt and the shifted text positions after it."""

    accepts_embeds = True

    def __init__(self, model, cache_api, engine):
        super().__init__(model, cache_api)
        self.engine = engine

    def prefill(self, cache, ids, embeds, pos3, delta, s, e, hidden=False):
        import mlx.core as mx
        if embeds is None:
            extra = {"delta": delta} if delta else {}       # text after an image sits at shifted positions
            logits = self.model(mx.array([ids[s:e]]), cache=cache, **extra)
        else:
            cos, sin = self.model.mrope.cos_sin(mx.array(pos3[:, s:e]))
            logits = self.model(cache=cache, embeds=embeds[:, s:e], cos=cos, sin=sin)
        mx.eval([c.state for c in cache])
        return logits

    def step(self, token, cache, delta):
        import mlx.core as mx
        ids = mx.array([[token]]) if isinstance(token, int) else token.reshape(1, 1)
        return self.model(ids, cache=cache, **({"delta": delta} if delta else {}))

    def verify(self, token, draft, cache, delta):
        import mlx.core as mx
        self._width = 1 + len(draft)
        return self.model(mx.array([[token] + draft]), cache=cache, **({"delta": delta} if delta else {}))

    def embed_media(self, ids, media):
        import mlx.core as mx
        from .runtime import mrope_positions
        pixels, grids = media
        eng = self.engine
        feats = self.model.vision(pixels, grids)
        emb = self.model.embed_tokens(mx.array([ids]))
        idx = mx.array([i for i, t in enumerate(ids) if t == eng.image_token])
        emb[0, idx] = feats.astype(emb.dtype)
        pos3, delta = mrope_positions(ids, grids, eng.image_token, eng.image_merge)
        return emb, pos3, delta


def _trim(cache, n):
    return [c.trim(n) for c in cache][0] if cache else 0


class GGUFEngine(CompatEngine):
    backend = "gguf-mlx"
    REQUIRES_MLX_LM = False

    def __init__(self, path, steer=None, weight_patch=None, speculative=True, draft_max=15, ngram_min=2,
                 vision=True, max_pixels=1024 * 1024, **kw):
        self.steer = steer              # (direction file or folder, alpha or None), see engine/steering.py
        self.weight_patch = weight_patch  # (candidate.patch folder, alpha or None), see engine/weightpatch.py
        self.max_pixels = max_pixels
        super().__init__(path, **kw)
        # images: the projector next to the file, when it is the Qwen3-VL tower this decoder runs (loaded on the
        # first image, ~0.2 GB, while the language model stays as it is — nothing restarts)
        from .vision import GGUF_PROJECTORS, gguf_projector_type
        self.projector = gguf.projector_for(path)
        self.vision_capable = bool(self.projector and gguf_projector_type(self.projector) in GGUF_PROJECTORS
                                   and getattr(self.model, "mrope", None) is not None)
        self.vision_allowed = self.vision_capable and bool(vision)
        self.image_token = None
        self.image_pp, self.image_merge = None, 2
        # prompt-lookup speculation: the key/value cache can be trimmed, so rejected drafts are simply forgotten
        # a hybrid model's recurrent state cannot be rewound past a rejected draft: no speculation there
        self.speculative = bool(speculative) and not getattr(self.model, "hybrid", False)
        self.draft_levels = tuple(sorted({min(l, max(1, min(15, draft_max))) for l in DRAFT_LEVELS}))
        self.ngram_min = max(1, min(4, ngram_min))

    # ---- what CompatEngine takes from mlx-lm
    def _read_configs(self, path):
        meta = gguf.read_header_full(path)
        arch = meta.get("general.architecture", "")
        g = lambda k, d=None: meta.get(f"{arch}.{k}", d)
        heads = int(g("attention.head_count", 32))
        config = {"model_type": arch, "num_hidden_layers": int(g("block_count", 32)),
                  "hidden_size": int(g("embedding_length", 4096)), "num_attention_heads": heads,
                  "num_key_value_heads": int(g("attention.head_count_kv", heads)),
                  "head_dim": int(g("attention.key_length", int(g("embedding_length", 4096)) // heads)),
                  "max_position_embeddings": int(g("context_length", 4096)), "vocab_size": 0}
        return config, {}               # a GGUF carries no generation_config: generic sampling presets

    def _attach_runtime(self):
        self.rt = GGUFRuntime(self.model, self._cache_api(), self)

    def build_prompt(self, messages, tools=None, thinking=None, effort=None, extra_kwargs=None):
        """The compat prompt, with each image of a message written as the model's own image slot (in its place
        among the text) so the template keeps it — the images themselves come back in order for encode()."""
        if not self.vision_allowed:
            return super().build_prompt(messages, tools, thinking, effort, extra_kwargs)
        images, flat = [], []
        for m in messages:
            if isinstance(m.get("content"), list):
                parts = []
                for it in m["content"]:
                    if isinstance(it, dict) and it.get("type") == "text":
                        parts.append(it.get("text", ""))
                    elif isinstance(it, dict):
                        images.append(it)
                        parts.append(IMAGE_SLOT)
                m = {**m, "content": "".join(parts)}
            flat.append(m)
        text, _, opens = super().build_prompt(flat, tools, thinking, effort, extra_kwargs)
        return text, images, opens

    def encode(self, text, images):
        if not images:
            return super().encode(text, images)
        if not self.vision_allowed:
            raise ValueError("this model/server has no vision support")
        self.ensure_awake()
        if self.model.vision is None:
            from .vision import load_gguf_projector
            self._reserve(4096)                       # the tower and an image's activations
            self.model.vision, self.image_pp, self.image_merge = load_gguf_projector(self.projector, log=self.log)
            ids = self.tok.encode(IMAGE_PAD, add_special_tokens=False)
            if len(ids) != 1:
                raise ValueError(f"the tokenizer has no single {IMAGE_PAD} token")
            self.image_token = ids[0]
        import mlx.core as mx
        from .vision import expand_image_tokens, load_image, preprocess
        pix, grids = [], []
        for ref in images:
            p, g = preprocess(load_image(ref), self.image_pp, self.max_pixels)
            pix.append(p)
            grids.append(g)
        ids = expand_image_tokens(lambda t: self.tok.encode(t, add_special_tokens=False), text, grids,
                                  self.image_token, self.image_merge, IMAGE_PAD)
        return ids, (mx.concatenate(pix), grids)

    def _cache_api(self):
        make = lambda model: model.make_cache()
        can_trim = lambda cache: all(c.is_trimmable() for c in cache)
        return make, can_trim, _trim

    def _load(self):
        from . import dense, gguf_tokenizer, steering
        from . import memory
        t0 = time.time()
        cache_plan = None
        stream_experts = None
        if os.environ.get("LLM_GGUF_STREAM_EXPERTS", "1") != "0" and "LLM_GGUF_EXPERT_CACHE" not in os.environ:
            cache_plan = dense.plan_expert_cache(self.path, memory.headroom_bytes(self.mem_limit))
        if cache_plan and not cache_plan["fits"]:
            raise MemoryLimitError(
                f"even streamed, this GGUF MoE needs about {cache_plan['minimum_gb']:.1f} GiB of current RAM "
                f"headroom for its fixed weights, {cache_plan['top_k']} routed experts and runtime reserve; "
                f"{cache_plan['budget_gb']:.1f} GiB is safely available")
        if cache_plan:
            # Lazy GGUF experts pay the quant-block decode/repack cost on every miss. When the complete decoded
            # set fits the current safe headroom, eager residency is both faster and simpler; SSD streaming is an
            # admission fallback for a model that otherwise cannot start, not a default tax on small MoEs.
            stream_experts = cache_plan["capacity"] < cache_plan["experts"]
        packed_root = os.path.join(os.path.expanduser(self.disk_cache_dir), "gguf-packed") \
            if self.disk_cache_dir else os.path.expanduser("~/.cache/ai-llm-engine/gguf-packed")
        packed_gb = self.disk_cache_gb
        if isinstance(packed_gb, str) and packed_gb.strip().lower() == "auto":
            os.makedirs(packed_root, exist_ok=True)
            packed_gb = max(1.0, shutil.disk_usage(packed_root).free * 0.10 / GiB)
        elif isinstance(packed_gb, str):
            packed_gb = float(packed_gb)
        model, cfg, gf = dense.build(self.path, stream_experts=stream_experts,
                                     expert_cache=cache_plan and cache_plan["capacity"],
                                     packed_cache_dir=packed_root,
                                     packed_cache_bytes=max(0, int(float(packed_gb) * GiB)))
        if cache_plan:
            if stream_experts:
                self.log(f"gguf moe: streaming {cache_plan['capacity']} of {cache_plan['experts']} experts per "
                         f"projection (~{cache_plan['resident_gb']:.2f} GiB weights within "
                         f"{cache_plan['budget_gb']:.2f} GiB current RAM headroom)")
                self.log(f"gguf moe: cache is smaller than the full expert set; source traffic and evictions "
                         f"are reported in /health (one decoded expert across the model is "
                         f"~{cache_plan['per_expert_mb']:.1f} MiB)")
                if float(packed_gb) > 0:
                    self.log(f"gguf moe: transformed experts persist in a bounded {float(packed_gb):g} GiB "
                             f"packed cache at {packed_root}; GGUF fingerprint and CRC are checked before reuse")
            else:
                self.log(f"gguf moe: all {cache_plan['experts']} experts fit safe RAM; loading them resident "
                         f"(~{cache_plan['resident_gb']:.2f} GiB weights within "
                         f"{cache_plan['budget_gb']:.2f} GiB current RAM headroom)")
        self.tok = gguf_tokenizer.GGUFTokenizer(self.path, gf.meta)
        if self.steer:
            file, rec = steering.resolve(self.steer[0])
            alpha = self.steer[1] if self.steer[1] is not None else (rec if rec is not None else 1.0)
            d = steering.load_direction(file, cfg.n_embd)
            model.steer = steering.projector(d, alpha)
            self.log(f"activation steering on: {file.rsplit('/', 1)[-1]}, alpha {alpha:g}")
        if self.weight_patch:
            from . import weightpatch
            patch_dir, patch_alpha = self.weight_patch
            patch_alpha = 1.0 if patch_alpha is None else patch_alpha
            manifest = weightpatch.apply_patch_dir(model, "gguf", patch_dir, patch_alpha)
            self.log(f"weight patch on: {', '.join(manifest['tensors'])}, alpha {patch_alpha:g} — "
                     f"{weightpatch.describe(manifest['applied'])}")
        self.model = model
        self.model_bytes = model.nbytes
        self.decoder = _Decoder(self.tok)
        self.log(f"loaded {self.path.rsplit('/', 1)[-1]} ({cfg.arch}, {cfg.n_layer} layers) on the MLX decoder: "
                 f"{self.model_bytes / GiB:.2f} GiB, {time.time() - t0:.1f}s")

    def moe_stats(self):
        """Cache and source traffic for GGUF experts streamed by the built-in decoder."""
        return self.model.moe_stats() if self.model is not None else None

    def sleep(self):
        if not self.sleeping and self.model is not None:
            self.model.close()
        super().sleep()
