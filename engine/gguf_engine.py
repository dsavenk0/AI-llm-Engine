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
from .compat import CompatEngine, _Decoder
from .generate import DRAFT_LEVELS, GiB, MemoryLimitError


def _trim(cache, n):
    return [c.trim(n) for c in cache][0] if cache else 0


class GGUFEngine(CompatEngine):
    backend = "gguf-mlx"
    REQUIRES_MLX_LM = False

    def __init__(self, path, steer=None, weight_patch=None, speculative=True, draft_max=15, ngram_min=2, **kw):
        self.steer = steer              # (direction file or folder, alpha or None), see engine/steering.py
        self.weight_patch = weight_patch  # (candidate.patch folder, alpha or None), see engine/weightpatch.py
        super().__init__(path, **kw)
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
