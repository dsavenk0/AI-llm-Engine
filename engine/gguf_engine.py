"""A GGUF model run by this engine's own MLX decoder (engine/dense.py) — no llama.cpp involved.

It plugs into the same server, prompt cache, stop handling, reasoning budget and memory watchdog as the
mlx-lm backend (engine/compat.py) by replacing only what that class takes from mlx-lm: the model, the
tokenizer and the cache. Which files qualify is decided by `engine.gguf.native_support`; everything
else keeps running on llama.cpp.
"""
import time

from . import gguf
from .compat import CompatEngine, _Decoder
from .generate import DRAFT_LEVELS, GiB


def _trim(cache, n):
    return [c.trim(n) for c in cache][0] if cache else 0


class GGUFEngine(CompatEngine):
    backend = "gguf-mlx"
    REQUIRES_MLX_LM = False

    def __init__(self, path, steer=None, speculative=True, draft_max=15, ngram_min=2, **kw):
        self.steer = steer              # (direction file or folder, alpha or None), see engine/steering.py
        super().__init__(path, **kw)
        # prompt-lookup speculation: the key/value cache can be trimmed, so rejected drafts are simply forgotten
        self.speculative = bool(speculative)
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
        t0 = time.time()
        model, cfg, gf = dense.build(self.path)
        self.tok = gguf_tokenizer.GGUFTokenizer(self.path, gf.meta)
        if self.steer:
            file, rec = steering.resolve(self.steer[0])
            alpha = self.steer[1] if self.steer[1] is not None else (rec if rec is not None else 1.0)
            d = steering.load_direction(file, cfg.n_embd)
            model.steer = steering.projector(d, alpha)
            self.log(f"activation steering on: {file.rsplit('/', 1)[-1]}, alpha {alpha:g}")
        self.model = model
        self.model_bytes = model.nbytes
        self.decoder = _Decoder(self.tok)
        self.log(f"loaded {self.path.rsplit('/', 1)[-1]} ({cfg.arch}, {cfg.n_layer} layers) on the MLX decoder: "
                 f"{self.model_bytes / GiB:.2f} GiB, {time.time() - t0:.1f}s")
