# Backend for every MLX architecture the native engine doesn't implement (GLM, Llama,
# Gemma, Mistral, Qwen2/3, MoE variants, ...), built on the optional `mlx-lm` package.
#
# The native engine (generate.py / model.py) is a specialist: Qwen3.5-hybrid with Hadamard
# rotation, fused projections, speculative decoding. Nobody reimplements 130 architectures
# alone, and mlx-lm already has them — so for those this class takes the model, tokenizer,
# chat template and tool-call parser from mlx-lm and keeps everything else ours:
#   * the same Engine interface, so server.py, the RAM guard, /health, /v1/stats, LLMBar
#     and the CLI treat both backends identically;
#   * our own decode loop (not mlx_lm.generate_step): the native engine's DecodeLoop
#     (engine/session.py) through MlxLmRuntime below — so the reasoning budget, repetition guard,
#     stop strings, cancellation, speculation and the memory guard are the same code on both;
#   * prefix reuse through mlx-lm's prompt cache where the cache can be trimmed.
# mlx-lm is imported lazily: without it the native engine stays dependency-light.
import gc
import json
import os
import threading
import time
from collections import deque

import mlx.core as mx

from . import memory
from .generate import (DRAFT_LEVELS, ContextError, GiB, MemoryLimitError, sampling_presets,  # noqa: F401
                       summarize)  # noqa: F401  (summarize is re-exported for the server)
from . import family
from .runtime import MlxHybridRuntime
from .session import DecodeLoop

INSTALL_HINT = ("this model's architecture isn't implemented by the native engine; it runs on the "
                "optional mlx-lm backend. Install it with:  ./llm setup --compat  "
                "(or  .venv/bin/pip install mlx-lm)")
# for a bundle with no generation_config at all; most ship one
GENERIC_SAMPLING = {"temperature": 0.7, "top_p": 0.9, "top_k": 40}


def _json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def pick_backend(path, requested="auto"):
    """"native" for the Qwen3.5-hybrid dense family (incl. its PTQ1 repack); "mlxlm" for the rest."""
    if requested in ("native", "mlxlm", "gguf"):
        return requested
    if os.path.isfile(path) and path.lower().endswith(".gguf"):
        return "gguf"                   # the engine's own decoder (engine/gguf_engine.py)
    cfg = _json(os.path.join(path, "config.json"))
    if family.native_capable(cfg) or str(cfg.get("schema", "")).startswith("prism_hadamard_qwen35_ptq1"):
        return "native"
    return "mlxlm"


def mlx_lm_available():
    try:
        import mlx_lm  # noqa: F401
        return True
    except ImportError:
        return False


def kv_bytes_per_token(cfg):
    """fp16 K+V bytes one token adds across the layers — a rough figure from the config
    (MLA / sliding-window / SSM layers make it an upper bound), used for the context budget."""
    tc = cfg.get("text_config") or cfg
    layers = tc.get("num_hidden_layers") or tc.get("n_layer") or tc.get("num_layers") or 32
    heads = tc.get("num_attention_heads") or tc.get("n_head") or 32
    kv = tc.get("num_key_value_heads") or tc.get("num_kv_heads") or heads
    hd = tc.get("head_dim") or ((tc.get("hidden_size") or 4096) // heads)
    return int(layers * 2 * kv * hd * 2)


class _Decoder:
    """What StreamDecoder needs from a tokenizer: decode() that keeps special tokens, so the
    reasoning / tool-call markers reach the output parser."""

    def __init__(self, tok):
        self.tok = tok

    def decode(self, ids):
        try:
            return self.tok.decode(ids, skip_special_tokens=False)
        except TypeError:
            return self.tok.decode(ids)


class CompatEngine(DecodeLoop):
    backend = "mlx-lm"
    REQUIRES_MLX_LM = True       # a subclass with its own model code (engine/gguf_engine.py) sets this False

    def __init__(self, path, vision=False, max_context=0, max_tokens=8192, prefill_step=512,
                 thinking=None, effort="medium", mem_limit=0.88, sampling_override=None,
                 cache_limit_mb=128, think_frac=0.6, prefill_guard=True, log=print, **_ignored):
        if self.REQUIRES_MLX_LM and not mlx_lm_available():
            raise RuntimeError(INSTALL_HINT)
        self.path, self.log = path, log
        self.max_tokens, self.prefill_step = max_tokens, prefill_step
        self.effort, self.mem_limit, self.think_frac = effort, mem_limit, think_frac
        self.prefill_guard, self.progress = prefill_guard, None
        self.config, self.gen_cfg = self._read_configs(path)
        self.vision_allowed = False      # text only; mlx-vlm would be a separate optional backend
        self.vision_capable = False
        self.speculative = False         # a subclass with a trimmable cache turns it on
        self.draft_levels, self.ngram_min = tuple(sorted({min(l, 15) for l in DRAFT_LEVELS})), 2
        self.store = None                # no SSD prefix cache on this backend
        self.snapshots, self.im_start = [], None     # nor restore points: a trimmable cache reuses any prefix
        self.moe = None                  # the expert pools of a streamed Mixture-of-Experts model (moe_stream.py)
        self.wired_limit_advice_mb = None
        mx.set_cache_limit(int(cache_limit_mb) * 2**20)

        self.sleeping, self.last_used = False, time.time()
        self.lock = threading.Lock()
        self.memory = memory.Manager(self, mx.clear_cache)   # what to give back under pressure, in order
        self.model = self.tok = self.rt = None
        self._load()
        self._attach_runtime()

        tpl = self._template_source()
        # Reasoning is a property of the *template* (a vocabulary that happens to contain <think> says
        # nothing). Reasons by default unless the template itself pre-fills an empty, closed think
        # block (Qwen3.5-0.8B does; Qwen3 leaves `<think>` to the model, so it thinks unless told not to).
        self.think_tags = self._think_tags(tpl)
        self.supports_thinking = bool(self.think_tags or "enable_thinking" in tpl)
        default_prompt = self._render_default([{"role": "user", "content": "hi"}])
        self.thinking_default = bool(self.supports_thinking and not (
            self.think_tags and default_prompt.rstrip().endswith(self.think_tags[1])))
        self.thinking = self.thinking_default if thinking is None else bool(thinking)
        self.efforts = []

        presets, self.sampling_source = sampling_presets({}, self.gen_cfg, fallback=GENERIC_SAMPLING)
        self.sampling = presets
        for mode in self.sampling.values():
            mode.update({k: v for k, v in (sampling_override or {}).items() if v is not None})

        eos = set(getattr(self.tok, "eos_token_ids", None) or [])
        g = self.gen_cfg.get("eos_token_id", self.config.get("eos_token_id", []))
        eos |= set(g if isinstance(g, list) else [g])
        self.eos = {e for e in eos if e is not None}
        self.vocab = (int((self.config.get("text_config") or self.config).get("vocab_size") or 0)
                      or int(getattr(getattr(self.model, "args", None), "vocab_size", 0)) or 262144)

        tc = self.config.get("text_config") or self.config
        native = tc.get("max_position_embeddings") or tc.get("max_seq_len") or tc.get("n_positions") or 32768
        self.kv_per_token = kv_bytes_per_token(self.config)
        self.base_work, self.act_per_token = 64 * 2**20, 512 * 1024      # what memory.reserve budgets with
        if not max_context:
            budget = memory.TOTAL * self.mem_limit - self.model_bytes - 4 * GiB
            max_context = int(max(4096, min(native, budget // max(1, self.kv_per_token) // 4096 * 4096)))
            log(f"context: auto → {max_context} tokens ({memory.TOTAL / GiB:.0f} GB RAM, "
                f"{self.kv_per_token // 1024} KB KV per token, estimated from the config)")
        self.max_context = max_context
        self.max_tokens = min(self.max_tokens, self.max_context // 2)

        self.cache, self.cache_tokens = None, []
        self.stats = {"requests": 0, "last": {}}
        self.totals = {"requests": 0, "prompt_tokens": 0, "cached_tokens": 0, "completion_tokens": 0,
                       "prefill_s": 0.0, "decode_s": 0.0, "drafted": 0, "accepted": 0, "since": time.time()}
        self.history = deque(maxlen=60)
        arch = tc.get("model_type") or self.config.get("model_type")
        log(f"mlx-lm backend: {arch}, {self.model_bytes / GiB:.2f} GiB, thinking="
            f"{'supported' if self.supports_thinking else 'no'}, tool calls="
            f"{'yes' if self.tok.has_tool_calling else 'no'}, sampling from {self.sampling_source}")

    # ───────────────────────── loading / sleep ─────────────────────────
    def _read_configs(self, path):
        return _json(os.path.join(path, "config.json")), _json(os.path.join(path, "generation_config.json"))

    def _cache_api(self):
        """(make_cache(model), can_trim(cache), trim(cache, n)) — mlx-lm's by default."""
        from mlx_lm.models.cache import can_trim_prompt_cache, make_prompt_cache, trim_prompt_cache
        return make_prompt_cache, can_trim_prompt_cache, trim_prompt_cache

    def _load(self):
        from mlx_lm import load
        from mlx.utils import tree_flatten
        from . import moe_stream
        t0 = time.time()
        streamed = ""
        plan = None
        if os.path.isdir(self.path):
            try:
                plan = moe_stream.plan(self.path, self.config, self.mem_limit, memory.TOTAL)
            except moe_stream.StreamError as e:
                raise MemoryLimitError(str(e)) from None
        if plan:
            try:
                self.model, self.tok, self.moe = moe_stream.load(self.path, plan, self.log)
                streamed = (f", experts streamed from disk: {plan['capacity']} of {plan['experts']} per layer "
                            f"resident ({plan['resident_gb']:.1f} of {plan['total_gb']:.1f} GiB)")
                if plan["thrashing"]:
                    self.log(f"moe: a pool of {plan['capacity']} is smaller than the {plan['top_k']} experts one "
                             f"token uses — every token will read from disk")
            except moe_stream.StreamError as e:
                self.log(f"moe: not streaming — {e}; loading the whole model")
                plan = None
        if not plan:
            self.model, self.tok = load(self.path)
        self.model_bytes = sum(v.nbytes for _, v in tree_flatten(self.model.parameters()))
        if self.moe:
            self.model_bytes += sum(p.nbytes for p in self.moe.pools.values())
        self.decoder = _Decoder(self.tok)
        self.log(f"loaded {os.path.basename(self.path.rstrip('/'))}: {self.model_bytes / GiB:.2f} GiB, "
                 f"{time.time() - t0:.1f}s{streamed}")

    def moe_stats(self):
        """Hit rate and traffic of the expert pools while a MoE model streams its experts, else None."""
        return self.moe.stats() if self.moe else None

    def sleep(self):
        if self.sleeping:
            return
        if getattr(self, "moe", None):
            self.moe.close()                 # saves the hot set, so the next load starts warm
            self.moe = None
        self.model, self.rt, self.cache, self.cache_tokens = None, None, None, []
        gc.collect()
        mx.clear_cache()
        self.sleeping = True
        self.log("sleep: model unloaded (idle); it wakes on the next request")

    def ensure_awake(self):
        self.last_used = time.time()
        if not self.sleeping:
            return
        t0 = time.time()
        self._load()
        self._attach_runtime()
        self.sleeping = False
        self.log(f"wake: model reloaded in {time.time() - t0:.1f}s")

    def persist_prefix(self):
        if getattr(self, "moe", None):
            self.moe.save_usage()

    # ───────────────────────── prompt ─────────────────────────
    def _template_source(self):
        tpl = getattr(self.tok, "chat_template", None)
        if isinstance(tpl, dict):
            tpl = " ".join(str(v) for v in tpl.values())
        return tpl or ""

    def _render(self, messages, tools, kw):
        return self.tok.apply_chat_template(messages, tools=tools or None, add_generation_prompt=True,
                                            tokenize=False, **kw)

    def _render_default(self, messages):
        """The generation prompt with enable_thinking left UNDEFINED, which is how the model's authors
        define its default. mlx-lm's wrapper injects enable_thinking=True whenever it can, so go to the
        underlying HF tokenizer for this one."""
        raw = getattr(self.tok, "_tokenizer", None)
        if raw is not None and getattr(self.tok, "_chat_template", None) is None:
            try:
                return raw.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            except Exception:
                pass
        return self._render(messages, None, {})

    def _think_tags(self, tpl):
        """(start, end) of this model's reasoning block: from its tokenizer when mlx-lm detected
        one, else the Qwen-style tags when the template speaks of them (mlx-lm misses some)."""
        if "think" not in tpl.lower():
            return None
        if self.tok.has_thinking:
            return self.tok.think_start, self.tok.think_end
        if "</think>" in tpl and "<think>" in tpl:
            return "<think>", "</think>"
        return None

    def _opens_reasoning(self, text):
        """True when the rendered prompt ends inside an open reasoning block."""
        if not self.think_tags:
            return False
        start, end = self.think_tags
        return text.rfind(start) > text.rfind(end)

    def build_prompt(self, messages, tools=None, thinking=None, effort=None, extra_kwargs=None):
        thinking = self.thinking if thinking is None else thinking
        kw = dict(extra_kwargs or {})
        tpl = self._template_source()
        if self.supports_thinking and "enable_thinking" in tpl:
            kw["enable_thinking"] = bool(thinking)
        eff = effort or self.effort
        if thinking and eff and "reasoning_effort" in tpl:
            kw["reasoning_effort"] = "high" if eff == "xhigh" and "xhigh" not in tpl else eff
        images = []
        flat = []
        for m in messages:
            m = dict(m)
            if isinstance(m.get("content"), list):
                texts = []
                for it in m["content"]:
                    if isinstance(it, dict) and it.get("type") == "text":
                        texts.append(it.get("text", ""))
                    elif isinstance(it, dict):
                        images.append(it)
                m["content"] = "".join(texts)
            flat.append(m)
        text = self._render(flat, tools, kw)
        opens = self._opens_reasoning(text)
        if thinking and self.think_tags and not opens and not text.rstrip().endswith(self.think_tags[1]):
            # the template leaves `<think>` to the model (Qwen3): write it ourselves so the server
            # parses the reasoning and the think budget applies, exactly as the model would begin
            text += self.think_tags[0] + "\n"
            opens = True
        return text, images, opens

    def encode(self, text, images):
        self.ensure_awake()
        if images:
            raise ValueError("this model/server has no vision support")
        return list(self.tok.encode(text, add_special_tokens=False)), None

    # ───────────────────────── output parsing ─────────────────────────
    def parser_markers(self):
        """Marker strings for the server's OutputParser, from this model's own tokenizer."""
        t = self.tok
        return {"think_end": self.think_tags[1] if self.think_tags else None,
                "tool_start": t.tool_call_start if t.has_tool_calling else None,
                "tool_end": t.tool_call_end if t.has_tool_calling else None}

    def parse_tool_calls(self, text, tools):
        """OpenAI tool_calls from the held-back tool text, via mlx-lm's parser for this family."""
        import uuid
        t = self.tok
        if not text or not t.has_tool_calling:
            return []
        start, end = t.tool_call_start, t.tool_call_end
        segments = []
        for part in text.split(start):
            part = part.strip()
            if end and end in part:
                part = part.split(end)[0]
            if part.strip():
                segments.append(part)
        calls = []
        for seg in segments or [text]:
            try:
                parsed = t.tool_parser(seg, tools)
            except Exception as e:  # truncated mid-generation, or not this family's format
                self.log(f"tool call not parsed ({type(e).__name__}: {e})")
                continue
            for tc in (parsed if isinstance(parsed, list) else [parsed]):
                if not isinstance(tc, dict) or not tc.get("name"):
                    continue
                args = tc.get("arguments", {})
                calls.append({"id": tc.get("id") or "call_" + uuid.uuid4().hex[:24], "type": "function",
                              "function": {"name": tc["name"],
                                           "arguments": args if isinstance(args, str)
                                           else json.dumps(args, ensure_ascii=False)}})
        return calls

    # ───────────────────────── memory ─────────────────────────
    def memory_breakdown(self):
        kv = 0
        try:
            kv = sum(getattr(c, "nbytes", 0) for c in list(self.cache or []))
        except Exception:
            pass
        br = {"weights": 0 if self.sleeping else self.model_bytes, "vision": 0, "kv": kv, "ssm": 0, "snapshots": 0}
        active, cached = mx.get_active_memory(), mx.get_cache_memory()
        known = br["weights"] + kv
        br["activations"] = max(0, active - known)
        br["mlx_cache"] = cached
        fp = memory.process_footprint()
        br["process"] = max(0, fp - active - cached) if fp else None
        br["footprint"] = fp
        return br

    def _check_memory(self, stage):
        return memory.check(self, stage, mx.clear_cache)

    def _reserve(self, new_tokens):
        # these backends prefill in one chunk size: no shrinking, the same refusal and grace band
        memory.reserve(self, new_tokens, mx.clear_cache, shrink=False)

    def _fit_output(self, max_tokens):
        return memory.fit_output(self, max_tokens)

    # ───────────────────────── generate (engine/session.py) ─────────────────────────
    # The decode loop is DecodeLoop's — the one the native engine runs. What differs here is the prefix
    # policy (mlx-lm's caches can be trimmed to any length, so any shared prefix is reused), the prefill
    # schedule, and how this tokenizer names the reasoning markers.
    def _attach_runtime(self):
        self.rt = MlxLmRuntime(self.model, self._cache_api())

    def _detokenizer(self):
        return self.decoder

    def _think_closing(self):
        end = self.think_tags[1] if self.think_tags else None
        return end, (list(self.tok.encode(f"\n{end}\n\n", add_special_tokens=False)) if end else [])

    def _restore_or_reset(self, ids):
        """Keep the cache for the longest shared prefix (< len(ids)), trimming what follows it; returns its
        length. Nothing to trim means any cache (a hybrid model's non-trimmable state too) is reusable."""
        rt, n = self.rt, 0
        if self.cache is not None and self.cache_tokens:
            m = min(len(self.cache_tokens), len(ids) - 1)
            while n < m and self.cache_tokens[n] == ids[n]:
                n += 1
        extra = len(self.cache_tokens) - n
        if n > 0 and (extra == 0 or rt.can_trim(self.cache)):
            if extra:
                rt.trim(self.cache, extra)
            self.cache_tokens = list(ids[:n])
            return n
        self.cache, self.cache_tokens = None, []
        rt.release()
        self.cache = rt.make_cache()
        return 0

    def _prefill(self, cache, ids, embeds=None, pos3=None, delta=0, start=0, hidden=False):
        """All but the last token in fixed chunks, each evaluated with the cache it leaves behind; then the
        last token alone, whose logits begin the decode."""
        if embeds is not None or hidden:
            raise ValueError("this model/server has no vision or embedding support")
        n, s, step = len(ids), start, max(64, self.prefill_step)
        try:
            while n - s > 1:
                self.progress = {"phase": "prefill", "done": s, "total": n}
                e = min(n - 1, s + step)
                self.rt.prefill(cache, ids, None, None, 0, s, e)
                s = e
                self.cache_tokens = list(ids[:s])
                self._check_memory("prefill")
        finally:
            self.progress = None
        return self.rt.prefill(cache, ids, None, None, 0, s, n)


class MlxLmRuntime(MlxHybridRuntime):
    """The runtime seam (engine/runtime.py) over an mlx-lm-style model: `model(ids, cache=...)` with the
    prompt caches of `cache_api` — mlx-lm's own, or engine/dense.py's for GGUF files. Sampling and reading
    tokens are the native runtime's; what differs is the call, the cache and how a verify pass is undone."""
    name = "mlx-lm"

    def __init__(self, model, cache_api):
        self.model = model
        self._make, self._can_trim, self._trim = cache_api
        self._width = 0          # tokens in the last verify pass

    def make_cache(self):
        return self._make(self.model)

    def settle(self, cache):
        mx.eval([c.state for c in cache])

    def can_trim(self, cache):
        return self._can_trim(cache)

    def trim(self, cache, n):
        self._trim(cache, n)

    def can_verify(self, cache):
        return self._can_trim(cache)

    def rewind(self, cache, n):
        extra = getattr(cache[0], "offset", n) - n
        if extra > 0:
            self._trim(cache, extra)
        return True

    def prefill(self, cache, ids, embeds, pos3, delta, s, e, hidden=False):
        """Evaluates the cache the chunk leaves; the logits stay lazy (only the last chunk's are read)."""
        logits = self.model(mx.array([ids[s:e]]), cache=cache)
        mx.eval([c.state for c in cache])
        return logits

    def step(self, token, cache, delta):
        ids = mx.array([[token]]) if isinstance(token, int) else token.reshape(1, 1)
        return self.model(ids, cache=cache)

    def verify(self, token, draft, cache, delta):
        self._width = 1 + len(draft)
        return self.model(mx.array([[token] + draft]), cache=cache)

    def commit(self, cache, m):
        if self._width > m:
            self._trim(cache, self._width - m)        # forget the rejected draft tokens

    def embed_media(self, ids, media):
        raise ValueError("this model/server has no vision support")
