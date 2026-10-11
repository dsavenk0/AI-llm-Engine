# Backend for every MLX architecture the native engine doesn't implement (GLM, Llama,
# Gemma, Mistral, Qwen2/3, MoE variants, ...), built on the optional `mlx-lm` package.
#
# The native engine (generate.py / model.py) is a specialist: Qwen3.5-hybrid with Hadamard
# rotation, fused projections, speculative decoding. Nobody reimplements 130 architectures
# alone, and mlx-lm already has them — so for those this class takes the model, tokenizer,
# chat template and tool-call parser from mlx-lm and keeps everything else ours:
#   * the same Engine interface, so server.py, the RAM guard, /health, /v1/stats, LLMBar
#     and the CLI treat both backends identically;
#   * our own decode loop (not mlx_lm.generate_step) — it's what lets the reasoning budget,
#     repetition guard, stop strings, cancellation and the memory guard work here too;
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
from .generate import (ContextError, GiB, MemoryLimitError, Sampler, is_repeating, sampling_presets,
                       summarize)  # noqa: F401  (summarize is re-exported for the server)
from .tokenizer import StreamDecoder

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


NATIVE_TYPES = ("qwen3_5", "qwen3_5_text")


def pick_backend(path, requested="auto"):
    """"native" for the Qwen3.5-hybrid dense family (incl. its PTQ1 repack); "mlxlm" for the rest."""
    if requested in ("native", "mlxlm"):
        return requested
    cfg = _json(os.path.join(path, "config.json"))
    tc = cfg.get("text_config") or {}
    types = {cfg.get("model_type"), tc.get("model_type")}
    if (types & set(NATIVE_TYPES) or str(cfg.get("schema", "")).startswith("prism_hadamard_qwen35_ptq1")) \
            and not (tc.get("num_experts") or cfg.get("num_experts")):
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


class CompatEngine:
    backend = "mlx-lm"

    def __init__(self, path, vision=False, max_context=0, max_tokens=8192, prefill_step=512,
                 thinking=None, effort="medium", mem_limit=0.88, sampling_override=None,
                 cache_limit_mb=128, think_frac=0.6, log=print, **_ignored):
        if not mlx_lm_available():
            raise RuntimeError(INSTALL_HINT)
        self.path, self.log = path, log
        self.max_tokens, self.prefill_step = max_tokens, prefill_step
        self.effort, self.mem_limit, self.think_frac = effort, mem_limit, think_frac
        self.config = _json(os.path.join(path, "config.json"))
        self.gen_cfg = _json(os.path.join(path, "generation_config.json"))
        self.vision_allowed = False      # text only; mlx-vlm would be a separate optional backend
        self.speculative = False
        self.store = None                # no SSD prefix cache on this backend
        self.wired_limit_advice_mb = None
        mx.set_cache_limit(int(cache_limit_mb) * 2**20)

        self.sleeping, self.last_used = False, time.time()
        self.lock = threading.Lock()
        self.model = self.tok = None
        self._load()

        tpl = self._template_source()
        self.supports_thinking = bool(self.tok.has_thinking or "enable_thinking" in tpl)
        # Reasons by default unless the template itself pre-fills an empty, closed think block
        # (Qwen3.5-0.8B does; Qwen3 leaves `<think>` to the model, so it thinks unless told not to).
        self.think_tags = self._think_tags(tpl)
        default_prompt = self._render([{"role": "user", "content": "hi"}], None, {})
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
        self.vocab = int((self.config.get("text_config") or self.config).get("vocab_size") or 0)

        tc = self.config.get("text_config") or self.config
        native = tc.get("max_position_embeddings") or tc.get("max_seq_len") or tc.get("n_positions") or 32768
        self.kv_per_token = kv_bytes_per_token(self.config)
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
    def _load(self):
        from mlx_lm import load
        from mlx.utils import tree_flatten
        t0 = time.time()
        self.model, self.tok = load(self.path)
        self.model_bytes = sum(v.nbytes for _, v in tree_flatten(self.model.parameters()))
        self.decoder = _Decoder(self.tok)
        self.log(f"loaded {os.path.basename(self.path.rstrip('/'))}: {self.model_bytes / GiB:.2f} GiB, "
                 f"{time.time() - t0:.1f}s")

    def sleep(self):
        if self.sleeping:
            return
        self.model, self.cache, self.cache_tokens = None, None, []
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
        self.sleeping = False
        self.log(f"wake: model reloaded in {time.time() - t0:.1f}s")

    def persist_prefix(self):
        pass

    # ───────────────────────── prompt ─────────────────────────
    def _template_source(self):
        tpl = getattr(self.tok, "chat_template", None)
        if isinstance(tpl, dict):
            tpl = " ".join(str(v) for v in tpl.values())
        return tpl or ""

    def _render(self, messages, tools, kw):
        return self.tok.apply_chat_template(messages, tools=tools or None, add_generation_prompt=True,
                                            tokenize=False, **kw)

    def _think_tags(self, tpl):
        """(start, end) of this model's reasoning block: from its tokenizer when mlx-lm detected
        one, else the Qwen-style tags when the template speaks of them (mlx-lm misses some)."""
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
        if memory.used_fraction() <= self.mem_limit:
            return True
        mx.clear_cache()
        if memory.used_fraction() <= self.mem_limit + 0.02:
            return True
        if stage == "prefill":
            raise MemoryLimitError(f"system memory reached {memory.used_fraction():.0%} during prefill "
                                   f"(limit {self.mem_limit:.0%}); request stopped to keep the Mac responsive")
        return False

    def _reserve(self, new_tokens):
        need = new_tokens * self.kv_per_token + 64 * 2**20 + min(new_tokens, self.prefill_step) * 512 * 1024
        mx.clear_cache()
        if memory.headroom_bytes(min(1.0, self.mem_limit + 0.02)) < need:
            raise MemoryLimitError(
                f"not enough free RAM: {memory.used_fraction():.0%} of system memory in use, limit "
                f"{self.mem_limit:.0%}; the prompt needs ~{need / GiB:.1f} GB more. Close other apps or "
                f"shorten the conversation.")

    # ───────────────────────── generate ─────────────────────────
    def _prefix_cache(self, ids):
        """A prompt cache holding ids[:n] for the largest reusable n (< len(ids)); returns (cache, n)."""
        from mlx_lm.models.cache import can_trim_prompt_cache, make_prompt_cache, trim_prompt_cache
        n = 0
        if self.cache is not None and self.cache_tokens:
            m = min(len(self.cache_tokens), len(ids) - 1)
            while n < m and self.cache_tokens[n] == ids[n]:
                n += 1
        if n > 0 and can_trim_prompt_cache(self.cache):
            extra = len(self.cache_tokens) - n
            if extra:
                trim_prompt_cache(self.cache, extra)
            return self.cache, n
        self.cache, self.cache_tokens = None, []
        mx.clear_cache()
        return make_prompt_cache(self.model), 0

    def generate(self, ids, media=None, max_tokens=None, sampling=None, stop=None, should_stop=None,
                 seed=None, thinking_open=False):
        """Yield (token_id, text) pairs; the final item is a stats dict (same shape as Engine's)."""
        self.ensure_awake()
        if len(ids) >= self.max_context:
            raise ContextError(f"prompt is {len(ids)} tokens, context limit is {self.max_context}")
        max_tokens = min(max_tokens or self.max_tokens, self.max_tokens, self.max_context - len(ids))
        vocab = self.vocab or int(getattr(getattr(self.model, "args", None), "vocab_size", 0)) or 262144
        sampler = Sampler(vocab, seed=seed, **(sampling or {}))
        t0 = time.time()

        cache, reused = self._prefix_cache(ids)
        self._reserve(len(ids) - reused)
        self.cache, self.cache_tokens = cache, list(ids[:reused])
        model, step = self.model, max(64, self.prefill_step)
        pos = reused
        while len(ids) - pos > 1:   # all but the last token: build the cache
            chunk = ids[pos:min(len(ids) - 1, pos + step)]
            model(mx.array([chunk]), cache=cache)
            mx.eval([c.state for c in cache])
            pos += len(chunk)
            self.cache_tokens = list(ids[:pos])
            self._check_memory("prefill")
        logits = model(mx.array([ids[pos:]]), cache=cache)
        self.cache_tokens = list(ids)
        t_prefill = time.time() - t0

        budget = int(max_tokens * self.think_frac) if thinking_open and self.think_frac else None
        in_think, forced = budget is not None, None
        think_end = self.think_tags[1] if self.think_tags else None
        closing = self.tok.encode(f"\n{think_end}\n\n", add_special_tokens=False) if think_end else []

        dec = StreamDecoder(self.decoder)
        out_tokens, text, finish = [], "", "length"
        t1 = time.time()
        fed = list(ids)

        def emit(tok):
            nonlocal text, finish
            out_tokens.append(tok)
            if tok in self.eos:
                finish = "stop"
                return True, None
            piece = dec.add(tok)
            text += piece
            if stop and any(s in text for s in stop):
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

        def think_check():
            nonlocal in_think, forced
            if not in_think:
                return False
            if think_end and think_end in text:
                in_think = False
                return False
            if budget and len(out_tokens) >= budget:
                forced = "budget"
                return True
            if is_repeating(text[-700:]):
                forced = "repetition"
                return True
            return False

        y = sampler(logits)
        mx.async_eval(y)
        for _ in range(max_tokens):
            nxt = sampler(model(y.reshape(1, 1), cache=cache))
            mx.async_eval(nxt)
            tok = y.item()
            fed.append(tok)
            done, piece = emit(tok)
            if piece:
                yield tok, piece
            if done:
                break
            if closing and think_check():
                in_think = False
                lg, stopped = None, False
                for cid in closing:   # teacher-force the close so the model sees reasoning ended
                    lg = model(mx.array([[cid]]), cache=cache)
                    fed.append(cid)
                    d2, p2 = emit(cid)
                    if p2:
                        yield cid, p2
                    if d2:
                        stopped = True
                        break
                if stopped:
                    break
                y = sampler(lg)
                mx.async_eval(y)
            else:
                y = nxt
        tail = dec.flush()
        if tail and finish == "length":
            yield None, tail
        mx.eval([c.state for c in cache])
        self.cache_tokens = fed[:-1] if len(fed) > len(ids) else fed   # the last sampled token isn't in the cache
        t_dec = time.time() - t1
        n_out = len(out_tokens)
        st = {
            "prompt_tokens": len(ids), "cached_tokens": reused, "completion_tokens": n_out,
            "prefill_s": round(t_prefill, 3), "decode_s": round(t_dec, 3),
            "prefill_tps": round((len(ids) - reused) / t_prefill, 1) if t_prefill > 0 else 0,
            "decode_tps": round(n_out / t_dec, 2) if t_dec > 0 else 0,
            "finish_reason": finish, "think_forced": forced, "peak_gb": round(mx.get_peak_memory() / GiB, 2),
            "draft_accept": None, "verifies": 0, "ram_used": round(memory.used_fraction(), 3),
        }
        self.stats["requests"] += 1
        self.stats["last"] = st
        T_ = self.totals
        T_["requests"] += 1
        for k in ("prompt_tokens", "cached_tokens", "completion_tokens", "prefill_s", "decode_s"):
            T_[k] += st[k]
        self.history.append({"t": round(time.time()), "prompt": st["prompt_tokens"], "cached": st["cached_tokens"],
                             "out": st["completion_tokens"], "prefill_tps": st["prefill_tps"],
                             "decode_tps": st["decode_tps"], "accept": None})
        mx.clear_cache()
        yield st
