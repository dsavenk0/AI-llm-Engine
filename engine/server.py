# OpenAI-compatible HTTP server (stdlib only).
#
#   python -m engine.server --model PATH [--port 8090] [--context 32768] ...
#
# Endpoints: GET /v1/models, GET /health, POST /v1/chat/completions
# (streaming SSE, reasoning_content, XML tool calls -> OpenAI tool_calls).
# max_tokens above the server limit is clamped, never rejected.
import argparse
import http.server
import json
import os
import queue
import re
import signal
import socketserver
import sys
import threading
import time
import uuid

ENGINE = None
ARGS = None
JOBS = queue.Queue()
_DONE = object()


def worker(init):
    """All MLX work runs on this one thread: MLX GPU streams are thread-local.

    The SSD prefix cache is persisted on sleep (Engine.sleep) and on a clean shutdown
    (_shutdown, below) — not after every request. A long conversation's snapshot point
    grows every turn, so saving it each time would mean writing the whole growing
    history to disk on every single message; saving only when the in-memory state is
    actually about to go away gets the same "resume without re-prefilling" benefit for
    a small fraction of the SSD writes.
    """
    global ENGINE
    ENGINE = init()
    READY.set()
    # how often an idle engine looks at the clock: 20 s, or a quarter of a very short idle-sleep
    poll = 20.0 if ARGS.idle_sleep <= 0 else max(0.2, min(20.0, ARGS.idle_sleep * 60 / 4))
    while True:
        try:
            fn, out = JOBS.get(timeout=poll)
        except queue.Empty:
            idle = time.time() - ENGINE.last_used
            if ARGS.idle_sleep > 0 and not ENGINE.sleeping and idle > ARGS.idle_sleep * 60:
                ENGINE.sleep()
            continue
        try:
            for item in fn():
                out.put(item)
        except BaseException as e:  # forwarded to the HTTP thread
            out.put(e)
        finally:
            out.put(_DONE)
        ENGINE.last_used = time.time()


def run_job(fn):
    """Run generator `fn` on the worker thread, yielding its items here."""
    out = queue.Queue()
    JOBS.put((fn, out))
    while True:
        item = out.get()
        if item is _DONE:
            return
        if isinstance(item, BaseException):
            raise item
        yield item


READY = threading.Event()


_log = None


def log(msg, level=None, group=None):
    """Written through engine.logs: a level (--log-level) and a group tag per line."""
    global _log
    if _log is None:
        from engine import logs
        _log = logs.make(getattr(ARGS, "log_level", "info") if ARGS is not None else "info")
    _log(msg, level, group)


def model_matches(requested):
    """Does a request's `model` name the model this server has loaded? Names are compared the way clients
    spell them: case-insensitively, with or without the "org/" prefix of a hub id."""
    if not requested:
        return True
    norm = lambda s: re.sub(r"[^a-z0-9._-]+", "-", str(s).lower()).strip("-")
    want = {norm(ARGS.name), norm(os.path.basename(ARGS.model.rstrip("/")))}
    got = norm(requested)
    return got in want or norm(str(requested).split("/")[-1]) in want


# ───────────────────────────── output parsing ─────────────────────────────
class OutputParser:
    """Split streamed text into reasoning / content and hold back tool calls."""

    THINK_END, TOOL = "</think>", "<tool_call>"

    def __init__(self, thinking, tools_enabled, think_end=None, tool_start=None, tool_end=None):
        # a family's markers (GLM, Llama, Qwen2...) replace Qwen's; None = "this model has none"
        self.THINK_END = think_end if think_end is not None else self.THINK_END
        self.TOOL = tool_start if tool_start is not None else self.TOOL
        self.mode = "reasoning" if thinking and self.THINK_END else "content"
        self.tools_enabled = tools_enabled and bool(self.TOOL)
        self.buf = ""
        self.tool_text = None
        self.content_started = False

    def _partial(self, s, tag):
        for n in range(min(len(tag) - 1, len(s)), 0, -1):
            if s.endswith(tag[:n]):
                return n
        return 0

    def feed(self, text):
        out = []
        self.buf += text
        while self.buf:
            if self.tool_text is not None:
                self.tool_text += self.buf
                self.buf = ""
                break
            if self.mode == "reasoning":
                i = self.buf.find(self.THINK_END)
                if i >= 0:
                    if i: out.append(("reasoning", self.buf[:i]))
                    self.buf = self.buf[i + len(self.THINK_END):].lstrip("\n")
                    self.mode = "content"
                    continue
                keep = self._partial(self.buf, self.THINK_END)
                emit = self.buf[:len(self.buf) - keep]
                if emit: out.append(("reasoning", emit))
                self.buf = self.buf[len(self.buf) - keep:]
                break
            # content
            if not self.content_started:
                self.buf = self.buf.lstrip("\n")
                if not self.buf: break
                self.content_started = True
            if self.tools_enabled:
                i = self.buf.find(self.TOOL)
                if i >= 0:
                    if i: out.append(("content", self.buf[:i]))
                    self.tool_text = self.buf[i:]
                    self.buf = ""
                    break
                keep = self._partial(self.buf, self.TOOL)
            else:
                keep = 0
            emit = self.buf[:len(self.buf) - keep]
            if emit: out.append(("content", emit))
            self.buf = self.buf[len(self.buf) - keep:]
            break
        return out

    def finish(self):
        out = []
        if self.buf:
            out.append((self.mode, self.buf))
            self.buf = ""
        return out


FUNC_RE = re.compile(r"<function=([^>\n]+)>(.*?)(?:</function>|$)", re.S)
PARAM_RE = re.compile(r"<parameter=([^>\n]+)>\n?(.*?)\n?(?:</parameter>|(?=<parameter=)|$)", re.S)


def parse_tool_calls(text, tools):
    schemas = {}
    for t in tools or []:
        f = t.get("function", t)
        schemas[f.get("name")] = (f.get("parameters") or {}).get("properties", {})
    calls = []
    for name, body in FUNC_RE.findall(text or ""):
        name = name.strip()
        props = schemas.get(name, {})
        args = {}
        for p, raw in PARAM_RE.findall(body):
            p = p.strip()
            typ = (props.get(p) or {}).get("type")
            val = raw
            if typ and typ != "string":
                try:
                    val = json.loads(raw)
                except ValueError:
                    val = raw
            elif not typ and raw.strip()[:1] in "[{":
                try:
                    val = json.loads(raw)
                except ValueError:
                    pass
            args[p] = val
        calls.append({"id": "call_" + uuid.uuid4().hex[:24], "type": "function",
                      "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}})
    return calls


# ───────────────────────────── request handling ─────────────────────────────
EFFORT_MAP = {"minimal": "low", "low": "low", "medium": "medium", "high": "xhigh", "xhigh": "xhigh"}


def normalize_messages(messages):
    out = []
    for m in messages:
        m = dict(m)
        if m.get("role") == "developer":
            m["role"] = "system"
        if m.get("role") == "assistant":
            calls = []
            for tc in m.get("tool_calls") or []:
                tc = json.loads(json.dumps(tc))
                fn = tc.get("function", tc)
                if isinstance(fn.get("arguments"), str):
                    try:
                        fn["arguments"] = json.loads(fn["arguments"] or "{}")
                    except ValueError:
                        fn["arguments"] = {"_raw": fn["arguments"]}
                calls.append(tc)
            if calls:
                m["tool_calls"] = calls
            if m.get("reasoning") and not m.get("reasoning_content"):
                m["reasoning_content"] = m["reasoning"]
        if isinstance(m.get("content"), list):
            items = []
            for it in m["content"]:
                if isinstance(it, dict) and it.get("type") in ("text", "input_text"):
                    items.append({"type": "text", "text": it.get("text", "")})
                else:
                    items.append(it)
            # collapse text-only lists to a string (keeps prompt-cache prefixes stable)
            if all(i.get("type") == "text" for i in items if isinstance(i, dict)):
                m["content"] = "".join(i.get("text", "") for i in items)
            else:
                m["content"] = items
        out.append(m)
    return out


def resolve_options(body):
    kw = dict(body.get("chat_template_kwargs") or {})
    thinking = kw.pop("enable_thinking", body.get("enable_thinking"))
    if isinstance(body.get("thinking"), dict):
        thinking = body["thinking"].get("type") != "disabled"
    effort = kw.pop("reasoning_effort", None) or body.get("reasoning_effort") \
        or (body.get("reasoning") or {}).get("effort")
    effort = EFFORT_MAP.get(effort, effort) if effort else None
    if effort == "none":
        thinking, effort = False, None
    return thinking, effort, kw


class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _json(self, code, obj):
        data = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(data)

    def _error(self, code, msg, typ="invalid_request_error", err_code=None):
        self._json(code, {"error": {"message": msg, "type": typ, "code": err_code}})

    def _auth(self):
        if not ARGS.api_key:
            return True
        if self.headers.get("Authorization", "") == f"Bearer {ARGS.api_key}":
            return True
        self._error(401, "invalid api key", "authentication_error")
        return False

    def do_OPTIONS(self):
        self.send_response(204)
        for k, v in (("Access-Control-Allow-Origin", "*"), ("Access-Control-Allow-Headers", "*"),
                     ("Access-Control-Allow-Methods", "GET, POST, OPTIONS"), ("Content-Length", "0")):
            self.send_header(k, v)
        self.end_headers()

    def do_GET(self):
        path = self.path.split("?")[0].rstrip("/")
        if path in ("/health", ""):
            import mlx.core as mx
            return self._json(200, {
                "status": "ok", "model": ARGS.name, "model_path": ARGS.model,
                "state": "sleeping" if ENGINE.sleeping else "awake",
                "idle_sleep_min": ARGS.idle_sleep,
                "disk_cache": {"entries": len(ENGINE.store.entries), "gb": round(ENGINE.store.total_bytes / 2**30, 2)}
                if ENGINE.store else None,
                "context": ENGINE.max_context, "max_tokens": ENGINE.max_tokens,
                "thinking": ENGINE.thinking, "thinking_default": ENGINE.thinking_default, "effort": ENGINE.effort,
                # the model is None while the engine sleeps — /health must still answer (the app polls it)
                "vision": ("loaded" if ENGINE.model is not None and "vision_tower" in ENGINE.model else "on demand")
                if ENGINE.vision_allowed else False,
                "busy": ENGINE.lock.locked(),
                "progress": getattr(ENGINE, "progress", None),
                "memory_gb": {"active": round(mx.get_active_memory() / 2**30, 2),
                              "peak": round(mx.get_peak_memory() / 2**30, 2),
                              "cache": round(mx.get_cache_memory() / 2**30, 2),
                              "model": round(ENGINE.model_bytes / 2**30, 2)},
                "system_ram_used": round(__import__("engine.memory", fromlist=["x"]).used_fraction(), 3),
                "mem_limit": ENGINE.mem_limit,
                "wired_limit_advice_mb": ENGINE.wired_limit_advice_mb,
                "memory_breakdown": ENGINE.memory_breakdown(),
                "cached_tokens": len(ENGINE.cache_tokens),
                "requests": ENGINE.stats["requests"], "last": ENGINE.stats["last"],
            })
        if not self._auth():
            return
        if path == "/v1/stats":
            from engine.generate import summarize
            return self._json(200, {"session": summarize(ENGINE.totals), "history": list(ENGINE.history),
                                    "all_time": summarize(all_time_stats())})
        if path == "/v1/models" or path.startswith("/v1/models/"):
            card = {"id": ARGS.name, "object": "model", "created": int(START), "owned_by": "local",
                    "context_length": ENGINE.max_context, "max_output_tokens": ENGINE.max_tokens}
            return self._json(200, card if path.startswith("/v1/models/") else {"object": "list", "data": [card]})
        self._error(404, f"unknown path {self.path}")

    def do_POST(self):
        if not self._auth():
            return
        path = self.path.split("?")[0].rstrip("/")
        try:
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(n) or b"{}")
        except ValueError:
            return self._error(400, "invalid JSON body")
        if path in ("/v1/chat/completions", "/chat/completions"):
            return self.chat(body)
        self._error(404, f"unknown path {self.path}")

    # ───────────────────────── chat ─────────────────────────
    def chat(self, body):
        from engine.generate import ContextError, MemoryLimitError
        from engine.tokenizer import TemplateError
        messages = body.get("messages") or []
        if not messages:
            return self._error(400, "messages is required")
        if ARGS.fallback_model == "off" and not model_matches(body.get("model")):
            return self._error(404, f"model {body.get('model')!r} is not loaded; the loaded model is "
                                    f"{ARGS.name!r} (fallback is off)", "invalid_request_error", "model_not_found")
        tools = body.get("tools") if body.get("tool_choice") != "none" else None
        thinking, effort, extra = resolve_options(body)
        log(f"trace: request model={body.get('model')!r} messages={len(messages)} tools={len(tools or [])} "
            f"stream={bool(body.get('stream'))} max_tokens={body.get('max_tokens') or body.get('max_completion_tokens')}")
        try:
            text, images, thinking_open = ENGINE.build_prompt(normalize_messages(messages), tools, thinking, effort, extra)
            ids, media = next(run_job(lambda: iter([ENGINE.encode(text, images)])))
        except (TemplateError, ValueError, OSError) as e:
            return self._error(400, f"prompt error: {e}")
        except MemoryLimitError as e:   # e.g. no room to load the vision tower: a clean answer, not a dropped connection
            log(f"memory: {e}")
            return self._error(503, str(e), "server_error", "insufficient_memory")
        except Exception as e:
            log(f"prompt encoding error: {e!r}")
            return self._error(500, f"prompt error: {e}", "server_error")
        if len(ids) >= ENGINE.max_context:
            return self._error(400, f"This model's maximum context length is {ENGINE.max_context} tokens, "
                                    f"but the prompt is {len(ids)} tokens.", err_code="context_length_exceeded")

        sampling = dict(ENGINE.sampling[bool(thinking_open)] or {})
        for k in ("temperature", "top_p", "top_k", "min_p", "presence_penalty", "repetition_penalty"):
            if body.get(k) is not None:
                sampling[k] = body[k]
        sampling = {k: v for k, v in sampling.items()
                    if k in ("temperature", "top_p", "top_k", "min_p", "presence_penalty", "repetition_penalty")}
        max_tokens = body.get("max_completion_tokens") or body.get("max_tokens")
        stop = body.get("stop")
        stop = [stop] if isinstance(stop, str) else (stop or None)
        stream = bool(body.get("stream"))
        include_usage = bool((body.get("stream_options") or {}).get("include_usage"))
        rid = "chatcmpl-" + uuid.uuid4().hex[:24]
        created = int(time.time())
        markers = ENGINE.parser_markers() if hasattr(ENGINE, "parser_markers") else {}
        parser = OutputParser(thinking_open, bool(tools), **markers)

        def chunk(delta, finish=None, usage=None):
            c = {"id": rid, "object": "chat.completion.chunk", "created": created, "model": ARGS.name,
                 "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
            if usage is not None:
                c["usage"] = usage
            return c

        cancelled = threading.Event()
        if stream:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.close_connection = True

        def send(obj):
            try:
                self.wfile.write(b"data: " + json.dumps(obj, ensure_ascii=False).encode() + b"\n\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                cancelled.set()

        reasoning, content, stats = [], [], {}
        with ENGINE.lock:
            if stream:
                send(chunk({"role": "assistant", "content": ""}))
            try:
                gen = lambda: ENGINE.generate(ids, media, max_tokens, sampling, stop, cancelled.is_set,
                                              seed=body.get("seed"), thinking_open=bool(thinking_open))
                for item in run_job(gen):
                    if isinstance(item, dict):
                        stats = item
                        break
                    for kind, s in parser.feed(item[1]):
                        (reasoning if kind == "reasoning" else content).append(s)
                        if stream:
                            send(chunk({"reasoning_content": s} if kind == "reasoning" else {"content": s}))
                for kind, s in parser.finish():
                    (reasoning if kind == "reasoning" else content).append(s)
                    if stream:
                        send(chunk({"reasoning_content": s} if kind == "reasoning" else {"content": s}))
            except MemoryLimitError as e:
                log(f"memory: {e}")
                if stream:
                    send({"error": {"message": str(e), "type": "server_error", "code": "insufficient_memory"}})
                    return
                return self._error(503, str(e), "server_error", "insufficient_memory")
            except ContextError as e:
                if stream:
                    send({"error": {"message": str(e), "code": "context_length_exceeded"}})
                    return
                return self._error(400, str(e), err_code="context_length_exceeded")
            except Exception as e:  # keep the server alive on model errors
                log(f"generation error: {e!r}")
                if stream:
                    send({"error": {"message": str(e), "type": "server_error"}})
                    return
                return self._error(500, str(e), "server_error")

        parse = ENGINE.parse_tool_calls if hasattr(ENGINE, "parse_tool_calls") else parse_tool_calls
        calls = parse(parser.tool_text, tools) if parser.tool_text else []
        finish = "tool_calls" if calls else ("length" if stats.get("finish_reason") == "length" else "stop")
        usage = {"prompt_tokens": stats.get("prompt_tokens", len(ids)),
                 "completion_tokens": stats.get("completion_tokens", 0),
                 "total_tokens": stats.get("prompt_tokens", len(ids)) + stats.get("completion_tokens", 0),
                 "prompt_tokens_details": {"cached_tokens": stats.get("cached_tokens", 0)}}
        record_stats(stats)
        log(f"{len(ids)} prompt ({stats.get('cached_tokens', 0)} cached, {stats.get('prefill_tps', 0)} tok/s) → "
            f"{stats.get('completion_tokens', 0)} out @ {stats.get('decode_tps', 0)} tok/s, "
            f"{finish}{', ' + str(len(calls)) + ' tool calls' if calls else ''}, peak {stats.get('peak_gb')} GB")

        if stream:
            if calls:
                send(chunk({"tool_calls": [dict(c, index=i) for i, c in enumerate(calls)]}))
            send(chunk({}, finish))
            if include_usage:
                last = chunk({}, None, usage)
                last["choices"] = []
                send(last)
            try:
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            except OSError:
                pass
            return
        msg = {"role": "assistant", "content": "".join(content) or (None if calls else "")}
        if reasoning:
            msg["reasoning_content"] = "".join(reasoning)
        if calls:
            msg["tool_calls"] = calls
        self._json(200, {"id": rid, "object": "chat.completion", "created": created, "model": ARGS.name,
                         "choices": [{"index": 0, "message": msg, "finish_reason": finish}], "usage": usage})


def record_stats(st):
    if not ARGS.stats_file or not st:
        return
    row = {"ts": round(time.time()), "model": ARGS.name, "prompt": st.get("prompt_tokens", 0),
           "cached": st.get("cached_tokens", 0), "out": st.get("completion_tokens", 0),
           "prefill_s": st.get("prefill_s", 0), "decode_s": st.get("decode_s", 0),
           "verifies": st.get("verifies", 0), "accept": st.get("draft_accept")}
    try:
        with open(ARGS.stats_file, "a") as f:
            f.write(json.dumps(row) + "\n")
    except OSError:
        pass


def all_time_stats():
    t = {"requests": 0, "prompt_tokens": 0, "cached_tokens": 0, "completion_tokens": 0,
         "prefill_s": 0.0, "decode_s": 0.0, "drafted": 0, "accepted": 0, "since": None}
    try:
        with open(ARGS.stats_file) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                t["requests"] += 1
                t["prompt_tokens"] += r.get("prompt", 0)
                t["cached_tokens"] += r.get("cached", 0)
                t["completion_tokens"] += r.get("out", 0)
                t["prefill_s"] += r.get("prefill_s", 0)
                t["decode_s"] += r.get("decode_s", 0)
                t["since"] = t["since"] or r.get("ts")
    except (OSError, TypeError):
        pass
    return t


class Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


START = time.time()


def _cache_size(s):
    """argparse type for --disk-cache-gb: a GB number, or "auto" for 10% of free disk space."""
    return s if s.strip().lower() == "auto" else float(s)


def main():
    global ENGINE, ARGS
    ap = argparse.ArgumentParser(description="MLX engine for Qwen3.5 / JANG-Hadamard models")
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", default=None, help="model id served in /v1/models")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--context", type=int, default=0, help="max prompt+output tokens (0 = auto from RAM and model size)")
    ap.add_argument("--max-tokens", type=int, default=8192, help="max output tokens (requests above are clamped)")
    ap.add_argument("--prefill-step", type=int, default=1024)
    ap.add_argument("--thinking", choices=("auto", "on", "off"), default="auto",
                    help="reasoning mode when a request doesn't say: auto = whatever the model's own "
                         "chat template does (on for Bonsai-27B, off for the small Qwen3.5 models)")
    ap.add_argument("--effort", default="medium")
    ap.add_argument("--backend", choices=("auto", "native", "mlxlm"), default="auto",
                    help="auto: the native engine for Qwen3.5-hybrid bundles, the optional mlx-lm backend "
                         "for every other MLX architecture")
    ap.add_argument("--ptq1", choices=("on", "off"), default="off",
                    help="experimental: run PTQ1_0 / TQ1_G128 (1.75-bit ternary) bundles on the Metal lookup-table kernel")
    ap.add_argument("--no-vision", action="store_true")
    ap.add_argument("--log-level", choices=("error", "warning", "info", "debug", "trace"), default="info",
                    help="what the engine writes to its log (each line is tagged with a group)")
    ap.add_argument("--fallback-model", choices=("on", "off"), default="on",
                    help="a request that names a model other than the loaded one is answered by the loaded "
                         "model (on) or refused with 404 (off)")
    ap.add_argument("--prefill-guard", choices=("on", "off"), default="on",
                    help="check free memory before a long prompt is prefilled and refuse what would not fit")
    ap.add_argument("--no-speculative", action="store_true", help="disable prompt-lookup speculative decoding")
    ap.add_argument("--stats-file", default="", help="append one JSON line per request (all-time token statistics)")
    ap.add_argument("--idle-sleep", type=float, default=15, help="unload the model after N idle minutes (0 = never)")
    ap.add_argument("--disk-cache-gb", type=_cache_size, default=8,
                    help="SSD prefix cache budget in GB, or \"auto\" for 10%% of free disk space (0 = off)")
    ap.add_argument("--disk-cache-dir", default="", help="where the SSD prefix cache lives (default: "
                    "~/.cache/ai-llm-engine/prefix) — point it at a faster/bigger external SSD if you have one")
    tune = ap.add_argument_group("fine tuning (defaults come from the model's presets)")
    for flag in ("temperature", "top-p", "top-k", "min-p", "presence-penalty", "repetition-penalty"):
        tune.add_argument(f"--{flag}", type=int if flag == "top-k" else float, default=None)
    tune.add_argument("--draft-max", type=int, default=15, help="longest speculative draft (1..15)")
    tune.add_argument("--ngram-min", type=int, default=2, help="shortest n-gram match that starts a draft (1..4)")
    tune.add_argument("--cache-limit-mb", type=int, default=128, help="MLX allocator cache kept between requests")
    tune.add_argument("--think-frac", type=float, default=0.6,
                      help="share of a reply's token budget reasoning may use before it's forced to close "
                           "(0 disables the budget/repetition cutoff)")
    ap.add_argument("--mem-limit", type=float, default=0.88,
                    help="max fraction of system RAM in use; requests are trimmed/stopped above it (default 0.88)")
    ap.add_argument("--max-pixels", type=int, default=1024 * 1024)
    ap.add_argument("--api-key", default=os.environ.get("ENGINE_API_KEY", ""))
    ARGS = ap.parse_args()
    ARGS.name = ARGS.name or os.path.basename(ARGS.model.rstrip("/")).lower()

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from engine.compat import CompatEngine, pick_backend
    from engine.generate import Engine
    backend = pick_backend(ARGS.model, ARGS.backend)
    Backend = Engine if backend == "native" else CompatEngine
    log(f"backend: {'native engine' if backend == 'native' else 'mlx-lm'}")
    init = lambda: Backend(ARGS.model, vision=not ARGS.no_vision, max_context=ARGS.context,
                           max_tokens=ARGS.max_tokens, prefill_step=ARGS.prefill_step, max_pixels=ARGS.max_pixels,
                           thinking={"auto": None, "on": True, "off": False}[ARGS.thinking], effort=ARGS.effort,
                           speculative=not ARGS.no_speculative, mem_limit=ARGS.mem_limit,
                           prefill_guard=ARGS.prefill_guard == "on",
                           sampling_override={"temperature": ARGS.temperature, "top_p": ARGS.top_p, "top_k": ARGS.top_k,
                                              "min_p": ARGS.min_p, "presence_penalty": ARGS.presence_penalty,
                                              "repetition_penalty": ARGS.repetition_penalty},
                           draft_max=ARGS.draft_max, ngram_min=ARGS.ngram_min, cache_limit_mb=ARGS.cache_limit_mb,
                           disk_cache_gb=ARGS.disk_cache_gb, disk_cache_dir=ARGS.disk_cache_dir,
                           think_frac=ARGS.think_frac, ptq1=ARGS.ptq1 == "on",
                           log=log)
    threading.Thread(target=worker, args=(init,), daemon=True).start()
    READY.wait()

    def _persist_job():
        ENGINE.persist_prefix()
        return iter(())  # no streamed output, just run on the worker thread and finish

    def _shutdown(signum, frame):
        # persist_prefix touches MLX arrays, so it has to run on the worker thread like
        # everything else — route it through the same job queue chat requests use, then
        # exit directly (stop/restart already has its own SIGKILL fallback if this hangs).
        try:
            for _ in run_job(_persist_job):
                pass
        except Exception:
            pass
        os._exit(0)

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    srv = Server((ARGS.host, ARGS.port), Handler)
    log(f"serving {ARGS.name} on http://{ARGS.host}:{ARGS.port}/v1  "
        f"(context {ENGINE.max_context}, max output {ENGINE.max_tokens})")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
