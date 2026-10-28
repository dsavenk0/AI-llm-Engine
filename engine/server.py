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


MAX_BODY = 64 * 2**20          # bytes of one request body


class Admission:
    """Caps how many generation requests are in flight at once (running on the engine plus waiting for the
    lock). Over the cap a request is refused with 503 instead of piling up unbounded threads and prompt
    buffers — a coding agent, a chat, the editor and voice all hitting one model stay bounded. 0 = no cap.

    This is the scheduler's admission half. Merging several requests into one forward pass is a separate,
    engine-level change (engine/batch.py proves the lossless batched decode) that is wired on with real-model
    concurrency measurement, since it has to carry speculation, stop strings and the thinking budget too."""

    def __init__(self, limit=0):
        self.limit = limit
        self.n = 0
        self.lock = threading.Lock()

    def acquire(self):
        with self.lock:
            if self.limit and self.n >= self.limit:
                return False
            self.n += 1
            return True

    def release(self):
        with self.lock:
            self.n = max(0, self.n - 1)

    @property
    def inflight(self):
        return self.n


ADMISSION = Admission(0)
GEN_PATHS = ("/v1/chat/completions", "/chat/completions", "/v1/messages", "/api/chat", "/api/generate")


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


def pid_alive(pid):
    """True while a process with this PID exists. A watchdog uses it to notice the app that launched the
    engine has gone away (crashed), so an orphaned engine does not keep the model and the port forever."""
    if not pid or pid <= 0:
        return True                                   # no parent to watch (the CLI daemon runs on its own)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True                                   # it exists, we just may not signal it
    return True


def watch_parent(pid, on_gone, interval=3.0, sleep=time.sleep, alive=pid_alive):
    """Block until `pid` disappears, then call `on_gone()`. Factored out so it is unit-testable."""
    while alive(pid):
        sleep(interval)
    on_gone()


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


from engine.parsers import FUNC_RE, PARAM_RE, parse_xml as parse_tool_calls  # noqa: E402,F401  (kept under its old name)


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


SAMPLING_KEYS = ("temperature", "top_p", "top_k", "min_p", "presence_penalty", "repetition_penalty")


def check_params(body):
    """None, or the sentence that says which request parameter is unusable (so the answer is a 400 before the
    model is woken or a prompt is rendered, not a 500 from deep inside the sampler)."""
    num = lambda v: isinstance(v, (int, float)) and not isinstance(v, bool) and v == v and abs(v) != float("inf")
    for k in SAMPLING_KEYS + ("seed",):
        v = body.get(k)
        if v is not None and not num(v):
            return f"{k} must be a number"
    for k in ("max_tokens", "max_completion_tokens"):
        v = body.get(k)
        if v is not None and not (isinstance(v, int) and not isinstance(v, bool) and v >= 1):
            return f"{k} must be a positive integer"
    stop = body.get("stop")
    if stop is not None and not (isinstance(stop, str) or (isinstance(stop, list) and len(stop) <= 16
                                                           and all(isinstance(x, str) for x in stop))):
        return "stop must be a string or a list of up to 16 strings"
    return None


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


class ChatError(Exception):
    """A chat request that cannot be served. Raised from the wire-agnostic chat core so each wire (OpenAI,
    Ollama) renders the failure in its own shape — as an HTTP error before anything is sent, or inline once a
    stream has started."""

    def __init__(self, status, message, typ="invalid_request_error", code=None):
        super().__init__(message)
        self.status, self.message, self.typ, self.code = status, message, typ, code


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
        import hmac
        bearer = hmac.compare_digest(self.headers.get("Authorization", "").encode(), f"Bearer {ARGS.api_key}".encode())
        xkey = hmac.compare_digest(self.headers.get("x-api-key", "").encode(), ARGS.api_key.encode())  # Anthropic clients
        if bearer or xkey:
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
                "vision_capable": bool(getattr(ENGINE, "vision_capable", False)),
                "parsers": {"reasoning": ARGS.reasoning_parser, "tool_call": ARGS.tool_parser},
                "images": self._images_ready(),
                "efforts": list(getattr(ENGINE, "efforts", []) or []),
                "thinking_supported": bool(getattr(ENGINE, "supports_thinking", True)),
                "steer": {"file": os.path.basename(ARGS.steer.rstrip("/")), "alpha": ARGS.steer_alpha}
                if ARGS.steer else None,
                "busy": ENGINE.lock.locked(),
                "inflight": ADMISSION.inflight, "max_queue": ADMISSION.limit,
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
        if path == "/api/version":                    # Ollama discovery — public, like /health
            from engine import ollama
            return self._json(200, ollama.version())
        if not self._auth():
            return
        if path == "/api/tags":                       # Ollama: the models available here
            from engine import ollama
            return self._json(200, ollama.tags([self._ollama_card()]))
        if path == "/api/ps":                         # Ollama: the models loaded right now
            from engine import ollama
            return self._json(200, ollama.ps([] if ENGINE.sleeping else [self._ollama_card()]))
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
            if n > MAX_BODY:                       # images arrive inline, but not gigabytes: never read that into memory
                return self._error(413, f"request body is larger than {MAX_BODY // 2**20} MB", "invalid_request_error")
            body = json.loads(self.rfile.read(n) or b"{}")
            if not isinstance(body, dict):
                raise ValueError("the body must be a JSON object")
        except ValueError:
            return self._error(400, "invalid JSON body")
        if path in GEN_PATHS:
            if not ADMISSION.acquire():
                return self._error(503, "the server is busy — too many requests in flight; retry shortly",
                                   "server_error", "overloaded")
            try:
                if path in ("/v1/chat/completions", "/chat/completions"):
                    return self.chat(body)
                if path == "/v1/messages":            # Anthropic Messages API (native, not an OpenAI wrapper)
                    return self.messages_anthropic(body)
                if path == "/api/chat":               # Ollama wire format (same loaded model, no adapter)
                    return self.chat_ollama(body)
                return self.generate_ollama(body)     # /api/generate
            finally:
                ADMISSION.release()
        if path == "/api/show":
            from engine import ollama
            return self._json(200, ollama.show(self._ollama_card(), context=ENGINE.max_context,
                                               capabilities=self._capabilities()))
        if path in ("/v1/images/generations", "/images/generations"):
            return self.images(body)
        self._error(404, f"unknown path {self.path}")

    # ───────────────────────── images (optional: mflux) ─────────────────────────
    def _images_ready(self):
        """True when /v1/images/generations would work: switched on and mflux installed."""
        if ARGS.images != "on":
            return False
        from engine import images
        home = os.environ.get("LLM_HOME") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return bool(images.tool_path(home))

    def images(self, body):
        """OpenAI-style /v1/images/generations through mflux. The text model is put to sleep first, so FLUX
        and the language model are never resident together."""
        import base64
        from engine import images
        if ARGS.images != "on":
            return self._error(501, "image generation is switched off (start the engine with --images on)",
                               "invalid_request_error", "images_disabled")
        home = os.environ.get("LLM_HOME") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if not images.tool_path(home):
            return self._error(501, "image generation is not installed — run: ./llm setup --images",
                               "invalid_request_error", "images_not_installed")
        try:
            if int(body.get("n") or 1) != 1:
                return self._error(400, "only n=1 is supported", "invalid_request_error")
            with ENGINE.lock:                  # wait for a running request, then free the language model
                if not ENGINE.sleeping:
                    next(run_job(lambda: iter([ENGINE.sleep()])))
            log("images: generating with mflux (the text model stays unloaded meanwhile)")
            png = images.generate(body.get("prompt"), data_dir=home, model=body.get("model") or images.DEFAULT_MODEL,
                                  size=body.get("size"), steps=body.get("steps"), seed=body.get("seed"))
        except images.ImageError as e:
            return self._error(e.status, str(e))
        except (TypeError, ValueError) as e:
            return self._error(400, f"bad request: {e}")
        self._json(200, {"created": int(time.time()), "data": [{"b64_json": base64.b64encode(png).decode()}]})

    # ───────────────────────── chat ─────────────────────────
    def _chat_core(self, body, cancelled):
        """One chat request, wire-agnostic (OpenAI and Ollama share it — the generation path exists once).
        Takes an OpenAI-shaped body and yields, in order:
            ("ready", rid, created)              — validation + prompt encoding done, nothing sent yet
            ("delta", "reasoning"|"content", s)  — streamed text pieces
            ("result", {...})                    — content/reasoning/calls/finish/stop_reason/usage/rid/created/stats
        and raises ChatError on any failure. `cancelled` is set by the caller when the client hangs up; the
        generation loop stops at the next token."""
        from engine.generate import ContextError, MemoryLimitError
        from engine.tokenizer import TemplateError
        messages = body.get("messages") or []
        if not messages:
            raise ChatError(400, "messages is required")
        if ARGS.fallback_model == "off" and not model_matches(body.get("model")):
            raise ChatError(404, f"model {body.get('model')!r} is not loaded; the loaded model is "
                                 f"{ARGS.name!r} (fallback is off)", "invalid_request_error", "model_not_found")
        tools = body.get("tools") if body.get("tool_choice") != "none" else None
        bad = check_params(body)
        if bad:
            raise ChatError(400, bad, "invalid_request_error")
        thinking, effort, extra = resolve_options(body)
        log(f"trace: request model={body.get('model')!r} messages={len(messages)} tools={len(tools or [])} "
            f"stream={bool(body.get('stream'))} max_tokens={body.get('max_tokens') or body.get('max_completion_tokens')}")
        if not getattr(ENGINE, "vision_allowed", False):      # no vision tower (or switched off): answer anyway
            from engine.media import strip_images
            messages, n_dropped = strip_images(messages)
            if n_dropped:
                log(f"trace: {n_dropped} image(s) dropped, this model does not read images")
        try:
            text, images, thinking_open = ENGINE.build_prompt(normalize_messages(messages), tools, thinking, effort, extra)
            ids, media = next(run_job(lambda: iter([ENGINE.encode(text, images)])))
        except (TemplateError, ValueError, OSError) as e:
            raise ChatError(400, f"prompt error: {e}")
        except MemoryLimitError as e:   # e.g. no room to load the vision tower: a clean answer, not a dropped connection
            log(f"memory: {e}")
            raise ChatError(503, str(e), "server_error", "insufficient_memory")
        except Exception as e:
            log(f"prompt encoding error: {e!r}")
            raise ChatError(500, f"prompt error: {e}", "server_error")
        if len(ids) >= ENGINE.max_context:
            raise ChatError(400, f"This model's maximum context length is {ENGINE.max_context} tokens, "
                                 f"but the prompt is {len(ids)} tokens.", code="context_length_exceeded")

        sampling = dict(ENGINE.sampling[bool(thinking_open)] or {})
        for k in SAMPLING_KEYS:
            if body.get(k) is not None:
                sampling[k] = body[k]
        sampling = {k: v for k, v in sampling.items() if k in SAMPLING_KEYS}
        max_tokens = body.get("max_completion_tokens") or body.get("max_tokens")
        stop = body.get("stop")
        stop = [stop] if isinstance(stop, str) else (stop or None)
        rid = "chatcmpl-" + uuid.uuid4().hex[:24]
        created = int(time.time())
        from engine import parsers
        markers, parse_calls = parsers.pick(
            ARGS.reasoning_parser, ARGS.tool_parser,
            ENGINE.parser_markers() if hasattr(ENGINE, "parser_markers") else {},
            ENGINE.parse_tool_calls if hasattr(ENGINE, "parse_tool_calls") else parse_tool_calls)
        parser = OutputParser(thinking_open, bool(tools), **markers)
        yield ("ready", rid, created)

        reasoning, content, stats = [], [], {}
        with ENGINE.lock:
            try:
                gen = lambda: ENGINE.generate(ids, media, max_tokens, sampling, stop, cancelled.is_set,
                                              seed=body.get("seed"), thinking_open=bool(thinking_open))
                for item in run_job(gen):
                    if isinstance(item, dict):
                        stats = item
                        break
                    for kind, s in parser.feed(item[1]):
                        (reasoning if kind == "reasoning" else content).append(s)
                        yield ("delta", kind, s)
                for kind, s in parser.finish():
                    (reasoning if kind == "reasoning" else content).append(s)
                    yield ("delta", kind, s)
            except MemoryLimitError as e:
                log(f"memory: {e}")
                raise ChatError(503, str(e), "server_error", "insufficient_memory")
            except ContextError as e:
                raise ChatError(400, str(e), code="context_length_exceeded")
            except Exception as e:  # keep the server alive on model errors
                log(f"generation error: {e!r}")
                raise ChatError(500, str(e), "server_error")

        calls = parse_calls(parser.tool_text, tools) if parser.tool_text else []
        finish, stop_reason = finish_reason(stats, bool(calls))
        usage = {"prompt_tokens": stats.get("prompt_tokens", len(ids)),
                 "completion_tokens": stats.get("completion_tokens", 0),
                 "total_tokens": stats.get("prompt_tokens", len(ids)) + stats.get("completion_tokens", 0),
                 "prompt_tokens_details": {"cached_tokens": stats.get("cached_tokens", 0)}}
        record_stats(stats)
        log(f"{len(ids)} prompt ({stats.get('cached_tokens', 0)} cached, {stats.get('prefill_tps', 0)} tok/s) → "
            f"{stats.get('completion_tokens', 0)} out @ {stats.get('decode_tps', 0)} tok/s, "
            f"{finish}{', ' + str(len(calls)) + ' tool calls' if calls else ''}, peak {stats.get('peak_gb')} GB")
        yield ("result", {"content": "".join(content), "reasoning": "".join(reasoning), "calls": calls,
                          "finish": finish, "stop_reason": stop_reason, "usage": usage,
                          "rid": rid, "created": created, "stats": stats})

    def chat(self, body):
        """OpenAI /v1/chat/completions — an SSE stream of chat.completion.chunk, or one chat.completion."""
        stream = bool(body.get("stream"))
        include_usage = bool((body.get("stream_options") or {}).get("include_usage"))
        cancelled = threading.Event()
        rid = created = None

        def chunk(delta, finish=None, usage=None, stop_reason=None):
            choice = {"index": 0, "delta": delta, "finish_reason": finish}
            if stop_reason is not None:               # why the answer ended, when it is not a plain stop
                choice["stop_reason"] = stop_reason
            c = {"id": rid, "object": "chat.completion.chunk", "created": created, "model": ARGS.name,
                 "choices": [choice]}
            if usage is not None:
                c["usage"] = usage
            return c

        def send(obj):
            try:
                self.wfile.write(b"data: " + json.dumps(obj, ensure_ascii=False).encode() + b"\n\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                cancelled.set()

        result = None
        try:
            for ev in self._chat_core(body, cancelled):
                if ev[0] == "ready":
                    rid, created = ev[1], ev[2]
                    if stream:
                        self.send_response(200)
                        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                        self.send_header("Cache-Control", "no-cache")
                        self.send_header("Connection", "close")
                        self.send_header("Access-Control-Allow-Origin", "*")
                        self.end_headers()
                        self.close_connection = True
                        send(chunk({"role": "assistant", "content": ""}))
                elif ev[0] == "delta":
                    if stream:
                        send(chunk({"reasoning_content": ev[2]} if ev[1] == "reasoning" else {"content": ev[2]}))
                elif ev[0] == "result":
                    result = ev[1]
        except ChatError as e:
            if stream and rid is not None:            # headers are already out — the error must go inline
                send({"error": {"message": e.message, "type": e.typ, "code": e.code}})
                return
            return self._error(e.status, e.message, e.typ, e.code)

        finish, stop_reason, usage, calls = result["finish"], result["stop_reason"], result["usage"], result["calls"]
        if stream:
            if calls:
                send(chunk({"tool_calls": [dict(c, index=i) for i, c in enumerate(calls)]}))
            send(chunk({}, finish, stop_reason=stop_reason))
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
        msg = {"role": "assistant", "content": result["content"] or (None if calls else "")}
        if result["reasoning"]:
            msg["reasoning_content"] = result["reasoning"]
        if calls:
            msg["tool_calls"] = calls
        choice = {"index": 0, "message": msg, "finish_reason": finish}
        if stop_reason is not None:
            choice["stop_reason"] = stop_reason
        self._json(200, {"id": result["rid"], "object": "chat.completion", "created": result["created"],
                         "model": ARGS.name, "choices": [choice], "usage": usage})

    # ───────────────────────── Ollama wire format ─────────────────────────
    def _begin_ndjson(self):
        """Ollama streams newline-delimited JSON (not SSE), one object per line."""
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.close_connection = True

    def chat_ollama(self, body):
        """Ollama POST /api/chat — the same chat core, framed as Ollama NDJSON or a single object."""
        from engine import ollama
        model = body.get("model") or ARGS.name
        oai = ollama.to_openai_chat(body)
        stream = oai.get("stream", True)
        cancelled = threading.Event()
        ready = False

        def send_line(obj):
            try:
                self.wfile.write(json.dumps(obj, ensure_ascii=False).encode() + b"\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                cancelled.set()

        result = None
        try:
            for ev in self._chat_core(oai, cancelled):
                if ev[0] == "ready":
                    ready = True
                    if stream:
                        self._begin_ndjson()
                elif ev[0] == "delta":
                    if stream:
                        send_line(ollama.chat_delta(model, content=ev[2]) if ev[1] == "content"
                                  else ollama.chat_delta(model, thinking=ev[2]))
                elif ev[0] == "result":
                    result = ev[1]
        except ChatError as e:
            if stream and ready:
                return send_line(ollama.error(e.message))
            return self._json(e.status, ollama.error(e.message))

        stats = result.get("stats") or {}
        timings = {"prefill_s": stats.get("prefill_s"), "decode_s": stats.get("decode_s")}
        if stream:
            if result["calls"]:
                send_line(ollama.chat_tool_delta(model, result["calls"]))
            send_line(ollama.chat_final(model, result["finish"], result["usage"], timings))
            return
        self._json(200, ollama.chat_response(model, result["content"], result["reasoning"],
                                             result["calls"], result["finish"], result["usage"], timings))

    def generate_ollama(self, body):
        """Ollama POST /api/generate (raw prompt) — the chat core, framed with the `response` field."""
        from engine import ollama
        model = body.get("model") or ARGS.name
        oai = ollama.to_openai_generate(body)
        stream = oai.get("stream", True)
        cancelled = threading.Event()
        ready = False

        def send_line(obj):
            try:
                self.wfile.write(json.dumps(obj, ensure_ascii=False).encode() + b"\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                cancelled.set()

        result = None
        try:
            for ev in self._chat_core(oai, cancelled):
                if ev[0] == "ready":
                    ready = True
                    if stream:
                        self._begin_ndjson()
                elif ev[0] == "delta":
                    if stream and ev[1] == "content":     # /api/generate streams only the answer text
                        send_line(ollama.generate_delta(model, ev[2]))
                elif ev[0] == "result":
                    result = ev[1]
        except ChatError as e:
            if stream and ready:
                return send_line(ollama.error(e.message))
            return self._json(e.status, ollama.error(e.message))

        stats = result.get("stats") or {}
        timings = {"prefill_s": stats.get("prefill_s"), "decode_s": stats.get("decode_s")}
        if stream:
            send_line(ollama.generate_final(model, result["finish"], result["usage"], timings))
            return
        self._json(200, ollama.generate_response(model, result["content"], result["finish"],
                                                 result["usage"], timings))

    # ───────────────────────── Anthropic Messages API ─────────────────────────
    def messages_anthropic(self, body):
        """Anthropic POST /v1/messages — the chat core, framed as an Anthropic message or its SSE event stream
        (message_start → content blocks → message_delta → message_stop)."""
        from engine import anthropic as A
        model = body.get("model") or ARGS.name
        internal = A.to_internal(body)
        stream = internal.get("stream", False)
        cancelled = threading.Event()
        rid = A.msg_id()
        started = False
        open_kind = None            # which content block is open while streaming: "thinking" | "text" | None
        index = -1
        result = None

        def sse(ev):
            try:
                self.wfile.write(f"event: {ev['type']}\ndata: ".encode()
                                 + json.dumps(ev, ensure_ascii=False).encode() + b"\n\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                cancelled.set()

        try:
            for e in self._chat_core(internal, cancelled):
                if e[0] == "ready":
                    started = True
                    if stream:
                        self.send_response(200)
                        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                        self.send_header("Cache-Control", "no-cache")
                        self.send_header("Connection", "close")
                        self.send_header("Access-Control-Allow-Origin", "*")
                        self.end_headers()
                        self.close_connection = True
                        sse(A.message_start(model, rid))
                elif e[0] == "delta":
                    if not stream:
                        continue
                    want = "thinking" if e[1] == "reasoning" else "text"
                    if open_kind != want:
                        if open_kind is not None:
                            sse(A.block_stop(index))
                        index += 1; open_kind = want
                        sse(A.block_start(index, want))
                    sse(A.block_delta(index, want, e[2]))
                elif e[0] == "result":
                    result = e[1]
        except ChatError as ex:
            if stream and started:
                sse({"type": "error", "error": {"type": ex.typ, "message": ex.message}})
                return
            return self._json(ex.status, A.error(ex.message, ex.typ))

        calls, finish = result["calls"], result["finish"]
        out_tokens = result["usage"].get("completion_tokens", 0)
        if stream:
            if open_kind is not None:
                sse(A.block_stop(index))
            for c in calls:
                index += 1
                start, inp = A.tool_block_start(index, c)
                sse(start)
                sse(A.input_json_delta(index, inp))
                sse(A.block_stop(index))
            sse(A.message_delta(finish, bool(calls), out_tokens))
            sse(A.message_stop())
            return
        self._json(200, A.message(model, result["content"], result["reasoning"], calls, finish, result["usage"], rid))

    def _ollama_card(self):
        from engine import ollama
        return ollama.model_card(ARGS.name, size_bytes=getattr(ENGINE, "model_bytes", 0), fmt="mlx")

    def _capabilities(self):
        caps = ["completion", "tools"]
        if getattr(ENGINE, "vision_allowed", False):
            caps.append("vision")
        if getattr(ENGINE, "supports_thinking", True):
            caps.append("thinking")
        return caps


def finish_reason(stats, has_calls):
    """(finish_reason, stop_reason) for a reply. A reply cut off because the model was looping ends as
    'length' (it was truncated, not a clean stop) and carries stop_reason 'repetition' so the UI can say
    why, instead of looking like a normal finish."""
    if has_calls:
        return "tool_calls", None
    if stats.get("loop_stopped"):
        return "length", "repetition"
    if stats.get("finish_reason") == "length":
        return "length", None
    return "stop", None


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
                if not isinstance(r, dict):
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
    ap.add_argument("--backend", choices=("auto", "native", "mlxlm", "gguf"), default="auto",
                    help="auto: the native engine for Qwen3.5-hybrid bundles, the optional mlx-lm backend "
                         "for every other MLX architecture")
    ap.add_argument("--ptq1", choices=("on", "off"), default="off",
                    help="experimental: run PTQ1_0 / TQ1_G128 (1.75-bit ternary) bundles on the Metal lookup-table kernel")
    ap.add_argument("--no-vision", action="store_true")
    ap.add_argument("--log-level", choices=("error", "warning", "info", "debug", "trace"), default="info",
                    help="what the engine writes to its log (each line is tagged with a group)")
    ap.add_argument("--reasoning-parser", default="auto", metavar="auto|think|none|tags:<close>",
                    help="where the reasoning ends in the output (auto = the model family's own)")
    ap.add_argument("--tool-parser", default="auto", choices=("auto", "qwen_xml", "json", "mlx-lm", "none"),
                    help="the tool-call format to read from the output (auto = the model family's own)")
    ap.add_argument("--images", choices=("on", "off"), default="off",
                    help="serve /v1/images/generations through mflux (./llm setup --images installs it)")
    ap.add_argument("--fallback-model", choices=("on", "off"), default="on",
                    help="a request that names a model other than the loaded one is answered by the loaded "
                         "model (on) or refused with 404 (off)")
    ap.add_argument("--prefill-guard", choices=("on", "off"), default="on",
                    help="check free memory before a long prompt is prefilled and refuse what would not fit")
    ap.add_argument("--no-speculative", action="store_true", help="disable prompt-lookup speculative decoding")
    ap.add_argument("--steer", default="", help="an activation edit shipped next to a base model: a direction file "
                    "(.safetensors/.npy/.bin) or a folder holding one; native route only (engine/steering.py)")
    ap.add_argument("--steer-alpha", type=float, default=None,
                    help="strength of --steer (default: the folder's adapter.json, else 1.0; 0 = off)")
    ap.add_argument("--stats-file", default="", help="append one JSON line per request (all-time token statistics)")
    ap.add_argument("--idle-sleep", type=float, default=15, help="unload the model after N idle minutes (0 = never)")
    ap.add_argument("--max-queue", type=int, default=0,
                    help="refuse (503) a generation request when this many are already in flight (0 = no cap)")
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
    ap.add_argument("--parent-pid", type=int, default=0,
                    help="shut down if this process (the app that launched the engine) goes away; "
                         "0 = run independently, as the CLI daemon does")
    ARGS = ap.parse_args()
    from engine import parsers
    try:
        parsers.validate(ARGS.reasoning_parser, ARGS.tool_parser)
    except ValueError as e:
        ap.error(str(e))
    ARGS.name = ARGS.name or os.path.basename(ARGS.model.rstrip("/")).lower()
    ADMISSION.limit = max(0, ARGS.max_queue)

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from engine.compat import CompatEngine, pick_backend
    from engine.generate import Engine
    if ARGS.steer and ARGS.backend == "mlxlm":
        ap.error("--steer works on the native engine and the built-in GGUF decoder, not on the mlx-lm backend")
    backend = pick_backend(ARGS.model, ARGS.backend)
    if ARGS.steer and backend == "mlxlm":
        ap.error("--steer works on the native engine and the built-in GGUF decoder, not on the mlx-lm backend "
                 "(this model runs on mlx-lm)")
    if backend == "gguf":
        from engine.gguf_engine import GGUFEngine
        Backend = GGUFEngine
    else:
        Backend = Engine if backend == "native" else CompatEngine
    log(f"backend: {'native engine' if backend == 'native' else 'GGUF on the MLX decoder' if backend == 'gguf' else 'mlx-lm'}")
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
                           steer=(ARGS.steer, ARGS.steer_alpha) if ARGS.steer else None,
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

    if ARGS.parent_pid:   # tie the engine's life to the app that launched it (SIGTERM runs the clean shutdown)
        log(f"watchdog: will stop if the launching process (PID {ARGS.parent_pid}) exits")
        threading.Thread(target=watch_parent, args=(ARGS.parent_pid, lambda: (
            log(f"watchdog: PID {ARGS.parent_pid} is gone — stopping the engine"),
            os.kill(os.getpid(), signal.SIGTERM))), daemon=True).start()

    srv = Server((ARGS.host, ARGS.port), Handler)
    log(f"serving {ARGS.name} on http://{ARGS.host}:{ARGS.port}/v1  "
        f"(context {ENGINE.max_context}, max output {ENGINE.max_tokens})")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
