"""OpenAI-compatible front for a GGUF model, served by llama.cpp's `llama-server` (stdlib only).

    python -m engine.gguf_server --model file.gguf --port 8090 [--llama-server PATH] ...

llama-server does the inference; this process keeps the engine's contract around it so every client and the
LLMBar see the same thing whatever the model's format: `/health` and `/v1/models`, the API key, the
model-name fallback switch, a leveled and grouped log, and the idle sleep (the model is unloaded after N
idle minutes and loaded again by the next request).
"""
import argparse
import http.client
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from engine import gguf, logs  # noqa: E402

ARGS = None
INNER = None
LOG = print
HOP = {"connection", "keep-alive", "transfer-encoding", "content-length", "server", "date"}


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


_SPEC = {}


def supports_spec(binary):
    """Does this llama-server know `--spec-type ngram-mod`? (older builds do not, and would refuse to start)"""
    if binary not in _SPEC:
        try:
            out = subprocess.run([binary, "--help"], capture_output=True, text=True, timeout=30)
            _SPEC[binary] = "ngram-mod" in (out.stdout + out.stderr)
        except (OSError, subprocess.TimeoutExpired):
            _SPEC[binary] = False
    return _SPEC[binary]


class Inner:
    """The llama-server process: started on demand, stopped when idle."""

    def __init__(self, binary, model, ctx, name, mmproj=None, ngl=99, log_path=None, speculative=True):
        self.binary, self.model, self.ctx, self.name = binary, model, ctx, name
        self.speculative = speculative
        self.mmproj, self.ngl, self.log_path = mmproj, ngl, log_path
        self.proc, self.port = None, None
        self.lock = threading.Lock()
        self.last_used = time.time()
        self.inflight = 0

    def command(self, port):
        cmd = [self.binary, "-m", self.model, "--host", "127.0.0.1", "--port", str(port), "-c", str(self.ctx),
               "-ngl", str(self.ngl), "-np", "1", "--jinja", "-a", self.name]
        if self.mmproj:
            cmd += ["--mmproj", self.mmproj]
        if self.speculative and supports_spec(self.binary):
            # prompt-lookup drafting: where the answer repeats the prompt (code edits, quotations) several
            # tokens are verified per step; where it does not, nothing is lost. Greedy output is identical.
            cmd += ["--spec-type", "ngram-mod"]
        return cmd

    @property
    def running(self):
        return self.proc is not None and self.proc.poll() is None

    def ensure(self, timeout=900):
        """Make sure the server is up; returns its port. Raises RuntimeError with the log's tail."""
        with self.lock:
            self.last_used = time.time()
            if self.running:
                return self.port
            self.port = free_port()
            sink = open(self.log_path, "a") if self.log_path else subprocess.DEVNULL
            LOG(f"model: loading {os.path.basename(self.model)} into llama-server (context {self.ctx})")
            self.proc = subprocess.Popen(self.command(self.port), stdout=sink, stderr=subprocess.STDOUT,
                                         stdin=subprocess.DEVNULL)
            t0 = time.time()
            while time.time() - t0 < timeout:
                if self.proc.poll() is not None:
                    raise RuntimeError(f"llama-server exited with code {self.proc.returncode}\n{self.tail()}")
                try:
                    c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=2)
                    c.request("GET", "/health")
                    r = c.getresponse()
                    r.read()
                    c.close()
                    if r.status == 200:
                        LOG(f"loaded {os.path.basename(self.model)}: llama-server ready in {time.time() - t0:.1f}s")
                        return self.port
                except OSError:
                    pass
                time.sleep(0.3)
            self.stop_locked()
            raise RuntimeError("llama-server did not become ready in time\n" + self.tail())

    def tail(self, n=12):
        try:
            with open(self.log_path) as f:
                return "".join(f.readlines()[-n:])
        except (OSError, TypeError):
            return ""

    def stop_locked(self):
        p, self.proc = self.proc, None
        if p and p.poll() is None:
            p.terminate()
            try:
                p.wait(10)
            except subprocess.TimeoutExpired:
                p.kill()

    def stop(self):
        with self.lock:
            self.stop_locked()


MAX_BODY = 64 * 2**20          # bytes of one request body


def model_matches(requested):
    if not requested:
        return True
    import re
    norm = lambda s: re.sub(r"[^a-z0-9._-]+", "-", str(s).lower()).strip("-")
    want = {norm(ARGS.name), norm(os.path.splitext(os.path.basename(ARGS.model))[0])}
    return norm(requested) in want or norm(str(requested).split("/")[-1]) in want


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "llm-gguf"

    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json; charset=utf-8", extra=None):
        data = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(data)

    def _error(self, code, message, kind="server_error", err_code=None):
        self._send(code, {"error": {"message": message, "type": kind, "code": err_code}})

    def _auth(self):
        import hmac
        if not ARGS.api_key or hmac.compare_digest(self.headers.get("Authorization", "").encode(),
                                                   f"Bearer {ARGS.api_key}".encode()):
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
        if not self._auth():
            return
        path = self.path.split("?")[0].rstrip("/")
        if path in ("/health", ""):
            return self._send(200, self.health())
        if path == "/v1/models" or path.startswith("/v1/models/"):
            card = {"id": ARGS.name, "object": "model", "created": int(START), "owned_by": "local",
                    "context_length": ARGS.context_effective, "max_output_tokens": ARGS.max_tokens}
            return self._send(200, card if path.startswith("/v1/models/") else {"object": "list", "data": [card]})
        if path == "/v1/stats":
            return self._send(200, {"session": STATS, "history": [], "all_time": STATS})
        self._error(404, f"unknown path {self.path}")

    def health(self):
        return {"status": "ok", "backend": "llama.cpp", "model": ARGS.name, "model_path": ARGS.model,
                "state": "awake" if INNER.running else "sleeping", "idle_sleep_min": ARGS.idle_sleep,
                "context": ARGS.context_effective, "max_tokens": ARGS.max_tokens,
                "thinking": False, "vision": bool(INNER.mmproj), "vision_capable": bool(INNER.mmproj),
                "busy": INNER.inflight > 0, "progress": None, "requests": STATS["requests"],
                "format": ARGS.format_info}

    def do_POST(self):
        if not self._auth():
            return
        path = self.path.split("?")[0].rstrip("/")
        try:
            n = int(self.headers.get("Content-Length") or 0)
            if n > MAX_BODY:
                return self._error(413, f"request body is larger than {MAX_BODY // 2**20} MB", "invalid_request_error")
            raw = self.rfile.read(n) if n > 0 else b""
            body = json.loads(raw or b"{}")
            if not isinstance(body, dict):
                raise ValueError("the body must be a JSON object")
        except ValueError:
            return self._error(400, "invalid JSON body", "invalid_request_error")
        if path not in ("/v1/chat/completions", "/chat/completions", "/v1/completions", "/v1/embeddings"):
            return self._error(404, f"unknown path {self.path}")
        if ARGS.fallback_model == "off" and not model_matches(body.get("model")):
            return self._error(404, f"model {body.get('model')!r} is not loaded; the loaded model is "
                                    f"{ARGS.name!r} (fallback is off)", "invalid_request_error", "model_not_found")
        mt = body.get("max_completion_tokens") or body.get("max_tokens")
        try:
            body["max_tokens"] = max(1, min(int(mt), ARGS.max_tokens)) if mt else ARGS.max_tokens
        except (TypeError, ValueError):
            return self._error(400, "max_tokens must be an integer", "invalid_request_error")
        body.pop("max_completion_tokens", None)
        body["model"] = ARGS.name
        if isinstance(body.get("messages"), list) and not INNER.mmproj:      # no projector: answer without pictures
            from engine.media import strip_images
            body["messages"], n_dropped = strip_images(body["messages"])
            if n_dropped:
                LOG(f"trace: {n_dropped} image(s) dropped, this model does not read images")
        self.proxy("/v1/chat/completions" if path == "/chat/completions" else path, body)

    def proxy(self, path, body):
        t0 = time.time()
        INNER.inflight += 1
        conn = None
        try:
            try:
                port = INNER.ensure()
            except RuntimeError as e:
                LOG(f"generation error: {e}")
                return self._error(503, str(e).splitlines()[0], "server_error", "model_unavailable")
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=3600)
            conn.request("POST", path, json.dumps(body).encode(), {"Content-Type": "application/json"})
            r = conn.getresponse()
            ctype = r.getheader("Content-Type", "application/json")
            if "text/event-stream" in ctype:
                self.send_response(r.status)
                self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.close_connection = True
                try:
                    while True:
                        chunk = r.readline()
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, OSError):
                    conn.close()                       # the client left: llama-server stops generating
                LOG(f"trace: streamed in {time.time() - t0:.1f}s")
            else:
                data = r.read()
                self._send(r.status, data, ctype)
                try:
                    u = json.loads(data).get("usage") or {}
                    LOG(f"{u.get('prompt_tokens', '?')} prompt → {u.get('completion_tokens', '?')} out, "
                        f"{time.time() - t0:.1f}s")
                except (ValueError, AttributeError):
                    pass
            STATS["requests"] += 1
        except (OSError, http.client.HTTPException) as e:
            if INNER.running:
                LOG(f"generation error: {e!r}")
                try:
                    self._error(502, f"the model server did not answer: {e}")
                except OSError:
                    pass
        finally:
            INNER.inflight -= 1
            INNER.last_used = time.time()
            if conn:
                conn.close()


START = time.time()
STATS = {"requests": 0}


def auto_context(model_ctx, ram_gb):
    cap = 32768 if ram_gb >= 32 else 16384 if ram_gb >= 16 else 8192
    return int(min(model_ctx or cap, cap))


def main(argv=None):
    global ARGS, INNER, LOG
    ap = argparse.ArgumentParser(description="OpenAI-compatible front for a GGUF model (llama.cpp)")
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", default=None)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--context", type=int, default=0, help="0 = auto from the model and RAM")
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--api-key", default=os.environ.get("ENGINE_API_KEY", ""))
    ap.add_argument("--idle-sleep", type=float, default=15, help="unload after N idle minutes (0 = never)")
    ap.add_argument("--log-level", choices=logs.LEVELS, default="info")
    ap.add_argument("--fallback-model", choices=("on", "off"), default="on")
    ap.add_argument("--llama-server", default="")
    ap.add_argument("--mmproj", default="")
    ap.add_argument("--no-vision", action="store_true")
    ap.add_argument("--no-speculative", action="store_true", help="no prompt-lookup speculative decoding")
    ap.add_argument("--inner-log", default="")
    ap.add_argument("--ram-gb", type=float, default=0)
    ap.add_argument("--parent-pid", type=int, default=0,
                    help="stop (and with it the llama-server child) if this process exits; 0 = independent")
    ARGS = ap.parse_args(argv)
    LOG = logs.make(ARGS.log_level)
    try:
        info = gguf.read_header(ARGS.model)
    except (gguf.GGUFError, OSError) as e:
        print(f"cannot read {ARGS.model}: {e}", file=sys.stderr)
        return 2
    binary = ARGS.llama_server or gguf.server_path(os.environ.get("LLM_HOME") or ROOT)
    if not binary:
        print("llama.cpp is not installed — run: ./llm setup --gguf", file=sys.stderr)
        return 2
    ram = ARGS.ram_gb or (os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30)
    ARGS.name = ARGS.name or os.path.splitext(os.path.basename(ARGS.model))[0].lower()
    ARGS.context_effective = ARGS.context or auto_context(info.get("context_length"), ram)
    ARGS.format_info = {"architecture": info["architecture"], "quant": info["quant"]}
    mmproj = None if ARGS.no_vision else (ARGS.mmproj or gguf.projector_for(ARGS.model))
    INNER = Inner(binary, ARGS.model, ARGS.context_effective, ARGS.name, mmproj, log_path=ARGS.inner_log or None,
                  speculative=not ARGS.no_speculative)
    LOG(f"backend: llama.cpp ({info['architecture']} {info['quant']})")
    try:
        INNER.ensure()
    except RuntimeError as e:
        print(f"✗ {e}", file=sys.stderr)
        return 3

    def idle_watch():
        while True:
            time.sleep(10)
            if (ARGS.idle_sleep and INNER.running and INNER.inflight == 0
                    and time.time() - INNER.last_used > ARGS.idle_sleep * 60):
                INNER.stop()
                LOG("sleep: model unloaded (idle); it wakes on the next request")
    threading.Thread(target=idle_watch, daemon=True).start()

    srv = ThreadingHTTPServer((ARGS.host, ARGS.port), Handler)
    srv.daemon_threads = True

    def bye(*_):
        INNER.stop()
        os._exit(0)
    signal.signal(signal.SIGTERM, bye)
    signal.signal(signal.SIGINT, bye)

    if ARGS.parent_pid:   # the app that launched us; if it crashes, stop so no orphan llama-server lingers
        from engine.server import watch_parent
        LOG(f"watchdog: will stop if the launching process (PID {ARGS.parent_pid}) exits")
        threading.Thread(target=watch_parent, args=(ARGS.parent_pid,
                         lambda: os.kill(os.getpid(), signal.SIGTERM)), daemon=True).start()
    LOG(f"serving {ARGS.name} on http://{ARGS.host}:{ARGS.port}/v1  (context {ARGS.context_effective}, "
        f"max output {ARGS.max_tokens})")
    try:
        srv.serve_forever()
    finally:
        INNER.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
