"""The browser assistant's local service — `./llm assistant serve`.

The Safari (and later Chrome) extension never talks to the engine directly. It talks to this small stdlib
HTTP service on 127.0.0.1, which

* knows the installed models and the global default, from the same `llm.py` code the CLI and LLMBar run;
* starts and stops the engine, switches its model, loads and unloads the weights — things the engine cannot
  do for itself, because it is not running, or holds one model at a time;
* proxies chat with per-session generations, so a session can cancel its own answer and nobody else's;
* publishes one state snapshot, versioned, as a stream every client can follow.

What reaches it is trusted only after four checks: the Host is loopback (DNS rebinding), the Origin, when
present, is a browser extension and never a web page, the bearer token matches, and the body is bounded.
See docs/safari-assistant/PROTOCOL.md for the wire format.
"""
import http.client
import http.server
import json
import os
import secrets
import socket
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request
import uuid

from . import prompts

PROTOCOL = 1
PORT = 8093
MAX_BODY = 4 * 2**20
EXTENSION_SCHEMES = ("safari-web-extension", "chrome-extension", "moz-extension")
HOME = os.environ.get("ASSISTANT_HOME") or os.path.expanduser("~/Library/Application Support/AI-llm-Engine")
TOKEN_FILE = os.path.join(HOME, "browser-assistant.json")
POLL_S = 1.0
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXT_BUILT = os.path.join(ROOT, "browser-assistant", "dist")      # what this copy of the code ships
EXT_HOME = os.path.join(HOME, "browser-assistant")                # what the browser is pointed at


# ───────────────────────────── the extension's files ─────────────────────────────
def _manifest_version(folder):
    try:
        with open(os.path.join(folder, "manifest.json")) as f:
            return str(json.load(f).get("version") or "")
    except (OSError, ValueError):
        return ""


def _tree_stamp(folder):
    """A cheap fingerprint of a built extension: every file's relative path, size and mtime."""
    out = []
    for base, _dirs, files in os.walk(folder):
        for name in sorted(files):
            st = os.stat(os.path.join(base, name))
            out.append((os.path.relpath(os.path.join(base, name), folder), st.st_size, int(st.st_mtime)))
    return sorted(out)


def sync_extension(built=None, home=None):
    """Copy the built extension to a folder that outlives app updates, and keep it current.

    The browser is pointed at `home`, never at the app bundle: an LLMBar update replaces the bundle (and may
    move it), and Safari would be left holding a path into the old one. When the shipped build differs from
    the installed copy — a new LLMBar brought a new extension — the copy is replaced, each browser's folder
    swapped in whole so a half-written extension is never loaded. Returns {browser: version} of what is
    installed now; {} when nothing was built."""
    import shutil
    built, home = built or EXT_BUILT, home or EXT_HOME
    out = {}
    for browser in ("safari", "chromium"):
        src, dst = os.path.join(built, browser), os.path.join(home, browser)
        if not os.path.isfile(os.path.join(src, "manifest.json")):
            if os.path.isfile(os.path.join(dst, "manifest.json")):
                out[browser] = _manifest_version(dst)          # an earlier install stays usable
            continue
        if not os.path.isdir(dst) or _tree_stamp(src) != _tree_stamp(dst):
            os.makedirs(home, mode=0o700, exist_ok=True)
            tmp = dst + ".new"
            shutil.rmtree(tmp, ignore_errors=True)
            shutil.copytree(src, tmp, copy_function=shutil.copy2)
            shutil.rmtree(dst, ignore_errors=True)
            os.replace(tmp, dst)
        out[browser] = _manifest_version(dst)
    return out


def code_stamp():
    """Changes when the service's own code or the shipped extension does — i.e. when LLMBar was updated
    under a running service."""
    stamp = []
    for path in (os.path.abspath(__file__), os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts.py"),
                 os.path.join(EXT_BUILT, "safari", "manifest.json")):
        try:
            st = os.stat(path)
            stamp.append((st.st_size, int(st.st_mtime)))
        except OSError:
            stamp.append(None)
    return stamp


# ───────────────────────────── token ─────────────────────────────
def load_or_create_token(path=None):
    """The pairing token, created on first use with owner-only permissions."""
    path = path or TOKEN_FILE
    try:
        with open(path) as f:
            tok = json.load(f).get("token")
        if isinstance(tok, str) and len(tok) >= 24:
            return tok
    except (OSError, ValueError):
        pass
    os.makedirs(os.path.dirname(path), mode=0o700, exist_ok=True)
    tok = secrets.token_urlsafe(32)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump({"token": tok, "port": PORT}, f)
    return tok


# ───────────────────────────── the engine, as the CLI sees it ─────────────────────────────
class EngineControl:
    """Everything the service needs from the engine and the CLI. Lifecycle actions run `./llm` itself, in a
    child process — exactly what LLMBar does — so the pid file, the memory checks, the fallback from the own
    GGUF decoder to llama.cpp and the config all behave as they do everywhere else."""

    def __init__(self):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sys.path.insert(0, root)
        import llm
        self.llm = llm
        self._models, self._models_at = [], 0.0

    def cfg(self):
        return self.llm.load_cfg()

    def default(self):
        return self.cfg().get("model") or ""

    def models(self):
        if time.time() - self._models_at > 5:
            ram = self.llm.RAM_GB
            self._models = [{"name": m["name"], "path": m["path"], "size_gb": round(m["size"], 2),
                             "format": m.get("format"), "route": m.get("engine") or m.get("route"),
                             "supported": bool(m.get("supported")), "problems": m.get("problems") or [],
                             "fits": m["size"] + 2 <= ram * 0.75, "vision": bool(m.get("vision")),
                             "thinking": bool(m.get("thinking")), "favorite": bool(m.get("favorite"))}
                            for m in self.llm.scan_models(self.cfg())]
            self._models_at = time.time()
        return self._models

    def health(self):
        return self.llm.health(self.cfg(), timeout=1.5)

    def pid_alive(self):
        return bool(self.llm.read_pid(self.llm.ENGINE_PID))

    def run(self, args, timeout=900):
        """(ok, last lines of output) of `./llm <args>`, non-interactive."""
        env = dict(os.environ, LLM_HOME=self.llm.DATA, NO_COLOR="1")
        try:
            r = subprocess.run([os.path.join(self.llm.BASE, "llm"), *args], cwd=self.llm.BASE, env=env,
                               stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return False, f"./llm {' '.join(args)} did not finish in {timeout} s"
        out = (r.stdout + r.stderr).strip().splitlines()
        return r.returncode == 0, "\n".join(out[-6:])

    def _base(self):
        cfg = self.cfg()
        return cfg["port"], cfg.get("api_key") or ""

    def post(self, path, body, timeout=600):
        """(status, json) of a POST to the engine."""
        port, key = self._base()
        req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json",
                                              **({"Authorization": f"Bearer {key}"} if key else {})})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.status, json.loads(r.read() or b"{}")
        except urllib.error.HTTPError as e:
            try:
                return e.code, json.loads(e.read() or b"{}")
            except ValueError:
                return e.code, {}
        except OSError as e:
            return 0, {"error": {"message": str(e)}}

    def open_chat(self, body):
        """(socket, response) of a streaming chat request. The socket is taken before the response, because
        http.client hands it over to the response and forgets it — and shutting it down from another thread
        is how a cancel interrupts the read."""
        port, key = self._base()
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=600)
        conn.connect()
        sock = conn.sock
        conn.request("POST", "/v1/chat/completions", body=json.dumps(body).encode(),
                     headers={"Content-Type": "application/json", "Accept": "text/event-stream",
                              **({"Authorization": f"Bearer {key}"} if key else {})})
        return sock, conn.getresponse()


# ───────────────────────────── generations ─────────────────────────────
class Generation:
    def __init__(self, session):
        self.id = uuid.uuid4().hex[:16]
        self.session = session
        self.started = time.time()
        self.cancelled = False
        self.sock = None

    def cancel(self):
        """Idempotent. Closing the upstream socket is what stops the engine: it notices the hang-up between
        two tokens and ends that request — only that one."""
        self.cancelled = True
        if self.sock is not None:
            try:
                self.sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass


class Conflict(Exception):
    def __init__(self, code, message, **extra):
        super().__init__(message)
        self.code, self.message, self.extra = code, message, extra


# ───────────────────────────── shared state ─────────────────────────────
class Assistant:
    """State, commands and the generations of every session."""

    def __init__(self, control):
        self.control = control
        self.lock = threading.Lock()
        self.changed = threading.Condition(self.lock)
        self.version = 0
        self.core = None
        self.op = None              # the lifecycle action in progress: start | stop | switch | load | unload
        self.error = None
        self.gens = {}
        self.unload_supported = True
        self.extension = {}         # {browser: version installed in EXT_HOME} — what an extension should be

    # ── snapshot ──
    def _core(self):
        h = self.control.health()
        op = self.op
        gens = list(self.gens.values())
        if op in ("start", "switch") or (not h and self.control.pid_alive()):
            engine = "starting"
        elif h:
            engine = "ready"
        else:
            engine = "error" if self.error else "offline"
        awake = bool(h) and h.get("state") != "sleeping"
        inflight = int((h or {}).get("inflight") or 0)
        if op == "unload":
            model = "unloading"
        elif op == "load" or engine == "starting":
            model = "loading"
        elif not h or not awake:
            model = "not_loaded"
        elif h.get("busy") or inflight:
            model = "generating"
        else:
            model = "ready"
        last = (h or {}).get("last") or {}
        mem = (h or {}).get("memory_gb") or {}
        metrics = {k: v for k, v in {
            "context_tokens": (h or {}).get("context"), "model_gb": mem.get("model") if awake else None,
            "active_gb": mem.get("active"), "system_ram_used": (h or {}).get("system_ram_used"),
            "last_decode_tps": last.get("decode_tps"), "last_prefill_tps": last.get("prefill_tps"),
        }.items() if v is not None}
        return {
            "protocol": PROTOCOL,
            "engine": {"state": engine, "error": self.error, "operation": op},
            "model": {"state": model, "name": (h or {}).get("model") if h else None,
                      "path": (h or {}).get("model_path") if h else None},
            "default": self.control.default(),
            "models": self.control.models(),
            "in_use": {"inflight": inflight, "assistant_generations": len(gens)},
            "generations": [{"id": g.id, "session": g.session, "since": round(g.started, 3)} for g in gens],
            "metrics": metrics,
            "extension": {"versions": self.extension, "folder": EXT_HOME},
            "capabilities": {
                "unload": self.unload_supported,
                "per_session_model": False,
                "per_session_model_reason": "the engine serves one model at a time: a different model for one "
                                            "session would mean switching it for every client",
            },
        }

    def refresh(self):
        core = self._core()
        with self.changed:
            if core != self.core:
                self.core = core
                self.version += 1
                self.changed.notify_all()
            return dict(self.core, version=self.version)

    def snapshot(self):
        with self.lock:
            if self.core is not None:
                return dict(self.core, version=self.version)
        return self.refresh()

    def wait_change(self, version, timeout):
        with self.changed:
            self.changed.wait_for(lambda: self.version > version, timeout)
            return dict(self.core, version=self.version) if self.core else None

    def poll_forever(self, stop):
        while not stop.is_set():
            try:
                self.refresh()
            except Exception:                  # a health probe must never take the service down
                pass
            stop.wait(POLL_S)

    # ── conflicts ──
    def others(self, session):
        """Requests on the engine that are not this session's: other tabs, the CLI, LLMBar, editors."""
        h = self.control.health() or {}
        mine = sum(1 for g in self.gens.values() if g.session == session)
        return max(0, int(h.get("inflight") or 0) - mine)

    def _guard(self, session, confirm, what):
        n = self.others(session)
        if n and not confirm:
            raise Conflict("model_in_use", f"{n} other request(s) are using the model; {what} would interrupt "
                                           f"them. Confirm to do it anyway.", others=n)

    def _begin(self, op):
        with self.lock:
            if self.op:
                raise Conflict("busy", f"another action is in progress: {self.op}", operation=self.op)
            self.op, self.error = op, None
        self.refresh()

    def _end(self, error=None):
        with self.lock:
            self.op, self.error = None, error
        self.refresh()

    def _background(self, op, fn):
        self._begin(op)

        def run():
            err = None
            try:
                err = fn()
            except Exception as e:
                err = f"{type(e).__name__}: {e}"
            self._end(err)
        threading.Thread(target=run, daemon=True).start()
        return self.snapshot()

    # ── commands ──
    def start_engine(self):
        if self.control.health():
            return self.snapshot()

        def go():
            ok, out = self.control.run(["start"])
            return None if ok and self.control.health() else (out or "the engine did not start")
        return self._background("start", go)

    def stop_engine(self, session, confirm):
        self._guard(session, confirm, "stopping the engine")
        self._begin("stop")
        ok, out = self.control.run(["stop"], timeout=60)
        self._end(None if ok else out)
        return self.snapshot()

    def load(self):
        h = self.control.health()
        if not h:
            return self.start_engine()
        if h.get("state") != "sleeping":
            return self.snapshot()

        def go():
            status, body = self.control.post("/v1/engine/wake", {})
            return None if status == 200 else _engine_error(status, body, "load the model")
        return self._background("load", go)

    def unload(self, session, confirm):
        h = self.control.health()
        if not h or h.get("state") == "sleeping":
            return self.snapshot()
        self._guard(session, confirm, "unloading the model")

        def go():
            status, body = self.control.post("/v1/engine/sleep", {"force": bool(confirm)})
            if status == 404:
                self.unload_supported = False
                return "this engine route cannot unload its model; stop the engine instead"
            return None if status == 200 else _engine_error(status, body, "unload the model")
        return self._background("unload", go)

    def set_default(self, path, session, expected, confirm):
        models = {m["path"]: m for m in self.control.models()}
        if path not in models:
            raise Conflict("unknown_model", "no installed model at that path")
        if not models[path]["supported"]:
            raise Conflict("unsupported_model", "the engine cannot run this model: " +
                           ", ".join(models[path]["problems"]))
        current = self.control.default()
        if expected is not None and expected != current:
            raise Conflict("stale", "the default model was changed by another client", default=current)
        if path == current:
            return self.snapshot()
        if self.control.health():
            self._guard(session, confirm, "switching the model")

        def go():
            ok, out = self.control.run(["use", path])
            return None if ok else (out or "could not switch the model")
        return self._background("switch", go)

    def cancel(self, gen_id, session):
        g = self.gens.get(gen_id)
        if g is None:
            return {"cancelled": False, "reason": "no such generation, or it has already finished"}
        if g.session != session:
            raise Conflict("not_yours", "that generation belongs to another session")
        g.cancel()
        return {"cancelled": True}

    def chat(self, req, emit):
        """Run one task. `emit(event)` sends one event to the client and returns False once it has gone."""
        session = str(req.get("session") or "")[:64]
        if not session:
            raise Conflict("bad_request", "a session id is needed")
        h = self.control.health()
        if not h:
            raise Conflict("engine_offline", "the engine is not running — start it first")
        max_tokens = max(64, min(int(req.get("max_tokens") or 1024), 4096))
        try:
            messages, shared = prompts.build(req, int(h.get("context") or 4096), max_tokens)
        except prompts.PromptError as e:
            raise Conflict("bad_request", str(e)) from None
        g = Generation(session)
        with self.lock:
            self.gens[g.id] = g
        self.refresh()
        try:
            if not emit({"type": "start", "id": g.id, "model": h.get("model"), "shared": shared}):
                return
            body = {"model": h.get("model"), "messages": messages, "stream": True, "max_tokens": max_tokens,
                    "stream_options": {"include_usage": True},
                    "chat_template_kwargs": {"enable_thinking": bool(req.get("thinking"))}}
            try:
                g.sock, resp = self.control.open_chat(body)
            except OSError as e:
                emit({"type": "error", "message": f"the engine did not answer: {e}"})
                return
            if resp.status != 200:
                try:
                    msg = json.loads(resp.read() or b"{}").get("error", {}).get("message")
                except ValueError:
                    msg = None
                emit({"type": "error", "message": msg or f"the engine answered {resp.status}"})
                return
            usage, finish = None, None
            try:
                for raw in resp:
                    line = raw.decode("utf-8", "replace").strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                    except ValueError:
                        continue
                    usage = chunk.get("usage") or usage
                    for ch in chunk.get("choices") or []:
                        d = ch.get("delta") or {}
                        finish = ch.get("finish_reason") or finish
                        if d.get("reasoning_content") and not emit({"type": "reasoning", "text": d["reasoning_content"]}):
                            g.cancel()
                            return
                        if d.get("content") and not emit({"type": "delta", "text": d["content"]}):
                            g.cancel()                 # the client went away: stop the engine too
                            return
            except (OSError, http.client.HTTPException):
                if not g.cancelled:
                    emit({"type": "error", "message": "the connection to the engine was lost"})
                    return
            emit({"type": "cancelled"} if g.cancelled else {"type": "done", "finish_reason": finish, "usage": usage})
        finally:
            if g.sock is not None:
                g.sock.close()
            with self.lock:
                self.gens.pop(g.id, None)
            self.refresh()


def _engine_error(status, body, what):
    msg = ((body or {}).get("error") or {}).get("message") if isinstance(body, dict) else None
    return f"could not {what}: " + (msg or (f"the engine answered {status}" if status else "the engine did not answer"))


# ───────────────────────────── HTTP ─────────────────────────────
class Handler(http.server.BaseHTTPRequestHandler):
    server_version = "ai-llm-engine-assistant"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):         # no request log: URLs never carry content, but keep it quiet
        pass

    # ── checks ──
    def _origin(self):
        """None for a client without an Origin (curl, the CLI), the origin for a browser extension, or False
        for anything else — a web page must never reach this service."""
        origin = self.headers.get("Origin")
        if not origin:
            return None
        scheme = urllib.parse.urlsplit(origin).scheme
        return origin if scheme in EXTENSION_SCHEMES else False

    def _host_ok(self):
        host = (self.headers.get("Host") or "").rsplit(":", 1)[0].strip("[]").lower()
        return host in ("127.0.0.1", "localhost", "::1")

    def _authed(self):
        got = self.headers.get("Authorization") or ""
        return secrets.compare_digest(got.encode(), f"Bearer {self.server.token}".encode())

    # ── replies ──
    def _cors(self):
        origin = self._origin()
        if origin:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")

    def _json(self, code, obj):
        data = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self._cors()
        self.end_headers()
        self.wfile.write(data)

    def _error(self, code, err, message, **extra):
        self._json(code, {"error": {"code": err, "message": message, **extra}})

    def _start_stream(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self._cors()
        self.end_headers()
        self.close_connection = True

    def _event(self, obj):
        try:
            self.wfile.write(b"data: " + json.dumps(obj, ensure_ascii=False).encode() + b"\n\n")
            self.wfile.flush()
            return True
        except OSError:
            return False

    def _gate(self):
        if not self._host_ok():
            self._error(403, "forbidden", "requests must be addressed to 127.0.0.1")
            return False
        if self._origin() is False:
            self._error(403, "forbidden", "web pages cannot use this service")
            return False
        return True

    # ── routing ──
    def do_OPTIONS(self):
        if not self._host_ok() or not self._origin():
            self.send_response(403)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        self.send_response(204)
        self._cors()
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type")
        self.send_header("Access-Control-Max-Age", "600")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):
        if not self._gate():
            return
        path = urllib.parse.urlsplit(self.path).path.rstrip("/")
        if path == "/v1/assistant/hello":              # discovery: says what is listening, nothing more
            return self._json(200, {"service": "ai-llm-engine-assistant", "protocol": PROTOCOL})
        if not self._authed():
            return self._error(401, "unauthorized", "missing or wrong token")
        a = self.server.assistant
        if path == "/v1/assistant/state":
            return self._json(200, a.refresh())
        if path == "/v1/assistant/events":
            self._start_stream()
            snap = a.refresh()
            if not self._event({"type": "state", "state": snap}):
                return
            while not self.server.stopping.is_set():
                nxt = a.wait_change(snap["version"], 15)
                if nxt and nxt["version"] > snap["version"]:
                    snap = nxt
                    if not self._event({"type": "state", "state": snap}):
                        return
                else:
                    try:
                        self.wfile.write(b": keep-alive\n\n")
                        self.wfile.flush()
                    except OSError:
                        return
            return
        self._error(404, "not_found", "unknown path")

    def do_POST(self):
        if not self._gate():
            return
        if not self._authed():
            return self._error(401, "unauthorized", "missing or wrong token")
        path = urllib.parse.urlsplit(self.path).path.rstrip("/")
        try:
            n = int(self.headers.get("Content-Length") or 0)
            if n < 0 or n > MAX_BODY:
                return self._error(413, "too_large", f"the body must be under {MAX_BODY // 2**20} MB")
            body = json.loads(self.rfile.read(n) or b"{}") if n else {}
            if not isinstance(body, dict):
                raise ValueError
        except ValueError:
            return self._error(400, "bad_request", "the body must be a JSON object")
        a = self.server.assistant
        session = str(body.get("session") or "")[:64]
        confirm = body.get("confirm") is True
        try:
            if path == "/v1/assistant/engine/start":
                return self._json(202, a.start_engine())
            if path == "/v1/assistant/engine/stop":
                return self._json(200, a.stop_engine(session, confirm))
            if path == "/v1/assistant/models/load":
                return self._json(202, a.load())
            if path == "/v1/assistant/models/unload":
                return self._json(202, a.unload(session, confirm))
            if path == "/v1/assistant/models/default":
                return self._json(202, a.set_default(str(body.get("path") or ""), session,
                                                     body.get("expected_default"), confirm))
            if path.startswith("/v1/assistant/generations/") and path.endswith("/cancel"):
                return self._json(200, a.cancel(path.split("/")[-2], session))
            if path == "/v1/assistant/chat":
                started = []

                def emit(ev):
                    if not started:
                        self._start_stream()
                        started.append(True)
                    return self._event(ev)
                try:
                    a.chat(body, emit)
                except Conflict as e:
                    if started:
                        emit({"type": "error", "code": e.code, "message": e.message})
                        return
                    raise
                return
        except Conflict as e:
            code = {"model_in_use": 409, "busy": 409, "stale": 409, "not_yours": 403, "engine_offline": 503,
                    "unknown_model": 404}.get(e.code, 400)
            return self._error(code, e.code, e.message, **e.extra,
                               **({"state": a.snapshot()} if e.code in ("stale", "model_in_use", "busy") else {}))
        self._error(404, "not_found", "unknown path")


class Server(http.server.ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, assistant, token, port=PORT):
        super().__init__(("127.0.0.1", port), Handler)
        self.assistant, self.token = assistant, token
        self.stopping = threading.Event()

    def shutdown(self):
        self.stopping.set()
        with self.assistant.changed:
            self.assistant.changed.notify_all()
        super().shutdown()


def _watch_for_update(assistant, srv, started, interval=20.0):
    """When the code this process was started from changes on disk (an LLMBar update), leave — launchd
    starts the new one, which syncs the new extension. Never in the middle of someone's answer."""
    while not srv.stopping.wait(interval):
        if code_stamp() != started and not assistant.gens and not assistant.op:
            print("the installed code changed: restarting to pick it up", flush=True)
            os._exit(0)


def serve(port=PORT, control=None, token=None):
    """Run until interrupted. Polls the engine once a second for the shared state."""
    assistant = Assistant(control or EngineControl())
    try:
        assistant.extension = sync_extension()
    except OSError as e:
        print(f"could not update the extension's files: {e}", flush=True)
    srv = Server(assistant, token or load_or_create_token(), port)
    threading.Thread(target=assistant.poll_forever, args=(srv.stopping,), daemon=True).start()
    threading.Thread(target=_watch_for_update, args=(assistant, srv, code_stamp()), daemon=True).start()
    print(f"browser assistant service on http://127.0.0.1:{srv.server_address[1]} (protocol {PROTOCOL})", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.stopping.set()
        srv.server_close()


if __name__ == "__main__":
    serve(int(sys.argv[1]) if len(sys.argv) > 1 else PORT)
