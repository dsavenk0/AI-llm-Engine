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
import base64
import hashlib
import http.client
import http.server
import json
import os
import secrets
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.parse
import urllib.request
import uuid

from . import agent, automation, describe, prompts, workspace

PROTOCOL = 1
PORT = 8093
MAX_BODY = 8 * 2**20          # page text and, when asked for, a screenshot
EXTENSION_SCHEMES = ("safari-web-extension", "chrome-extension", "moz-extension")
HOME = os.environ.get("ASSISTANT_HOME") or os.path.expanduser("~/Library/Application Support/AI-llm-Engine")
TOKEN_FILE = os.path.join(HOME, "browser-assistant.json")
POLL_S = 1.0
KEEP_AWAKE_S = 240.0          # how often an open side panel tells the engine it is in use
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXT_BUILT = os.path.join(ROOT, "browser-assistant", "dist")      # what this copy of the code ships
EXT_HOME = os.path.join(HOME, "browser-assistant")                # what the browser is pointed at


def _frame_meta(image, supplied):
    """Opaque frame identity plus the extension's tiny perceptual signature; never retains screenshot bytes."""
    if not image: return None
    try: raw = base64.b64decode(image.split(",", 1)[1], validate=True)
    except Exception: raise Conflict("bad_request", "invalid screenshot encoding") from None
    supplied = supplied if isinstance(supplied, dict) else {}
    width, height = supplied.get("width"), supplied.get("height")
    signature = supplied.get("signature")
    if not isinstance(width, int) or not 0 <= width <= 10_000: width = 0
    if not isinstance(height, int) or not 0 <= height <= 10_000: height = 0
    if not isinstance(signature, str) or not re_full_hex(signature, 34): signature = ""
    return {"id": hashlib.sha256(raw).hexdigest()[:24], "bytes": len(raw), "width": width, "height": height,
            "signature": signature}


def re_full_hex(value, max_len):
    return 1 <= len(value) <= max_len and all(c in "0123456789abcdef" for c in value.lower())


def _frame_changed(before, after):
    if not before or not after: return True
    if before["width"] and after["width"] and (before["width"], before["height"]) != (after["width"], after["height"]):
        return True
    a, b = before.get("signature", ""), after.get("signature", "")
    if a and b and len(a) == len(b):
        different = sum((int(x, 16) ^ int(y, 16)).bit_count() for x, y in zip(a, b))
        if different / (len(a) * 4) > 0.34: return True
    lo, hi = sorted((max(1, before["bytes"]), max(1, after["bytes"])))
    return hi / lo > 1.7


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
    for browser in ("safari", "chromium", "firefox"):
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

    def vision_model(self):
        """Configured vision role, or the largest installed vision model that safely fits memory."""
        try:
            from harness import conductor
            spec, _ = conductor.resolve_roles(self.llm.BASE, local=None)
            name = (spec.get("vision") or {}).get("local")
        except Exception:
            name = None
        for model in self.models():
            if name and model["vision"] and model["supported"] and (model["name"] == name or model["path"] == name):
                return model["path"]
        choices = [m for m in self.models() if m["vision"] and m["supported"] and m["fits"]]
        return max(choices, key=lambda m: (m["favorite"], m["size_gb"]), default={}).get("path")

    def vision_source(self):
        try:
            from harness import conductor
            raw = conductor.load_mapping(self.llm.BASE).get("vision")
            configured = raw.get("local") if isinstance(raw, dict) else None
        except Exception: configured = None
        return "configured" if configured else "automatic"

    def install_vision_model(self):
        """Find and download a compatible Qwen vision model that fits this Mac. Called only by the explicit
        install button; searches current Hub metadata instead of baking a repository revision into the app."""
        existing = self.vision_model()
        if existing:
            return existing
        choices = []
        errors = []
        for query in ("Qwen3.5-VL-4B-Instruct", "Qwen3-VL-4B-Instruct", "Qwen3-VL-2B-Instruct"):
            try:
                found = self.llm.hf_search(query, limit=12)
            except Exception as e:
                errors.append(str(e)); continue
            choices.extend(x for x in found if x.get("compatible") and x.get("fits") and x.get("size"))
            if choices: break
        if not choices:
            raise RuntimeError("no compatible vision model that fits this Mac was found" +
                               (f": {errors[-1]}" if errors else ""))
        # Search results are popularity-sorted. Prefer a complete local result, then the first current, fitting
        # result; do not silently choose a larger model merely because more RAM happens to be free today.
        choice = next((x for x in choices if x.get("local")), choices[0])
        ok, out = self.run(["pull", choice["id"]], timeout=7200)
        if not ok:
            raise RuntimeError(out or "the vision model download failed")
        self._models_at = 0
        model = self.vision_model()
        if not model:
            raise RuntimeError("the downloaded model is not supported as a vision model")
        return model

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
                                     headers={"Content-Type": "application/json", "X-Priority": "interactive",
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
                              "X-Priority": "interactive",      # someone is reading the answer in the browser
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
        self.watchers = 0           # open event streams: a side panel (or popup) someone has in front of them
        self.kept_awake = 0.0
        self.automation = automation.Automation(ROOT, HOME)
        self.workspace = workspace.Workspace(HOME, ROOT)
        self.visual_actions = {}  # opaque ticket -> action issued by agent_step; expires in a few seconds
        self.workspace_actions = {}  # opaque ticket -> narrow Workspace operation issued by agent_step
        self.workspace_undos = {}    # one-use, in-memory reversals; never OAuth credentials
        self.vision_lock = threading.Lock()

    def installed_versions(self):
        """The extension's version in each browser's folder, read now: `llm assistant setup` or an LLMBar update
        can replace the files while this service keeps running, and an extension reloads itself when it sees a
        newer one here (browser-assistant/src/background/main.ts)."""
        out = dict(self.extension)
        for browser in ("safari", "chromium", "firefox"):
            v = _manifest_version(os.path.join(EXT_HOME, browser))
            if v:
                out[browser] = v
        return out

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
            "extension": {"versions": self.installed_versions(), "folder": EXT_HOME},
            "capabilities": {
                "unload": self.unload_supported,
                "per_session_model": False,
                "per_session_model_reason": "the engine serves one model at a time: a different model for one "
                                            "session would mean switching it for every client",
                "hands": self.automation.status(),
                "eyes": {"model": getattr(self.control, "vision_model", lambda: None)(),
                         "source": getattr(self.control, "vision_source", lambda: "automatic")()},
                "workspace": self.workspace.status(),
                "dictation": {"installed": self._dictation_available(), "local": True},
            },
        }

    @staticmethod
    def _dictation_available():
        try:
            from engine import audio
            return audio.whisper_available()
        except Exception:
            return False

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

    def keep_awake(self, now=None):
        """While a side panel is open, the engine must not unload the model for being idle between questions —
        the conversation is still going on. Every KEEP_AWAKE_S the service tells an awake engine it is in use;
        an engine already asleep is left asleep (it loads again with the next question)."""
        now = time.time() if now is None else now
        if self.watchers <= 0 or now - self.kept_awake < KEEP_AWAKE_S:
            return False
        h = self.control.health()
        if not h or h.get("state") == "sleeping":
            return False
        self.kept_awake = now
        self.control.post("/v1/engine/wake", {})
        return True

    def poll_forever(self, stop):
        while not stop.is_set():
            try:
                self.refresh()
                self.keep_awake()
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

    def _complete(self, g, health, messages, max_tokens):
        """The text of one non-streamed-to-the-client completion that asks for a JSON object, as one string.
        `g` is registered, so it can be cancelled like any generation."""
        body = {"model": health.get("model"), "messages": messages, "stream": True, "max_tokens": max_tokens,
                "temperature": 0.2, "response_format": {"type": "json_object"},
                "chat_template_kwargs": {"enable_thinking": False}}
        try:
            g.sock, resp = self.control.open_chat(body)
        except OSError as e:
            raise Conflict("engine_error", f"the engine did not answer: {e}") from None
        if resp.status != 200:
            try:
                msg = json.loads(resp.read() or b"{}").get("error", {}).get("message")
            except ValueError:
                msg = None
            raise Conflict("engine_error", msg or f"the engine answered {resp.status}")
        out = []
        try:
            for chunk in _sse_lines(resp):
                for ch in chunk.get("choices") or []:
                    piece = (ch.get("delta") or {}).get("content")
                    if piece:
                        out.append(piece)
        except (OSError, http.client.HTTPException):
            if not g.cancelled:
                raise Conflict("engine_error", "the connection to the engine was lost") from None
        finally:
            if g.sock is not None:
                g.sock.close()
                g.sock = None
        return "".join(out)

    def agent_step(self, req):
        """One step of the browser agent: the model's next action for this page and goal, validated. The
        extension decides whether it may run and asks the person; nothing here touches a browser."""
        import re
        session = str(req.get("session") or "")[:64]
        if not session:
            raise Conflict("bad_request", "a session id is needed")
        if self.op == "vision":
            raise Conflict("busy", "the vision model is examining a page; try again in a moment", operation="vision")
        h = self.control.health()
        if not h:
            raise Conflict("engine_offline", "the engine is not running — start it first")
        restore = None
        role_locked = False
        # The configured vision role can be a different local model.  Local models take turns: while it is
        # loaded we refuse other assistant work, then restore the person's text-model default in finally.
        image_for_role = req.get("screenshot")
        vision_path = getattr(self.control, "vision_model", lambda: None)() if image_for_role and not h.get("vision") else None
        if vision_path and vision_path != self.control.default():
            if not self.vision_lock.acquire(blocking=False):
                vision_path = None
            else:
                role_locked = True
                try:
                    self._guard(session, False, "switching to the vision model")
                    restore = self.control.default()
                    self._begin("vision")
                    ok, out = self.control.run(["use", vision_path])
                    h = self.control.health() or h
                    if not ok or not h.get("vision"):
                        # Continue with the original model plus local OCR instead of leaving the service wedged.
                        if restore and self.control.default() == vision_path:
                            restored, restore_out = self.control.run(["use", restore])
                            if not restored:
                                out = (out + "\n" if out else "") + (restore_out or "could not restore the text model")
                        self._end(out or "could not load the configured vision model")
                        self.vision_lock.release()
                        role_locked, restore = False, None
                except Conflict:
                    self.vision_lock.release(); role_locked = False; raise
                except Exception as e:
                    if restore and self.control.default() == vision_path:
                        try: self.control.run(["use", restore])
                        except Exception: pass
                    if self.op == "vision":
                        self._end(f"could not load the configured vision model: {e}")
                    self.vision_lock.release(); role_locked = False
                    raise Conflict("engine_error", "could not load the configured vision model") from None

        def restore_text_role():
            nonlocal role_locked
            if not role_locked:
                return
            error = None
            try:
                # Do not silently overwrite a model choice someone made while the visual step ran.
                if restore and self.control.default() == vision_path:
                    ok, out = self.control.run(["use", restore])
                    if not ok:
                        error = out or "could not restore the text model"
            finally:
                self._end(error)
                self.vision_lock.release()
                role_locked = False
        try:
            image = prompts.image_url(req.get("screenshot"))
            frame_meta = _frame_meta(image, req.get("frame"))
            # A vision model receives the pixels. For a text-only model, provide local OCR and scene labels so
            # the agent can still use visible text and controls instead of silently ignoring the screenshot.
            shot = image if h.get("vision") else None
            visual_description = None
            if image and not h.get("vision"):
                visual_description = describe.as_text(describe.describe([image]), True)
            messages, snap = agent.build_messages(req.get("goal"), req.get("snapshot"), req.get("history"),
                                                  shot, visual_description, self.workspace.status().get("connected", False))
        except (agent.AgentError, prompts.PromptError) as e:
            restore_text_role()
            raise Conflict("bad_request", str(e)) from None
        g = Generation(session)
        rid = req.get("request_id")
        if isinstance(rid, str) and re.match(r"^[a-z0-9-]{8,40}$", rid):
            g.id = rid                        # the caller can name it, so it can cancel before the answer comes
        with self.lock:
            self.gens[g.id] = g
        self.refresh()
        try:
            problem = ""
            for attempt in (1, 2):
                text = self._complete(g, h, messages, 500)
                if g.cancelled:
                    raise Conflict("cancelled", "stopped")
                try:
                    action = agent.validate(agent.parse_action(text), snap, allow_visual=shot is not None)
                    if action["action"].startswith("visual_"):
                        ticket = secrets.token_urlsafe(24)
                        with self.lock:
                            self.visual_actions[ticket] = {"at": time.time(), "session": session,
                                                           "url": snap["url"], "tab_id": req.get("tab_id"),
                                                           "frame": frame_meta, "action": action}
                        action = dict(action, ticket=ticket)
                    elif action["action"].startswith("workspace_"):
                        ticket = secrets.token_urlsafe(24)
                        with self.lock:
                            self.workspace_actions[ticket] = {"at": time.time(), "session": session,
                                                              "url": snap["url"], "action": action}
                        action = dict(action, ticket=ticket)
                    return {"action": action, "attempts": attempt,
                            "saw_screenshot": shot is not None, "described_screenshot": visual_description is not None}
                except agent.AgentError as e:
                    problem = str(e)
                    messages = messages + [
                        {"role": "assistant", "content": text[:1500]},
                        {"role": "user", "content": f"That was not usable: {problem}. Reply with one valid JSON "
                                                    f"action, using only refs from the list."}]
            raise Conflict("bad_action", f"the model did not give a usable action: {problem}")
        finally:
            with self.lock:
                self.gens.pop(g.id, None)
            self.refresh()
            restore_text_role()

    def visual_execute(self, req):
        """Run a short-lived visual action through the private macOS helper.

        A ticket can only have been produced by this service for the same browser session and page.  The extension
        cannot ask the helper to move or type arbitrary input, and the helper independently refuses a background
        or non-browser window.
        """
        session = str(req.get("session") or "")[:64]
        ticket = str(req.get("ticket") or "")[:128]
        now = time.time()
        with self.lock:
            item = self.visual_actions.pop(ticket, None)
            # Bound memory even when a client abandons a task.
            self.visual_actions = {k: v for k, v in self.visual_actions.items() if now - v["at"] < 20}
        if not item or item["session"] != session or now - item["at"] > 12:
            raise Conflict("stale_visual_action", "that visual step expired; look at the page again")
        if str(req.get("url") or "") != item["url"]:
            raise Conflict("stale_visual_action", "the page changed since the visual step was proposed")
        if req.get("tab_id") != item.get("tab_id"):
            raise Conflict("stale_visual_action", "the active tab changed since the visual step was proposed")
        fresh = req.get("screenshot")
        if not isinstance(fresh, str) or not fresh.startswith("data:image/") or len(fresh) > MAX_BODY:
            raise Conflict("bad_request", "a fresh screenshot is required before a visual action")
        try: fresh_image = prompts.image_url(fresh)
        except prompts.PromptError as e: raise Conflict("bad_request", str(e)) from None
        fresh_meta = _frame_meta(fresh_image, req.get("frame"))
        if _frame_changed(item.get("frame"), fresh_meta):
            raise Conflict("stale_visual_action", "the visible page changed substantially; look at it again")
        browser = str(req.get("browser") or "")
        # The helper resolves the active member of the Chromium family itself, so Edge/Brave/Arc/Vivaldi cannot
        # be impersonated by a stale browser name from the extension.
        bundles = {"safari": "com.apple.Safari", "chromium": "chromium", "firefox": "org.mozilla.firefox"}
        bundle = bundles.get(browser)
        if not bundle:
            raise Conflict("bad_request", "unknown browser for visual action")
        action = item["action"]
        try:
            if action["action"] == "visual_click":
                body = {"action": "click", "bundle": bundle, "x": action["x"], "y": action["y"]}
            elif action["action"] == "visual_type":
                # Focus a visible target first.  The value is never logged by this service.
                focus = self.automation.request({"action": "click", "bundle": bundle, "x": action["x"], "y": action["y"]})
                if not focus.get("ok"):
                    return {"ok": False, "error": focus.get("error") or "could not focus the visual target"}
                body = {"action": "type", "bundle": bundle, "value": action["value"]}
            elif action["action"] == "visual_press":
                body = {"action": "press", "bundle": bundle, "key": action["key"]}
            else:
                raise Conflict("bad_action", "this is not a visual action")
            out = self.automation.request(body)
        except automation.AutomationError as e:
            return {"ok": False, "error": str(e), "route": "macos"}
        return {"ok": bool(out.get("ok")), "error": out.get("error"), "route": out.get("route") or "macos"}

    def workspace_configure(self, req):
        try: return self.workspace.configure(req.get("client_id"))
        except workspace.WorkspaceError as e: raise Conflict("workspace", str(e)) from None

    def accessibility_settings(self):
        try: return self.automation.open_settings()
        except OSError as e: raise Conflict("automation", f"could not open Accessibility settings: {e}") from None

    def workspace_authorize(self, req):
        try: return self.workspace.authorize(req.get("scopes") if isinstance(req.get("scopes"), list) else [])
        except workspace.WorkspaceError as e: raise Conflict("workspace", str(e)) from None

    def workspace_disconnect(self):
        return self.workspace.disconnect()

    def vision_configure(self, req):
        path = str(req.get("path") or "")
        model = next((m for m in self.control.models() if m["path"] == path and m["vision"] and m["supported"]), None)
        if not model: raise Conflict("bad_request", "choose an installed vision-capable model")
        from harness import conductor
        conductor.set_role(ROOT, "vision", {"local": path})
        return {"model": path, "source": "configured"}

    def vision_install(self):
        try:
            path = self.control.install_vision_model()
            from harness import conductor
            conductor.set_role(ROOT, "vision", {"local": path})
            return {"model": path, "source": "configured"}
        except Exception as e:
            raise Conflict("vision_install", str(e)) from None

    def workspace_execute(self, req):
        session, ticket = str(req.get("session") or "")[:64], str(req.get("ticket") or "")[:128]
        now = time.time()
        with self.lock:
            item = self.workspace_actions.pop(ticket, None)
            self.workspace_actions = {k: v for k, v in self.workspace_actions.items() if now - v["at"] < 300}
        if not item or item["session"] != session or now - item["at"] > 300:
            raise Conflict("stale_workspace_action", "that Workspace step expired; ask the model to prepare it again")
        if str(req.get("url") or "") != item["url"]:
            raise Conflict("stale_workspace_action", "the page changed since the Workspace step was proposed")
        try: out = self.workspace.execute(item["action"])
        except workspace.WorkspaceError as e: raise Conflict("workspace", str(e)) from None
        undo, label = out.pop("_undo", None), out.pop("_undo_label", None)
        if undo:
            opaque = secrets.token_urlsafe(24)
            with self.lock:
                self.workspace_undos[opaque] = {"at": now, "session": session, "record": undo}
            out.update(undo=opaque, undo_label=label or "Undo Workspace change")
        return out

    def workspace_undo(self, req):
        session, token = str(req.get("session") or "")[:64], str(req.get("undo") or "")[:128]
        now = time.time()
        with self.lock:
            item = self.workspace_undos.pop(token, None)
            self.workspace_undos = {k: v for k, v in self.workspace_undos.items() if now - v["at"] < 600}
        if not item or item["session"] != session or now - item["at"] > 600:
            raise Conflict("stale_workspace_action", "that undo expired or was already used")
        try: return self.workspace.undo(item["record"])
        except workspace.WorkspaceError as e: raise Conflict("workspace", str(e)) from None

    def forget_session(self, req):
        session = str(req.get("session") or "")[:64]
        if not session: raise Conflict("bad_request", "a session id is needed")
        with self.lock:
            before = len(self.visual_actions) + len(self.workspace_actions) + len(self.workspace_undos)
            self.visual_actions = {k: v for k, v in self.visual_actions.items() if v["session"] != session}
            self.workspace_actions = {k: v for k, v in self.workspace_actions.items() if v["session"] != session}
            self.workspace_undos = {k: v for k, v in self.workspace_undos.items() if v["session"] != session}
            removed = before - len(self.visual_actions) - len(self.workspace_actions) - len(self.workspace_undos)
        return {"forgotten": removed}

    def audio_setup(self):
        """Install the optional local audio environment after an explicit click in the extension."""
        if self._dictation_available():
            return {"ready": True}
        ok, out = self.control.run(["setup", "--audio"], timeout=1800)
        if not ok or not self._dictation_available():
            raise Conflict("audio", out or "could not install local dictation")
        self.refresh()
        return {"ready": True}

    def transcribe(self, req):
        """Decode one bounded browser recording, transcribe it locally, and always erase the temporary file."""
        encoded = req.get("audio")
        if not isinstance(encoded, str) or not encoded.startswith("data:audio/") or "," not in encoded:
            raise Conflict("bad_request", "audio must be an audio data URL")
        header, payload = encoded.split(",", 1)
        mime = header[5:].split(";", 1)[0].lower()
        suffixes = {"audio/webm": ".webm", "audio/ogg": ".ogg", "audio/mp4": ".m4a", "audio/mpeg": ".mp3",
                    "audio/wav": ".wav", "audio/x-wav": ".wav", "audio/aac": ".aac"}
        if mime not in suffixes or ";base64" not in header.lower():
            raise Conflict("bad_request", "unsupported audio recording format")
        try:
            raw = base64.b64decode(payload, validate=True)
        except (ValueError, TypeError):
            raise Conflict("bad_request", "invalid audio encoding") from None
        if not raw or len(raw) > 5 * 2**20:
            raise Conflict("bad_request", "the recording must be between 1 byte and 5 MB")
        from engine import audio
        if not audio.whisper_available():
            raise Conflict("audio_not_installed", "local dictation is not installed yet")
        path = None
        try:
            with tempfile.NamedTemporaryFile(prefix="llmbar-dictation-", suffix=suffixes[mime], delete=False) as f:
                os.chmod(f.name, 0o600)
                f.write(raw)
                path = f.name
            result = audio.transcribe(path)
            return {"text": result.get("text") or "", "language": result.get("language")}
        except audio.AudioError as e:
            raise Conflict("audio", str(e)) from None
        finally:
            if path:
                try: os.unlink(path)
                except OSError: pass

    def chat(self, req, emit):
        """Run one task. `emit(event)` sends one event to the client and returns False once it has gone."""
        session = str(req.get("session") or "")[:64]
        if not session:
            raise Conflict("bad_request", "a session id is needed")
        if self.op == "vision":
            raise Conflict("busy", "the vision model is examining a page; try again in a moment", operation="vision")
        h = self.control.health()
        if not h:
            raise Conflict("engine_offline", "the engine is not running — start it first")
        max_tokens = max(64, min(int(req.get("max_tokens") or 1024), 4096))
        if (req.get("image") or req.get("images")) and not h.get("vision"):
            # a model without eyes gets the pictures in words: Apple Vision's labels and the text it read
            try:
                pics = prompts.images_of(req)
            except prompts.PromptError as e:
                raise Conflict("bad_request", str(e)) from None
            words = describe.as_text(describe.describe(pics), bool(req.get("image")))
            req = {k: v for k, v in req.items() if k not in ("image", "images")}
            req["described"] = words
        try:
            messages, shared = prompts.build(dict(req, can_see=bool(h.get("vision"))), int(h.get("context") or 4096),
                                             max_tokens)
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


def _sse_lines(resp):
    """The decoded JSON of each `data:` line of a streamed response, until [DONE]."""
    for raw in resp:
        line = raw.decode("utf-8", "replace").strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            return
        try:
            yield json.loads(data)
        except ValueError:
            continue


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
        if path == "/v1/assistant/workspace/status":
            return self._json(200, a.workspace.status())
        if path == "/v1/assistant/events":
            self._start_stream()
            with a.lock:
                a.watchers += 1                       # someone has a panel open: keep the model loaded
            try:
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
            finally:
                with a.lock:
                    a.watchers -= 1
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
            if path == "/v1/assistant/agent/step":
                return self._json(200, a.agent_step(body))
            if path == "/v1/assistant/agent/visual/execute":
                return self._json(200, a.visual_execute(body))
            if path == "/v1/assistant/accessibility/settings":
                return self._json(200, a.accessibility_settings())
            if path == "/v1/assistant/workspace/configure":
                return self._json(200, a.workspace_configure(body))
            if path == "/v1/assistant/vision/configure":
                return self._json(200, a.vision_configure(body))
            if path == "/v1/assistant/vision/install":
                return self._json(200, a.vision_install())
            if path == "/v1/assistant/workspace/authorize":
                return self._json(202, a.workspace_authorize(body))
            if path == "/v1/assistant/workspace/disconnect":
                return self._json(200, a.workspace_disconnect())
            if path == "/v1/assistant/workspace/execute":
                return self._json(200, a.workspace_execute(body))
            if path == "/v1/assistant/workspace/undo":
                return self._json(200, a.workspace_undo(body))
            if path == "/v1/assistant/session/forget":
                return self._json(200, a.forget_session(body))
            if path == "/v1/assistant/audio/setup":
                return self._json(200, a.audio_setup())
            if path == "/v1/assistant/audio/transcribe":
                return self._json(200, a.transcribe(body))
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
                    "unknown_model": 404, "bad_action": 422, "cancelled": 409, "engine_error": 502,
                    "stale_visual_action": 409, "stale_workspace_action": 409, "workspace": 422,
                    "audio": 422, "audio_not_installed": 501, "vision_install": 502}.get(e.code, 400)
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
