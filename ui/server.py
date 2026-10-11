"""`./llm ui` backend: a small localhost HTTP server for the code-editor UI (stdlib only).

    python -m ui.server --root <project> [--port 0] [--engine-port 8090] [--open]

Everything is confined to the project folder; the API needs the random token printed at start, and
requests must come from localhost (Host/Origin checked, so a web page cannot drive it).
"""
import argparse
import json
import os
import re
import secrets
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)
from ui import agentloop, codeinfo  # noqa: E402

SKIP_DIRS = {".git", ".llm", "node_modules", "__pycache__", ".venv", "venv", ".build", "DerivedData", ".idea",
             ".DS_Store"}
MAX_FILE = 2_000_000          # larger files are listed but not opened
MAX_ATTACH_TOTAL = 200_000    # bytes of attachments inlined into one prompt
MAX_BODY = 8_000_000
GIT_TIMEOUT = 60


class Project:
    """The workspace root and every filesystem/git operation on it."""

    skip = SKIP_DIRS
    max_file = MAX_FILE

    def __init__(self, root):
        self.root = os.path.realpath(root)

    def search(self, pattern, rel=".", glob=None, limit=200):
        """path:line: text for lines matching a regex (a plain substring when it is not a valid regex)."""
        import fnmatch
        try:
            rx = re.compile(pattern)
        except re.error:
            rx = re.compile(re.escape(pattern))
        out = []
        for d, dirs, files in os.walk(self.resolve(rel)):
            dirs[:] = sorted(x for x in dirs if x not in SKIP_DIRS and not x.startswith("."))
            for f in sorted(files):
                if glob and not fnmatch.fnmatch(f, glob):
                    continue
                full = os.path.join(d, f)
                try:
                    if os.path.getsize(full) > MAX_FILE:
                        continue
                    with open(full, encoding="utf-8") as fh:
                        for n, line in enumerate(fh, 1):
                            if rx.search(line):
                                out.append(f"{self.rel(full)}:{n}: {line.rstrip()[:200]}")
                                if len(out) >= limit:
                                    return out
                except (UnicodeDecodeError, OSError):
                    continue
        return out

    # ---- paths
    def resolve(self, rel, must_exist=True):
        full = os.path.realpath(os.path.join(self.root, rel or "."))
        if full != self.root and not full.startswith(self.root + os.sep):
            raise PermissionError("path is outside the project")
        parts = os.path.relpath(full, self.root).split(os.sep)
        if ".git" in parts:
            raise PermissionError("the .git folder is not accessible")
        if must_exist and not os.path.exists(full):
            raise FileNotFoundError(rel)
        return full

    def rel(self, full):
        return os.path.relpath(full, self.root)

    # ---- tree / files
    def tree(self, rel=""):
        base = self.resolve(rel)
        if not os.path.isdir(base):
            raise NotADirectoryError(rel)
        out = []
        for name in sorted(os.listdir(base), key=lambda n: (not os.path.isdir(os.path.join(base, n)), n.lower())):
            if name in SKIP_DIRS:
                continue
            p = os.path.join(base, name)
            try:
                st = os.stat(p)
            except OSError:
                continue
            out.append({"name": name, "path": self.rel(p), "type": "dir" if os.path.isdir(p) else "file",
                        "size": st.st_size})
        return out

    def read(self, rel):
        full = self.resolve(rel)
        if os.path.isdir(full):
            raise IsADirectoryError(rel)
        st = os.stat(full)
        if st.st_size > MAX_FILE:
            return {"path": self.rel(full), "too_large": True, "size": st.st_size}
        with open(full, "rb") as f:
            raw = f.read()
        if b"\x00" in raw[:8192]:
            return {"path": self.rel(full), "binary": True, "size": st.st_size}
        text = raw.decode("utf-8", errors="replace")
        lang = codeinfo.detect_language(full, text)
        return {"path": self.rel(full), "text": text, "language": lang, "mtime": st.st_mtime_ns,
                "size": st.st_size, "outline": codeinfo.outline(text, lang),
                "spans": codeinfo.highlight(text, lang) if len(text) < 400_000 else []}

    def write(self, rel, text, expected_mtime=None):
        full = self.resolve(rel, must_exist=False)
        if os.path.isdir(full):
            raise IsADirectoryError(rel)
        if expected_mtime is not None and os.path.exists(full) and os.stat(full).st_mtime_ns != int(expected_mtime):
            raise FileExistsError("the file changed on disk since it was opened")
        os.makedirs(os.path.dirname(full), exist_ok=True)
        tmp = full + ".ui-tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, full)
        return os.stat(full).st_mtime_ns

    # ---- git
    def git(self, *args, timeout=GIT_TIMEOUT, check=False):
        env = dict(os.environ, GIT_TERMINAL_PROMPT="0", LC_ALL="C")
        p = subprocess.run(["git", "-C", self.root, *args], capture_output=True, text=True, timeout=timeout, env=env)
        if check and p.returncode != 0:
            raise RuntimeError((p.stderr or p.stdout).strip() or f"git {args[0]} failed")
        return p

    def is_repo(self):
        return self.git("rev-parse", "--is-inside-work-tree").stdout.strip() == "true"

    def git_status(self):
        if not self.is_repo():
            return {"repo": False}
        p = self.git("status", "--porcelain=v1", "-b", check=True)
        lines = p.stdout.splitlines()
        head = lines[0][3:] if lines else ""
        branch, ahead, behind, upstream = head, 0, 0, None
        m = re.match(r"(.+?)\.\.\.(\S+)(?: \[(.*)\])?$", head)
        if m:
            branch, upstream = m.group(1), m.group(2)
            for part in (m.group(3) or "").split(","):
                part = part.strip()
                if part.startswith("ahead "):
                    ahead = int(part[6:])
                elif part.startswith("behind "):
                    behind = int(part[7:])
        elif head.startswith("No commits yet on "):
            branch = head[len("No commits yet on "):]
        files = []
        for line in lines[1:]:
            code, path = line[:2], line[3:]
            if " -> " in path:
                path = path.split(" -> ")[1]
            files.append({"path": path.strip('"'), "index": code[0], "worktree": code[1],
                          "status": "untracked" if code == "??" else "modified" if "M" in code else
                          "added" if "A" in code else "deleted" if "D" in code else "renamed" if "R" in code else code})
        remotes = self.git("remote").stdout.split()
        return {"repo": True, "branch": branch, "upstream": upstream, "ahead": ahead, "behind": behind,
                "files": files, "remotes": remotes}

    def git_diff(self, rel):
        self.resolve(rel, must_exist=False)
        p = self.git("diff", "HEAD", "--", rel)
        if not p.stdout.strip():
            p = self.git("diff", "--no-index", "--", os.devnull, os.path.join(self.root, rel)) \
                if os.path.exists(os.path.join(self.root, rel)) else p
        return p.stdout

    def git_commit(self, message, paths=None, all_changes=False):
        message = (message or "").strip()
        if not message:
            raise ValueError("a commit message is required")
        if not self.is_repo():
            raise RuntimeError("not a git repository")
        if all_changes:
            self.git("add", "-A", check=True)
        elif paths:
            for rel in paths:
                self.resolve(rel, must_exist=False)
            self.git("add", "--", *paths, check=True)
        else:
            raise ValueError("choose files to commit (or all changes)")
        if not self.git("diff", "--cached", "--name-only").stdout.strip():
            raise RuntimeError("nothing staged to commit")
        self.git("commit", "-m", message, check=True)
        return self.git("log", "-1", "--format=%h %s").stdout.strip()

    def git_push(self, remote=None, branch=None):
        st = self.git_status()
        if not st.get("repo"):
            raise RuntimeError("not a git repository")
        if not st["remotes"]:
            raise RuntimeError("no remote configured (git remote add origin <url>)")
        remote = remote or ("origin" if "origin" in st["remotes"] else st["remotes"][0])
        if remote not in st["remotes"]:
            raise ValueError(f"unknown remote {remote!r}")
        branch = branch or st["branch"]
        args = ["push", remote, branch] if st["upstream"] else ["push", "-u", remote, branch]
        p = self.git(*args)
        out = (p.stdout + p.stderr).strip()
        if p.returncode != 0:
            raise RuntimeError(out or "git push failed")
        return out or "pushed"

    # ---- prompt attachments
    def attachments(self, paths):
        """Files inlined into a prompt: [(path, text)], refusing anything outside the project, binary or huge."""
        out, total = [], 0
        for rel in paths or []:
            full = self.resolve(rel)
            if os.path.isdir(full):
                raise ValueError(f"{rel} is a folder")
            if os.path.getsize(full) > MAX_FILE:
                raise ValueError(f"{rel} is too large to attach")
            with open(full, "rb") as f:
                raw = f.read()
            if b"\x00" in raw[:8192]:
                raise ValueError(f"{rel} is a binary file")
            total += len(raw)
            if total > MAX_ATTACH_TOTAL:
                raise ValueError(f"attachments exceed {MAX_ATTACH_TOTAL // 1000} KB; attach fewer or smaller files")
            out.append((self.rel(full), raw.decode("utf-8", errors="replace")))
        return out


def attachment_block(items):
    parts = []
    for path, text in items:
        fence = "````" if "```" in text else "```"
        lang = codeinfo.detect_language(path, text)
        parts.append(f"File `{path}`:\n{fence}{'' if lang == 'plaintext' else lang}\n{text.rstrip()}\n{fence}")
    return "\n\n".join(parts)


# ───────────────────────────── HTTP ─────────────────────────────
class Handler(BaseHTTPRequestHandler):
    server_version = "llm-ui"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        pass

    # ---- helpers
    def _headers(self, code, ctype, length=None, extra=None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        if length is not None:
            self.send_header("Content-Length", str(length))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()

    def _json(self, code, obj):
        body = json.dumps(obj, ensure_ascii=False).encode()
        self._headers(code, "application/json; charset=utf-8", len(body))
        self.wfile.write(body)

    def _error(self, code, message):
        self._json(code, {"error": message})

    def _host_ok(self):
        host = (self.headers.get("Host") or "").rsplit(":", 1)[0].strip("[]").lower()
        if host not in ("127.0.0.1", "localhost", "::1"):
            return False
        origin = self.headers.get("Origin")
        if origin:
            o = urllib.parse.urlsplit(origin)
            if (o.hostname or "") not in ("127.0.0.1", "localhost", "::1") or o.port != self.server.server_address[1]:
                return False
        return True

    def _authed(self):
        tok = self.headers.get("X-UI-Token") or urllib.parse.parse_qs(urllib.parse.urlsplit(self.path).query).get("token", [""])[0]
        return secrets.compare_digest(tok or "", self.server.token)

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        if n > MAX_BODY:
            raise ValueError("request body too large")
        raw = self.rfile.read(n) if n else b""
        return json.loads(raw) if raw else {}

    # ---- routing
    def do_GET(self):
        self._route("GET")

    def do_POST(self):
        self._route("POST")

    def do_PUT(self):
        self._route("PUT")

    def _route(self, method):
        if not self._host_ok():
            return self._error(403, "requests must come from localhost")
        u = urllib.parse.urlsplit(self.path)
        path, q = u.path, {k: v[0] for k, v in urllib.parse.parse_qs(u.query).items()}
        if method == "GET" and path == "/":
            body = b"AI-llm-Engine editor backend. Use the native editor app (./llm ui).\n"
            self._headers(200, "text/plain; charset=utf-8", len(body))
            return self.wfile.write(body)
        if not path.startswith("/api/"):
            return self._error(404, "not found")
        if not self._authed():
            return self._error(401, "missing or wrong token")
        P = self.server.project
        try:
            body = self._body() if method in ("POST", "PUT") else {}
            handler = getattr(self, "api_" + path[5:].replace("/", "_"), None)
            if handler is None:
                return self._error(404, f"unknown endpoint {path}")
            return handler(P, q, body)
        except PermissionError as e:
            return self._error(403, str(e))
        except (FileNotFoundError, NotADirectoryError, IsADirectoryError) as e:
            return self._error(404, f"not found: {e}")
        except FileExistsError as e:
            return self._error(409, str(e))
        except (ValueError, json.JSONDecodeError) as e:
            return self._error(400, str(e))
        except subprocess.TimeoutExpired:
            return self._error(504, "git timed out")
        except RuntimeError as e:
            return self._error(500, str(e))

    # ---- API
    def api_info(self, P, q, body):
        eng = None
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{self.server.engine_port}/health", timeout=1.5) as r:
                h = json.load(r)
            eng = {"model": h.get("model"), "state": h.get("state"), "context": h.get("context")}
        except (OSError, ValueError):
            pass
        return self._json(200, {"root": P.root, "name": os.path.basename(P.root), "engine": eng,
                                "git": P.git_status()})

    def api_tree(self, P, q, body):
        return self._json(200, {"path": q.get("path", ""), "entries": P.tree(q.get("path", ""))})

    def api_file(self, P, q, body):
        if self.command == "GET":
            return self._json(200, P.read(q.get("path", "")))
        if self.command == "PUT":
            mtime = P.write(body.get("path", ""), body.get("text", ""), body.get("expected_mtime"))
            return self._json(200, {"ok": True, "mtime": mtime})
        return self._error(405, "method not allowed")

    def api_analyze(self, P, q, body):
        text = body.get("text", "")
        if len(text) > MAX_FILE:
            raise ValueError("text too large to analyze")
        lang = body.get("language") or codeinfo.detect_language(body.get("path", ""), text)
        return self._json(200, {"language": lang, "spans": codeinfo.highlight(text, lang),
                                "outline": codeinfo.outline(text, lang)})

    def api_search(self, P, q, body):
        pattern = q.get("q", "")
        if not pattern:
            raise ValueError("empty query")
        hits = P.search(pattern, q.get("path", "."), q.get("glob"), limit=200)
        return self._json(200, {"hits": [dict(zip(("path", "line", "text"), h.split(":", 2))) for h in hits]})

    def api_agent_answer(self, P, q, body):
        run = agentloop.RUNS.get(body.get("run", ""))
        if run is None:
            return self._error(404, "no such run (it may have finished)")
        key = ("approval:" if body.get("kind") == "approval" else "question:") + str(body.get("id", ""))
        run.answer(key, {"approve": bool(body.get("approve")), "text": body.get("text", "")})
        return self._json(200, {"ok": True})

    def api_agent(self, P, q, body):
        messages = body.get("messages") or []
        if not messages or messages[-1].get("role") != "user":
            raise ValueError("the last message must be from the user")
        items = P.attachments(body.get("attachments"))
        msgs = [{"role": m["role"], "content": m["content"]} for m in messages]
        if items:
            msgs[-1]["content"] = f"{msgs[-1]['content']}\n\n{attachment_block(items)}"
        def alive():
            try:                                  # an orderly close shows up as an empty peek
                return self.connection.recv(1, socket.MSG_PEEK | socket.MSG_DONTWAIT) != b""
            except BlockingIOError:
                return True
            except OSError:
                return False

        run = agentloop.Run(alive)
        agentloop.RUNS[run.id] = run
        self._headers(200, "text/event-stream", extra={"Connection": "close"})
        self.close_connection = True

        def emit(ev):
            try:
                self.wfile.write(b"data: " + json.dumps(ev, ensure_ascii=False).encode() + b"\n\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                run.cancel()

        try:
            emit({"type": "run", "id": run.id})
            extra = ""
            try:
                from harness import memory
                extra = memory.system_block(P.root)
            except Exception:
                pass
            finished = agentloop.run_agent(P, self.server.engine_port, msgs, emit, run, think=bool(body.get("think")),
                                           auto_edit=bool(body.get("auto_edit")), system_extra=extra)
            if finished:
                emit({"type": "done"})
        except RuntimeError as e:
            emit({"type": "error", "message": str(e)})
        finally:
            run.cancel()
            agentloop.RUNS.pop(run.id, None)

    def api_git_status(self, P, q, body):
        return self._json(200, P.git_status())

    def api_git_diff(self, P, q, body):
        return self._json(200, {"diff": P.git_diff(q.get("path", ""))})

    def api_git_commit(self, P, q, body):
        summary = P.git_commit(body.get("message"), body.get("paths"), bool(body.get("all")))
        return self._json(200, {"ok": True, "commit": summary, "git": P.git_status()})

    def api_git_push(self, P, q, body):
        if body.get("confirm") is not True:           # a push leaves the machine: never implicit
            return self._error(400, "push needs an explicit confirmation")
        out = P.git_push(body.get("remote"), body.get("branch"))
        return self._json(200, {"ok": True, "output": out, "git": P.git_status()})

    def api_chat(self, P, q, body):
        messages = body.get("messages") or []
        if not messages or messages[-1].get("role") != "user":
            raise ValueError("the last message must be from the user")
        items = P.attachments(body.get("attachments"))
        msgs = [dict(m) for m in messages]
        if items:
            msgs[-1]["content"] = f"{msgs[-1]['content']}\n\n{attachment_block(items)}"
        msgs.insert(0, {"role": "system", "content": self.server.system_prompt()})
        payload = {"model": "local", "messages": msgs, "stream": True,
                   "chat_template_kwargs": {"enable_thinking": bool(body.get("think", False))}}
        req = urllib.request.Request(f"http://127.0.0.1:{self.server.engine_port}/v1/chat/completions",
                                     data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
        try:
            r = urllib.request.urlopen(req, timeout=3600)
        except urllib.error.HTTPError as e:
            return self._error(e.code, e.read(300).decode("utf-8", "replace"))
        except (urllib.error.URLError, OSError):
            return self._error(502, "the model is not running — start it first (./llm start)")
        self._headers(200, "text/event-stream", extra={"Connection": "close"})
        self.close_connection = True
        try:
            for line in r:
                self.wfile.write(line)
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            r.close()


class UIServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, project, port=0, engine_port=8090, token=None):
        super().__init__(("127.0.0.1", port), Handler)
        self.project = project
        self.engine_port = engine_port
        self.token = token or secrets.token_urlsafe(18)

    def system_prompt(self):
        sys.path.insert(0, ROOT_DIR)
        try:
            from harness import memory
            extra = memory.system_block(self.project.root)
        except Exception:
            extra = ""
        return ("You are a coding assistant inside a code editor, working on the project "
                f"{os.path.basename(self.project.root)}. Files the user attached are quoted in their message. "
                "Answer concisely; show code in fenced blocks and say which file it belongs to." + extra)

    @property
    def url(self):
        return f"http://127.0.0.1:{self.server_address[1]}/?token={self.token}"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=os.getcwd())
    ap.add_argument("--port", type=int, default=0)
    ap.add_argument("--engine-port", type=int, default=8090)
    ap.add_argument("--open", action="store_true", help="open the browser")
    args = ap.parse_args(argv)
    root = os.path.abspath(args.root)
    if not os.path.isdir(root):
        print(f"not a directory: {root}", file=sys.stderr)
        return 2
    srv = UIServer(Project(root), args.port, args.engine_port)
    print(f"Editor UI for {root}\n  {srv.url}\n  (Ctrl+C to stop)", flush=True)
    if args.open:
        threading.Timer(0.4, lambda: subprocess.run(["open", srv.url])).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
