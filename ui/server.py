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
import shutil
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
from ui import agentloop, codeinfo, diagnostics  # noqa: E402

SKIP_DIRS = {".git", ".llm", "node_modules", "__pycache__", ".venv", "venv", ".build", "DerivedData", ".idea",
             ".DS_Store"}
MAX_FILE = 2_000_000          # larger files are listed but not opened
MAX_ATTACH_TOTAL = 200_000    # bytes of attachments inlined into one prompt
MAX_BODY = 40_000_000         # images arrive inline (base64) with the request
MAX_IMAGES = 6
MAX_IMAGE_BYTES = 10_000_000
GIT_TIMEOUT = 60


class Project:
    """The workspace root and every filesystem/git operation on it."""

    skip = SKIP_DIRS
    max_file = MAX_FILE

    def __init__(self, root):
        self.root = os.path.realpath(root)

    SEARCH_SECONDS = 10           # a search that has run this long returns what it has

    def search(self, pattern, rel=".", glob=None, limit=200):
        """path:line: text for lines matching a regex (a plain substring when it is not a valid regex)."""
        return [f"{p}:{n}: {t}" for p, n, t in self.search_hits(pattern, rel, glob, limit)]

    def search_hits(self, pattern, rel=".", glob=None, limit=200):
        """[(path, line, text)] — structured, so a path that contains a colon is not cut in the wrong place."""
        import fnmatch
        # a quantified group that itself holds a quantifier — (a+)+ , (.*)* , (\\w+\\s*)* — is the classic shape of
        # catastrophic backtracking, and the pattern usually comes from a model: such a pattern is searched for
        # literally instead of being able to hold this server's thread for minutes
        nested = re.search(r"\((?:[^()\\]|\\.)*[+*](?:[^()\\]|\\.)*\)\s*(?:[+*]|\{\d+,?\d*\})", pattern) \
            or re.search(r"\((?:[^()\\]|\\.)*\|(?:[^()\\]|\\.)*\)\s*(?:[+*]|\{\d+,?\d*\})", pattern)   # (a|aa)+
        try:
            rx = re.compile(re.escape(pattern) if nested else pattern)
        except re.error:
            rx = re.compile(re.escape(pattern))
        out = []
        deadline = time.time() + self.SEARCH_SECONDS
        for d, dirs, files in os.walk(self.resolve(rel)):
            dirs[:] = sorted(x for x in dirs if x not in SKIP_DIRS and not x.startswith("."))
            for f in sorted(files):
                if time.time() > deadline:
                    return out
                if glob and not fnmatch.fnmatch(f, glob):
                    continue
                full = os.path.join(d, f)
                if self.escapes(full):
                    continue
                try:
                    if os.path.getsize(full) > MAX_FILE:
                        continue
                    with open(full, encoding="utf-8") as fh:
                        for n, line in enumerate(fh, 1):
                            if rx.search(line[:4000]):          # a minified file's one huge line is not scanned whole
                                out.append((self.rel(full), n, line.rstrip()[:200]))
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

    def escapes(self, full):
        """True for a symlink whose target is outside the project: listed or searched, it would show
        content that read() and the attachments refuse."""
        if not os.path.islink(full):
            return False
        real = os.path.realpath(full)
        return not (real == self.root or real.startswith(self.root + os.sep)) or ".git" in real.split(os.sep)

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

    def list_files(self, limit=5000):
        """Every file path under the project (for quick-open), respecting the skip list."""
        out = []
        for d, dirs, files in os.walk(self.root):
            dirs[:] = sorted(x for x in dirs if x not in SKIP_DIRS and not x.startswith("."))
            for f in sorted(files):
                if not f.startswith(".") and not self.escapes(os.path.join(d, f)):
                    out.append(self.rel(os.path.join(d, f)))
                    if len(out) >= limit:
                        return out
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
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            return {"path": self.rel(full), "not_utf8": True, "size": st.st_size}
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
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                f.write(text)
            if os.path.exists(full):
                shutil.copymode(full, tmp)          # a saved script stays executable
            os.replace(tmp, full)
        except BaseException:
            try:
                os.remove(tmp)
            except OSError:
                pass
            raise
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
        p = self.git("status", "--porcelain=v1", "-z", "-b", check=True)
        lines, items = [], p.stdout.split("\0")
        i = 0
        while i < len(items):
            entry = items[i]
            i += 1
            if not entry:
                continue
            if entry[:1] in "RC" or entry[1:2] in "RC":       # a rename/copy is followed by its source path
                i += 1
            lines.append(entry)
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
            files.append({"path": path, "index": code[0], "worktree": code[1],
                          "status": "untracked" if code == "??" else "modified" if "M" in code else
                          "added" if "A" in code else "deleted" if "D" in code else "renamed" if "R" in code else code})
        remotes = self.git("remote").stdout.split()
        return {"repo": True, "branch": branch, "upstream": upstream, "ahead": ahead, "behind": behind,
                "files": files, "remotes": remotes}

    MAX_DIFF = 2_000_000

    def git_diff(self, rel):
        self.resolve(rel, must_exist=False)
        p = self.git("diff", "HEAD", "--", rel)
        if not p.stdout.strip():
            p = self.git("diff", "--no-index", "--", os.devnull, os.path.join(self.root, rel)) \
                if os.path.exists(os.path.join(self.root, rel)) else p
        out = p.stdout
        if len(out) > self.MAX_DIFF:
            out = out[:self.MAX_DIFF] + f"\n… {len(out) - self.MAX_DIFF} more characters of diff not shown\n"
        return out

    @staticmethod
    def _paths(paths):
        """A list of non-empty path strings — a bare string would be walked character by character, and "." among
        the characters of "a.txt" stages the whole repository."""
        if paths is None:
            return []
        if not isinstance(paths, list) or not all(isinstance(x, str) and x for x in paths):
            raise ValueError("paths must be a list of file paths")
        return paths

    def git_commit(self, message, paths=None, all_changes=False):
        message = (message or "").strip()
        if not message:
            raise ValueError("a commit message is required")
        if not self.is_repo():
            raise RuntimeError("not a git repository")
        paths = self._paths(paths)
        only = []
        if all_changes:
            self.git("add", "-A", check=True)
        elif paths:
            for rel in paths:
                self.resolve(rel, must_exist=False)
            self.git("add", "--", *paths, check=True)
            only = ["--", *paths]                   # not whatever else happens to be staged already
        else:
            raise ValueError("choose files to commit (or all changes)")
        if not self.git("diff", "--cached", "--name-only", *only).stdout.strip():
            raise RuntimeError("nothing staged to commit")
        self.git("commit", "-m", message, *only, check=True)
        return self.git("log", "-1", "--format=%h %s").stdout.strip()

    def git_discard(self, paths=None, all_changes=False):
        """Throw away uncommitted work: tracked files go back to HEAD (index and working tree), untracked ones
        are deleted, files added to the index are removed. Returns the paths that were discarded.
        Nothing here can be undone — the caller has already asked the person."""
        st = self.git_status()
        if not st.get("repo"):
            raise RuntimeError("not a git repository")
        changed = {f["path"]: f for f in st["files"]}
        wanted = list(changed) if all_changes else self._paths(paths)
        if not wanted:
            raise ValueError("choose files to discard (or all changes)")
        for rel in wanted:
            self.resolve(rel, must_exist=False)
            if rel not in changed:
                raise ValueError(f"{rel} has no uncommitted changes")
            if changed[rel]["status"] == "renamed":
                raise ValueError(f"{rel} is a rename: undo it in the terminal (git restore --staged / git mv)")
        restore, remove, untracked = [], [], []
        for rel in wanted:
            f = changed[rel]
            if f["status"] == "untracked":
                untracked.append(rel)
            elif f["index"] == "A":
                remove.append(rel)
            else:
                restore.append(rel)
        if restore:
            self.git("restore", "--source=HEAD", "--staged", "--worktree", "--", *restore, check=True)
        if remove:
            self.git("rm", "-f", "-q", "--", *remove, check=True)
        if untracked:
            self.git("clean", "-f", "-d", "-q", "--", *untracked, check=True)
        return wanted

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
        if branch.startswith("-") or self.git("check-ref-format", "--branch", branch).returncode != 0:
            raise ValueError(f"not a branch name: {branch!r}")
        args = ["push", remote, branch] if st["upstream"] else ["push", "-u", remote, branch]
        p = self.git(*args)
        out = (p.stdout + p.stderr).strip()
        if p.returncode != 0:
            raise RuntimeError(out or "git push failed")
        return out or "pushed"

    # ---- GitHub (through the `gh` CLI: its login, its credentials — nothing of ours is stored)
    def gh(self, *args, timeout=60):
        exe = shutil.which("gh")
        if not exe:
            raise RuntimeError("GitHub CLI not found — install it with: brew install gh")
        env = dict(os.environ, GH_PROMPT_DISABLED="1", NO_COLOR="1", GH_NO_UPDATE_NOTIFIER="1")
        return subprocess.run([exe, *args], cwd=self.root, capture_output=True, text=True, timeout=timeout, env=env)

    def github_status(self):
        out = {"installed": bool(shutil.which("gh")), "authenticated": False, "user": None, "repo": None, "remote": None}
        if self.is_repo():
            url = self.git("remote", "get-url", "origin").stdout.strip()
            out["remote"] = url or None
            m = re.search(r"github\.com[:/]([^/\s]+)/([^/\s]+?)(?:\.git)?$", url)
            if m:
                out["repo"] = f"{m.group(1)}/{m.group(2)}"
        if out["installed"]:
            who = self.gh("api", "user", "--jq", ".login", timeout=20)
            if who.returncode == 0 and who.stdout.strip():
                out.update(authenticated=True, user=who.stdout.strip())
        return out

    def github_prs(self):
        p = self.gh("pr", "list", "--limit", "20", "--json", "number,title,headRefName,url,isDraft,author")
        if p.returncode != 0:
            raise RuntimeError((p.stderr or p.stdout).strip() or "gh pr list failed")
        return json.loads(p.stdout or "[]")

    def github_pr(self, title, body="", base=None, draft=False):
        title = (title or "").strip()
        if not title:
            raise ValueError("a title is required")
        args = ["pr", "create", "--title", title, "--body", body or ""]
        if base:
            if base.startswith("-") or self.git("check-ref-format", "--branch", base).returncode != 0:
                raise ValueError(f"not a branch name: {base!r}")
            args += ["--base", base]
        if draft:
            args.append("--draft")
        p = self.gh(*args, timeout=120)
        if p.returncode != 0:
            raise RuntimeError((p.stderr or p.stdout).strip() or "gh pr create failed")
        return p.stdout.strip().splitlines()[-1] if p.stdout.strip() else ""

    def github_create_repo(self, name, private=True):
        name = (name or "").strip()
        if not re.fullmatch(r"[A-Za-z0-9._-]{1,100}", name):
            raise ValueError("a repository name may contain letters, digits, '.', '_' and '-'")
        if self.git("remote", "get-url", "origin").returncode == 0:
            raise ValueError("this project already has a remote named origin")
        p = self.gh("repo", "create", name, "--private" if private else "--public", "--source=.", "--remote=origin",
                    "--push", timeout=180)
        if p.returncode != 0:
            raise RuntimeError((p.stderr or p.stdout).strip() or "gh repo create failed")
        return (p.stdout + p.stderr).strip().splitlines()[-1]

    # ---- branches
    def branches(self):
        if not self.is_repo():
            return {"repo": False, "current": None, "local": [], "remote": []}
        cur = self.git("branch", "--show-current").stdout.strip()
        local = self.git("for-each-ref", "--format=%(refname:short)", "refs/heads").stdout.split()
        remote = [r for r in self.git("for-each-ref", "--format=%(refname:short)", "refs/remotes").stdout.split()
                  if not r.endswith("/HEAD")]
        return {"repo": True, "current": cur or None, "local": sorted(local), "remote": sorted(remote)}

    def checkout(self, branch, create=False):
        """Switch (or create and switch to) a branch. git refuses when local changes would be lost, and its
        message is passed on; nothing is forced."""
        branch = (branch or "").strip()
        if not branch or branch.startswith("-") or self.git("check-ref-format", "--branch", branch).returncode != 0:
            raise ValueError(f"not a branch name: {branch!r}")
        if not self.is_repo():
            raise RuntimeError("not a git repository")
        info = self.branches()
        if create:
            if branch in info["local"]:
                raise ValueError(f"branch {branch!r} already exists")
            self.git("checkout", "-b", branch, check=True)
        elif branch in info["local"]:
            self.git("checkout", branch, check=True)
        elif branch in info["remote"]:                  # origin/x: a local branch that tracks it
            self.git("checkout", "--track", branch, check=True)
        else:
            raise ValueError(f"no such branch: {branch!r}")
        return self.branches()

    # ---- prompt attachments
    def inline_attachments(self, items):
        """Files that are not in the project (picked from anywhere on disk by the editor): [(name, text)]."""
        out, total = [], 0
        for it in items or []:
            name, text = str(it.get("name") or "attachment"), it.get("text")
            if not isinstance(text, str):
                raise ValueError(f"{name}: no text")
            total += len(text.encode("utf-8"))
            if total > MAX_ATTACH_TOTAL:
                raise ValueError(f"attachments exceed {MAX_ATTACH_TOTAL // 1000} KB; attach fewer or smaller files")
            out.append((os.path.basename(name), text))
        return out

    @staticmethod
    def images(items):
        """['data:image/png;base64,…', …] after checking type and size."""
        import base64
        urls = []
        for it in items or []:
            url = it.get("data") if isinstance(it, dict) else None
            if not isinstance(url, str) or not re.match(r"data:image/(png|jpeg|jpg|gif|webp);base64,", url):
                raise ValueError(f"{(it or {}).get('name', 'image')}: not a PNG/JPEG/GIF/WebP image")
            try:
                size = len(base64.b64decode(url.split(",", 1)[1], validate=False))
            except Exception:
                raise ValueError("the image data is not valid base64")
            if size > MAX_IMAGE_BYTES:
                raise ValueError(f"{(it or {}).get('name', 'image')} is larger than {MAX_IMAGE_BYTES // 1_000_000} MB")
            urls.append(url)
        if len(urls) > MAX_IMAGES:
            raise ValueError(f"at most {MAX_IMAGES} images per message")
        return urls
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


def user_content(text, images):
    """A user message's content: plain text, or text plus image parts when images are attached."""
    if not images:
        return text
    return [{"type": "text", "text": text}] + [{"type": "image_url", "image_url": {"url": u}} for u in images]


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
        self._sent = True
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
        try:
            n = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            raise ValueError("bad Content-Length")
        if n < 0:
            raise ValueError("bad Content-Length")
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
        self._sent = False
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
        except (KeyError, TypeError, AttributeError, IndexError) as e:     # a malformed request body
            if self._sent:
                self.close_connection = True
                return None
            return self._error(400, f"bad request: {type(e).__name__}: {e}")

    # ---- API
    def api_info(self, P, q, body):
        eng = None
        try:
            ep = self.server.endpoint
            if ep.remote:                         # a remote server: the OpenAI model list is all there is to ask
                req = urllib.request.Request(ep.base + "/models", headers=ep.headers())
                with urllib.request.urlopen(req, timeout=4) as r:
                    listing = json.load(r)
                ids = [m.get("id") for m in listing.get("data", [])]
                eng = {"model": ep.model if ep.model != "local" else (ids[0] if ids else "remote"),
                       "state": "remote", "context": None, "vision": False, "vision_capable": False}
            else:
                with urllib.request.urlopen(ep.health_url, timeout=1.5) as r:
                    h = json.load(r)
                eng = {"model": h.get("model"), "state": h.get("state"), "context": h.get("context"),
                       "vision": bool(h.get("vision")), "vision_capable": bool(h.get("vision_capable")),
                       "efforts": [e for e in (h.get("efforts") or []) if isinstance(e, str)],
                       "thinking_supported": h.get("thinking_supported", True) is not False}
        except (OSError, ValueError):
            pass
        return self._json(200, {"root": P.root, "name": os.path.basename(P.root), "engine": eng,
                                "git": P.git_status(), "remote": self.server.endpoint.remote,
                                "endpoint": self.server.endpoint.base if self.server.endpoint.remote else None})

    def api_endpoint(self, P, q, body):
        """GET: where the model is. PUT {url, key, model, confirm} points the assistant at another server
        (the key stays in memory of this process; the editor keeps it in the Keychain); PUT {reset: true}
        returns to the engine on this Mac."""
        srv = self.server
        if self.command == "GET":
            ep = srv.endpoint
            return self._json(200, {"remote": ep.remote, "url": ep.base if ep.remote else "", "model": ep.model,
                                    "has_key": bool(ep.key)})
        if body.get("reset"):
            srv.endpoint = agentloop.Endpoint.local(srv.engine_port)
            return self._json(200, {"remote": False})
        url = str(body.get("url") or "").strip()
        parts = urllib.parse.urlsplit(url)
        if parts.scheme not in ("http", "https") or not parts.hostname:
            raise ValueError("the address must start with http:// or https://")
        if parts.scheme == "http" and parts.hostname not in ("127.0.0.1", "localhost", "::1") and body.get("key"):
            raise ValueError("a key is not sent over plain http to another computer — use https://")
        if body.get("confirm") is not True:         # your messages and attached files will go to that server
            return self._error(400, "pointing the assistant at another server needs an explicit confirmation")
        srv.endpoint = agentloop.Endpoint(url.rstrip("/"), body.get("key") or "", body.get("model") or "local")
        return self._json(200, {"remote": srv.endpoint.remote, "url": srv.endpoint.base})

    def api_tree(self, P, q, body):
        return self._json(200, {"path": q.get("path", ""), "entries": P.tree(q.get("path", ""))})

    def api_files(self, P, q, body):
        return self._json(200, {"files": P.list_files()})

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
                                "outline": codeinfo.outline(text, lang),
                                "diagnostics": diagnostics.diagnose(text, lang)})

    def api_diagnostics(self, P, q, body):
        """The language's own parser (swiftc/node/bash…) when installed — slower, so asked for separately."""
        text = body.get("text", "")
        if len(text) > MAX_FILE:
            raise ValueError("text too large to check")
        lang = body.get("language") or codeinfo.detect_language(body.get("path", ""), text)
        found = diagnostics.diagnose(text, lang)
        deep = diagnostics.deep_check(text, lang) if body.get("deep", True) else []
        seen, merged = set(), []
        for d in found + deep:                       # the same problem from two tiers is shown once
            key = (d["line"], d["message"])
            if key not in seen:
                seen.add(key)
                merged.append(d)
        return self._json(200, {"language": lang, "diagnostics": merged, "deep": diagnostics.available(lang)})

    def api_search(self, P, q, body):
        pattern = q.get("q", "")
        if not pattern:
            raise ValueError("empty query")
        hits = P.search_hits(pattern, q.get("path", "."), q.get("glob"), limit=200)
        # `line` stays a string: the editor app decodes it as one (a number here broke its search panel)
        return self._json(200, {"hits": [{"path": p, "line": str(n), "text": t} for p, n, t in hits]})

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
        items = P.attachments(body.get("attachments")) + P.inline_attachments(body.get("inline"))
        pictures = P.images(body.get("images"))
        msgs = [{"role": m["role"], "content": m["content"]} for m in messages]
        if items:
            msgs[-1]["content"] = f"{msgs[-1]['content']}\n\n{attachment_block(items)}"
        if pictures:
            msgs[-1]["content"] = user_content(msgs[-1]["content"], pictures)

        def alive():
            try:                                  # an orderly close shows up as an empty peek
                return self.connection.recv(1, socket.MSG_PEEK | socket.MSG_DONTWAIT) != b""
            except BlockingIOError:
                return True
            except OSError:
                return False

        run = agentloop.Run(alive)
        agentloop.RUNS[run.id] = run
        self._headers(200, "text/event-stream; charset=utf-8", extra={"Connection": "close"})
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
            effort = body.get("effort")
            finished = agentloop.run_agent(P, self.server.endpoint, msgs, emit, run, think=bool(body.get("think")),
                                           auto_edit=bool(body.get("auto_edit")), system_extra=extra,
                                           web=bool(body.get("web")), mode=body.get("mode"),
                                           effort=effort if re.fullmatch(r"[a-z]{2,10}", str(effort or "")) else None)
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

    def api_git_branches(self, P, q, body):
        return self._json(200, P.branches())

    def api_git_checkout(self, P, q, body):
        info = P.checkout(body.get("branch"), bool(body.get("create")))
        return self._json(200, {"ok": True, "branches": info, "git": P.git_status()})

    def api_github_status(self, P, q, body):
        return self._json(200, P.github_status())

    def api_github_prs(self, P, q, body):
        return self._json(200, {"prs": P.github_prs()})

    def api_github_pr(self, P, q, body):
        if body.get("confirm") is not True:               # opening a PR is public: never implicit
            return self._error(400, "creating a pull request needs an explicit confirmation")
        url = P.github_pr(body.get("title"), body.get("body", ""), body.get("base"), bool(body.get("draft")))
        return self._json(200, {"ok": True, "url": url})

    def api_github_create_repo(self, P, q, body):
        if body.get("confirm") is not True:
            return self._error(400, "creating a repository needs an explicit confirmation")
        url = P.github_create_repo(body.get("name"), bool(body.get("private", True)))
        return self._json(200, {"ok": True, "url": url, "git": P.git_status()})

    def api_git_commit(self, P, q, body):
        summary = P.git_commit(body.get("message"), body.get("paths"), bool(body.get("all")))
        return self._json(200, {"ok": True, "commit": summary, "git": P.git_status()})

    def api_git_discard(self, P, q, body):
        if body.get("confirm") is not True:           # destroys work that git cannot bring back
            return self._error(400, "discarding changes needs an explicit confirmation")
        done = P.git_discard(body.get("paths"), bool(body.get("all")))
        return self._json(200, {"ok": True, "discarded": done, "git": P.git_status()})

    # ---- Open VSX extensions (declarative parts only: themes, snippets; never runs their code)
    def api_extensions_search(self, P, q, body):
        from ui import extensions
        query = (q.get("q") or "").strip()
        if not query:
            raise ValueError("a search text is required")
        try:
            return self._json(200, {"results": extensions.search(query)})
        except extensions.ExtensionError as e:
            return self._error(502, str(e))

    def api_extensions_installed(self, P, q, body):
        from ui import extensions
        return self._json(200, {"extensions": extensions.installed()})

    def api_extensions_install(self, P, q, body):
        from ui import extensions
        if body.get("confirm") is not True:           # downloads from the internet and writes to ~/.llm
            return self._error(400, "installing an extension needs an explicit confirmation")
        try:
            m = extensions.install(body.get("namespace"), body.get("name"), body.get("version"))
        except extensions.ExtensionError as e:
            return self._error(502 if "open-vsx.org" in str(e) else 400, str(e))
        return self._json(200, {"ok": True, "extension": m})

    def api_extensions_remove(self, P, q, body):
        from ui import extensions
        try:
            n = extensions.remove(body.get("id"))
        except extensions.ExtensionError as e:
            return self._error(400, str(e))
        return self._json(200, {"ok": True, "removed": n})

    def api_extensions_theme(self, P, q, body):
        from ui import extensions
        try:
            return self._json(200, extensions.theme(q.get("id", ""), q.get("label", "")))
        except extensions.ExtensionError as e:
            return self._error(404, str(e))

    def api_extensions_snippets(self, P, q, body):
        from ui import extensions
        return self._json(200, {"snippets": extensions.snippets(q.get("language", ""))})

    def api_git_push(self, P, q, body):
        if body.get("confirm") is not True:           # a push leaves the machine: never implicit
            return self._error(400, "push needs an explicit confirmation")
        out = P.git_push(body.get("remote"), body.get("branch"))
        return self._json(200, {"ok": True, "output": out, "git": P.git_status()})

    def api_compact(self, P, q, body):
        """{messages: [{role, content}…], keep_user_turns} → {summary, folded}: a note that replaces everything
        before the last N user turns (the same two-stage logic the terminal agent uses, harness/compact.py)."""
        from harness import compact
        history = [{"role": m["role"], "content": str(m.get("content") or "")} for m in (body.get("messages") or [])
                   if isinstance(m, dict) and m.get("role") in ("user", "assistant")]
        keep = max(1, min(6, int(body.get("keep_user_turns") or compact.KEEP_USER_TURNS)))
        msgs = [{"role": "system", "content": "-"}] + history
        cut = compact._tail_start(msgs, keep)
        if cut <= 1:
            return self._json(200, {"summary": "", "folded": 0})
        summary = agentloop.complete(self.server.endpoint, [
            {"role": "user", "content": compact.SUMMARY_PROMPT + compact.transcript(msgs[1:cut])}], max_tokens=900)
        return self._json(200, {"summary": summary, "folded": cut - 1})

    def api_chat(self, P, q, body):
        messages = body.get("messages") or []
        if not messages or messages[-1].get("role") != "user":
            raise ValueError("the last message must be from the user")
        items = P.attachments(body.get("attachments")) + P.inline_attachments(body.get("inline"))
        pictures = P.images(body.get("images"))
        msgs = [dict(m) for m in messages]
        if items:
            msgs[-1]["content"] = f"{msgs[-1]['content']}\n\n{attachment_block(items)}"
        if pictures:
            msgs[-1]["content"] = user_content(msgs[-1]["content"], pictures)
        msgs.insert(0, {"role": "system", "content": self.server.system_prompt()})
        ep = self.server.endpoint
        req = urllib.request.Request(ep.chat_url, data=json.dumps(ep.payload(msgs, bool(body.get("think", False)))).encode(),
                                     headers=ep.headers())
        try:
            r = urllib.request.urlopen(req, timeout=3600)
        except urllib.error.HTTPError as e:
            return self._error(e.code, e.read(300).decode("utf-8", "replace"))
        except (urllib.error.URLError, OSError):
            return self._error(502, "the model is not running — start it first (./llm start)")
        self._headers(200, "text/event-stream; charset=utf-8", extra={"Connection": "close"})
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
        self.endpoint = agentloop.Endpoint.local(engine_port)
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
    ap.add_argument("--exit-with-parent", type=int, nargs="?", const=0, default=None, metavar="PID",
                    help="stop when process PID is gone (the editor app passes its own); without a PID: "
                         "when the starting process is")
    args = ap.parse_args(argv)
    root = os.path.abspath(args.root)
    if not os.path.isdir(root):
        print(f"not a directory: {root}", file=sys.stderr)
        return 2
    srv = UIServer(Project(root), args.port, args.engine_port)
    if args.exit_with_parent is not None:
        parent = args.exit_with_parent or os.getppid()

        def gone():
            try:
                os.kill(parent, 0)
                return False
            except ProcessLookupError:
                return True
            except PermissionError:
                return False

        def watch():
            while not gone() and (args.exit_with_parent or os.getppid() == parent):
                time.sleep(1)
            os._exit(0)              # the app crashed or was killed: no orphan with a live token
        threading.Thread(target=watch, daemon=True).start()
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
