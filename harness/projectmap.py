"""A cheap, model-free map of the workspace, kept in `.llm/map.md`.

Built from file names, sizes and the first meaningful line / top-level names of each file, so the
agent starts a session knowing the layout instead of listing and grepping the tree again. Rebuilt
only when something changed (fingerprint of path + mtime + size, stored in the file's first line).
"""
import ast
import hashlib
import os
import re
import subprocess

MAP_PATH = os.path.join(".llm", "map.md")
SKIP_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__", ".llm", "dist", "build", ".build",
             ".idea", ".vscode", "target", ".next", ".cache", "Pods", "DerivedData"}
TEXT_EXT = {".py", ".md", ".txt", ".json", ".toml", ".yaml", ".yml", ".sh", ".js", ".ts", ".tsx", ".jsx",
            ".swift", ".rs", ".go", ".c", ".h", ".cpp", ".hpp", ".java", ".kt", ".rb", ".html", ".css",
            ".sql", ".cfg", ".ini", ".mk", ""}
IMAGES = (".png", ".jpg", ".jpeg", ".gif", ".webp", ".ico", ".icns", ".svg", ".pdf", ".mp4", ".mov")
MAX_FILES = 600
MAX_BYTES = 400_000          # files bigger than this are listed, not read
PROMPT_CHARS = 4000          # what goes into the system prompt
DECL = re.compile(r"^\s*(?:export\s+)?(?:public\s+|private\s+|final\s+|async\s+)*"
                  r"(?:def|class|func|function|struct|enum|protocol|interface|fn|type|impl)\s+([A-Za-z_]\w*)", re.M)


def _files(root):
    try:
        out = subprocess.run(["git", "-C", root, "ls-files", "-co", "--exclude-standard"], capture_output=True,
                             text=True, timeout=10)
        if out.returncode == 0 and out.stdout.strip():
            return sorted(p for p in out.stdout.split("\n") if p and not set(p.split("/")[:-1]) & SKIP_DIRS)
    except (OSError, subprocess.TimeoutExpired):
        pass
    found = []
    for d, dirs, names in os.walk(root):
        dirs[:] = sorted(x for x in dirs if x not in SKIP_DIRS and not x.startswith("."))
        for n in sorted(names):
            if not n.startswith("."):
                found.append(os.path.relpath(os.path.join(d, n), root))
    return found


def _stat(root, rel):
    try:
        st = os.stat(os.path.join(root, rel))
        return st.st_mtime_ns, st.st_size
    except OSError:
        return None


def fingerprint(root, files=None):
    h = hashlib.sha1()
    for rel in files if files is not None else _files(root):
        s = _stat(root, rel)
        if s:
            h.update(f"{rel}\0{s[0]}\0{s[1]}\n".encode())
    return h.hexdigest()[:16]


def describe(root, rel):
    """(lines, one-line description) of a file; '' when nothing useful can be said cheaply."""
    ext = os.path.splitext(rel)[1].lower()
    full = os.path.join(root, rel)
    s = _stat(root, rel)
    if not s or ext not in TEXT_EXT or s[1] > MAX_BYTES:
        return None, ""
    try:
        with open(full, encoding="utf-8") as f:
            text = f.read()
    except (OSError, UnicodeDecodeError):
        return None, ""
    lines = text.count("\n") + (1 if text and not text.endswith("\n") else 0)
    desc = ""
    if ext == ".py":
        try:
            tree = ast.parse(text)
            doc = ast.get_docstring(tree)
            names = [n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                     and not n.name.startswith("_")]
            desc = (doc.strip().split("\n")[0][:110] if doc else "")
            if names:
                desc += (" — " if desc else "") + ", ".join(names[:8]) + ("…" if len(names) > 8 else "")
        except SyntaxError:
            pass
    elif ext == ".md":
        m = re.search(r"^#{1,3}\s+(.+)$", text, re.M)
        desc = m.group(1).strip()[:110] if m else ""
    elif ext in (".json", ".toml", ".yaml", ".yml", ".txt", ".cfg", ".ini"):
        desc = ""
    else:
        names = DECL.findall(text)
        first = next((ln.strip().lstrip("/#*- ").strip() for ln in text.split("\n")[:12]
                      if ln.strip().startswith(("//", "#", "/*", "*")) and len(ln.strip()) > 8
                      and not ln.startswith("#!")), "")
        desc = first[:90]
        if names:
            seen = list(dict.fromkeys(names))
            desc += (" — " if desc else "") + ", ".join(seen[:8]) + ("…" if len(seen) > 8 else "")
    return lines, desc


def build(root, files=None, detail=0):
    """detail 0: every file with size and description; 1: short descriptions, no sizes;
    2: one line per directory listing file names (what a big repo can afford in a prompt)."""
    files = (files if files is not None else _files(root))[:MAX_FILES * 3]
    out, by_dir = [], {}
    for rel in files:
        if rel.lower().endswith(IMAGES):
            continue
        by_dir.setdefault(os.path.dirname(rel) or ".", []).append(rel)
    shown = 0
    for d in sorted(by_dir, key=lambda x: (x != ".", x)):
        head = f"{d}/" if d != "." else "./"
        if detail >= 2:
            names = [os.path.basename(r) for r in by_dir[d]]
            if names:
                out.append(f"{head} " + ", ".join(names[:40]) + (f", … (+{len(names) - 40})" if len(names) > 40 else ""))
            shown += len(by_dir[d])
            continue
        out.append(f"\n## {head}")
        for rel in by_dir[d]:
            if shown >= MAX_FILES:
                break
            lines, desc = describe(root, rel)
            name = os.path.basename(rel)
            if detail == 1:
                desc = desc[:50]
                out.append(f"- {name}" + (f" — {desc}" if desc else ""))
            else:
                size = f" ({lines} lines)" if lines else ""
                out.append(f"- {name}{size}" + (f" — {desc}" if desc else ""))
            shown += 1
    omitted = sum(len(v) for v in by_dir.values()) - shown
    if omitted > 0:
        out.append(f"\n… {omitted} more files not listed")
    return "\n".join(out).strip()


def ensure(root):
    """Return the map text, rebuilding `.llm/map.md` only when the tree changed."""
    files = _files(root)
    if not files:
        return ""
    fp = fingerprint(root, files)
    path = os.path.join(root, MAP_PATH)
    try:
        with open(path, encoding="utf-8") as f:
            head, _, body = f.read().partition("\n")
        if head == f"<!-- fp:{fp} -->":
            return body
    except OSError:
        pass
    body = ("# Project map (generated; do not edit)\n"
            "Files with line counts and top-level names. Use read_file/search for details.\n\n" + build(root, files))
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"<!-- fp:{fp} -->\n{body}")
    except OSError:
        pass
    return body


def system_block(root, limit=PROMPT_CHARS):
    """The map for the system prompt: full when it fits, else progressively terser — never a
    hard cut that would lose whole directories."""
    body = ensure(root)
    if not body:
        return ""
    if len(body) > limit:
        files = _files(root)
        for detail in (1, 2):
            body = ("# Project map (condensed)\n" + build(root, files, detail)).strip()
            if len(body) <= limit:
                break
        else:
            body = body[:limit].rsplit("\n", 1)[0] + "\n… (truncated; use list_dir / search for the rest)"
    return "\n\n" + body
