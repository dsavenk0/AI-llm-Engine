"""Errors and syntax problems for an editor buffer, in two tiers:

  diagnose(text, language)              instant, built in: Python (ast), JSON, TOML, XML, tabs in YAML,
                                        unbalanced / mismatched brackets for brace languages
  deep_check(text, language, path)      optional, a few hundred ms: the language's own parser when it is
                                        installed (swiftc -parse, node --check, bash -n, ruby -c, php -l, gofmt -e)

Each diagnostic: {line, col, end_col, severity: error|warning, message, source}; line/col are 1-based and
count characters (code points), like the rest of the backend.
"""
import ast
import json
import os
import re
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET

from ui import codeinfo

BRACES = {"javascript", "typescript", "swift", "go", "rust", "java", "kotlin", "c", "cpp", "objc",
          "csharp", "php", "css"}
PAIRS = {")": "(", "]": "[", "}": "{"}


def _d(line, col, message, source, severity="error", end_col=None):
    return {"line": max(1, line), "col": max(1, col), "end_col": end_col or max(1, col) + 1,
            "severity": severity, "message": message, "source": source}


def diagnose(text, language):
    """Built-in checks; never raises."""
    try:
        if language == "python":
            return _python(text)
        if language == "json":
            return _json(text)
        if language == "toml":
            return _toml(text)
        if language == "xml":
            return _xml(text)
        if language == "yaml":
            return _yaml(text)
        if language in BRACES:
            return _brackets(text, language)
    except Exception:
        return []
    return []


def _python(text):
    try:
        ast.parse(text)
        return []
    except SyntaxError as e:
        line = e.lineno or 1
        col = e.offset or 1
        end_col = (e.end_offset if getattr(e, "end_offset", None) and e.end_lineno == line else col + 1)
        msg = e.msg or "invalid syntax"
        return [_d(line, col, msg, "python", end_col=end_col)]
    except (ValueError, RecursionError) as e:
        return [_d(1, 1, str(e), "python")]


def _json(text):
    if not text.strip():
        return []
    try:
        json.loads(text)
        return []
    except json.JSONDecodeError as e:
        return [_d(e.lineno, e.colno, e.msg, "json")]


def _toml(text):
    try:
        import tomllib
    except ImportError:
        return []
    try:
        tomllib.loads(text)
        return []
    except tomllib.TOMLDecodeError as e:
        m = re.search(r"\(at line (\d+), column (\d+)\)", str(e))
        line, col = (int(m.group(1)), int(m.group(2))) if m else (1, 1)
        return [_d(line, col, re.sub(r"\s*\(at line.*\)$", "", str(e)), "toml")]


def _xml(text):
    try:
        ET.fromstring(text)
        return []
    except ET.ParseError as e:
        line, col = e.position
        return [_d(line, col + 1, re.sub(r":\s*line \d+, column \d+", "", str(e)), "xml")]


def _yaml(text):
    out = []
    for i, line in enumerate(text.split("\n"), 1):
        m = re.match(r"^( *)\t", line)
        if m and line.strip():
            out.append(_d(i, len(m.group(1)) + 1, "tabs are not allowed for indentation in YAML", "yaml"))
    return out[:50]


def _brackets(text, language):
    """Unclosed, unexpected and mismatched ()[]{} — strings and comments are blanked first."""
    masked = list(text)
    for a, b, kind in codeinfo.highlight(text, language):
        if kind in ("comment", "string"):
            for k in range(a, b):
                if masked[k] != "\n":
                    masked[k] = " "
    stack, out = [], []
    line, col = 1, 1
    for ch in masked:
        if ch == "\n":
            line, col = line + 1, 1
            continue
        if ch in "([{":
            stack.append((ch, line, col))
        elif ch in ")]}":
            if not stack:
                out.append(_d(line, col, f"unexpected '{ch}'", "brackets"))
            elif stack[-1][0] != PAIRS[ch]:
                o, ol, oc = stack[-1]
                out.append(_d(line, col, f"'{ch}' does not match '{o}' opened at line {ol}", "brackets"))
                stack.pop()
            else:
                stack.pop()
        col += 1
        if len(out) >= 20:
            return out
    for o, l, c in stack[-10:]:
        out.append(_d(l, c, f"'{o}' is never closed", "brackets"))
    return out


# ───────────────────────────── deep checks (external parsers) ─────────────────────────────
def _swift(path):
    p = subprocess.run(["swiftc", "-parse", path], capture_output=True, text=True, timeout=8)
    return p.stderr, r"^(?:.*?):(\d+):(\d+): (error|warning): (.*)$", "swiftc"


def _node(path):
    p = subprocess.run(["node", "--check", path], capture_output=True, text=True, timeout=8)
    return p.stderr, None, "node"


def _bash(path, shell="bash"):
    p = subprocess.run([shell, "-n", path], capture_output=True, text=True, timeout=8)
    return p.stderr, None, shell


TOOLS = {
    "swift": ("swiftc", ".swift", _swift),
    "javascript": ("node", ".js", _node),
    "shell": ("bash", ".sh", _bash),
    "ruby": ("ruby", ".rb", lambda p: (subprocess.run(["ruby", "-c", p], capture_output=True, text=True, timeout=8).stderr, None, "ruby")),
    "php": ("php", ".php", lambda p: (subprocess.run(["php", "-l", p], capture_output=True, text=True, timeout=8).stdout, None, "php")),
    "go": ("gofmt", ".go", lambda p: (subprocess.run(["gofmt", "-e", p], capture_output=True, text=True, timeout=8).stderr,
                                      r"^(?:.*?):(\d+):(\d+): (.*)$", "gofmt")),
}


def available(language):
    t = TOOLS.get(language)
    return bool(t and shutil.which(t[0]))


def deep_check(text, language):
    """Diagnostics from the language's own parser, or [] when it is not installed / nothing to report."""
    tool = TOOLS.get(language)
    if not tool or not shutil.which(tool[0]):
        return []
    binary, ext, runner = tool
    path = ""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "buffer" + ext)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        try:
            output, pattern, source = runner(path)
        except (subprocess.TimeoutExpired, OSError):
            return []
    out = []
    if pattern:
        for m in re.finditer(pattern, output, re.M):
            g = m.groups()
            if len(g) == 4:
                out.append(_d(int(g[0]), int(g[1]), g[3], source, severity=g[2]))
            else:
                out.append(_d(int(g[0]), int(g[1]), g[2], source))
    else:                                                # "file: line N: msg" (bash) / "...:N" + "SyntaxError: msg" (node)
        m = re.search(r"line (\d+): (.*)", output)
        if m:
            out.append(_d(int(m.group(1)), 1, m.group(2).strip(), source))
        else:
            m = re.search(r"\.\w+:(\d+)\n(?:.*\n)*?\s*(\w*Error: .*)", output)
            if m:
                out.append(_d(int(m.group(1)), 1, m.group(2).strip(), source))
            elif output.strip() and "Syntax OK" not in output and "No syntax errors" not in output:
                m = re.search(r"on line (\d+)", output) or re.search(r":(\d+)", output)
                out.append(_d(int(m.group(1)) if m else 1, 1, output.strip().splitlines()[-1][:200], source))
    for x in out:                                        # the temp file name is noise in a message
        x["message"] = x["message"].replace(path, "buffer")
    return out[:50]
