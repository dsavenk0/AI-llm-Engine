"""What an editor knows about a source file, with no dependencies:

  detect_language(path, text)   → "python" | "typescript" | "swift" | … | "plaintext"
  highlight(text, language)     → [(start, end, kind)] character spans for syntax colouring
  outline(text, language)       → nested symbols (classes, functions, methods, headings, tables…)

The tokenizer is deliberately small — keywords, strings, comments, numbers, calls, types — the same
level of detail an editor shows before a language server arrives, and it never raises on broken code.
"""
import ast
import json
import os
import re

# ───────────────────────────── language detection ─────────────────────────────
EXT = {
    ".py": "python", ".pyw": "python", ".pyi": "python",
    ".js": "javascript", ".mjs": "javascript", ".cjs": "javascript", ".jsx": "javascript",
    ".ts": "typescript", ".tsx": "typescript", ".mts": "typescript",
    ".swift": "swift", ".go": "go", ".rs": "rust", ".java": "java", ".kt": "kotlin", ".kts": "kotlin",
    ".c": "c", ".h": "c", ".cc": "cpp", ".cpp": "cpp", ".cxx": "cpp", ".hpp": "cpp", ".hh": "cpp",
    ".m": "objc", ".mm": "objc", ".cs": "csharp", ".rb": "ruby", ".php": "php", ".lua": "lua",
    ".sh": "shell", ".bash": "shell", ".zsh": "shell", ".fish": "shell", ".command": "shell",
    ".html": "html", ".htm": "html", ".xml": "xml", ".svg": "xml", ".plist": "xml",
    ".css": "css", ".scss": "css", ".json": "json", ".jsonc": "json", ".yaml": "yaml", ".yml": "yaml",
    ".toml": "toml", ".md": "markdown", ".markdown": "markdown", ".sql": "sql", ".txt": "plaintext",
    ".ini": "ini", ".cfg": "ini", ".conf": "ini",
}
NAMES = {"dockerfile": "dockerfile", "makefile": "makefile", "gnumakefile": "makefile", "rakefile": "ruby",
         "gemfile": "ruby", "podfile": "ruby", "cmakelists.txt": "cmake", ".gitignore": "plaintext",
         ".zshrc": "shell", ".bashrc": "shell", ".env": "ini"}
SHEBANG = [("python", "python"), ("node", "javascript"), ("deno", "typescript"), ("ruby", "ruby"),
           ("php", "php"), ("perl", "perl"), ("lua", "lua"), ("bash", "shell"), ("zsh", "shell"),
           ("/sh", "shell"), (" sh", "shell"), ("fish", "shell")]


def detect_language(path="", text=""):
    name = os.path.basename(path or "").lower()
    if name in NAMES:
        return NAMES[name]
    ext = os.path.splitext(name)[1]
    if ext in EXT:
        return EXT[ext]
    first = (text or "").lstrip("﻿").split("\n", 1)[0]
    if first.startswith("#!"):
        for needle, lang in SHEBANG:
            if needle in first:
                return lang
    head = (text or "")[:4000].lstrip()
    if not head:
        return "plaintext"
    low = head.lower()
    if low.startswith("<!doctype html") or low.startswith("<html") or re.search(r"<(div|body|head|script)\b", low):
        return "html"
    if low.startswith("<?xml"):
        return "xml"
    if head[0] in "{[":
        try:
            json.loads(text)
            return "json"
        except ValueError:
            pass
    lines = [l for l in head.splitlines() if l.strip()][:40]
    score = lambda *pats: sum(bool(re.match(p, l)) for l in lines for p in pats)
    guesses = {
        "python": score(r"\s*def \w+\(", r"\s*class \w+.*:", r"(from \S+ )?import \S+", r'\s*if __name__ == '),
        "swift": score(r"\s*(func|struct|enum|protocol|extension) \w+", r"import (Foundation|SwiftUI|UIKit|AppKit)",
                       r"\s*(let|var) \w+: \w+"),
        "go": score(r"package \w+", r"func (\(\w+ \*?\w+\) )?\w+\(", r'import \('),
        "rust": score(r"\s*(pub )?fn \w+", r"use \w+(::\w+)+", r"\s*impl\b", r"\s*let mut "),
        "javascript": score(r"\s*(const|let|var) \w+ = ", r"\s*function \w+\(", r"\s*(import|export) .* from ",
                            r"module\.exports"),
        "java": score(r"\s*public (static )?(class|void|int|String)", r"package [\w.]+;", r"import [\w.]+;"),
        "c": score(r"#include\s*[<\"]", r"\s*(int|void|char) \w+\(.*\)\s*\{?"),
        "markdown": score(r"#{1,6} \S", r"\s*[-*] \S", r"```"),
        "shell": score(r"\s*(echo|export|cd|if \[|fi$|for \w+ in)\b", r"\w+=\S*$"),
        "yaml": score(r"[\w-]+:( |$)", r"\s*- \S"),
    }
    best = max(guesses, key=guesses.get)
    return best if guesses[best] >= 2 else "plaintext"


# ───────────────────────────── highlighting ─────────────────────────────
def _kw(s):
    return frozenset(s.split())


C_LIKE = _kw("if else for while do switch case default break continue return goto sizeof typedef struct union "
             "enum static const extern volatile register inline void int char short long float double signed "
             "unsigned auto bool true false NULL nullptr class public private protected virtual new delete "
             "namespace using template typename this try catch throw")
LANGS = {
    "python": dict(
        kw=_kw("False None True and as assert async await break class continue def del elif else except finally "
               "for from global if import in is lambda nonlocal not or pass raise return try while with yield "
               "match case self cls"),
        line=["#"], block=[], strings=["'''", '"""', "'", '"'], decorator="@"),
    "javascript": dict(
        kw=_kw("async await break case catch class const continue debugger default delete do else export extends "
               "finally for from function if import in instanceof let new of return static super switch this "
               "throw try typeof var void while with yield true false null undefined"),
        line=["//"], block=[("/*", "*/")], strings=["`", "'", '"']),
    "typescript": dict(
        kw=_kw("async await break case catch class const continue debugger default delete do else enum export "
               "extends finally for from function if implements import in instanceof interface let namespace new "
               "of private protected public readonly return static super switch this throw try type typeof var "
               "void while with yield abstract declare as is keyof true false null undefined any string number "
               "boolean unknown never"),
        line=["//"], block=[("/*", "*/")], strings=["`", "'", '"']),
    "swift": dict(
        kw=_kw("actor as associatedtype async await break case catch class continue default defer deinit do "
               "else enum extension fallthrough false fileprivate final for func guard if import in indirect "
               "init inout internal is lazy let mutating nil nonmutating open operator override private "
               "protocol public repeat rethrows return self Self static struct subscript super switch throws "
               "throw true try typealias var weak where while @MainActor @Published @State @ObservedObject"),
        line=["//"], block=[("/*", "*/")], strings=['"""', '"']),
    "go": dict(
        kw=_kw("break case chan const continue default defer else fallthrough for func go goto if import "
               "interface map package range return select struct switch type var true false nil iota"),
        line=["//"], block=[("/*", "*/")], strings=["`", '"', "'"]),
    "rust": dict(
        kw=_kw("as async await break const continue crate dyn else enum extern false fn for if impl in let "
               "loop match mod move mut pub ref return self Self static struct super trait true type unsafe "
               "use where while"),
        line=["//"], block=[("/*", "*/")], strings=['"', "'"]),
    "java": dict(kw=C_LIKE | _kw("abstract boolean byte extends final finally implements import instanceof "
                                 "interface native package super synchronized throws transient null"),
                 line=["//"], block=[("/*", "*/")], strings=['"""', '"', "'"]),
    "kotlin": dict(kw=_kw("as break class continue do else false for fun if in interface is null object "
                          "package return super this throw true try typealias val var when while override "
                          "open data sealed companion suspend private public internal import"),
                   line=["//"], block=[("/*", "*/")], strings=['"""', '"', "'"]),
    "c": dict(kw=C_LIKE, line=["//"], block=[("/*", "*/")], strings=['"', "'"], preproc=True),
    "cpp": dict(kw=C_LIKE | _kw("constexpr override final noexcept explicit friend mutable operator"),
                line=["//"], block=[("/*", "*/")], strings=['"', "'"], preproc=True),
    "objc": dict(kw=C_LIKE | _kw("@interface @implementation @end @property @selector self super nil YES NO id"),
                 line=["//"], block=[("/*", "*/")], strings=['@"', '"', "'"], preproc=True),
    "csharp": dict(kw=C_LIKE | _kw("abstract as base bool checked class delegate event explicit finally "
                                   "foreach implicit in interface internal is lock namespace null object "
                                   "out override params readonly ref sealed string using var async await"),
                   line=["//"], block=[("/*", "*/")], strings=['@"', '"', "'"]),
    "ruby": dict(kw=_kw("alias and begin break case class def defined? do else elsif end ensure false for if "
                        "in module next nil not or redo rescue retry return self super then true undef unless "
                        "until when while yield require include attr_accessor"),
                 line=["#"], block=[("=begin", "=end")], strings=['"', "'"]),
    "php": dict(kw=_kw("abstract and array as break case catch class const continue declare default do echo "
                       "else elseif empty extends final finally fn for foreach function global if implements "
                       "include interface isset namespace new null or private protected public require "
                       "return static switch this throw trait true false try use var while yield"),
                line=["//", "#"], block=[("/*", "*/")], strings=['"', "'"]),
    "lua": dict(kw=_kw("and break do else elseif end false for function goto if in local nil not or repeat "
                       "return then true until while"), line=["--"], block=[("--[[", "]]")], strings=['"', "'"]),
    "shell": dict(kw=_kw("if then else elif fi for while until do done case esac in function select time "
                         "return exit export local readonly unset source echo cd set shift trap"),
                  line=["#"], block=[], strings=['"', "'", "`"]),
    "sql": dict(kw=_kw("select from where group by order having insert into values update set delete create "
                       "table index view drop alter add column primary key foreign references join left right "
                       "inner outer on as and or not null is in like between limit offset union distinct "
                       "case when then else end begin commit rollback unique default constraint function "
                       "SELECT FROM WHERE GROUP BY ORDER HAVING INSERT INTO VALUES UPDATE SET DELETE CREATE "
                       "TABLE INDEX VIEW DROP ALTER ADD COLUMN PRIMARY KEY FOREIGN REFERENCES JOIN LEFT RIGHT "
                       "INNER OUTER ON AS AND OR NOT NULL IS IN LIKE BETWEEN LIMIT OFFSET UNION DISTINCT CASE "
                       "WHEN THEN ELSE END BEGIN COMMIT ROLLBACK UNIQUE DEFAULT CONSTRAINT FUNCTION"),
                line=["--"], block=[("/*", "*/")], strings=["'", '"']),
    "css": dict(kw=_kw("important @media @import @font-face @keyframes @supports"), line=[],
                block=[("/*", "*/")], strings=['"', "'"]),
    "yaml": dict(kw=_kw("true false null yes no on off"), line=["#"], block=[], strings=['"', "'"]),
    "toml": dict(kw=_kw("true false"), line=["#"], block=[], strings=['"""', "'''", '"', "'"]),
    "ini": dict(kw=_kw("true false"), line=["#", ";"], block=[], strings=['"', "'"]),
    "json": dict(kw=_kw("true false null"), line=[], block=[], strings=['"']),
    "dockerfile": dict(kw=_kw("FROM RUN CMD LABEL EXPOSE ENV ADD COPY ENTRYPOINT VOLUME USER WORKDIR ARG "
                              "ONBUILD STOPSIGNAL HEALTHCHECK SHELL AS"), line=["#"], block=[], strings=['"', "'"]),
    "makefile": dict(kw=_kw("ifeq ifneq ifdef ifndef else endif include define endef export .PHONY"),
                     line=["#"], block=[], strings=['"', "'"]),
}
_NUM = re.compile(r"(?<![\w.])(?:0[xX][0-9a-fA-F_]+|0[bB][01_]+|\d[\d_]*\.?\d*(?:[eE][+-]?\d+)?)(?![\w])")
_WORD = re.compile(r"[A-Za-z_$@#][\w$]*(?:\?)?")


MAX_HIGHLIGHT = 1_500_000      # characters; beyond this an editor shows plain text
MAX_OUTLINE = 2_000_000


def highlight(text, language):
    """Character spans [(start, end, kind)] for syntax colouring; kinds: comment string keyword number
    function type decorator tag property heading. Anything it does not understand is left unmarked."""
    if len(text) > MAX_HIGHLIGHT:
        return []
    if language == "markdown":
        return _highlight_markdown(text)
    if language in ("html", "xml"):
        return _highlight_markup(text)
    cfg = LANGS.get(language)
    if not cfg:
        return []
    spans, i, n = [], 0, len(text)
    kw, line_c, block_c, strings = cfg["kw"], cfg["line"], cfg["block"], cfg["strings"]
    at_line_start = True
    while i < n:
        c = text[i]
        if c == "\n":
            at_line_start = True
            i += 1
            continue
        if c in " \t\r":
            i += 1
            continue
        matched = False
        for b0, b1 in block_c:
            if text.startswith(b0, i):
                j = text.find(b1, i + len(b0))
                j = n if j < 0 else j + len(b1)
                spans.append((i, j, "comment"))
                i, matched = j, True
                break
        if matched:
            at_line_start = False
            continue
        for lc in line_c:
            if text.startswith(lc, i):
                j = text.find("\n", i)
                j = n if j < 0 else j
                spans.append((i, j, "comment"))
                i, matched = j, True
                break
        if matched:
            continue
        if cfg.get("preproc") and at_line_start and c == "#":
            j = text.find("\n", i)
            j = n if j < 0 else j
            spans.append((i, j, "decorator"))
            i = j
            continue
        at_line_start = False
        for q in strings:
            if text.startswith(q, i):
                end = _string_end(text, i + len(q), q, multiline=len(q) == 3 or q == "`")
                spans.append((i, end, "string"))
                i, matched = end, True
                break
        if matched:
            continue
        if language == "python" and c == "@" and (i == 0 or text[i - 1] == "\n" or text[:i].rstrip(" \t").endswith("\n")):
            m = re.match(r"@[\w.]+", text[i:])
            if m:
                spans.append((i, i + m.end(), "decorator"))
                i += m.end()
                continue
        m = _NUM.match(text, i)
        if m and (c.isdigit()):
            spans.append((i, m.end(), "number"))
            i = m.end()
            continue
        m = _WORD.match(text, i)
        if m:
            word = m.group(0)
            end = m.end()
            if word in kw:
                kind = "keyword"
            elif re.match(r"\s*\(", text[end:end + 8]) and language not in ("json", "yaml", "toml", "ini"):
                kind = "function"
            elif word[:1].isupper() and len(word) > 1 and not word.isupper():
                kind = "type"
            elif word.isupper() and len(word) > 2 and "_" in word or word in ("NULL", "YES", "NO"):
                kind = "constant"
            else:
                kind = None
            if kind:
                spans.append((i, end, kind))
            i = end
            continue
        i += 1
    if language in ("yaml", "toml", "ini", "json"):
        spans = _add_keys(text, spans, language)
    return sorted(spans)


def _string_end(text, i, quote, multiline):
    n = len(text)
    while i < n:
        c = text[i]
        if c == "\\":
            i += 2
            continue
        if text.startswith(quote, i):
            return i + len(quote)
        if c == "\n" and not multiline:
            return i
        i += 1
    return n


def _add_keys(text, spans, language):
    """Keys of key: value / key = value lines (and "key": in JSON) as `property`. One pass with a dict of
    replaced starts — rescanning the span list per key made large JSON files take minutes."""
    comments = [(a, b) for a, b, k in spans if k == "comment"]
    ci = 0
    keys = {}                                   # start -> end of every key found
    if language == "json":
        for m in re.finditer(r'("(?:[^"\\\n]|\\.)*")\s*:', text):
            keys[m.start(1)] = m.end(1)
    else:
        for m in re.finditer(r"(?m)^(\s*-?\s*)([\w.\-\"']+)(\s*[:=])", text):
            s0 = m.start(2)
            while ci < len(comments) and comments[ci][1] <= s0:     # spans are in text order: advance a cursor
                ci += 1
            if ci < len(comments) and comments[ci][0] <= s0 < comments[ci][1]:
                continue
            keys[s0] = m.end(2)
    out = []
    seen = set()
    for a, b, k in spans:
        if a in keys and (k == "string" or language != "json"):
            continue                                                  # replaced by the property span below
        out.append((a, b, k))
    for a, b in keys.items():
        out.append((a, b, "property"))
    return out


def _highlight_markdown(text):
    spans, pos, in_fence = [], 0, False
    for line in text.split("\n"):
        end = pos + len(line)
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            spans.append((pos, end, "comment"))
        elif in_fence:
            spans.append((pos, end, "string"))
        elif re.match(r"#{1,6}\s", line):
            spans.append((pos, end, "heading"))
        else:
            for m in re.finditer(r"`[^`\n]+`", line):
                spans.append((pos + m.start(), pos + m.end(), "string"))
            for m in re.finditer(r"\[[^\]\n]+\]\([^)\n]+\)", line):
                spans.append((pos + m.start(), pos + m.end(), "function"))
        pos = end + 1
    return sorted(spans)


def _highlight_markup(text):
    spans = []
    for m in re.finditer(r"<!--.*?-->", text, re.S):
        spans.append((m.start(), m.end(), "comment"))
    for m in re.finditer(r"</?([A-Za-z][\w:-]*)", text):
        spans.append((m.start(1), m.end(1), "tag"))
    for m in re.finditer(r'([\w:-]+)=("[^"]*"|\'[^\']*\')', text):
        spans.append((m.start(1), m.end(1), "property"))
        spans.append((m.start(2), m.end(2), "string"))
    return sorted(spans)


# ───────────────────────────── structure (outline) ─────────────────────────────
def _sym(kind, name, line, end=None, children=None):
    return {"kind": kind, "name": name, "line": line, "end": end or line, "children": children or []}


def outline(text, language):
    """Nested symbols with 1-based line numbers. Never raises: a file that does not parse still gets a
    best-effort outline."""
    if len(text) > MAX_OUTLINE:
        return []
    try:
        if language == "python":
            return _outline_python(text)
        if language == "markdown":
            return _outline_markdown(text)
        if language == "json":
            return _outline_json(text)
        if language in ("yaml",):
            return _outline_indent_keys(text)
        if language in ("toml", "ini"):
            return _outline_sections(text)
        if language in ("html", "xml"):
            return _outline_html(text)
        if language == "makefile":
            return _outline_regex_flat(text, [(r"^([A-Za-z0-9_.\-/%$()]+)\s*:(?!=)", "target")])
        if language == "dockerfile":
            return _outline_regex_flat(text, [(r"(?i)^FROM\s+\S+(?:\s+AS\s+(\S+))?", "stage")])
        if language == "sql":
            return _outline_regex_flat(text, [(r"(?i)^\s*CREATE\s+(?:OR\s+REPLACE\s+)?(?:TEMP\w*\s+)?"
                                               r"(TABLE|VIEW|FUNCTION|PROCEDURE|INDEX|TRIGGER)\s+(?:IF\s+NOT\s+EXISTS\s+)?([\w.\"`]+)",
                                               "create")])
        if language in BRACE or language in ("ruby", "shell", "lua"):
            return _outline_scan(text, language)
    except Exception:
        return []
    return []


def _outline_python(text):
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return _outline_scan(text, "python")

    def walk(body, in_class=False):
        out = []
        for node in body:
            if isinstance(node, ast.ClassDef):
                out.append(_sym("class", node.name, node.lineno, node.end_lineno, walk(node.body, True)))
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                kind = "method" if in_class else "function"
                out.append(_sym(kind, node.name, node.lineno, node.end_lineno, walk(node.body, False)))
            elif isinstance(node, ast.Assign) and not in_class:
                for t in node.targets:
                    if isinstance(t, ast.Name) and t.id.isupper():
                        out.append(_sym("constant", t.id, node.lineno, node.end_lineno))
            elif isinstance(node, (ast.If, ast.Try, ast.With, ast.For, ast.While)):
                out += walk(getattr(node, "body", []), in_class)
                for attr in ("orelse", "finalbody"):
                    out += walk(getattr(node, attr, []), in_class)
        return out
    return walk(tree.body)


def _outline_markdown(text):
    root, stack, in_fence = [], [], False
    lines = text.split("\n")
    for i, line in enumerate(lines, 1):
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            continue
        m = None if in_fence else re.match(r"(#{1,6})\s+(.*?)\s*#*\s*$", line)
        if not m:
            continue
        level, sym = len(m.group(1)), _sym("heading", m.group(2), i)
        sym["level"] = level
        while stack and stack[-1]["level"] >= level:
            stack.pop()["end"] = i - 1
        (stack[-1]["children"] if stack else root).append(sym)
        stack.append(sym)
    for s in stack:
        s["end"] = len(lines)
    return root


def _outline_json(text):
    """Top-level keys of an object (or the length of an array) with the line each starts on."""
    try:
        data = json.loads(text)
    except ValueError:
        return []
    if not isinstance(data, dict):
        return [_sym("array" if isinstance(data, list) else "value", f"[{len(data)}]" if isinstance(data, list) else "value", 1)]
    out = []
    depth = 0
    in_str = False
    esc = False
    line = 1
    key_buf, i, n = None, 0, len(text)
    while i < n:
        c = text[i]
        if c == "\n":
            line += 1
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        elif c == '"':
            in_str = True
            if depth == 1:
                j = i + 1
                while j < n and (text[j] != '"' or text[j - 1] == "\\"):
                    j += 1
                k = j + 1
                while k < n and text[k] in " \t\r\n":
                    k += 1
                if k < n and text[k] == ":":
                    key = text[i + 1:j]
                    v = data.get(json.loads('"' + key + '"')) if key else None
                    kind = "object" if isinstance(v, dict) else "array" if isinstance(v, list) else "key"
                    out.append(_sym(kind, key, line))
                    i = j
                    in_str = False
        elif c in "{[":
            depth += 1
        elif c in "}]":
            depth -= 1
        i += 1
    for a, b in zip(out, out[1:]):
        a["end"] = max(a["line"], b["line"] - 1)
    if out:
        out[-1]["end"] = line
    return out


def _outline_indent_keys(text):
    root, stack = [], []
    for i, raw in enumerate(text.split("\n"), 1):
        m = re.match(r"^(\s*)([\w.\-\"']+)\s*:(?:\s|$)", raw)
        if not m or raw.lstrip().startswith("#") or raw.lstrip().startswith("- "):
            continue
        indent = len(m.group(1).replace("\t", "  "))
        sym = _sym("key", m.group(2).strip("\"'"), i)
        sym["indent"] = indent
        while stack and stack[-1]["indent"] >= indent:
            stack.pop()
        (stack[-1]["children"] if stack else root).append(sym)
        stack.append(sym)
    return _trim_indent(root)


def _trim_indent(items, depth=0):
    for s in items:
        s.pop("indent", None)
        _trim_indent(s["children"], depth + 1)
    return [s for s in items]


def _outline_sections(text):
    out = []
    for i, line in enumerate(text.split("\n"), 1):
        m = re.match(r"^\s*\[{1,2}\s*([^\]]+?)\s*\]{1,2}\s*$", line)
        if m:
            out.append(_sym("section", m.group(1), i))
    for a, b in zip(out, out[1:]):
        a["end"] = b["line"] - 1
    return out


def _outline_html(text):
    out = []
    for i, line in enumerate(text.split("\n"), 1):
        for m in re.finditer(r"<(h[1-6])\b[^>]*>(.*?)</\1>", line, re.I):
            out.append(_sym("heading", re.sub(r"<[^>]+>", "", m.group(2)).strip(), i))
        for m in re.finditer(r"<(\w+)\b[^>]*\bid=[\"']([^\"']+)[\"']", line):
            if not m.group(1).lower().startswith("h") or len(m.group(1)) != 2:
                out.append(_sym("id", f"#{m.group(2)}", i))
        for m in re.finditer(r"<(script|style)\b", line, re.I):
            out.append(_sym("block", m.group(1).lower(), i))
    return out


def _outline_regex_flat(text, patterns):
    out = []
    for i, line in enumerate(text.split("\n"), 1):
        for pat, kind in patterns:
            m = re.match(pat, line)
            if m:
                groups = [g for g in m.groups() if g]
                out.append(_sym(kind, (groups[-1] if groups else m.group(0)).strip(), i))
                break
    for a, b in zip(out, out[1:]):
        a["end"] = b["line"] - 1
    return out


# declarations for languages with braces (or end / indentation); (regex, kind, container?)
_ID = r"[A-Za-z_$][\w$]*"
BRACE = {
    "javascript": [
        (rf"^\s*(?:export\s+)?(?:default\s+)?(?:abstract\s+)?class\s+({_ID})", "class", True),
        (rf"^\s*(?:export\s+)?(?:default\s+)?(?:async\s+)?function\s*\*?\s*({_ID})", "function", True),
        (rf"^\s*(?:export\s+)?(?:const|let|var)\s+({_ID})\s*=\s*(?:async\s*)?(?:\([^)]*\)|{_ID})\s*=>", "function", True),
        (rf"^\s*(?:export\s+)?(?:const|let|var)\s+({_ID})\s*=\s*(?:async\s+)?function", "function", True),
        (rf"^\s*(?:static\s+)?(?:async\s+)?(?:get\s+|set\s+)?(?!if\b|for\b|while\b|switch\b|catch\b|return\b)({_ID})\s*\([^)]*\)\s*\{{", "method", True),
    ],
    "typescript": [
        (rf"^\s*(?:export\s+)?(?:default\s+)?(?:abstract\s+)?class\s+({_ID})", "class", True),
        (rf"^\s*(?:export\s+)?interface\s+({_ID})", "interface", True),
        (rf"^\s*(?:export\s+)?(?:const\s+)?enum\s+({_ID})", "enum", True),
        (rf"^\s*(?:export\s+)?type\s+({_ID})\s*(?:<[^>]*>)?\s*=", "type", False),
        (rf"^\s*(?:export\s+)?(?:default\s+)?(?:async\s+)?function\s*\*?\s*({_ID})", "function", True),
        (rf"^\s*(?:export\s+)?(?:const|let|var)\s+({_ID})\s*(?::[^=]+)?=\s*(?:async\s*)?(?:\([^)]*\)|{_ID})\s*(?::[^=]+)?=>", "function", True),
        (rf"^\s*(?:(?:public|private|protected|static|async|readonly|override)\s+)*(?!if\b|for\b|while\b|switch\b|catch\b|return\b)({_ID})\s*(?:<[^>]*>)?\([^)]*\)\s*(?::[^{{]+)?\{{", "method", True),
    ],
    "swift": [
        (rf"^\s*(?:(?:public|private|fileprivate|internal|open|final)\s+)*(class|struct|enum|protocol|actor|extension)\s+({_ID})", "type", True),
        (rf"^\s*(?:@\w+\s+)*(?:(?:public|private|fileprivate|internal|open|static|class|final|override|mutating)\s+)*func\s+({_ID})", "function", True),
        (rf"^\s*(?:(?:public|private|internal|convenience|required)\s+)*(init)(?=\s*[?(])", "init", True),
    ],
    "go": [
        (rf"^type\s+({_ID})\s+(?:struct|interface)", "type", True),
        (rf"^func\s+\(\s*\w+\s+\*?({_ID})\)\s*({_ID})", "method", True),
        (rf"^func\s+({_ID})", "function", True),
    ],
    "rust": [
        (rf"^\s*(?:pub(?:\([^)]*\))?\s+)?(struct|enum|trait|mod|union)\s+({_ID})", "type", True),
        (rf"^\s*impl(?:<[^>]*>)?\s+(?:[\w:<>,\s]+\s+for\s+)?({_ID})", "impl", True),
        (rf"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?(?:unsafe\s+)?fn\s+({_ID})", "function", True),
    ],
    "java": [
        (rf"^\s*(?:(?:public|private|protected|static|final|abstract)\s+)*(class|interface|enum|record)\s+({_ID})", "class", True),
        (rf"^\s*(?!(?:return|new|else|throw|case|if|for|while|switch|catch|do|try|assert|yield)\b)"
         rf"(?:(?:public|private|protected|static|final|abstract|synchronized|native|default)\s+)*"
         rf"[\w<>\[\],.?]+\s+({_ID})\s*\([^;=]*\)\s*(?:throws\s+[\w,.\s]+)?\{{", "method", True),
    ],
    "kotlin": [
        (rf"^\s*(?:(?:public|private|internal|data|sealed|open|abstract|enum)\s+)*(class|interface|object)\s+({_ID})", "class", True),
        (rf"^\s*(?:(?:public|private|internal|override|suspend|open)\s+)*fun\s+(?:<[^>]*>\s*)?(?:[\w.]+\.)?({_ID})", "function", True),
    ],
    "c": [
        (rf"^\s*(?:typedef\s+)?(struct|enum|union)\s+({_ID})", "type", True),
        (rf"^(?!\s*(?:if|for|while|switch|return|else)\b)[A-Za-z_][\w\s\*]*?\s+\**({_ID})\s*\([^;]*\)\s*\{{?\s*$", "function", True),
    ],
    "csharp": [
        (rf"^\s*(?:(?:public|private|protected|internal|static|sealed|abstract|partial)\s+)*(class|interface|enum|struct|record)\s+({_ID})", "class", True),
        (rf"^\s*(?!(?:return|new|else|throw|case|if|for|foreach|while|switch|catch|using|lock|do|try)\b)"
         rf"(?:(?:public|private|protected|internal|static|virtual|override|async|abstract)\s+)*"
         rf"[\w<>\[\],.?]+\s+({_ID})\s*\([^;=]*\)\s*\{{?\s*$", "method", True),
    ],
    "php": [
        (rf"^\s*(?:abstract\s+|final\s+)?(class|interface|trait)\s+({_ID})", "class", True),
        (rf"^\s*(?:(?:public|private|protected|static)\s+)*function\s+&?({_ID})", "function", True),
    ],
    "objc": [
        (rf"^@(interface|implementation|protocol)\s+({_ID})", "class", False),
        (rf"^[-+]\s*\([^)]*\)\s*({_ID})", "method", False),
    ],
}
BRACE["cpp"] = BRACE["c"] + [(rf"^\s*(?:template\s*<[^>]*>\s*)?(class|namespace)\s+({_ID})", "class", True)]
INDENT_BASED = {
    "ruby": [(rf"^\s*(class|module)\s+({_ID}(?:::{_ID})*)", "class"), (rf"^\s*def\s+(?:self\.)?({_ID}[?!=]?)", "function")],
    "shell": [(rf"^\s*(?:function\s+)?({_ID})\s*\(\s*\)\s*\{{?", "function"), (rf"^\s*function\s+({_ID})", "function")],
    "lua": [(rf"^\s*(?:local\s+)?function\s+([\w.:]+)", "function")],
    "python": [(rf"^\s*class\s+({_ID})", "class"), (rf"^\s*(?:async\s+)?def\s+({_ID})", "function")],
}


def _mask(text, language):
    """The text with comments and string contents blanked (newlines and length kept), so braces and
    keywords inside them cannot confuse the scanner."""
    out = list(text)
    for a, b, kind in highlight(text, language):
        if kind in ("comment", "string"):
            for k in range(a, b):
                if out[k] != "\n":
                    out[k] = " "
    return "".join(out)


def _outline_scan(text, language):
    masked = _mask(text, language)
    lines = masked.split("\n")
    if language in INDENT_BASED:
        return _outline_by_indent(lines, INDENT_BASED[language])
    pats = [(re.compile(p), k, c) for p, k, c in BRACE[language]]
    root, stack, depth = [], [], 0
    for i, line in enumerate(lines, 1):
        parent = stack[-1][0] if stack else None
        sym = None
        for rx, kind, container in pats:
            m = rx.match(line)
            if m:
                groups = [g for g in m.groups() if g]
                name = groups[-1] if groups else m.group(0).strip()
                if kind == "method" and language in ("javascript", "typescript") and not (
                        stack and stack[-1][0]["kind"] in ("class", "type", "interface", "impl", "enum")):
                    kind = "function"           # `name(...) {` outside a class body is a call/statement shape, a function
                if kind == "type" and len(groups) > 1:
                    kind = groups[0]
                sym = _sym(kind, name, i)
                sym["_container"] = container
                break
        opens, closes = line.count("{"), line.count("}")
        if sym:
            (parent["children"] if parent else root).append(sym)
            if sym["_container"] and opens > closes:
                stack.append((sym, depth))
            elif sym["_container"] and opens == closes and opens:
                sym["end"] = i
            if not sym["_container"] or opens <= closes:
                sym.pop("_container", None)
        depth += opens - closes
        while stack and depth <= stack[-1][1]:
            done = stack.pop()[0]
            done["end"] = i
            done.pop("_container", None)
    for s, _ in stack:
        s["end"] = len(lines)
        s.pop("_container", None)
    return root


def _outline_by_indent(lines, patterns):
    pats = [(re.compile(p), k) for p, k in patterns]
    root, stack = [], []
    for i, line in enumerate(lines, 1):
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip())
        while stack and stack[-1][1] >= indent and not line.strip().startswith(("end", ")", "}")):
            stack.pop()[0]["end"] = i - 1
        for rx, kind in pats:
            m = rx.match(line)
            if m:
                groups = [g for g in m.groups() if g]
                sym = _sym(kind, groups[-1], i)
                if kind == "function" and stack and stack[-1][0]["kind"] == "class":
                    sym["kind"] = "method"
                (stack[-1][0]["children"] if stack else root).append(sym)
                stack.append((sym, indent))
                break
    for sym, _ in stack:
        sym["end"] = len(lines)
    return root


def flatten(symbols, depth=0):
    """[(depth, symbol)] in document order — convenient for a flat list view."""
    out = []
    for s in symbols:
        out.append((depth, s))
        out += flatten(s["children"], depth + 1)
    return out
