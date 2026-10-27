"""Open VSX extensions, the light way (an optional module of the editor, off by default; stdlib only).

An extension is a .vsix — a zip with a package.json. This module installs one and uses only what is
*declarative* in it: colour themes, snippets and language names. It never runs an extension's code (that
needs VS Code's Node extension host and API, which a native editor does not have), and it says so for the
parts it skips. Everything comes from open-vsx.org over https; the microsoft marketplace is not used (its
terms limit it to Microsoft's own products).

Storage: ~/.llm/extensions/<publisher>.<name>-<version>/ holding package.json, a manifest.json written here
and just the theme/snippet files the extension points at.
"""
import hashlib
import io
import json
import os
import re
import shutil
import tempfile
import urllib.parse
import urllib.request
import zipfile

HOST = "open-vsx.org"
# open-vsx.org redirects file downloads to the Eclipse Foundation's CDN; nothing else is ever contacted
HOSTS = (HOST, "openvsx.eclipsecontent.org")
API = f"https://{HOST}/api"
MAX_VSIX = 40 * 1024 * 1024
MAX_MEMBER = 6 * 1024 * 1024           # one extracted file
MAX_UNPACKED = 60 * 1024 * 1024        # what a zip may claim to unpack to
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,99}$")
VERSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.+-]{0,63}$")


class ExtensionError(ValueError):
    pass


def store_dir():
    return os.environ.get("LLM_EXTENSIONS_DIR") or os.path.join(os.path.expanduser("~"), ".llm", "extensions")


# ───────────────────────────── network (open-vsx.org only) ─────────────────────────────
def _check_url(url):
    p = urllib.parse.urlsplit(url)
    if p.scheme != "https" or (p.hostname or "").lower() not in HOSTS:
        raise ExtensionError(f"only https://{HOST} is used, not {url!r}")
    return url


def fetch(url, limit=2 * 1024 * 1024, timeout=30):
    """GET `url` (open-vsx.org over https, redirects included) and return at most `limit` bytes; more is an error."""
    _check_url(url)

    class _Same(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            _check_url(newurl)
            return super().redirect_request(req, fp, code, msg, headers, newurl)

    opener = urllib.request.build_opener(_Same)
    req = urllib.request.Request(url, headers={"User-Agent": "llm-editor/1", "Accept": "application/json, */*"})
    try:
        with opener.open(req, timeout=timeout) as r:
            data = r.read(limit + 1)
    except urllib.error.HTTPError as e:
        e.close()
        raise ExtensionError(f"open-vsx.org answered {e.code}")
    except (urllib.error.URLError, OSError) as e:
        raise ExtensionError(f"cannot reach open-vsx.org: {e}")
    if len(data) > limit:
        raise ExtensionError("the download is larger than allowed")
    return data


CATEGORIES = ("Themes", "Snippets", "Programming Languages")      # what this editor can use of an extension


def search(query, size=20, get=fetch, category=None):
    """`category`: only extensions Open VSX files under it (Themes / Snippets / Programming Languages) — most of
    the others are code-only (agents, debuggers) and give this editor nothing to use."""
    params = {"query": query, "size": max(1, min(int(size), 50)), "sortBy": "relevance"}
    if category in CATEGORIES:
        params["category"] = category
    q = urllib.parse.urlencode(params)
    try:
        data = json.loads(get(f"{API}/-/search?{q}"))
        listing = data.get("extensions") or []
        if not isinstance(listing, list):
            raise ValueError("extensions is not a list")
    except (ValueError, AttributeError):
        raise ExtensionError("open-vsx.org answered with something unexpected")
    out = []
    for e in listing:
        if not isinstance(e, dict) or not e.get("namespace") or not e.get("name"):
            continue
        out.append({"id": f"{e.get('namespace')}.{e.get('name')}", "namespace": e.get("namespace"),
                    "name": e.get("name"), "version": e.get("version"),
                    "displayName": e.get("displayName") or e.get("name"),
                    "description": e.get("description") or "", "downloads": e.get("downloadCount") or 0,
                    "verified": bool(e.get("verified"))})
    return out


# ───────────────────────────── JSON with comments (themes use it) ─────────────────────────────
def jsonc(text):
    """json.loads for VS Code files: // and /* */ comments and trailing commas are allowed there."""
    out, i, n, in_str = [], 0, len(text), False
    while i < n:
        c = text[i]
        if in_str:
            out.append(c)
            if c == "\\" and i + 1 < n:
                out.append(text[i + 1])
                i += 1
            elif c == '"':
                in_str = False
        elif c == '"':
            in_str = True
            out.append(c)
        elif text.startswith("//", i):
            while i < n and text[i] != "\n":
                i += 1
            continue
        elif text.startswith("/*", i):
            j = text.find("*/", i + 2)
            i = n if j < 0 else j + 2
            continue
        else:
            out.append(c)
        i += 1
    return json.loads(re.sub(r",(\s*[}\]])", r"\1", "".join(out)).lstrip("﻿"))


# ───────────────────────────── installing ─────────────────────────────
def _safe_member(name):
    """A zip member's path under extension/, or None when it is not a plain relative path."""
    norm = os.path.normpath(name.replace("\\", "/"))
    if norm.startswith(("/", "..")) or ".." in norm.split("/") or os.path.isabs(norm):
        return None
    return norm


def _read_member(z, name):
    info = z.getinfo(name)
    if info.file_size > MAX_MEMBER:
        raise ExtensionError(f"{name} is too large")
    return z.read(info)


def _root(z):
    """The folder inside the zip that holds package.json (normally `extension/`)."""
    for n in z.namelist():
        if n.endswith("package.json") and n.count("/") <= 1 and _safe_member(n):
            return os.path.dirname(n)
    raise ExtensionError("not an extension: package.json is missing")


def _url(v):
    """An http(s) address from a package.json field, or None (git+https://… and git@… are tidied or dropped)."""
    if not isinstance(v, str):
        return None
    v = v.strip()
    if v.startswith("git+"):
        v = v[4:]
    if v.endswith(".git"):
        v = v[:-4]
    return v if re.match(r"^https?://[^\s]+$", v) and len(v) < 300 else None


def _inspect(z, root):
    """The manifest dict and the files to keep, from a .vsix opened as `z`."""
    rel = lambda p: _safe_member((root + "/" if root else "") + p.lstrip("./"))
    pkg = jsonc(_read_member(z, rel("package.json") or "").decode("utf-8", "replace")) if rel("package.json") else {}
    if not isinstance(pkg, dict) or not pkg.get("name") or not pkg.get("publisher"):
        raise ExtensionError("package.json has no name/publisher")
    contrib = pkg.get("contributes") if isinstance(pkg.get("contributes"), dict) else {}
    names = set(z.namelist())
    keep, themes, snippets, languages = {"package.json"}, [], [], []

    def need(path):
        member = rel(path)
        if member and member in names:
            keep.add(os.path.relpath(member, root) if root else member)
            return os.path.relpath(member, root) if root else member
        return None

    for t in contrib.get("themes") or []:
        f = need(str(t.get("path", ""))) if isinstance(t, dict) else None
        if f:
            themes.append({"label": str(t.get("label") or t.get("id") or f), "uiTheme": str(t.get("uiTheme") or "vs-dark"),
                           "file": f})
            # a theme may `include` another one, relative to itself (a cycle must not loop forever)
            todo, seen = [f], {f}
            while todo:
                cur = todo.pop()
                try:
                    inc = jsonc(_read_member(z, rel(cur)).decode("utf-8", "replace")).get("include")
                except (ValueError, KeyError, AttributeError):
                    inc = None
                if isinstance(inc, str):
                    nxt = os.path.normpath(os.path.join(os.path.dirname(cur), inc))
                    if nxt not in seen and need(nxt):
                        seen.add(nxt)
                        todo.append(nxt)
    for s in contrib.get("snippets") or []:
        f = need(str(s.get("path", ""))) if isinstance(s, dict) else None
        if f and s.get("language"):
            snippets.append({"language": str(s["language"]), "file": f})
    for lang in contrib.get("languages") or []:
        if isinstance(lang, dict) and lang.get("id"):
            languages.append({"id": str(lang["id"]), "extensions": [str(x) for x in lang.get("extensions") or []],
                              "aliases": [str(x) for x in lang.get("aliases") or []]})
    skipped = sorted(k for k in contrib if k not in ("themes", "snippets", "languages", "iconThemes") and contrib[k])
    code = bool(pkg.get("main") or pkg.get("browser"))
    manifest = {
        "id": f"{pkg['publisher']}.{pkg['name']}", "publisher": str(pkg["publisher"]), "name": str(pkg["name"]),
        "displayName": str(pkg.get("displayName") or pkg["name"]), "version": str(pkg.get("version") or "0"),
        "description": str(pkg.get("description") or ""), "themes": themes, "snippets": snippets,
        "languages": languages, "skipped": skipped, "has_code": code,
        "homepage": _url(pkg.get("homepage")) or _url((pkg.get("repository") or {}).get("url")
                                                      if isinstance(pkg.get("repository"), dict) else pkg.get("repository")),
    }
    return manifest, sorted(keep)


def install_vsix(data, expected_sha256=None, dest_root=None):
    """Unpack the declarative parts of a .vsix (bytes). Returns the manifest. Nothing is executed."""
    if len(data) > MAX_VSIX:
        raise ExtensionError("the extension package is too large")
    if expected_sha256 is not None:
        want = str(expected_sha256).strip().split()
        if not want or hashlib.sha256(data).hexdigest() != want[0].lower():
            raise ExtensionError("the download does not match its published checksum")
    try:
        z = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile:
        raise ExtensionError("not a .vsix (zip) file")
    with z:
        if sum(i.file_size for i in z.infolist()) > MAX_UNPACKED:
            raise ExtensionError("the package unpacks to more than allowed")
        root = _root(z)
        manifest, keep = _inspect(z, root)
        base = dest_root or store_dir()
        folder = os.path.join(base, f"{manifest['id']}-{manifest['version']}")
        if not NAME_RE.match(manifest["publisher"]) or not NAME_RE.match(manifest["name"]) \
                or not VERSION_RE.match(manifest["version"]):
            raise ExtensionError("the extension's id is not a plain name")
        tmp = tempfile.mkdtemp(dir=base) if os.path.isdir(base) else (os.makedirs(base, exist_ok=True) or tempfile.mkdtemp(dir=base))
        try:
            for rel_path in keep:
                member = (root + "/" if root else "") + rel_path
                target = os.path.join(tmp, rel_path)
                if not os.path.realpath(target).startswith(os.path.realpath(tmp) + os.sep):
                    raise ExtensionError("unsafe path in the package")
                os.makedirs(os.path.dirname(target), exist_ok=True)
                with open(target, "wb") as f:
                    f.write(_read_member(z, member))
            with open(os.path.join(tmp, "manifest.json"), "w", encoding="utf-8") as f:
                json.dump(manifest, f, ensure_ascii=False, indent=1)
            for old in _folders(base, manifest["id"]):             # one version of an extension at a time
                shutil.rmtree(old, ignore_errors=True)
            os.replace(tmp, folder)
        except BaseException:
            shutil.rmtree(tmp, ignore_errors=True)
            raise
    return manifest


def install(namespace, name, version=None, get=fetch):
    """Download from Open VSX and install; returns the manifest."""
    for part in (namespace, name):
        if not isinstance(part, str) or not NAME_RE.match(part):
            raise ExtensionError("invalid extension id")
    if version is not None and not VERSION_RE.match(str(version)):
        raise ExtensionError("invalid version")
    try:
        meta = json.loads(get(f"{API}/{urllib.parse.quote(namespace)}/{urllib.parse.quote(name)}"
                              + (f"/{urllib.parse.quote(str(version))}" if version else "")))
    except ValueError:
        raise ExtensionError("open-vsx.org answered with something unexpected")
    if not isinstance(meta, dict):
        raise ExtensionError("open-vsx.org answered with something unexpected")
    files = meta.get("files") if isinstance(meta.get("files"), dict) else {}
    if not files.get("download"):
        raise ExtensionError("this extension has no download")
    if not files.get("sha256"):                    # the file comes from a CDN: its checksum is what vouches for it
        raise ExtensionError("open-vsx.org publishes no checksum for this extension; not installing it")
    digest = get(_check_url(files["sha256"]), limit=4096).decode("ascii", "replace")
    data = get(_check_url(files["download"]), limit=MAX_VSIX)
    return install_vsix(data, digest)


# ───────────────────────────── what is installed ─────────────────────────────
def _folders(base, ext_id):
    """Folders holding exactly the extension `ext_id` (any version). Matching on the folder-name prefix would
    take "acme.night-extra-1.0" for a version of "acme.night"; the manifest says what a folder really is."""
    if not os.path.isdir(base):
        return []
    found = []
    for d in os.listdir(base):
        path = os.path.join(base, d, "manifest.json")
        if not d.startswith(ext_id + "-") or not os.path.isfile(path):
            continue
        try:
            with open(path, encoding="utf-8") as f:
                if json.load(f).get("id") == ext_id:
                    found.append(os.path.join(base, d))
        except (OSError, ValueError, AttributeError):
            continue
    return found


def _disabled_path(base):
    return os.path.join(base, "disabled.json")


def disabled_ids(base=None):
    """Ids the person switched off (kept installed, their themes and snippets ignored)."""
    base = base or store_dir()
    try:
        with open(_disabled_path(base), encoding="utf-8") as f:
            data = json.load(f)
        return {x for x in data if isinstance(x, str)} if isinstance(data, list) else set()
    except (OSError, ValueError):
        return set()


def set_enabled(ext_id, enabled, base=None):
    base = base or store_dir()
    if not isinstance(ext_id, str) or not re.match(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$", ext_id):
        raise ExtensionError("invalid extension id")
    if not _folders(base, ext_id):
        raise ExtensionError("that extension is not installed")
    off = disabled_ids(base)
    (off.discard if enabled else off.add)(ext_id)
    os.makedirs(base, exist_ok=True)
    tmp = _disabled_path(base) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(sorted(off), f)
    os.replace(tmp, _disabled_path(base))
    return not (ext_id in off)


def installed(base=None):
    base = base or store_dir()
    off = disabled_ids(base)
    out = []
    if os.path.isdir(base):
        for d in sorted(os.listdir(base)):
            p = os.path.join(base, d, "manifest.json")
            try:
                with open(p, encoding="utf-8") as f:
                    m = json.load(f)
                if not isinstance(m, dict):
                    continue
                m["folder"] = d
                m["enabled"] = m.get("id") not in off
                m["usable"] = bool(m.get("themes") or m.get("snippets"))
                out.append(m)
            except (OSError, ValueError):
                continue
    return out


def remove(ext_id, base=None):
    base = base or store_dir()
    if not isinstance(ext_id, str) or not re.match(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$", ext_id):
        raise ExtensionError("invalid extension id")
    found = _folders(base, ext_id)
    for folder in found:
        shutil.rmtree(folder, ignore_errors=True)
    if found and ext_id in disabled_ids(base):          # a later install starts switched on
        set_enabled_raw(base, disabled_ids(base) - {ext_id})
    return len(found)


def set_enabled_raw(base, ids):
    os.makedirs(base, exist_ok=True)
    tmp = _disabled_path(base) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(sorted(ids), f)
    os.replace(tmp, _disabled_path(base))


def detail(ext_id, base=None, limit=300):
    """What one installed extension gives this editor: its themes, its snippets (language, prefix, description)
    and what it carries that is not used — the "what can I do with it" page of the extensions sheet."""
    base = base or store_dir()
    for m in installed(base):
        if m.get("id") == ext_id:
            snips = []
            for s in m.get("snippets", []):
                try:
                    with open(os.path.join(base, m["folder"], s["file"]), encoding="utf-8", errors="replace") as f:
                        data = jsonc(f.read())
                except (OSError, ValueError):
                    continue
                for title, snip in (data.items() if isinstance(data, dict) else []):
                    if not isinstance(snip, dict):
                        continue
                    p = snip.get("prefix")
                    for prefix in ([p] if isinstance(p, str) else [x for x in p or [] if isinstance(x, str)]):
                        snips.append({"language": s["language"], "prefix": prefix,
                                      "description": str(snip.get("description") or title)})
            return {"id": ext_id, "enabled": m["enabled"], "usable": m["usable"], "version": m.get("version"),
                    "homepage": m.get("homepage"), "themes": [t["label"] for t in m.get("themes", [])],
                    "languages": [lang["id"] for lang in m.get("languages", [])],
                    "snippets": snips[:limit], "snippet_total": len(snips), "skipped": m.get("skipped", []),
                    "has_code": bool(m.get("has_code")), "folder": os.path.join(base, m["folder"])}
    raise ExtensionError("that extension is not installed")


def _version_key(v):
    return tuple(int(x) if x.isdigit() else 0 for x in re.split(r"[.\-+]", str(v)))


def updates(base=None, get=fetch):
    """[{id, installed, latest}] for installed extensions whose newest Open VSX version is higher (one request
    each, so it is asked for on demand; an extension Open VSX does not answer for is skipped)."""
    base = base or store_dir()
    out = []
    for m in installed(base):
        ns, _, name = str(m.get("id", "")).partition(".")
        if not NAME_RE.match(ns) or not NAME_RE.match(name):
            continue
        try:
            meta = json.loads(get(f"{API}/{urllib.parse.quote(ns)}/{urllib.parse.quote(name)}"))
            latest = str(meta.get("version") or "")
        except (ExtensionError, ValueError, AttributeError):
            continue
        if VERSION_RE.match(latest) and _version_key(latest) > _version_key(m.get("version", "0")):
            out.append({"id": m["id"], "installed": m.get("version"), "latest": latest})
    return out


# ───────────────────────────── themes ─────────────────────────────
# our highlighter's token kinds → TextMate scopes to look for, most specific first
KIND_SCOPES = {
    "keyword": ["keyword.control", "keyword.operator.new", "keyword", "storage.modifier", "storage.type", "storage"],
    "string": ["string.quoted", "string"],
    "comment": ["comment"],
    "number": ["constant.numeric", "constant"],
    "function": ["entity.name.function", "support.function", "meta.function-call", "meta.function"],
    "type": ["entity.name.type", "entity.name.class", "support.type", "support.class", "storage.type"],
    "decorator": ["meta.decorator", "entity.name.function.decorator", "storage.type.annotation",
                  "punctuation.decorator", "meta.annotation"],
    "property": ["variable.other.property", "variable.other.object.property", "support.type.property-name",
                 "meta.object-literal.key", "variable.other.member", "variable.object.property"],
    "constant": ["constant.language", "constant.other", "variable.other.constant", "constant"],
    "tag": ["entity.name.tag", "meta.tag", "punctuation.definition.tag"],
    "heading": ["markup.heading", "entity.name.section", "markup.heading.markdown"],
}
HEX = re.compile(r"^#([0-9a-fA-F]{3,4}|[0-9a-fA-F]{6}|[0-9a-fA-F]{8})$")


def _hex(c):
    """#rgb/#rgba/#rrggbb/#rrggbbaa → #rrggbb (alpha dropped), or None."""
    if not isinstance(c, str) or not HEX.match(c):
        return None
    h = c[1:]
    if len(h) in (3, 4):
        h = "".join(ch * 2 for ch in h[:3])
    return "#" + h[:6].lower()


def _luminance(hexc):
    r, g, b = (int(hexc[i:i + 2], 16) / 255 for i in (1, 3, 5))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _load_theme(folder, rel_file, depth=0):
    with open(os.path.join(folder, rel_file), encoding="utf-8", errors="replace") as f:
        data = jsonc(f.read())
    if not isinstance(data, dict):
        raise ExtensionError("not a theme file")
    colors, tokens = {}, []
    inc = data.get("include")
    if isinstance(inc, str) and depth < 4:
        sub = os.path.normpath(os.path.join(os.path.dirname(rel_file), inc))
        if not sub.startswith(".."):
            try:
                parent = _load_theme(folder, sub, depth + 1)
                colors.update(parent["colors"])
                tokens += parent["tokens"]
            except (OSError, ValueError):
                pass
    colors.update({k: v for k, v in (data.get("colors") or {}).items() if isinstance(v, str)})
    tokens += [t for t in (data.get("tokenColors") or data.get("settings") or []) if isinstance(t, dict)]
    return {"colors": colors, "tokens": tokens, "type": data.get("type")}


def theme(ext_id, label, base=None):
    """A theme from an installed extension, converted to what the editor uses:
    {"label", "dark", "background", "foreground", "kinds": {kind: "#rrggbb"}}."""
    base = base or store_dir()
    if ext_id in disabled_ids(base):
        raise ExtensionError("that extension is switched off")
    for folder in _folders(base, ext_id):
        with open(os.path.join(folder, "manifest.json"), encoding="utf-8") as f:
            manifest = json.load(f)
        for t in manifest.get("themes", []):
            if t["label"] != label:
                continue
            raw = _load_theme(folder, t["file"])
            colors = raw["colors"]
            bg = _hex(colors.get("editor.background")) or ("#1e1e1e" if t["uiTheme"] != "vs" else "#ffffff")
            fg = _hex(colors.get("editor.foreground"))
            rules = []                                  # (scope, order, colour)
            for order, rule in enumerate(raw["tokens"]):
                settings = rule.get("settings") or {}
                colour = _hex(settings.get("foreground"))
                scope = rule.get("scope")
                if colour is None or scope is None:
                    if scope is None and colour and fg is None:       # a scope-less rule is the default colours
                        fg = colour
                    continue
                for s in (scope if isinstance(scope, list) else str(scope).split(",")):
                    s = str(s).strip()
                    if s:
                        rules.append((s, order, colour))
            kinds = {}
            for kind, candidates in KIND_SCOPES.items():
                for cand in candidates:
                    best = None
                    for s, order, colour in rules:
                        if cand == s or cand.startswith(s + "."):
                            key = (len(s), order)
                            if best is None or key >= best[0]:
                                best = (key, colour)
                    if best:
                        kinds[kind] = best[1]
                        break
            dark = t["uiTheme"] != "vs" if t["uiTheme"] in ("vs", "vs-dark", "hc-black", "hc-light") else _luminance(bg) < 0.5
            if raw["type"] in ("light", "dark"):
                dark = raw["type"] == "dark"
            return {"label": label, "id": ext_id, "dark": dark, "background": bg,
                    "foreground": fg or ("#d4d4d4" if dark else "#1f1f1f"), "kinds": kinds}
    raise ExtensionError("theme not found")


# ───────────────────────────── snippets ─────────────────────────────
def snippets(language, base=None):
    """[{prefix, body, description}] for `language` from all installed extensions (body is a list of lines)."""
    base = base or store_dir()
    out = []
    for m in installed(base):
        if not m.get("enabled", True):
            continue
        for s in m.get("snippets", []):
            if s.get("language") != language:
                continue
            try:
                with open(os.path.join(base, m["folder"], s["file"]), encoding="utf-8", errors="replace") as f:
                    data = jsonc(f.read())
            except (OSError, ValueError):
                continue
            if not isinstance(data, dict):
                continue
            for title, snip in data.items():
                if not isinstance(snip, dict):
                    continue
                prefixes = snip.get("prefix")
                prefixes = [prefixes] if isinstance(prefixes, str) else [p for p in prefixes or [] if isinstance(p, str)]
                body = snip.get("body")
                body = [body] if isinstance(body, str) else [b for b in body or [] if isinstance(b, str)]
                for p in prefixes:
                    out.append({"prefix": p, "body": body, "description": str(snip.get("description") or title)})
    return out
