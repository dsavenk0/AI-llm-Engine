"""Web access for the agent (stdlib only): search the internet and read pages as clean text.

  search(query)  → [{title, url, snippet}]   DuckDuckGo's HTML endpoint, no key needed; a
                                             SearxNG-style JSON endpoint can replace it
                                             (LLM_SEARCH_URL, e.g. http://host/search)
  fetch(url)     → "title / url / text"      page → readable text: navigation, scripts,
                                             forms and footers dropped, headings/lists/code kept

Private and loopback addresses are refused (a page must not be able to steer the agent into
the user's LAN), redirects are re-checked, downloads are size-capped.
"""
import html
import ipaddress
import json
import os
import re
import socket
import urllib.error
import urllib.parse
import urllib.request
from html.parser import HTMLParser

# search engines and many docs sites turn away unfamiliar agents; this is the user's own request
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15"
MAX_BYTES = 1_500_000
TEXT_TYPES = ("text/", "application/json", "application/xml", "application/xhtml", "application/javascript")
SKIP = {"script", "style", "noscript", "svg", "nav", "header", "footer", "aside", "form", "iframe",
        "button", "select", "template", "head"}
BLOCK = {"p", "div", "section", "article", "main", "br", "tr", "table", "ul", "ol", "blockquote", "hr",
         "dl", "dt", "dd", "figure", "figcaption", "details", "summary"}


# ───────────────────────────── safety ─────────────────────────────
def check_url(url, allow_private=False):
    u = urllib.parse.urlsplit(url)
    if u.scheme not in ("http", "https") or not u.hostname:
        raise ValueError(f"only http(s) URLs are allowed: {url}")
    if allow_private:
        return url
    try:
        infos = socket.getaddrinfo(u.hostname, u.port or (443 if u.scheme == "https" else 80))
    except socket.gaierror as e:
        raise ValueError(f"cannot resolve {u.hostname}: {e}")
    for info in infos:
        ip = ipaddress.ip_address(info[4][0].split("%")[0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
            raise ValueError(f"refusing to fetch a private/local address ({u.hostname} → {ip})")
    return url


class _Redirects(urllib.request.HTTPRedirectHandler):
    allow_private = False

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        check_url(urllib.parse.urljoin(req.full_url, newurl), self.allow_private)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _open(url, allow_private, timeout, data=None, headers=None):
    check_url(url, allow_private)
    handler = _Redirects()
    handler.allow_private = allow_private
    opener = urllib.request.build_opener(handler)
    req = urllib.request.Request(url, data=data, headers={"User-Agent": UA, "Accept-Language": "en,ru;q=0.8",
                                                          **(headers or {})})
    return opener.open(req, timeout=timeout)


def _read(resp, limit=MAX_BYTES):
    body = resp.read(limit + 1)
    return body[:limit], len(body) > limit


# ───────────────────────────── page → text ─────────────────────────────
VOID = {"br", "hr", "img", "input", "meta", "link", "area", "base", "col", "embed", "source", "track", "wbr"}
CHROME_ROLES = {"navigation", "banner", "contentinfo", "complementary", "search"}
CHROME_WORDS = {"nav", "navbar", "navigation", "sidebar", "menu", "breadcrumb", "breadcrumbs", "related",
                "footer", "cookie", "cookies", "toc", "skip", "sphinxsidebar"}


def _is_chrome(tag, attrs):
    """Page furniture (navigation, sidebars, footers…) by tag, ARIA role or class/id word."""
    if tag in SKIP:
        return True
    a = dict(attrs)
    if (a.get("role") or "") in CHROME_ROLES or a.get("aria-hidden") == "true":
        return True
    words = set(re.split(r"[\s_-]+", ((a.get("class") or "") + " " + (a.get("id") or "")).lower()))
    return bool(words & CHROME_WORDS)


class _Text(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.out, self.stack, self.skip, self.pre, self.title, self.in_title = [], [], 0, 0, "", False

    def handle_starttag(self, tag, attrs):
        if tag == "title":
            self.in_title = True
        if tag not in VOID:
            chrome = _is_chrome(tag, attrs)
            self.stack.append((tag, chrome))
            self.skip += chrome
            if chrome:
                return
        if self.skip:
            return
        if re.fullmatch(r"h[1-6]", tag):
            self.out.append("\n\n" + "#" * int(tag[1]) + " ")
        elif tag == "li":
            self.out.append("\n- ")
        elif tag == "pre":
            self.pre += 1
            self.out.append("\n```\n")
        elif tag == "code" and not self.pre:
            self.out.append("`")
        elif tag in BLOCK:
            self.out.append("\n")

    def handle_startendtag(self, tag, attrs):
        if tag in BLOCK and not self.skip:
            self.out.append("\n")

    def handle_endtag(self, tag):
        if tag == "title":
            self.in_title = False
        if tag in VOID:
            return
        for k in range(len(self.stack) - 1, -1, -1):  # close the nearest matching open tag, tolerate junk
            if self.stack[k][0] == tag:
                for _, chrome in self.stack[k:]:
                    self.skip -= chrome
                del self.stack[k:]
                break
        else:
            return
        if self.skip:
            return
        if re.fullmatch(r"h[1-6]", tag):
            self.out.append("\n")
        elif tag == "pre":
            self.pre = max(0, self.pre - 1)
            self.out.append("\n```\n")
        elif tag == "code" and not self.pre:
            self.out.append("`")
        elif tag in BLOCK:
            self.out.append("\n")

    def handle_data(self, data):
        if self.in_title:
            self.title += data
        if self.skip:
            return
        self.out.append(data if self.pre else re.sub(r"\s+", " ", data))


def _tidy(text):
    """Collapse blank runs and repeated lines in prose; code inside ``` fences is left exactly as is."""
    parts = text.split("```")
    for k in range(0, len(parts), 2):                      # even parts are prose
        lines, last = [], None
        for ln in re.sub(r"[ \t]+\n", "\n", parts[k]).split("\n"):
            ln = ln.lstrip(" ")
            if ln.strip() and ln == last:                  # tables of links repeat their label
                continue
            lines.append(ln)
            last = ln if ln.strip() else last
        parts[k] = re.sub(r"\n{3,}", "\n\n", "\n".join(lines))
    return "```".join(parts)


def html_to_text(src):
    """(title, text) of an HTML document."""
    p = _Text()
    p.feed(src)
    return re.sub(r"\s+", " ", p.title).strip(), _tidy("".join(p.out)).strip()


def _decode(body, content_type):
    m = re.search(r"charset=([\w-]+)", content_type or "", re.I) or re.search(rb"<meta[^>]+charset=[\"']?([\w-]+)", body[:4096], re.I)
    enc = (m.group(1).decode() if m and isinstance(m.group(1), bytes) else (m.group(1) if m else "utf-8"))
    try:
        return body.decode(enc, "replace")
    except LookupError:
        return body.decode("utf-8", "replace")


def fetch(url, max_chars=8000, allow_private=False, timeout=20):
    """A page as readable text, at most `max_chars` long (the rest is reported, not silently cut)."""
    try:
        with _open(url, allow_private, timeout) as r:
            ctype = r.headers.get("Content-Type", "")
            final = r.geturl()
            if ctype and not ctype.lower().startswith(TEXT_TYPES):
                u = urllib.parse.urlsplit(final)
                return f"error: {u.hostname}{u.path[:80]} is {ctype.split(';')[0]}, not a text page"
            body, cut = _read(r)
    except urllib.error.HTTPError as e:
        e.close()
        return f"error: HTTP {e.code} for {url}"
    except (urllib.error.URLError, OSError, ValueError) as e:
        return f"error: {e}"
    src = _decode(body, ctype)
    if "html" in ctype.lower() or re.search(r"<html|<body|<!doctype html", src[:2000], re.I):
        title, text = html_to_text(src)
    else:
        title, text = "", src.strip()
    note = ""
    if len(text) > max_chars:
        note = f"\n… [{len(text) - max_chars} more characters; ask for a narrower page or search within it]"
        text = text[:max_chars]
    elif cut:
        note = "\n… [page larger than 1.5 MB, truncated]"
    return f"{title}\n{final}\n\n{text}{note}".strip()


# ───────────────────────────── search ─────────────────────────────
class _DDG(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.results, self.cur, self.field = [], None, None

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        cls = a.get("class") or ""
        if tag == "a" and "result__a" in cls.split():
            href = a.get("href", "")
            q = urllib.parse.parse_qs(urllib.parse.urlsplit(href).query)
            url = q["uddg"][0] if "uddg" in q else urllib.parse.urljoin("https://duckduckgo.com", href)
            self.cur = {"title": "", "url": url, "snippet": ""}
            self.results.append(self.cur)
            self.field = "title"
        elif tag == "a" and "result__snippet" in cls.split() and self.cur is not None:
            self.field = "snippet"

    def handle_endtag(self, tag):
        if tag == "a":
            self.field = None

    def handle_data(self, data):
        if self.field and self.cur is not None:
            self.cur[self.field] += data


def parse_results(page):
    p = _DDG()
    p.feed(page)
    out = []
    for r in p.results:
        r = {k: re.sub(r"\s+", " ", v).strip() for k, v in r.items()}
        if r["title"] and r["url"].startswith("http") and "duckduckgo.com/y.js" not in r["url"]:
            out.append(r)
    return out


def search(query, n=5, timeout=20, allow_private=False):
    custom = os.environ.get("LLM_SEARCH_URL")
    try:
        if custom:  # SearxNG-style: GET <url>?q=…&format=json → {"results": [{title, url, content}]}
            with _open(custom + ("&" if "?" in custom else "?") + urllib.parse.urlencode(
                    {"q": query, "format": "json"}), True, timeout) as r:
                data = json.loads(_read(r)[0])
            return [{"title": x.get("title", ""), "url": x.get("url", ""), "snippet": x.get("content", "")}
                    for x in data.get("results", [])[:n]]
        with _open("https://html.duckduckgo.com/html/?" + urllib.parse.urlencode({"q": query}),
                   allow_private, timeout) as r:
            return parse_results(_decode(_read(r)[0], r.headers.get("Content-Type")))[:n]
    except (urllib.error.URLError, OSError, ValueError) as e:
        raise RuntimeError(f"search failed: {e}")


def format_results(results):
    if not results:
        return "(no results)"
    return "\n\n".join(f"{i}. {r['title']}\n   {r['url']}\n   {r['snippet']}" for i, r in enumerate(results, 1))


# ───────────────────────────── tool definitions ─────────────────────────────
def tools(allow=lambda what: True, allow_private=False):
    """{name: (OpenAI tool schema, callable(**args) -> str)}. `allow(what)` is asked before anything
    leaves the machine; returning False makes the tool report that the user declined."""
    def web_search(query, max_results=5):
        if not allow(f"search the web for: {query}"):
            return "user declined web access"
        try:
            return format_results(search(query, max(1, min(int(max_results), 10)), allow_private=allow_private))
        except RuntimeError as e:
            return f"error: {e}"

    def web_fetch(url, max_chars=8000):
        if not allow(f"fetch {url}"):
            return "user declined web access"
        return fetch(url, max(500, min(int(max_chars), 20000)), allow_private=allow_private)

    return {
        "web_search": ({"type": "function", "function": {
            "name": "web_search",
            "description": "Search the internet. Use it for current facts, library/API changes and error messages "
                           "you do not know. Returns titles, URLs and snippets; follow up with web_fetch.",
            "parameters": {"type": "object", "properties": {
                "query": {"type": "string"},
                "max_results": {"type": "integer", "description": "1-10, default 5"}}, "required": ["query"]}}},
            web_search),
        "web_fetch": ({"type": "function", "function": {
            "name": "web_fetch",
            "description": "Download a web page and return it as clean text (navigation and scripts removed).",
            "parameters": {"type": "object", "properties": {
                "url": {"type": "string"},
                "max_chars": {"type": "integer", "description": "default 8000, max 20000"}}, "required": ["url"]}}},
            web_fetch),
    }
