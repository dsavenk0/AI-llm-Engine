"""Web access for the agent (stdlib only): search the internet and read pages as clean text.

  search(query)  → [{title, url, snippet}]   DuckDuckGo's HTML endpoint, no key needed; a
                                             SearxNG-style JSON endpoint can replace it
                                             (LLM_SEARCH_URL, e.g. http://host/search)
  fetch(url)     → "title / url / text"      page → readable text: navigation, scripts,
                                             forms and footers dropped, headings/lists/code kept
  multi_search(queries, focus)               several queries × the sources that suit them
                                             (DuckDuckGo, Wikipedia, GitHub, arXiv, OpenAlex,
                                             Hacker News, Stack Exchange — their documented
                                             APIs), at once under one deadline; merged, de-duplicated
                                             and ranked (reciprocal rank fusion + lexical relevance)
  read(url, query)                           a page or PDF with its metadata and the passages most
                                             relevant to the question, with their real offsets

harness/research.py builds the bounded, cited research on top of these.

Private and loopback addresses are refused (a page must not be able to steer the agent into
the user's LAN), redirects are re-checked, downloads are size-capped.
"""
import gzip
import http.client
import ipaddress
import json
import math
import os
import re
import socket
import threading
import time
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
        _require_public(ipaddress.ip_address(info[4][0].split("%")[0]), u.hostname)
    return url


def _require_public(ip, host=""):
    """Refuse anything that is not an ordinary public address: private, loopback, link-local, shared (CGNAT),
    reserved, multicast, unspecified — and an IPv4 address wearing an IPv6 mask (::ffff:127.0.0.1)."""
    if ip.version == 6 and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    if not ip.is_global or ip.is_multicast or ip.is_loopback or ip.is_link_local:
        raise ValueError(f"refusing to fetch a private/local address ({host} → {ip})")


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
    resp = opener.open(req, timeout=timeout)
    if not allow_private:
        # the name was checked before connecting; check where the connection really went too, so a name that
        # resolves differently the second time (DNS rebinding) cannot be used to read a local service
        try:
            peer = resp.fp.raw._sock.getpeername()[0]
        except (AttributeError, OSError, IndexError):
            peer = None
        if peer:
            try:
                _require_public(ipaddress.ip_address(peer.split("%")[0]), urllib.parse.urlsplit(url).hostname or "")
            except ValueError:
                resp.close()
                raise
    return resp


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


MAX_PDF_BYTES = 12_000_000          # a paper's PDF; read through engine/documents.py's text-layer parser


def _meta(src):
    """(canonical URL, published date) a page states about itself: <link rel=canonical>, article:published_time,
    JSON-LD datePublished, <time datetime>. Only what the page says — never a guess."""
    head = src[:200_000]
    canon = re.search(r"<link[^>]+rel=[\"']canonical[\"'][^>]*href=[\"']([^\"']+)", head, re.I) or \
        re.search(r"<link[^>]+href=[\"']([^\"']+)[\"'][^>]*rel=[\"']canonical", head, re.I)
    pub = re.search(r"<meta[^>]+(?:property|name)=[\"'](?:article:published_time|datePublished|date|dc\.date)[\"']"
                    r"[^>]*content=[\"']([^\"']{8,40})", head, re.I) or \
        re.search(r"\"datePublished\"\s*:\s*\"([^\"]{8,40})\"", head) or \
        re.search(r"<time[^>]+datetime=[\"']([^\"']{8,40})", head, re.I)
    return (canon.group(1) if canon else None), (pub.group(1) if pub else None)


def _get(url, allow_private=False, timeout=20):
    """{url, final_url, title, text, cut, kind, canonical, published, retrieved_at} or {"error": …} — the one
    reader under fetch() and read(). HTML is turned to text here, a PDF goes through engine/documents.py."""
    retrieved = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    try:
        with _open(url, allow_private, timeout) as r:
            ctype = r.headers.get("Content-Type", "")
            final = r.geturl()
            pdf = "pdf" in ctype.lower() or (not ctype and final.lower().endswith(".pdf"))
            if ctype and not pdf and not ctype.lower().startswith(TEXT_TYPES):
                u = urllib.parse.urlsplit(final)
                return {"error": f"{u.hostname}{u.path[:80]} is {ctype.split(';')[0]}, not a text page"}
            body, cut = _read(r, MAX_PDF_BYTES if pdf else MAX_BYTES)
    except urllib.error.HTTPError as e:
        e.close()
        return {"error": f"HTTP {e.code} for {url}"}
    except (urllib.error.URLError, OSError, ValueError, http.client.HTTPException) as e:
        return {"error": str(e) or type(e).__name__}
    out = {"url": url, "final_url": final, "retrieved_at": retrieved, "cut": cut, "canonical": None, "published": None}
    if pdf or body[:5] == b"%PDF-":
        try:
            from engine import documents
            doc = documents.pdf_text(body)
        except Exception as e:                                  # a damaged PDF is a failed read, not a crash
            return {"error": f"could not read the PDF: {e}"}
        if not doc["text"]:
            return {"error": doc.get("note") or "the PDF has no text layer"}
        return {**out, "title": "", "text": doc["text"], "kind": "document"}
    src = _decode(body, ctype)
    if "html" in ctype.lower() or re.search(r"<html|<body|<!doctype html", src[:2000], re.I):
        title, text = html_to_text(src)
        out["canonical"], out["published"] = _meta(src)
    else:
        title, text = "", src.strip()
    return {**out, "title": title, "text": text, "kind": "page"}


def fetch(url, max_chars=8000, allow_private=False, timeout=20):
    """A page as readable text, at most `max_chars` long (the rest is reported, not silently cut)."""
    page = _get(url, allow_private, timeout)
    if "error" in page:
        return f"error: {page['error']}"
    title, text, final, cut = page["title"], page["text"], page["final_url"], page["cut"]
    note = ""
    if len(text) > max_chars:
        note = f"\n… [{len(text) - max_chars} more characters; ask for a narrower page or search within it]"
        text = text[:max_chars]
    elif cut:
        note = "\n… [page larger than 1.5 MB, truncated]"
    return f"{title}\n{final}\n\n{text}{note}".strip()


def _sections(text):
    """[(offset, heading)] of the markdown-style headings html_to_text keeps, to name a passage's section."""
    return [(m.start(), m.group(2).strip()) for m in re.finditer(r"^(#{1,6}) (.+)$", text, re.M)]


def read(url, query="", budget=6000, allow_private=False, timeout=20, max_passages=6):
    """A page (or PDF) for research: its metadata and the passages most relevant to `query` — by lexical
    relevance over the whole text, not its first N characters (engine/documents.select_excerpts) — each with
    its section heading and real offsets into `text`. {"error": …} when it cannot be read."""
    page = _get(url, allow_private, timeout)
    if "error" in page:
        return page
    from engine import documents
    text = page["text"]
    sel = documents.select_excerpts(text, query, budget=budget, size=900, max_passages=max_passages)
    heads = _sections(text)
    passages = []
    for p in sel["passages"]:
        sec = next((h for off, h in reversed(heads) if off <= p["start"]), None)
        passages.append({"text": text[p["start"]:p["end"]].strip(), "section": sec,
                         "start": p["start"], "end": p["end"], "score": p["score"]})
    if not passages and text:                                   # nothing matched the words: the opening
        end = min(len(text), budget)
        passages = [{"text": text[:end].strip(), "section": None, "start": 0, "end": end, "score": 0.0}]
    return {**{k: v for k, v in page.items() if k != "text"}, "chars": len(text), "passages": passages}


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


# ───────────────────────────── more sources: their documented APIs ─────────────────────────────
# Each source is a function (query, n, lang, timeout) → [{title, url, snippet, published?}] that raises on failure.
# Only documented, keyless (or optionally keyed) endpoints: no scraping of search pages that forbid it, no CAPTCHA
# games. A source that fails is skipped for this search and rests for a while (cooldown), the others still answer.
API_UA = "AI-llm-Engine/1 (local research assistant; stdlib urllib)"
MAX_API_BYTES = 2_000_000


def _api(url, timeout, headers=None, allow_private=False):
    """GET a public API: JSON or XML bytes, gzip undone within a bound (a small reply cannot expand without limit)."""
    h = {"User-Agent": API_UA, "Accept": "application/json", "Accept-Encoding": "gzip", **(headers or {})}
    with _open(url, allow_private, timeout, headers=h) as r:
        body, cut = _read(r, MAX_API_BYTES)
        if cut:
            raise ValueError("reply too large")
        if r.headers.get("Content-Encoding", "").lower() == "gzip" or body[:2] == b"\x1f\x8b":
            d = gzip.GzipFile(fileobj=__import__("io").BytesIO(body))
            body = d.read(MAX_API_BYTES + 1)
            if len(body) > MAX_API_BYTES:
                raise ValueError("reply too large once decompressed")
        return body


def _strip_tags(s):
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", __import__("html").unescape(s or ""))).strip()


def _lang_of(text):
    """'ru' when the text is mostly Cyrillic, 'es' for a few Spanish markers, else 'en' (for Wikipedia's language)."""
    t = text or ""
    if sum(1 for c in t if "Ѐ" <= c <= "ӿ") > len(t) * 0.3:
        return "ru"
    if re.search(r"[ñ¿¡]|\b(qué|cómo|cuál|dónde|por qué)\b", t, re.I):
        return "es"
    return "en"


FILLER = {"documentation", "docs", "overview", "guide", "tutorial", "experience", "survey", "arxiv", "how", "does",
          "best", "using", "use"}


def _short(query, k=4):
    """A few key terms of a query, for APIs that want every word to match (GitHub, Stack Exchange)."""
    ws = [w for w in terms(re.sub(r"\bsite:\S+|\"", " ", query)) if w not in FILLER and not w.isdigit()]
    return " ".join(ws[:k]) or query


def src_duckduckgo(query, n, lang, timeout):
    return search(query, n=n, timeout=timeout)


def src_wikipedia(query, n, lang, timeout):
    lang = lang or _lang_of(query)
    data = json.loads(_api(f"https://{lang}.wikipedia.org/w/api.php?" + urllib.parse.urlencode(
        {"action": "query", "list": "search", "srsearch": query, "format": "json", "srlimit": n, "utf8": 1}), timeout))
    return [{"title": x["title"], "snippet": _strip_tags(x.get("snippet")),
             "url": f"https://{lang}.wikipedia.org/wiki/" + urllib.parse.quote(x["title"].replace(" ", "_")),
             "published": (x.get("timestamp") or None)} for x in data.get("query", {}).get("search", [])]


def src_github(query, n, lang, timeout):
    """Repositories (the code-search API needs a login: not used). GITHUB_TOKEN / GH_TOKEN raises the anonymous
    10-per-minute limit; it goes in the header only, never in a URL, a prompt or a log."""
    tok = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    h = {"Accept": "application/vnd.github+json", **({"Authorization": f"Bearer {tok}"} if tok else {})}
    data = json.loads(_api("https://api.github.com/search/repositories?" + urllib.parse.urlencode(
        {"q": _short(query, 3), "per_page": n, "sort": "stars"}), timeout, h))
    return [{"title": x["full_name"], "url": x["html_url"],
             "snippet": f"{x.get('description') or ''} (★{x.get('stargazers_count', 0)}, "
                        f"updated {(x.get('pushed_at') or '')[:10]})".strip(),
             "published": None} for x in data.get("items", [])]


ATOM = "{http://www.w3.org/2005/Atom}"


def parse_feed(xml_bytes, n=20):
    """RSS 2.0 or Atom → [{title, url, snippet, published}] with the stdlib parser, bounded input. `published` is
    the feed entry's own date — when it was published, not when anything it describes happened."""
    import xml.etree.ElementTree as ET
    if len(xml_bytes) > MAX_API_BYTES or b"<!ENTITY" in xml_bytes[:4096]:
        raise ValueError("feed too large or declares entities")
    root = ET.fromstring(xml_bytes)
    out = []
    for e in root.iter(ATOM + "entry"):
        link = next((l.get("href") for l in e.findall(ATOM + "link") if l.get("rel") in (None, "alternate")), None)
        out.append({"title": _strip_tags(e.findtext(ATOM + "title")), "url": link or (e.findtext(ATOM + "id") or ""),
                    "snippet": _strip_tags(e.findtext(ATOM + "summary") or e.findtext(ATOM + "content"))[:400],
                    "published": e.findtext(ATOM + "published") or e.findtext(ATOM + "updated")})
    for it in root.iter("item"):
        out.append({"title": _strip_tags(it.findtext("title")), "url": (it.findtext("link") or "").strip(),
                    "snippet": _strip_tags(it.findtext("description"))[:400], "published": it.findtext("pubDate")})
    return [r for r in out if r["title"] and r["url"].startswith("http")][:n]


def src_arxiv(query, n, lang, timeout):
    q = re.sub(r"\bsite:\S+|\"", "", query).strip()
    return parse_feed(_api("https://export.arxiv.org/api/query?" + urllib.parse.urlencode(
        {"search_query": "all:" + q, "max_results": n}), timeout, {"Accept": "application/atom+xml"}), n)


def _openalex_abstract(inv):
    if not isinstance(inv, dict):
        return ""
    words = sorted(((p, w) for w, ps in inv.items() for p in ps[:50]), key=lambda x: x[0])[:80]
    return " ".join(w for _, w in words)


def src_openalex(query, n, lang, timeout):
    data = json.loads(_api("https://api.openalex.org/works?" + urllib.parse.urlencode(
        {"search": re.sub(r"\bsite:\S+", "", query).strip(), "per-page": n}), timeout))
    out = []
    for x in data.get("results", []):
        url = x.get("doi") or (x.get("primary_location") or {}).get("landing_page_url") or x.get("id")
        if x.get("title") and url:
            out.append({"title": x["title"], "url": url, "snippet": _openalex_abstract(x.get("abstract_inverted_index")),
                        "published": x.get("publication_date")})
    return out


def src_hackernews(query, n, lang, timeout):
    data = json.loads(_api("https://hn.algolia.com/api/v1/search?" + urllib.parse.urlencode(
        {"query": query, "hitsPerPage": n, "tags": "story"}), timeout))
    return [{"title": x.get("title") or "", "snippet": f"Hacker News discussion, {x.get('num_comments', 0)} comments",
             "url": x.get("url") or f"https://news.ycombinator.com/item?id={x.get('objectID')}",
             "published": x.get("created_at"), "discussion": True} for x in data.get("hits", []) if x.get("title")]


def src_stackexchange(query, n, lang, timeout):
    data = json.loads(_api("https://api.stackexchange.com/2.3/search/advanced?" + urllib.parse.urlencode(
        {"q": _short(query), "site": "stackoverflow", "pagesize": n, "order": "desc", "sort": "relevance"}), timeout))
    return [{"title": _strip_tags(x["title"]), "url": x["link"], "discussion": True,
             "snippet": f"Stack Overflow, score {x.get('score', 0)}, {'answered' if x.get('is_answered') else 'unanswered'}",
             "published": time.strftime("%Y-%m-%d", time.gmtime(x["creation_date"])) if x.get("creation_date") else None}
            for x in data.get("items", [])]


SOURCES = {"duckduckgo": src_duckduckgo, "wikipedia": src_wikipedia, "github": src_github, "arxiv": src_arxiv,
           "openalex": src_openalex, "hackernews": src_hackernews, "stackexchange": src_stackexchange}
# which sources suit a focus (the first is the general fallback when the others fail)
FOCUS = {"web": ("duckduckgo", "wikipedia"),
         "docs": ("duckduckgo", "github", "stackexchange"),
         "academic": ("arxiv", "openalex", "duckduckgo"),
         "news": ("duckduckgo", "hackernews"),
         "discussions": ("hackernews", "stackexchange", "duckduckgo")}
_COOLDOWN, _COOL_LOCK = {}, threading.Lock()
COOLDOWN_S = 120


def _resting(name):
    with _COOL_LOCK:
        return _COOLDOWN.get(name, 0) > time.time()


def _rest(name, err):
    """A source that refused (429/5xx/timeout) rests: Retry-After when it says so, else COOLDOWN_S."""
    wait = COOLDOWN_S
    if isinstance(err, urllib.error.HTTPError):
        ra = err.headers.get("Retry-After") if err.headers else None
        if ra and ra.isdigit():
            wait = min(int(ra), 900)
    with _COOL_LOCK:
        _COOLDOWN[name] = time.time() + wait


def detect_focus(question):
    """The sources a question calls for, by plain cues (no model call): academic, docs, news, discussions or web."""
    q = (question or "").lower()
    if re.search(r"\b(paper|papers|arxiv|study|studies|research|survey|doi|citation)\b|исследован|стать[яи] |научн", q):
        return "academic"
    if re.search(r"\b(api|library|sdk|error|exception|traceback|install|github|repo|function|python|swift|"
                 r"javascript|typescript|rust|mlx|cli|docs?|documentation)\b|ошибк|библиотек|документац|репозитор", q):
        return "docs"
    if re.search(r"\b(today|latest|news|this week|announced|released|2025|2026|2027)\b|новост|сегодня|вышел|анонс", q):
        return "news"
    if re.search(r"\b(opinion|opinions|experience|reddit|forum|vs\.?|recommend)\b|мнени|отзыв|опыт|посовет", q):
        return "discussions"
    return "web"


# ───────────────────────────── merging and ranking ─────────────────────────────
TRACKING = re.compile(r"^(utm_\w+|fbclid|gclid|dclid|yclid|msclkid|mc_cid|mc_eid|igshid|_hsenc|_hsmi|ref_src|"
                      r"spm|scid|cmpid|__twitter_impression)$", re.I)


def canonical_url(url):
    """The key two links to the same page share: lower-case host without "www.", no default port, no fragment, no
    tracking parameters, no trailing slash — parameters that name the resource are kept, in their order."""
    try:
        u = urllib.parse.urlsplit(url.strip())
    except ValueError:
        return url
    host = (u.hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    if u.port and not ((u.scheme == "http" and u.port == 80) or (u.scheme == "https" and u.port == 443)):
        host += f":{u.port}"
    seen, query = set(), []
    for k, v in urllib.parse.parse_qsl(u.query, keep_blank_values=True):
        if TRACKING.match(k) or (k, v) in seen:
            continue
        seen.add((k, v))
        query.append((k, v))
    path = re.sub(r"/{2,}", "/", u.path or "/")
    if len(path) > 1:
        path = path.rstrip("/")
    if host == "arxiv.org":                     # /pdf/2511.10676v2(.pdf) and /abs/2511.10676 are one paper
        m = re.match(r"^/(?:abs|pdf)/([\w.\-/]+?)(?:v\d+)?(?:\.pdf)?$", path)
        if m:
            path = "/abs/" + m.group(1)
    return urllib.parse.urlunsplit(("", host, path, urllib.parse.urlencode(query), "")).lstrip("/")


_WORD = re.compile(r"[\w][\w+#.-]*", re.U)
STOP = set("the a an of to in on for and or is are was were be by with from as at it this that what how why which who "
           "when does do can could should vs и в во не на с со по к ко о об от до за из у же ли как что это для или".split())


def terms(text):
    return [w for w in (t.lower().strip(".-") for t in _WORD.findall(text or "")) if len(w) > 1 and w not in STOP]


def _norm_title(t):
    return " ".join(terms(t))


def merge(hits):
    """hits: [(query, source, rank, result)] → one entry per page (canonical URL, or the same title on the same
    host), keeping every source and query that found it and the best rank under each."""
    by_key, order = {}, []
    by_title = {}
    for query, source, rank, r in hits:
        key = canonical_url(r.get("url", ""))
        tkey = (key.split("/")[0], _norm_title(r.get("title", "")))
        if key not in by_key and tkey in by_title and tkey[1]:
            key = by_title[tkey]
        e = by_key.get(key)
        if e is None:
            e = by_key[key] = {"title": r.get("title", ""), "url": r.get("url", ""), "snippet": r.get("snippet", ""),
                               "published": r.get("published"), "sources": [], "queries": [], "ranks": {},
                               "discussion": bool(r.get("discussion")), "key": key}
            order.append(key)
            by_title.setdefault(tkey, key)
        if source not in e["sources"]:
            e["sources"].append(source)
        if query not in e["queries"]:
            e["queries"].append(query)
        e["ranks"][(query, source)] = min(rank, e["ranks"].get((query, source), rank))
        if len(r.get("snippet") or "") > len(e["snippet"] or ""):
            e["snippet"] = r.get("snippet") or ""
        e["published"] = e["published"] or r.get("published")
    return [by_key[k] for k in order]


def rank(entries, question, fresh=False, k=60):
    """Order merged results for `question`: reciprocal rank fusion over every (query, source) list each appeared in
    (agreement counts, position counts), plus a BM25-style lexical score of title and snippet against the
    question's terms, a bonus for an exact phrase, and — only when `fresh` — for a recent date. Deterministic;
    no model, no embedding. Each entry gets "score"; ties keep their discovery order."""
    q = terms(question)
    docs = [terms(e["title"]) * 2 + terms(e["snippet"]) for e in entries]     # the title weighs twice
    N = len(docs) or 1
    avg = sum(len(d) for d in docs) / N or 1
    df = {t: sum(1 for d in docs if t in d) for t in set(q)}
    phrase = (question or "").strip().strip('"').lower()
    year = time.gmtime().tm_year
    for e, d in zip(entries, docs):
        rrf = sum(1.0 / (k + r) for r in e["ranks"].values())
        bm = 0.0
        for t in set(q):
            f = d.count(t)
            if f:
                idf = math.log(1 + (N - df[t] + 0.5) / (df[t] + 0.5))
                bm += idf * f * 2.2 / (f + 1.2 * (0.25 + 0.75 * len(d) / avg))
        cover = sum(1 for t in set(q) if t in d) / (len(set(q)) or 1)
        score = 30 * rrf + 0.35 * bm + 0.6 * cover
        if len(phrase) > 8 and phrase in (e["title"] + " " + e["snippet"]).lower():
            score += 0.5
        if fresh and e.get("published"):
            m = re.search(r"(20\d\d)", str(e["published"]))
            if m:
                score += max(0.0, 0.4 - 0.2 * (year - int(m.group(1))))
        e["score"] = round(score, 4)
    return sorted(entries, key=lambda e: -e["score"])


def diversify(entries, per_host=2):
    """The same ranking with at most `per_host` results of one site among the leaders; the rest follow."""
    first, rest, count = [], [], {}
    for e in entries:
        h = e["key"].split("/")[0]
        count[h] = count.get(h, 0) + 1
        (first if count[h] <= per_host else rest).append(e)
    return first + rest


def multi_search(queries, focus="web", n=6, deadline=15.0, per_source_timeout=8.0, sources=None,
                 max_workers=6, should_stop=None, fresh=False, question=None):
    """Every query on every source of `focus` at once, under one `deadline`: what answered in time is merged,
    de-duplicated and ranked; what failed is named in "errors" and rests (cooldown). Returns
    {"results": [entries], "errors": {source: why}, "requests": n, "seconds": t}. Never raises for a source."""
    from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
    names = [s for s in (sources or FOCUS.get(focus, FOCUS["web"])) if s in SOURCES]
    if os.environ.get("LLM_SEARCH_URL") and "duckduckgo" in names:
        names = ["duckduckgo"] + [s for s in names if s != "duckduckgo"]   # the user's own SearxNG stands in
    t0, hits, errors = time.time(), [], {}
    jobs = [(q, s) for q in queries for s in names if not _resting(s)]
    for s in names:
        if _resting(s):
            errors[s] = "resting after recent failures"
    ex = ThreadPoolExecutor(max_workers=max(1, min(max_workers, len(jobs) or 1)), thread_name_prefix="web-search")
    try:
        pending = {ex.submit(SOURCES[s], q, n, None, per_source_timeout): (q, s) for q, s in jobs}
        while pending:
            left = deadline - (time.time() - t0)
            if left <= 0 or (should_stop and should_stop()):
                for f, (q, s) in pending.items():
                    errors.setdefault(s, "no answer before the deadline")
                    f.cancel()
                break
            done, _ = wait(list(pending), timeout=min(left, 0.25), return_when=FIRST_COMPLETED)
            for f in done:
                q, s = pending.pop(f)
                try:
                    for i, r in enumerate(f.result() or []):
                        if r.get("url", "").startswith("http") and r.get("title"):
                            hits.append((q, s, i + 1, r))
                except Exception as e:                       # one source failing is one missing list
                    errors[s] = str(e)[:160]
                    if isinstance(e, (urllib.error.URLError, OSError, ValueError, RuntimeError)):
                        _rest(s, e)
    finally:
        ex.shutdown(wait=False, cancel_futures=True)
    ranked = diversify(rank(merge(hits), question or " ".join(queries[:1]), fresh=fresh))
    return {"results": ranked, "errors": errors, "requests": len(jobs), "seconds": round(time.time() - t0, 2),
            "sources": names}


def format_entries(entries, limit=10):
    if not entries:
        return "(no results)"
    return "\n\n".join(f"{i}. {e['title']}\n   {e['url']}\n   {e['snippet'][:300]}\n   (via {', '.join(e['sources'])})"
                       for i, e in enumerate(entries[:limit], 1))


# ───────────────────────────── tool definitions ─────────────────────────────
def tools(allow=lambda what: True, allow_private=False, model=None, progress=None, should_stop=None):
    """{name: (OpenAI tool schema, callable(**args) -> str)}. `allow(what)` is asked before anything
    leaves the machine; returning False makes the tool report that the user declined.

    `model(messages, max_tokens) -> text` is the already loaded model, used by web_research to plan queries and
    judge gaps (optional); `progress(stage, text)` receives its activity notes; `should_stop()` cancels it."""
    def web_search(query, max_results=5, focus=None):
        if not allow(f"search the web for: {query}"):
            return "user declined web access"
        n = max(1, min(int(max_results), 10))
        if focus == "auto" or (focus in FOCUS and focus != "web"):
            f = detect_focus(query) if focus == "auto" else focus
            res = multi_search([query], focus=f, n=n, question=query, should_stop=should_stop)
            errs = ("\n(unavailable: " + ", ".join(res["errors"]) + ")") if res["errors"] else ""
            return format_entries(res["results"], n) + errs
        try:
            return format_results(search(query, n, allow_private=allow_private))
        except RuntimeError as e:
            return f"error: {e}"

    def web_fetch(url, max_chars=8000):
        if not allow(f"fetch {url}"):
            return "user declined web access"
        return fetch(url, max(500, min(int(max_chars), 20000)), allow_private=allow_private)

    def web_research(question, mode="smart", focus="auto"):
        if not allow(f"research on the web: {question}"):
            return "user declined web access"
        from harness import research
        result = research.run(question, mode=mode if mode in research.MODES else "smart", focus=focus or "auto",
                              model=model, progress=progress, should_stop=should_stop, allow_private=allow_private)
        return research.report(result)

    focus_prop = {"type": "string", "enum": ["auto", "web", "docs", "academic", "news", "discussions"],
                  "description": "optional: which kind of sources (several at once); omit for the plain web search"}
    return {
        "web_search": ({"type": "function", "function": {
            "name": "web_search",
            "description": "Search the internet. Use it for current facts, library/API changes and error messages "
                           "you do not know. Returns titles, URLs and snippets; follow up with web_fetch.",
            "parameters": {"type": "object", "properties": {
                "query": {"type": "string"},
                "max_results": {"type": "integer", "description": "1-10, default 5"},
                "focus": focus_prop}, "required": ["query"]}}},
            web_search),
        "web_fetch": ({"type": "function", "function": {
            "name": "web_fetch",
            "description": "Download a web page and return it as clean text (navigation and scripts removed).",
            "parameters": {"type": "object", "properties": {
                "url": {"type": "string"},
                "max_chars": {"type": "integer", "description": "default 8000, max 20000"}}, "required": ["url"]}}},
            web_fetch),
        "web_research": ({"type": "function", "function": {
            "name": "web_research",
            "description": "Research a question on the web in one step: several queries over suitable sources, the "
                           "best pages read, evidence returned as numbered sources [S1], [S2]… to cite. Use it for "
                           "questions that need several sources, comparisons or a careful check; web_search for a "
                           "quick lookup. mode: quick (a fact), smart (default), deep (a thorough investigation).",
            "parameters": {"type": "object", "properties": {
                "question": {"type": "string", "description": "the question, in the user's words"},
                "mode": {"type": "string", "enum": ["quick", "smart", "deep"]},
                "focus": focus_prop}, "required": ["question"]}}},
            web_research),
    }
