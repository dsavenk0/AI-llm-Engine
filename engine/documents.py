# Document text extraction — stdlib only, no new dependencies.
#
# Turns a file the user attaches into plain text that can be sent to the model as one source:
#   extract_text(path) -> {"text", "kind", "pages", "truncated", "note"}
#
# Office files (.docx/.pptx) are zipped XML, read with zipfile + a small regex. PDFs are parsed
# best-effort: stream objects are inflated (FlateDecode via zlib) and the text operators (Tj/TJ) are
# pulled out — enough for a text PDF, and a scanned one (no text layer) is reported so OCR can be
# offered instead of sending an empty source. Plain text / markup is read directly.
import html
import math
import os
import re
import subprocess
import zipfile
import zlib

MAX_CHARS = 400_000          # a long document is truncated to this (≈100K tokens); `truncated` says so

TEXT_EXTS = (".txt", ".text", ".log", ".md", ".markdown", ".rst", ".csv", ".tsv", ".json", ".jsonl",
             ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf")
MARKUP_EXTS = (".html", ".htm", ".xml", ".svg")
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".gif", ".heic", ".heif", ".webp")


class DocError(Exception):
    pass


def ocr_tool():
    """The local Apple-Vision OCR helper, if it was built (env LLM_OCR, else bin/ocr next to the code).
    None means scanned PDFs and images cannot be read here — build it with `./llm setup --ocr`."""
    p = os.environ.get("LLM_OCR")
    if p and os.path.isfile(p):
        return p
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))     # repo root / Resources/llm
    for cand in (os.path.join(here, "bin", "ocr"), os.path.join(here, "ocr")):
        if os.path.isfile(cand) and os.access(cand, os.X_OK):
            return cand
    return None


def _run_ocr(path):
    tool = ocr_tool()
    if not tool:
        return ""
    try:
        return subprocess.run([tool, path], capture_output=True, text=True, timeout=180).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def _clip(text):
    text = text.strip()
    if len(text) > MAX_CHARS:
        return text[:MAX_CHARS], True
    return text, False


def _result(text, kind, pages=0, note=""):
    text, truncated = _clip(text)
    return {"text": text, "kind": kind, "pages": pages, "truncated": truncated, "note": note}


# ───────────────────────────── plain text / markup ─────────────────────────────
def _read_text(data):
    return data.decode("utf-8", "replace")


_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"[ \t]*\n(?:[ \t]*\n)+")


def _strip_markup(data):
    s = data.decode("utf-8", "replace")
    s = re.sub(r"(?is)<(script|style)\b.*?</\1>", " ", s)   # drop scripts/styles whole
    s = re.sub(r"(?i)<(br|/p|/div|/h[1-6]|/li|/tr)\b[^>]*>", "\n", s)  # block ends → newlines
    s = _TAG.sub("", s)
    s = html.unescape(s)
    return _WS.sub("\n\n", s)


# ───────────────────────────── office (zipped XML) ─────────────────────────────
def _office_text(data, tag, part_filter):
    """Join the text in every <tag>…</tag> of the matching parts of a zipped Office file, one line per
    paragraph. `part_filter` picks the XML parts to read (the document body, or every slide)."""
    rx = re.compile(r"<%s\b[^>]*>(.*?)</%s>" % (tag, tag), re.S)
    try:
        with zipfile.ZipFile(data if hasattr(data, "read") else _BytesIO(data)) as z:
            parts = sorted(n for n in z.namelist() if part_filter(n))
            if not parts:
                raise DocError("no document body in the archive")
            out = []
            for name in parts:
                xml = z.read(name).decode("utf-8", "replace")
                xml = re.sub(r"<w:br/>|<a:br/>", "</w:t><w:t>\n</w:t><w:t>", xml)  # line break inside a run
                # a paragraph's runs join together; paragraphs become separate lines
                for para in re.split(r"</w:p>|</a:p>", xml):
                    runs = "".join(html.unescape(m.group(1)) for m in rx.finditer(para))
                    if runs:
                        out.append(runs)
            return "\n".join(out)
    except zipfile.BadZipFile:
        raise DocError("not a valid Office file (bad zip)")


def _BytesIO(data):
    import io
    return io.BytesIO(data)


def _docx(data):
    return _result(_office_text(data, "w:t", lambda n: n == "word/document.xml"), "docx")


def _pptx(data):
    text = _office_text(data, "a:t", lambda n: n.startswith("ppt/slides/slide") and n.endswith(".xml"))
    pages = text.count("\n\n") + 1 if text else 0
    return _result(text, "pptx", pages=pages)


# ───────────────────────────── pdf (best-effort) ─────────────────────────────
_STREAM = re.compile(rb"stream\r?\n(.*?)\r?\nendstream", re.S)
_PDF_STR = re.compile(rb"\((?:\\.|[^\\()])*\)", re.S)


def _pdf_unescape(s):
    out, i = [], 0
    s = s[1:-1]  # strip the surrounding ()
    repl = {b"n": "\n", b"r": "\r", b"t": "\t", b"b": "\b", b"f": "\f", b"(": "(", b")": ")", b"\\": "\\"}
    while i < len(s):
        c = s[i:i + 1]
        if c == b"\\" and i + 1 < len(s):
            nxt = s[i + 1:i + 2]
            if nxt in repl:
                out.append(repl[nxt]); i += 2; continue
            if nxt.isdigit():                              # octal \ddd
                oct_ = s[i + 1:i + 4]
                m = re.match(rb"[0-7]{1,3}", oct_)
                if m:
                    out.append(chr(int(m.group(), 8))); i += 1 + len(m.group()); continue
            i += 1; continue
        out.append(c.decode("latin-1", "replace")); i += 1
    return "".join(out)


def _pdf(data):
    pages = len(re.findall(rb"/Type\s*/Page[^s]", data)) or data.count(b"/Page")
    chunks = []
    for m in _STREAM.finditer(data):
        raw = m.group(1)
        try:
            content = zlib.decompress(raw)
        except zlib.error:
            content = raw                                   # uncompressed stream
        if b"Tj" not in content and b"TJ" not in content and b"BT" not in content:
            continue
        for sm in _PDF_STR.finditer(content):
            piece = _pdf_unescape(sm.group())
            if piece:
                chunks.append(piece)
    text = " ".join(chunks)
    text = re.sub(r"[ \t]{2,}", " ", text).strip()
    note = ""
    if not text:
        note = "no text layer found — this PDF looks scanned; OCR would be needed to read it"
    return _result(text, "pdf", pages=pages, note=note)


# ───────────────────────────── dispatch ─────────────────────────────
def extract_text(path):
    """Extract plain text from a document. Returns a dict (see the module docstring); raises DocError when
    the file cannot be read as any supported kind."""
    if not os.path.isfile(path):
        raise DocError(f"no such file: {path}")
    ext = os.path.splitext(path)[1].lower()
    with open(path, "rb") as f:
        data = f.read()
    if not data:
        return _result("", "text")
    if ext == ".pdf" or data[:5] == b"%PDF-":
        r = _pdf(data)
        if not r["text"] and ocr_tool():                  # scanned PDF: recover the text with Vision OCR
            text = _run_ocr(path)
            if text:
                return _result(text, "pdf-ocr", pages=r["pages"], note="text recovered by OCR")
        return r
    if ext in IMAGE_EXTS or data[:4] in (b"\x89PNG", b"GIF8") or data[:3] == b"\xff\xd8\xff":
        if not ocr_tool():
            raise DocError("reading an image needs the Vision OCR helper — build it with ./llm setup --ocr")
        text = _run_ocr(path)
        return _result(text, "image", note="" if text else "no text found in the image")
    if ext == ".docx":
        return _docx(data)
    if ext == ".pptx":
        return _pptx(data)
    if ext in MARKUP_EXTS:
        return _result(_strip_markup(data), "markup")
    if ext in TEXT_EXTS:
        return _result(_read_text(data), "text")
    # unknown extension: accept it if it is decodable text, else refuse (a binary we cannot read)
    sample = data[:4096]
    if b"\x00" in sample:
        if data[:2] == b"PK":
            raise DocError("looks like an Office/zip file with an unsupported layout")
        raise DocError(f"cannot read {os.path.basename(path)} as text (binary file)")
    return _result(_read_text(data), "text")


# ───────────────────────────── excerpts for long documents ─────────────────────────────
_STOP = frozenset("a an the and or of to in on for is are be was were this that it its with as at by from "
                  "not but if then else what which who how why when where".split())
_TOK = re.compile(r"[A-Za-z0-9_]+")


def _passages(text, size):
    """Split text into ~size-char passages, snapping each end back to a nearby paragraph/line break so a
    passage is whole sentences, not a mid-word cut. Returns (start, end, text) with real char offsets."""
    out, i, n = [], 0, len(text)
    while i < n:
        end = min(i + size, n)
        if end < n:
            window = text[max(i + size // 2, end - 300):end]
            br = window.rfind("\n\n")
            if br == -1:
                br = window.rfind("\n")
            if br == -1:
                br = window.rfind(". ")
            if br != -1:
                end = max(i + size // 2, end - 300) + br + 1
        out.append((i, end, text[i:end]))
        if end <= i:
            break
        i = end
    return out


def select_excerpts(text, query, budget=MAX_CHARS, size=1400, max_passages=12):
    """The passages of `text` most relevant to `query`, in document order, within a character budget — so a
    long document contributes the parts that bear on the question instead of just its opening. Lexical
    scoring (TF·IDF over passages, length-normalised); no model, no embedding, no dependency.

    Returns {"text", "passages": [{start, end, score}], "selected", "total"}. With no query or no match it
    falls back to the head of the document (same as a plain truncation)."""
    terms = {t for t in (w.lower() for w in _TOK.findall(query or "")) if len(t) >= 2 and t not in _STOP}
    passes = _passages(text, size)
    if not terms or not passes:
        return {"text": text[:budget], "passages": [], "selected": 0, "total": len(passes)}
    toks_per = [[w.lower() for w in _TOK.findall(p)] for _, _, p in passes]
    df = {t: sum(1 for toks in toks_per if t in set(toks)) for t in terms}
    n = len(passes)
    idf = {t: math.log(1 + n / (1 + df[t])) for t in terms}
    scored = []
    for idx, toks in enumerate(toks_per):
        if not toks:
            continue
        tf = {}
        for w in toks:
            if w in terms:
                tf[w] = tf.get(w, 0) + 1
        score = sum(c * idf[t] for t, c in tf.items()) / (1 + len(toks) / 200)
        if score > 0:
            scored.append((score, idx))
    if not scored:
        return {"text": text[:budget], "passages": [], "selected": 0, "total": n}
    scored.sort(reverse=True)
    chosen, used = [], 0
    for score, idx in scored:
        s, e, p = passes[idx]
        if chosen and used + len(p) > budget:
            break
        chosen.append((idx, s, e, p.strip(), score)); used += len(p)
        if len(chosen) >= max_passages:
            break
    chosen.sort(key=lambda x: x[0])                    # back into document order
    pieces, passages = [], []
    for k, (idx, s, e, p, score) in enumerate(chosen):
        if k and chosen[k - 1][0] != idx - 1:          # a gap between passages
            pieces.append("[…]")
        pieces.append(p)
        passages.append({"start": s, "end": e, "score": round(score, 3)})
    return {"text": "\n\n".join(pieces), "passages": passages, "selected": len(chosen), "total": n}


SUPPORTED = ("pdf", "docx", "pptx", "html/xml", "txt/md/csv/json/yaml/…")
