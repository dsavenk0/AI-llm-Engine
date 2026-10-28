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
import os
import re
import zipfile
import zlib

MAX_CHARS = 400_000          # a long document is truncated to this (≈100K tokens); `truncated` says so

TEXT_EXTS = (".txt", ".text", ".log", ".md", ".markdown", ".rst", ".csv", ".tsv", ".json", ".jsonl",
             ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf")
MARKUP_EXTS = (".html", ".htm", ".xml", ".svg")


class DocError(Exception):
    pass


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
        return _pdf(data)
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


SUPPORTED = ("pdf", "docx", "pptx", "html/xml", "txt/md/csv/json/yaml/…")
