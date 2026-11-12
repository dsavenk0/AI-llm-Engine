"""A tiny, valid multi-page PDF writer for tests — standard Helvetica text, no external libraries or samples.

    make_pdf(["Page one text", None, "Page three"], rotate={2: 90})   → bytes

A None page has no content stream: a page with no text layer, which is what a scan looks like to an extractor.
"""


def _esc(s):
    return s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def make_pdf(pages, rotate=None):
    rotate = rotate or {}
    objs = {1: b"<< /Type /Catalog /Pages 2 0 R >>", 3: b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"}
    kids, n = [], 4
    for i, text in enumerate(pages):
        page_id, content_id = n, n + 1
        n += 2
        kids.append(page_id)
        rot = f" /Rotate {rotate[i]}" if i in rotate else ""
        if text is None:
            stream = b""
        else:
            lines = text.split("\n")
            ops = ["BT /F1 14 Tf 72 720 Td 18 TL"] + [f"({_esc(line)}) Tj T*" for line in lines] + ["ET"]
            stream = " ".join(ops).encode("latin-1")
        objs[page_id] = (f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]{rot} "
                         f"/Resources << /Font << /F1 3 0 R >> >> /Contents {content_id} 0 R >>").encode()
        objs[content_id] = b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream"
    objs[2] = f"<< /Type /Pages /Kids [{' '.join(f'{k} 0 R' for k in kids)}] /Count {len(kids)} >>".encode()
    out = bytearray(b"%PDF-1.4\n")
    offsets = {}
    for oid in sorted(objs):
        offsets[oid] = len(out)
        out += b"%d 0 obj\n" % oid + objs[oid] + b"\nendobj\n"
    xref = len(out)
    size = max(objs) + 1
    out += b"xref\n0 %d\n0000000000 65535 f \n" % size
    for oid in range(1, size):
        out += b"%010d 00000 n \n" % offsets[oid]
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (size, xref)
    return bytes(out)
