"""Pictures in words, for a model that cannot see: Apple Vision (the local OCR helper's `--describe` mode) reads
the text in each picture and names what it shows, and the prompt carries that instead of the pixels. It works
with any model — text-only MLX, GGUF — and nothing leaves the Mac.

It is a description, not sight: scene labels ("screenshot", "food", "plate") and the recognised text. Enough for
screenshots, charts and slides with words on them; rough for photographs. The prompt says so.
"""
import base64
import json
import os
import subprocess
import tempfile

from engine.documents import ocr_tool

MAX_TEXT = 4000           # characters of recognised text per picture


def describe(data_urls, timeout=60):
    """One {"text", "labels"} per picture (empty ones when the helper is missing or fails)."""
    tool = ocr_tool()
    if not tool or not data_urls:
        return [{"text": "", "labels": []} for _ in data_urls]
    with tempfile.TemporaryDirectory() as tmp:
        paths = []
        for n, url in enumerate(data_urls):
            head, _, b64 = url.partition(",")
            ext = "png" if "png" in head else "webp" if "webp" in head else "jpg"
            p = os.path.join(tmp, f"{n}.{ext}")
            with open(p, "wb") as f:
                f.write(base64.b64decode(b64))
            paths.append(p)
        try:
            out = subprocess.run([tool, "--describe", *paths], capture_output=True, text=True, timeout=timeout).stdout
        except (OSError, subprocess.SubprocessError):
            out = ""
    found = []
    for line in out.splitlines():
        try:
            d = json.loads(line)
        except ValueError:
            continue
        found.append({"text": str(d.get("text") or "")[:MAX_TEXT], "labels": [str(x) for x in d.get("labels") or []][:8]})
    found += [{"text": "", "labels": []}] * (len(data_urls) - len(found))
    return found[:len(data_urls)]


def as_text(descriptions, has_screenshot):
    """The descriptions as prompt text, naming each picture."""
    parts = []
    for n, d in enumerate(descriptions):
        name = "The screenshot of the page" if (n == 0 and has_screenshot) else f"Picture {n + (0 if has_screenshot else 1)} from the page"
        bits = []
        if d["labels"]:
            bits.append("shows: " + ", ".join(d["labels"]))
        if d["text"].strip():
            bits.append("text in it:\n" + d["text"].strip())
        parts.append(f"{name} — " + ("; ".join(bits) if bits else "nothing recognisable"))
    return ("You cannot see pictures; they were described for you by the Mac's image recognition (scene labels "
            "and the text it read), which is rough for photographs. Use the descriptions, and say so when they "
            "are not enough.\n" + "\n\n".join(parts))
