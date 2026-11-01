#!/usr/bin/env python3
"""Normalized README tiles from the screenshots.

The screenshots are different shapes, so side by side in the README they read as a pile rather than a set.
This puts every one of them on the same canvas at the same margin, so the row looks like one set, and trims
the dead field each one carries around its window. The canvas stays transparent: the window brings its own
shadow and the page shows through, in either GitHub theme.

    .venv/bin/python tools/make-tiles.py

Run it after regenerating any screenshot; it only reads docs/assets and writes docs/assets/tile-*.png.
"""
import os
import sys

from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CANVAS = (1200, 900)              # 4:3, the shape the README grid is laid out in
MARGIN = 44
BACKGROUND = (0, 0, 0, 0)         # transparent: the window keeps its own shadow, the page shows through

TILES = [
    ("screenshot-engine.png", "tile-engine.png"),
    ("screenshot-editor-ide.png", "tile-editor.png"),
    ("screenshot-voice.png", "tile-voice.png"),
    ("screenshot-models.png", "tile-models.png"),
]


FLOOR = 80                        # alpha above this is the window, below it is the shadow fading out
HALO = 0.03                       # of the window's longest side: how much of that shadow to keep


def trim(im):
    """Finds the window inside a screenshot and keeps a little of its shadow.

    The screenshots carry a soft shadow that fades almost to nothing but still covers the whole canvas, so
    trimming at "any alpha at all" keeps the entire dead field around the window — which is what made the
    tiles look like small pictures swimming in black. This looks for pixels that really belong to the
    window, then grows the box by a fraction of its size so the shadow is not cut off flat.
    """
    a = im.getchannel("A") if im.mode == "RGBA" else im.convert("L")
    box = a.point(lambda v: 255 if v > FLOOR else 0).getbbox()
    if not box:
        return im
    pad = round(HALO * max(box[2] - box[0], box[3] - box[1]))
    return im.crop((max(0, box[0] - pad), max(0, box[1] - pad),
                    min(im.width, box[2] + pad), min(im.height, box[3] + pad)))


def tile(src, dst):
    im = trim(Image.open(src).convert("RGBA"))
    room = (CANVAS[0] - 2 * MARGIN, CANVAS[1] - 2 * MARGIN)
    scale = min(room[0] / im.width, room[1] / im.height)
    im = im.resize((max(1, round(im.width * scale)), max(1, round(im.height * scale))), Image.LANCZOS)
    out = Image.new("RGBA", CANVAS, BACKGROUND)
    out.paste(im, ((CANVAS[0] - im.width) // 2, (CANVAS[1] - im.height) // 2), im)
    out.save(dst)
    return im.size


def main():
    assets = os.path.join(ROOT, "docs", "assets")
    for src, dst in TILES:
        s = os.path.join(assets, src)
        if not os.path.exists(s):
            print(f"missing: {src}", file=sys.stderr)
            return 1
        w, h = tile(s, os.path.join(assets, dst))
        print(f"{dst}: {CANVAS[0]}x{CANVAS[1]} (content {w}x{h})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
