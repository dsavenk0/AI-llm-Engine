#!/usr/bin/env python3
"""The README's screenshot carousel: the tiles from make-tiles.py as one animated WebP that shows each screenshot
for a few seconds and cross-fades to the next.

GitHub strips scripts and styles from a README, so a carousel with buttons is impossible there; an animated image
is the one moving thing it shows. The tiles stay in the README too (under "All screenshots"), for a close look.

    .venv/bin/python tools/make-carousel.py

Run it after make-tiles.py; it reads docs/assets/tile-*.png and writes docs/assets/carousel.webp.
"""
import os

from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(ROOT, "docs", "assets")
ORDER = ["tile-engine.png", "tile-browser.png", "tile-editor.png", "tile-voice.png", "tile-models.png"]
SIZE = (960, 720)          # the tiles' 4:3, small enough to keep the file light
HOLD_MS = 2800             # how long each screenshot stays
FADE_STEPS = 8             # frames in a cross-fade
FADE_MS = 45               # per fade frame


def main():
    tiles = [Image.open(os.path.join(ASSETS, name)).convert("RGBA").resize(SIZE, Image.LANCZOS)
             for name in ORDER if os.path.exists(os.path.join(ASSETS, name))]
    if len(tiles) < 2:
        raise SystemExit("need at least two tiles in docs/assets (run tools/make-tiles.py first)")
    frames, durations = [], []
    for i, tile in enumerate(tiles):
        frames.append(tile)
        durations.append(HOLD_MS)
        nxt = tiles[(i + 1) % len(tiles)]
        for k in range(1, FADE_STEPS + 1):
            frames.append(Image.blend(tile, nxt, k / (FADE_STEPS + 1)))
            durations.append(FADE_MS)
    out = os.path.join(ASSETS, "carousel.webp")
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=durations, loop=0,
                   quality=82, method=6, lossless=False)
    print(f"carousel.webp: {len(tiles)} screenshots, {len(frames)} frames, {os.path.getsize(out) // 1024} KB")


if __name__ == "__main__":
    main()
