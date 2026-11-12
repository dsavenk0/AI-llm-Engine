"""Lumo's greeting as it looks in the terminal, as SVG for the READMEs (English and Russian).

    python3 tools/make-lumo.py            → docs/assets/lumo-en.svg, docs/assets/lumo-ru.svg

The pixels, colours and words come from llm.py (`_MASCOT`, `_MASCOT_COLORS`, the banner strings) and
locales/ru.json, so a change to Lumo is one re-run away. Stdlib only.
"""
import html
import json
import os
import sys
import textwrap

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import llm  # noqa: E402

PX = 14                       # one mascot pixel
FONT = 19                     # monospace text size
CHAR = FONT * 0.6             # monospace advance
LINE = 29
WRAP = 46                     # the tip, wrapped like the terminal's bubble
TIP = "./llm perf — measure this Mac against the memory bus it is limited by"

BG, PANEL, CYAN = "#151b22", "#1b232c", "#00e5ff"
TEXT, DIM, MINT, TIPC = "#eaf4f7", "#7f8d95", "#4df2c2", "#39c6d6"


def rgb(c):
    return "#%02x%02x%02x" % c


def card(lang):
    tr = {}
    if lang == "ru":
        with open(os.path.join(ROOT, "locales", "ru.json"), encoding="utf-8") as f:
            tr = json.load(f)
    t = lambda s: tr.get(s, s)                                         # noqa: E731
    lines = [[(t("Good evening") + "! ", TEXT, True), (t("I'm Lumo."), TEXT, False)],
             [("Ai>_ ", MINT, True), ("AI-llm-Engine", TEXT, True)],
             [(t("Your models, your Mac, no cloud."), DIM, False)],
             []]
    tip_label = t("Tip:")
    for k, w in enumerate(textwrap.wrap(tip_label + " " + t(TIP), WRAP)):
        lines.append([(tip_label, TIPC, False), (w[len(tip_label):], TEXT, False)] if k == 0 else [(w, TEXT, False)])

    grid = llm._MASCOT
    mw, mh = len(grid[0]) * PX, len(grid) * PX
    inner = max(sum(len(s) for s, _, _ in ln) for ln in lines if ln) * CHAR
    bx, by = 24 + mw + 34, 24 + 2 * PX
    bw, bh = inner + 2 * 22, len(lines) * LINE + 2 * 18
    width, height = int(bx + bw + 24), int(max(24 + mh, by + bh) + 24)

    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
           f'role="img" aria-label="Lumo, the AI-llm-Engine mascot, greeting you in the terminal">',
           f'<rect width="{width}" height="{height}" rx="14" fill="{BG}"/>',
           '<g shape-rendering="crispEdges">']
    for y, row in enumerate(grid):
        for x, ch in enumerate(row):
            c = llm._MASCOT_COLORS.get(ch)
            if c:
                out.append(f'<rect x="{24 + x * PX}" y="{24 + y * PX}" width="{PX}" height="{PX}" fill="{rgb(c)}"/>')
    out.append("</g>")
    # the bubble, pointing at Lumo's eyes from its second line
    tail_y = by + 18 + LINE * 1.5
    out.append(f'<rect x="{bx}" y="{by}" width="{bw:.0f}" height="{bh}" rx="10" fill="{PANEL}" '
               f'stroke="{CYAN}" stroke-width="2"/>')
    out.append(f'<path d="M{bx - 12} {tail_y} L{bx + 1} {tail_y - 9} L{bx + 1} {tail_y + 9} Z" fill="{CYAN}"/>')
    out.append(f'<g font-family="SF Mono, Menlo, Consolas, monospace" font-size="{FONT}" xml:space="preserve">')
    for i, ln in enumerate(lines):
        if not ln:
            continue
        bold = ' font-weight="700"'
        spans = "".join(f'<tspan fill="{c}"{bold if b else ""}>{html.escape(s)}</tspan>' for s, c, b in ln)
        out.append(f'<text x="{bx + 22}" y="{by + 18 + LINE * i + FONT}">{spans}</text>')
    out.append("</g></svg>")
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    for lang in ("en", "ru"):
        path = os.path.join(ROOT, "docs", "assets", f"lumo-{lang}.svg")
        with open(path, "w", encoding="utf-8") as f:
            f.write(card(lang))
        print(path)
