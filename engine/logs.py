"""Engine log with levels and groups (stdlib only).

Every line carries a group tag — `[request]`, `[memory]`, `[model]`, `[cache]`, `[sleep]`, `[quality]`,
`[error]` — so a long log can be filtered with grep, and a level decides what is written at all:
error < warning < info (default) < debug < trace. Call sites keep passing plain strings: the level and the
group are derived from how the message starts, which keeps the (many) existing log() calls untouched.
"""
import re
import time

LEVELS = ("error", "warning", "info", "debug", "trace")

# (pattern, level, group); first match wins
RULES = [
    (r"^(generation error|prompt encoding error|prefix cache: save failed)", "error", "error"),
    (r"^trace:", "trace", "request"),
    (r"^memory: (not enough|system memory)", "warning", "memory"),
    (r"^memory: Metal's recommended", "debug", "memory"),
    (r"^memory:", "warning", "memory"),
    (r"^repetition:", "warning", "quality"),
    (r"^tool call not parsed", "warning", "quality"),
    (r"^(PTQ1 model|warning)", "warning", "model"),
    (r"^(sleep|wake):", "info", "sleep"),
    (r"^prefix cache:", "info", "cache"),
    (r"^\d+ prompt ", "info", "request"),
    (r"^(sampling|disk cache|context): ", "debug", "model"),
    (r"^vision tower", "debug", "model"),
    (r"^(loaded|serving|backend|mlx-lm backend)", "info", "model"),
]
_COMPILED = [(re.compile(p), lvl, grp) for p, lvl, grp in RULES]


def classify(msg):
    """(level, group) for a log message."""
    for rx, level, group in _COMPILED:
        if rx.search(msg):
            return level, group
    return "info", "engine"


def _print(line):
    print(line, flush=True)


def make(level="info", sink=_print):
    """A log function writing `[time] [group] message` for messages at or above `level`."""
    limit = LEVELS.index(level) if level in LEVELS else LEVELS.index("info")

    def log(msg, level=None, group=None):
        lvl, grp = classify(msg)
        lvl, grp = level or lvl, group or grp
        if LEVELS.index(lvl) <= limit:
            sink(f"[{time.strftime('%H:%M:%S')}] [{grp}] {msg}")
    return log
