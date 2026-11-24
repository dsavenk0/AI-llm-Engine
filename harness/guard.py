"""Keeping an agent turn from running away: a reply that loops on itself, a tool called the same way again and again.

A small local model can fall into a loop — the same line written over and over, or the same tool call repeated —
and without a limit that looks like a hung agent. These checks are cheap, look only at the tail of what was
produced, and stay conservative: real output (a table, a list of similar files) must not be cut off.
"""
import json

LINE_REPEATS = 5          # the same non-trivial line this many times among the last lines written
TAIL = 160                # characters of the reply's end that must recur …
TAIL_REPEATS = 4          # … this many times in the last WINDOW characters
WINDOW = 4000
SAME_CALLS = 3            # the same tool call (name and arguments) this many times in a row


def looping(text):
    """True when the end of `text` repeats itself: one line many times among the recent ones, or the last
    TAIL characters several times over in the recent window."""
    if not text or len(text) < TAIL * TAIL_REPEATS:
        return False
    recent = text[-WINDOW:]
    lines = [l.strip() for l in recent.splitlines() if len(l.strip()) > 12]
    last = lines[-(LINE_REPEATS * 3):]
    if last and any(last.count(l) >= LINE_REPEATS for l in set(last)):
        return True
    tail = recent[-TAIL:]
    return len(tail.strip()) > 40 and recent.count(tail) >= TAIL_REPEATS


def call_key(name, args):
    """A comparable form of one tool call."""
    try:
        return name + json.dumps(args, sort_keys=True, ensure_ascii=False)
    except (TypeError, ValueError):
        return name + repr(args)


def same_calls(history, key):
    """True when `key` would be the SAME_CALLS-th identical call in a row (history: the keys so far, oldest first)."""
    return len(history) >= SAME_CALLS - 1 and all(k == key for k in history[-(SAME_CALLS - 1):])
