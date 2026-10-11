"""Project memory: standing instructions the agent should always have in front of it.

Read (first match wins) from the workspace:  .llm/memory.md  →  AGENTS.md  →  CLAUDE.md
`/remember <text>` appends a line to .llm/memory.md (created on demand, never elsewhere).
"""
import os

CANDIDATES = (".llm/memory.md", "AGENTS.md", "CLAUDE.md")
MAX_CHARS = 6000          # a standing prompt must stay small; the tail of a long file is cut
WRITABLE = ".llm/memory.md"


def find(root):
    for rel in CANDIDATES:
        p = os.path.join(root, rel)
        if os.path.isfile(p):
            return p
    return None


def load(root):
    """(relative_path, text) of the project's instructions, or (None, "")."""
    p = find(root)
    if not p:
        return None, ""
    try:
        with open(p, encoding="utf-8", errors="replace") as f:
            text = f.read().strip()
    except OSError:
        return None, ""
    if len(text) > MAX_CHARS:
        text = text[:MAX_CHARS].rsplit("\n", 1)[0] + "\n… (truncated)"
    return os.path.relpath(p, root), text


def remember(root, note):
    """Append one note to .llm/memory.md; returns the path written."""
    note = " ".join(note.split())
    if not note:
        raise ValueError("nothing to remember")
    p = os.path.join(root, WRITABLE)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(f"- {note}\n")
    return p


def system_block(root):
    rel, text = load(root)
    if not text:
        return ""
    return f"\n\nProject instructions (from {rel}) — follow them:\n{text}"


# ───────────────────────── session notes (written by the agent itself) ─────────────────────────
SESSION = ".llm/session.md"
SESSION_MAX = 3000


def load_session(root):
    p = os.path.join(root, SESSION)
    try:
        with open(p, encoding="utf-8", errors="replace") as f:
            text = f.read().strip()
    except OSError:
        return ""
    return text[:SESSION_MAX].rsplit("\n", 1)[0] + "\n…" if len(text) > SESSION_MAX else text


def save_session(root, text):
    text = text.strip()
    if not text:
        return None
    p = os.path.join(root, SESSION)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    os.replace(tmp, p)
    return p


def session_block(root):
    text = load_session(root)
    if not text:
        return ""
    return ("\n\nNotes you saved at the end of the previous session (they may be outdated — verify "
            "against the files before relying on them):\n" + text)
