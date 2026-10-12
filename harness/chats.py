"""Chat history shared by the terminal chat and the code editor.

The editor keeps one JSON file per project in ~/Library/Application Support/LLMBar/editor-chats
(an array of sessions: id, title, updated, items). `llm chat` reads and writes the same file, so a
conversation started in either can be opened, renamed, deleted and continued from the other.

The files are the editor's Swift `Codable` output: UUIDs are upper-case strings, `updated` is seconds
since 2001-01-01, and every field of a session and of an item is required by its decoder — so what
is written here always carries all of them (and keeps the items it does not understand untouched).
"""
import json
import os
import time
import uuid

KEEP = 40                       # the editor keeps the 40 most recent chats of a project
TITLE_CHARS = 48
REFERENCE_DATE = 978307200      # 2001-01-01 in Unix seconds (Swift's Date encoding)
COUNTS = ("searches", "explores", "reads", "edits", "commands", "questions", "webs", "images")


def default_dir():
    return os.path.expanduser("~/Library/Application Support/LLMBar/editor-chats")


def project_key(root):
    """The editor's file name for a project: <folder name>-<djb2 of the path, hex>.json"""
    h = 5381
    for b in root.encode("utf-8"):
        h = (h * 33 + b) & 0xFFFFFFFFFFFFFFFF
    return f"{os.path.basename(root)}-{h:x}.json"


def chats_file(root, directory=None):
    return os.path.join(directory or default_dir(), project_key(os.path.abspath(root)))


def _now():
    return time.time() - REFERENCE_DATE


def load(root, directory=None):
    """The project's sessions, newest first. A missing or damaged file is an empty history."""
    try:
        with open(chats_file(root, directory), encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return []
    if not isinstance(data, list):
        return []
    out = [s for s in data if isinstance(s, dict) and isinstance(s.get("id"), str)
           and isinstance(s.get("items", []), list)]
    for s in out:
        s.setdefault("items", [])
        s["items"] = [i for i in s["items"] if isinstance(i, dict)]
        if not isinstance(s.get("title"), str):
            s["title"] = "New chat"
        if not isinstance(s.get("updated"), (int, float)):
            s["updated"] = 0
    out.sort(key=lambda s: s["updated"], reverse=True)
    return out


def save(root, sessions, directory=None):
    """Writes the newest KEEP chats (atomically). Empty chats are not stored."""
    path = chats_file(root, directory)
    keep = [s for s in sorted(sessions, key=lambda s: s["updated"], reverse=True) if s["items"]][:KEEP]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(keep, f, ensure_ascii=False)
    os.replace(tmp, path)


def new_session():
    return {"id": str(uuid.uuid4()).upper(), "title": "New chat", "updated": _now(), "items": []}


def title_for(text):
    t = " ".join(str(text).split())
    if len(t) <= TITLE_CHARS:
        return t or "New chat"
    cut = t[:TITLE_CHARS]
    return (cut.rsplit(" ", 1)[0] if " " in cut else cut) + " …"       # whole words, like the Voice titles


def add_turn(session, role, text, reasoning="", attached=()):
    """Appends a user or assistant message in the editor's item format."""
    item = {"id": str(uuid.uuid4()).upper(), "kind": role, "text": text, "reasoning": reasoning,
            "attached": list(attached), "steps": [], "counts": {k: 0 for k in COUNTS}}
    session["items"].append(item)
    session["updated"] = _now()
    if role == "user" and session.get("title") in (None, "", "New chat"):
        session["title"] = title_for(text)


def messages(session):
    """The conversation as chat-API messages (user and assistant text; the editor's activity,
    approval and image cards are not part of what the model is sent)."""
    out = []
    for i in session.get("items", []):
        kind, text = i.get("kind"), i.get("text")
        if kind in ("user", "assistant") and isinstance(text, str) and text:
            m = {"role": kind, "content": text}
            if kind == "assistant":
                m["reasoning_content"] = i.get("reasoning") if isinstance(i.get("reasoning"), str) else ""
            out.append(m)
    return out


def find(sessions, ref):
    """A session by its 1-based number in the list, or by a unique id prefix; None if there is none."""
    ref = str(ref).strip()
    if ref.isdigit():
        n = int(ref)
        return sessions[n - 1] if 1 <= n <= len(sessions) else None
    hits = [s for s in sessions if ref and s["id"].upper().startswith(ref.upper())]
    return hits[0] if len(hits) == 1 else None


def rename(session, title):
    t = " ".join(str(title).split())[:80]
    if not t:
        return False
    session["title"] = t
    return True


def delete(sessions, session):
    sessions[:] = [s for s in sessions if s["id"] != session["id"]]


def listing(sessions, current_id=None):
    """Lines for a terminal list: number, mark of the open chat, title, message count, age."""
    lines, now = [], _now()
    for n, s in enumerate(sessions, 1):
        age = max(0, now - s["updated"])
        when = (f"{int(age // 60)} min" if age < 3600 else f"{int(age // 3600)} h" if age < 86400
                else f"{int(age // 86400)} d")
        turns = sum(1 for i in s["items"] if i.get("kind") in ("user", "assistant"))
        mark = "▸" if s["id"] == current_id else " "
        lines.append(f"{mark}{n:>3}  {s['title']}  ·  {turns} · {when}")
    return lines
