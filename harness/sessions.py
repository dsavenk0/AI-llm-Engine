"""Saved terminal chats: many conversations, each one continued or deleted later (`llm -c`, `llm -r`, /chats).

One JSON-lines file per chat in ~/Library/Application Support/LLMBar/sessions/<id>.jsonl, written as the chat
goes — every message is appended the moment it exists, so a Ctrl+C, a crash or a closed window loses nothing:

    {"chat": "<id>", "created": 1732450000.0, "project": "/path/or/null", "title": "…"}   the header (first line)
    {"m": {"role": "user", "content": "…"}}                                             one message
    {"reset": [ …messages… ]}                       after compaction: the conversation is replaced by this
    {"title": "…"} / {"project": "…"}               a rename, a project attached later

The system prompt is never stored (it is rebuilt from the project when a chat is opened). Chats older than
KEEP_DAYS and those beyond KEEP_CHATS (oldest first) are removed when a new chat starts.
"""
import json
import os
import time
import uuid

KEEP_DAYS = 30
KEEP_CHATS = 100
TITLE_CHARS = 60


def default_dir():
    return os.environ.get("LLM_SESSIONS_DIR") or os.path.expanduser("~/Library/Application Support/LLMBar/sessions")


def title_for(text):
    t = " ".join(str(text).split())
    if len(t) <= TITLE_CHARS:
        return t
    cut = t[:TITLE_CHARS]
    return (cut.rsplit(" ", 1)[0] if " " in cut else cut) + " …"


class Chat:
    """One saved chat. `messages` excludes the system prompt."""

    def __init__(self, path, header, messages, updated):
        self.path, self.messages, self.updated = path, messages, updated
        self.id = header["chat"]
        self.created = header.get("created") or updated
        self.title = header.get("title") or ""
        self.project = header.get("project")
        self.written = len(messages)          # how many of `messages` are already in the file
        self.stale = False                    # True: the file differs from `messages` (repaired on reading)

    @property
    def questions(self):
        return sum(1 for m in self.messages if m.get("role") == "user")

    # ── writing ──
    def _append(self, *objs):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            for o in objs:
                f.write(json.dumps(o, ensure_ascii=False) + "\n")
        self.updated = time.time()

    def sync(self, messages):
        """Bring the file up to `messages` (the conversation without its system prompt): new messages are
        appended; a conversation that shrank or changed (compaction, /retry) is written as one reset line."""
        messages = [m for m in messages if m.get("role") != "system"]
        if not messages and not os.path.exists(self.path):
            return                                   # an empty chat is not stored
        if not self.title:                           # named after its first question
            first = next((m for m in messages if m.get("role") == "user"), None)
            if first:
                self.title = title_for(_text(first.get("content")))
        if not os.path.exists(self.path):
            self._append({"chat": self.id, "created": self.created, "project": self.project, "title": self.title})
            self.written = 0
        same_prefix = (not self.stale and len(messages) >= self.written
                       and self.messages[:self.written] == messages[:self.written])
        if same_prefix:
            new = messages[self.written:]
            if new:
                self._append(*({"m": m} for m in new))
        else:
            self._append({"reset": messages})
        self.messages = [dict(m) for m in messages]
        self.written = len(messages)
        self.stale = False

    def rename(self, title):
        t = title_for(title)
        if not t:
            return False
        self.title = t
        if os.path.exists(self.path):
            self._append({"title": t})
        return True

    def set_project(self, root):
        self.project = root
        if os.path.exists(self.path):
            self._append({"project": root})


def _text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):                  # a message with an image: its text parts
        return " ".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text")
    return ""


def new(project=None, directory=None):
    cid = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    path = os.path.join(directory or default_dir(), cid + ".jsonl")
    return Chat(path, {"chat": cid, "created": time.time(), "project": project}, [], time.time())


def read(path):
    """The chat in `path`, or None when the file is not a chat (damaged lines are skipped)."""
    header, messages = None, []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    o = json.loads(line)
                except ValueError:
                    continue                         # a line cut by a crash mid-write
                if not isinstance(o, dict):
                    continue
                if header is None:
                    if "chat" not in o:
                        return None
                    header = o
                elif "m" in o and isinstance(o["m"], dict):
                    messages.append(o["m"])
                elif isinstance(o.get("reset"), list):
                    messages = [m for m in o["reset"] if isinstance(m, dict)]
                elif "title" in o:
                    header["title"] = o["title"]
                elif "project" in o:
                    header["project"] = o["project"]
        updated = os.path.getmtime(path)
    except OSError:
        return None
    if header is None:
        return None
    repaired = _repair(messages)
    chat = Chat(path, header, repaired, updated)
    chat.stale = len(repaired) != len(messages)      # its next write replaces the file's conversation with this one
    return chat


def _repair(messages):
    """A chat cut off mid-turn may end with tool calls that never got their results; the next request would be
    malformed, so those calls are answered as cancelled (the same as a Ctrl+C during the turn)."""
    out = list(messages)
    for i, m in enumerate(messages):
        calls = m.get("tool_calls") if m.get("role") == "assistant" else None
        if not calls:
            continue
        answered = {x.get("tool_call_id") for x in messages[i + 1:] if x.get("role") == "tool"}
        missing = [c for c in calls if c.get("id") not in answered]
        if missing:
            at = i + 1
            while at < len(out) and out[at].get("role") == "tool":
                at += 1
            for c in missing:
                out.insert(at, {"role": "tool", "tool_call_id": c.get("id"), "content": "cancelled (the chat was closed)"})
                at += 1
    return out


def all_chats(directory=None):
    """Every saved chat, the most recently used first."""
    d = directory or default_dir()
    try:
        names = [n for n in os.listdir(d) if n.endswith(".jsonl")]
    except OSError:
        return []
    chats = [c for c in (read(os.path.join(d, n)) for n in names) if c and c.messages]
    chats.sort(key=lambda c: c.updated, reverse=True)
    return chats


def latest(directory=None, project=None):
    """The most recent chat (of `project`, when given and it has one)."""
    chats = all_chats(directory)
    if project:
        mine = [c for c in chats if c.project == project]
        if mine:
            return mine[0]
    return chats[0] if chats else None


def find(chats, ref):
    """A chat by its number in the list (1 = newest), a unique id prefix, or its name (the title, or a unique
    beginning of it, ignoring case); None when there is none or several match."""
    ref = str(ref or "").strip()
    if not ref:
        return None
    if ref.isdigit() and len(ref) < 5:
        n = int(ref)
        return chats[n - 1] if 1 <= n <= len(chats) else None
    hits = [c for c in chats if c.id.startswith(ref)]
    if not hits:
        low = ref.lower()
        hits = [c for c in chats if c.title.lower() == low] or [c for c in chats if c.title.lower().startswith(low)]
    return hits[0] if len(hits) == 1 else None


def fork(chat, directory=None):
    """A new chat that starts as a copy of `chat` (which stays as it is)."""
    copy = new(chat.project, directory or os.path.dirname(chat.path))
    copy.title = title_for(f"{chat.title} (fork)") if chat.title else ""
    copy.sync([dict(m) for m in chat.messages])
    return copy


def delete(chat):
    try:
        os.remove(chat.path)
        return True
    except OSError:
        return False


def prune(directory=None, now=None):
    """Drop chats unused for KEEP_DAYS, then the oldest beyond KEEP_CHATS. Returns how many were removed."""
    now = now or time.time()
    chats = all_chats(directory)
    gone = [c for c in chats if now - c.updated > KEEP_DAYS * 86400]
    kept = [c for c in chats if c not in gone]
    gone += kept[KEEP_CHATS:]
    return sum(delete(c) for c in gone)


def age(seconds):
    s = max(0, seconds)
    return (f"{int(s // 60)} min" if s < 3600 else f"{int(s // 3600)} h" if s < 86400 else f"{int(s // 86400)} d")


def listing(chats, current=None, now=None):
    """Lines for the terminal: number, mark of the open chat, title, project, questions, age."""
    now = now or time.time()
    lines = []
    for n, c in enumerate(chats, 1):
        mark = "▸" if current is not None and c.id == current else " "
        where = f"  · {os.path.basename(c.project)}" if c.project else ""
        lines.append(f"{mark}{n:>3}  {c.title or '…'}{where}  · {c.questions} · {age(now - c.updated)}")
    return lines
