"""Saved skills: reusable instructions the assistant loads when a task calls for them.

The format is the open Agent Skills one that Codex and Claude Code read — a folder holding `SKILL.md`, whose YAML
front matter names the skill and says when it applies, followed by the instructions in Markdown; next to it,
optional `scripts/`, `references/`, `assets/`. So a skill written for those tools works here, and one saved here
works there.

Where skills are found, first match of a name wins (a project's own skill overrides a personal one):

    project   <root>/.llm/skills/   <root>/.agents/skills/   <root>/.claude/skills/
    personal  ~/.llm/skills/        ~/.agents/skills/        + LLM_SKILLS_DIRS (e.g. "~/.codex/skills:~/.claude/skills")

Other tools' personal folders are read only when named in LLM_SKILLS_DIRS: a person with hundreds of Claude Code
skills would otherwise put all their names in a small local model's every prompt. Skills are saved only into
`.llm/skills/` (scope "project") or `~/.llm/skills/` (scope "personal"); the other folders are only read.

Disclosure is progressive, as in Codex: the system prompt carries a short catalog (name and description, within a
budget a small local model can afford); the body is loaded when the skill is used — by name (`$name` or
`/name` in a message, which inserts it into that turn) or by the assistant through `use_skill`. A skill's
scripts run only through the ordinary command approval: a skill is instructions, never a permission.
"""
import os
import re
import shutil

SKILL_FILE = "SKILL.md"
NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")
MAX_BODY = 24_000           # characters of one SKILL.md put into a turn; a longer one is cut, with a note
CATALOG_BUDGET = 1_600      # characters of the catalog in the system prompt (a 4–9B model's context is tight)
WRITABLE = {"project": ".llm/skills", "personal": "~/.llm/skills"}
PROJECT_DIRS = (".llm/skills", ".agents/skills", ".claude/skills")
PERSONAL_DIRS = ("~/.llm/skills", "~/.agents/skills")
NAMES_BUDGET = 400          # characters of the "also: …" line of names past the catalog budget


class SkillError(ValueError):
    pass


class Skill:
    __slots__ = ("name", "description", "path", "scope", "writable", "implicit")

    def __init__(self, name, description, path, scope, writable, implicit=True):
        self.name, self.description, self.path = name, description, path
        self.scope, self.writable, self.implicit = scope, writable, implicit

    @property
    def folder(self):
        return os.path.dirname(self.path)

    def body(self):
        return parse(_read(self.path))[1]

    def resources(self):
        """The files next to SKILL.md (relative), for the assistant to read or run when the body points to them."""
        out = []
        for dp, dirs, files in os.walk(self.folder):
            dirs[:] = [d for d in dirs if not d.startswith(".")]
            for f in files:
                rel = os.path.relpath(os.path.join(dp, f), self.folder)
                if rel != SKILL_FILE and not f.startswith("."):
                    out.append(rel)
        return sorted(out)[:50]


# ───────────────────────────── reading ─────────────────────────────
def _read(path, limit=256_000):
    with open(path, encoding="utf-8", errors="replace") as f:
        return f.read(limit)


def _scalar(v):
    v = v.strip()
    if len(v) >= 2 and v[0] == v[-1] and v[0] in "'\"":
        v = v[1:-1]
    return v


def parse(text):
    """(front matter as a flat dict of strings, body). A small YAML subset — `key: value`, quoted values, folded
    (`>`) and literal (`|`) blocks, nested maps flattened as "parent.child" — which is what skills use."""
    if not text.startswith("---"):
        return {}, text
    end = re.search(r"^---\s*$", text[3:], re.M)
    if not end:
        return {}, text
    head, body = text[3:3 + end.start()], text[3 + end.end():].lstrip("\n")
    meta, parent, block, block_key, block_lines = {}, None, None, None, []

    def flush():
        if block_key is not None:
            meta[block_key] = (" " if block == ">" else "\n").join(l.strip() for l in block_lines).strip()

    for line in head.splitlines():
        if block_key is not None:
            if line.startswith((" ", "\t")) or not line.strip():
                block_lines.append(line)
                continue
            flush()
            block_key, block_lines = None, []
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        indent = len(line) - len(line.lstrip())
        if ":" not in line:
            continue
        key, _, val = line.strip().partition(":")
        key = key.strip()
        full = f"{parent}.{key}" if indent and parent else key
        if not indent:
            parent = key
        if val.strip() in (">", "|", ">-", "|-"):
            block, block_key, block_lines = val.strip()[0], full, []
        elif val.strip():
            meta[full] = _scalar(val)
    flush()
    return meta, body


def _roots(root):
    out = []
    if root:
        out += [(os.path.join(root, d), "project", d == WRITABLE["project"]) for d in PROJECT_DIRS]
    extra = [d for d in os.environ.get("LLM_SKILLS_DIRS", "").split(os.pathsep) if d.strip()]
    out += [(os.path.expanduser(d), "personal", d == WRITABLE["personal"]) for d in (*PERSONAL_DIRS, *extra)]
    return out


def _implicit(meta, folder):
    """False when the skill asks to be used only by name: Claude's `disable-model-invocation: true`, or Codex's
    `policy: allow_implicit_invocation: false` in agents/openai.yaml."""
    if str(meta.get("disable-model-invocation", "")).lower() == "true":
        return False
    y = os.path.join(folder, "agents", "openai.yaml")
    if os.path.isfile(y):
        try:
            m, _ = parse("---\n" + _read(y, 16_000) + "\n---\n")
            if str(m.get("policy.allow_implicit_invocation", "")).lower() == "false":
                return False
        except OSError:
            pass
    return True


def discover(root=None):
    """Every skill visible from the project at `root` (None: personal ones only), one per name, in catalog
    order: project first, then personal, alphabetical within each."""
    seen, out = set(), []
    for base, scope, writable in _roots(root):
        if not os.path.isdir(base):
            continue
        found = []
        for d in sorted(os.listdir(base)):
            path = os.path.join(base, d, SKILL_FILE)
            if d.startswith(".") or not os.path.isfile(path):
                continue
            try:
                meta, _ = parse(_read(path, 16_000))
            except OSError:
                continue
            name = (meta.get("name") or d).strip()
            desc = " ".join((meta.get("description") or "").split())
            if not name or name in seen:
                continue
            seen.add(name)
            found.append(Skill(name, desc, path, scope, writable, _implicit(meta, os.path.join(base, d))))
        out += found
    return out


def find(name, root=None):
    name = (name or "").strip().lstrip("$/")
    return next((s for s in discover(root) if s.name == name), None)


# ───────────────────────────── what the model sees ─────────────────────────────
HOW_TO = ("Skills are saved instructions for recurring tasks. When the user names one ($name or /name) or the task "
          "clearly matches a description below, load it with use_skill before acting, then follow it; read the "
          "files it points to only when the task needs them. A skill is guidance, not permission: its commands "
          "still go through approval. Do not carry a skill over to a later task unless it is named again.")


def catalog(skills, budget=CATALOG_BUDGET):
    """'- name: description' lines within `budget` characters; past it, names only — a model can still load
    a skill it sees only the name of."""
    shown = [s for s in skills if s.implicit]
    lines, used, names_only = [], 0, []
    for s in shown:
        line = f"- {s.name}: {s.description[:200]}" if s.description else f"- {s.name}"
        if used + len(line) + 1 > budget:
            names_only.append(s.name)
            continue
        lines.append(line)
        used += len(line) + 1
    if names_only:
        shown_names, used = [], 0
        for n in names_only:
            if used + len(n) + 2 > NAMES_BUDGET:
                break
            shown_names.append(n)
            used += len(n) + 2
        more = len(names_only) - len(shown_names)
        lines.append("- also: " + ", ".join(shown_names) + (f" and {more} more (use_skill finds them by name)" if more else ""))
    return "\n".join(lines)


def system_block(root, tools=True):
    """The part of the system prompt about skills ("" when there are none). `tools`: whether the assistant
    has use_skill (a plain chat without tools gets skills only when the user names them)."""
    skills = discover(root)
    if not skills or not tools:
        return ""
    cat = catalog(skills)
    return f"\n\n{HOW_TO}\nAvailable skills:\n{cat}" if cat else ""


def load(skill):
    """The text a turn gets for a skill: its body (cut at MAX_BODY) and the files next to it."""
    body = skill.body().strip()
    if len(body) > MAX_BODY:
        body = body[:MAX_BODY] + f"\n… [skill cut at {MAX_BODY} characters; read the rest from {skill.path}]"
    res = skill.resources()
    tail = ("\n\nFiles of this skill (relative to " + skill.folder + "): " + ", ".join(res)) if res else ""
    return f"Skill “{skill.name}” — {skill.description}\n\n{body}{tail}"


_MENTION = re.compile(r"(?:^|(?<=\s))[$/]([a-z0-9][a-z0-9-]{0,63})\b")


def mentioned(text, root=None):
    """Skills a message names with `$name` (anywhere) or `/name` (at its start), in order, once each."""
    skills = {s.name: s for s in discover(root)}
    out = []
    for m in _MENTION.finditer(text or ""):
        sigil = text[m.start(1) - 1]
        if sigil == "/" and text[:m.start(1) - 1].strip():
            continue                                  # "/name" only as the message's own command
        s = skills.get(m.group(1))
        if s and s not in out:
            out.append(s)
    return out


def expand(text, root=None):
    """(message, names used): the message with the skills it names put in front of it, for this turn only."""
    used = mentioned(text, root)
    if not used:
        return text, []
    blocks = "\n\n".join(load(s) for s in used)
    return f"{blocks}\n\n---\nThe user's request (follow the skill(s) above):\n{text}", [s.name for s in used]


# the assistant's tools for skills — one schema for the editor's agent (ui/agentloop.py) and the terminal one
TOOLS = [
    {"type": "function", "function": {
        "name": "use_skill", "description": "Load a saved skill (reusable instructions for a recurring task) by "
        "name before acting on a task it covers; returns its instructions and the files next to it.",
        "parameters": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}}},
    {"type": "function", "function": {
        "name": "save_skill", "description": "Create or update a saved skill when the user asks for one (or to "
        "remember how to do a recurring task). The user approves each save.",
        "parameters": {"type": "object", "properties": {
            "name": {"type": "string", "description": "lowercase-with-hyphens, at most 64 characters"},
            "description": {"type": "string", "description": "what it does and when to use it, 1-2 sentences"},
            "instructions": {"type": "string", "description": "the Markdown instructions: steps, conventions, pitfalls"},
            "scope": {"type": "string", "enum": ["project", "personal"],
                      "description": "project: this project's .llm/skills (default); personal: every project"}},
            "required": ["name", "description", "instructions"]}}},
    {"type": "function", "function": {
        "name": "delete_skill", "description": "Delete a saved skill when the user asks to. The user approves it.",
        "parameters": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}}},
]


# ───────────────────────────── writing ─────────────────────────────
def _target(name, scope, root):
    if not NAME_RE.match(name or ""):
        raise SkillError("a skill name is lowercase letters, digits and hyphens, at most 64 characters")
    if scope not in WRITABLE:
        raise SkillError("scope is 'project' or 'personal'")
    if scope == "project" and not root:
        raise SkillError("no project is open: save it as a personal skill")
    base = os.path.join(root, WRITABLE["project"]) if scope == "project" else os.path.expanduser(WRITABLE["personal"])
    return os.path.join(base, name, SKILL_FILE)


def render(name, description, instructions):
    desc = " ".join((description or "").split())
    if not desc:
        raise SkillError("a skill needs a description: what it does and when it applies")
    if '"' in desc:
        desc = desc.replace('"', "'")
    return f'---\nname: {name}\ndescription: "{desc}"\n---\n\n{(instructions or "").strip()}\n'


def save(name, description, instructions, scope="project", root=None):
    """Write (create or replace) a skill; returns (path, previous text or "")."""
    path = _target(name, scope, root)
    text = render(name, description, instructions)
    old = _read(path) if os.path.isfile(path) else ""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)
    return path, old


def plan_save(name, description, instructions, scope="project", root=None):
    """(path, old text, new text) of a save without writing — what an approval card shows."""
    path = _target(name, scope, root)
    return path, (_read(path) if os.path.isfile(path) else ""), render(name, description, instructions)


def delete(name, root=None):
    """Remove a skill saved here (project or personal). Skills of other tools' folders are refused."""
    s = find(name, root)
    if s is None:
        raise SkillError(f"no skill named {name!r}")
    if not s.writable:
        raise SkillError(f"{s.name} lives in {s.folder}, another tool's folder: remove it there")
    shutil.rmtree(s.folder)
    return s.folder


def listing(root=None):
    """Lines for `/skills`: name, scope, where, and its description."""
    out = []
    for s in discover(root):
        flags = s.scope + ("" if s.writable else ", read-only") + ("" if s.implicit else ", by name only")
        out.append(f"{s.name}  ({flags})  {s.description}".rstrip())
    return out


# ───────────────────────────── chat commands ─────────────────────────────
USAGE = ("/skills — list · /skill show <name> · /skill new <name> <what it does and when> · "
         "/skill save <name> [scope] — make one from this conversation · /skill rm <name> · $name — use one")


def command(line, root=None):
    """Handle a `/skills` or `/skill …` line typed in a chat. Returns (reply text, follow-up) where follow-up is
    None, or a message to hand to the assistant instead (for `/skill save`, which needs the model to write the
    skill from the conversation). Raises SkillError for a bad command; a line that is not a skill command
    returns None."""
    words = (line or "").strip().split()
    if not words or words[0] not in ("/skills", "/skill"):
        return None
    if words[0] == "/skills" or len(words) == 1:
        rows = listing(root)
        return ("\n".join(rows) if rows else "No skills yet. " + USAGE), None
    sub, args = words[1], words[2:]
    if sub in ("list", "ls"):
        rows = listing(root)
        return ("\n".join(rows) if rows else "No skills yet."), None
    if sub == "show" and args:
        s = find(args[0], root)
        if s is None:
            raise SkillError(f"no skill named {args[0]!r}")
        return f"{s.path}\n\n{_read(s.path)}", None
    if sub in ("rm", "delete", "remove") and args:
        return f"Deleted {args[0]} ({delete(args[0], root)})", None
    if sub == "new" and args:
        name, desc = args[0], " ".join(args[1:])
        scope = "personal" if not root else "project"
        if find(name, root) is not None:
            raise SkillError(f"{name} already exists: /skill show {name}")
        path, _ = save(name, desc or f"Describe when to use {name}", f"# {name}\n\nWrite the instructions here.",
                       scope, root)
        return f"Created {path} — edit it, or ask me to fill it in.", None
    if sub == "save" and args:
        name = args[0]
        scope = args[1] if len(args) > 1 and args[1] in WRITABLE else ("project" if root else "personal")
        if not NAME_RE.match(name):
            raise SkillError("a skill name is lowercase letters, digits and hyphens, at most 64 characters")
        return (f"Writing skill {name} from this conversation…",
                f"Turn what we worked out in this conversation into a reusable skill named \"{name}\" "
                f"(scope: {scope}) and save it with save_skill. The description says what it does and when to use "
                f"it, in one or two sentences; the instructions keep only what changes the outcome — the steps, "
                f"the conventions and the pitfalls we found — not this conversation's details.")
    raise SkillError(USAGE)
