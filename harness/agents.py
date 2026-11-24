"""Sub-agents: named helpers the chat starts for a self-contained task — each with its own conversation, its own
instructions and its own set of tools — that come back with a short report.

A definition is a Markdown file with front matter, the format Claude Code uses for `.claude/agents/`, so one
written for it works here:

    ---
    name: explorer
    description: Maps an unfamiliar part of the code and reports where things are.
    tools: read_file, search, list_dir          (optional; Claude Code names such as Read, Grep, Bash work too;
    ---                                          without it the agent gets every tool the chat has)
    You are …  (the instructions)

Found, first match of a name wins (a project's own overrides a personal one, both override a built-in):

    project   <root>/.llm/agents/*.md   <root>/.claude/agents/*.md
    personal  ~/.llm/agents/*.md        + LLM_AGENTS_DIRS (e.g. "~/.claude/agents")
    built-in  explorer, reviewer, fixer

They are saved only into `.llm/agents/` (scope "project") or `~/.llm/agents/` (scope "personal"). The running of
agents — in the background, with their log, status and approvals — is the chat's (agent.py, SubAgent / Run).
"""
import os
import re

from harness.skills import parse

NAME_RE = re.compile(r"^[a-z][a-z0-9-]{0,39}$")
RESERVED = {"list", "new", "create", "rm", "delete", "stop", "say", "show", "log", "result", "retry"}   # /agents …
WRITABLE = {"project": ".llm/agents", "personal": "~/.llm/agents"}
PROJECT_DIRS = (".llm/agents", ".claude/agents")
PERSONAL_DIRS = ("~/.llm/agents",)

READ_ONLY = ["list_dir", "read_file", "search", "use_skill", "web_search", "web_fetch", "web_research"]
# Claude Code's tool names → ours (a definition written for it keeps its meaning here)
ALIASES = {"read": "read_file", "grep": "search", "glob": "list_dir", "ls": "list_dir", "bash": "run",
           "edit": "edit_file", "multiedit": "edit_file", "write": "write_file", "websearch": "web_search",
           "webfetch": "web_fetch", "skill": "use_skill"}

BUILTIN = {
    "explorer": ("Finds where things are in the project and how they connect; reads, never changes anything.",
                 READ_ONLY,
                 "You explore a code base for the main conversation. Search and read only what the task needs, "
                 "follow the calls that matter, and stop as soon as you can answer. Report: the answer first, then "
                 "the files and functions involved as path:line, then anything uncertain. Never change files."),
    "reviewer": ("Reviews changes or a part of the code for bugs and risks; can run tests and read-only commands.",
                 READ_ONLY + ["run"],
                 "You review code for the main conversation. Look for real defects: wrong behaviour, missed cases, "
                 "broken contracts, security problems; not style. Use `git diff` (through run) when asked to review "
                 "changes, and run tests only when that settles a question. Report each finding as path:line, what "
                 "goes wrong and when, most serious first; say plainly if you found nothing. Never edit files."),
    "fixer": ("Makes a focused change or fixes a failing test; edits files and runs commands (each with approval).",
              None,
              "You make one focused change for the main conversation. Read before you edit, keep the change "
              "minimal and in the style of the code around it, run the relevant test or program to check it. "
              "Report what you changed (files), how you checked it, and what is left."),
}


class AgentError(ValueError):
    pass


class Definition:
    __slots__ = ("name", "description", "tools", "prompt", "path", "scope")

    def __init__(self, name, description, tools, prompt, path=None, scope="built-in"):
        self.name, self.description, self.tools = name, description, tools
        self.prompt, self.path, self.scope = prompt, path, scope

    @property
    def writable(self):
        if not self.path:
            return False
        folder = os.path.dirname(self.path)
        return folder.endswith(os.sep + WRITABLE["project"]) or folder == os.path.expanduser(WRITABLE["personal"])


def tool_list(value):
    """The `tools:` field → our tool names, or None (all tools). Accepts "a, b", "[a, b]" and Claude Code's names
    (an argument pattern such as `Bash(git diff:*)` keeps only the tool)."""
    if value is None:
        return None
    text = str(value).strip().strip("[]")
    if not text or text in ("*", "all"):
        return None
    out = []
    for raw in re.split(r"[,\s]+(?![^()]*\))", text):
        name = re.sub(r"\(.*\)$", "", raw.strip().strip("'\""))
        if not name:
            continue
        name = ALIASES.get(name.lower(), name if not name[0].isupper() else name.lower())
        if name not in out:
            out.append(name)
    return out


def _dirs(root):
    out = []
    if root:
        out += [(os.path.join(root, d), "project") for d in PROJECT_DIRS]
    out += [(os.path.expanduser(d), "personal") for d in PERSONAL_DIRS]
    extra = os.environ.get("LLM_AGENTS_DIRS", "")
    out += [(os.path.expanduser(d), "personal") for d in extra.split(os.pathsep) if d.strip()]
    return out


def load(path, scope):
    with open(path, encoding="utf-8", errors="replace") as f:
        meta, body = parse(f.read(128_000))
    name = (meta.get("name") or os.path.splitext(os.path.basename(path))[0]).strip().lower()
    if not NAME_RE.match(name) or name in RESERVED:
        return None
    return Definition(name, " ".join((meta.get("description") or "").split()), tool_list(meta.get("tools")),
                      body.strip(), path, scope)


def discover(root=None):
    """Every definition by name: project first, then personal, then the built-ins."""
    found = {}
    for folder, scope in _dirs(root):
        try:
            names = sorted(n for n in os.listdir(folder) if n.endswith(".md"))
        except OSError:
            continue
        for n in names:
            try:
                d = load(os.path.join(folder, n), scope)
            except OSError:
                continue
            if d and d.name not in found:
                found[d.name] = d
    for name, (desc, tools, prompt) in BUILTIN.items():
        found.setdefault(name, Definition(name, desc, tools, prompt))
    return found


def find(name, root=None):
    return discover(root).get((name or "").strip().lower())


def render(name, description, tools, instructions):
    desc = " ".join((description or "").split()).replace('"', "'")
    if not desc:
        raise AgentError("an agent needs a description: what it does and when to use it")
    if not (instructions or "").strip():
        raise AgentError("an agent needs instructions")
    line = f"tools: {', '.join(tools)}\n" if tools else ""
    return f'---\nname: {name}\ndescription: "{desc}"\n{line}---\n\n{instructions.strip()}\n'


def target(name, scope, root):
    if not NAME_RE.match(name or ""):
        raise AgentError("an agent name is lowercase letters, digits and hyphens, starting with a letter")
    if name in RESERVED:
        raise AgentError(f"{name!r} is a word of the /agents command; choose another name")
    if scope not in WRITABLE:
        raise AgentError("scope is 'project' or 'personal'")
    if scope == "project" and not root:
        raise AgentError("no project is attached: save it as a personal agent")
    base = os.path.join(root, WRITABLE["project"]) if scope == "project" else os.path.expanduser(WRITABLE["personal"])
    return os.path.join(base, name + ".md")


def plan_save(name, description, tools, instructions, scope="project", root=None):
    """(path, old text, new text) of a save, without writing — what the approval shows."""
    path = target(name, scope, root)
    old = ""
    if os.path.isfile(path):
        with open(path, encoding="utf-8", errors="replace") as f:
            old = f.read()
    return path, old, render(name, description, tool_list(tools) if isinstance(tools, str) else tools, instructions)


def save(name, description, tools, instructions, scope="project", root=None):
    path, _, text = plan_save(name, description, tools, instructions, scope, root)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(path + ".tmp", path)
    return path


def delete(name, root=None):
    d = find(name, root)
    if d is None:
        raise AgentError(f"no agent named {name!r}")
    if not d.writable:
        raise AgentError(f"{d.name} is {'built in' if not d.path else 'in ' + os.path.dirname(d.path)}: "
                         "it can't be removed from here")
    os.remove(d.path)
    return d.path


def catalog(root=None, budget=900):
    """The short list of agents for the system prompt."""
    lines, used = [], 0
    for d in discover(root).values():
        line = f"- {d.name}: {d.description}"
        if used + len(line) > budget:
            break
        lines.append(line)
        used += len(line)
    return "\n".join(lines)
