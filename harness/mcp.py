"""A small MCP client (stdlib only): stdio and streamable-HTTP servers, tools only.

Config — `.llm/mcp.json` in the workspace, else `~/.llm/mcp.json`:

    {"servers": {
       "context7": {"url": "https://mcp.context7.com/mcp",
                    "headers": {"CONTEXT7_API_KEY": "${CONTEXT7_API_KEY}"}},
       "files":    {"command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", "."],
                    "enabled": false,
                    "tools": {"write_file": false}}}}

`enabled` switches a whole server, `tools` switches single tools (default: all on). Servers
are contacted only when asked for (`/mcp on` in the agent) and never at import time.
Tools reach the model as `<server>__<tool>`.
"""
import json
import os
import queue
import re
import subprocess
import threading
import urllib.error
import urllib.request

PROTOCOL = "2025-03-26"
MAX_RESULT = 100_000          # characters of one tool result that reach the model
MAX_PAGES = 50                # pages of tools/list followed before giving up (a server that repeats its cursor)
PRESETS = {
    "context7": {"url": "https://mcp.context7.com/mcp"},
    # Google Stitch (UI designs / mockups). The key is created in Stitch → Settings → API keys and
    # read from the environment. Only the core tools: all 15 would cost ~15k tokens of every prompt.
    "stitch": {"url": "https://stitch.googleapis.com/mcp",
               "headers": {"X-Goog-Api-Key": "${STITCH_API_KEY}"},
               "only": ["list_projects", "create_project", "get_project", "list_screens", "get_screen",
                        "generate_screen_from_text", "edit_screens"]},
}
CLIENT = {"name": "llm-agent", "version": "1"}


class MCPError(RuntimeError):
    pass


def _expand(v):
    if isinstance(v, str):
        return re.sub(r"\$\{(\w+)\}", lambda m: os.environ.get(m.group(1), ""), v)
    if isinstance(v, list):
        return [_expand(x) for x in v]
    if isinstance(v, dict):
        return {k: _expand(x) for k, x in v.items()}
    return v


# ───────────────────────────── config ─────────────────────────────
def config_path(root):
    for p in (os.path.join(root, ".llm", "mcp.json"), os.path.expanduser("~/.llm/mcp.json")):
        if os.path.isfile(p):
            return p
    return os.path.join(root, ".llm", "mcp.json")


def workspace_config_path(root):
    """Only the project's own `.llm/mcp.json` (no `~/.llm` fallback) — what the editor's MCP panel edits."""
    return os.path.join(root, ".llm", "mcp.json")


def load_config(root, path=None):
    """{name: spec} with `enabled` defaulting to True. Accepts the common {"mcpServers": …} spelling.
    `path` reads a specific file (the editor passes the workspace one); None uses `config_path`."""
    p = path or config_path(root)
    try:
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return {}
    servers = (data.get("servers") or data.get("mcpServers") or {}) if isinstance(data, dict) else {}
    if not isinstance(servers, dict):
        return {}
    return {n: dict(s, enabled=s.get("enabled", True)) for n, s in servers.items() if isinstance(s, dict)}


def describe(root):
    """One line per configured server saying what turning it on does: the program that will be started, or
    the address that will be contacted and which environment variables go into the request."""
    out = []
    for name, spec in load_config(root).items():
        if not spec["enabled"]:
            continue
        env = sorted(set(re.findall(r"\$\{(\w+)\}", json.dumps(spec))))
        use = f" — uses ${', $'.join(env)}" if env else ""
        if spec.get("url"):
            out.append(f"{name}: connects to {spec['url']}{use}")
        elif spec.get("command"):
            out.append(f"{name}: runs  {spec['command']} {' '.join(map(str, spec.get('args') or []))}{use}")
        else:
            out.append(f"{name}: (incomplete)")
    return out


def add_preset(root, name, path=None):
    if name not in PRESETS:
        raise ValueError(f"unknown preset {name!r}; available: {', '.join(PRESETS)}")
    p = path or config_path(root)
    try:
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        data = {}
    if not isinstance(data, dict):
        data = {}
    key = "mcpServers" if "mcpServers" in data else "servers"
    if not isinstance(data.get(key), dict):
        data[key] = {}
    data[key][name] = dict(PRESETS[name])
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return p


def _edit_config(root, mutate, path=None):
    """Read `.llm/mcp.json`, let `mutate(servers)` change the servers dict in place, write it back
    (keeping the file's `servers`/`mcpServers` spelling). Returns the written path."""
    p = path or config_path(root)
    try:
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        data = {}
    if not isinstance(data, dict):
        data = {}
    key = "mcpServers" if "mcpServers" in data else "servers"
    if not isinstance(data.get(key), dict):
        data[key] = {}
    mutate(data[key])
    os.makedirs(os.path.dirname(p), exist_ok=True)
    tmp = f"{p}.{os.getpid()}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, p)
    return p


_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 _.-]{0,39}$")


def valid_spec(spec):
    """A server spec is usable when it names either a URL or a command. Returns a reason it is not, or ''."""
    if not isinstance(spec, dict):
        return "the server description must be an object"
    if spec.get("url"):
        return "" if isinstance(spec["url"], str) else "`url` must be text"
    if spec.get("command"):
        return "" if isinstance(spec["command"], str) else "`command` must be text"
    return "a server needs either `url` (streamable HTTP) or `command` (a program to run over stdio)"


def set_server(root, name, spec, path=None):
    """Add or replace one server. Raises ValueError for a bad name or incomplete spec."""
    name = str(name).strip()
    if not _NAME_RE.match(name):
        raise ValueError("the name may hold letters, digits, spaces, '.', '_' and '-' (1–40 characters)")
    bad = valid_spec(spec)
    if bad:
        raise ValueError(bad)
    _edit_config(root, lambda servers: servers.__setitem__(name, spec), path)
    return name


def remove_server(root, name, path=None):
    found = [False]

    def mut(servers):
        found[0] = servers.pop(name, None) is not None
    _edit_config(root, mut, path)
    return found[0]


def set_enabled(root, name, on, path=None):
    def mut(servers):
        if name not in servers:
            raise ValueError(f"no server named {name!r}")
        servers[name] = dict(servers[name], enabled=bool(on))
    _edit_config(root, mut, path)
    return bool(on)


def set_tool(root, name, tool, on, path=None):
    """Switch one tool of a server on or off (default: all on)."""
    def mut(servers):
        if name not in servers:
            raise ValueError(f"no server named {name!r}")
        s = dict(servers[name])
        toggles = dict(s.get("tools") or {})
        if on:
            toggles.pop(tool, None)                 # back to the default (on); keep the map small
        else:
            toggles[tool] = False
        if toggles:
            s["tools"] = toggles
        else:
            s.pop("tools", None)
        servers[name] = s
    _edit_config(root, mut, path)
    return bool(on)


def probe(root, name, path=None):
    """Connect one configured server and report what it offers, for the UI's 'Test' button. Starts the
    program / contacts the address, so callers gate this behind the person's confirmation."""
    spec = load_config(root, path).get(name)
    if spec is None:
        raise ValueError(f"no server named {name!r}")
    s = Server(name, spec, cwd=root)
    try:
        s.connect()
    except MCPError as e:
        return {"ok": False, "error": str(e), "tools": [], "info": {}}
    toggles = spec.get("tools") or {}
    only = spec.get("only")
    tools_out = [{"name": t["name"], "description": (t.get("description") or "").strip().split("\n")[0][:200],
                  "enabled": toggles.get(t["name"], True) and (only is None or t["name"] in only)}
                 for t in s.tools_list]
    info = s.info
    s.close()
    return {"ok": True, "error": None, "info": info, "tools": tools_out}


# ───────────────────────────── transports ─────────────────────────────
class _Stdio:
    def __init__(self, spec, cwd):
        env = dict(os.environ, **_expand(spec.get("env") or {}))
        try:
            self.proc = subprocess.Popen([_expand(spec["command"])] + _expand(spec.get("args") or []),
                                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                         cwd=cwd, env=env, text=True, bufsize=1)
        except OSError as e:
            raise MCPError(f"cannot start {spec.get('command')!r}: {e}")
        self.inbox = queue.Queue()
        self.exited = False
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self):
        for line in self.proc.stdout:
            try:
                m = json.loads(line)
            except ValueError:
                continue                           # servers sometimes log to stdout
            if isinstance(m, dict):                # a bare number or list is noise, not a JSON-RPC message
                self.inbox.put(m)
        self.inbox.put(None)

    def send(self, msg, want_id=None, timeout=60):
        if self.exited:
            raise MCPError("server exited")
        try:
            self.proc.stdin.write(json.dumps(msg) + "\n")
            self.proc.stdin.flush()
        except (OSError, ValueError) as e:
            raise MCPError(f"server closed the connection: {e}")
        if want_id is None:
            return None
        while True:
            try:
                m = self.inbox.get(timeout=timeout)
            except queue.Empty:
                raise MCPError(f"no answer within {timeout} s")
            if m is None:
                self.exited = True
                raise MCPError("server exited")
            if m.get("id") == want_id and ("result" in m or "error" in m):
                return m
            if "method" in m and "id" in m:        # a request from the server (sampling, roots…): decline
                self.send({"jsonrpc": "2.0", "id": m["id"],
                           "error": {"code": -32601, "message": "not supported by this client"}})

    def close(self):
        try:
            self.proc.stdin.close()
            self.proc.terminate()
            self.proc.wait(3)
        except Exception:
            self.proc.kill()


class _Http:
    def __init__(self, spec):
        self.url = _expand(spec["url"])
        self.headers = {k: v for k, v in _expand(spec.get("headers") or {}).items() if v != ""}
        self.session = None

    @staticmethod
    def _parse(raw, ctype, want_id):
        if "text/event-stream" in ctype:
            data = []
            for line in raw.splitlines() + [""]:
                if line.startswith("data:"):
                    data.append(line[5:].lstrip())
                elif not line.strip() and data:
                    try:
                        m = json.loads("\n".join(data))
                    except ValueError:
                        m = None
                    data = []
                    if isinstance(m, dict) and m.get("id") == want_id and ("result" in m or "error" in m):
                        return m
            return None
        try:
            m = json.loads(raw) if raw.strip() else None
        except ValueError:
            return None                            # an HTML error page or other noise
        for x in (m if isinstance(m, list) else [m]):
            if isinstance(x, dict) and x.get("id") == want_id:
                return x
        return None

    def send(self, msg, want_id=None, timeout=60):
        h = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream",
             "User-Agent": "llm-agent/1", **self.headers}
        if self.session:
            h["Mcp-Session-Id"] = self.session
            h["MCP-Protocol-Version"] = PROTOCOL
        req = urllib.request.Request(self.url, data=json.dumps(msg).encode(), headers=h, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                self.session = r.headers.get("Mcp-Session-Id") or self.session
                raw = r.read(8_000_000).decode("utf-8", "replace")
                ctype = r.headers.get("Content-Type", "")
        except urllib.error.HTTPError as e:
            body = e.read(500).decode("utf-8", "replace")
            e.close()
            raise MCPError(f"HTTP {e.code} from {self.url}: {body[:200]}")
        except (urllib.error.URLError, OSError) as e:
            raise MCPError(f"cannot reach {self.url}: {e}")
        if want_id is None:
            return None
        m = self._parse(raw, ctype, want_id)
        if m is None:
            raise MCPError("no response to the request")
        return m

    def close(self):
        if self.session:
            try:
                req = urllib.request.Request(self.url, headers={**self.headers, "Mcp-Session-Id": self.session},
                                             method="DELETE")
                urllib.request.urlopen(req, timeout=3).close()
            except Exception:
                pass


# ───────────────────────────── server ─────────────────────────────
class Server:
    def __init__(self, name, spec, cwd="."):
        self.name, self.spec, self.cwd = name, spec, cwd
        self.transport, self.tools_list, self.error, self.info = None, [], None, {}
        self._id = 0

    def _rpc(self, method, params=None, timeout=60):
        self._id += 1
        m = self.transport.send({"jsonrpc": "2.0", "id": self._id, "method": method, "params": params or {}},
                                self._id, timeout)
        if "error" in m:
            e = m["error"]
            raise MCPError(f"{method}: {e.get('message', e) if isinstance(e, dict) else e}")
        r = m.get("result")
        return r if isinstance(r, dict) else {}

    def connect(self):
        """Handshake + tool listing. Raises MCPError; the reason is also kept in `self.error`."""
        try:
            if self.spec.get("url"):
                self.transport = _Http(self.spec)
            elif isinstance(self.spec.get("command"), str) and self.spec["command"]:
                self.transport = _Stdio(self.spec, self.cwd)
            else:
                raise MCPError("needs either `url` or `command`")
            r = self._rpc("initialize", {"protocolVersion": PROTOCOL, "capabilities": {}, "clientInfo": CLIENT})
            self.info = r.get("serverInfo") or {}
            self.transport.send({"jsonrpc": "2.0", "method": "notifications/initialized"})
            tools, cursor = [], None
            for _ in range(MAX_PAGES):
                r = self._rpc("tools/list", {"cursor": cursor} if cursor else {})
                got = r.get("tools")
                tools += [t for t in got if isinstance(t, dict) and isinstance(t.get("name"), str)] \
                    if isinstance(got, list) else []
                nxt = r.get("nextCursor")
                if not nxt or nxt == cursor:
                    break
                cursor = nxt
            self.tools_list = tools
            self.error = None
        except MCPError as e:
            self.error = str(e)
            self.close()
            raise
        return self

    def enabled_tools(self):
        toggles = self.spec.get("tools") or {}
        only = self.spec.get("only")
        return [t for t in self.tools_list
                if toggles.get(t["name"], True) and (only is None or t["name"] in only)]

    def call(self, tool, arguments, timeout=300):
        r = self._rpc("tools/call", {"name": tool, "arguments": arguments or {}}, timeout)
        parts = []
        content = r.get("content")
        for c in content if isinstance(content, list) else []:
            if not isinstance(c, dict):
                continue
            if c.get("type") == "text":
                parts.append(c.get("text", ""))
            elif c.get("type") == "resource" and (c.get("resource") or {}).get("text"):
                parts.append(c["resource"]["text"])
            else:
                parts.append(f"[{c.get('type', 'content')} omitted]")
        text = "\n".join(str(p) for p in parts) or json.dumps(r.get("structuredContent") or "(empty result)")
        if len(text) > MAX_RESULT:
            text = text[:MAX_RESULT] + f"\n[… {len(text) - MAX_RESULT} more characters cut]"
        return ("error: " + text) if r.get("isError") else text

    def close(self):
        if self.transport:
            self.transport.close()
            self.transport = None


def slim(schema, keep=160, depth=0):
    """Shrink a JSON schema for the prompt: keep names, types, enums and required, and only a short
    description on top-level parameters. Nested objects (Stitch's design-system tokens run to 14 KB)
    lose their prose; the server still validates whatever the model sends."""
    if isinstance(schema, list):
        return [slim(x, keep, depth) for x in schema]
    if not isinstance(schema, dict):
        return schema
    out = {}
    for k, v in schema.items():
        if k.startswith("x-"):                      # vendor extensions (enum prose etc.)
            continue
        if k in ("title", "examples", "example", "$comment", "$schema", "additionalProperties", "format",
                 "minLength", "maxLength", "pattern", "default"):
            continue
        if k == "description":
            if depth <= 2 and isinstance(v, str):
                out[k] = v.strip().split("\n")[0][:keep]
            continue
        if k in ("properties",) and isinstance(v, dict):
            out[k] = {n: slim(x, keep, depth + 1) for n, x in v.items()}
        elif k in ("items", "anyOf", "oneOf", "allOf", "$defs", "definitions"):
            if k in ("$defs", "definitions") and isinstance(v, dict):
                out[k] = {n: slim(x, keep, depth + 2) for n, x in v.items()}
            else:
                out[k] = slim(v, keep, depth + 1)
        else:
            out[k] = v
    return out


def schema_tokens(tool_map):
    """Rough prompt cost (tokens) of a {name: (schema, fn)} map: ~3 characters per token."""
    return sum(len(json.dumps(sch)) for sch, _ in tool_map.values()) // 3


def _safe(s):
    return re.sub(r"[^a-zA-Z0-9_-]", "_", s)


def tool_name(server, tool):
    return f"{_safe(server)}__{_safe(tool)}"[:64]


def load_servers(root, only=None, path=None):
    """Connect every enabled server (or just `only`). Returns ([Server…], {name: error})."""
    servers, failed = [], {}
    for name, spec in load_config(root, path).items():
        if not spec["enabled"] or (only and name not in only):
            continue
        s = Server(name, spec, cwd=root)
        try:
            servers.append(s.connect())
        except MCPError as e:
            failed[name] = str(e)
    return servers, failed


def tools(servers, allow=lambda what: True):
    """{name: (OpenAI tool schema, callable)} for the enabled tools of connected servers."""
    out = {}
    for s in servers:
        for t in s.enabled_tools():
            name = tool_name(s.name, t["name"])
            base, n = name, 1
            while name in out:                              # two names that sanitise (or truncate) the same
                n += 1
                name = f"{base[:60]}_{n}"

            def call(_s=s, _t=t["name"], **args):
                if not allow(f"call {_s.name}: {_t}"):
                    return "user declined"
                try:
                    return _s.call(_t, args)
                except MCPError as e:
                    return f"error: {e}"
            schema = t.get("inputSchema")
            if not isinstance(schema, dict) or schema.get("type") != "object":
                schema = {"type": "object", "properties": {}}
            schema = slim(schema)
            desc = (t.get("description") or t.get("title") or t["name"]).strip().replace("\n\n", "\n")
            out[name] = ({"type": "function", "function": {
                "name": name, "description": f"[{s.name}] {desc[:300]}", "parameters": schema}}, call)
    return out


def status(root, servers, failed):
    """Lines describing every configured server: connected / failed / off, with tool counts."""
    lines, live = [], {s.name: s for s in servers}
    cost = schema_tokens(tools(servers))
    for name, spec in load_config(root).items():
        if name in live:
            s = live[name]
            on = len(s.enabled_tools())
            lines.append(f"✓ {name}  {s.info.get('name', '')} {s.info.get('version', '')}".rstrip()
                         + f"  — {on}/{len(s.tools_list)} tools on")
            if s is servers[-1] and cost:
                lines.append(f"  ≈ {cost} tokens of tool definitions in every request")
        elif name in failed:
            lines.append(f"✗ {name}  — {failed[name]}")
        else:
            lines.append(f"· {name}  — {'off' if not spec['enabled'] else 'not connected'}")
    return lines
