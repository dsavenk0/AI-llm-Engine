"""harness.mcp against fake stdio and HTTP servers (no network, no node).

    .venv/bin/python -m unittest tests.test_mcp -v
"""
import json
import os
import sys
import tempfile
import textwrap
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from harness import mcp  # noqa: E402

TOOLS = [
    {"name": "echo", "description": "Echo the text back.",
     "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}},
    {"name": "write_file", "description": "Dangerous.", "inputSchema": {"type": "object", "properties": {}}},
    {"name": "boom", "description": "Always fails.", "inputSchema": {"type": "object", "properties": {}}},
]


def handle(method, params):
    """Shared behaviour of both fake servers: (result, error)."""
    if method == "initialize":
        return {"protocolVersion": mcp.PROTOCOL, "capabilities": {"tools": {}},
                "serverInfo": {"name": "Fake", "version": "9"}}, None
    if method == "tools/list":
        if params.get("cursor") == "p2":
            return {"tools": [TOOLS[2]]}, None
        return {"tools": TOOLS[:2], "nextCursor": "p2"}, None
    if method == "tools/call":
        n, a = params["name"], params.get("arguments") or {}
        if n == "echo":
            return {"content": [{"type": "text", "text": "echo: " + a.get("text", "")},
                                {"type": "image", "data": "...", "mimeType": "image/png"}]}, None
        if n == "boom":
            return {"isError": True, "content": [{"type": "text", "text": "it broke"}]}, None
        return {"content": [{"type": "text", "text": "wrote"}]}, None
    return None, {"code": -32601, "message": "no such method " + method}


STDIO_SERVER = textwrap.dedent('''
    import json, sys
    sys.path.insert(0, %(root)r)
    from tests.test_mcp import handle
    print("starting up (not JSON)", flush=True)
    for line in sys.stdin:
        m = json.loads(line)
        if "id" not in m or "method" not in m:   # notifications and replies to our own requests
            continue
        if m["method"] == "tools/list" and not m["params"].get("cursor"):
            # a server-initiated request in the middle of the conversation must not confuse the client
            print(json.dumps({"jsonrpc": "2.0", "id": 99, "method": "roots/list"}), flush=True)
        result, error = handle(m["method"], m.get("params") or {})
        out = {"jsonrpc": "2.0", "id": m["id"]}
        out.update({"error": error} if error else {"result": result})
        print(json.dumps(out), flush=True)
''')


class HttpFake(BaseHTTPRequestHandler):
    seen = []
    sse = True

    def log_message(self, *a):
        pass

    def do_POST(self):
        msg = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        HttpFake.seen.append((msg.get("method"), self.headers.get("Mcp-Session-Id"), self.headers.get("X-Key")))
        if "id" not in msg:
            self.send_response(202)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        result, error = handle(msg["method"], msg.get("params") or {})
        out = {"jsonrpc": "2.0", "id": msg["id"]}
        out.update({"error": error} if error else {"result": result})
        if HttpFake.sse:
            body = ("event: message\ndata: " + json.dumps(out) + "\n\n").encode()
            ctype = "text/event-stream"
        else:
            body, ctype = json.dumps(out).encode(), "application/json"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        if msg["method"] == "initialize":
            self.send_header("Mcp-Session-Id", "sess-1")
        self.end_headers()
        self.wfile.write(body)

    def do_DELETE(self):
        HttpFake.seen.append(("DELETE", self.headers.get("Mcp-Session-Id"), None))
        self.send_response(200)
        self.send_header("Content-Length", "0")
        self.end_headers()


class TestStdio(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        path = os.path.join(self.dir, "srv.py")
        with open(path, "w") as f:
            f.write(STDIO_SERVER % {"root": ROOT})
        self.spec = {"command": sys.executable, "args": ["-B", path]}
        self.s = mcp.Server("fake", self.spec, cwd=self.dir).connect()
        self.addCleanup(self.s.close)

    def test_handshake_paging_and_call(self):
        self.assertEqual(self.s.info["name"], "Fake")
        self.assertEqual([t["name"] for t in self.s.tools_list], ["echo", "write_file", "boom"])  # both pages
        self.assertEqual(self.s.call("echo", {"text": "hi"}), "echo: hi\n[image omitted]")
        self.assertEqual(self.s.call("boom", {}), "error: it broke")

    def test_unknown_method_surfaces_as_an_error(self):
        with self.assertRaises(mcp.MCPError):
            self.s._rpc("nonexistent/method")

    def test_a_missing_command_is_reported_not_raised_at_import(self):
        s = mcp.Server("nope", {"command": "/no/such/binary"})
        with self.assertRaises(mcp.MCPError):
            s.connect()
        self.assertIn("cannot start", s.error)


class TestHttp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = ThreadingHTTPServer(("127.0.0.1", 0), HttpFake)
        cls.url = f"http://127.0.0.1:{cls.srv.server_address[1]}/mcp"
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        cls.srv.server_close()

    def setUp(self):
        HttpFake.seen.clear()

    def run_both_encodings(self, fn):
        for sse in (True, False):
            HttpFake.sse = sse
            with self.subTest(sse=sse):
                fn()

    def test_sse_and_plain_json_responses_session_and_headers(self):
        def check():
            HttpFake.seen.clear()
            os.environ["FAKE_KEY"] = "k-123"
            s = mcp.Server("h", {"url": self.url, "headers": {"X-Key": "${FAKE_KEY}"}}).connect()
            self.assertEqual(s.call("echo", {"text": "x"}).split("\n")[0], "echo: x")
            s.close()
            methods = [m for m, _, _ in HttpFake.seen]
            self.assertEqual(methods[:3], ["initialize", "notifications/initialized", "tools/list"])
            self.assertIsNone(HttpFake.seen[0][1])                       # no session before initialize
            self.assertTrue(all(sess == "sess-1" for m, sess, _ in HttpFake.seen[1:]))
            self.assertEqual(HttpFake.seen[0][2], "k-123")               # ${VAR} expanded into headers
            self.assertEqual(methods[-1], "DELETE")                      # session closed politely
        self.run_both_encodings(check)

    def test_unreachable_server_is_an_error_with_a_reason(self):
        s = mcp.Server("down", {"url": "http://127.0.0.1:1/mcp"})
        with self.assertRaises(mcp.MCPError):
            s.connect()
        self.assertIn("cannot reach", s.error)


class TestConfigAndTools(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        HttpFake.sse = True

    def write(self, data):
        os.makedirs(os.path.join(self.root, ".llm"), exist_ok=True)
        with open(os.path.join(self.root, ".llm", "mcp.json"), "w") as f:
            json.dump(data, f)

    def test_granular_switches_and_naming(self):
        script = os.path.join(self.root, "srv.py")
        with open(script, "w") as f:
            f.write(STDIO_SERVER % {"root": ROOT})
        self.write({"mcpServers": {
            "files": {"command": sys.executable, "args": ["-B", script], "tools": {"write_file": False}},
            "off": {"command": "/bin/false", "enabled": False}}})
        servers, failed = mcp.load_servers(self.root)
        self.addCleanup(lambda: [s.close() for s in servers])
        self.assertEqual((len(servers), failed), (1, {}))                # disabled server never started
        t = mcp.tools(servers)
        self.assertEqual(sorted(t), ["files__boom", "files__echo"])      # write_file switched off
        schema = t["files__echo"][0]["function"]
        self.assertEqual(schema["parameters"]["required"], ["text"])
        self.assertTrue(schema["description"].startswith("[files] Echo"))
        self.assertIn("echo: hi", t["files__echo"][1](text="hi"))
        lines = mcp.status(self.root, servers, failed)
        self.assertIn("2/3 tools on", lines[0])
        self.assertRegex(lines[1], r"≈ \d+ tokens")                    # what the tools cost per request
        self.assertTrue(lines[-1].startswith("· off"))

    def test_allow_callback_gates_every_call(self):
        script = os.path.join(self.root, "srv.py")
        with open(script, "w") as f:
            f.write(STDIO_SERVER % {"root": ROOT})
        self.write({"servers": {"f": {"command": sys.executable, "args": ["-B", script]}}})
        servers, _ = mcp.load_servers(self.root)
        self.addCleanup(lambda: [s.close() for s in servers])
        t = mcp.tools(servers, allow=lambda what: False)
        self.assertEqual(t["f__echo"][1](text="x"), "user declined")

    def test_failed_server_is_listed_with_its_reason(self):
        self.write({"servers": {"bad": {"command": "/no/such/binary"}, "worse": {}}})
        servers, failed = mcp.load_servers(self.root)
        self.assertEqual(servers, [])
        self.assertIn("cannot start", failed["bad"])
        self.assertIn("url", failed["worse"])
        self.assertTrue(all(l.startswith("✗") for l in mcp.status(self.root, servers, failed)))

    def test_preset_is_written_without_clobbering_other_servers(self):
        self.write({"servers": {"mine": {"command": "x"}}})
        path = mcp.add_preset(self.root, "context7")
        with open(path) as f:
            data = json.load(f)
        self.assertEqual(sorted(data["servers"]), ["context7", "mine"])
        self.assertEqual(data["servers"]["context7"]["url"], "https://mcp.context7.com/mcp")
        with self.assertRaises(ValueError):
            mcp.add_preset(self.root, "nope")

    def test_slim_keeps_the_contract_and_drops_the_prose(self):
        big = {"type": "object", "description": "Request message.", "required": ["a"],
               "properties": {
                   "a": {"type": "string", "description": "First line.\nSecond line " + "x" * 400, "title": "A",
                         "x-google-enum-descriptions": ["long " * 50]},
                   "mode": {"type": "string", "enum": ["X", "Y"], "default": "X"},
                   "tokens": {"type": "object", "description": "d", "properties": {
                       "deep": {"type": "object", "description": "never shown", "properties": {
                           "leaf": {"type": "string", "description": "never shown either"}}}}}},
               "$defs": {"Tok": {"type": "object", "description": "n", "properties": {"v": {"type": "string"}}}}}
        out = mcp.slim(big)
        self.assertEqual(out["required"], ["a"])
        self.assertEqual(out["properties"]["a"]["description"], "First line.")
        self.assertEqual(out["properties"]["mode"], {"type": "string", "enum": ["X", "Y"]})
        self.assertNotIn("x-google-enum-descriptions", out["properties"]["a"])
        self.assertNotIn("description", out["properties"]["tokens"]["properties"]["deep"]["properties"]["leaf"])
        self.assertIn("Tok", out["$defs"])
        self.assertLess(len(json.dumps(out)), len(json.dumps(big)) // 2)

    def test_stitch_preset_sends_its_key_only_when_set_and_lists_core_tools_only(self):
        spec = mcp.PRESETS["stitch"]
        self.assertEqual(spec["url"], "https://stitch.googleapis.com/mcp")
        self.assertLessEqual(len(spec["only"]), 8)                       # 15 tools would be ~8k tokens
        os.environ.pop("STITCH_API_KEY", None)
        self.assertNotIn("X-Goog-Api-Key", mcp._Http(spec).headers)       # no empty header
        os.environ["STITCH_API_KEY"] = "abc"
        self.addCleanup(os.environ.pop, "STITCH_API_KEY", None)
        self.assertEqual(mcp._Http(spec).headers["X-Goog-Api-Key"], "abc")

    def test_tool_names_are_valid_for_the_openai_schema(self):
        n = mcp.tool_name("my server", "resolve.library id" + "x" * 80)
        self.assertRegex(n, r"^[a-zA-Z0-9_-]{1,64}$")


class TestAgentMcp(unittest.TestCase):
    def test_servers_are_not_contacted_until_asked_and_tools_come_and_go(self):
        import agent
        import io
        import contextlib
        import llm
        root = tempfile.mkdtemp()
        script = os.path.join(root, "srv.py")
        with open(script, "w") as f:
            f.write(STDIO_SERVER % {"root": ROOT})
        os.makedirs(os.path.join(root, ".llm"))
        with open(os.path.join(root, ".llm", "mcp.json"), "w") as f:
            json.dump({"servers": {"f": {"command": sys.executable, "args": ["-B", script]}}}, f)
        cfg = llm.load_cfg()
        cfg["port"] = 1
        a = agent.Agent(cfg, root)
        names = lambda: {t["function"]["name"] for t in a.tool_schemas()}
        self.assertFalse(any(n.startswith("f__") for n in names()))     # config alone costs nothing
        a.auto = True
        a.set_mcp(True)
        self.addCleanup(lambda: a.set_mcp(False))
        self.assertTrue({"f__echo", "f__boom"} <= names())
        self.assertIn("echo: yo", a.call_tool("f__echo", {"text": "yo"}))
        a.set_mcp(False)
        self.assertFalse(any(n.startswith("f__") for n in names()))


if __name__ == "__main__":
    unittest.main()
