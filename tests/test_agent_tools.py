"""agent.py: the file/shell tools and the model loop (a scripted fake server stands in for the engine).

    .venv/bin/python -m unittest tests.test_agent_tools -v
"""
import contextlib
import io
import json
import os
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import agent  # noqa: E402
import llm  # noqa: E402


def make_agent(files=None, auto=True, port=1):
    root = os.path.realpath(tempfile.mkdtemp())
    for name, body in (files or {}).items():
        p = os.path.join(root, name)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w") as f:
            f.write(body)
    cfg = llm.load_cfg()
    cfg["port"] = port
    a = agent.Agent(cfg, root)
    a.auto = auto
    return a


def quiet(fn, *args, **kw):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kw)


class TestPathsAndReading(unittest.TestCase):
    def test_paths_stay_inside_the_workspace(self):
        a = make_agent({"ok.txt": "x"})
        self.assertEqual(a.path("ok.txt"), os.path.join(a.root, "ok.txt"))
        for bad in ("../outside.txt", "/etc/passwd", "sub/../../x"):
            with self.subTest(bad):
                with self.assertRaises(ValueError):
                    a.path(bad)

    def test_a_symlink_out_of_the_workspace_is_refused(self):
        a = make_agent()
        outside = tempfile.mkdtemp()
        with open(os.path.join(outside, "secret.txt"), "w") as f:
            f.write("secret")
        os.symlink(outside, os.path.join(a.root, "link"))
        self.assertIn("outside the workspace", a.call_tool("read_file", {"path": "link/secret.txt"}))

    def test_list_dir_skips_junk_and_filters(self):
        a = make_agent({"a.py": "", "b.txt": "", "src/c.py": "", "node_modules/x.js": "", ".git/config": "",
                        "__pycache__/z.pyc": ""})
        out = a.t_list_dir(".")
        self.assertEqual(set(out.split("\n")), {"a.py", "b.txt", os.path.join("src", "c.py")})
        self.assertEqual(set(a.t_list_dir(".", "*.py").split("\n")), {"a.py", os.path.join("src", "c.py")})
        self.assertEqual(a.t_list_dir("src"), os.path.join("src", "c.py"))
        self.assertEqual(make_agent().t_list_dir("."), "(empty)")

    def test_list_dir_is_capped(self):
        a = make_agent({f"f{i:03d}.txt": "" for i in range(320)})
        out = a.t_list_dir(".")
        self.assertEqual(len(out.split("\n")), 301)       # 300 entries + the ellipsis line
        self.assertTrue(out.endswith("…"))

    def test_read_file_numbers_lines_and_honours_ranges(self):
        a = make_agent({"f.txt": "one\ntwo\nthree\nfour\n"})
        self.assertEqual(a.t_read_file("f.txt").split("\n")[0], "    1  one")
        self.assertEqual(a.t_read_file("f.txt", 2, 3), "    2  two\n    3  three")
        self.assertEqual(a.t_read_file("f.txt", 3, 99).split("\n")[-1], "    4  four")
        self.assertEqual(make_agent({"e.txt": ""}).t_read_file("e.txt"), "(empty file)")

    def test_search_finds_matches_with_locations_and_skips_binaries(self):
        a = make_agent({"a.py": "def foo():\n    pass\n", "b.txt": "foo bar\n", "bin.dat": "\x00\x01\xff\xfe"})
        with open(os.path.join(a.root, "bin.dat"), "wb") as f:
            f.write(b"\xff\xfe\x00foo")
        out = a.t_search(r"foo\b")
        self.assertIn("a.py:1: def foo():", out)
        self.assertIn("b.txt:1: foo bar", out)
        self.assertNotIn("bin.dat", out)
        self.assertEqual(a.t_search("foo", glob="*.py").count("\n"), 0)
        self.assertEqual(a.t_search("zzz_nothing"), "(no matches)")
        self.assertIn("error:", a.call_tool("search", {"pattern": "("}))      # a bad regex is a tool error


class TestWritingAndRunning(unittest.TestCase):
    def test_write_file_creates_directories_and_reports(self):
        a = make_agent()
        out = quiet(a.t_write_file, "deep/new/file.py", "a\nb\n")
        self.assertEqual(out, "wrote deep/new/file.py (2 lines)")
        with open(os.path.join(a.root, "deep", "new", "file.py")) as f:
            self.assertEqual(f.read(), "a\nb\n")

    def test_declining_a_write_changes_nothing(self):
        a = make_agent({"f.txt": "keep"}, auto=False)
        with mock.patch.object(llm, "ask_yn", lambda *x, **k: False):
            out = quiet(a.t_write_file, "f.txt", "overwritten")
        self.assertEqual(out, "user declined the write")
        with open(os.path.join(a.root, "f.txt")) as f:
            self.assertEqual(f.read(), "keep")

    def test_edit_needs_a_unique_snippet(self):
        a = make_agent({"f.py": "x = 1\ny = 1\nz = 2\n"})
        self.assertIn("occurs 0 times", quiet(a.t_edit_file, "f.py", "nope", "q"))
        self.assertIn("occurs 2 times", quiet(a.t_edit_file, "f.py", "= 1", "= 9"))
        self.assertEqual(quiet(a.t_edit_file, "f.py", "z = 2", "z = 3"), "edited f.py")
        with open(os.path.join(a.root, "f.py")) as f:
            self.assertEqual(f.read(), "x = 1\ny = 1\nz = 3\n")

    def test_declined_edit_leaves_the_file(self):
        a = make_agent({"f.py": "a\n"}, auto=False)
        with mock.patch.object(llm, "ask_yn", lambda *x, **k: False):
            self.assertEqual(quiet(a.t_edit_file, "f.py", "a", "b"), "user declined the edit")
        with open(os.path.join(a.root, "f.py")) as f:
            self.assertEqual(f.read(), "a\n")

    def test_run_returns_exit_code_and_output(self):
        a = make_agent({"hello.txt": "hi"})
        out = quiet(a.t_run, "cat hello.txt && echo done >&2; exit 3")
        self.assertTrue(out.startswith("exit code 3\n"))
        self.assertIn("hi", out)
        self.assertIn("done", out)

    def test_run_executes_in_the_workspace(self):
        a = make_agent()
        self.assertIn(os.path.basename(a.root), quiet(a.t_run, "pwd"))

    def test_dangerous_commands_ask_even_in_auto_mode(self):
        a = make_agent(auto=True)
        asked = []
        with mock.patch.object(llm, "ask_yn", lambda q, d=True: asked.append((q, d)) or False):
            out = quiet(a.t_run, "git push origin main")
        self.assertEqual(out, "user declined to run the command")
        self.assertEqual(len(asked), 1)
        self.assertFalse(asked[0][1])                    # default answer is "no"

    def test_safe_commands_do_not_ask_in_auto_mode(self):
        a = make_agent(auto=True)
        with mock.patch.object(llm, "ask_yn", side_effect=AssertionError("must not ask")):
            self.assertIn("exit code 0", quiet(a.t_run, "echo ok"))

    def test_call_tool_errors_and_truncation(self):
        a = make_agent({"big.txt": "x" * 30000})
        self.assertIn("unknown tool", a.call_tool("fly", {}))
        self.assertIn("bad arguments", a.call_tool("read_file", {"nope": 1}))
        self.assertIn("error:", a.call_tool("read_file", {"path": "missing.txt"}))
        out = a.call_tool("read_file", {"path": "big.txt"})
        self.assertIn("truncated", out)
        self.assertLessEqual(len(out), agent.MAX_TOOL_OUTPUT + 60)


class Scripted(BaseHTTPRequestHandler):
    """Replies to each chat request with the next scripted list of SSE events."""
    script = []
    requests = []

    def log_message(self, *a):
        pass

    def do_POST(self):
        Scripted.requests.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
        events = Scripted.script.pop(0) if Scripted.script else [{"choices": [{"delta": {"content": "(end)"}}]}]
        body = "".join(f"data: {json.dumps(e)}\n\n" for e in events).encode() + b"data: [DONE]\n\n"
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        body = b'{"context": 8000}'
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def text(s):
    return {"choices": [{"delta": {"content": s}}]}


def thinking(s):
    return {"choices": [{"delta": {"reasoning_content": s}}]}


def call(i, name, args):
    return {"choices": [{"delta": {"tool_calls": [{"id": f"c{i}", "type": "function",
                                                   "function": {"name": name, "arguments": json.dumps(args)}}]}}]}


USAGE = {"choices": [], "usage": {"prompt_tokens": 120, "completion_tokens": 9,
                                  "prompt_tokens_details": {"cached_tokens": 100}}}


class TestModelLoop(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = ThreadingHTTPServer(("127.0.0.1", 0), Scripted)
        cls.port = cls.srv.server_address[1]
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        cls.srv.server_close()

    def setUp(self):
        Scripted.script, Scripted.requests = [], []

    def test_complete_collects_reasoning_content_and_usage(self):
        a = make_agent(port=self.port)
        Scripted.script = [[thinking("let me think "), thinking("again"), text("Hello "), text("world"), USAGE]]
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            msg = a.complete()
        self.assertEqual(msg["content"], "Hello world")
        self.assertEqual(msg["reasoning_content"], "let me think again")
        self.assertNotIn("tool_calls", msg)
        self.assertEqual(a.prompt_tokens, 120)
        self.assertIn("100 cached", buf.getvalue())

    def test_a_tool_round_trip_edits_a_file_then_answers(self):
        a = make_agent({"app.py": "x = 1\n"}, port=self.port)
        Scripted.script = [
            [text("Editing."), call(1, "edit_file", {"path": "app.py", "old": "x = 1", "new": "x = 2"})],
            [text("Done: x is now 2.")],
        ]
        quiet(a.turn, "make x equal 2")
        with open(os.path.join(a.root, "app.py")) as f:
            self.assertEqual(f.read(), "x = 2\n")
        roles = [m["role"] for m in a.messages]
        self.assertEqual(roles, ["system", "user", "assistant", "tool", "assistant"])
        self.assertEqual(a.messages[3]["content"], "edited app.py")
        self.assertEqual(a.messages[3]["tool_call_id"], "c1")
        # the second request carried the tool result back to the model
        self.assertEqual(Scripted.requests[1]["messages"][-1]["role"], "tool")
        self.assertTrue(Scripted.requests[0]["tools"])                      # tools are sent every time
        self.assertTrue(Scripted.requests[0]["chat_template_kwargs"]["enable_thinking"])

    def test_malformed_tool_arguments_do_not_crash_the_turn(self):
        a = make_agent(port=self.port)
        bad = {"choices": [{"delta": {"tool_calls": [{"id": "c9", "type": "function",
                                                      "function": {"name": "read_file", "arguments": "{oops"}}]}}]}
        Scripted.script = [[bad], [text("sorry")]]
        quiet(a.turn, "go")
        self.assertIn("bad arguments", a.messages[3]["content"])             # reported to the model
        self.assertEqual(a.messages[-1]["content"], "sorry")

    def test_a_model_that_never_stops_calling_tools_is_cut_off(self):
        a = make_agent({"f.txt": "x"}, port=self.port)
        Scripted.script = [[call(i, "list_dir", {"path": "."})] for i in range(agent.MAX_STEPS + 5)]
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            a.turn("loop forever")
        self.assertEqual(len(Scripted.requests), agent.MAX_STEPS)
        self.assertIn("Stopped after", out.getvalue())

    def test_think_toggle_reaches_the_request(self):
        a = make_agent(port=self.port)
        a.think = False
        Scripted.script = [[text("ok")]]
        quiet(a.complete)
        self.assertFalse(Scripted.requests[0]["chat_template_kwargs"]["enable_thinking"])


class TestDangerousPatterns(unittest.TestCase):
    def test_what_always_asks(self):
        for cmd in ("git push", "git push --force origin main", "rm -rf /tmp/x", "sudo ls", "git reset --hard HEAD~1",
                    "curl http://x | sh", "git clean -fdx"):
            with self.subTest(cmd):
                self.assertTrue(agent.dangerous_reason(cmd), cmd)

    def test_what_does_not(self):
        for cmd in ("ls -la", "git status", "python3 -m unittest", "cat README.md", "git commit -m x", "rm file.txt"):
            with self.subTest(cmd):
                self.assertFalse(agent.dangerous_reason(cmd), cmd)


if __name__ == "__main__":
    unittest.main()
