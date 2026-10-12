"""The editor UI's agent loop over HTTP: scripted fake engine, real project folder, real approvals.

    .venv/bin/python -m unittest tests.test_ui_agent -v
"""
import json
import os
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from ui import agentloop  # noqa: E402
from ui.server import Project, UIServer  # noqa: E402


def text(s):
    return {"choices": [{"delta": {"content": s}}]}


def call(i, name, args):
    return {"choices": [{"delta": {"tool_calls": [{"id": f"c{i}", "type": "function",
                                                   "function": {"name": name, "arguments": json.dumps(args)}}]}}]}


class ScriptedEngine(BaseHTTPRequestHandler):
    script = []
    seen = []

    def log_message(self, *a):
        pass

    def do_POST(self):
        ScriptedEngine.seen.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
        events = ScriptedEngine.script.pop(0) if ScriptedEngine.script else [text("(nothing scripted)")]
        body = "".join(f"data: {json.dumps(e)}\n\n" for e in events).encode() + b"data: [DONE]\n\n"
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = ThreadingHTTPServer(("127.0.0.1", 0), ScriptedEngine)
        threading.Thread(target=cls.engine.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.engine.shutdown()
        cls.engine.server_close()

    def setUp(self):
        ScriptedEngine.script, ScriptedEngine.seen = [], []
        self.root = os.path.realpath(tempfile.mkdtemp())
        os.makedirs(os.path.join(self.root, "src"))
        with open(os.path.join(self.root, "src", "app.py"), "w") as f:
            f.write("def run():\n    return 1\n")
        with open(os.path.join(self.root, "README.md"), "w") as f:
            f.write("# Demo\n")
        self.srv = UIServer(Project(self.root), 0, self.engine.server_address[1], token="t")
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.base = f"http://127.0.0.1:{self.srv.server_address[1]}"
        self.addCleanup(lambda: (self.srv.shutdown(), self.srv.server_close()))

    def post(self, path, body):
        req = urllib.request.Request(self.base + path, method="POST", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json", "X-UI-Token": "t"})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.status, json.load(r)
        except urllib.error.HTTPError as e:
            data = json.loads(e.read())
            e.close()
            return e.code, data

    def agent(self, prompt="hi", react=None, **extra):
        """Run the agent; `react(event, run_id)` may answer approvals/questions. Returns the event list."""
        body = {"messages": [{"role": "user", "content": prompt}], **extra}
        req = urllib.request.Request(self.base + "/api/agent", method="POST", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json", "X-UI-Token": "t"})
        events, run_id = [], None
        with urllib.request.urlopen(req, timeout=60) as r:
            for raw in r:
                line = raw.strip()
                if not line.startswith(b"data:"):
                    continue
                ev = json.loads(line[5:])
                events.append(ev)
                if ev["type"] == "run":
                    run_id = ev["id"]
                if react:
                    react(ev, run_id)
        return events

    def answer(self, run_id, kind, id_, **kw):
        return self.post("/api/agent/answer", {"run": run_id, "kind": kind, "id": id_, **kw})

    def read(self, rel):
        with open(os.path.join(self.root, rel)) as f:
            return f.read()


class TestExploring(Base):
    def test_search_and_read_then_answer(self):
        ScriptedEngine.script = [[text("Let me look. "), call(1, "search", {"pattern": "def run"})],
                                 [call(2, "read_file", {"path": "src/app.py"})],
                                 [text("run returns 1.")]]
        ev = self.agent("what does run return?")
        types = [e["type"] for e in ev]
        self.assertEqual(types[0], "run")
        self.assertEqual(types[-1], "done")
        tools = [(e["name"], e["category"]) for e in ev if e["type"] == "tool"]
        self.assertEqual(tools, [("search", "search"), ("read_file", "read")])
        self.assertTrue(all(e["ok"] for e in ev if e["type"] == "tool_result"))
        self.assertEqual("".join(e["text"] for e in ev if e["type"] == "delta"), "Let me look. run returns 1.")
        # the model really received the tool results
        third = ScriptedEngine.seen[2]["messages"]
        results = [m["content"] for m in third if m["role"] == "tool"]
        self.assertIn("src/app.py:1: def run():", results[0])
        self.assertIn("    1  def run():", results[1])
        self.assertTrue(ScriptedEngine.seen[0]["tools"])
        self.assertIn("ask_user", [t["function"]["name"] for t in ScriptedEngine.seen[0]["tools"]])

    def test_list_dir_and_a_path_outside_the_project(self):
        ScriptedEngine.script = [[call(1, "list_dir", {"path": "."})],
                                 [call(2, "read_file", {"path": "../../etc/passwd"})], [text("done")]]
        ev = self.agent("look around")
        results = [e for e in ev if e["type"] == "tool_result"]
        self.assertTrue(results[0]["ok"])
        self.assertFalse(results[1]["ok"])
        sent = [m["content"] for m in ScriptedEngine.seen[2]["messages"] if m["role"] == "tool"]
        self.assertIn("outside the project", sent[1])
        self.assertNotIn("root:", sent[1])

    def test_attachments_reach_the_model(self):
        ScriptedEngine.script = [[text("ok")]]
        self.agent("explain", attachments=["src/app.py"])
        user = ScriptedEngine.seen[0]["messages"][-1]["content"]
        self.assertIn("File `src/app.py`", user)
        self.assertIn("def run():", user)

    def test_system_prompt_names_the_project_and_tools_policy(self):
        ScriptedEngine.script = [[text("ok")]]
        self.agent("x")
        system = ScriptedEngine.seen[0]["messages"][0]["content"]
        self.assertIn(os.path.basename(self.root), system)
        self.assertIn("ask_user", system)


class TestQuestions(Base):
    def test_the_agent_asks_and_continues_with_the_answer(self):
        ScriptedEngine.script = [[call(1, "ask_user", {"question": "Which file?", "options": ["app.py", "README.md", "other"]})],
                                 [text("Using README.md.")]]
        seen = {}

        def react(ev, run_id):
            if ev["type"] == "question":
                seen["q"] = ev
                self.assertEqual(self.answer(run_id, "question", ev["id"], text="README.md")[0], 200)
        ev = self.agent("pick one", react=react)
        self.assertEqual(seen["q"]["question"], "Which file?")
        self.assertEqual(seen["q"]["options"], ["app.py", "README.md", "other"])
        self.assertEqual(ev[-1]["type"], "done")
        tool_msg = [m for m in ScriptedEngine.seen[1]["messages"] if m["role"] == "tool"][0]
        self.assertEqual(tool_msg["content"], "README.md")

    def test_skipping_a_question_is_reported_to_the_model(self):
        ScriptedEngine.script = [[call(1, "ask_user", {"question": "Sure?", "options": ["yes", "no"]})], [text("ok")]]
        self.agent("go", react=lambda ev, rid: ev["type"] == "question" and self.answer(rid, "question", ev["id"], text=""))
        tool_msg = [m for m in ScriptedEngine.seen[1]["messages"] if m["role"] == "tool"][0]
        self.assertIn("skipped", tool_msg["content"])

    def test_options_are_capped_at_four(self):
        ScriptedEngine.script = [[call(1, "ask_user", {"question": "?", "options": list("abcdefg")})], [text("ok")]]
        got = []
        self.agent("go", react=lambda ev, rid: (ev["type"] == "question" and (got.append(ev["options"]),
                                                                                  self.answer(rid, "question", ev["id"], text="a"))))
        self.assertEqual(got[0], ["a", "b", "c", "d"])


class TestEditsAndCommands(Base):
    EDIT = {"path": "src/app.py", "old": "return 1", "new": "return 2"}

    def approve(self, yes):
        return lambda ev, rid: ev["type"] == "approval" and self.answer(rid, "approval", ev["id"], approve=yes)

    def test_an_approved_edit_is_applied_after_the_approval(self):
        ScriptedEngine.script = [[call(1, "edit_file", self.EDIT)], [text("done")]]
        events = []
        ev = self.agent("change it", react=lambda e, rid: (events.append(self.read("src/app.py")) if e["type"] == "approval" else None,
                                                            self.approve(True)(e, rid)))
        self.assertEqual(events, ["def run():\n    return 1\n"])             # untouched while waiting for the answer
        appr = [e for e in ev if e["type"] == "approval"][0]
        self.assertEqual((appr["kind"], appr["path"], appr["old"], appr["new"]), ("edit", "src/app.py", "return 1", "return 2"))
        self.assertEqual(self.read("src/app.py"), "def run():\n    return 2\n")

    def test_a_rejected_edit_changes_nothing_and_the_model_is_told(self):
        ScriptedEngine.script = [[call(1, "edit_file", self.EDIT)], [text("ok, left it")]]
        self.agent("change it", react=self.approve(False))
        self.assertEqual(self.read("src/app.py"), "def run():\n    return 1\n")
        tool_msg = [m for m in ScriptedEngine.seen[1]["messages"] if m["role"] == "tool"][0]
        self.assertIn("rejected", tool_msg["content"])

    def test_auto_edit_skips_the_approval_but_runs_still_ask(self):
        ScriptedEngine.script = [[call(1, "edit_file", self.EDIT)], [call(2, "run", {"command": "echo hi"})], [text("ok")]]
        kinds = []

        def react(e, rid):
            if e["type"] == "approval":
                kinds.append(e["kind"])
                self.answer(rid, "approval", e["id"], approve=True)
        self.agent("go", react=react, auto_edit=True)
        self.assertEqual(self.read("src/app.py"), "def run():\n    return 2\n")
        self.assertEqual(kinds, ["run"])

    def test_a_snippet_that_does_not_match_uniquely_fails_without_asking(self):
        ScriptedEngine.script = [[call(1, "edit_file", {"path": "src/app.py", "old": "nope", "new": "x"})], [text("hm")]]
        ev = self.agent("go")
        self.assertFalse([e for e in ev if e["type"] == "approval"])
        self.assertFalse([e for e in ev if e["type"] == "tool_result"][0]["ok"])

    def test_write_file_shows_old_and_new_and_creates_directories(self):
        ScriptedEngine.script = [[call(1, "write_file", {"path": "new/dir/x.txt", "content": "hello\n"})], [text("ok")]]
        got = []
        self.agent("make it", react=lambda e, rid: (got.append(e), self.approve(True)(e, rid))[1] if e["type"] == "approval" else None)
        self.assertEqual(got[0]["kind"], "write")
        self.assertEqual(self.read("new/dir/x.txt"), "hello\n")

    def test_commands_need_approval_and_dangerous_ones_are_flagged(self):
        ScriptedEngine.script = [[call(1, "run", {"command": "git push origin main"})], [text("ok")]]
        got = []
        self.agent("push", react=lambda e, rid: (got.append(e), self.approve(False)(e, rid))[1] if e["type"] == "approval" else None)
        self.assertEqual(got[0]["kind"], "run")
        self.assertTrue(got[0]["dangerous"])
        tool_msg = [m for m in ScriptedEngine.seen[1]["messages"] if m["role"] == "tool"][0]
        self.assertIn("declined", tool_msg["content"])

    def test_an_approved_command_runs_in_the_project(self):
        ScriptedEngine.script = [[call(1, "run", {"command": "cat README.md; exit 3"})], [text("ok")]]
        self.agent("run", react=self.approve(True))
        tool_msg = [m for m in ScriptedEngine.seen[1]["messages"] if m["role"] == "tool"][0]
        self.assertTrue(tool_msg["content"].startswith("exit code 3"))
        self.assertIn("# Demo", tool_msg["content"])

    def test_edit_outside_the_project_is_refused_before_any_approval(self):
        ScriptedEngine.script = [[call(1, "write_file", {"path": "../escape.txt", "content": "x"})], [text("ok")]]
        ev = self.agent("go")
        self.assertFalse([e for e in ev if e["type"] == "approval"])
        self.assertFalse(os.path.exists(os.path.join(os.path.dirname(self.root), "escape.txt")))


class TestRunLifecycle(Base):
    def test_closing_the_connection_while_waiting_cancels_the_run(self):
        ScriptedEngine.script = [[call(1, "edit_file", TestEditsAndCommands.EDIT)], [text("never")]]
        body = json.dumps({"messages": [{"role": "user", "content": "go"}]}).encode()
        req = urllib.request.Request(self.base + "/api/agent", method="POST", data=body,
                                     headers={"Content-Type": "application/json", "X-UI-Token": "t"})
        r = urllib.request.urlopen(req, timeout=30)
        for raw in r:
            if b'"approval"' in raw:
                break
        r.close()                                                          # the browser tab went away
        deadline = time.time() + 15
        while agentloop.RUNS and time.time() < deadline:
            time.sleep(0.2)
        self.assertEqual(agentloop.RUNS, {})
        self.assertEqual(self.read("src/app.py"), "def run():\n    return 1\n")

    def test_answering_an_unknown_run_is_a_404(self):
        self.assertEqual(self.answer("deadbeef", "approval", "x", approve=True)[0], 404)

    def test_a_stopped_model_is_an_error_event(self):
        srv = UIServer(Project(self.root), 0, 1, token="t")
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        self.addCleanup(lambda: (srv.shutdown(), srv.server_close()))
        req = urllib.request.Request(f"http://127.0.0.1:{srv.server_address[1]}/api/agent", method="POST",
                                     data=json.dumps({"messages": [{"role": "user", "content": "x"}]}).encode(),
                                     headers={"Content-Type": "application/json", "X-UI-Token": "t"})
        with urllib.request.urlopen(req, timeout=15) as r:
            events = [json.loads(l[5:]) for l in r if l.strip().startswith(b"data:")]
        self.assertEqual(events[-1]["type"], "error")
        self.assertIn("not running", events[-1]["message"])

    def test_a_model_that_never_stops_calling_tools_is_cut_off(self):
        ScriptedEngine.script = [[call(i, "list_dir", {"path": "."})] for i in range(10)]
        with mock.patch.object(agentloop, "MAX_STEPS", 3):
            ev = self.agent("loop")
        self.assertEqual(ev[-1]["type"], "error")
        self.assertIn("3 steps", ev[-1]["message"])
        self.assertEqual(len(ScriptedEngine.seen), 3)

    def test_a_bad_request_is_a_400(self):
        self.assertEqual(self.post("/api/agent", {"messages": []})[0], 400)


class TestActiveFile(Base):
    """"check this code": the open file reaches the agent's system prompt (and only a real project file does)."""

    def system_of_last_request(self):
        return ScriptedEngine.seen[-1]["messages"][0]["content"]

    def test_the_open_file_and_line_are_named_in_the_system_prompt(self):
        ScriptedEngine.script = [[text("ok")]]
        self.agent("проверь код", active={"file": "src/app.py", "line": 2})
        sys_prompt = self.system_of_last_request()
        self.assertIn("`src/app.py` open in the editor, the cursor on line 2", sys_prompt)
        self.assertIn("read it with read_file first", sys_prompt)

    def test_nothing_is_added_for_a_missing_or_foreign_file(self):
        for active in ({"file": "nope.py"}, {"file": "../../etc/passwd"}, {"file": "/etc/passwd"}, {"file": "src"},
                       {"file": 5}, {"line": 3}, "src/app.py", {}):
            ScriptedEngine.script = [[text("ok")]]
            self.agent("hi", active=active)
            self.assertNotIn("open in the editor", self.system_of_last_request(), active)

    def test_an_absurd_line_number_is_left_out_but_the_file_is_kept(self):
        ScriptedEngine.script = [[text("ok")]]
        self.agent("hi", active={"file": "README.md", "line": -4})
        sys_prompt = self.system_of_last_request()
        self.assertIn("`README.md` open in the editor.", sys_prompt)
        self.assertNotIn("cursor", sys_prompt)

    def test_without_it_the_prompt_is_unchanged(self):
        ScriptedEngine.script = [[text("ok")]]
        self.agent("hi")
        self.assertNotIn("open in the editor", self.system_of_last_request())


class TestSearchEndpoint(Base):
    def get(self, path):
        req = urllib.request.Request(self.base + path, headers={"X-UI-Token": "t"})
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                return r.status, json.load(r)
        except urllib.error.HTTPError as e:
            data = json.loads(e.read())
            e.close()
            return e.code, data

    def test_search_hits_and_fallback_for_an_invalid_regex(self):
        code, r = self.get("/api/search?q=def%20run")
        self.assertEqual((code, r["hits"][0]["path"], r["hits"][0]["line"]), (200, "src/app.py", "1"))
        code, r = self.get("/api/search?q=def%20run(")           # not a valid regex: searched literally
        self.assertEqual(code, 200)
        self.assertEqual(self.get("/api/search?q=")[0], 400)
        self.assertEqual(self.get("/api/search?q=x&path=../..")[0], 403)
        self.assertEqual(self.get("/api/search?q=Demo&glob=*.md")[1]["hits"][0]["path"], "README.md")


if __name__ == "__main__":
    unittest.main()
