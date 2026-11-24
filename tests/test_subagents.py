"""Sub-agents: definitions (harness/agents.py) and running them from the chat (agent.py) — inline for the model's
`run_agent`, in their own Terminal window for `/agents <name> <task>` (the window is replaced by a direct call).

    .venv/bin/python -m unittest tests.test_subagents -v
"""
import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("LLM_SESSIONS_DIR", tempfile.mkdtemp(prefix="llm-test-chats-"))  # never the real chats
import agent  # noqa: E402
import llm  # noqa: E402
from harness import agents, sessions  # noqa: E402


def write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)


class TestDefinitions(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.home = tempfile.mkdtemp()
        p = mock.patch.dict(os.environ, {"HOME": self.home, "LLM_AGENTS_DIRS": ""})
        p.start()
        self.addCleanup(p.stop)

    def test_builtins_are_there_and_a_project_one_overrides(self):
        found = agents.discover(self.root)
        self.assertTrue({"explorer", "reviewer", "fixer"} <= set(found))
        self.assertNotIn("write_file", found["explorer"].tools)
        self.assertIsNone(found["fixer"].tools)                          # all tools
        write(os.path.join(self.root, ".llm/agents/explorer.md"), "---\nname: explorer\ndescription: mine\n---\nX")
        self.assertEqual(agents.find("explorer", self.root).description, "mine")

    def test_claude_code_definitions_and_tool_names(self):
        write(os.path.join(self.root, ".claude/agents/doc-writer.md"),
              '---\nname: doc-writer\ndescription: "Writes docs"\ntools: Read, Grep, Glob, Bash(git diff:*), Write\n'
              'model: sonnet\n---\nYou write documentation.')
        d = agents.find("doc-writer", self.root)
        self.assertEqual(d.tools, ["read_file", "search", "list_dir", "run", "write_file"])
        self.assertEqual((d.scope, d.prompt), ("project", "You write documentation."))
        self.assertFalse(d.writable)                                      # another tool's folder: read only

    def test_save_delete_and_names(self):
        path = agents.save("api-checker", "Checks the API", "read_file, search", "Check it.", "project", self.root)
        self.assertTrue(path.endswith(".llm/agents/api-checker.md"))
        d = agents.find("api-checker", self.root)
        self.assertEqual((d.tools, d.writable), (["read_file", "search"], True))
        agents.delete("api-checker", self.root)
        self.assertIsNone(agents.find("api-checker", self.root))
        for bad in ("Bad Name", "1st", "stop", "result"):
            with self.subTest(bad), self.assertRaises(agents.AgentError):
                agents.save(bad, "d", None, "x", "project", self.root)
        with self.assertRaises(agents.AgentError):
            agents.delete("explorer", self.root)                          # built in
        with self.assertRaises(agents.AgentError):
            agents.save("x-y", "d", None, "x", "project", None)           # no project attached


def script(*replies):
    out = []
    for r in replies:
        if isinstance(r, tuple):
            ev = {"choices": [{"delta": {"tool_calls": [{"id": f"c{len(out)}", "type": "function",
                                                         "function": {"name": r[0], "arguments": json.dumps(r[1])}}]}}]}
        else:
            ev = {"choices": [{"delta": {"content": r}}]}
        out.append([b"data: " + json.dumps(ev).encode(), b"data: [DONE]"])
    return out


class Stream:
    def __init__(self, lines):
        self.lines = lines

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def __iter__(self):
        return iter(self.lines)


class Case(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.root = os.path.realpath(tempfile.mkdtemp())
        write(os.path.join(self.root, "app.py"), "def main():\n    return 42\n")
        self.sent, self.replies = [], []
        for p in (mock.patch.dict(os.environ, {"LLM_SESSIONS_DIR": self.dir, "LLM_CWD": self.root}),
                  mock.patch("urllib.request.urlopen", self.fake_urlopen),
                  mock.patch.object(llm, "health", return_value=None), mock.patch.object(llm, "set_lang")):
            p.start()
            self.addCleanup(p.stop)

    def fake_urlopen(self, req, timeout=None):
        self.sent.append((json.loads(req.data), dict(req.header_items())))
        return Stream(self.replies.pop(0) if self.replies else script("(end)")[0])

    def chat(self, root=None):
        a = agent.Agent(dict(llm.load_cfg(), port=1), root)
        a.auto = True
        return a


class TestInline(Case):
    def test_the_model_hands_a_task_to_an_agent_and_gets_its_report(self):
        a = self.chat(self.root)
        self.replies = script(("run_agent", {"agent": "explorer", "task": "what does main return?"}),
                              ("read_file", {"path": "app.py"}), "main returns 42 (app.py:2)", "It returns 42.")
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            a.run_turn("ask the explorer")
        body = [b for b, _ in self.sent]
        sub_tools = {t["function"]["name"] for t in body[1]["tools"]}
        self.assertIn("read_file", sub_tools)
        self.assertFalse({"write_file", "run", "run_agent", "attach_project"} & sub_tools)   # explorer: read only
        self.assertIn("sub-agent “explorer”", body[1]["messages"][0]["content"])
        self.assertEqual(body[1]["messages"][1]["content"], "what does main return?")      # only the task
        tool_result = [m for m in body[3]["messages"] if m["role"] == "tool"][-1]["content"]
        self.assertIn("main returns 42", tool_result)
        self.assertIn("│ explorer", out.getvalue())                     # its steps are shown in the chat
        self.assertEqual(a.messages[-1]["content"], "It returns 42.")

    def test_a_tool_outside_its_list_is_refused(self):
        a = self.chat(self.root)
        run = agent.Run(1, agents.find("explorer", self.root), "t", a)
        sub = agent.SubAgent(a, run.definition, run)
        self.assertIn("not available", sub.call_tool("write_file", {"path": "x", "content": "y"}))
        self.assertFalse(os.path.exists(os.path.join(self.root, "x")))

    def test_save_agent_asks_even_in_auto(self):
        a = self.chat(self.root)
        with mock.patch.object(llm, "ask_yn", return_value=False) as asked, \
                contextlib.redirect_stdout(io.StringIO()):
            out = a.call_tool("save_agent", {"name": "doc-bot", "description": "d", "instructions": "i"})
        asked.assert_called_once()
        self.assertIn("declined", out)
        with mock.patch.object(llm, "ask_yn", return_value=True), contextlib.redirect_stdout(io.StringIO()):
            a.call_tool("save_agent", {"name": "doc-bot", "description": "d", "instructions": "i", "tools": "read_file"})
        self.assertEqual(agents.find("doc-bot", self.root).tools, ["read_file"])


class TestWindows(Case):
    def start(self, a, name, task):
        """`/agents <name> <task>` with the Terminal window replaced: returns the run and its window's args."""
        with mock.patch.object(llm, "new_window", return_value=True) as win, \
                mock.patch.object(a, "watch_runs"), contextlib.redirect_stdout(io.StringIO()):
            a.agent_command(f"/agents {name} {task}")                     # (the watcher thread is the test here)
        run = a.runs[-1]
        return run, win.call_args

    def window(self, run, replies):
        """What the agent's window does, run here."""
        self.replies = script(*replies)
        with mock.patch.object(agent.time, "sleep"), contextlib.redirect_stdout(io.StringIO()):
            return agent.agent_window(run.folder)

    def test_an_agent_window_reports_back_to_the_chat(self):
        a = self.chat(self.root)
        run, call = self.start(a, "explorer", "find main")
        args, kw = call
        self.assertEqual(args[0], ["chat", "--agent-run", run.folder])
        self.assertTrue(kw["close"])                                      # the window closes itself when done
        self.assertEqual(kw["env"]["LLM_SESSIONS_DIR"], self.dir)
        self.assertTrue(self.window(run, [("search", {"pattern": "def main"}), "main is in app.py:1"]))
        _, headers = self.sent[0]
        self.assertEqual(headers.get("X-priority"), "background")         # the chat goes first on the engine
        with contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertTrue(run.refresh())
            a.run_finished(run)
        self.assertEqual((run.status, run.steps), ("done", 2))
        self.assertIn("/agents result 1", out.getvalue())
        with open(os.path.join(run.folder, "log.txt")) as f:
            self.assertIn("⚙ search", f.read())
        # the report rides with the next message, once
        self.replies = script("noted")
        with contextlib.redirect_stdout(io.StringIO()):
            a.run_turn("thanks")
        sent = self.sent[-1][0]["messages"][-1]["content"]
        self.assertIn("main is in app.py:1", sent)
        self.assertTrue(sent.endswith("(My message:)\nthanks"))
        self.assertTrue(os.path.exists(os.path.join(run.folder, "delivered")))
        with contextlib.redirect_stdout(io.StringIO()):
            a.run_turn("next")
        self.assertEqual(self.sent[-1][0]["messages"][-1]["content"], "next")

    def test_the_watcher_notices_a_question_and_the_end_on_its_own(self):
        a = self.chat(self.root)
        run, _ = self.start(a, "fixer", "edit it")
        seen = []
        with mock.patch.object(a, "notify", side_effect=seen.append):
            a.watch_runs()
            agent._write_json(os.path.join(run.folder, "state.json"),
                              {"pid": os.getpid(), "status": "waiting", "question": "Edit app.py?"})
            for _ in range(50):
                if seen:
                    break
                agent.time.sleep(0.1)
            agent._write_json(os.path.join(run.folder, "state.json"),
                              {"pid": os.getpid(), "status": "done", "result": "edited"})
            for _ in range(50):
                if run.reported:
                    break
                agent.time.sleep(0.1)
        self.assertIn("Edit app.py?", seen[0])
        self.assertTrue(run.reported)
        self.assertEqual(a.reports, [run])

    def test_say_reaches_a_running_agent_and_continues_a_finished_one(self):
        a = self.chat(self.root)
        run, _ = self.start(a, "fixer", "bump the answer")
        run.status = "running"
        self.assertEqual(run.say("use 43, not 44"), "sent")
        self.window(run, ["done: 43"])
        self.assertIn("use 43, not 44", self.sent[0][0]["messages"][-1]["content"])
        run.refresh()
        with mock.patch.object(llm, "new_window", return_value=True) as win, mock.patch.object(a, "watch_runs"):
            self.assertEqual(run.say("also add a test"), "continued")
        self.assertEqual(win.call_count, 1)
        self.window(run, ["added"])
        msgs = [m["content"] for m in self.sent[-1][0]["messages"][1:] if m["role"] == "user"]
        self.assertEqual(msgs[-1], "also add a test")                     # the same conversation goes on
        self.assertIn("bump the answer", msgs[0])

    def test_a_closed_window_is_noticed_and_reports_come_after_reopening(self):
        a = self.chat(self.root)
        run, _ = self.start(a, "explorer", "look around")
        agent._write_json(os.path.join(run.folder, "state.json"), {"pid": 999999, "status": "running"})
        run.refresh()
        self.assertEqual(run.status, "stopped")
        self.assertIn("closed", run.result)
        # another one finishes after the chat was closed: reopening the chat brings its report in
        run2, _ = self.start(a, "explorer", "later")
        self.window(run2, ["found it later"])
        b = agent.Agent(dict(llm.load_cfg(), port=1), self.root, a.chat)
        with contextlib.redirect_stdout(io.StringIO()) as out:
            b.collect_reports()
        self.assertIn("finished while the chat was closed", out.getvalue())
        self.assertIn("found it later", b.take_reports("hi"))

    def test_a_reopened_chat_does_not_reuse_an_earlier_runs_folder(self):
        a = self.chat(self.root)
        first, _ = self.start(a, "explorer", "one")
        b = agent.Agent(dict(llm.load_cfg(), port=1), self.root, a.chat)     # the same chat, a later session
        second, _ = self.start(b, "explorer", "two")
        self.assertNotEqual(first.folder, second.folder)
        self.assertEqual(second.id, first.id + 1)

    def test_a_late_report_is_kept_until_a_message_carries_it(self):
        a = self.chat(self.root)
        run, _ = self.start(a, "explorer", "later")
        self.window(run, ["found it"])
        b = agent.Agent(dict(llm.load_cfg(), port=1), self.root, a.chat)
        with contextlib.redirect_stdout(io.StringIO()):
            b.collect_reports()
        self.assertFalse(os.path.exists(os.path.join(run.folder, "delivered")))   # not yet: nothing was sent
        c = agent.Agent(dict(llm.load_cfg(), port=1), self.root, a.chat)         # closed again, reopened
        with contextlib.redirect_stdout(io.StringIO()):
            c.collect_reports()
        self.assertIn("found it", c.take_reports("hi"))
        self.assertTrue(os.path.exists(os.path.join(run.folder, "delivered")))

    def test_stopped_while_asking_is_stopped_not_done(self):
        a = self.chat(self.root)
        a.auto = False
        run, _ = self.start(a, "fixer", "edit")
        self.replies = script(("write_file", {"path": "x.py", "content": "y"}))
        with mock.patch.object(llm, "ask_yn", side_effect=KeyboardInterrupt), mock.patch.object(agent.time, "sleep"), \
                contextlib.redirect_stdout(io.StringIO()):
            agent.agent_window(run.folder)
        run.refresh()
        self.assertEqual(run.status, "stopped")
        self.assertFalse(os.path.exists(os.path.join(self.root, "x.py")))

    def test_commands_for_unknown_things_say_so(self):
        a = self.chat(self.root)
        with contextlib.redirect_stdout(io.StringIO()) as out:
            a.agent_command("/agents nobody do it")
            a.agent_command("/agents stop 7")
            a.agent_command("/agents explorer")
            a.agent_command("/agents")
        text = out.getvalue()
        self.assertIn(llm.T("No agent named {name}.").format(name="nobody"), text)
        self.assertIn(llm.T("No such agent run. /agents lists them."), text)
        self.assertIn(llm.T("Give it a task: /agents {name} <task>").format(name="explorer"), text)
        self.assertIn("reviewer", text)


if __name__ == "__main__":
    unittest.main()
