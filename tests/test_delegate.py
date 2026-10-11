"""harness.delegate: handing sub-tasks to other models (fake endpoint + fake CLI, no network).

    .venv/bin/python -m unittest tests.test_delegate -v
"""
import io
import contextlib
import json
import os
import stat
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from harness import delegate  # noqa: E402


class Fake(BaseHTTPRequestHandler):
    seen = []
    status = 200

    def log_message(self, *a):
        pass

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        Fake.seen.append({"path": self.path, "auth": self.headers.get("Authorization"), "body": req})
        if Fake.status != 200:
            body = json.dumps({"error": {"message": "bad key"}}).encode()
        else:
            body = json.dumps({"choices": [{"message": {"content": "ANSWER: " + req["messages"][0]["content"][:40]}}]}).encode()
        self.send_response(Fake.status)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = ThreadingHTTPServer(("127.0.0.1", 0), Fake)
        cls.url = f"http://127.0.0.1:{cls.srv.server_address[1]}/v1"
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        cls.srv.server_close()

    def setUp(self):
        Fake.seen.clear()
        Fake.status = 200
        self.root = tempfile.mkdtemp()
        with open(os.path.join(self.root, "a.py"), "w") as f:
            f.write("print('a')\n")
        os.makedirs(os.path.join(self.root, ".llm"))

    def write_cfg(self, workers):
        with open(os.path.join(self.root, ".llm", "workers.json"), "w") as f:
            json.dump({"workers": workers}, f)


class TestDelegate(Base):
    def test_openai_worker_gets_task_and_files_with_key_from_env(self):
        os.environ["FAKE_WORKER_KEY"] = "sk-test"
        self.addCleanup(os.environ.pop, "FAKE_WORKER_KEY", None)
        self.write_cfg({"big": {"type": "openai", "url": self.url, "model": "m1", "key_env": "FAKE_WORKER_KEY",
                                "about": "second opinion"}})
        asked = []
        t = delegate.tools(self.root, delegate.load(self.root), allow=lambda w: asked.append(w) or True)
        out = t["delegate"][1](worker="big", task="Review this", files=["a.py"])
        self.assertTrue(out.startswith("ANSWER: Review this"))
        req = Fake.seen[0]
        self.assertEqual(req["path"], "/v1/chat/completions")
        self.assertEqual(req["auth"], "Bearer sk-test")
        self.assertEqual(req["body"]["model"], "m1")
        self.assertEqual(len(req["body"]["messages"]), 1)              # no conversation history is sent
        self.assertIn("### a.py", req["body"]["messages"][0]["content"])
        self.assertIn("print('a')", req["body"]["messages"][0]["content"])
        # the user is shown what goes where, including the file list (loopback: not "leaves this machine")
        self.assertIn("a.py", asked[0])
        self.assertNotIn("leaves this machine", asked[0])

    def test_remote_worker_is_flagged_as_leaving_the_machine(self):
        os.environ["K"] = "x"
        self.addCleanup(os.environ.pop, "K", None)
        self.write_cfg({"cloud": {"type": "openai", "url": "https://api.example.com/v1", "model": "m", "key_env": "K"}})
        asked = []
        t = delegate.tools(self.root, delegate.load(self.root), allow=lambda w: asked.append(w) or False)
        self.assertEqual(t["delegate"][1](worker="cloud", task="x"), "user declined to delegate")
        self.assertIn("leaves this machine", asked[0])
        self.assertEqual(Fake.seen, [])

    def test_missing_key_hides_the_worker_but_it_is_listed(self):
        os.environ.pop("NO_SUCH_KEY", None)
        self.write_cfg({"cloud": {"type": "openai", "url": self.url, "model": "m", "key_env": "NO_SUCH_KEY"}})
        w = delegate.load(self.root)
        self.assertEqual(delegate.tools(self.root, w), {})
        self.assertIn("set $NO_SUCH_KEY", delegate.summary(w)[0])

    def test_schema_lists_only_usable_workers(self):
        self.write_cfg({"a": {"type": "openai", "url": self.url, "model": "m", "about": "fast"},
                        "b": {"type": "cli", "command": []}})
        t = delegate.tools(self.root, delegate.load(self.root))
        fn = t["delegate"][0]["function"]
        self.assertEqual(fn["parameters"]["properties"]["worker"]["enum"], ["a"])
        self.assertIn("a — fast", fn["description"])

    def test_files_outside_the_workspace_and_oversized_context_are_refused_before_asking(self):
        self.write_cfg({"w": {"type": "openai", "url": self.url, "model": "m"}})
        with open(os.path.join(self.root, "big.txt"), "w") as f:
            f.write("x" * (delegate.MAX_CONTEXT_BYTES + 10))
        asked = []
        fn = delegate.tools(self.root, delegate.load(self.root), allow=lambda w: asked.append(w) or True)["delegate"][1]
        self.assertIn("outside the workspace", fn(worker="w", task="t", files=["../../etc/passwd"]))
        self.assertIn("exceed", fn(worker="w", task="t", files=["big.txt"]))
        self.assertIn("unknown worker", fn(worker="nope", task="t"))
        self.assertEqual((asked, Fake.seen), ([], []))

    def test_http_errors_come_back_as_tool_errors_not_exceptions(self):
        self.write_cfg({"w": {"type": "openai", "url": self.url, "model": "m"}})
        Fake.status = 401
        fn = delegate.tools(self.root, delegate.load(self.root))["delegate"][1]
        out = fn(worker="w", task="t")
        self.assertTrue(out.startswith("error: w: HTTP 401"))
        self.assertIn("bad key", out)

    def test_cli_worker_runs_in_the_workspace_and_says_it_may_edit(self):
        script = os.path.join(self.root, "worker.sh")
        with open(script, "w") as f:
            f.write('#!/bin/sh\necho "cwd=$(basename "$PWD")"\necho "task=$1"\n')
        os.chmod(script, os.stat(script).st_mode | stat.S_IEXEC)
        self.write_cfg({"cli": {"type": "cli", "command": [script], "about": "has file tools"}})
        asked = []
        fn = delegate.tools(self.root, delegate.load(self.root), allow=lambda w: asked.append(w) or True)["delegate"][1]
        out = fn(worker="cli", task="do it", files=["a.py"])
        self.assertIn("cwd=" + os.path.basename(self.root), out)
        self.assertIn("task=do it", out)
        self.assertIn("may change files", asked[0])

    def test_failing_cli_reports_exit_code(self):
        self.write_cfg({"bad": {"type": "cli", "command": ["/bin/sh", "-c", "echo oops >&2; exit 3", "x"]},
                        "gone": {"type": "cli", "command": ["/no/such/cli"]}})
        fn = delegate.tools(self.root, delegate.load(self.root))["delegate"][1]
        self.assertIn("exit code 3: oops", fn(worker="bad", task="t"))
        self.assertIn("command not found", fn(worker="gone", task="t"))


class TestAgentWorkers(Base):
    def test_off_by_default_and_toggle(self):
        import agent
        import llm
        self.write_cfg({"w": {"type": "openai", "url": self.url, "model": "m"}})
        cfg = llm.load_cfg()
        cfg["port"] = 1
        a = agent.Agent(cfg, self.root)
        names = lambda: {t["function"]["name"] for t in a.tool_schemas()}
        self.assertNotIn("delegate", names())               # configured is not offered until asked
        from unittest import mock
        from harness import trust
        with mock.patch.object(trust, "STORE", os.path.join(self.root, "trusted.json")), \
                mock.patch.object(agent.llm, "ask_yn", lambda q, default=True: True):
            self.assertTrue(a.set_workers(True))
        self.assertIn("delegate", names())
        a.auto = True
        self.assertTrue(a.call_tool("delegate", {"worker": "w", "task": "hi"}).startswith("ANSWER: hi"))
        a.set_workers(False)
        self.assertNotIn("delegate", names())


if __name__ == "__main__":
    unittest.main()
