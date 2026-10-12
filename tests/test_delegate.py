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


class TestDelegateDetails(Base):
    def setUp(self):
        Fake.seen.clear()
        Fake.status = 200
        self.root = tempfile.mkdtemp()

    def spec(self, **kw):
        return dict({"type": "openai", "url": self.url, "model": "m"}, **kw)

    def test_config_that_is_not_an_object_is_empty_and_a_preset_repairs_it(self):
        os.makedirs(os.path.join(self.root, ".llm"))
        p = os.path.join(self.root, ".llm", "workers.json")
        for body in ("[1]", '{"workers": [1]}', '"x"', "{"):
            with open(p, "w") as f:
                f.write(body)
            self.assertEqual(delegate.load(self.root), {}, body)
        with open(p, "w") as f:
            f.write("[1]")
        delegate.add_preset(self.root, "openrouter")
        self.assertIn("openrouter", delegate.load(self.root))
        with self.assertRaises(ValueError):
            delegate.add_preset(self.root, "nope")

    def test_only_loopback_hosts_are_local(self):
        for url in ("http://localhost:8091/v1", "http://127.0.0.1/v1", "https://[::1]:1/v1", "http://LOCALHOST/x"):
            self.assertTrue(delegate.is_local_url(url), url)
        for url in ("http://127.evil.com/v1", "https://api.openai.com/v1", "http://192.168.1.5/v1", "", "not a url"):
            self.assertFalse(delegate.is_local_url(url), url)

    def test_what_a_worker_needs_is_explained(self):
        self.assertEqual(delegate.usable({"type": "openai", "url": "u"}), (False, "needs url and model"))
        self.assertEqual(delegate.usable({"type": "openai", "model": "m"}), (False, "needs url and model"))
        os.environ.pop("DELEGATE_TEST_KEY", None)
        self.assertEqual(delegate.usable({"type": "openai", "url": "u", "model": "m", "key_env": "DELEGATE_TEST_KEY"}),
                         (False, "set $DELEGATE_TEST_KEY"))
        self.assertEqual(delegate.usable({"type": "cli"}), (False, "needs command"))
        self.assertEqual(delegate.usable({"type": "cli", "command": ["x"]}), (True, ""))
        self.assertEqual(delegate.usable({"type": "openai", "url": "u", "model": "m"}), (True, ""))

    def test_describe_and_summary_lines(self):
        lines = delegate.describe({"c": {"type": "cli", "command": ["gemini", "-p"]},
                                   "o": {"type": "openai", "url": "https://x/v1", "key_env": "K"},
                                   "l": {"type": "openai", "url": "http://localhost:1/v1"}})
        self.assertEqual(lines[0], "c: runs  gemini -p  in the workspace (may edit files)")
        self.assertEqual(lines[1], "o: sends tasks to https://x/v1 with $K  (leaves this machine)")
        self.assertEqual(lines[2], "l: sends tasks to http://localhost:1/v1  (this machine)")
        s = delegate.summary({"a": {"type": "openai", "url": "u", "model": "m", "about": "fast"},
                              "b": {"type": "cli"}})
        self.assertEqual(s[0], "✓ a  [openai] m u  · fast")
        self.assertEqual(s[1], "✗ b  [cli]  — needs command")

    def test_the_request_has_the_default_token_cap_and_the_answer_is_cut_with_a_count(self):
        delegate.ask_openai(self.spec(), "hello")
        self.assertEqual(Fake.seen[-1]["body"]["max_tokens"], 4096)
        self.assertEqual(Fake.seen[-1]["path"], "/v1/chat/completions")
        self.assertEqual(Fake.seen[-1]["body"]["stream"], False)
        delegate.ask_openai(self.spec(max_tokens=7), "hello")
        self.assertEqual(Fake.seen[-1]["body"]["max_tokens"], 7)
        out = delegate.ask("w", {"type": "cli", "command": [sys.executable, "-c",
                                 f"import sys; print('x' * {delegate.MAX_ANSWER + 500})", "--"]}, "task", self.root)
        self.assertTrue(out.startswith("x" * 100))
        self.assertTrue(out.endswith("\n… [500 more characters]"))
        self.assertEqual(len(out), delegate.MAX_ANSWER + len("\n… [500 more characters]"))

    def test_an_empty_answer_is_named_and_the_prompt_carries_the_context(self):
        script = os.path.join(self.root, "none.sh")
        with open(script, "w") as f:
            f.write("#!/bin/sh\n")
        os.chmod(script, os.stat(script).st_mode | stat.S_IEXEC)
        self.assertEqual(delegate.ask("w", {"type": "cli", "command": [script]}, "t", self.root), "(empty answer)")
        self.assertEqual(delegate.build_prompt("  do it  ", ""), "do it")
        self.assertEqual(delegate.build_prompt("do it", "CTX"), "do it\n\nContext files:\n\nCTX")

    def test_files_are_attached_in_order_and_the_limit_is_in_bytes(self):
        for n, body in (("a.txt", "AAA"), ("b.txt", "BBB")):
            with open(os.path.join(self.root, n), "w") as f:
                f.write(body)
        block = delegate.attach(self.root, ["a.txt", "b.txt"])
        self.assertEqual(block, "### a.txt\n```\nAAA\n```\n\n### b.txt\n```\nBBB\n```")
        with open(os.path.join(self.root, "u.txt"), "w", encoding="utf-8") as f:
            f.write("я" * 40_000)                                # 80 000 bytes, 40 000 characters
        with self.assertRaises(ValueError):
            delegate.attach(self.root, ["u.txt"])
        self.assertEqual(delegate.attach(self.root, None), "")

    def test_a_cli_worker_that_fails_or_is_missing_says_why(self):
        with self.assertRaisesRegex(RuntimeError, "command not found: /no/such/bin"):
            delegate.ask_cli({"command": ["/no/such/bin"]}, "t", self.root)
        sh = os.path.join(self.root, "fail.sh")
        with open(sh, "w") as f:
            f.write("#!/bin/sh\necho oops >&2\nexit 2\n")
        os.chmod(sh, os.stat(sh).st_mode | stat.S_IEXEC)
        with self.assertRaisesRegex(RuntimeError, "exit code 2: oops"):
            delegate.ask_cli({"command": [sh]}, "t", self.root)
        slow = os.path.join(self.root, "slow.sh")
        with open(slow, "w") as f:
            f.write("#!/bin/sh\nsleep 5\n")
        os.chmod(slow, os.stat(slow).st_mode | stat.S_IEXEC)
        with self.assertRaisesRegex(RuntimeError, "no answer within 1 s"):
            delegate.ask_cli({"command": [slow]}, "t", self.root, timeout=1)


class TestParallel(Base):
    def setUp(self):
        super().setUp()
        self.write_cfg({"a": {"type": "openai", "url": self.url, "model": "ma", "about": "first"},
                        "b": {"type": "openai", "url": self.url, "model": "mb", "about": "second"},
                        "c": {"type": "openai", "url": "https://api.example.com/v1", "model": "mc", "key_env": "K_PAR"}})
        os.environ["K_PAR"] = "x"
        self.addCleanup(os.environ.pop, "K_PAR", None)
        self.asked = []
        self.fn = delegate.tools(self.root, delegate.load(self.root), allow=lambda w: self.asked.append(w) or True)["delegate"][1]

    def test_the_same_task_to_several_workers_comes_back_in_one_labelled_answer(self):
        out = self.fn(workers=["a", "b"], task="Review this", files=["a.py"])
        self.assertIn("## a\nANSWER: Review this", out)
        self.assertIn("## b\nANSWER: Review this", out)
        self.assertLess(out.index("## a"), out.index("## b"))
        self.assertEqual(sorted(r["body"]["model"] for r in Fake.seen), ["ma", "mb"])
        self.assertEqual(len(self.asked), 1)                       # one confirmation for the whole fan-out
        self.assertIn("2 tasks at the same time", self.asked[0])

    def test_different_jobs_and_one_failure_does_not_hide_the_others(self):
        out = self.fn(jobs=[{"worker": "a", "task": "task one"}, {"worker": "b", "task": "task two", "files": "a.py"}])
        self.assertIn("ANSWER: task one", out)
        self.assertIn("ANSWER: task two", out)
        Fake.status = 500
        out = self.fn(workers=["a", "b"], task="t")
        self.assertEqual(out.count("error: "), 2)
        self.assertIn("## a\nerror: a:", out)

    def test_they_really_run_at_the_same_time(self):
        import time
        started = []
        orig = delegate.ask

        def slow(name, spec, task, root, files=None):
            started.append((name, time.time()))
            time.sleep(0.4)
            return "done " + name
        delegate.ask = slow
        self.addCleanup(setattr, delegate, "ask", orig)
        t0 = time.time()
        out = self.fn(workers=["a", "b"], task="t")
        self.assertLess(time.time() - t0, 0.7)                     # two 0.4 s jobs: ~0.4, not ~0.8
        self.assertIn("done a", out)
        self.assertIn("done b", out)

    def test_limits_and_bad_requests_are_refused_before_asking(self):
        self.assertIn("at most 4", self.fn(workers=["a", "b", "a", "b", "a"], task="t"))
        self.assertIn("name a worker", self.fn())
        self.assertIn("unknown worker", self.fn(workers=["a", "zzz"], task="t"))
        self.assertIn("no task for a", self.fn(jobs=[{"worker": "a", "task": "  "}]))
        self.assertIn("jobs must be a list", self.fn(jobs=["a"]))
        self.assertIn("outside the workspace", self.fn(workers=["a"], task="t", files=["../../etc/passwd"]))
        self.assertEqual((self.asked, Fake.seen), ([], []))

    def test_a_remote_worker_in_the_group_is_flagged_and_a_decline_sends_nothing(self):
        fn = delegate.tools(self.root, delegate.load(self.root), allow=lambda w: self.asked.append(w) or False)["delegate"][1]
        self.assertEqual(fn(workers=["a", "c"], task="t"), "user declined to delegate")
        self.assertIn("leaves this machine", self.asked[-1])
        self.assertEqual(Fake.seen, [])

    def test_schema_offers_workers_and_jobs(self):
        props = delegate.tools(self.root, delegate.load(self.root))["delegate"][0]["function"]["parameters"]["properties"]
        self.assertEqual(props["workers"]["items"]["enum"], ["a", "b", "c"])
        self.assertIn("jobs", props)


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
