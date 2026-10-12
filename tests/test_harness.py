"""Harness modules (stdlib only): project memory and context compaction.

    .venv/bin/python -m unittest tests.test_harness -v
"""
import json
import os
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from harness import compact, memory  # noqa: E402


def convo(turns, tool_chars=5000):
    """system + `turns` user turns, each with a tool round-trip and a final answer."""
    m = [{"role": "system", "content": "SYS"}]
    for i in range(turns):
        m.append({"role": "user", "content": f"task {i}"})
        m.append({"role": "assistant", "content": "", "reasoning_content": "thinking " * 50,
                  "tool_calls": [{"id": f"c{i}", "type": "function",
                                  "function": {"name": "read_file", "arguments": json.dumps({"path": f"f{i}.py"})}}]})
        m.append({"role": "tool", "tool_call_id": f"c{i}", "content": "x" * tool_chars})
        m.append({"role": "assistant", "content": f"done {i}"})
    return m


class TestMemory(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()

    def test_nothing_without_a_file(self):
        self.assertEqual(memory.load(self.root), (None, ""))
        self.assertEqual(memory.system_block(self.root), "")

    def test_first_candidate_wins(self):
        with open(os.path.join(self.root, "CLAUDE.md"), "w") as f:
            f.write("from claude")
        with open(os.path.join(self.root, "AGENTS.md"), "w") as f:
            f.write("from agents")
        self.assertEqual(memory.load(self.root), ("AGENTS.md", "from agents"))
        memory.remember(self.root, "mine")
        self.assertEqual(memory.load(self.root)[0], os.path.join(".llm", "memory.md"))

    def test_remember_appends_and_never_touches_other_files(self):
        with open(os.path.join(self.root, "CLAUDE.md"), "w") as f:
            f.write("keep me")
        memory.remember(self.root, "use  tabs\nnever spaces")
        memory.remember(self.root, "run tests with pytest")
        _, text = memory.load(self.root)
        self.assertEqual(text, "- use tabs never spaces\n- run tests with pytest")
        with open(os.path.join(self.root, "CLAUDE.md")) as f:
            self.assertEqual(f.read(), "keep me")
        with self.assertRaises(ValueError):
            memory.remember(self.root, "   ")

    def test_long_file_is_cut_on_a_line_boundary(self):
        with open(os.path.join(self.root, "AGENTS.md"), "w") as f:
            f.write("\n".join(f"rule {i}" for i in range(2000)))
        text = memory.load(self.root)[1]
        self.assertLessEqual(len(text), memory.MAX_CHARS + 20)
        self.assertTrue(text.endswith("(truncated)"))


class TestCompaction(unittest.TestCase):
    def test_trigger(self):
        self.assertFalse(compact.needs_compaction(0, 32768))
        self.assertFalse(compact.needs_compaction(1000, 0))
        self.assertFalse(compact.needs_compaction(20000, 32768))
        self.assertTrue(compact.needs_compaction(24000, 32768))

    def test_shrink_touches_only_old_turns(self):
        m = convo(5)
        out, saved = compact.shrink(m, keep_user_turns=2)
        self.assertGreater(saved, 4000)
        self.assertEqual(len(out), len(m))
        self.assertLess(len(out[3]["content"]), 400)            # old tool output cut
        self.assertNotIn("reasoning_content", out[2])           # old reasoning dropped
        self.assertEqual(out[-2]["content"], "x" * 5000)        # recent turn verbatim
        self.assertIn("reasoning_content", out[-3])
        self.assertIn("reasoning_content", m[2])                # the input is not mutated

    def test_compact_replaces_the_middle_and_keeps_system_and_tail(self):
        m = convo(5)
        seen = []
        out = compact.compact(m, lambda p: seen.append(p) or "goal: fix f0..f2; f0.py edited")
        self.assertEqual(out[0], m[0])
        self.assertIn("goal: fix f0..f2", out[1]["content"])
        self.assertEqual(out[1]["role"], "user")
        self.assertEqual(out[2]["role"], "assistant")           # roles keep alternating for the template
        self.assertEqual(out[3:], m[-8:])                       # last two user turns verbatim
        self.assertIn("task 0", seen[0])
        self.assertNotIn("task 4", seen[0])                     # the tail is not summarized
        self.assertIn("read_file(path='f0.py')", seen[0])       # tool calls are described, not dumped
        self.assertLess(len(seen[0]), 4000)                     # old tool output is clipped

    def test_nothing_to_fold_or_empty_summary_leaves_history_alone(self):
        short = convo(2)
        self.assertIs(compact.compact(short, lambda p: "never called"), short)
        long = convo(4)
        self.assertIs(compact.compact(long, lambda p: "  "), long)

    def test_the_trigger_is_inclusive_and_constants_are_what_the_docs_say(self):
        self.assertEqual((compact.TRIGGER, compact.KEEP_USER_TURNS, compact.OLD_TOOL_KEEP, compact.TRANSCRIPT_MAX),
                         (0.70, 2, 300, 24000))
        self.assertTrue(compact.needs_compaction(70, 100))               # exactly at the share: compact
        self.assertFalse(compact.needs_compaction(69, 100))
        self.assertTrue(compact.needs_compaction(50, 100, trigger=0.5))

    def test_the_tail_starts_at_the_nth_user_message_from_the_end_and_never_at_the_system(self):
        m = [{"role": "system"}, {"role": "user"}, {"role": "assistant"}, {"role": "user"}, {"role": "assistant"},
             {"role": "user"}, {"role": "assistant"}]
        self.assertEqual(compact._tail_start(m, 1), 5)
        self.assertEqual(compact._tail_start(m, 2), 3)
        self.assertEqual(compact._tail_start(m, 3), 1)
        self.assertEqual(compact._tail_start(m, 9), 1)                   # fewer user turns than asked: all but the system
        self.assertEqual(compact._tail_start([{"role": "system"}, {"role": "assistant"}], 1), 1)

    def test_shrink_counts_every_saved_character_and_keeps_the_system_and_the_tail(self):
        long = "x" * 1000
        m = [{"role": "system", "content": "S" * 900, "reasoning_content": "keep me"},
             {"role": "user", "content": "q1"},
             {"role": "assistant", "content": "", "reasoning_content": "think", "tool_calls": []},
             {"role": "tool", "content": long},
             {"role": "tool", "content": "y" * 300},                      # exactly the limit: left alone
             {"role": "user", "content": "q2"}, {"role": "assistant", "content": "a2", "reasoning_content": "t2"}]
        out, saved = compact.shrink(m, keep_user_turns=1)
        self.assertEqual(out[0], m[0])                                    # the system message is never touched
        self.assertNotIn("reasoning_content", out[2])
        self.assertEqual(out[3]["content"], "x" * 300 + "\n… [700 chars of old tool output dropped]")
        self.assertEqual(out[4], m[4])
        self.assertEqual(out[5:], m[5:])                                  # the tail stays as it was
        self.assertEqual(saved, 1 + 700)                                  # one dropped reasoning + the cut characters

    def test_transcript_describes_tool_calls_and_hides_edit_bodies(self):
        import json
        m = [{"role": "assistant", "content": "looking", "tool_calls": [
                {"function": {"name": "edit_file", "arguments": json.dumps({"path": "a.py", "old": "SECRET", "new": "x"})}},
                {"function": {"name": "read_file", "arguments": "{broken"}}]},
             {"role": "tool", "content": "z" * 500}, {"role": "user", "content": None}]
        t = compact.transcript(m)
        self.assertIn("assistant: looking [tool calls: edit_file(path='a.py'); read_file()]", t)
        self.assertNotIn("SECRET", t)
        self.assertIn("tool: " + "z" * 300 + "\n", t + "\n")
        self.assertNotIn("z" * 301, t)
        self.assertTrue(t.rstrip().endswith("user:"))

    def test_compact_passes_the_summary_prompt_and_trims_the_note(self):
        m = convo(4)
        got = []
        out = compact.compact(m, lambda p: got.append(p) or "  the note \n")
        self.assertTrue(got[0].startswith(compact.SUMMARY_PROMPT))
        self.assertTrue(out[1]["content"].endswith("\nthe note"))
        self.assertEqual(out[1]["content"].split("\n")[0],
                         "[Summary of the earlier conversation — the transcript was compacted to save context]")
        self.assertEqual(compact.compact(m, lambda p: None), m)         # None from the summarizer: unchanged

    def test_transcript_is_capped_keeping_the_newest(self):
        m = convo(200, tool_chars=100)
        t = compact.transcript(m[1:])
        self.assertLessEqual(len(t), compact.TRANSCRIPT_MAX + 5)
        self.assertIn("task 199", t)
        self.assertNotIn("task 0\n", t)


class FakeEngine(BaseHTTPRequestHandler):
    calls = []

    def log_message(self, *a):
        pass

    def do_GET(self):
        body = json.dumps({"context": 1000}).encode()
        self.send_response(200); self.send_header("Content-Length", str(len(body))); self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeEngine.calls.append(req)
        if req.get("stream"):
            events = [{"choices": [{"delta": {"content": "ok"}}]},
                      {"choices": [], "usage": {"prompt_tokens": 900, "completion_tokens": 1}}]
            body = "".join(f"data: {json.dumps(e)}\n\n" for e in events).encode() + b"data: [DONE]\n\n"
        else:
            body = json.dumps({"choices": [{"message": {"content": "SUMMARY-NOTE"}}]}).encode()
        self.send_response(200); self.send_header("Content-Length", str(len(body))); self.end_headers()
        self.wfile.write(body)


class TestAgentCompaction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = ThreadingHTTPServer(("127.0.0.1", 0), FakeEngine)
        cls.port = cls.srv.server_address[1]
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()

    def agent(self):
        import agent
        import llm
        cfg = llm.load_cfg()
        cfg["port"] = self.port
        FakeEngine.calls.clear()
        return agent.Agent(cfg, tempfile.mkdtemp())

    def test_a_turn_near_the_limit_compacts_before_the_next_call(self):
        a = self.agent()
        self.assertEqual(a.context, 1000)
        a.messages += convo(4, tool_chars=200)[1:]   # little for the free stage to reclaim
        a.prompt_tokens = 900               # what the previous completion reported
        import io, contextlib
        with contextlib.redirect_stdout(io.StringIO()):
            a.turn("next task")
        kinds = [bool(c.get("stream")) for c in FakeEngine.calls]
        self.assertEqual(kinds, [False, True])          # summary first, then the real request
        sent = FakeEngine.calls[1]["messages"]
        self.assertIn("SUMMARY-NOTE", sent[1]["content"])
        self.assertEqual(sent[-1]["content"], "next task")
        self.assertLess(len(sent), 12)

    def test_shrinking_alone_is_enough_when_old_tool_output_was_the_bulk(self):
        a = self.agent()
        a.messages += convo(4, tool_chars=20000)[1:]  # ~6.6k tokens of old tool output
        a.prompt_tokens = 900
        import io, contextlib
        with contextlib.redirect_stdout(io.StringIO()):
            a.turn("next task")
        self.assertEqual([bool(c.get("stream")) for c in FakeEngine.calls], [True])  # no summary call
        sent = FakeEngine.calls[0]["messages"]
        tools = [len(m["content"]) for m in sent if m["role"] == "tool"]
        self.assertEqual(sorted(tools)[-1], 20000)                   # the latest turn is verbatim
        self.assertTrue(all(n < 400 for n in sorted(tools)[:-1]), tools)  # the older ones are stubs

    def test_project_memory_reaches_the_system_prompt(self):
        a = self.agent()
        memory.remember(a.root, "answer in haiku")
        self.assertIn("answer in haiku", a.system_prompt())


class TestShellRun(unittest.TestCase):
    """harness.shell.run: a deadline that also ends the command's children."""

    def test_output_and_exit_code(self):
        from harness import shell
        r = shell.run("echo out; echo err 1>&2; exit 3", timeout=10)
        self.assertEqual((r.returncode, r.stdout.strip(), r.stderr.strip(), r.timed_out), (3, "out", "err", False))

    def test_timeout_returns_the_output_so_far_and_kills_the_whole_group(self):
        import subprocess
        import time
        from harness import shell
        marker = f"sleep {7000 + os.getpid() % 900}"      # unique, so pgrep sees only this test's child
        t0 = time.time()
        r = shell.run(f"echo started; ({marker} &) ; {marker}", timeout=1.5)
        self.assertTrue(r.timed_out)
        self.assertEqual(r.returncode, 124)
        self.assertIn("started", r.stdout)
        self.assertLess(time.time() - t0, 8)
        time.sleep(0.3)
        left = subprocess.run(["pgrep", "-f", marker], capture_output=True, text=True).stdout.split()
        self.assertEqual(left, [], "the shell's children survived the timeout")

    def test_a_flood_of_output_is_cut_and_the_command_stopped(self):
        import time
        from harness import shell
        t0 = time.time()
        r = shell.run("yes abcdefghij", timeout=30)              # would fill memory if collected whole
        self.assertTrue(r.truncated)
        self.assertEqual((r.returncode, r.timed_out), (125, False))
        self.assertLessEqual(len(r.stdout), shell.MAX_OUTPUT)
        self.assertGreater(len(r.stdout), shell.MAX_OUTPUT - 1000)   # the kept part is the beginning, in full
        self.assertTrue(r.stdout.startswith("abcdefghij\nabcdefghij"))
        self.assertLess(time.time() - t0, 10)
        # stderr is limited on its own, and a normal command is untouched
        e = shell.run("yes e 1>&2", timeout=30)
        self.assertTrue(e.truncated)
        self.assertEqual(e.stdout, "")
        ok = shell.run("printf 'x%.0s' $(seq 1 1000)", timeout=10)
        self.assertFalse(ok.truncated)
        self.assertEqual((ok.returncode, len(ok.stdout)), (0, 1000))

    def test_output_that_ends_without_a_newline_and_non_utf8_bytes_survive(self):
        from harness import shell
        r = shell.run("printf 'no newline'; printf '\\377\\376'", timeout=10)
        self.assertEqual(r.returncode, 0)
        self.assertTrue(r.stdout.startswith("no newline"))        # invalid bytes become U+FFFD, nothing is lost

    def test_an_argument_list_runs_without_a_shell(self):
        from harness import shell
        r = shell.run(["/bin/echo", "a b"], shell=False, timeout=10)
        self.assertEqual(r.stdout.strip(), "a b")


if __name__ == "__main__":
    unittest.main()