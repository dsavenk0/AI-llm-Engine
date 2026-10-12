"""harness.fixtests and the agent's /fix-tests loop (no model needed: the "model" is a stub).

    .venv/bin/python -m unittest tests.test_fixtests -v
"""
import io
import contextlib
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from harness import fixtests  # noqa: E402

PYTEST_OUT = """\
============================= test session starts ==============================
collected 3 items

tests/test_calc.py .FF                                                   [100%]

=================================== FAILURES ===================================
________________________________ test_add_zero _________________________________

    def test_add_zero():
>       assert add(0, 0) == 1
E       assert 0 == 1

tests/test_calc.py:7: AssertionError
________________________________ test_sub _____________________________________

    def test_sub():
>       assert sub(3, 1) == 3
E       assert 2 == 3

tests/test_calc.py:11: AssertionError
=========================== short test summary info ============================
FAILED tests/test_calc.py::test_add_zero - assert 0 == 1
FAILED tests/test_calc.py::test_sub - assert 2 == 3
========================= 2 failed, 1 passed in 0.03s ==========================
"""


def project(files):
    d = tempfile.mkdtemp()
    for name, body in files.items():
        p = os.path.join(d, name)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w") as f:
            f.write(textwrap.dedent(body))
    return d


class TestDetect(unittest.TestCase):
    def test_explicit_file_wins_then_ecosystems(self):
        self.assertEqual(fixtests.detect_command(project({".llm/test_cmd": "make check\n", "go.mod": "x"})),
                         "make check")
        self.assertEqual(fixtests.detect_command(project({"package.json": json.dumps({"scripts": {"test": "jest"}})})),
                         "npm test --silent")
        self.assertIsNone(fixtests.detect_command(project({"package.json": json.dumps({"scripts": {}})})))
        self.assertEqual(fixtests.detect_command(project({"Cargo.toml": ""})), "cargo test")
        self.assertEqual(fixtests.detect_command(project({"go.mod": ""})), "go test ./...")
        self.assertIn("-m pytest", fixtests.detect_command(project({"pytest.ini": "[pytest]"})))
        self.assertIn("-m unittest discover -s tests", fixtests.detect_command(project({"tests/test_a.py": ""})))
        self.assertEqual(fixtests.detect_command(project({"Makefile": "all:\n\techo\ntest:\n\techo t\n"})), "make test")
        self.assertIsNone(fixtests.detect_command(project({"README.md": "hi"})))


class TestDetectMore(unittest.TestCase):
    def test_swift_make_pyproject_and_the_unittest_layouts(self):
        self.assertEqual(fixtests.detect_command(project({"Package.swift": ""})), "swift test")
        self.assertEqual(fixtests.detect_command(project({"makefile": "x:\n\ttrue\ntest:\n\ttrue\n"})), "make test")
        self.assertIsNone(fixtests.detect_command(project({"Makefile": "contest:\n\ttrue\n"})))   # "test:" must start a line
        self.assertIn("-m pytest -x -q", fixtests.detect_command(project({"pyproject.toml": "[tool.pytest.ini_options]"})))
        self.assertIn("-m pytest", fixtests.detect_command(project({"setup.cfg": "[tool:pytest]\n"})))
        self.assertIn("-m pytest", fixtests.detect_command(project({"conftest.py": ""})))
        with_pkg = fixtests.detect_command(project({"tests/__init__.py": "", "tests/test_a.py": ""}))
        self.assertTrue(with_pkg.endswith("discover -s tests -t ."))
        bare = fixtests.detect_command(project({"test/test_a.py": ""}))
        self.assertTrue(bare.endswith("discover -s test"))
        self.assertIn("-B", bare)
        # an empty .llm/test_cmd falls through to the ecosystems
        self.assertEqual(fixtests.detect_command(project({".llm/test_cmd": "  \n", "Cargo.toml": ""})), "cargo test")
        self.assertIsNone(fixtests.detect_command(project({"package.json": "{broken"})))

    def test_the_prompt_names_the_command_the_round_and_the_previous_attempt(self):
        first = fixtests.prompt("pytest -q", "FAILED x", 1)
        self.assertIn("`pytest -q`", first)
        self.assertIn("Round 1. Failing output:\n\nFAILED x", first)
        self.assertNotIn("previous attempt", first)
        again = fixtests.prompt("pytest -q", "FAILED x", 2, previous=True)
        self.assertIn("previous attempt did not fix it", again)
        self.assertIn("Round 2.", again)

    def test_constants(self):
        self.assertEqual((fixtests.MAX_EXCERPT, fixtests.TIMEOUT), (6000, 900))

    def test_run_returns_the_exit_code_and_both_streams(self):
        code, out = fixtests.run(f"{sys.executable} -c \"import sys; print('A'); print('B', file=sys.stderr); sys.exit(3)\"",
                                 tempfile.mkdtemp())
        self.assertEqual(code, 3)
        self.assertIn("A", out)
        self.assertIn("B", out)


class TestFailuresMore(unittest.TestCase):
    def test_the_fallback_keeps_the_last_sixty_lines_and_at_most_the_limit(self):
        log = "\n".join(f"line {i:03d}" for i in range(100))
        ex = fixtests.failures(log)
        self.assertEqual(ex.split("\n")[0], "line 040")
        self.assertEqual(len(ex.split("\n")), 60)
        self.assertEqual(len(fixtests.failures("x" * 20000, limit=100)), 100)

    def test_an_oversized_block_is_cut_in_the_middle_and_the_rest_are_counted(self):
        big = f"{'=' * 70}\nFAIL: test_big (m.C.test_big)\n{'-' * 70}\n" + "H" * 5000 + "T" * 5000 + "\n"
        two = f"{'=' * 70}\nFAIL: test_two (m.C.test_two)\n{'-' * 70}\ntraceback two\n"
        ex = fixtests.failures(big + two, limit=1000)
        self.assertIn("\n…\n", ex)
        self.assertTrue(ex.startswith("FAIL: test_big"))
        self.assertIn("T" * 100, ex)                                 # the tail of the block survives
        self.assertIn("… and 1 more failing test(s) not shown", ex)
        self.assertNotIn("traceback two", ex)

    def test_summary_lines_of_other_runners(self):
        out = f"{'_' * 20} test_a {'_' * 20}\nassert 1 == 2\n{'=' * 10} short test summary info {'=' * 10}\nFAILED t.py::test_a\n{'=' * 5} 1 failed in 0.1s {'=' * 5}\n"
        self.assertIn("1 failed in 0.1s", fixtests.failures(out))
        self.assertIn("test result: FAILED. 0 passed; 1 failed",
                      fixtests.failures("running 1 test\nsomething\ntest result: FAILED. 0 passed; 1 failed\n"))


class TestFailures(unittest.TestCase):
    def test_pytest_failures_are_cut_to_the_failing_tests(self):
        ex = fixtests.failures(PYTEST_OUT)
        self.assertIn("FAILED: test_add_zero", ex)
        self.assertIn("assert 0 == 1", ex)
        self.assertIn("assert 2 == 3", ex)
        self.assertIn("2 failed, 1 passed", ex)
        self.assertNotIn("test session starts", ex)

    def test_real_unittest_output(self):
        d = project({"t.py": """
            import unittest
            class T(unittest.TestCase):
                def test_ok(self): pass
                def test_bad(self): self.assertEqual(1 + 1, 3)
                def test_err(self): raise KeyError("boom")
            unittest.main()
            """})
        p = subprocess.run([sys.executable, "-B", "t.py"], cwd=d, capture_output=True, text=True)
        ex = fixtests.failures(p.stderr)
        self.assertIn("FAIL: test_bad", ex)
        self.assertIn("AssertionError: 2 != 3", ex)
        self.assertIn("ERROR: test_err", ex)
        self.assertIn("KeyError: 'boom'", ex)
        self.assertIn("FAILED (failures=1, errors=1)", ex)
        self.assertNotIn("test_ok", ex)

    def test_unknown_runner_falls_back_to_the_tail_and_long_output_is_capped(self):
        log = "\n".join(f"line {i}" for i in range(500))
        ex = fixtests.failures(log)
        self.assertTrue(ex.endswith("line 499"))
        self.assertNotIn("line 100\n", ex)
        many = "".join(f"{'=' * 70}\nFAIL: test_{i} (m.C.test_{i})\n{'-' * 70}\nTraceback: {'x' * 900}\n" for i in range(40))
        capped = fixtests.failures(many + "\nFAILED (failures=40)\n")
        self.assertLess(len(capped), fixtests.MAX_EXCERPT + 400)
        self.assertIn("more failing test(s) not shown", capped)

    def test_signature_ignores_timings_and_line_numbers_but_not_content(self):
        a = "FAIL: test_x\n  File \"a.py\", line 12, in f\nAssertionError: 2 != 3\nRan 3 tests in 0.012s"
        b = "FAIL: test_x\n  File \"a.py\", line 15, in f\nAssertionError: 2 != 3\nRan 3 tests in 0.340s"
        c = "FAIL: test_x\n  File \"a.py\", line 12, in f\nAssertionError: 2 != 4\nRan 3 tests in 0.012s"
        self.assertEqual(fixtests.signature(a), fixtests.signature(b))
        self.assertNotEqual(fixtests.signature(a), fixtests.signature(c))

    def test_timeout_is_a_failure_not_a_hang(self):
        code, out = fixtests.run(f"{sys.executable} -c 'import time; time.sleep(5)'", tempfile.mkdtemp(), timeout=1)
        self.assertEqual(code, 124)
        self.assertIn("timed out", out)


class TestAgentLoop(unittest.TestCase):
    def agent(self, files):
        import agent
        import llm
        cfg = llm.load_cfg()
        cfg["port"] = 1
        a = agent.Agent(cfg, project(files))
        a.auto = True
        return a

    def run_loop(self, a, **kw):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            ok = a.fix_tests(**kw)
        return ok, buf.getvalue()

    BROKEN = {"calc.py": "def add(a, b):\n    return a - b\n",
              "tests/test_calc.py": """
                  import sys, os, unittest
                  sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                  from calc import add
                  class T(unittest.TestCase):
                      def test_add(self): self.assertEqual(add(2, 3), 5)
                  """}

    def test_loop_feeds_failures_to_the_model_and_stops_when_green(self):
        a = self.agent(self.BROKEN)
        prompts = []

        def stub_turn(text):
            prompts.append(text)
            with open(os.path.join(a.root, "calc.py"), "w") as f:     # what the model's edit_file would do
                f.write("def add(a, b):\n    return a + b\n")
        a.turn = stub_turn
        ok, out = self.run_loop(a)
        self.assertTrue(ok)
        self.assertEqual(len(prompts), 1)
        self.assertIn("FAIL: test_add", prompts[0])
        self.assertIn("AssertionError: -1 != 5", prompts[0])
        self.assertNotIn("Ran 1 test", prompts[0].split("Failing output:")[0])
        self.assertIn("Tests pass", out)

    def test_loop_stops_when_edits_change_nothing(self):
        a = self.agent(self.BROKEN)
        calls = []
        a.turn = lambda text: calls.append(text)               # a model that does nothing useful
        ok, out = self.run_loop(a, max_rounds=5)
        self.assertFalse(ok)
        self.assertEqual(len(calls), 2)                        # not five: two attempts changed nothing → stop
        self.assertIn("same failure after two attempts", out)

    def test_loop_respects_the_round_limit(self):
        a = self.agent(self.BROKEN)
        n = [0]

        def churn(text):                                      # changes the failure every round, never fixes it
            n[0] += 1
            with open(os.path.join(a.root, "calc.py"), "w") as f:
                f.write(f"def add(a, b):\n    return a - b - {n[0]}\n")
        a.turn = churn
        ok, out = self.run_loop(a, max_rounds=3)
        self.assertFalse(ok)
        self.assertEqual(n[0], 3)
        self.assertIn("Out of rounds", out)

    def test_no_command_is_reported(self):
        a = self.agent({"README.md": "x"})
        ok, out = self.run_loop(a)
        self.assertFalse(ok)
        self.assertIn("No test command", out)

    def test_declining_the_run_runs_nothing(self):
        a = self.agent(self.BROKEN)
        a.auto = False
        import llm
        orig, llm.ask_yn = llm.ask_yn, lambda *x, **k: False
        self.addCleanup(setattr, llm, "ask_yn", orig)
        called = []
        a.turn = lambda t: called.append(t)
        ok, _ = self.run_loop(a)
        self.assertFalse(ok)
        self.assertEqual(called, [])


if __name__ == "__main__":
    unittest.main()
