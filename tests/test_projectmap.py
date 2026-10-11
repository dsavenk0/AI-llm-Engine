"""Project map and session notes: what lets a restarted agent skip re-scanning the project.

    .venv/bin/python -m unittest tests.test_projectmap -v
"""
import contextlib
import io
import os
import sys
import tempfile
import textwrap
import time
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from harness import memory, projectmap  # noqa: E402


def project(files):
    d = tempfile.mkdtemp()
    for name, body in files.items():
        p = os.path.join(d, name)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w") as f:
            f.write(textwrap.dedent(body))
    return d


FILES = {
    "app/main.py": '"""Entry point: wires the CLI."""\ndef main():\n    pass\nclass App:\n    pass\ndef _private(): pass\n',
    "app/util.js": "// string helpers\nexport function slug(s) {}\nclass Parser {}\n",
    "README.md": "# Demo project\ntext\n",
    "node_modules/x/index.js": "junk",
    "assets/logo.png": "binary-ish",
    "data.bin": "\x00\x01",
}


class TestMap(unittest.TestCase):
    def test_describes_files_and_skips_junk(self):
        root = project(FILES)
        m = projectmap.ensure(root)
        self.assertIn("main.py (6 lines) — Entry point: wires the CLI. — main, App", m)
        self.assertNotIn("_private", m)
        self.assertIn("util.js (3 lines) — string helpers — slug, Parser", m)
        self.assertIn("README.md (2 lines) — Demo project", m)
        for junk in ("node_modules", "logo.png"):
            self.assertNotIn(junk, m)
        self.assertTrue(os.path.isfile(os.path.join(root, ".llm", "map.md")))

    def test_rebuilt_only_when_something_changed(self):
        root = project(FILES)
        projectmap.ensure(root)
        path = os.path.join(root, projectmap.MAP_PATH)
        before = os.stat(path).st_mtime_ns
        time.sleep(0.02)
        projectmap.ensure(root)
        self.assertEqual(os.stat(path).st_mtime_ns, before)               # untouched
        with open(os.path.join(root, "app", "new.py"), "w") as f:
            f.write("def fresh(): pass\n")
        self.assertIn("new.py (1 lines) — fresh", projectmap.ensure(root))  # picked up
        self.assertNotEqual(os.stat(path).st_mtime_ns, before)

    def test_big_projects_are_condensed_not_cut_off(self):
        files = {f"pkg{d}/mod{i}.py": f'"""Module {d}.{i} does something quite specific and long-winded."""\n'
                                      f"def f{i}(): pass\n" for d in range(30) for i in range(6)}
        root = project(files)
        full = projectmap.ensure(root)
        self.assertGreater(len(full), 6000)
        small = projectmap.system_block(root, limit=4000)
        self.assertLessEqual(len(small), 4000)
        for d in (0, 15, 29):                                              # the last directories survive
            self.assertIn(f"pkg{d}/", small)

    def test_empty_workspace_gives_nothing(self):
        self.assertEqual(projectmap.system_block(tempfile.mkdtemp()), "")


class TestSessionNotes(unittest.TestCase):
    def test_roundtrip_and_block(self):
        root = tempfile.mkdtemp()
        self.assertEqual(memory.session_block(root), "")
        memory.save_session(root, "## Goal\nport the parser\n## Open\n- tests for utf-8")
        block = memory.session_block(root)
        self.assertIn("previous session", block)
        self.assertIn("port the parser", block)
        self.assertIn("verify", block)                                     # warns that notes may be stale
        self.assertIsNone(memory.save_session(root, "   "))                # nothing to save, nothing clobbered
        self.assertIn("port the parser", memory.load_session(root))

    def test_oversized_notes_are_cut_on_a_line(self):
        root = tempfile.mkdtemp()
        memory.save_session(root, "\n".join(f"- item {i}" for i in range(2000)))
        self.assertLessEqual(len(memory.load_session(root)), memory.SESSION_MAX + 3)


class TestAgentRestart(unittest.TestCase):
    def agent(self, root):
        import agent
        import llm
        cfg = llm.load_cfg()
        cfg["port"] = 1
        return agent.Agent(cfg, root)

    def test_a_new_session_starts_knowing_the_project_and_the_last_session(self):
        root = project(FILES)
        a = self.agent(root)
        self.assertIn("main.py (6 lines)", a.messages[0]["content"])        # the map is in the prompt
        a.messages += [{"role": "user", "content": "add logging"},
                       {"role": "assistant", "content": "done"},
                       {"role": "user", "content": "thanks"},
                       {"role": "assistant", "content": "ok"}]
        a.oneshot = lambda prompt: "## Goal\nlogging added to app/main.py"
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertTrue(a.save_notes())
        b = self.agent(root)                                                # "restart"
        sys_prompt = b.messages[0]["content"]
        self.assertIn("logging added to app/main.py", sys_prompt)
        self.assertIn("main.py (6 lines)", sys_prompt)

    def test_notes_merge_the_previous_ones_and_trivial_sessions_save_nothing(self):
        root = project(FILES)
        a = self.agent(root)
        seen = []
        a.oneshot = lambda prompt: seen.append(prompt) or "notes"
        self.assertFalse(a.save_notes())                                    # nothing happened yet
        self.assertEqual(seen, [])
        memory.save_session(root, "OLD-NOTES-MARKER")
        a.messages += [{"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1"},
                       {"role": "user", "content": "q2"}, {"role": "assistant", "content": "a2"}]
        with contextlib.redirect_stdout(io.StringIO()):
            a.save_notes()
        self.assertIn("OLD-NOTES-MARKER", seen[0])
        self.assertIn("q2", seen[0])

    def test_a_dead_server_at_exit_does_not_raise_or_clobber(self):
        root = project(FILES)
        memory.save_session(root, "KEEP ME")
        a = self.agent(root)                                                # port 1: nothing listens
        a.messages += [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"},
                       {"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]
        self.assertFalse(a.save_notes(quiet=True))
        self.assertEqual(memory.load_session(root), "KEEP ME")


class TestBigTrees(unittest.TestCase):
    def test_the_home_folder_is_not_a_project(self):
        import time
        with tempfile.TemporaryDirectory() as home:
            os.makedirs(os.path.join(home, "Library", "x"))
            with open(os.path.join(home, "Library", "x", "f.txt"), "w") as f:
                f.write("x")
            with mock.patch.dict(os.environ, {"HOME": home}):
                t0 = time.time()
                self.assertEqual(projectmap._files(home), [])
                self.assertEqual(projectmap.system_block(home), "")
                self.assertLess(time.time() - t0, 1)
            self.assertFalse(os.path.exists(os.path.join(home, ".llm")))      # and nothing is written there

    def test_a_huge_tree_without_git_is_cut_short(self):
        with tempfile.TemporaryDirectory() as d:
            for i in range(30):
                os.makedirs(os.path.join(d, f"pkg{i}"))
                for j in range(40):
                    open(os.path.join(d, f"pkg{i}", f"m{j}.py"), "w").close()
            with mock.patch.object(projectmap, "WALK_MAX_FILES", 100):
                files = projectmap._files(d)
            self.assertLessEqual(len(files), 140)         # stops at the end of the directory that crossed the cap
            self.assertGreater(len(files), 0)


if __name__ == "__main__":
    unittest.main()
