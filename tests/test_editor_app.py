"""The native editor app, headless: compiled with swiftc and run with --selftest against the real Python
backend on a throw-away git project (no window, no focus, no model).

    .venv/bin/python -m unittest tests.test_editor_app -v
"""
import os
import shutil
import subprocess
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@unittest.skipUnless(shutil.which("swiftc") and shutil.which("python3") and shutil.which("git"), "needs swiftc, python3, git")
class TestEditorApp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.binary = os.path.join(cls.tmp, "LLMEditor")
        files = sorted(os.path.join(ROOT, "app/editor", f) for f in os.listdir(os.path.join(ROOT, "app/editor"))
                       if f.endswith(".swift"))
        build = subprocess.run(["swiftc", "-O", *files, "-o", cls.binary], capture_output=True, text=True, timeout=900)
        if build.returncode != 0:
            raise AssertionError("the editor app does not compile:\n" + build.stderr[-3000:])
        cls.project = os.path.realpath(os.path.join(cls.tmp, "proj"))
        os.makedirs(cls.project)
        with open(os.path.join(cls.project, "hello.py"), "w") as f:
            f.write("class Greeter:\n    def hello(self):\n        return 'hi'\n")
        with open(os.path.join(cls.project, "broken.py"), "w") as f:
            f.write("x = 1\ndef f(:\n    pass\n")
        env = dict(os.environ, GIT_AUTHOR_NAME="T", GIT_AUTHOR_EMAIL="t@t", GIT_COMMITTER_NAME="T",
                   GIT_COMMITTER_EMAIL="t@t", GIT_CONFIG_GLOBAL="/dev/null")
        for args in (["init", "-q", "-b", "main"], ["add", "-A"], ["commit", "-q", "-m", "init"]):
            subprocess.run(["git", "-C", cls.project, *args], check=True, capture_output=True, env=env)

    def test_the_app_drives_the_backend_end_to_end(self):
        run = subprocess.run([self.binary, "--selftest", "--root", self.project, "--llm-root", ROOT, "--engine-port", "1"],
                             capture_output=True, text=True, timeout=120)
        self.assertEqual(run.returncode, 0, run.stdout[-3000:] + run.stderr[-1500:])
        self.assertIn("SELFTEST OK", run.stdout)
        self.assertNotIn("FAIL ", run.stdout)
        with open(os.path.join(self.project, "hello.py")) as f:
            self.assertIn("def added", f.read())                  # the save really reached the disk


if __name__ == "__main__":
    unittest.main()
