"""The editor module of the one LLMBar app, headless: the whole app is compiled with swiftc and run with
`--module editor --selftest` against the real Python backend on a throw-away git project (no window, no focus,
no model).

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
        cls.binary = os.path.join(cls.tmp, "LLMBar")
        files = [os.path.join(ROOT, "app/main.swift")]
        for d in ("core", "bar", "editor", "voice"):
            folder = os.path.join(ROOT, "app", d)
            files += sorted(os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".swift"))
        build = subprocess.run(["swiftc", "-O", *files, "-o", cls.binary], capture_output=True, text=True, timeout=1500)
        if build.returncode != 0:
            raise AssertionError("the app does not compile:\n" + build.stderr[-3000:])
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
        run = subprocess.run([self.binary, "--module", "editor", "--selftest", "--root", self.project, "--llm-root", ROOT,
                              "--engine-port", "1"],
                             capture_output=True, text=True, timeout=120)
        self.assertEqual(run.returncode, 0, run.stdout[-3000:] + run.stderr[-1500:])
        self.assertIn("SELFTEST OK", run.stdout)
        self.assertNotIn("FAIL ", run.stdout)
        with open(os.path.join(self.project, "hello.py")) as f:
            self.assertIn("def added", f.read())                  # the save really reached the disk


class TestModuleLinks(unittest.TestCase):
    """The llmbar:// links are the one way in for the widget, `./llm ui` and other apps."""

    def read(self, *parts):
        with open(os.path.join(ROOT, *parts), encoding="utf-8") as f:
            return f.read()

    def test_the_scheme_is_registered_and_every_route_has_a_handler(self):
        self.assertIn("<string>llmbar</string>", self.read("build-app.sh"))
        host = self.read("app", "core", "Host.swift")
        for route in ('case "toggle"', 'case "editor"', 'case "cli"'):
            self.assertIn(route, host)
        widget = self.read("app", "widget", "LLMBarWidget.swift")
        for link in ("llmbar://toggle", "llmbar://launch?target="):
            self.assertIn(link, widget)

    def test_ui_command_and_the_host_agree_on_the_editor_link(self):
        self.assertIn('"llmbar://editor"', self.read("llm.py"))

    def test_there_is_one_app_not_three(self):
        for gone in ("build-editor.sh", "build-voice.sh"):
            self.assertFalse(os.path.exists(os.path.join(ROOT, gone)), gone)
        self.assertEqual(
            sorted(f for f in os.listdir(os.path.join(ROOT, "app")) if f.endswith(".swift")), ["main.swift"])


if __name__ == "__main__":
    unittest.main()
