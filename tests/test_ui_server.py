"""ui/server.py end to end: a real UI server on a temp project with a real git repo and a local bare
"remote", and a scripted fake engine behind the chat endpoint.

    .venv/bin/python -m unittest tests.test_ui_server -v
"""
import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from ui.server import Project, UIServer  # noqa: E402


def git(cwd, *args):
    env = dict(os.environ, GIT_AUTHOR_NAME="T", GIT_AUTHOR_EMAIL="t@t", GIT_COMMITTER_NAME="T",
               GIT_COMMITTER_EMAIL="t@t", GIT_CONFIG_GLOBAL="/dev/null", GIT_CONFIG_SYSTEM="/dev/null")
    return subprocess.run(["git", "-C", cwd, *args], capture_output=True, text=True, env=env, check=True).stdout


class FakeEngine(BaseHTTPRequestHandler):
    seen = []

    def log_message(self, *a):
        pass

    def do_GET(self):
        body = json.dumps({"status": "ok", "model": "fake-model", "state": "awake", "context": 4096}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        FakeEngine.seen.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
        events = [{"choices": [{"delta": {"content": "Hel"}}]}, {"choices": [{"delta": {"content": "lo"}}]}]
        body = "".join(f"data: {json.dumps(e)}\n\n" for e in events).encode() + b"data: [DONE]\n\n"
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = ThreadingHTTPServer(("127.0.0.1", 0), FakeEngine)
        threading.Thread(target=cls.engine.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.engine.shutdown()
        cls.engine.server_close()

    def setUp(self):
        FakeEngine.seen.clear()
        self.root = os.path.realpath(tempfile.mkdtemp())
        self.remote = os.path.realpath(tempfile.mkdtemp())
        subprocess.run(["git", "init", "-q", "--bare", self.remote], check=True)
        git(self.root, "init", "-q", "-b", "main")
        git(self.root, "remote", "add", "origin", self.remote)
        self.put("src/app.py", "class App:\n    def run(self):\n        return 1\n")
        self.put("README.md", "# Demo\n\n## Usage\n")
        self.put("data.bin", "\x00\x01\x02")
        git(self.root, "add", "-A")
        git(self.root, "commit", "-q", "-m", "init")
        self.srv = UIServer(Project(self.root), 0, self.engine.server_address[1], token="tok123")
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.base = f"http://127.0.0.1:{self.srv.server_address[1]}"
        self.addCleanup(self.stop)

    def stop(self):
        self.srv.shutdown()
        self.srv.server_close()

    def put(self, rel, text):
        p = os.path.join(self.root, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w") as f:
            f.write(text)

    def call(self, method, path, body=None, token="tok123", headers=None, raw=False):
        h = {"Content-Type": "application/json", **(headers or {})}
        if token:
            h["X-UI-Token"] = token
        req = urllib.request.Request(self.base + path, method=method, headers=h,
                                     data=json.dumps(body).encode() if body is not None else None)
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                data = r.read()
                return r.status, (data if raw else json.loads(data))
        except urllib.error.HTTPError as e:
            data = e.read()
            e.close()
            try:
                return e.code, json.loads(data)
            except ValueError:
                return e.code, data


class TestSecurity(Base):
    def test_the_api_needs_the_token(self):
        self.assertEqual(self.call("GET", "/api/info", token=None)[0], 401)
        self.assertEqual(self.call("GET", "/api/info", token="wrong")[0], 401)
        self.assertEqual(self.call("GET", "/api/info")[0], 200)
        self.assertEqual(self.call("GET", f"/api/info?token=tok123", token=None)[0], 200)

    def test_requests_not_aimed_at_localhost_are_refused(self):
        code, _ = self.call("GET", "/api/info", headers={"Host": "evil.example.com"})
        self.assertEqual(code, 403)
        code, _ = self.call("GET", "/api/info", headers={"Origin": "http://evil.example.com"})
        self.assertEqual(code, 403)
        code, _ = self.call("GET", "/", token=None, headers={"Host": "evil.example.com"})
        self.assertEqual(code, 403)

    def test_paths_cannot_leave_the_project(self):
        for bad in ("../outside.txt", "/etc/passwd", "src/../../x"):
            with self.subTest(bad):
                self.assertEqual(self.call("GET", "/api/file?path=" + urllib.request.quote(bad))[0], 403)
                self.assertEqual(self.call("PUT", "/api/file", {"path": bad, "text": "x"})[0], 403)
        self.assertFalse(os.path.exists(os.path.join(os.path.dirname(self.root), "outside.txt")))

    def test_a_symlink_to_the_outside_is_refused(self):
        outside = tempfile.mkdtemp()
        with open(os.path.join(outside, "secret.txt"), "w") as f:
            f.write("secret")
        os.symlink(outside, os.path.join(self.root, "link"))
        self.assertEqual(self.call("GET", "/api/file?path=link/secret.txt")[0], 403)
        self.assertEqual(self.call("POST", "/api/chat", {"messages": [{"role": "user", "content": "x"}],
                                                         "attachments": ["link/secret.txt"]})[0], 403)

    def test_the_git_folder_is_not_reachable(self):
        self.assertEqual(self.call("GET", "/api/file?path=.git/config")[0], 403)
        self.assertEqual(self.call("PUT", "/api/file", {"path": ".git/hooks/pre-commit", "text": "#!/bin/sh"})[0], 403)
        names = [e["name"] for e in self.call("GET", "/api/tree")[1]["entries"]]
        self.assertNotIn(".git", names)

    def test_there_is_no_web_client_and_no_static_file_serving(self):
        self.assertEqual(self.call("GET", "/static/../server.py", token=None)[0], 404)
        self.assertEqual(self.call("GET", "/static/app.js", token=None)[0], 404)


class TestFiles(Base):
    def test_tree_orders_folders_first(self):
        entries = self.call("GET", "/api/tree")[1]["entries"]
        self.assertEqual([e["name"] for e in entries], ["src", "data.bin", "README.md"])
        sub = self.call("GET", "/api/tree?path=src")[1]["entries"]
        self.assertEqual(sub[0]["path"], os.path.join("src", "app.py"))

    def test_open_file_returns_language_and_outline(self):
        r = self.call("GET", "/api/file?path=src/app.py")[1]
        self.assertEqual(r["language"], "python")
        self.assertEqual(r["outline"][0]["name"], "App")
        self.assertEqual(r["outline"][0]["children"][0]["name"], "run")
        md = self.call("GET", "/api/file?path=README.md")[1]
        self.assertEqual([s["name"] for s in md["outline"][0]["children"]], ["Usage"])

    def test_binary_and_missing_and_large(self):
        self.assertTrue(self.call("GET", "/api/file?path=data.bin")[1]["binary"])
        self.assertEqual(self.call("GET", "/api/file?path=nope.txt")[0], 404)
        self.put("big.txt", "x" * 2_100_000)
        self.assertTrue(self.call("GET", "/api/file?path=big.txt")[1]["too_large"])

    def test_save_is_atomic_and_detects_a_concurrent_edit(self):
        opened = self.call("GET", "/api/file?path=src/app.py")[1]
        code, r = self.call("PUT", "/api/file", {"path": "src/app.py", "text": "x = 1\n", "expected_mtime": opened["mtime"]})
        self.assertEqual(code, 200)
        with open(os.path.join(self.root, "src", "app.py")) as f:
            self.assertEqual(f.read(), "x = 1\n")
        self.assertFalse([n for n in os.listdir(os.path.join(self.root, "src")) if n.endswith(".ui-tmp")])
        # someone else changes the file; saving with the old mtime must not overwrite it
        self.put("src/app.py", "changed elsewhere\n")
        os.utime(os.path.join(self.root, "src", "app.py"), (1, 1))
        code, r = self.call("PUT", "/api/file", {"path": "src/app.py", "text": "mine\n", "expected_mtime": r["mtime"]})
        self.assertEqual(code, 409)
        with open(os.path.join(self.root, "src", "app.py")) as f:
            self.assertEqual(f.read(), "changed elsewhere\n")

    def test_new_files_and_folders_are_created(self):
        self.assertEqual(self.call("PUT", "/api/file", {"path": "new/dir/f.txt", "text": "hi"})[0], 200)
        self.assertTrue(os.path.isfile(os.path.join(self.root, "new", "dir", "f.txt")))

    def test_analyze_unsaved_text(self):
        code, r = self.call("POST", "/api/analyze", {"path": "x.py", "text": "def f():\n    return 'a'\n"})
        self.assertEqual(r["language"], "python")
        kinds = {k for _, _, k in r["spans"]}
        self.assertTrue({"keyword", "string", "function"} <= kinds)
        self.assertEqual(r["outline"][0]["name"], "f")
        _, r = self.call("POST", "/api/analyze", {"text": "# T\n## S\n", "language": "markdown"})
        self.assertEqual(r["outline"][0]["children"][0]["name"], "S")


class TestGit(Base):
    def test_status_lists_changes_with_branch_and_remote(self):
        self.put("src/app.py", "class App:\n    pass\n")
        self.put("new.txt", "n")
        st = self.call("GET", "/api/git/status")[1]
        self.assertEqual((st["repo"], st["branch"], st["remotes"]), (True, "main", ["origin"]))
        by = {f["path"]: f["status"] for f in st["files"]}
        self.assertEqual(by["src/app.py"], "modified")
        self.assertEqual(by["new.txt"], "untracked")
        self.assertIsNone(st["upstream"])

    def test_diff_for_a_modified_and_an_untracked_file(self):
        self.put("src/app.py", "class App:\n    pass\n")
        self.put("new.txt", "brand new\n")
        d = self.call("GET", "/api/git/diff?path=src/app.py")[1]["diff"]
        self.assertIn("-    def run(self):", d)
        self.assertIn("brand new", self.call("GET", "/api/git/diff?path=new.txt")[1]["diff"])

    def test_commit_only_what_was_chosen(self):
        self.put("a.txt", "a")
        self.put("b.txt", "b")
        code, r = self.call("POST", "/api/git/commit", {"message": "add a", "paths": ["a.txt"]})
        self.assertEqual(code, 200)
        self.assertIn("add a", r["commit"])
        self.assertEqual(git(self.root, "show", "--name-only", "--format=", "HEAD").split(), ["a.txt"])
        self.assertEqual([f["path"] for f in r["git"]["files"]], ["b.txt"])      # b is still untracked
        self.assertNotIn("Co-Authored-By", git(self.root, "log", "-1", "--format=%B"))

    def test_commit_validation(self):
        self.put("a.txt", "a")
        self.assertEqual(self.call("POST", "/api/git/commit", {"message": "  ", "paths": ["a.txt"]})[0], 400)
        self.assertEqual(self.call("POST", "/api/git/commit", {"message": "m"})[0], 400)          # nothing chosen
        self.assertEqual(self.call("POST", "/api/git/commit", {"message": "m", "paths": ["../x"]})[0], 403)
        self.assertEqual(self.call("POST", "/api/git/commit", {"message": "m", "paths": ["README.md"]})[0], 500)  # unchanged

    def test_commit_all(self):
        self.put("a.txt", "a")
        self.put("src/app.py", "changed\n")
        self.assertEqual(self.call("POST", "/api/git/commit", {"message": "all", "all": True})[0], 200)
        self.assertEqual(self.call("GET", "/api/git/status")[1]["files"], [])

    def test_push_needs_confirmation_then_reaches_the_remote(self):
        self.assertEqual(self.call("POST", "/api/git/push", {})[0], 400)
        self.assertEqual(self.call("POST", "/api/git/push", {"confirm": "yes"})[0], 400)         # truthy is not True
        self.assertEqual(git(self.remote, "branch", "--list").strip(), "")                       # nothing left the machine
        code, r = self.call("POST", "/api/git/push", {"confirm": True})
        self.assertEqual(code, 200, r)
        self.assertIn("main", git(self.remote, "branch", "--list"))
        self.assertEqual(r["git"]["upstream"], "origin/main")
        self.assertEqual(r["git"]["ahead"], 0)

    def test_ahead_count_and_a_second_push(self):
        self.call("POST", "/api/git/push", {"confirm": True})
        self.put("a.txt", "a")
        self.call("POST", "/api/git/commit", {"message": "more", "paths": ["a.txt"]})
        self.assertEqual(self.call("GET", "/api/git/status")[1]["ahead"], 1)
        self.assertEqual(self.call("POST", "/api/git/push", {"confirm": True})[0], 200)
        self.assertEqual(git(self.remote, "log", "-1", "--format=%s", "main").strip(), "more")

    def test_push_without_a_remote_or_to_an_unknown_one_is_an_error(self):
        self.assertEqual(self.call("POST", "/api/git/push", {"confirm": True, "remote": "nope"})[0], 400)
        git(self.root, "remote", "remove", "origin")
        code, r = self.call("POST", "/api/git/push", {"confirm": True})
        self.assertEqual(code, 500)
        self.assertIn("no remote", r["error"])

    def test_a_folder_that_is_not_a_repo(self):
        plain = os.path.realpath(tempfile.mkdtemp())
        srv = UIServer(Project(plain), 0, 1, token="t")
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        self.addCleanup(lambda: (srv.shutdown(), srv.server_close()))
        req = urllib.request.Request(f"http://127.0.0.1:{srv.server_address[1]}/api/git/status", headers={"X-UI-Token": "t"})
        with urllib.request.urlopen(req) as r:
            self.assertEqual(json.load(r), {"repo": False})


class TestChat(Base):
    def chat(self, body):
        req = urllib.request.Request(self.base + "/api/chat", method="POST", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json", "X-UI-Token": "tok123"})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.status, r.read().decode()
        except urllib.error.HTTPError as e:
            data = e.read().decode()
            e.close()
            return e.code, data

    def test_attachments_are_quoted_into_the_prompt_and_the_stream_is_relayed(self):
        code, out = self.chat({"messages": [{"role": "user", "content": "what does run return?"}],
                               "attachments": ["src/app.py", "README.md"]})
        self.assertEqual(code, 200)
        text = "".join(json.loads(l[6:])["choices"][0]["delta"]["content"]
                       for l in out.splitlines() if l.startswith("data: {"))
        self.assertEqual(text, "Hello")
        self.assertIn("data: [DONE]", out)
        sent = FakeEngine.seen[-1]
        self.assertEqual(sent["messages"][0]["role"], "system")
        user = sent["messages"][-1]["content"]
        self.assertIn("what does run return?", user)
        self.assertIn("File `src/app.py`", user)
        self.assertIn("```python\nclass App:", user)
        self.assertIn("File `README.md`", user)
        self.assertTrue(sent["stream"])

    def test_history_is_kept_and_attachments_only_go_on_the_last_message(self):
        self.chat({"messages": [{"role": "user", "content": "first"}, {"role": "assistant", "content": "ok"},
                                {"role": "user", "content": "second"}], "attachments": ["README.md"]})
        msgs = FakeEngine.seen[-1]["messages"]
        self.assertEqual([m["role"] for m in msgs], ["system", "user", "assistant", "user"])
        self.assertNotIn("README", msgs[1]["content"])
        self.assertIn("README", msgs[3]["content"])

    def test_refused_attachments(self):
        user = [{"role": "user", "content": "x"}]
        self.assertEqual(self.chat({"messages": user, "attachments": ["data.bin"]})[0], 400)
        self.assertEqual(self.chat({"messages": user, "attachments": ["src"]})[0], 400)
        self.assertEqual(self.chat({"messages": user, "attachments": ["../etc/passwd"]})[0], 403)
        self.put("huge.txt", "y" * 150_000)
        self.put("huge2.txt", "y" * 150_000)
        self.assertEqual(self.chat({"messages": user, "attachments": ["huge.txt", "huge2.txt"]})[0], 400)
        self.assertEqual(FakeEngine.seen, [])                      # nothing reached the model

    def test_a_missing_or_bad_conversation(self):
        self.assertEqual(self.chat({"messages": []})[0], 400)
        self.assertEqual(self.chat({"messages": [{"role": "assistant", "content": "x"}]})[0], 400)

    def test_a_stopped_model_is_a_readable_error(self):
        srv = UIServer(Project(self.root), 0, 1, token="t")           # nothing listens on port 1
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        self.addCleanup(lambda: (srv.shutdown(), srv.server_close()))
        req = urllib.request.Request(f"http://127.0.0.1:{srv.server_address[1]}/api/chat", method="POST",
                                     data=json.dumps({"messages": [{"role": "user", "content": "x"}]}).encode(),
                                     headers={"Content-Type": "application/json", "X-UI-Token": "t"})
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            urllib.request.urlopen(req, timeout=10)
        self.assertEqual(ctx.exception.code, 502)
        ctx.exception.close()

    def test_info_reports_the_engine_and_git(self):
        info = self.call("GET", "/api/info")[1]
        self.assertEqual(info["engine"]["model"], "fake-model")
        self.assertEqual(info["git"]["branch"], "main")
        self.assertEqual(info["name"], os.path.basename(self.root))


if __name__ == "__main__":
    unittest.main()
