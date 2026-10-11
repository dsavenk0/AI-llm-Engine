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


class TestReviewFixes(Base):
    """Defects found reviewing the server: each of these used to be wrong."""

    def test_symlinked_files_leaving_the_project_are_neither_searched_nor_listed(self):
        outside = tempfile.mkdtemp()
        with open(os.path.join(outside, "secret.txt"), "w") as f:
            f.write("TOP-SECRET-TOKEN\n")
        os.symlink(os.path.join(outside, "secret.txt"), os.path.join(self.root, "notes.txt"))
        self.put("ok.txt", "TOP-SECRET-TOKEN lives here too\n")        # a real file in the project still matches
        hits = self.call("GET", "/api/search?q=TOP-SECRET")[1]["hits"]
        self.assertEqual([h["path"] for h in hits], ["ok.txt"])
        self.assertNotIn("notes.txt", self.call("GET", "/api/files")[1]["files"])
        # the agent's own listing and search go through the same project object
        from ui import agentloop
        tools = agentloop.Tools(Project(self.root), agentloop.Run(), lambda ev: None)
        self.assertNotIn("notes.txt", tools.list_dir("."))
        self.assertNotIn("notes.txt", tools.search("TOP-SECRET"))

    def test_a_symlink_inside_the_project_is_still_fine(self):
        self.put("real.txt", "inside\n")
        os.symlink(os.path.join(self.root, "real.txt"), os.path.join(self.root, "alias.txt"))
        self.assertIn("alias.txt", self.call("GET", "/api/files")[1]["files"])

    def test_non_utf8_files_are_not_opened_as_text(self):
        with open(os.path.join(self.root, "legacy.txt"), "wb") as f:
            f.write("привет".encode("cp1251"))
        code, r = self.call("GET", "/api/file?path=legacy.txt")
        self.assertEqual(code, 200)
        self.assertTrue(r.get("not_utf8"))
        self.assertNotIn("text", r)                 # nothing the editor could save back with U+FFFD in it

    def test_the_agent_does_not_edit_a_file_it_cannot_decode(self):
        from ui import agentloop
        with open(os.path.join(self.root, "legacy.txt"), "wb") as f:
            f.write(b"caf\xe9 latte\n")
        tools = agentloop.Tools(Project(self.root), agentloop.Run(), lambda ev: None, auto_edit=True)
        out = tools.edit_file("legacy.txt", "latte", "mocha")
        self.assertTrue(out.startswith("error:") and "UTF-8" in out, out)
        with open(os.path.join(self.root, "legacy.txt"), "rb") as f:
            self.assertEqual(f.read(), b"caf\xe9 latte\n")

    def test_saving_keeps_the_executable_bit_and_leaves_no_temp_file(self):
        self.put("run.sh", "#!/bin/sh\necho 1\n")
        os.chmod(os.path.join(self.root, "run.sh"), 0o755)
        code, r = self.call("PUT", "/api/file", {"path": "run.sh", "text": "#!/bin/sh\necho 2\n"})
        self.assertEqual(code, 200)
        self.assertEqual(os.stat(os.path.join(self.root, "run.sh")).st_mode & 0o777, 0o755)
        self.assertEqual([n for n in os.listdir(self.root) if n.endswith(".ui-tmp")], [])

    def test_git_status_reads_cyrillic_and_awkward_names(self):
        for name in ("заметки.txt", "a -> b.txt", 'quo"te.txt'):
            self.put(name, "x\n")
        paths = {f["path"] for f in self.call("GET", "/api/git/status")[1]["files"]}
        self.assertEqual(paths, {"заметки.txt", "a -> b.txt", 'quo"te.txt'})

    def test_a_rename_is_one_entry_with_the_new_name(self):
        git(self.root, "mv", "README.md", "README2.md")
        files = self.call("GET", "/api/git/status")[1]["files"]
        self.assertEqual([f["path"] for f in files], ["README2.md"])
        self.assertEqual(files[0]["status"], "renamed")

    def test_committing_chosen_paths_leaves_other_staged_files_alone(self):
        self.put("a.txt", "a")
        self.put("b.txt", "b")
        git(self.root, "add", "b.txt")                       # staged earlier, not part of this commit
        code, r = self.call("POST", "/api/git/commit", {"message": "only a", "paths": ["a.txt"]})
        self.assertEqual(code, 200, r)
        self.assertEqual(git(self.root, "show", "--name-only", "--format=", "HEAD").split(), ["a.txt"])
        self.assertEqual(git(self.root, "diff", "--cached", "--name-only").split(), ["b.txt"])

    def test_push_branch_is_a_name_not_an_option(self):
        for bad in ("--force", "-f", "--delete", "has space", "a..b"):
            with self.subTest(bad):
                code, r = self.call("POST", "/api/git/push", {"confirm": True, "branch": bad})
                self.assertEqual(code, 400, r)
        self.assertEqual(git(self.remote, "branch", "--list").strip(), "")

    def test_bad_requests_get_an_answer_not_a_dropped_connection(self):
        # a body that parses but has the wrong shape used to raise out of the handler
        code, r = self.call("POST", "/api/agent", {"messages": [{"role": "user"}]})
        self.assertEqual(code, 400, r)
        code, r = self.call("POST", "/api/chat", {"messages": ["not an object"]})
        self.assertIn(code, (400, 500))
        import http.client
        conn = http.client.HTTPConnection("127.0.0.1", self.srv.server_address[1], timeout=5)
        conn.request("POST", "/api/analyze", body=b"{}", headers={"X-UI-Token": "tok123", "Content-Length": "-5",
                                                                 "Content-Type": "application/json"})
        resp = conn.getresponse()
        self.assertEqual(resp.status, 400)
        conn.close()


class TestBranches(Base):
    def test_listing_and_switching(self):
        code, r = self.call("GET", "/api/git/branches")
        self.assertEqual((code, r["current"], r["local"]), (200, "main", ["main"]))
        code, r = self.call("POST", "/api/git/checkout", {"branch": "feature/x", "create": True})
        self.assertEqual(code, 200, r)
        self.assertEqual(r["branches"]["current"], "feature/x")
        self.assertEqual(r["git"]["branch"], "feature/x")
        code, r = self.call("POST", "/api/git/checkout", {"branch": "main"})
        self.assertEqual((code, r["branches"]["current"]), (200, "main"))
        self.assertEqual(sorted(self.call("GET", "/api/git/branches")[1]["local"]), ["feature/x", "main"])

    def test_bad_names_and_unknown_branches_are_refused(self):
        for bad in ("", "--orphan", "-b", "a..b", "with space"):
            with self.subTest(bad):
                self.assertEqual(self.call("POST", "/api/git/checkout", {"branch": bad})[0], 400)
        self.assertEqual(self.call("POST", "/api/git/checkout", {"branch": "nope"})[0], 400)
        self.assertEqual(self.call("POST", "/api/git/checkout", {"branch": "main", "create": True})[0], 400)

    def test_git_refuses_to_lose_local_changes_and_says_so(self):
        self.call("POST", "/api/git/checkout", {"branch": "other", "create": True})
        self.put("README.md", "# on other\n")
        git(self.root, "commit", "-qam", "other")
        self.call("POST", "/api/git/checkout", {"branch": "main"})
        self.put("README.md", "# local edit\n")                       # conflicts with the commit on 'other'
        code, r = self.call("POST", "/api/git/checkout", {"branch": "other"})
        self.assertEqual(code, 500)
        self.assertIn("README.md", r["error"])
        self.assertEqual(self.call("GET", "/api/git/branches")[1]["current"], "main")

    def test_a_remote_branch_gets_a_tracking_local_branch(self):
        git(self.root, "push", "-q", "origin", "main")
        git(self.root, "checkout", "-q", "-b", "side")
        self.put("side.txt", "s")
        git(self.root, "add", "-A")
        git(self.root, "commit", "-qm", "side")
        git(self.root, "push", "-q", "origin", "side")
        git(self.root, "checkout", "-q", "main")
        git(self.root, "branch", "-q", "-D", "side")
        git(self.root, "fetch", "-q", "origin")
        info = self.call("GET", "/api/git/branches")[1]
        self.assertIn("origin/side", info["remote"])
        code, r = self.call("POST", "/api/git/checkout", {"branch": "origin/side"})
        self.assertEqual((code, r["branches"]["current"]), (200, "side"))


class TestGitHub(Base):
    """`gh` is replaced by a script that records its arguments: no network, no account."""

    def setUp(self):
        super().setUp()
        self.bin = tempfile.mkdtemp()
        self.log = os.path.join(self.bin, "calls.txt")
        script = os.path.join(self.bin, "gh")
        with open(script, "w") as f:
            f.write('#!/bin/sh\necho "$@" >> "%s"\n'
                    'case "$1 $2" in\n'
                    '  "api user") echo octocat ;;\n'
                    '  "pr list") echo \'[{"number":7,"title":"Fix","headRefName":"fix","url":"https://github.com/o/r/pull/7",'
                    '"isDraft":false,"author":{"login":"octocat"}}]\' ;;\n'
                    '  "pr create") echo "https://github.com/o/r/pull/8" ;;\n'
                    '  "repo create") echo "https://github.com/octocat/$3" ;;\n'
                    'esac\n' % self.log)
        os.chmod(script, 0o755)
        from unittest import mock
        p = mock.patch.dict(os.environ, {"PATH": self.bin + os.pathsep + os.environ["PATH"]})
        p.start()
        self.addCleanup(p.stop)

    def calls(self):
        return open(self.log).read().splitlines() if os.path.exists(self.log) else []

    def test_status_reads_the_login_and_the_repo_from_the_remote(self):
        git(self.root, "remote", "set-url", "origin", "git@github.com:octocat/hello.git")
        code, r = self.call("GET", "/api/github/status")
        self.assertEqual(code, 200)
        self.assertEqual((r["installed"], r["authenticated"], r["user"], r["repo"]),
                         (True, True, "octocat", "octocat/hello"))

    def test_without_gh_everything_says_so_and_status_still_answers(self):
        from unittest import mock
        with mock.patch.dict(os.environ, {"PATH": "/usr/bin:/bin"}):
            code, r = self.call("GET", "/api/github/status")
            self.assertEqual((code, r["installed"], r["authenticated"]), (200, False, False))
            code, r = self.call("GET", "/api/github/prs")
            self.assertEqual(code, 500)
            self.assertIn("brew install gh", r["error"])

    def test_pull_requests_are_listed(self):
        code, r = self.call("GET", "/api/github/prs")
        self.assertEqual((code, r["prs"][0]["number"]), (200, 7))

    def test_creating_a_pr_or_a_repo_needs_confirmation_and_valid_input(self):
        self.assertEqual(self.call("POST", "/api/github/pr", {"title": "x"})[0], 400)
        self.assertEqual(self.call("POST", "/api/github/pr", {"title": "x", "confirm": "yes"})[0], 400)
        self.assertEqual(self.call("POST", "/api/github/pr", {"title": "  ", "confirm": True})[0], 400)
        self.assertEqual(self.call("POST", "/api/github/pr", {"title": "x", "confirm": True, "base": "--force"})[0], 400)
        self.assertEqual(self.call("POST", "/api/github/create_repo", {"name": "r"})[0], 400)
        self.assertEqual(self.call("POST", "/api/github/create_repo", {"name": "bad name!", "confirm": True})[0], 400)
        self.assertFalse([c for c in self.calls() if c.startswith(("pr create", "repo create"))])   # nothing left the machine

    def test_a_confirmed_pr_runs_gh_with_exactly_those_arguments(self):
        code, r = self.call("POST", "/api/github/pr", {"title": "Add thing", "body": "why", "base": "main",
                                                       "draft": True, "confirm": True})
        self.assertEqual((code, r["url"]), (200, "https://github.com/o/r/pull/8"))
        self.assertIn("pr create --title Add thing --body why --base main --draft", self.calls())

    def test_a_repo_is_not_created_over_an_existing_origin(self):
        code, r = self.call("POST", "/api/github/create_repo", {"name": "again", "confirm": True})
        self.assertEqual(code, 400)                    # the test project already has origin
        self.assertIn("origin", r["error"])


class TestPicturesAndOutsideFiles(Base):
    PNG = ("data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==")

    def chat(self, **extra):
        body = {"messages": [{"role": "user", "content": "look"}], **extra}
        return self.call("POST", "/api/chat", body, raw=True)

    def test_inline_text_is_quoted_like_a_project_file(self):
        code, _ = self.chat(inline=[{"name": "/Users/x/Downloads/notes.md", "text": "# outside\n"}])
        self.assertEqual(code, 200)
        content = FakeEngine.seen[-1]["messages"][-1]["content"]
        self.assertIn("notes.md", content)
        self.assertNotIn("/Users/x/Downloads", content)           # only the base name travels
        self.assertIn("# outside", content)

    def test_images_become_image_parts_for_the_engine(self):
        code, _ = self.chat(images=[{"name": "shot.png", "data": self.PNG}])
        self.assertEqual(code, 200)
        content = FakeEngine.seen[-1]["messages"][-1]["content"]
        self.assertIsInstance(content, list)
        self.assertEqual(content[0], {"type": "text", "text": "look"})
        self.assertEqual(content[1]["image_url"]["url"], self.PNG)

    def test_bad_images_and_too_much_are_refused(self):
        self.assertEqual(self.chat(images=[{"name": "x", "data": "data:text/html;base64,PGI+"}])[0], 400)
        self.assertEqual(self.chat(images=[{"name": "x", "data": "http://evil/x.png"}])[0], 400)
        self.assertEqual(self.chat(images=[{"name": f"{i}.png", "data": self.PNG} for i in range(7)])[0], 400)
        big = "x" * (ATTACH_LIMIT + 10)
        self.assertEqual(self.chat(inline=[{"name": "big.txt", "text": big}])[0], 400)


ATTACH_LIMIT = 200_000


class TestExitWithParent(unittest.TestCase):
    def test_the_backend_follows_the_named_process_out(self):
        """The editor app passes its own pid: if the app is killed the backend must not stay behind with
        its token and its agent runs."""
        import time
        root = tempfile.mkdtemp()
        parent = subprocess.Popen(["sleep", "60"])
        srv = subprocess.Popen([sys.executable, "-B", "-m", "ui.server", "--root", root,
                                "--exit-with-parent", str(parent.pid)], cwd=ROOT,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        try:
            self.assertIn("Editor UI", srv.stdout.readline())          # it is up
            time.sleep(1.2)
            self.assertIsNone(srv.poll(), "it exited while its parent was alive")
            parent.kill()
            parent.wait()
            try:
                srv.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.fail("the backend outlived its parent")
        finally:
            srv.kill()
            parent.kill()


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
