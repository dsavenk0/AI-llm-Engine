"""The private Python download (a clean Mac with no Python 3.10+), against fake release servers.

    .venv/bin/python -m unittest tests.test_pyruntime -v
"""
import contextlib
import hashlib
import io
import json
import os
import sys
import tarfile
import tempfile
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import llm  # noqa: E402
from engine import pyruntime  # noqa: E402

NAME = "cpython-3.12.7+20241016-aarch64-apple-darwin-install_only.tar.gz"


def make_tgz(members):
    """members: {path: bytes | ("link", target)} -> tar.gz bytes."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as t:
        for path, body in members.items():
            if isinstance(body, tuple):
                ti = tarfile.TarInfo(path)
                ti.type, ti.linkname = tarfile.SYMTYPE, body[1]
                t.addfile(ti)
                continue
            ti = tarfile.TarInfo(path)
            ti.size, ti.mode = len(body), 0o755
            t.addfile(ti, io.BytesIO(body))
    return buf.getvalue()


GOOD = make_tgz({"python/bin/python3": b"#!/bin/sh\necho True\n", "python/lib/x.txt": b"x"})


class Resp:
    def __init__(self, body):
        self.body, self.headers, self.pos = body, {"Content-Length": str(len(body))}, 0

    def __enter__(self): return self
    def __exit__(self, *a): return False

    def read(self, n=-1):
        if n < 0:
            n = len(self.body)
        chunk, self.pos = self.body[self.pos:self.pos + n], self.pos + n
        return chunk


def server(tgz=GOOD, digest=True, sums="match", assets=None):
    sha = hashlib.sha256(tgz).hexdigest()
    asset = {"name": NAME, "browser_download_url": "https://dl/py.tgz"}
    if digest:
        asset["digest"] = "sha256:" + (sha if digest is True else digest)
    release = {"tag_name": "20241016", "assets": assets if assets is not None else [asset]}
    if sums:
        release["assets"] = release["assets"] + [{"name": "SHA256SUMS", "browser_download_url": "https://dl/sums"}]

    def opener(req, timeout=None):
        url = req.full_url
        if url == pyruntime.RELEASES:
            return Resp(json.dumps(release).encode())
        if url.endswith("py.tgz"):
            return Resp(tgz)
        if url.endswith("sums"):
            line = (sha if sums == "match" else "0" * 64) + "  " + NAME + "\n"
            return Resp(line.encode())
        raise AssertionError(url)
    return opener


class TestPick(unittest.TestCase):
    def test_prefers_312_aarch64_install_only(self):
        names = ["cpython-3.13.0+1-aarch64-apple-darwin-install_only.tar.gz",
                 "cpython-3.12.7+1-aarch64-apple-darwin-install_only.tar.gz",
                 "cpython-3.12.7+1-x86_64-apple-darwin-install_only.tar.gz",
                 "cpython-3.12.7+1-aarch64-apple-darwin-debug-full.tar.zst",
                 "cpython-3.9.20+1-aarch64-apple-darwin-install_only.tar.gz"]
        rel = {"tag_name": "t", "assets": [{"name": n, "browser_download_url": "u/" + n, "digest": "sha256:ab"} for n in names]}
        tag, name, url, digest, sums = pyruntime.pick_asset(rel)
        self.assertEqual((tag, name, digest, sums), ("t", names[1], "ab", None))

    def test_nothing_suitable(self):
        self.assertIsNone(pyruntime.pick_asset({"assets": [{"name": "x.tar.gz", "browser_download_url": "u"}]}))
        self.assertIsNone(pyruntime.pick_asset({}))


class TestInstall(unittest.TestCase):
    def setUp(self):
        self.data = tempfile.mkdtemp()

    def install(self, **kw):
        return pyruntime.install(self.data, log=lambda s: None, opener=kw.pop("opener", None) or server(**kw))

    def test_installs_with_the_release_digest(self):
        p = self.install()
        self.assertEqual(p, pyruntime.python_path(self.data))
        self.assertTrue(os.access(p, os.X_OK))
        self.assertEqual(os.listdir(self.data), ["python"])           # no temp leftovers

    def test_installs_with_the_sums_file_when_there_is_no_digest(self):
        self.assertTrue(os.path.exists(self.install(digest=False)))

    def test_refuses_a_checksum_mismatch_and_leaves_nothing(self):
        with self.assertRaisesRegex(pyruntime.PythonInstallError, "checksum"):
            self.install(digest="0" * 64)
        with self.assertRaisesRegex(pyruntime.PythonInstallError, "checksum"):
            self.install(digest=False, sums="wrong")
        self.assertEqual(os.listdir(self.data), [])

    def test_refuses_when_no_checksum_is_published(self):
        with self.assertRaisesRegex(pyruntime.PythonInstallError, "no checksum"):
            self.install(digest=False, sums=None)

    def test_refuses_unsafe_archives(self):
        for members in ({"../evil": b"x"}, {"/abs/evil": b"x"}, {"python/bin/python3": ("link", "/etc/passwd")},
                        {"python/a": ("link", "../../b")}):
            with self.assertRaisesRegex(pyruntime.PythonInstallError, "unsafe"):
                self.install(tgz=make_tgz(members))
        self.assertEqual(os.listdir(self.data), [])

    def test_refuses_an_archive_without_python(self):
        with self.assertRaisesRegex(pyruntime.PythonInstallError, "not in the downloaded"):
            self.install(tgz=make_tgz({"python/lib/x": b"x"}))

    def test_intel_and_missing_builds_are_refused(self):
        with self.assertRaisesRegex(pyruntime.PythonInstallError, "Apple Silicon"):
            pyruntime.install(self.data, log=print, opener=server(), machine="x86_64")
        with self.assertRaisesRegex(pyruntime.PythonInstallError, "no macOS Python"):
            self.install(assets=[])

    def test_an_older_copy_is_replaced(self):
        os.makedirs(os.path.join(self.data, "python", "bin"))
        with open(os.path.join(self.data, "python", "old.txt"), "w") as f:
            f.write("old")
        self.install()
        self.assertFalse(os.path.exists(os.path.join(self.data, "python", "old.txt")))


class TestFindPython(unittest.TestCase):
    def test_uses_the_system_python_when_there_is_one(self):
        with mock.patch.object(llm, "_python_ok", side_effect=lambda p: p == sys.executable), \
                mock.patch.object(pyruntime, "install") as inst:
            self.assertEqual(llm.find_python(), sys.executable)
        inst.assert_not_called()

    def test_downloads_a_private_copy_when_none_is_new_enough(self):
        ok = {"calls": 0}

        def python_ok(p):
            return p == "/data/python/bin/python3" and ok["calls"] > 0

        def install(data, log=print):
            ok["calls"] += 1
            return "/data/python/bin/python3"

        with mock.patch.object(llm, "_python_ok", side_effect=python_ok), \
                mock.patch.object(pyruntime, "install", side_effect=install), \
                mock.patch.object(pyruntime, "python_path", return_value="/data/python/bin/python3"), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(llm.find_python(), "/data/python/bin/python3")

    def test_a_failed_download_ends_with_instructions(self):
        with mock.patch.object(llm, "_python_ok", return_value=False), \
                mock.patch.object(pyruntime, "install", side_effect=pyruntime.PythonInstallError("offline")), \
                contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as cm:
                llm.find_python()
        self.assertIn("offline", str(cm.exception))
        self.assertIn("brew install python", str(cm.exception))

    def test_an_outdated_python_is_not_used_and_the_private_copy_is_downloaded(self):
        def fake_run(cmd, **kw):
            return mock.Mock(returncode=0, stdout="False\n" if cmd[0] != "/data/python/bin/python3" else "True\n")

        installed = []
        with mock.patch("os.path.exists", return_value=True), mock.patch("subprocess.run", side_effect=fake_run), \
                mock.patch.object(pyruntime, "python_path", return_value="/data/python/bin/python3"), \
                mock.patch.object(pyruntime, "install", side_effect=lambda *a, **k: installed.append(1) or "/data/python/bin/python3"), \
                contextlib.redirect_stdout(io.StringIO()):
            # python3 3.9 everywhere: every candidate answers False, so nothing old is picked
            self.assertEqual(llm.find_python(), "/data/python/bin/python3")
        self.assertEqual(installed, [])        # the private copy that is already there is simply used

    def test_the_xcode_stub_is_not_run_without_command_line_tools(self):
        with mock.patch("os.path.exists", return_value=True), \
                mock.patch("subprocess.run", return_value=mock.Mock(returncode=2, stdout="")) as run:
            self.assertFalse(llm._python_ok("/usr/bin/python3"))
        self.assertEqual(run.call_args_list[0][0][0], ["xcode-select", "-p"])
        self.assertEqual(run.call_count, 1)


if __name__ == "__main__":
    unittest.main()
