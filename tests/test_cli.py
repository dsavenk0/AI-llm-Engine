"""CLI tests (stdlib only): client config writers and translations.

    .venv/bin/python -m unittest tests.test_cli -v
"""
import ast
import json
import os
import re
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import llm  # noqa: E402

MODEL = {"name": "Demo-Model", "path": "/tmp/demo", "size": 4.0, "vision": True, "thinking": True}


def cfg():
    c = llm.load_cfg()
    c.update(port=8090, api_key="", mem_limit=88, models={})
    return c


class TestContinueConfig(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.path = os.path.join(self.dir, "config.yaml")

    def write(self):
        llm._write_continue(self.path, cfg(), MODEL, 40960, 8192)
        with open(self.path) as f:
            return f.read()

    def test_new_file(self):
        text = self.write()
        self.assertIn("schema: v1", text)
        self.assertIn("apiBase: http://localhost:8090/v1", text)
        self.assertIn("contextLength: 40960", text)
        self.assertIn("capabilities: [tool_use, image_input]", text)

    def test_keeps_other_models_and_is_idempotent(self):
        with open(self.path, "w") as f:
            f.write("name: Mine\nversion: 1\nschema: v1\nmodels:\n  - name: Other\n    provider: ollama\n"
                    "    model: qwen\nrules:\n  - be nice\n")
        first = self.write()
        second = self.write()
        self.assertEqual(first, second)
        self.assertEqual(first.count(llm.BEGIN), 1)
        self.assertIn("  - name: Other", first)
        self.assertIn("rules:\n  - be nice", first)
        self.assertTrue(os.path.exists(self.path + ".bak"))
        # our block sits inside the models list (indented under "models:")
        self.assertLess(first.index("models:"), first.index(llm.BEGIN))
        self.assertLess(first.index(llm.END), first.index("rules:"))

    def test_empty_models_list(self):
        with open(self.path, "w") as f:
            f.write("name: x\nversion: 1\nschema: v1\nmodels: []\n")
        text = self.write()
        self.assertNotIn("models: []", text)
        self.assertIn("models:\n  " + llm.BEGIN, text)


class TestOpencodeStyleConfig(unittest.TestCase):
    def test_kilo_aliases_legacy_provider_and_keeps_others(self):
        d = tempfile.mkdtemp()
        p = os.path.join(d, "kilo.jsonc")
        with open(p, "w") as f:
            f.write('{\n  // comment\n  "provider": {"local_qwen": {}, "other": {"x": 1},},\n  "theme": "dark"\n}\n')
        llm._write_opencode_style(p, cfg(), MODEL, 40960, 8192, alias=("local_qwen",))
        with open(p) as f:
            data = json.load(f)
        # local_qwen mirrors local_mlx (old Kilo sessions still resolve it), not removed
        self.assertEqual(data["provider"]["local_qwen"], data["provider"]["local_mlx"])
        self.assertEqual(data["provider"]["other"], {"x": 1})
        self.assertEqual(data["theme"], "dark")
        m = data["provider"]["local_mlx"]["models"]["demo-model"]
        self.assertEqual(m["limit"], {"context": 40960, "output": 8192})
        self.assertEqual(data["model"], "local_mlx/demo-model")


class TestTranslations(unittest.TestCase):
    def test_every_ui_string_has_russian(self):
        with open(os.path.join(ROOT, "locales", "ru.json"), encoding="utf-8") as f:
            ru = json.load(f)
        keys = set()
        for name in ("llm.py", "agent.py"):
            with open(os.path.join(ROOT, name)) as f:
                src = f.read()
            for n in ast.walk(ast.parse(src)):
                if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "T" and n.args \
                        and isinstance(n.args[0], ast.Constant):
                    keys.add(n.args[0].value)
            keys |= set(re.findall(r'\("([A-Z][^"]+)", (?:lambda|logs|run_tests)', src))
        self.assertEqual(sorted(keys - set(ru)), [])
        for k, v in ru.items():
            self.assertEqual(set(re.findall(r"\{(\w+)", k)), set(re.findall(r"\{(\w+)", v)), k)


if __name__ == "__main__":
    unittest.main()


class TestDownloadFile(unittest.TestCase):
    """download_file() against a local server that misbehaves the way real CDNs do."""

    @classmethod
    def setUpClass(cls):
        import hashlib
        import http.server
        import threading

        cls.data = os.urandom(300_000)
        cls.sha = hashlib.sha256(cls.data).hexdigest()
        outer = cls

        class H(http.server.BaseHTTPRequestHandler):
            log = []
            cut = []        # per request: how many bytes to send before dropping the connection (None = all)
            ignore_range = False

            def log_message(self, *a):
                pass

            def do_GET(self):
                rng = self.headers.get("Range")
                H.log.append(rng)
                start = int(rng.split("=")[1].rstrip("-")) if rng and not H.ignore_range else 0
                body = outer.data[start:]
                self.send_response(206 if start else 200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                cut = H.cut.pop(0) if H.cut else None
                if cut is not None:
                    body = body[:cut]
                self.wfile.write(body)
                self.wfile.flush()
                self.close_connection = True  # a short body with a full Content-Length: the early close

        cls.H = H
        cls.srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
        cls.url = f"http://127.0.0.1:{cls.srv.server_address[1]}/f.bin"
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()

    def setUp(self):
        import tempfile
        from unittest import mock
        self.dir = tempfile.mkdtemp()
        self.path = os.path.join(self.dir, "f.bin")
        self.H.log.clear()
        self.H.cut.clear()
        self.H.ignore_range = False
        p = mock.patch.object(llm.time, "sleep", lambda s: None)  # no backoff delays in tests
        p.start()
        self.addCleanup(p.stop)

    def read(self, path=None):
        with open(path or self.path, "rb") as f:
            return f.read()

    def test_a_connection_that_closes_early_is_resumed_not_accepted(self):
        self.H.cut.append(120_000)  # the first response ends 120 KB in, with no error raised client-side
        llm.download_file(self.url, self.path, len(self.data), self.sha)
        self.assertEqual(self.read(), self.data)
        self.assertEqual(self.H.log, [None, "bytes=120000-"])
        self.assertFalse(os.path.exists(self.path + ".part"))

    def test_a_short_file_under_the_final_name_is_resumed(self):
        # what an earlier version of the downloader left behind after accepting a short read
        with open(self.path, "wb") as f:
            f.write(self.data[:50_000])
        llm.download_file(self.url, self.path, len(self.data), self.sha)
        self.assertEqual(self.read(), self.data)
        self.assertEqual(self.H.log, ["bytes=50000-"])

    def test_a_server_that_ignores_range_does_not_get_its_answer_appended(self):
        with open(self.path + ".part", "wb") as f:
            f.write(self.data[:50_000])
        self.H.ignore_range = True
        llm.download_file(self.url, self.path, len(self.data), self.sha)
        self.assertEqual(self.read(), self.data)  # not 50 KB + the whole file

    def test_a_wrong_checksum_is_refused_and_leaves_no_final_file(self):
        with self.assertRaises(ValueError):
            llm.download_file(self.url, self.path, len(self.data), "0" * 64)
        self.assertFalse(os.path.exists(self.path))

    def test_a_server_that_always_cuts_off_fails_loudly(self):
        self.H.cut.extend([10_000] * 10)
        with self.assertRaises(OSError):
            llm.download_file(self.url, self.path, len(self.data), self.sha, retries=2)
        self.assertFalse(os.path.exists(self.path))  # never a short file under the final name

    def test_unknown_size_still_downloads(self):
        llm.download_file(self.url, self.path, 0)
        self.assertEqual(self.read(), self.data)

    def test_complete_file_is_left_alone(self):
        with open(self.path, "wb") as f:
            f.write(self.data)
        llm.download_file(self.url, self.path, len(self.data), self.sha)
        self.assertEqual(self.H.log, [])  # nothing fetched
