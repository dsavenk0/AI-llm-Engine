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


class TestModelRoutes(unittest.TestCase):
    """Which engine a model needs — from a local config.json or from a hub repo's tags."""

    def test_local_configs(self):
        r = llm.engine_route
        self.assertEqual(r({"model_type": "qwen3_5", "text_config": {"model_type": "qwen3_5_text"}}), "native")
        self.assertEqual(r({"model_type": "prism_hadamard_qwen35"}), "native")
        self.assertEqual(r({"model_type": "prism_hadamard_qwen35_ptq1", "schema": "prism_hadamard_qwen35_ptq1_v1"}), "ptq1")
        # inductiveML's lossless 1.75-bit repack: its text_config is plain qwen3_5, the top-level type decides
        self.assertEqual(r({"model_type": "ternel_hadamard_qwen35", "text_config": {"model_type": "qwen3_5_text"}}), "ptq1")
        self.assertEqual(r({"model_type": "qwen3"}), "mlxlm")
        self.assertEqual(r({"model_type": "glm4_moe"}), "mlxlm")
        self.assertEqual(r({"model_type": "qwen3_5", "text_config": {"num_experts": 8}}), "mlxlm")  # MoE → not native

    def test_hub_tags(self):
        h = llm.hub_route
        self.assertEqual(h(["mlx", "safetensors", "prism_hadamard_qwen35", "ternary"]), "native")
        self.assertEqual(h(["custom", "safetensors", "prism_hadamard_qwen35_ptq1", "mlx"]), "ptq1")
        self.assertEqual(h(["mlx", "safetensors", "ternel_hadamard_qwen35", "qwen3_5", "ternary", "vision"]), "ptq1")
        self.assertEqual(h(["mlx", "safetensors", "qwen3"]), "mlxlm")
        self.assertEqual(h(["mlx", "safetensors", "qwen3_5_moe"]), "mlxlm")
        self.assertEqual(h(["gguf", "qwen3_5"]), "none")
        self.assertEqual(h(["pytorch", "llama"]), "none")


class TestRouteBlockers(unittest.TestCase):
    def setUp(self):
        self.cfg = cfg()
        self.cfg["advanced"] = {}
        from unittest import mock
        self.save = mock.patch.object(llm, "save_cfg", lambda c: None)
        self.save.start()
        self.addCleanup(self.save.stop)

    def test_native_is_never_blocked(self):
        self.assertIsNone(llm.route_blocker(self.cfg, {"route": "native"}, False))

    def test_ptq1_is_not_blocked_the_model_is_the_opt_in(self):
        self.assertIsNone(llm.route_blocker(self.cfg, {"route": "ptq1"}, False))

    def test_engine_command_enables_ptq1_by_itself_and_only_for_ptq1(self):
        base = {"name": "m", "path": "/tmp/m", "thinking_default": False, "efforts": []}
        on = llm.engine_cmd(self.cfg, dict(base, route="ptq1"))
        self.assertEqual(on[on.index("--ptq1") + 1], "on")
        self.assertNotIn("--ptq1", llm.engine_cmd(self.cfg, dict(base, route="native")))

    def test_mlxlm_points_at_the_installer_when_missing(self):
        from unittest import mock
        with mock.patch.object(llm, "mlxlm_installed", lambda: False):
            self.assertIn("setup --compat", llm.route_blocker(self.cfg, {"route": "mlxlm"}, False))
        with mock.patch.object(llm, "mlxlm_installed", lambda: True):
            self.assertIsNone(llm.route_blocker(self.cfg, {"route": "mlxlm"}, False))


class TestDeleteModel(unittest.TestCase):
    def setUp(self):
        from unittest import mock
        self.home = os.path.realpath(tempfile.mkdtemp())
        self.cfg = cfg()
        self.cfg.update(dirs=[self.home], model="", models={})
        # no real engine may influence the outcome: delete_model refuses the model a running engine has loaded
        for name, val in (("save_cfg", lambda c: None), ("read_pid", lambda p: None), ("health", lambda c, *a: None)):
            p = mock.patch.object(llm, name, val)
            p.start()
            self.addCleanup(p.stop)

    def make(self, *parts, size=1000):
        d = os.path.join(self.home, *parts)
        os.makedirs(d)
        with open(os.path.join(d, "model.safetensors"), "wb") as f:
            f.write(b"\0" * size)
        return d

    def test_cache_layout_removes_the_whole_models_folder(self):
        snap = self.make("models--org--demo", "snapshots", "abc123")
        os.makedirs(os.path.join(self.home, "models--org--demo", "blobs"))
        self.cfg["models"][snap] = {"context": 4096}
        self.assertTrue(llm.delete_model(self.cfg, {"path": snap, "name": "demo"}, assume_yes=True))
        self.assertFalse(os.path.exists(os.path.join(self.home, "models--org--demo")))
        self.assertTrue(os.path.isdir(self.home))
        self.assertNotIn(snap, self.cfg["models"])

    def test_pull_layout_and_selected_model_is_cleared(self):
        d = self.make("org", "demo")
        self.cfg["model"] = d
        self.assertTrue(llm.delete_model(self.cfg, {"path": d, "name": "demo"}, assume_yes=True))
        self.assertFalse(os.path.exists(d))
        self.assertEqual(self.cfg["model"], "")
        self.assertTrue(os.path.isdir(os.path.join(self.home, "org")))   # siblings untouched

    def test_refuses_outside_model_dirs_and_the_dir_itself(self):
        outside = tempfile.mkdtemp()
        self.assertFalse(llm.delete_model(self.cfg, {"path": outside, "name": "x"}, assume_yes=True))
        self.assertTrue(os.path.isdir(outside))
        self.assertFalse(llm.delete_model(self.cfg, {"path": self.home, "name": "x"}, assume_yes=True))
        self.assertTrue(os.path.isdir(self.home))

    def test_refuses_the_loaded_model_and_asks_otherwise(self):
        from unittest import mock
        d = self.make("org", "demo")
        self.cfg["model"] = d
        with mock.patch.object(llm, "read_pid", lambda p: 1), mock.patch.object(llm, "health", lambda c: {"ok": 1}):
            self.assertFalse(llm.delete_model(self.cfg, {"path": d, "name": "demo"}, assume_yes=True))
        self.assertTrue(os.path.isdir(d))
        with mock.patch.object(llm, "ask_yn", lambda *a, **k: False):
            self.assertFalse(llm.delete_model(self.cfg, {"path": d, "name": "demo"}))
        self.assertTrue(os.path.isdir(d))


class TestConfigKeepsAppSettings(unittest.TestCase):
    """LLMBar writes UI settings into the same file the CLI loads and saves; the CLI must not drop them."""

    def test_ui_keys_survive_a_load_save_roundtrip(self):
        from unittest import mock
        d = tempfile.mkdtemp()
        path = os.path.join(d, "config.json")
        with open(path, "w") as f:
            json.dump({"theme": "light", "glass": True, "glass_transparency": 0.3, "port": 8123, "unknown": 1}, f)
        with mock.patch.object(llm, "CFG_FILE", path), mock.patch.object(llm, "STATE", d):
            c = llm.load_cfg()
            self.assertEqual((c["theme"], c["glass"], c["glass_transparency"], c["port"]), ("light", True, 0.3, 8123))
            llm.save_cfg(c)
            again = llm.load_cfg()
        self.assertEqual((again["theme"], again["glass"], again["glass_transparency"]), ("light", True, 0.3))
        self.assertEqual((llm.DEFAULTS["theme"], llm.DEFAULTS["glass"]), ("system", False))


class TestEditorLauncher(unittest.TestCase):
    def test_find_editor_prefers_the_folder_next_to_the_code(self):
        from unittest import mock
        base = tempfile.mkdtemp()
        res = os.path.dirname(base)
        with mock.patch.object(llm, "BASE", base):
            self.assertIsNone(llm.find_editor() if not os.path.isdir(os.path.join(res, "LLMEditor.app")) else None)
            os.makedirs(os.path.join(base, "LLMEditor.app"))
            self.assertEqual(llm.find_editor(), os.path.join(base, "LLMEditor.app"))

    def test_ui_command_opens_the_app_with_root_and_ports(self):
        from unittest import mock
        calls = []
        base = tempfile.mkdtemp()
        os.makedirs(os.path.join(base, "LLMEditor.app"))
        proj = tempfile.mkdtemp()
        fake = mock.Mock(returncode=0)
        with mock.patch.object(llm, "BASE", base), mock.patch.object(llm.subprocess, "run",
                                                                      lambda cmd, **k: calls.append(cmd) or fake):
            self.assertTrue(llm.run_ui([proj]))
            self.assertTrue(llm.run_ui(["--choose"]))
        first, second = calls
        self.assertEqual(first[:4], ["open", "-n", os.path.join(base, "LLMEditor.app"), "--args"])
        self.assertEqual(first[first.index("--root") + 1], os.path.abspath(proj))
        self.assertEqual(first[first.index("--llm-root") + 1], base)
        self.assertIn("--engine-port", first)
        self.assertNotIn("--root", second)

    def test_missing_editor_is_explained_not_launched(self):
        from unittest import mock
        with mock.patch.object(llm, "BASE", tempfile.mkdtemp()), mock.patch.object(llm, "find_editor", lambda: None):
            ok = llm.run_ui([])
        self.assertFalse(ok)


class TestEditorInTheBundle(unittest.TestCase):
    def test_layout_of_a_bundled_app_is_found_from_the_cli(self):
        """LLMBar.app/Contents/Resources/{LLMEditor.app, llm/} — `llm` runs with BASE = .../Resources/llm."""
        from unittest import mock
        res = tempfile.mkdtemp()
        base = os.path.join(res, "llm")
        os.makedirs(base)
        os.makedirs(os.path.join(res, "LLMEditor.app"))
        with mock.patch.object(llm, "BASE", base):
            self.assertEqual(llm.find_editor(), os.path.join(res, "LLMEditor.app"))

    @unittest.skipUnless(os.path.isdir(os.path.join(ROOT, "LLMBar.app")), "LLMBar.app has not been built here")
    def test_the_built_app_carries_the_editor_and_its_backend(self):
        res = os.path.join(ROOT, "LLMBar.app", "Contents", "Resources")
        self.assertTrue(os.path.isfile(os.path.join(res, "LLMEditor.app", "Contents", "MacOS", "LLMEditor")))
        self.assertTrue(os.path.isfile(os.path.join(res, "LLMEditor.app", "Contents", "Info.plist")))
        for rel in ("ui/server.py", "ui/codeinfo.py", "ui/agentloop.py", "ui/diagnostics.py", "harness/memory.py"):
            self.assertTrue(os.path.isfile(os.path.join(res, "llm", rel)), rel)
