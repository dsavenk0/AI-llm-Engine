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
import urllib.parse

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


class TestClientsKnowEveryInstalledModel(unittest.TestCase):
    """Kilo said "Model not found: local_qwen/dealignai/Bonsai-…": its config listed only the model that
    was selected when it was connected."""

    def setUp(self):
        from unittest import mock
        self.home = tempfile.mkdtemp()
        hub = os.path.join(self.home, ".cache", "huggingface", "hub")
        self.models = []
        for org, name in (("dealignai", "Bonsai-2-27B"), ("mlx-community", "Qwen3-0.6B-4bit")):
            d = os.path.join(hub, org, name)
            os.makedirs(d)
            self.models.append({"name": name, "path": d, "size": 1.0, "vision": False, "thinking": True,
                                "supported": True})
        self.cfg = cfg()
        self.cfg["dirs"] = [hub]
        self.cfg["model"] = self.models[1]["path"]
        for patch in (mock.patch.object(llm, "HOME", self.home),
                      mock.patch.object(llm, "scan_models", lambda c: list(self.models)),
                      mock.patch.object(llm, "current_model", lambda c: self.models[1]),
                      mock.patch.object(llm, "health", lambda c, t=3: None)):
            patch.start()
            self.addCleanup(patch.stop)

    def write(self, which="kilo"):
        p = os.path.join(self.home, "kilo.jsonc")
        cur = self.models[1]
        ctx, out = llm.client_limits(self.cfg, cur)
        llm._write_opencode_style(p, self.cfg, cur, ctx, out, alias=("local_qwen",),
                                  others=llm._other_models(self.cfg, cur))
        with open(p) as f:
            return json.load(f), p

    def test_every_installed_model_is_offered_under_both_ids(self):
        data, _ = self.write()
        models = data["provider"]["local_mlx"]["models"]
        self.assertIn("qwen3-0.6b-4bit", models)
        self.assertIn("bonsai-2-27b", models)                                  # the other installed model
        self.assertIn("dealignai/Bonsai-2-27B", models)                        # …and the id older versions used
        self.assertEqual(data["provider"]["local_qwen"], data["provider"]["local_mlx"])
        self.assertEqual(data["model"], "local_mlx/qwen3-0.6b-4bit")           # the selected one stays the default

    def test_sync_refreshes_a_connected_client_and_never_creates_one(self):
        from unittest import mock
        p = os.path.join(self.home, "kilo.jsonc")
        with mock.patch.dict(llm.CLIENTS, {"kilo": ("Kilo Code", p), "opencode": ("opencode", os.path.join(self.home, "oc.json"))}):
            self.assertEqual(llm.sync_clients(self.cfg), [])
            self.assertFalse(os.path.exists(p))
            with open(p, "w") as f:
                json.dump({"provider": {"local_mlx": {"models": {"old": {}}}}, "theme": "dark"}, f)
            self.assertEqual(llm.sync_clients(self.cfg), ["Kilo Code"])
            with open(p, encoding="utf-8") as f:
                data = json.load(f)
        self.assertIn("bonsai-2-27b", data["provider"]["local_mlx"]["models"])
        self.assertNotIn("old", data["provider"]["local_mlx"]["models"])
        self.assertEqual(data["theme"], "dark")


class TestNetworkAndEngineFlags(unittest.TestCase):
    def test_proxy_and_ca_settings_become_the_variables_urllib_and_ssl_read(self):
        c = cfg()
        c.update(proxy_http="http://proxy.local:8080", proxy_https="http://proxy.local:8443",
                 no_proxy="*.internal", ca_bundle="/etc/ssl/corp.pem")
        env = llm.network_env(c, {"PATH": "/usr/bin"})
        self.assertEqual((env["HTTP_PROXY"], env["http_proxy"]), ("http://proxy.local:8080",) * 2)
        self.assertEqual(env["HTTPS_PROXY"], "http://proxy.local:8443")
        self.assertEqual(env["SSL_CERT_FILE"], "/etc/ssl/corp.pem")
        self.assertIn("*.internal", env["NO_PROXY"])
        for host in ("localhost", "127.0.0.1"):      # the engine on this Mac is never sent through the proxy
            self.assertIn(host, env["NO_PROXY"].split(","))

    def test_nothing_set_changes_nothing(self):
        c = cfg()
        c.update(proxy_http="", proxy_https="", no_proxy="", ca_bundle="")
        self.assertEqual(llm.network_env(c, {"PATH": "/x", "HTTP_PROXY": "keep"}), {"PATH": "/x", "HTTP_PROXY": "keep"})

    def test_engine_flags_follow_the_settings_and_stay_out_of_the_command_by_default(self):
        c = cfg()
        c.update(log_level="info", fallback_model=True, prefill_guard=True)
        m = {"name": "m", "path": "/tmp/m", "size": 1.0, "vision": False, "thinking": False, "route": "native"}
        base = llm.engine_cmd(c, m)
        for flag in ("--log-level", "--fallback-model", "--prefill-guard"):
            self.assertNotIn(flag, base)
        c.update(log_level="debug", fallback_model=False, prefill_guard=False)
        cmd = llm.engine_cmd(c, m)
        self.assertEqual(cmd[cmd.index("--log-level") + 1], "debug")
        self.assertEqual(cmd[cmd.index("--fallback-model") + 1], "off")
        self.assertEqual(cmd[cmd.index("--prefill-guard") + 1], "off")


class TestSteerFlags(unittest.TestCase):
    def test_a_models_edit_reaches_the_engine_command_on_the_native_route_only(self):
        c = cfg()
        m = {"name": "m", "path": "/tmp/m", "size": 1.0, "vision": False, "thinking": False, "route": "native"}
        self.assertNotIn("--steer", llm.engine_cmd(c, m))
        c["models"]["/tmp/m"] = {"steer": "/edits/dir.safetensors", "steer_alpha": 1.5}
        cmd = llm.engine_cmd(c, m)
        self.assertEqual(cmd[cmd.index("--steer") + 1], "/edits/dir.safetensors")
        self.assertEqual(cmd[cmd.index("--steer-alpha") + 1], "1.5")
        c["models"]["/tmp/m"] = {"steer": "/edits/dir.safetensors"}
        self.assertNotIn("--steer-alpha", llm.engine_cmd(c, m))               # the edit's own recommendation applies
        self.assertNotIn("--steer", llm.engine_cmd(c, dict(m, route="mlxlm")))


class TestSteerCommand(unittest.TestCase):
    """`./llm steer <repo>`: the edit, and the base model it was made for, are fetched and set up."""

    def setUp(self):
        import tempfile
        from unittest import mock
        self.mock = mock
        self.tmp = tempfile.mkdtemp()
        self.edit = os.path.join(self.tmp, "edit")
        os.makedirs(os.path.join(self.edit, "runtime-adapter"))
        with open(os.path.join(self.edit, "direction.json"), "w") as f:
            json.dump({"model": "acme/base-27b", "hidden_size": 8}, f)
        with open(os.path.join(self.edit, "runtime-adapter", "adapter.json"), "w") as f:
            json.dump({"alpha": 1.5, "base_revision": "3f926b415992eaa2ae9dd7b573706494d6bbf787"}, f)
        import numpy as np
        import mlx.core as mx
        mx.save_safetensors(os.path.join(self.edit, "refusal_dir.safetensors"),
                            {"direction": mx.array(np.ones(8, np.float32))})
        self.base_dir = os.path.join(self.tmp, "hub", "acme", "base-27b")
        self.entry = {"name": "base-27b", "path": self.base_dir, "supported": True, "route": "native",
                      "problems": [], "efforts": [], "size": 7.0, "vision": False, "thinking": False}
        self.cfg = cfg()
        self.cfg["models"] = {}
        self.cfg["model"] = ""
        self.pulls = []
        self.installed = False

        def pull(repo, progress=False, revision=None):
            self.pulls.append((repo, revision))
            self.installed = True
            os.makedirs(self.base_dir, exist_ok=True)
            return True
        self.patches = [
            mock.patch.object(llm, "hf_pull", pull),
            mock.patch.object(llm, "install_state", lambda repo, siblings=None: "complete" if self.installed else "none"),
            mock.patch.object(llm, "_pull_dest", lambda repo: self.base_dir),
            mock.patch.object(llm, "scan_models", lambda c: [self.entry] if self.installed else []),
            mock.patch.object(llm, "save_cfg", lambda c: None),
            mock.patch.object(llm, "read_pid", lambda p: None),
            mock.patch.object(llm, "sync_clients", lambda c: None),
            mock.patch.object(llm, "current_model",
                              lambda c: self.entry if c.get("model") == self.base_dir else None),
        ]
        for p in self.patches:
            p.start()
            self.addCleanup(p.stop)

    def test_the_missing_base_is_downloaded_at_its_pinned_revision_selected_and_given_the_edit(self):
        self.assertTrue(llm.steer_cmd(self.cfg, [self.edit]))
        self.assertEqual(self.pulls, [("acme/base-27b", "3f926b415992eaa2ae9dd7b573706494d6bbf787")])
        self.assertEqual(self.cfg["model"], self.base_dir)                       # selected
        mc = self.cfg["models"][self.base_dir]
        self.assertEqual(mc["steer"], os.path.realpath(self.edit))
        self.assertIsNone(mc["steer_alpha"])                                     # the edit's own recommendation applies

    def test_a_base_downloaded_by_another_tool_in_the_hub_cache_layout_is_found(self):
        layered = os.path.join(self.tmp, "hub", "models--acme--base-27b", "snapshots", "abc123")
        os.makedirs(layered)
        self.entry["path"] = layered
        self.installed = True
        self.assertTrue(llm.steer_cmd(self.cfg, [self.edit]))
        self.assertEqual(self.pulls, [])                         # not fetched a second time into a flat folder
        self.assertEqual(self.cfg["model"], layered)

    def test_an_installed_base_is_not_downloaded_again_and_alpha_can_be_given(self):
        self.installed = True
        os.makedirs(self.base_dir, exist_ok=True)
        self.assertTrue(llm.steer_cmd(self.cfg, [self.edit, "--alpha", "0.8"]))
        self.assertEqual(self.pulls, [])
        self.assertEqual(self.cfg["models"][self.base_dir]["steer_alpha"], 0.8)

    def test_a_base_on_another_route_is_refused_and_nothing_is_set(self):
        self.installed = True
        self.entry["route"] = "mlxlm"
        self.assertFalse(llm.steer_cmd(self.cfg, [self.edit]))
        self.assertEqual(self.cfg["models"], {})

    def test_off_clears_the_edit_of_the_selected_model(self):
        self.installed = True
        self.assertTrue(llm.steer_cmd(self.cfg, [self.edit]))
        self.assertTrue(llm.steer_cmd(self.cfg, ["off"]))
        self.assertEqual(self.cfg["models"][self.base_dir]["steer"], "")

    def test_a_pinned_revision_must_look_like_a_commit_hash(self):
        with open(os.path.join(self.edit, "runtime-adapter", "adapter.json"), "w") as f:
            json.dump({"alpha": 1.0, "base_revision": "../../etc"}, f)
        self.assertEqual(llm.steer_base(self.edit), ("acme/base-27b", None))
        with open(os.path.join(self.edit, "direction.json"), "w") as f:
            json.dump({"model": "../x"}, f)
        self.assertEqual(llm.steer_base(self.edit)[0], None)


class TestRouteFeatures(unittest.TestCase):
    """What the model list says a route can do: shown, not discovered later."""

    def test_native_ptq1_mlxlm_and_gguf_routes(self):
        from unittest import mock
        base = {"vision": False, "thinking": True, "tool_calls": False, "path": "/x", "supported": True}
        self.assertEqual(llm.route_features({}, dict(base, route="native", vision=True)),
                         ("engine", ["vision", "thinking", "tools", "ssd-cache", "speculation", "steering"]))
        self.assertEqual(llm.route_features({}, dict(base, route="ptq1")),
                         ("engine-ptq1", ["thinking", "tools", "ssd-cache"]))              # no speculation, no steering
        self.assertEqual(llm.route_features({}, dict(base, route="mlxlm")), ("mlx-lm", ["thinking"]))
        self.assertEqual(llm.route_features({}, dict(base, route="mlxlm", tool_calls=True)), ("mlx-lm", ["thinking", "tools"]))
        with mock.patch.object(llm, "gguf_native", return_value=True):
            self.assertEqual(llm.route_features({}, dict(base, route="gguf", thinking=False)),
                             ("engine-gguf", ["speculation", "steering"]))
            self.assertEqual(llm.route_features({}, dict(base, route="gguf", tool_calls=True)),
                             ("engine-gguf", ["thinking", "tools", "speculation", "steering"]))
        with mock.patch.object(llm, "gguf_native", return_value=False):
            self.assertEqual(llm.route_features({}, dict(base, route="gguf", vision=True)),
                             ("llama.cpp", ["vision", "thinking", "tools"]))

    def test_scan_models_carries_the_route_and_its_features(self):
        from unittest import mock
        m = {"path": "/m/a", "name": "a", "size": 1.0, "format": "x", "type": "t", "route": "mlxlm", "vision": False,
             "thinking": False, "supported": True, "problems": [], "efforts": [], "hadamard": False}
        with mock.patch.object(llm, "inspect_model", return_value=m), mock.patch.object(llm, "scan_gguf", return_value=[]), \
                mock.patch("os.walk", return_value=[("/m", [], ["config.json"])]):
            got = llm.scan_models({"dirs": ["/m"], "favorites": []})
        self.assertEqual((got[0]["engine"], got[0]["features"]), ("mlx-lm", []))


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


class TestUpdate(unittest.TestCase):
    """`./llm update` — version comparison, release parsing, the GitHub check (mocked)."""

    def capture(self, fn, *a):
        import contextlib
        import io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rv = fn(*a)
        return rv, buf.getvalue()

    def test_version_compare(self):
        self.assertEqual(llm._parse_version("v1.2.3-beta"), (1, 2, 3))
        self.assertTrue(llm._version_newer("1.2.0", "1.1.9"))
        self.assertTrue(llm._version_newer("1.10", "1.9"))         # not string order
        self.assertFalse(llm._version_newer("1.2", "1.2.0"))       # padded equal
        self.assertFalse(llm._version_newer("1.0", "2.0"))

    def test_latest_release_prefers_stable_dmg(self):
        rel = llm._latest_release({"tag_name": "v2.1", "body": "notes", "assets": [
            {"name": "LLMBar-2.1.dmg", "browser_download_url": "u1"},
            {"name": "LLMBar.dmg", "browser_download_url": "u2"}]})
        self.assertEqual(rel["version"], "2.1")
        self.assertEqual(rel["dmg"], "u2")                         # the stable name wins

    def _patch_release(self, obj):
        from unittest import mock

        class Resp:
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def read(self): return json.dumps(obj).encode()
        return mock.patch.object(llm.urllib.request, "urlopen", lambda *a, **k: Resp())

    def test_update_json_reports_newer(self):
        with self._patch_release({"tag_name": "v9.9", "assets": [
                {"name": "LLMBar.dmg", "browser_download_url": "http://x/LLMBar.dmg"}]}):
            rv, out = self.capture(llm.update_cmd, cfg(), ["--json"])
        card = json.loads(out)
        self.assertTrue(card["update"])
        self.assertEqual(card["latest"], "9.9")
        self.assertEqual(card["current"], llm.VERSION)

    def test_update_json_up_to_date(self):
        with self._patch_release({"tag_name": "v" + llm.VERSION, "assets": []}):
            rv, out = self.capture(llm.update_cmd, cfg(), ["--json"])
        self.assertFalse(json.loads(out)["update"])

    def test_update_check_announces_without_downloading(self):
        with self._patch_release({"tag_name": "v9.9", "assets": [
                {"name": "LLMBar.dmg", "browser_download_url": "http://x/LLMBar.dmg"}]}):
            rv, out = self.capture(llm.update_cmd, cfg(), ["--check"])
        self.assertTrue(rv)
        self.assertIn("9.9", out)


class TestBackgroundService(unittest.TestCase):
    """The optional launchd service: plist shape, the foreground `serve` guard, install/status/uninstall."""

    def capture(self, fn, *a):
        import contextlib
        import io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rv = fn(*a)
        return rv, buf.getvalue()

    def test_plist_shape(self):
        import plistlib
        d = plistlib.loads(llm.service_plist())
        self.assertEqual(d["Label"], "com.ai-llm-engine")
        self.assertEqual(d["ProgramArguments"][-1], "serve")
        self.assertTrue(d["RunAtLoad"])
        self.assertTrue(d["KeepAlive"])
        self.assertIn("LLM_HOME", d["EnvironmentVariables"])

    def test_serve_without_a_model_does_not_exec(self):
        from unittest import mock
        with mock.patch.object(llm, "current_model", lambda c: None):
            rv, out = self.capture(llm.serve_cmd, cfg())   # returns before os.execv
        self.assertFalse(rv)

    def test_install_writes_plist_and_bootstraps(self):
        from unittest import mock
        d = tempfile.mkdtemp()
        agent = os.path.join(d, "com.ai-llm-engine.plist")
        calls = []
        fake = mock.Mock(returncode=0, stdout="", stderr="")
        with mock.patch.object(llm, "LAUNCH_AGENT", agent), mock.patch.object(llm, "STATE", d), \
             mock.patch.object(llm.subprocess, "run", lambda *a, **k: calls.append(a[0]) or fake):
            rv, out = self.capture(llm.service_cmd, cfg(), ["install"])
        self.assertTrue(rv)
        self.assertTrue(os.path.exists(agent))
        self.assertTrue(any("bootstrap" in c for c in calls))

    def test_status_not_installed(self):
        from unittest import mock
        with mock.patch.object(llm, "LAUNCH_AGENT", os.path.join(tempfile.mkdtemp(), "none.plist")):
            rv, out = self.capture(llm.service_cmd, cfg(), ["status"])
        self.assertTrue(rv)
        self.assertIn("not installed", out.lower())

    def test_uninstall_removes_plist(self):
        from unittest import mock
        d = tempfile.mkdtemp()
        agent = os.path.join(d, "a.plist")
        with open(agent, "wb") as f:
            f.write(llm.service_plist())
        with mock.patch.object(llm, "LAUNCH_AGENT", agent), \
             mock.patch.object(llm.subprocess, "run", mock.Mock(return_value=mock.Mock(returncode=0))):
            rv, out = self.capture(llm.service_cmd, cfg(), ["uninstall"])
        self.assertTrue(rv)
        self.assertFalse(os.path.exists(agent))


class TestFetchCustomSource(unittest.TestCase):
    """`./llm fetch <url|magnet>` — a user-provided source: name derivation, archive safety, dispatch."""

    def test_basename_strips_query_and_slash(self):
        self.assertEqual(llm._fetch_basename("https://x.io/a/model.gguf?token=1"), "model.gguf")
        self.assertEqual(llm._fetch_basename("https://x.io/a/"), "a")

    def test_name_from_magnet_archive_and_gguf(self):
        self.assertEqual(llm._fetch_name("magnet:?xt=urn:btih:abc&dn=My+Model"), "My Model")
        self.assertEqual(llm._fetch_name("https://x.io/pack.tar.gz"), "pack")
        self.assertEqual(llm._fetch_name("https://x.io/Qwen3-4B.gguf"), "Qwen3-4B")

    def test_safe_extract_accepts_normal_zip(self):
        import zipfile
        d = tempfile.mkdtemp()
        arc = os.path.join(d, "a.zip")
        with zipfile.ZipFile(arc, "w") as z:
            z.writestr("config.json", "{}")
            z.writestr("sub/weights.bin", "x")
        out = tempfile.mkdtemp()
        llm._safe_extract(arc, out)
        self.assertTrue(os.path.exists(os.path.join(out, "config.json")))
        self.assertTrue(os.path.exists(os.path.join(out, "sub", "weights.bin")))

    def test_safe_extract_rejects_path_traversal(self):
        import zipfile
        d = tempfile.mkdtemp()
        arc = os.path.join(d, "evil.zip")
        with zipfile.ZipFile(arc, "w") as z:
            z.writestr("../escape.txt", "pwn")
        with self.assertRaises(ValueError):
            llm._safe_extract(arc, tempfile.mkdtemp())

    def test_fetch_dispatch(self):
        from unittest import mock
        c = cfg()
        c["dirs"] = [tempfile.mkdtemp()]
        seen = {}
        with mock.patch.object(llm, "_fetch_torrent", lambda u, d, p: seen.update(torrent=u) or True), \
             mock.patch.object(llm, "_fetch_http", lambda u, d, p: seen.update(http=u) or True):
            llm.fetch_cmd(c, ["magnet:?xt=urn:btih:abc&dn=M"])
            llm.fetch_cmd(c, ["https://x.io/thing.torrent"])
            llm.fetch_cmd(c, ["https://x.io/model.gguf"])
        self.assertTrue(seen["torrent"].startswith("magnet:") or seen["torrent"].endswith(".torrent"))
        self.assertEqual(seen["http"], "https://x.io/model.gguf")

    def test_fetch_without_args(self):
        import contextlib
        import io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rv = llm.fetch_cmd(cfg(), [])
        self.assertFalse(rv)


class TestOllamaStyleCommands(unittest.TestCase):
    """resolve_model and the Ollama-style verbs run / show / ps."""

    def setUp(self):
        from unittest import mock
        self.models = [
            {"name": "Bonsai-2-27B", "path": "/m/bonsai", "format": "JANG", "route": "native", "size": 7.7,
             "thinking": True, "vision": False, "tool_calls": True, "efforts": ["low", "high"],
             "supported": True, "problems": []},
            {"name": "Qwen3-0.6B", "path": "/m/qwen", "format": "MLX 4-bit", "route": "native", "size": 0.5,
             "thinking": False, "vision": False, "tool_calls": True, "efforts": [], "supported": True,
             "problems": []},
        ]
        self.cfg = cfg()
        self.cfg["model"] = "/m/bonsai"
        for patch in (mock.patch.object(llm, "scan_models", lambda c: list(self.models)),):
            patch.start()
            self.addCleanup(patch.stop)

    def capture(self, fn, *a):
        import contextlib
        import io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rv = fn(*a)
        return rv, buf.getvalue()

    # ── resolve_model ──
    def test_resolve_by_path_index_and_name(self):
        self.assertEqual(llm.resolve_model(self.models, "/m/qwen")[0]["name"], "Qwen3-0.6B")
        self.assertEqual(llm.resolve_model(self.models, "1")[0]["name"], "Bonsai-2-27B")
        self.assertEqual(llm.resolve_model(self.models, "qwen")[0]["name"], "Qwen3-0.6B")

    def test_resolve_ambiguous_and_missing(self):
        pick, hits = llm.resolve_model(self.models, "n")           # "Bonsai" and "Qwen3" both contain 'n'
        self.assertIsNone(pick)
        self.assertEqual(len(hits), 2)
        self.assertEqual(llm.resolve_model(self.models, "nope"), (None, []))

    # ── show ──
    def test_show_json_for_named_model(self):
        from unittest import mock
        with mock.patch.object(llm, "health", lambda c, t=3: None):
            rv, out = self.capture(llm.show_cmd, self.cfg, ["qwen", "--json"])
        self.assertTrue(rv)
        card = json.loads(out)
        self.assertEqual(card["name"], "Qwen3-0.6B")
        self.assertEqual(card["capabilities"], ["tools"])         # no thinking/vision for this one
        self.assertEqual(card["size_gb"], 0.5)

    def test_show_defaults_to_selected_and_reports_context_when_loaded(self):
        from unittest import mock
        with mock.patch.object(llm, "health", lambda c, t=3: {"status": "ok", "context": 40960}):
            rv, out = self.capture(llm.show_cmd, self.cfg, ["--json"])
        card = json.loads(out)
        self.assertEqual(card["name"], "Bonsai-2-27B")
        self.assertEqual(card["context"], 40960)
        self.assertIn("thinking", card["capabilities"])

    def test_show_unknown_model(self):
        rv, out = self.capture(llm.show_cmd, self.cfg, ["nope"])
        self.assertFalse(rv)

    # ── ps ──
    def test_ps_empty_when_engine_down(self):
        from unittest import mock
        with mock.patch.object(llm, "read_pid", lambda p: None):
            rv, out = self.capture(llm.ps_cmd, self.cfg, ["--json"])
        self.assertEqual(json.loads(out), {"models": []})

    def test_ps_lists_the_loaded_model(self):
        from unittest import mock
        with mock.patch.object(llm, "read_pid", lambda p: 123), \
             mock.patch.object(llm, "health", lambda c, t=3: {"status": "ok", "state": "awake",
                              "context": 40960, "memory_gb": {"model": 7.7}}), \
             mock.patch.object(llm, "current_model", lambda c: self.models[0]):
            rv, out = self.capture(llm.ps_cmd, self.cfg, ["--json"])
        row = json.loads(out)["models"][0]
        self.assertEqual((row["name"], row["state"], row["size_gb"], row["port"]),
                         ("Bonsai-2-27B", "awake", 7.7, 8090))

    # ── run (non-network paths) ──
    def test_run_without_args_is_usage(self):
        rv, out = self.capture(llm.run_cmd, self.cfg, [])
        self.assertFalse(rv)

    def test_run_unknown_local_name_without_slash_fails(self):
        rv, out = self.capture(llm.run_cmd, self.cfg, ["does-not-exist"])
        self.assertFalse(rv)

    # ── create / profiles ──
    def test_coerce_types(self):
        self.assertEqual(llm._coerce("true"), True)
        self.assertEqual(llm._coerce("0.2"), 0.2)
        self.assertEqual(llm._coerce("8"), 8)
        self.assertEqual(llm._coerce("hello"), "hello")

    def test_create_builds_a_profile(self):
        from unittest import mock
        with mock.patch.object(llm, "save_cfg", lambda c: None):
            rv, out = self.capture(llm.create_cmd, self.cfg,
                                   ["coder", "--from", "qwen", "--system", "be brief", "--param", "temperature=0.2",
                                    "--param", "top_k=20"])
        self.assertTrue(rv)
        p = self.cfg["profiles"]["coder"]
        self.assertEqual(p["from"], "qwen")
        self.assertEqual(p["system"], "be brief")
        self.assertEqual(p["params"], {"temperature": 0.2, "top_k": 20})

    def test_create_requires_a_base(self):
        rv, out = self.capture(llm.create_cmd, self.cfg, ["coder", "--system", "hi"])
        self.assertFalse(rv)

    def test_create_rm_removes_a_profile(self):
        from unittest import mock
        self.cfg["profiles"]["coder"] = {"from": "qwen", "system": "", "params": {}}
        with mock.patch.object(llm, "save_cfg", lambda c: None):
            rv, out = self.capture(llm.create_cmd, self.cfg, ["coder", "--rm"])
        self.assertTrue(rv)
        self.assertNotIn("coder", self.cfg["profiles"])

    def test_show_profile(self):
        self.cfg["profiles"]["coder"] = {"from": "qwen", "system": "be brief", "params": {"temperature": 0.2}}
        rv, out = self.capture(llm.show_cmd, self.cfg, ["coder", "--json"])
        card = json.loads(out)
        self.assertEqual(card["profile"], "coder")
        self.assertEqual(card["from"], "qwen")

    def test_run_profile_selects_base_and_passes_system_and_params(self):
        from unittest import mock
        self.cfg["profiles"]["coder"] = {"from": "qwen", "system": "be brief", "params": {"temperature": 0.1}}
        seen = {}
        with mock.patch.object(llm, "choose_model", lambda c, p: seen.update(selected=p) or True), \
             mock.patch.object(llm, "read_pid", lambda p: 1), \
             mock.patch.object(llm, "health", lambda c, t=3: {"status": "ok"}), \
             mock.patch.object(llm, "run_once",
                               lambda c, prompt, system="", params=None: seen.update(prompt=prompt, system=system, params=params) or True):
            rv, out = self.capture(llm.run_cmd, self.cfg, ["coder", "write", "a", "test"])
        self.assertTrue(rv)
        self.assertEqual(seen["selected"], "/m/qwen")        # resolved the profile's base model
        self.assertEqual(seen["prompt"], "write a test")
        self.assertEqual(seen["system"], "be brief")
        self.assertEqual(seen["params"], {"temperature": 0.1})


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


class TestTokenStaysWithTheHub(unittest.TestCase):
    """The hub answers /resolve/ with a redirect to a CDN: the Authorization header must not follow it."""

    def test_authorization_is_dropped_on_a_redirect_to_another_host_but_kept_on_the_same_one(self):
        import http.server
        import threading
        seen = {}

        class Target(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                seen["cdn"] = self.headers.get("Authorization")
                body = b"payload"
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        cdn = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Target)
        cdn_url = f"http://127.0.0.1:{cdn.server_address[1]}/blob"

        class Hub(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                seen["hub"] = self.headers.get("Authorization")
                if self.path.startswith("/same"):
                    seen["same_hit"] = True
                    body = b"local"
                    self.send_response(200)
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                else:
                    self.send_response(302)
                    self.send_header("Location", cdn_url)
                    self.send_header("Content-Length", "0")
                    self.end_headers()

        hub = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Hub)
        for srv in (cdn, hub):
            threading.Thread(target=srv.serve_forever, daemon=True).start()
        self.addCleanup(cdn.shutdown)
        self.addCleanup(hub.shutdown)
        d = tempfile.mkdtemp()
        out = os.path.join(d, "f")
        llm.download_file(f"http://127.0.0.1:{hub.server_address[1]}/resolve/f", out, 0, None,
                          {"Authorization": "Bearer secret", "User-Agent": "t"})
        self.assertEqual(seen["hub"], "Bearer secret")             # the hub itself gets it
        self.assertIsNone(seen["cdn"])                             # the CDN does not
        self.assertEqual(open(out, "rb").read(), b"payload")


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
        # decided by the architecture's fields, not by the name: a type nobody listed still runs natively…
        hybrid = {"hidden_size": 2048, "num_attention_heads": 16, "linear_num_value_heads": 16, "linear_num_key_heads": 8,
                  "linear_key_head_dim": 128, "linear_value_head_dim": 128, "linear_conv_kernel_dim": 4}
        self.assertEqual(r({"model_type": "qwen3_9", "text_config": dict(hybrid, model_type="qwen3_9_text")}), "native")
        self.assertEqual(r({"model_type": "some_new_name", "text_config": hybrid}), "native")
        # …unless it is a mixture of experts, and a dense model without those fields is not the hybrid
        self.assertEqual(r({"model_type": "some_new_name", "text_config": dict(hybrid, num_experts=8)}), "mlxlm")
        self.assertEqual(r({"model_type": "llama", "text_config": {"hidden_size": 2048, "num_attention_heads": 16}}), "mlxlm")
        self.assertEqual(r({"model_type": "qwen3_5", "text_config": {"num_experts": 8}}), "mlxlm")  # MoE → not native

    def test_hub_tags(self):
        h = llm.hub_route
        self.assertEqual(h(["mlx", "safetensors", "prism_hadamard_qwen35", "ternary"]), "native")
        self.assertEqual(h(["custom", "safetensors", "prism_hadamard_qwen35_ptq1", "mlx"]), "ptq1")
        self.assertEqual(h(["mlx", "safetensors", "ternel_hadamard_qwen35", "qwen3_5", "ternary", "vision"]), "ptq1")
        self.assertEqual(h(["mlx", "safetensors", "qwen3_6"]), "native")      # a later family member, by name
        self.assertEqual(h(["mlx", "safetensors", "qwen3_8b", "qwen3"]), "mlxlm")
        self.assertEqual(h(["mlx", "safetensors", "qwen3"]), "mlxlm")
        self.assertEqual(h(["mlx", "safetensors", "qwen3_5_moe"]), "mlxlm")
        self.assertEqual(h(["gguf", "qwen3_5"]), "gguf")                  # llama.cpp runs it
        self.assertEqual(h(["mlx", "gguf", "qwen3_5"]), "native")         # MLX weights win when the repo has them
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
    """`./llm ui` opens the editor module of the one LLMBar app through its llmbar:// link."""

    def test_the_app_is_found_next_to_the_code_in_a_bundle_or_in_applications(self):
        from unittest import mock
        base = os.path.realpath(tempfile.mkdtemp())
        with mock.patch.object(llm, "BASE", base):
            self.assertNotIn(llm.find_app(), (os.path.join(base, "LLMBar.app"),))
            os.makedirs(os.path.join(base, "LLMBar.app"))
            self.assertEqual(llm.find_app(), os.path.join(base, "LLMBar.app"))
        bundle = os.path.join(os.path.realpath(tempfile.mkdtemp()), "LLMBar.app")
        inner = os.path.join(bundle, "Contents", "Resources", "llm")
        os.makedirs(inner)
        with mock.patch.object(llm, "BASE", inner):
            self.assertEqual(llm.find_app(), bundle)                  # the CLI inside the bundle names its own app

    def test_ui_command_sends_the_editor_link_with_the_project(self):
        from unittest import mock
        calls = []
        base = os.path.realpath(tempfile.mkdtemp())
        os.makedirs(os.path.join(base, "LLMBar.app"))
        proj = os.path.realpath(tempfile.mkdtemp())
        fake = mock.Mock(returncode=0)
        with mock.patch.object(llm, "BASE", base), mock.patch.object(llm.subprocess, "run",
                                                                      lambda cmd, **k: calls.append(cmd) or fake):
            self.assertTrue(llm.run_ui([proj]))
            self.assertTrue(llm.run_ui(["--choose"]))
        first, second = calls
        self.assertEqual(first[:3], ["open", "-a", os.path.join(base, "LLMBar.app")])
        self.assertEqual(first[3], "llmbar://editor?root=" + urllib.parse.quote(proj, safe=""))
        self.assertEqual(second[3], "llmbar://editor")                   # no root: the last project, or the picker

    def test_a_missing_app_is_explained_not_launched(self):
        from unittest import mock
        with mock.patch.object(llm, "find_app", lambda: None):
            self.assertFalse(llm.run_ui([]))


class TestOneAppBundle(unittest.TestCase):
    @unittest.skipUnless(os.path.isdir(os.path.join(ROOT, "LLMBar.app")), "LLMBar.app has not been built here")
    def test_the_built_app_is_one_bundle_with_the_backend_and_the_widget(self):
        app = os.path.join(ROOT, "LLMBar.app", "Contents")
        self.assertTrue(os.path.isfile(os.path.join(app, "MacOS", "LLMBar")))
        res = os.path.join(app, "Resources")
        for rel in ("ui/server.py", "ui/codeinfo.py", "ui/agentloop.py", "ui/diagnostics.py", "harness/memory.py"):
            self.assertTrue(os.path.isfile(os.path.join(res, "llm", rel)), rel)
        self.assertFalse(os.path.exists(os.path.join(res, "LLMEditor.app")))      # no second app inside
        self.assertFalse(os.path.exists(os.path.join(res, "LLMVoice.app")))
        import plistlib
        with open(os.path.join(app, "Info.plist"), "rb") as f:
            plist = plistlib.load(f)
        schemes = [s for t in plist.get("CFBundleURLTypes", []) for s in t.get("CFBundleURLSchemes", [])]
        self.assertIn("llmbar", schemes)
        self.assertIn("NSMicrophoneUsageDescription", plist)


class TestPullValidation(unittest.TestCase):
    """A download target is a repository id — never a path that could climb out of the models folder."""

    def test_bad_ids_never_reach_the_network(self):
        from unittest import mock
        for bad in ("../../x", "/etc/passwd", "org/../x", "a", "org/", "ms:../x/y", "org/name/extra", ".hidden/x"):
            with self.subTest(bad), mock.patch.object(llm, "_hf") as hf, mock.patch.object(llm, "ms_files") as ms:
                self.assertFalse(llm.hf_pull(bad))
                hf.assert_not_called()
                ms.assert_not_called()


class TestFavouriteModels(unittest.TestCase):
    """`./llm fav`: a star on a model, kept in the config, listed first, never undone by a double click."""

    def setUp(self):
        from unittest import mock
        self.home = os.path.realpath(tempfile.mkdtemp())
        self.paths = {}
        for n in ("alpha", "beta", "gamma"):
            d = os.path.join(self.home, n)
            os.makedirs(d)
            with open(os.path.join(d, "config.json"), "w") as f:
                f.write("{}")
            self.paths[n] = d
        self.cfg = cfg()
        self.cfg.update(dirs=[self.home], model="", models={}, favorites=[])
        self.saved = []

        def fake_inspect(path):
            name = os.path.basename(path)
            return {"path": path, "name": name, "size": 1.0, "format": "x", "type": "t", "vision": False,
                    "thinking": False, "hadamard": False, "supported": name != "gamma", "problems": [], "efforts": []}
        for name, val in (("inspect_model", fake_inspect), ("scan_gguf", lambda c: []),
                          ("save_cfg", lambda c: self.saved.append(list(c["favorites"])))):
            p = mock.patch.object(llm, name, val)
            p.start()
            self.addCleanup(p.stop)

    def names(self):
        return [m["name"] for m in llm.scan_models(self.cfg)]

    def test_next_favourite_wraps_and_starts_with_the_first(self):
        self.assertEqual(llm.next_favorite(["a", "b", "c"], "a"), "b")
        self.assertEqual(llm.next_favorite(["a", "b", "c"], "c"), "a")
        self.assertEqual(llm.next_favorite(["a", "b"], "zzz"), "a")           # the current one is not a favourite
        self.assertIsNone(llm.next_favorite(["a"], "a"))                      # nothing to switch to
        self.assertEqual(llm.next_favorite(["a"], "b"), "a")
        self.assertIsNone(llm.next_favorite([], "a"))

    def test_fav_next_selects_the_next_usable_favourite(self):
        from unittest import mock
        self.cfg["favorites"] = [self.paths["alpha"], self.paths["gamma"], self.paths["beta"]]   # gamma is unsupported
        self.cfg["model"] = self.paths["alpha"]
        with mock.patch.object(llm, "choose_model", return_value=True) as ch:
            self.assertTrue(llm.fav_cmd(self.cfg, ["--next"]))
        ch.assert_called_once_with(self.cfg, self.paths["beta"])               # gamma is skipped
        self.cfg["favorites"] = [self.paths["alpha"]]
        with mock.patch.object(llm, "choose_model", return_value=True) as ch:
            self.assertFalse(llm.fav_cmd(self.cfg, ["--next"]))
        ch.assert_not_called()

    def test_the_default_is_no_favourites_and_the_flag_is_false(self):
        self.assertTrue(all(m["favorite"] is False for m in llm.scan_models(self.cfg)))
        self.assertEqual(llm.DEFAULTS["favorites"], [])

    def test_a_favourite_comes_first_even_before_a_supported_model(self):
        self.assertEqual(self.names(), ["alpha", "beta", "gamma"])               # unsupported last
        self.cfg["favorites"] = [self.paths["gamma"]]
        self.assertEqual(self.names(), ["gamma", "alpha", "beta"])
        m = llm.scan_models(self.cfg)[0]
        self.assertTrue(m["favorite"])

    def test_toggle_by_name_number_and_path_and_explicit_on_off(self):
        self.assertTrue(llm.fav_cmd(self.cfg, ["beta"]))
        self.assertEqual(self.cfg["favorites"], [self.paths["beta"]])
        self.assertTrue(llm.fav_cmd(self.cfg, ["beta"]))                       # a second toggle removes it
        self.assertEqual(self.cfg["favorites"], [])
        llm.fav_cmd(self.cfg, [self.paths["alpha"], "--on"])
        llm.fav_cmd(self.cfg, [self.paths["alpha"], "--on"])                   # explicit: twice is still on, once
        self.assertEqual(self.cfg["favorites"], [self.paths["alpha"]])
        llm.fav_cmd(self.cfg, [self.paths["alpha"], "--off"])
        llm.fav_cmd(self.cfg, [self.paths["alpha"], "--off"])
        self.assertEqual(self.cfg["favorites"], [])
        self.assertTrue(llm.fav_cmd(self.cfg, ["1"]))                          # number in the list order
        self.assertEqual(self.cfg["favorites"], [self.paths["alpha"]])
        self.assertTrue(self.saved)                                            # every change was written

    def test_unknown_and_ambiguous_names_change_nothing(self):
        import io
        from contextlib import redirect_stdout
        buf = io.StringIO()
        with redirect_stdout(buf):
            self.assertFalse(llm.fav_cmd(self.cfg, ["nope"]))
            self.assertFalse(llm.fav_cmd(self.cfg, ["a"]))                     # alpha, beta and gamma all contain "a"
        self.assertEqual(self.cfg["favorites"], [])

    def test_listing_json_and_a_removed_model_is_not_listed(self):
        import io
        import json
        from contextlib import redirect_stdout
        self.cfg["favorites"] = [self.paths["beta"], "/gone/model"]
        buf = io.StringIO()
        with redirect_stdout(buf):
            llm.fav_cmd(self.cfg, ["--json"])
        self.assertEqual(json.loads(buf.getvalue()), [self.paths["beta"]])     # only what exists on disk
