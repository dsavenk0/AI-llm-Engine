"""The launch-time readiness check: what is reported on a clean Mac, offline, with little disk, and when all is well.

    .venv/bin/python -m unittest tests.test_preflight -v
"""
import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
import urllib.error
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import llm  # noqa: E402
from engine import preflight  # noqa: E402


def run(env_ok=True, python="/usr/bin/python3", online=True, free=50.0, models=3, machine="arm64", mac="15.1"):
    probes = []

    def probe(url, **kw):
        probes.append(url)
        return (True, "") if online else (False, "[Errno 8] nodename nor servname provided")

    items = preflight.run("/data", "/data/.venv/bin/python", lambda: python, models, probe=probe,
                          env_ok=lambda p: env_ok, machine=machine, mac_ver=mac, free_gb=free)
    return {i["id"]: i for i in items}, probes


class TestRun(unittest.TestCase):
    def test_everything_in_place_needs_no_network(self):
        with mock.patch("os.path.exists", return_value=True):
            items, probes = run()
        self.assertEqual({k: v["level"] for k, v in items.items()}, {"environment": "ok"})
        self.assertEqual(probes, [])                     # a working setup works offline

    def test_clean_mac_with_network_is_a_warning_naming_both_downloads(self):
        with mock.patch("os.path.exists", return_value=False):
            items, probes = run(env_ok=False, python=None)
        self.assertEqual(items["environment"]["level"], "warn")
        self.assertIn("Python", items["environment"]["detail"])
        self.assertIn("libraries", items["environment"]["detail"])
        self.assertNotIn("network", items)
        self.assertEqual(len(probes), 2)

    def test_offline_is_an_error_only_when_something_must_be_downloaded(self):
        with mock.patch("os.path.exists", return_value=False):
            items, _ = run(env_ok=False, python="/usr/bin/python3", online=False)
        self.assertEqual(items["network"]["level"], "error")
        self.assertIn("No internet connection", items["network"]["title"])
        self.assertIn("nodename", items["network"]["detail"])
        with mock.patch("os.path.exists", return_value=True):
            items, probes = run(env_ok=True, online=False)
        self.assertNotIn("network", items)
        self.assertEqual(probes, [])

    def test_a_damaged_environment_is_reported_as_such(self):
        with mock.patch("os.path.exists", return_value=True):
            items, _ = run(env_ok=False)
        self.assertIn("damaged", items["environment"]["title"])

    def test_disk_space(self):
        with mock.patch("os.path.exists", return_value=False):
            items, _ = run(env_ok=False, free=0.6)
        self.assertEqual(items["disk"]["level"], "error")
        with mock.patch("os.path.exists", return_value=True):
            items, _ = run(free=3.0)
            self.assertEqual(items["disk"]["level"], "warn")
            items, _ = run(free=3.0, env_ok=True)
            self.assertNotIn("disk", run(free=80)[0])

    def test_platform_and_models(self):
        with mock.patch("os.path.exists", return_value=True):
            items, _ = run(machine="x86_64", mac="12.6", models=0)
        self.assertEqual(items["apple-silicon"]["level"], "error")
        self.assertEqual(items["macos"]["level"], "error")
        self.assertEqual(items["models"]["level"], "warn")
        with mock.patch("os.path.exists", return_value=True):
            self.assertNotIn("models", run(models=None)[0])
            self.assertNotIn("macos", run(mac="")[0])
            self.assertNotIn("macos", run(mac="not-a-version")[0])

    def test_summary_is_ok_without_errors_only(self):
        self.assertTrue(preflight.summary([{"level": "warn"}, {"level": "ok"}])["ok"])
        self.assertFalse(preflight.summary([{"level": "error"}])["ok"])


class TestProbe(unittest.TestCase):
    def test_http_errors_still_mean_the_network_works(self):
        err = urllib.error.HTTPError("u", 403, "forbidden", {}, None)
        with mock.patch("urllib.request.urlopen", side_effect=err):
            self.assertEqual(preflight.reachable("https://x"), (True, ""))

    def test_dns_and_timeout_failures_say_why(self):
        with mock.patch("urllib.request.urlopen", side_effect=urllib.error.URLError("no route")):
            ok, why = preflight.reachable("https://x")
        self.assertEqual((ok, why), (False, "no route"))
        with mock.patch("urllib.request.urlopen", side_effect=TimeoutError("timed out")):
            self.assertEqual(preflight.reachable("https://x"), (False, "timed out"))

    def test_environment_check_runs_the_imports(self):
        with tempfile.TemporaryDirectory() as d:
            py = os.path.join(d, "python")
            self.assertFalse(preflight.environment_ok(py))
            with open(py, "w") as f:
                f.write("#!/bin/sh\nexit 0\n")
            os.chmod(py, 0o755)
            self.assertTrue(preflight.environment_ok(py))
            with open(py, "w") as f:
                f.write("#!/bin/sh\nexit 1\n")
            self.assertFalse(preflight.environment_ok(py))


class TestCommand(unittest.TestCase):
    def test_json_and_exit_status(self):
        res = {"ok": False, "items": [{"id": "network", "level": "error", "title": "No internet connection", "detail": "x"}]}
        with mock.patch.object(preflight, "run", return_value=res["items"]), \
                mock.patch.object(llm, "scan_models", return_value=[]), contextlib.redirect_stdout(io.StringIO()) as out:
            ok = llm.check_cmd({}, as_json=True)
        self.assertFalse(ok)
        self.assertEqual(json.loads(out.getvalue()), res)
        with mock.patch.object(preflight, "run", return_value=[{"id": "environment", "level": "ok", "title": "Ready", "detail": ""}]), \
                mock.patch.object(llm, "scan_models", return_value=[1]), contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertTrue(llm.check_cmd({}))
        self.assertIn("Ready", out.getvalue())

    def test_setup_without_network_explains_and_changes_nothing(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.object(llm, "DATA", d), \
                mock.patch.object(llm, "VENV_PY", os.path.join(d, ".venv", "bin", "python")), \
                mock.patch.object(llm, "find_python", return_value="/usr/bin/python3"), \
                mock.patch("subprocess.run") as run, \
                mock.patch.object(preflight, "environment_ok", return_value=False), \
                mock.patch.object(preflight, "reachable", return_value=(False, "offline")), \
                contextlib.redirect_stdout(io.StringIO()) as out:
            # the venv "exists" so no venv creation is attempted; only the pip step is guarded
            os.makedirs(os.path.dirname(llm.VENV_PY))
            open(llm.VENV_PY, "w").close()
            self.assertFalse(llm.setup())
        run.assert_not_called()
        self.assertIn("No internet connection", out.getvalue())
        self.assertIn("offline", out.getvalue())


if __name__ == "__main__":
    unittest.main()


class TestOfflineBuild(unittest.TestCase):
    def test_bundled_wheels_need_no_network(self):
        probes = []
        with mock.patch("os.path.exists", return_value=False):
            items = preflight.run("/data", "/data/.venv/bin/python", lambda: "/app/python/bin/python3", 1,
                                  probe=lambda u, **k: probes.append(u) or (False, "offline"),
                                  env_ok=lambda p: False, machine="arm64", mac_ver="15.0", free_gb=40,
                                  wheels_bundled=True)
        by = {i["id"]: i for i in items}
        self.assertEqual(probes, [])
        self.assertNotIn("network", by)
        self.assertIn("no network", by["environment"]["detail"])

    def test_bundled_wheels_with_no_python_still_need_python(self):
        with mock.patch("os.path.exists", return_value=False):
            items = preflight.run("/data", "/data/.venv/bin/python", lambda: None, 1,
                                  probe=lambda u, **k: (True, ""), env_ok=lambda p: False,
                                  machine="arm64", mac_ver="15.0", free_gb=40, wheels_bundled=True)
        self.assertIn("Python", {i["id"]: i for i in items}["environment"]["detail"])


class TestOfflineSetup(unittest.TestCase):
    def test_pip_installs_from_the_bundled_wheels_without_the_network(self):
        with tempfile.TemporaryDirectory() as d:
            wheels = os.path.join(d, "wheels")
            os.makedirs(wheels)
            open(os.path.join(wheels, "x-1-py3-none-any.whl"), "w").close()
            py = os.path.join(d, ".venv", "bin", "python")
            os.makedirs(os.path.dirname(py))
            open(py, "w").close()
            calls = []
            with mock.patch.object(llm, "DATA", d), mock.patch.object(llm, "VENV_PY", py), \
                    mock.patch.object(llm, "ENV_READY", os.path.join(d, ".venv", ".ready")), \
                    mock.patch.object(llm, "BUNDLED_WHEELS", wheels), \
                    mock.patch.object(preflight, "environment_ok", return_value=False), \
                    mock.patch.object(preflight, "reachable", side_effect=AssertionError("no network wanted")), \
                    mock.patch("subprocess.run", side_effect=lambda cmd, **k: calls.append(cmd) or mock.Mock(returncode=0)), \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertTrue(llm.setup())
            pip = [c for c in calls if "pip" in c][0]
            self.assertIn("--no-index", pip)
            self.assertEqual(pip[pip.index("--find-links") + 1], wheels)
            self.assertTrue(os.path.exists(os.path.join(d, ".venv", ".ready")))

    def test_a_checkout_has_no_bundled_wheels(self):
        with mock.patch.object(llm, "BUNDLED_WHEELS", None):
            self.assertIsNone(llm.bundled_wheels())


class TestEnvReady(unittest.TestCase):
    def test_a_venv_without_libraries_is_not_ready_and_the_marker_is_remembered(self):
        with tempfile.TemporaryDirectory() as d:
            py = os.path.join(d, ".venv", "bin", "python")
            os.makedirs(os.path.dirname(py))
            open(py, "w").close()
            marker = os.path.join(d, ".venv", ".ready")
            with mock.patch.object(llm, "VENV_PY", py), mock.patch.object(llm, "ENV_READY", marker):
                with mock.patch.object(preflight, "environment_ok", return_value=False):
                    self.assertFalse(llm.env_ready())
                self.assertFalse(os.path.exists(marker))
                with mock.patch.object(preflight, "environment_ok", return_value=True):
                    self.assertTrue(llm.env_ready())
                self.assertTrue(os.path.exists(marker))
                with mock.patch.object(preflight, "environment_ok", side_effect=AssertionError("cached")):
                    self.assertTrue(llm.env_ready())
                # other requirements (a newer app) or another interpreter: verified again, once
                with mock.patch.object(llm, "_ready_stamp", return_value="changed\n"), \
                        mock.patch.object(preflight, "environment_ok", return_value=False) as ok:
                    self.assertFalse(llm.env_ready())
                ok.assert_called_once()
