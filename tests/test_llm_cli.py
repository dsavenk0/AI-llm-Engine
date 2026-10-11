"""llm.py flows without a model: discovery, start/stop and port recovery against fake engine processes,
hub search and download against a local fake hub, selection, deletion and per-model settings.

    .venv/bin/python -m unittest tests.test_llm_cli -v
"""
import contextlib
import hashlib
import io
import json
import os
import socket
import struct
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import llm  # noqa: E402


def quiet(fn, *a, **k):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        r = fn(*a, **k)
    return r, buf.getvalue()


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def write_model(path, config=None, files=("model.safetensors",), template=None, jang=None):
    os.makedirs(path, exist_ok=True)
    cfg = {"model_type": "qwen3_5", "text_config": {"model_type": "qwen3_5_text"},
           "quantization": {"bits": 4, "group_size": 64}}
    cfg.update(config or {})
    with open(os.path.join(path, "config.json"), "w") as f:
        json.dump(cfg, f)
    for name in files:
        with open(os.path.join(path, name), "wb") as f:
            header = json.dumps({}).encode()
            f.write(struct.pack("<Q", len(header)) + header)
    if template is not None:
        with open(os.path.join(path, "chat_template.jinja"), "w") as f:
            f.write(template)
    if jang is not None:
        with open(os.path.join(path, "jang_config.json"), "w") as f:
            json.dump(jang, f)
    return path


class State(unittest.TestCase):
    """Isolates llm.py's state files and config in a temp dir."""

    def setUp(self):
        self.tmp = os.path.realpath(tempfile.mkdtemp())
        self.state = os.path.join(self.tmp, ".llm")
        os.makedirs(self.state)
        for name, val in (("STATE", self.state), ("CFG_FILE", os.path.join(self.state, "config.json")),
                          ("ENGINE_PID", os.path.join(self.state, "engine.pid")),
                          ("ENGINE_LOG", os.path.join(self.state, "engine.log"))):
            p = mock.patch.object(llm, name, val)
            p.start()
            self.addCleanup(p.stop)
        self.cfg = llm.load_cfg()
        self.cfg.update(dirs=[os.path.join(self.tmp, "models")], model="", models={}, auto_free=False,
                        port=free_port(), advanced={})
        os.makedirs(self.cfg["dirs"][0])


class TestInspect(State):
    def test_native_quantized_model(self):
        d = write_model(os.path.join(self.cfg["dirs"][0], "Qwen3.5-X-4bit"),
                        template="{% if enable_thinking is defined and enable_thinking is true %}<think>{% endif %}")
        m = llm.inspect_model(d)
        self.assertEqual((m["route"], m["format"], m["supported"], m["problems"]), ("native", "MLX 4-bit", True, []))
        self.assertTrue(m["thinking"])
        self.assertFalse(m["thinking_default"])        # only opens a think block on an explicit True
        self.assertFalse(m["vision"])

    def test_default_thinking_template_and_efforts_and_vision(self):
        tpl = ("{% if enable_thinking is undefined or enable_thinking is true %}<think>{% endif %}"
               "{% if reasoning_effort == 'low' %}a{% elif reasoning_effort == 'high' %}b{% endif %}")
        d = write_model(os.path.join(self.cfg["dirs"][0], "m"), config={"vision_config": {"depth": 1}}, template=tpl)
        m = llm.inspect_model(d)
        self.assertTrue(m["thinking_default"])
        self.assertEqual(m["efforts"], ["high", "low"])
        self.assertTrue(m["vision"])

    def test_routes_ptq1_and_other_families(self):
        d = os.path.join(self.cfg["dirs"][0], "p")
        self.assertEqual(llm.inspect_model(write_model(d, config={"schema": "prism_hadamard_qwen35_ptq1_v1",
                                                                  "quantization": {}}))["route"], "ptq1")
        d3 = os.path.join(self.cfg["dirs"][0], "t")
        m = llm.inspect_model(write_model(d3, config={"model_type": "ternel_hadamard_qwen35",
                                                      "text_config": {"model_type": "qwen3_5_text"}}))
        self.assertEqual((m["route"], m["format"], m["supported"]), ("ptq1", "TQ1", True))
        d2 = os.path.join(self.cfg["dirs"][0], "g")
        m = llm.inspect_model(write_model(d2, config={"model_type": "glm4", "text_config": {}}))
        self.assertEqual((m["route"], m["supported"]), ("mlxlm", True))

    def test_packed_ternary_jang_runs_as_ptq1_and_1bit_affine_is_reported(self):
        d = write_model(os.path.join(self.cfg["dirs"][0], "j"),
                        jang={"profile": "J", "runtime": {"requires_jang_ternary_packed_expansion": True}})
        m = llm.inspect_model(d)
        self.assertEqual((m["route"], m["supported"], m["problems"]), ("ptq1", True, []))
        self.assertTrue(m["format"].startswith("JANG"))          # the profile stays the label
        d = write_model(os.path.join(self.cfg["dirs"][0], "j1"),
                        jang={"profile": "J", "runtime": {"requires_jang_affine1_expansion": True}})
        m = llm.inspect_model(d)
        self.assertFalse(m["supported"])
        self.assertTrue(m["problems"])

    def test_non_models_are_ignored(self):
        self.assertIsNone(llm.inspect_model(write_model(os.path.join(self.tmp, "nowt"), files=())))   # no weights
        os.makedirs(os.path.join(self.tmp, "empty"))
        self.assertIsNone(llm.inspect_model(os.path.join(self.tmp, "empty")))

    def test_scan_names_dedups_and_orders(self):
        root = self.cfg["dirs"][0]
        write_model(os.path.join(root, "zeta"))
        write_model(os.path.join(root, "alpha"))
        snap = os.path.join(root, "models--org--cached", "snapshots", "abc123")
        write_model(snap)
        os.symlink(os.path.join(root, "alpha"), os.path.join(root, "alias"))
        write_model(os.path.join(root, "bad"), config={"model_type": "x"}, jang={"runtime": {"requires_jang_affine1_expansion": True}})
        names = [m["name"] for m in llm.scan_models(self.cfg)]
        self.assertEqual(names[:3], ["alpha", "cached", "zeta"])    # supported first, then by name
        self.assertEqual(names[-1], "bad")                           # unsupported sinks
        self.assertEqual(len([n for n in names if n in ("alpha", "alias")]), 1)   # the symlink is not a second model


class FakeTime:
    """A clock that only moves when the code under test sleeps."""
    def __init__(self):
        self.now = 1000.0

    def time(self):
        return self.now

    def sleep(self, s):
        self.now += s

    def ctime(self, *a):
        return "now"


ENGINE_SCRIPT = textwrap.dedent('''
    import http.server, json, sys, time
    port, delay = int(sys.argv[1]), float(sys.argv[2])
    time.sleep(delay)
    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a): pass
        def do_GET(self):
            body = json.dumps({"status": "ok", "model": "fake", "context": 4096}).encode()
            self.send_response(200); self.send_header("Content-Length", str(len(body))); self.end_headers()
            self.wfile.write(body)
    http.server.ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
''')


class ReapingPopen(subprocess.Popen):
    """In production the engine's parent is long gone, so init reaps it; here the test process is the
    parent and an unreaped child stays a zombie that kill_pid() would wait 15 s for."""
    spawned = []

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        ReapingPopen.spawned.append(self.pid)
        threading.Thread(target=self.wait, daemon=True).start()


class TestStartStop(State):
    def setUp(self):
        super().setUp()
        p = mock.patch.object(llm.subprocess, "Popen", ReapingPopen)
        p.start()
        self.addCleanup(p.stop)
        self.model = write_model(os.path.join(self.cfg["dirs"][0], "m"))
        self.cfg["model"] = self.model
        script = os.path.join(self.tmp, "engine.py")
        with open(script, "w") as f:
            f.write(ENGINE_SCRIPT)
        self.script = script
        self.spawned = []
        for target, val in (("VENV_PY", sys.executable),):
            p = mock.patch.object(llm, target, val)
            p.start()
            self.addCleanup(p.stop)
        self.addCleanup(self.cleanup)

    def cleanup(self):
        for pid in [llm.read_pid(llm.ENGINE_PID)] + self.spawned:
            if pid:
                with contextlib.suppress(OSError):
                    os.killpg(pid, 9)

    def engine(self, delay=0.0):
        return lambda cfg, m: [sys.executable, "-B", self.script, str(cfg["port"]), str(delay)]

    def test_a_stale_pid_file_never_makes_us_signal_someone_else(self):
        """After a crash or a reboot the pid in the file can belong to any program. stop() and a start that
        finds the port busy must leave it alone and drop the stale file."""
        victim = subprocess.Popen(["sleep", "30"], start_new_session=True)
        self.addCleanup(victim.kill)
        os.makedirs(os.path.dirname(llm.ENGINE_PID), exist_ok=True)
        with open(llm.ENGINE_PID, "w") as f:
            f.write(f"{victim.pid}\nengine.server")
        self.assertIsNone(llm.read_pid(llm.ENGINE_PID))
        self.assertFalse(os.path.exists(llm.ENGINE_PID))
        with open(llm.ENGINE_PID, "w") as f:
            f.write(str(victim.pid))                       # an old file without the second line
        quiet(llm.stop, self.cfg)
        time.sleep(0.2)
        self.assertIsNone(victim.poll(), "an unrelated process was signalled")

    def test_start_waits_for_health_then_stop_clears_the_pid(self):
        with mock.patch.object(llm, "engine_cmd", self.engine(0.3)), mock.patch.object(llm, "show_endpoints", lambda *a: None):
            ok, out = quiet(llm.start, self.cfg, False)
        self.assertTrue(ok, out)
        self.assertIn("ready in", out)
        pid = llm.read_pid(llm.ENGINE_PID)
        self.assertTrue(pid)
        self.assertEqual(llm.health(self.cfg)["model"], "fake")
        again, out2 = quiet(llm.start, self.cfg, False)                  # already running: no second process
        self.assertTrue(again)
        self.assertIn("already running", out2)
        self.assertEqual(llm.read_pid(llm.ENGINE_PID), pid)
        _, out3 = quiet(llm.stop, self.cfg)
        self.assertIn(str(pid), out3)
        self.assertFalse(os.path.exists(llm.ENGINE_PID))
        self.assertFalse(llm.port_busy(self.cfg["port"]))
        _, out4 = quiet(llm.stop, self.cfg)
        self.assertIn("Nothing is running", out4)

    def test_a_stuck_engine_of_ours_on_the_port_is_replaced(self):
        stuck = subprocess.Popen([sys.executable, "-c",
                                  f"import socket,time;s=socket.socket();s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1);"
                                  f"s.bind(('127.0.0.1',{self.cfg['port']}));s.listen();time.sleep(60)"],
                                 start_new_session=True)
        self.spawned.append(stuck.pid)
        deadline = time.time() + 5
        while not llm.port_busy(self.cfg["port"]) and time.time() < deadline:
            time.sleep(0.05)
        with open(llm.ENGINE_PID, "w") as f:
            f.write(f"{stuck.pid}\nsocket.socket()")      # what start() records: a fragment of the command line
        with mock.patch.object(llm, "engine_cmd", self.engine()), mock.patch.object(llm, "show_endpoints", lambda *a: None):
            ok, out = quiet(llm.start, self.cfg, False)
        self.assertTrue(ok, out)
        self.assertIn("stuck engine", out)
        stuck.wait(5)
        self.assertIsNotNone(stuck.returncode)

    def test_a_foreign_process_on_the_port_is_left_alone(self):
        s = socket.socket()
        s.bind(("127.0.0.1", self.cfg["port"]))
        s.listen()
        self.addCleanup(s.close)
        ok, out = quiet(llm.start, self.cfg, False)
        self.assertFalse(ok)
        self.assertIn("busy", out)

    def listener(self):
        """Somebody else's process (not started by llm.py) holding the engine port."""
        p = subprocess.Popen([sys.executable, "-c",
                              "import socket,time;s=socket.socket();s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1);"
                              f"s.bind(('127.0.0.1',{self.cfg['port']}));s.listen();time.sleep(60)"],
                             start_new_session=True)
        self.addCleanup(p.kill)
        deadline = time.time() + 5
        while not llm.port_busy(self.cfg["port"]) and time.time() < deadline:
            time.sleep(0.05)
        return p

    def test_the_message_names_who_holds_the_port_and_nothing_is_stopped_unattended(self):
        p = self.listener()
        ok, out = quiet(llm.start, self.cfg, False)          # non-interactive: LLMBar, scripts
        self.assertFalse(ok)
        self.assertIn(f"PID {p.pid}", out)
        time.sleep(0.2)
        self.assertIsNone(p.poll(), "a process that is not ours was stopped without asking")

    def test_a_person_is_asked_and_the_default_is_no(self):
        p = self.listener()
        asked = []

        def answer(q, default=True):
            asked.append((q, default))
            return False
        with mock.patch.object(llm, "ask_yn", answer), mock.patch.object(llm.sys.stdin, "isatty", lambda: True):
            ok, out = quiet(llm.start, self.cfg, True)
        self.assertFalse(ok)
        self.assertEqual([d for _, d in asked if str(p.pid) in _], [False])
        time.sleep(0.2)
        self.assertIsNone(p.poll())

    def test_agreeing_stops_the_foreign_process_and_starts(self):
        p = self.listener()
        with mock.patch.object(llm, "ask_yn", lambda q, default=True: str(p.pid) in q), \
                mock.patch.object(llm.sys.stdin, "isatty", lambda: True), \
                mock.patch.object(llm, "engine_cmd", self.engine()), mock.patch.object(llm, "show_endpoints", lambda *a: None):
            ok, out = quiet(llm.start, self.cfg, True)
        self.assertTrue(ok, out)
        p.wait(5)
        self.assertIsNotNone(p.returncode)

    def test_a_crashing_engine_reports_its_log_and_does_not_hang(self):
        crash = lambda cfg, m: [sys.executable, "-c", "import sys; print('boom: no such weights'); sys.exit(3)"]
        with mock.patch.object(llm, "engine_cmd", crash):
            ok, out = quiet(llm.start, self.cfg, False)
        self.assertFalse(ok)
        self.assertIn("code 3", out)
        self.assertIn("boom: no such weights", out)

    def test_a_load_that_never_finishes_is_killed_after_the_timeout(self):
        hang = lambda cfg, m: [sys.executable, "-c", "import time; time.sleep(60)"]
        with mock.patch.object(llm, "engine_cmd", hang), mock.patch.object(llm, "time", FakeTime()):
            ok, out = quiet(llm.start, self.cfg, False)
        self.assertFalse(ok)
        self.assertIn("timed out", out)
        self.assertFalse(os.path.exists(llm.ENGINE_PID))               # no orphan left holding the port
        pid = ReapingPopen.spawned[-1]
        deadline = time.time() + 5
        while time.time() < deadline:
            try:
                os.kill(pid, 0)
            except OSError:
                break
            time.sleep(0.05)
        with self.assertRaises(OSError):                                # the hung process itself is gone
            os.kill(pid, 0)

    def test_a_model_that_needs_mlx_lm_is_refused_with_the_fix(self):
        write_model(self.model, config={"model_type": "glm4", "text_config": {}})
        with mock.patch.object(llm, "mlxlm_installed", lambda: False):
            ok, out = quiet(llm.start, self.cfg, False)
        self.assertFalse(ok)
        self.assertIn("setup --compat", out)


class TestVisionSwitch(State):
    def setUp(self):
        super().setUp()
        self.model = write_model(os.path.join(self.cfg["dirs"][0], "v"), config={
            "model_type": "qwen3_5", "text_config": {"model_type": "qwen3_5_text"}, "vision_config": {"depth": 1}})
        self.cfg["model"] = self.model
        self.cfg["models"][self.model] = {"vision": False}

    def test_it_sets_the_model_setting_and_restarts_only_a_running_engine(self):
        calls = []
        with mock.patch.object(llm, "restart", lambda c, i=True: calls.append("restart") or True), \
                mock.patch.object(llm, "read_pid", lambda p: None):
            ok, out = quiet(llm.set_vision, self.cfg, "on")
        self.assertTrue(ok, out)
        self.assertTrue(llm.mcfg(self.cfg)["vision"])
        self.assertEqual(calls, [])
        with mock.patch.object(llm, "restart", lambda c, i=True: calls.append("restart") or True), \
                mock.patch.object(llm, "read_pid", lambda p: 123):
            quiet(llm.set_vision, self.cfg, "off")
        self.assertFalse(llm.mcfg(self.cfg)["vision"])
        self.assertEqual(calls, ["restart"])

    def test_a_model_without_a_vision_tower_says_so(self):
        plain = write_model(os.path.join(self.cfg["dirs"][0], "p"))
        self.cfg["model"] = plain
        ok, out = quiet(llm.set_vision, self.cfg, "on")
        self.assertFalse(ok)
        self.assertIn("vision", out.lower())


class TestSettingsAndSelection(State):
    def test_per_model_settings_override_defaults_only_for_that_model(self):
        a = write_model(os.path.join(self.cfg["dirs"][0], "a"))
        b = write_model(os.path.join(self.cfg["dirs"][0], "b"))
        self.cfg["model"] = a
        llm.set_mcfg(self.cfg, "context", 12345)
        self.assertEqual(llm.mcfg(self.cfg)["context"], 12345)
        self.assertEqual(llm.mcfg(self.cfg, b)["context"], llm.MODEL_DEFAULTS["context"])
        self.assertEqual(llm.mcfg(self.cfg)["max_output"], llm.MODEL_DEFAULTS["max_output"])

    def test_choose_model_by_number_name_and_ambiguity(self):
        root = self.cfg["dirs"][0]
        write_model(os.path.join(root, "alpha-4bit"))
        write_model(os.path.join(root, "alpha-8bit"))
        write_model(os.path.join(root, "beta"))
        with mock.patch.object(llm, "restart", lambda *a: None):
            ok, out = quiet(llm.choose_model, self.cfg, "beta")
            self.assertTrue(ok)
            self.assertTrue(self.cfg["model"].endswith("beta"))
            ok, out = quiet(llm.choose_model, self.cfg, "alpha")
            self.assertFalse(ok)
            self.assertIn("Several matches", out)
            ok, out = quiet(llm.choose_model, self.cfg, "nope")
            self.assertFalse(ok)
            ok, _ = quiet(llm.choose_model, self.cfg, "1")
            self.assertTrue(ok)

    def test_rm_by_name_asks_unless_yes_and_reports_ambiguity(self):
        root = self.cfg["dirs"][0]
        a = write_model(os.path.join(root, "alpha-4bit"))
        write_model(os.path.join(root, "alpha-8bit"))
        ok, out = quiet(llm.remove_cmd, self.cfg, ["alpha", "--yes"])
        self.assertFalse(ok)
        self.assertIn("Several matches", out)
        self.assertTrue(os.path.isdir(a))
        with mock.patch.object(llm, "ask_yn", lambda *x, **k: False):
            ok, _ = quiet(llm.remove_cmd, self.cfg, ["alpha-4bit"])
        self.assertFalse(ok)
        self.assertTrue(os.path.isdir(a))
        ok, out = quiet(llm.remove_cmd, self.cfg, ["alpha-4bit", "--yes"])
        self.assertTrue(ok)
        self.assertFalse(os.path.exists(a))


# ───────────────────────────── hub ─────────────────────────────
WEIGHTS = os.urandom(200_000)
CONFIG = json.dumps({"model_type": "qwen3_5", "text_config": {"model_type": "qwen3_5_text"},
                     "quantization": {"bits": 4, "group_size": 64}}).encode()
REPOS = {
    "org/native-4bit": {"tags": ["mlx", "safetensors", "qwen3_5"], "downloads": 500,
                        "files": {"config.json": CONFIG, "model.safetensors": WEIGHTS, "preview.png": b"png",
                                  "tokenizer.json": b"{}"}},
    "org/packed-jang": {"tags": ["mlx", "safetensors", "qwen3_5", "packed-trits"], "downloads": 90,
                        "files": {"config.json": CONFIG, "jang_config.json": json.dumps(
                            {"runtime": {"requires_jang_ternary_packed_expansion": True}}).encode(),
                            "model.safetensors": WEIGHTS}},
    "org/gguf-only": {"tags": ["gguf", "llama"], "downloads": 5000, "files": {"m.gguf": b"x"}},
    "org/mlx-and-gguf": {"tags": ["mlx", "gguf", "qwen3_5"], "downloads": 70,
                         "files": {"config.json": CONFIG, "model.safetensors": WEIGHTS}},
    "org/glm-mlx": {"tags": ["mlx", "safetensors", "glm4"], "downloads": 40,
                    "files": {"config.json": CONFIG, "model.safetensors": WEIGHTS}},
}


class FakeHub(BaseHTTPRequestHandler):
    corrupt = False

    def log_message(self, *a):
        pass

    def send(self, code, body, ctype="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/api/models":
            return self.send(200, json.dumps([{"id": r, "downloads": v["downloads"], "likes": 1, "tags": v["tags"]}
                                              for r, v in REPOS.items()]).encode())
        if path.startswith("/api/models/"):
            repo = path[len("/api/models/"):]
            if repo not in REPOS:
                return self.send(404, b"{}")
            sib = [{"rfilename": n, "size": len(b), "lfs": {"sha256": hashlib.sha256(b).hexdigest()}}
                   for n, b in REPOS[repo]["files"].items()]
            return self.send(200, json.dumps({"id": repo, "sha": "main", "tags": REPOS[repo]["tags"], "siblings": sib}).encode())
        parts = path.lstrip("/").split("/")                      # org/name/resolve/rev/file
        if len(parts) >= 5 and parts[2] == "resolve":
            repo, name = "/".join(parts[:2]), "/".join(parts[4:])
            body = REPOS.get(repo, {}).get("files", {}).get(name)
            if body is None:
                return self.send(404, b"")
            if FakeHub.corrupt and name == "model.safetensors":
                body = bytes(len(body))
            return self.send(200, body, "application/octet-stream")
        self.send(404, b"")


class TestHub(State):
    @classmethod
    def setUpClass(cls):
        cls.srv = ThreadingHTTPServer(("127.0.0.1", 0), FakeHub)
        cls.endpoint = f"http://127.0.0.1:{cls.srv.server_address[1]}"
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        cls.srv.server_close()

    def setUp(self):
        super().setUp()
        FakeHub.corrupt = False
        self.home = os.path.join(self.tmp, "home")
        for name, val in (("HF_ENDPOINT", self.endpoint), ("HOME", self.home)):
            p = mock.patch.object(llm, name, val)
            p.start()
            self.addCleanup(p.stop)

    def test_search_reports_size_route_and_local_copies(self):
        os.makedirs(llm._pull_dest("org/glm-mlx"))
        res = {r["id"]: r for r in llm.hf_search("anything")}
        self.assertEqual(res["org/native-4bit"]["route"], "native")
        self.assertTrue(res["org/native-4bit"]["compatible"])
        self.assertAlmostEqual(res["org/native-4bit"]["size"] * 2**30, len(WEIGHTS) + len(CONFIG) + 2, delta=2)  # .png skipped
        self.assertEqual(res["org/glm-mlx"]["route"], "mlxlm")
        self.assertFalse(res["org/glm-mlx"]["local"])     # the folder exists but nothing is in it
        self.assertTrue(res["org/glm-mlx"]["partial"])
        self.assertEqual(res["org/gguf-only"]["route"], "none")
        self.assertEqual(res["org/mlx-and-gguf"]["route"], "none")      # a GGUF tag wins over mlx/qwen3_5
        self.assertFalse(res["org/gguf-only"]["compatible"])
        self.assertTrue(res["org/packed-jang"]["compatible"])         # packed ternary runs on the ptq1 kernel
        self.assertEqual(res["org/packed-jang"]["route"], "ptq1")
        self.assertTrue(res["org/native-4bit"]["fits"])

    def test_a_half_downloaded_model_is_partial_not_installed(self):
        dest = llm._pull_dest("org/native-4bit")
        os.makedirs(dest)
        with open(os.path.join(dest, "config.json"), "wb") as f:
            f.write(CONFIG)
        with open(os.path.join(dest, "model.safetensors.part"), "wb") as f:
            f.write(WEIGHTS[:1000])
        res = {r["id"]: r for r in llm.hf_search("x")}["org/native-4bit"]
        self.assertFalse(res["local"])                    # a folder that merely exists is not a model
        self.assertTrue(res["partial"])
        self.assertEqual(llm.install_state("org/native-4bit"), "partial")
        self.assertEqual(llm.install_state("org/never-seen"), "none")
        # resuming finishes it, and only then does it count as installed
        ok, _ = quiet(llm.hf_pull, "org/native-4bit")
        self.assertTrue(ok)
        res = {r["id"]: r for r in llm.hf_search("x")}["org/native-4bit"]
        self.assertTrue(res["local"])
        self.assertFalse(res["partial"])

    def test_a_short_file_under_the_final_name_is_partial_too(self):
        dest = llm._pull_dest("org/glm-mlx")
        os.makedirs(dest)
        for name, body in (("config.json", CONFIG), ("model.safetensors", WEIGHTS[:5000])):
            with open(os.path.join(dest, name), "wb") as f:
                f.write(body)
        # (without the hub listing sizes are unknown, so only hf_search below — which has it — can tell)
        res = {r["id"]: r for r in llm.hf_search("x")}["org/glm-mlx"]
        self.assertTrue(res["partial"])
        self.assertFalse(res["local"])

    def test_pull_downloads_verifies_and_skips_previews(self):
        ok, out = quiet(llm.hf_pull, "org/native-4bit")
        self.assertTrue(ok, out)
        dest = llm._pull_dest("org/native-4bit")
        with open(os.path.join(dest, "model.safetensors"), "rb") as f:
            self.assertEqual(f.read(), WEIGHTS)
        self.assertFalse(os.path.exists(os.path.join(dest, "preview.png")))
        self.assertFalse([n for n in os.listdir(dest) if n.endswith(".part")])
        self.assertIn("Downloaded", out)
        # a second pull finds everything complete and fetches nothing new
        ok2, _ = quiet(llm.hf_pull, "org/native-4bit")
        self.assertTrue(ok2)

    def test_progress_mode_prints_machine_readable_lines(self):
        _, out = quiet(llm.hf_pull, "org/glm-mlx", True)
        lines = [l for l in out.splitlines() if l.startswith("PROGRESS")]
        self.assertTrue(lines)
        self.assertTrue(lines[-1].endswith("done"))
        total = len(WEIGHTS) + len(CONFIG)
        self.assertEqual(lines[-1].split()[1:3], [str(total), str(total)])

    def test_a_corrupt_download_fails_loudly_and_leaves_no_model(self):
        FakeHub.corrupt = True
        ok, out = quiet(llm.hf_pull, "org/native-4bit")
        self.assertFalse(ok)
        self.assertIn("Download failed", out)
        self.assertFalse(os.path.exists(os.path.join(llm._pull_dest("org/native-4bit"), "model.safetensors")))

    def test_not_enough_disk_space_is_refused_before_downloading(self):
        usage = shutil_usage = mock.Mock(free=1 * 2**30)
        with mock.patch.object(llm.shutil, "disk_usage", lambda p: usage):
            ok, out = quiet(llm.hf_pull, "org/native-4bit")
        self.assertFalse(ok)
        self.assertIn("Not enough disk space", out)
        self.assertFalse(os.path.exists(os.path.join(llm._pull_dest("org/native-4bit"), "model.safetensors")))

    def test_search_cmd_json_is_what_the_app_parses(self):
        _, out = quiet(llm.search_cmd, "x", True)
        data = json.loads(out)
        self.assertEqual({"id", "downloads", "size", "route", "compatible", "local", "fits"} - set(data[0]), set())


if __name__ == "__main__":
    unittest.main()
