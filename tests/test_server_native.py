"""The native engine behind its HTTP API, end to end: a real engine.server process on a synthetic
bundle (random weights, see tests/synth.py) — no model download needed.

    .venv/bin/python -m unittest tests.test_server_native -v
"""
import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from tests.synth import CHATML, make_bundle  # noqa: E402


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Server:
    def __init__(self, model, *extra):
        self.port = free_port()
        self.base = f"http://127.0.0.1:{self.port}"
        self.log = tempfile.NamedTemporaryFile("w+", suffix=".log", delete=False)
        self.proc = subprocess.Popen(
            [sys.executable, "-B", "-m", "engine.server", "--model", model, "--port", str(self.port),
             "--name", "synth", "--context", "1024", "--max-tokens", "16", "--disk-cache-gb", "0",
             "--backend", "native", *extra],
            cwd=ROOT, stdout=self.log, stderr=subprocess.STDOUT)
        deadline = time.time() + 120
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError("server exited:\n" + open(self.log.name).read()[-2000:])
            try:
                self.get("/health")
                return
            except OSError:
                time.sleep(0.3)
        raise RuntimeError("server did not come up")

    def stop(self):
        self.proc.terminate()
        try:
            self.proc.wait(10)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        self.log.close()

    def request(self, method, path, body=None, headers=None, raw=None):
        data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
        req = urllib.request.Request(self.base + path, data=data, method=method,
                                     headers={"Content-Type": "application/json", **(headers or {})})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.status, dict(r.headers), r.read()
        except urllib.error.HTTPError as e:
            try:
                return e.code, dict(e.headers), e.read()
            finally:
                e.close()

    def get(self, path, headers=None):
        code, h, b = self.request("GET", path, headers=headers)
        return json.loads(b) if b else {}

    def json(self, method, path, body=None, headers=None, raw=None):
        code, h, b = self.request(method, path, body, headers, raw)
        return code, (json.loads(b) if b else {})

    def chat(self, content="hello world", **kw):
        body = {"model": "synth", "temperature": 0, "max_tokens": 8,
                "messages": [{"role": "user", "content": content}]}
        body.update(kw)
        return self.json("POST", "/v1/chat/completions", body)

    def stream(self, content="hello world", **kw):
        body = {"model": "synth", "temperature": 0, "max_tokens": 8, "stream": True,
                "messages": [{"role": "user", "content": content}]}
        body.update(kw)
        code, h, raw = self.request("POST", "/v1/chat/completions", body)
        events, done = [], False
        for line in raw.decode().split("\n"):
            if line.startswith("data: "):
                if line == "data: [DONE]":
                    done = True
                else:
                    events.append(json.loads(line[6:]))
        return code, events, done


class TestNativeServer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.mkdtemp()
        make_bundle(cls.dir, CHATML, seed=11)
        cls.stats = os.path.join(cls.dir, "stats.jsonl")
        cls.srv = Server(cls.dir, "--stats-file", cls.stats)

    @classmethod
    def tearDownClass(cls):
        cls.srv.stop()

    # ───────────── discovery ─────────────
    def test_health_describes_the_model_and_limits(self):
        h = self.srv.get("/health")
        self.assertEqual((h["status"], h["model"], h["state"]), ("ok", "synth", "awake"))
        self.assertEqual(h["max_tokens"], 16)
        self.assertLessEqual(h["context"], 1024)
        self.assertFalse(h["thinking"])                    # the synthetic template has no reasoning
        self.assertFalse(h["vision"])
        for k in ("memory_gb", "memory_breakdown", "system_ram_used", "mem_limit", "requests", "busy"):
            self.assertIn(k, h)

    def test_models_endpoint(self):
        code, r = self.srv.json("GET", "/v1/models")
        self.assertEqual((code, r["object"], r["data"][0]["id"]), (200, "list", "synth"))
        self.assertEqual(r["data"][0]["max_output_tokens"], 16)
        code, one = self.srv.json("GET", "/v1/models/synth")
        self.assertEqual((code, one["id"]), (200, "synth"))

    def test_unknown_paths_and_cors_preflight(self):
        code, r = self.srv.json("GET", "/v1/nope")
        self.assertEqual(code, 404)
        self.assertIn("error", r)
        code, headers, _ = self.srv.request("OPTIONS", "/v1/chat/completions")
        self.assertEqual(code, 204)
        self.assertEqual({k.lower(): v for k, v in headers.items()}["access-control-allow-origin"], "*")

    # ───────────── chat ─────────────
    def test_chat_shape_usage_and_output_cap(self):
        code, r = self.srv.chat("count to ten", max_tokens=1000)         # asks for more than the server allows
        self.assertEqual(code, 200)
        self.assertTrue(r["id"].startswith("chatcmpl-"))
        self.assertEqual(r["object"], "chat.completion")
        c = r["choices"][0]
        self.assertEqual((c["index"], c["message"]["role"]), (0, "assistant"))
        self.assertIn(c["finish_reason"], ("stop", "length"))
        u = r["usage"]
        self.assertEqual(u["total_tokens"], u["prompt_tokens"] + u["completion_tokens"])
        self.assertGreater(u["prompt_tokens"], 0)
        self.assertLessEqual(u["completion_tokens"], 16)                 # clamped, not rejected

    def test_temperature_zero_is_deterministic_and_second_call_hits_the_prefix_cache(self):
        code, a = self.srv.chat("the quick brown fox jumps over the lazy dog " * 3)
        code, b = self.srv.chat("the quick brown fox jumps over the lazy dog " * 3)
        self.assertEqual(a["choices"][0]["message"]["content"], b["choices"][0]["message"]["content"])
        self.assertGreater(b["usage"]["prompt_tokens_details"]["cached_tokens"], 0)
        self.assertGreater(b["usage"]["prompt_tokens_details"]["cached_tokens"],
                           a["usage"]["prompt_tokens_details"]["cached_tokens"])

    def test_seed_makes_sampling_reproducible(self):
        kw = dict(temperature=1.0, seed=1234)
        _, a = self.srv.chat("sample something", **kw)
        _, b = self.srv.chat("sample something else entirely", **kw)     # evict the prefix, same seed below
        _, c = self.srv.chat("sample something", **kw)
        self.assertEqual(a["choices"][0]["message"]["content"], c["choices"][0]["message"]["content"])
        self.assertIn("choices", b)

    def test_streaming_matches_the_plain_answer_and_ends_properly(self):
        _, plain = self.srv.chat("what is two plus two")
        code, ev, done = self.srv.stream("what is two plus two", stream_options={"include_usage": True})
        self.assertEqual(code, 200)
        self.assertTrue(done)                                            # data: [DONE]
        self.assertEqual(ev[0]["choices"][0]["delta"].get("role"), "assistant")
        text = "".join(c["delta"].get("content") or "" for e in ev for c in e["choices"])
        self.assertEqual(text, plain["choices"][0]["message"]["content"])
        finals = [c["finish_reason"] for e in ev for c in e["choices"] if c.get("finish_reason")]
        self.assertEqual(len(finals), 1)
        self.assertEqual(ev[-1]["choices"], [])                          # usage chunk has no choices
        self.assertEqual(ev[-1]["usage"]["completion_tokens"], plain["usage"]["completion_tokens"])

    def test_stop_sequences_cut_the_answer(self):
        _, full = self.srv.chat("tell me about the fox", max_tokens=16)
        text = full["choices"][0]["message"]["content"]
        if len(text) < 6:
            self.skipTest("the random model answered too briefly to cut")
        stop = text[3:5]
        _, cut = self.srv.chat("tell me about the fox", max_tokens=16, stop=[stop])
        got = cut["choices"][0]["message"]["content"]
        self.assertNotIn(stop, got)
        self.assertTrue(text.startswith(got))
        self.assertEqual(cut["choices"][0]["finish_reason"], "stop")

    def test_tools_are_accepted_by_a_model_that_never_calls_them(self):
        tools = [{"type": "function", "function": {"name": "f", "description": "d",
                                                   "parameters": {"type": "object", "properties": {}}}}]
        code, r = self.srv.chat("use the tool", tools=tools)
        self.assertEqual(code, 200)
        self.assertNotIn("tool_calls", r["choices"][0]["message"])

    # ───────────── errors ─────────────
    def test_bad_requests_are_400s_with_error_bodies(self):
        code, r = self.srv.json("POST", "/v1/chat/completions", {"model": "synth", "messages": []})
        self.assertEqual(code, 400)
        self.assertIn("messages", r["error"]["message"])
        code, r = self.srv.json("POST", "/v1/chat/completions", raw=b"{not json")
        self.assertEqual(code, 400)
        code, r = self.srv.chat("word " * 3000)
        self.assertEqual(code, 400)
        self.assertEqual(r["error"]["code"], "context_length_exceeded")

    # ───────────── behaviour under load ─────────────
    def test_parallel_requests_are_serialized_not_dropped(self):
        results = []

        def go(i):
            results.append(self.srv.chat(f"request number {i}")[0])
        threads = [threading.Thread(target=go, args=(i,)) for i in range(4)]
        [t.start() for t in threads]
        [t.join(60) for t in threads]
        self.assertEqual(results, [200] * 4)

    def test_a_client_that_hangs_up_mid_stream_does_not_hurt_the_server(self):
        s = socket.create_connection(("127.0.0.1", self.srv.port))
        body = json.dumps({"model": "synth", "stream": True, "max_tokens": 16, "temperature": 0,
                           "messages": [{"role": "user", "content": "a long answer please"}]}).encode()
        s.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"
                  + f"Content-Length: {len(body)}\r\n\r\n".encode() + body)
        s.recv(200)
        s.close()
        time.sleep(0.5)
        self.assertEqual(self.srv.get("/health")["status"], "ok")
        self.assertEqual(self.srv.chat("still alive?")[0], 200)

    def test_stats_are_recorded_per_request_and_exposed(self):
        before = self.srv.get("/v1/stats")["all_time"]["requests"]
        self.srv.chat("count me")
        after = self.srv.get("/v1/stats")
        self.assertEqual(after["all_time"]["requests"], before + 1)
        self.assertIn("session", after)
        with open(self.stats) as f:
            row = json.loads(f.readlines()[-1])
        self.assertEqual(row["model"], "synth")
        self.assertGreater(row["prompt"], 0)


class TestNativeServerNoRoomForVision(unittest.TestCase):
    """Loading the vision tower is refused when RAM is over the limit: the client gets a JSON 503, not a
    dropped connection (the error used to escape the handler)."""

    @classmethod
    def setUpClass(cls):
        from tests.synth import make_vision_bundle
        cls.dir = tempfile.mkdtemp()
        make_vision_bundle(cls.dir, seed=12)
        cls.srv = Server(cls.dir, "--mem-limit", "0.01")      # 1 % of RAM: always "over the limit"

    @classmethod
    def tearDownClass(cls):
        cls.srv.stop()

    def test_an_image_request_gets_a_clean_503(self):
        import base64
        import struct
        import zlib
        raw = b"".join(b"\x00" + b"\x80\x20\x20" * 16 for _ in range(16))

        def chunk(t, d):
            return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
        png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 16, 16, 8, 2, 0, 0, 0))
               + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))
        url = "data:image/png;base64," + base64.b64encode(png).decode()
        code, r = self.srv.json("POST", "/v1/chat/completions", {
            "model": "synth", "max_tokens": 4, "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": url}}, {"type": "text", "text": "what is this?"}]}]})
        self.assertEqual(code, 503)
        self.assertEqual(r["error"]["code"], "insufficient_memory")
        # and the server is still serving
        self.assertEqual(self.srv.get("/health")["status"], "ok")


class TestNativeServerSleep(unittest.TestCase):
    """An idle engine unloads its weights; /health must keep answering (the app polls it) and the
    next request must wake it. Regression: /health raised TypeError while the model was None."""

    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.mkdtemp()
        make_bundle(cls.dir, CHATML, seed=13)
        # advertise a vision tower (loaded lazily, so no weights are needed): this is what makes the
        # engine report vision as "on demand" — the branch of /health that used to dereference the model
        with open(os.path.join(cls.dir, "config.json")) as f:
            cfg = json.load(f)
        cfg["vision_config"] = {"depth": 1}
        with open(os.path.join(cls.dir, "config.json"), "w") as f:
            json.dump(cfg, f)
        with open(os.path.join(cls.dir, "model.safetensors.index.json"), "w") as f:
            json.dump({"weight_map": {"vision_tower.blocks.0.attn.qkv.weight": "model.safetensors"}}, f)
        cls.srv = Server(cls.dir, "--idle-sleep", "0.03")          # ~1.8 s of idleness

    @classmethod
    def tearDownClass(cls):
        cls.srv.stop()

    def wait_for(self, state, timeout=20):
        deadline = time.time() + timeout
        while time.time() < deadline:
            code, h = self.srv.json("GET", "/health")
            if code == 200 and h.get("state") == state:
                return h
            time.sleep(0.2)
        self.fail(f"engine never reported state {state!r}; last: {code} {h}")

    def test_health_answers_while_sleeping_and_a_request_wakes_the_model(self):
        h = self.wait_for("sleeping")
        self.assertEqual(h["status"], "ok")
        self.assertEqual(h["memory_breakdown"]["weights"], 0)      # the weights really are gone
        self.assertEqual(h["vision"], "on demand")                 # the field that used to crash
        code, r = self.srv.chat("wake up")
        self.assertEqual(code, 200)
        self.assertIn("choices", r)
        self.assertEqual(self.srv.get("/health")["state"], "awake")
        self.assertGreater(self.srv.get("/health")["memory_breakdown"]["weights"], 0)
        self.wait_for("sleeping")                                   # and it can go back to sleep again


class TestNativeServerAuth(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.mkdtemp()
        make_bundle(cls.dir, CHATML, seed=12)
        cls.srv = Server(cls.dir, "--api-key", "s3cret")

    @classmethod
    def tearDownClass(cls):
        cls.srv.stop()

    def test_key_is_required_for_the_api_but_not_for_health(self):
        self.assertEqual(self.srv.json("GET", "/health")[0], 200)
        self.assertEqual(self.srv.json("GET", "/v1/models")[0], 401)
        code, r = self.srv.chat("hi")
        self.assertEqual(code, 401)
        self.assertEqual(r["error"]["type"], "authentication_error")
        auth = {"Authorization": "Bearer s3cret"}
        self.assertEqual(self.srv.json("GET", "/v1/models", headers=auth)[0], 200)
        code, r = self.srv.json("POST", "/v1/chat/completions", headers=auth,
                                body={"model": "synth", "max_tokens": 4, "temperature": 0,
                                      "messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(code, 200)
        self.assertEqual(self.srv.json("GET", "/v1/models", headers={"Authorization": "Bearer nope"})[0], 401)


if __name__ == "__main__":
    unittest.main()
