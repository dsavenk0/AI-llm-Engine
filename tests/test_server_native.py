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


class TestWatchdog(unittest.TestCase):
    """The engine stops itself when the app that launched it goes away (no orphaned engine / llama-server)."""

    def test_pid_alive(self):
        from engine.server import pid_alive
        self.assertTrue(pid_alive(0))                 # no parent to watch
        self.assertTrue(pid_alive(os.getpid()))
        p = subprocess.Popen([sys.executable, "-c", "pass"])
        p.wait()
        self.assertFalse(pid_alive(p.pid))            # it has exited

    def test_watch_parent_fires_once_when_the_parent_goes(self):
        from engine.server import watch_parent
        seen = {"alive": [True, True, False], "slept": 0, "gone": 0}

        def alive(_pid):
            return seen["alive"].pop(0) if seen["alive"] else False

        def sleep(_s):
            seen["slept"] += 1
        watch_parent(1234, lambda: seen.__setitem__("gone", seen["gone"] + 1), sleep=sleep, alive=alive)
        self.assertEqual((seen["gone"], seen["slept"]), (1, 2))


class TestFinishReason(unittest.TestCase):
    """A reply cut off for looping must read as a clear reason, not a plain stop."""

    def test_mapping(self):
        from engine.server import finish_reason
        self.assertEqual(finish_reason({"loop_stopped": True}, False), ("length", "repetition"))
        self.assertEqual(finish_reason({"finish_reason": "length"}, False), ("length", None))
        self.assertEqual(finish_reason({"finish_reason": "stop"}, False), ("stop", None))
        self.assertEqual(finish_reason({}, False), ("stop", None))
        self.assertEqual(finish_reason({"loop_stopped": True}, True), ("tool_calls", None))   # tool calls win


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

    def test_health_reports_prefill_progress_while_a_long_prompt_is_processed(self):
        import threading
        seen = []
        stop = threading.Event()

        def poll():
            while not stop.is_set():
                try:
                    p = self.srv.get("/health").get("progress")
                    if p:
                        seen.append(p)
                except OSError:
                    pass
                time.sleep(0.01)
        t = threading.Thread(target=poll)
        t.start()
        try:
            self.srv.chat("the quick brown fox jumps over the lazy dog " * 90, max_tokens=2)
        finally:
            stop.set()
            t.join()
        self.assertIsNone(self.srv.get("/health")["progress"])           # and it is gone afterwards
        for p in seen:                                                      # whatever was caught is well formed
            self.assertEqual(p["phase"], "prefill")
            self.assertTrue(0 <= p["done"] <= p["total"])

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

    def test_a_picture_for_a_text_only_model_is_dropped_not_refused(self):
        url = "data:image/png;base64,iVBORw0KGgo="
        parts = [{"type": "image_url", "image_url": {"url": url}}, {"type": "text", "text": "hello world"}]
        code, r = self.srv.chat(parts)
        self.assertEqual(code, 200, r)
        self.assertGreater(r["usage"]["prompt_tokens"], 0)
        code, text_only = self.srv.chat("hello world")
        self.assertGreater(r["usage"]["prompt_tokens"], text_only["usage"]["prompt_tokens"])    # the note was added
        self.assertEqual(self.srv.get("/health")["vision"], False)
        status, h, raw = self.srv.request("POST", "/v1/chat/completions",
                                          {"model": "synth", "temperature": 0, "max_tokens": 4, "stream": True,
                                           "messages": [{"role": "user", "content": parts}]})
        self.assertEqual(status, 200)
        self.assertIn(b"[DONE]", raw)

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

    def test_unusable_parameters_are_400s_before_anything_runs(self):
        for kw, word in (({"max_tokens": "many"}, "max_tokens"), ({"max_tokens": 0}, "max_tokens"),
                         ({"max_tokens": True}, "max_tokens"), ({"max_completion_tokens": -3}, "max_completion_tokens"),
                         ({"temperature": "hot"}, "temperature"), ({"top_p": [1]}, "top_p"),
                         ({"seed": "x"}, "seed"), ({"stop": 5}, "stop"), ({"stop": ["a", 1]}, "stop"),
                         ({"stop": ["a"] * 17}, "stop"), ({"temperature": float("inf")}, "temperature")):
            with self.subTest(kw):
                code, r = self.srv.chat("hi", **kw)
                self.assertEqual(code, 400, r)
                self.assertIn(word, r["error"]["message"])
        code, r = self.srv.chat("hi", max_tokens=3, temperature=0.0, top_k=5, seed=3, stop=["zz"])
        self.assertEqual(code, 200, r)

    def test_a_damaged_stats_file_does_not_break_health(self):
        with open(self.stats, "a") as f:
            f.write("5\n[1]\n\"x\"\nnot json\n")
        self.assertEqual(self.srv.get("/v1/stats")["all_time"]["requests"] >= 0, True)
        self.assertEqual(self.srv.get("/health")["status"], "ok")

    def test_an_oversized_or_non_object_body_is_refused_without_being_read(self):
        import http.client
        conn = http.client.HTTPConnection("127.0.0.1", self.srv.port, timeout=10)
        conn.putrequest("POST", "/v1/chat/completions")
        conn.putheader("Content-Type", "application/json")
        conn.putheader("Content-Length", str(10 ** 11))           # announced, never sent
        conn.endheaders()
        r = conn.getresponse()
        self.assertEqual(r.status, 413)
        conn.close()
        code, r = self.srv.json("POST", "/v1/chat/completions", raw=b'["not", "an", "object"]')
        self.assertEqual(code, 400)

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

    # ───────────── Ollama wire format ─────────────
    def _ndjson(self, path, body):
        code, h, raw = self.srv.request("POST", path, body)
        lines = [json.loads(x) for x in raw.decode().split("\n") if x.strip()]
        return code, h, lines

    def test_ollama_discovery(self):
        self.assertIn("version", self.srv.get("/api/version"))
        tags = self.srv.get("/api/tags")
        self.assertEqual([m["name"] for m in tags["models"]], ["synth"])
        self.assertTrue(tags["models"][0]["digest"].startswith("sha256:"))
        code, show = self.srv.json("POST", "/api/show", {"model": "synth"})
        self.assertEqual(code, 200)
        self.assertIn("completion", show["capabilities"])
        self.assertIn("tools", show["capabilities"])
        self.assertGreater(show["model_info"]["general.context_length"], 0)
        ps = self.srv.get("/api/ps")                       # awake → the model is listed
        self.assertEqual([m["name"] for m in ps["models"]], ["synth"])

    def test_ollama_chat_nonstream(self):
        code, r = self.srv.json("POST", "/api/chat", {"model": "synth", "stream": False,
                                "options": {"temperature": 0, "num_predict": 8},
                                "messages": [{"role": "user", "content": "hello world"}]})
        self.assertEqual(code, 200, r)
        self.assertTrue(r["done"])
        self.assertIn(r["done_reason"], ("stop", "length"))   # synth truncates at num_predict → "length"
        self.assertEqual(r["message"]["role"], "assistant")
        self.assertIsInstance(r["message"]["content"], str)
        self.assertGreater(r["prompt_eval_count"], 0)

    def test_ollama_chat_stream_is_ndjson_ending_done(self):
        code, h, lines = self._ndjson("/api/chat", {"model": "synth",
                                      "options": {"temperature": 0, "num_predict": 8},
                                      "messages": [{"role": "user", "content": "what is two plus two"}]})
        self.assertEqual(code, 200)
        self.assertIn("x-ndjson", h.get("Content-Type", ""))
        self.assertFalse(lines[0]["done"])                 # the first line is a delta, not the end
        self.assertTrue(lines[-1]["done"])                 # the last one is the terminator
        self.assertIn(lines[-1]["done_reason"], ("stop", "length"))
        streamed = "".join(x["message"]["content"] for x in lines if not x["done"])
        self.assertTrue(streamed)                          # something was actually streamed

    def test_ollama_generate_uses_response_field(self):
        code, r = self.srv.json("POST", "/api/generate", {"model": "synth", "stream": False,
                                "options": {"temperature": 0, "num_predict": 8}, "prompt": "hello"})
        self.assertEqual(code, 200, r)
        self.assertTrue(r["done"])
        self.assertIsInstance(r["response"], str)
        self.assertGreater(r["prompt_eval_count"], 0)

    def test_ollama_chat_without_messages_is_a_clean_error(self):
        code, r = self.srv.json("POST", "/api/chat", {"model": "synth", "stream": False, "messages": []})
        self.assertEqual(code, 400)
        self.assertIn("error", r)                           # Ollama's bare {"error": "..."} shape

    # ───────────── Anthropic Messages API ─────────────
    def test_anthropic_messages_nonstream(self):
        code, r = self.srv.json("POST", "/v1/messages", {"model": "synth", "max_tokens": 8, "temperature": 0,
                                "messages": [{"role": "user", "content": "hello world"}]})
        self.assertEqual(code, 200, r)
        self.assertEqual(r["type"], "message")
        self.assertEqual(r["role"], "assistant")
        self.assertTrue(r["id"].startswith("msg_"))
        self.assertIn(r["stop_reason"], ("end_turn", "max_tokens", "tool_use"))
        self.assertEqual([b["type"] for b in r["content"] if b["type"] == "text"] or ["text"], ["text"])
        self.assertGreater(r["usage"]["input_tokens"], 0)

    def test_anthropic_messages_stream_events(self):
        code, h, raw = self.srv.request("POST", "/v1/messages", {"model": "synth", "max_tokens": 8,
                                        "temperature": 0, "stream": True,
                                        "messages": [{"role": "user", "content": "two plus two"}]})
        self.assertEqual(code, 200)
        self.assertIn("event-stream", h.get("Content-Type", ""))
        types = [ln[len("event: "):] for ln in raw.decode().split("\n") if ln.startswith("event: ")]
        self.assertEqual(types[0], "message_start")
        self.assertEqual(types[-1], "message_stop")
        self.assertIn("content_block_start", types)
        self.assertIn("content_block_delta", types)
        self.assertIn("message_delta", types)

    def test_anthropic_messages_without_messages_is_a_clean_error(self):
        code, r = self.srv.json("POST", "/v1/messages", {"model": "synth", "max_tokens": 8, "messages": []})
        self.assertEqual(code, 400)
        self.assertEqual(r["type"], "error")                # Anthropic {"type":"error","error":{...}} shape


class TestNativeServerOptions(unittest.TestCase):
    """--fallback-model, --log-level (with group tags), --prefill-guard."""

    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.mkdtemp()
        make_bundle(cls.dir, CHATML, seed=13)

    def with_server(self, *extra):
        srv = Server(self.dir, *extra)
        self.addCleanup(srv.stop)
        return srv

    def test_fallback_on_answers_any_name(self):
        srv = self.with_server()
        code, r = srv.chat(model="gpt-4o")
        self.assertEqual(code, 200, r)

    def test_fallback_off_refuses_another_model_but_accepts_the_loaded_one_in_any_spelling(self):
        srv = self.with_server("--fallback-model", "off")
        code, r = srv.chat(model="some-other-model")
        self.assertEqual(code, 404)
        self.assertEqual(r["error"]["code"], "model_not_found")
        self.assertIn("synth", r["error"]["message"])
        for ok in ("synth", "SYNTH", "org/synth"):
            self.assertEqual(srv.chat(model=ok)[0], 200, ok)

    def test_log_lines_carry_a_group_and_the_level_filters(self):
        quiet = self.with_server("--log-level", "warning")
        quiet.chat()
        loud = self.with_server("--log-level", "trace")
        loud.chat()
        for s in (quiet, loud):
            s.log.flush()
        q, l = open(quiet.log.name).read(), open(loud.log.name).read()
        self.assertNotIn("[request]", q)                     # info lines are below "warning"
        self.assertNotIn("[model]", q)
        self.assertIn("[request]", l)
        self.assertIn("[model]", l)
        self.assertIn("trace: request", l)                    # only at trace level

    def test_parsers_are_reported_and_validated_at_start(self):
        srv = self.with_server("--reasoning-parser", "tags:</reasoning>", "--tool-parser", "json")
        self.assertEqual(srv.get("/health")["parsers"], {"reasoning": "tags:</reasoning>", "tool_call": "json"})
        import subprocess
        bad = subprocess.run([sys.executable, "-B", "-m", "engine.server", "--model", self.dir,
                              "--reasoning-parser", "bogus"], cwd=ROOT, capture_output=True, text=True, timeout=60)
        self.assertNotEqual(bad.returncode, 0)
        self.assertIn("--reasoning-parser", bad.stderr)

    def test_an_activation_edit_is_refused_where_it_cannot_be_applied(self):
        import subprocess
        bad = subprocess.run([sys.executable, "-B", "-m", "engine.server", "--model", self.dir, "--backend", "mlxlm",
                              "--steer", "/nonexistent/direction.safetensors"],
                             cwd=ROOT, capture_output=True, text=True, timeout=60)
        self.assertNotEqual(bad.returncode, 0)
        self.assertIn("--steer", bad.stderr)
        self.assertIn("mlx-lm", bad.stderr)

    def test_the_prefill_guard_can_be_switched_off(self):
        srv = self.with_server("--prefill-guard", "off")
        self.assertEqual(srv.chat()[0], 200)


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
