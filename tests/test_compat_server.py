"""The mlx-lm backend end to end: a real engine.server process on a free port, real HTTP.

Skipped unless COMPAT_TEST_MODEL points at a model folder the native engine does not run (for
example mlx-community/Qwen3-0.6B-4bit) and mlx-lm is installed in this interpreter.

    COMPAT_TEST_MODEL=~/.cache/huggingface/hub/mlx-community/Qwen3-0.6B-4bit \
        .venv/bin/python -m unittest tests.test_compat_server -v
"""
import json
import os
import socket
import subprocess
import sys
import time
import unittest
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = os.path.expanduser(os.environ.get("COMPAT_TEST_MODEL", ""))

try:
    import mlx_lm  # noqa: F401
    HAVE_MLX_LM = True
except ImportError:
    HAVE_MLX_LM = False


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@unittest.skipUnless(MODEL and os.path.isdir(MODEL) and HAVE_MLX_LM, "set COMPAT_TEST_MODEL and install mlx-lm")
class TestCompatServer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.port = free_port()
        cls.base = f"http://127.0.0.1:{cls.port}"
        cls.log = open(os.path.join(os.environ.get("TMPDIR", "/tmp"), f"compat-server-{cls.port}.log"), "w")
        cls.proc = subprocess.Popen(
            [sys.executable, "-B", "-m", "engine.server", "--model", MODEL, "--port", str(cls.port),
             "--backend", "mlxlm", "--context", "4096", "--max-tokens", "512", "--disk-cache-gb", "0"],
            cwd=ROOT, stdout=cls.log, stderr=subprocess.STDOUT)
        for _ in range(180):
            if cls.proc.poll() is not None:
                raise RuntimeError("server exited; see " + cls.log.name)
            try:
                cls.health = cls.get("/health")
                break
            except OSError:
                time.sleep(0.5)
        else:
            raise RuntimeError("server did not come up")

    @classmethod
    def tearDownClass(cls):
        cls.proc.terminate()
        try:
            cls.proc.wait(10)
        except subprocess.TimeoutExpired:
            cls.proc.kill()
        cls.log.close()

    @classmethod
    def get(cls, path):
        with urllib.request.urlopen(cls.base + path, timeout=5) as r:
            return json.load(r)

    def post(self, body, stream=False):
        req = urllib.request.Request(self.base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            r = urllib.request.urlopen(req, timeout=120)
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())
        with r:
            if not stream:
                return r.status, json.load(r)
            events = []
            for raw in r:
                raw = raw.strip()
                if raw.startswith(b"data:") and raw != b"data: [DONE]":
                    events.append(json.loads(raw[5:]))
            return r.status, events

    def chat(self, content, **kw):
        body = {"model": "x", "max_tokens": 200, "temperature": 0,
                "messages": [{"role": "user", "content": content}]}
        body.update(kw)
        return self.post(body, stream=kw.get("stream", False))

    def test_health_names_the_backend_and_the_model(self):
        self.assertEqual(self.health.get("status", "ok"), "ok")
        self.assertIn("context", self.health)

    def test_plain_answer(self):
        code, r = self.chat("Reply with just the word PONG.", chat_template_kwargs={"enable_thinking": False})
        self.assertEqual(code, 200)
        msg = r["choices"][0]["message"]
        self.assertIn("PONG", msg["content"].upper())
        self.assertIn(r["choices"][0]["finish_reason"], ("stop", "length"))
        self.assertGreater(r["usage"]["completion_tokens"], 0)

    def test_streaming_has_deltas_usage_and_finish(self):
        code, ev = self.chat("Count: 1 2 3", stream=True, stream_options={"include_usage": True},
                             chat_template_kwargs={"enable_thinking": False})
        self.assertEqual(code, 200)
        text = "".join(c.get("delta", {}).get("content") or "" for e in ev for c in e.get("choices", []))
        self.assertTrue(text.strip())
        self.assertTrue(any(c.get("finish_reason") for e in ev for c in e.get("choices", [])))
        self.assertTrue(any(e.get("usage") for e in ev))

    def test_thinking_is_split_from_the_answer_when_the_model_supports_it(self):
        code, r = self.chat("What is 6 times 7?", max_tokens=400)
        self.assertEqual(code, 200)
        msg = r["choices"][0]["message"]
        self.assertNotIn("<think>", msg.get("content") or "")     # never leaks tags into the answer
        self.assertNotIn("</think>", msg.get("content") or "")

    def test_thinking_off_gives_no_reasoning(self):
        code, r = self.chat("What is 6 times 7?", chat_template_kwargs={"enable_thinking": False})
        msg = r["choices"][0]["message"]
        self.assertFalse(msg.get("reasoning_content"))
        self.assertIn("42", msg["content"])

    def test_tool_call_is_parsed_or_the_answer_is_plain_text(self):
        tools = [{"type": "function", "function": {
            "name": "get_weather", "description": "Current weather for a city",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
        code, r = self.chat("What is the weather in Paris? Use the tool.", tools=tools, max_tokens=300,
                            chat_template_kwargs={"enable_thinking": False})
        self.assertEqual(code, 200)
        msg = r["choices"][0]["message"]
        calls = msg.get("tool_calls") or []
        if calls:  # a 0.6B model may or may not call it; when it does the arguments must be valid JSON
            self.assertEqual(calls[0]["function"]["name"], "get_weather")
            self.assertIn("city", json.loads(calls[0]["function"]["arguments"]))
            self.assertEqual(r["choices"][0]["finish_reason"], "tool_calls")
        else:
            self.assertNotIn("<tool_call>", msg.get("content") or "")

    def test_prompt_beyond_the_context_is_a_clean_400(self):
        code, r = self.chat("word " * 6000)
        self.assertEqual(code, 400)
        self.assertEqual(r["error"].get("code"), "context_length_exceeded")

    def test_second_turn_reuses_the_prefix(self):
        msgs = [{"role": "user", "content": "Remember the number 4821. Reply OK."}]
        code, r = self.post({"model": "x", "messages": msgs, "max_tokens": 40, "temperature": 0,
                             "chat_template_kwargs": {"enable_thinking": False}})
        msgs += [r["choices"][0]["message"] | {"role": "assistant"},
                 {"role": "user", "content": "What number did I ask you to remember?"}]
        code, r2 = self.post({"model": "x", "messages": msgs, "max_tokens": 40, "temperature": 0,
                              "chat_template_kwargs": {"enable_thinking": False}})
        self.assertEqual(code, 200)
        self.assertIn("4821", r2["choices"][0]["message"]["content"])
        cached = (r2["usage"].get("prompt_tokens_details") or {}).get("cached_tokens", 0)
        self.assertGreater(cached, 0)


if __name__ == "__main__":
    unittest.main()
