"""The Ollama wire-format adapter (engine/ollama.py) — pure request/response translation, no model needed.

    .venv/bin/python -m unittest tests.test_ollama -v
"""
import json
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from engine import ollama  # noqa: E402


class TestRequestToOpenAI(unittest.TestCase):
    def test_chat_defaults_to_streaming(self):
        # Ollama streams unless told not to; OpenAI is the opposite, so the default must be set here.
        self.assertTrue(ollama.to_openai_chat({"messages": []})["stream"])
        self.assertFalse(ollama.to_openai_chat({"messages": [], "stream": False})["stream"])

    def test_options_map_to_openai_keys(self):
        out = ollama.to_openai_chat({"messages": [], "options": {
            "temperature": 0.3, "top_p": 0.9, "top_k": 20, "num_predict": 128, "repeat_penalty": 1.1, "seed": 7,
            "stop": ["X"]}})
        self.assertEqual(out["temperature"], 0.3)
        self.assertEqual(out["top_p"], 0.9)
        self.assertEqual(out["top_k"], 20)
        self.assertEqual(out["max_tokens"], 128)            # num_predict -> max_tokens
        self.assertEqual(out["repetition_penalty"], 1.1)    # repeat_penalty -> repetition_penalty
        self.assertEqual(out["seed"], 7)
        self.assertEqual(out["stop"], ["X"])

    def test_num_predict_unlimited_is_dropped(self):
        # Ollama uses -1/-2 for "no limit"; mapping that to max_tokens=-1 would break the server.
        self.assertNotIn("max_tokens", ollama.to_openai_chat({"messages": [], "options": {"num_predict": -1}}))

    def test_think_toggles_thinking(self):
        self.assertTrue(ollama.to_openai_chat({"messages": [], "think": True})["enable_thinking"])
        self.assertFalse(ollama.to_openai_chat({"messages": [], "think": False})["enable_thinking"])

    def test_images_become_content_parts(self):
        out = ollama.to_openai_chat({"messages": [{"role": "user", "content": "what is this", "images": ["QUJD"]}]})
        parts = out["messages"][0]["content"]
        self.assertEqual(parts[0], {"type": "text", "text": "what is this"})
        self.assertEqual(parts[1]["type"], "image_url")
        self.assertTrue(parts[1]["image_url"]["url"].startswith("data:image/png;base64,QUJD"))

    def test_plain_text_message_stays_a_string(self):
        out = ollama.to_openai_chat({"messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(out["messages"][0]["content"], "hi")

    def test_tool_calls_pass_through(self):
        calls = [{"function": {"name": "f", "arguments": {"a": 1}}}]
        out = ollama.to_openai_chat({"messages": [{"role": "assistant", "content": "", "tool_calls": calls}]})
        self.assertEqual(out["messages"][0]["tool_calls"], calls)

    def test_generate_builds_system_and_user_turns(self):
        out = ollama.to_openai_generate({"system": "be brief", "prompt": "hello", "options": {"temperature": 0}})
        self.assertEqual(out["messages"], [{"role": "system", "content": "be brief"},
                                           {"role": "user", "content": "hello"}])
        self.assertEqual(out["temperature"], 0)


class TestResponses(unittest.TestCase):
    def test_chat_response_shape(self):
        r = ollama.chat_response("m", "hello", "", [], "stop",
                                 {"prompt_tokens": 5, "completion_tokens": 3}, {"prefill_s": 0.1, "decode_s": 0.2})
        self.assertEqual(r["model"], "m")
        self.assertTrue(r["done"])
        self.assertEqual(r["done_reason"], "stop")
        self.assertEqual(r["message"], {"role": "assistant", "content": "hello"})
        self.assertEqual(r["prompt_eval_count"], 5)
        self.assertEqual(r["eval_count"], 3)
        self.assertEqual(r["eval_duration"], 200_000_000)   # 0.2 s in ns
        self.assertEqual(r["total_duration"], 300_000_000)

    def test_chat_response_thinking_and_tools(self):
        r = ollama.chat_response("m", "ans", "because", [{"function": {"name": "f", "arguments": "{\"a\": 1}"}}],
                                 "stop", {"prompt_tokens": 1, "completion_tokens": 1})
        self.assertEqual(r["message"]["thinking"], "because")
        self.assertEqual(r["message"]["tool_calls"], [{"function": {"name": "f", "arguments": {"a": 1}}}])
        self.assertEqual(r["done_reason"], "stop")

    def test_length_finish_maps_to_length(self):
        self.assertEqual(ollama.chat_response("m", "x", "", [], "length")["done_reason"], "length")
        self.assertEqual(ollama.chat_final("m", "length")["done_reason"], "length")

    def test_stream_lines(self):
        d = ollama.chat_delta("m", content="hi")
        self.assertFalse(d["done"])
        self.assertEqual(d["message"], {"role": "assistant", "content": "hi"})
        td = ollama.chat_tool_delta("m", [{"function": {"name": "f", "arguments": "{}"}}])
        self.assertEqual(td["message"]["tool_calls"], [{"function": {"name": "f", "arguments": {}}}])
        self.assertFalse(td["done"])
        f = ollama.chat_final("m", "stop", {"prompt_tokens": 2, "completion_tokens": 4})
        self.assertTrue(f["done"])
        self.assertEqual(f["eval_count"], 4)

    def test_generate_response_uses_response_field(self):
        r = ollama.generate_response("m", "the answer", "stop", {"prompt_tokens": 1, "completion_tokens": 2})
        self.assertEqual(r["response"], "the answer")
        self.assertTrue(r["done"])
        self.assertIn("context", r)
        gd = ollama.generate_delta("m", "chunk")
        self.assertEqual(gd["response"], "chunk")
        self.assertFalse(gd["done"])

    def test_error_shape(self):
        self.assertEqual(ollama.error("boom"), {"error": "boom"})


class TestDiscovery(unittest.TestCase):
    def test_model_card_and_tags(self):
        card = ollama.model_card("my-model", size_bytes=1234, family="qwen3", params="4B", quant="Q4")
        self.assertEqual(card["name"], "my-model")
        self.assertEqual(card["model"], "my-model")
        self.assertEqual(card["size"], 1234)
        self.assertTrue(card["digest"].startswith("sha256:"))
        self.assertEqual(card["details"]["family"], "qwen3")
        self.assertEqual(ollama.tags([card]), {"models": [card]})

    def test_ps_marks_loaded(self):
        card = ollama.model_card("m", size_bytes=10)
        ps = ollama.ps([card])["models"][0]
        self.assertEqual(ps["size_vram"], 10)
        self.assertIn("expires_at", ps)

    def test_show_and_version(self):
        card = ollama.model_card("m")
        show = ollama.show(card, context=4096, capabilities=["completion", "tools"])
        self.assertEqual(show["capabilities"], ["completion", "tools"])
        self.assertEqual(show["model_info"]["general.context_length"], 4096)
        self.assertIn("version", ollama.version())


if __name__ == "__main__":
    unittest.main()
