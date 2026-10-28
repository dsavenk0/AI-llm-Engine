"""The Anthropic Messages API adapter (engine/anthropic.py) — pure translation, no model needed.

    .venv/bin/python -m unittest tests.test_anthropic -v
"""
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from engine import anthropic as A  # noqa: E402


class TestRequest(unittest.TestCase):
    def test_system_and_sampling_and_max_tokens(self):
        out = A.to_internal({"model": "m", "max_tokens": 64, "system": "be brief",
                             "messages": [{"role": "user", "content": "hi"}],
                             "temperature": 0.3, "top_p": 0.9, "top_k": 20, "stop_sequences": ["X"]})
        self.assertEqual(out["messages"][0], {"role": "system", "content": "be brief"})
        self.assertEqual(out["max_tokens"], 64)
        self.assertEqual((out["temperature"], out["top_p"], out["top_k"]), (0.3, 0.9, 20))
        self.assertEqual(out["stop"], ["X"])
        self.assertFalse(out["stream"])          # Anthropic defaults to non-streaming, like OpenAI

    def test_system_blocks(self):
        out = A.to_internal({"system": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}],
                             "messages": [{"role": "user", "content": "x"}]})
        self.assertEqual(out["messages"][0]["content"], "ab")

    def test_image_block(self):
        out = A.to_internal({"messages": [{"role": "user", "content": [
            {"type": "text", "text": "what is this"},
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "QUJD"}}]}]})
        parts = out["messages"][0]["content"]
        self.assertEqual(parts[0], {"type": "text", "text": "what is this"})
        self.assertEqual(parts[1]["image_url"]["url"], "data:image/jpeg;base64,QUJD")

    def test_tools_map_input_schema_to_parameters(self):
        out = A.to_internal({"messages": [], "tools": [
            {"name": "get_weather", "description": "d", "input_schema": {"type": "object", "properties": {}}}]})
        self.assertEqual(out["tools"][0]["function"]["name"], "get_weather")
        self.assertEqual(out["tools"][0]["function"]["parameters"], {"type": "object", "properties": {}})

    def test_tool_choice_none_and_thinking(self):
        out = A.to_internal({"messages": [], "tool_choice": {"type": "none"}, "thinking": {"type": "enabled"}})
        self.assertEqual(out["tool_choice"], "none")
        self.assertTrue(out["enable_thinking"])

    def test_tool_use_and_tool_result_round_trip(self):
        # an assistant tool_use turn, then a user tool_result turn → OpenAI assistant+tool messages
        out = A.to_internal({"messages": [
            {"role": "assistant", "content": [{"type": "tool_use", "id": "tu1", "name": "f", "input": {"a": 1}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "tu1", "content": "42"}]}]})
        self.assertEqual(out["messages"][0]["tool_calls"][0]["id"], "tu1")
        self.assertEqual(out["messages"][0]["tool_calls"][0]["function"]["arguments"], {"a": 1})
        self.assertEqual(out["messages"][1], {"role": "tool", "tool_call_id": "tu1", "content": "42"})


class TestResponse(unittest.TestCase):
    def test_message_blocks_and_usage(self):
        r = A.message("m", "hello", "because", [{"id": "t1", "function": {"name": "f", "arguments": "{\"a\": 1}"}}],
                      "stop", {"prompt_tokens": 5, "completion_tokens": 3}, "msg_x")
        self.assertEqual(r["type"], "message")
        self.assertEqual([b["type"] for b in r["content"]], ["thinking", "text", "tool_use"])
        self.assertEqual(r["content"][2]["input"], {"a": 1})
        self.assertEqual(r["stop_reason"], "tool_use")       # tool calls win
        self.assertEqual(r["usage"], {"input_tokens": 5, "output_tokens": 3})

    def test_stop_reason_mapping(self):
        self.assertEqual(A.stop_reason("stop", False), "end_turn")
        self.assertEqual(A.stop_reason("length", False), "max_tokens")
        self.assertEqual(A.stop_reason("length", True), "tool_use")

    def test_text_only_has_no_thinking_block(self):
        r = A.message("m", "hi", "", [], "stop", {"prompt_tokens": 1, "completion_tokens": 1}, "msg_y")
        self.assertEqual([b["type"] for b in r["content"]], ["text"])

    def test_stream_event_shapes(self):
        self.assertEqual(A.message_start("m", "msg_1")["message"]["usage"]["output_tokens"], 0)
        self.assertEqual(A.block_start(0, "thinking")["content_block"]["type"], "thinking")
        self.assertEqual(A.block_delta(1, "text", "hi")["delta"], {"type": "text_delta", "text": "hi"})
        start, inp = A.tool_block_start(2, {"id": "t", "function": {"name": "f", "arguments": {"a": 1}}})
        self.assertEqual(start["content_block"]["name"], "f")
        self.assertEqual(inp, {"a": 1})
        self.assertIn("partial_json", A.input_json_delta(2, {"a": 1})["delta"])
        self.assertEqual(A.message_delta("length", False, 7)["delta"]["stop_reason"], "max_tokens")
        self.assertEqual(A.message_delta("length", False, 7)["usage"]["output_tokens"], 7)
        self.assertEqual(A.message_stop()["type"], "message_stop")

    def test_error_shape(self):
        self.assertEqual(A.error("boom")["error"]["type"], "invalid_request_error")


if __name__ == "__main__":
    unittest.main()
