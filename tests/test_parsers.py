"""engine/parsers.py: the output formats, and the manual override of the automatic choice.

    .venv/bin/python -m unittest tests.test_parsers -v
"""
import json
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from engine import parsers  # noqa: E402
from engine.server import OutputParser  # noqa: E402

TOOLS = [{"type": "function", "function": {"name": "get_weather", "parameters": {
    "type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"}}}}}]
AUTO = {"think_end": "</think>", "tool_start": "<tool_call>", "tool_end": "</tool_call>"}


def run(chunks, markers, thinking=True, tools=True):
    p, out = OutputParser(thinking, tools, **markers), []
    for c in chunks:
        out += p.feed(c)
    out += p.finish()
    return ("".join(s for k, s in out if k == "reasoning"), "".join(s for k, s in out if k == "content"), p.tool_text)


class TestJsonToolCalls(unittest.TestCase):
    def test_one_call(self):
        c = parsers.parse_json('<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris", "days": 3}}\n</tool_call>', TOOLS)
        self.assertEqual(len(c), 1)
        self.assertEqual(c[0]["function"]["name"], "get_weather")
        self.assertEqual(json.loads(c[0]["function"]["arguments"]), {"city": "Paris", "days": 3})

    def test_several_calls_and_string_arguments(self):
        text = ('<tool_call>{"name": "a", "arguments": "{\\"x\\": 1}"}</tool_call>'
                '<tool_call>{"name": "b", "parameters": {"y": 2}}</tool_call>')
        c = parsers.parse_json(text, [])
        self.assertEqual([x["function"]["name"] for x in c], ["a", "b"])
        self.assertEqual(json.loads(c[0]["function"]["arguments"]), {"x": 1})
        self.assertEqual(json.loads(c[1]["function"]["arguments"]), {"y": 2})

    def test_a_cut_off_call_is_not_a_call(self):
        self.assertEqual(parsers.parse_json('<tool_call>{"name": "get_wea', TOOLS), [])
        self.assertEqual(parsers.parse_json("", TOOLS), [])

    def test_the_xml_format_is_untouched(self):
        text = "<tool_call><function=get_weather><parameter=city>Paris</parameter><parameter=days>2</parameter></function></tool_call>"
        c = parsers.parse_xml(text, TOOLS)
        self.assertEqual(json.loads(c[0]["function"]["arguments"]), {"city": "Paris", "days": 2})


class TestPick(unittest.TestCase):
    def test_auto_changes_nothing(self):
        markers, parse = parsers.pick("auto", "auto", dict(AUTO), parsers.parse_xml)
        self.assertEqual(markers, AUTO)
        self.assertIs(parse, parsers.parse_xml)

    def test_a_reasoning_override_leaves_tool_calls_alone(self):
        markers, parse = parsers.pick("tags:</reasoning>", "auto", dict(AUTO), parsers.parse_xml)
        self.assertEqual((markers["think_end"], markers["tool_start"]), ("</reasoning>", "<tool_call>"))
        self.assertIs(parse, parsers.parse_xml)

    def test_a_tool_override_leaves_reasoning_alone(self):
        markers, parse = parsers.pick("auto", "json", {"think_end": "</think>"}, parsers.parse_xml)
        self.assertEqual(markers["think_end"], "</think>")
        self.assertIs(parse, parsers.parse_json)

    def test_validation_names_the_bad_value(self):
        parsers.validate("auto", "auto")
        parsers.validate("tags:</r>", "json")
        for r, t in (("bogus", "auto"), ("tags:", "auto"), ("auto", "yaml")):
            with self.assertRaises(ValueError):
                parsers.validate(r, t)


class TestEndToEndWithTheOutputParser(unittest.TestCase):
    def test_a_custom_closing_marker_splits_reasoning_from_the_answer(self):
        markers, _ = parsers.pick("tags:</reasoning>", "auto", dict(AUTO), parsers.parse_xml)
        r, c, tool = run(["I should add ", "2+2</reaso", "ning>The answer is 4."], markers)
        self.assertEqual((r, c, tool), ("I should add 2+2", "The answer is 4.", None))

    def test_none_keeps_everything_as_the_answer(self):
        markers, _ = parsers.pick("none", "auto", dict(AUTO), parsers.parse_xml)
        r, c, _ = run(["<think>x</think>done"], markers)
        self.assertEqual((r, c), ("", "<think>x</think>done"))

    def test_tool_none_does_not_hold_back_tool_text(self):
        markers, parse = parsers.pick("auto", "none", dict(AUTO), parsers.parse_xml)
        r, c, tool = run(["hello <tool_call>{}</tool_call>"], markers, thinking=False)
        self.assertEqual((c, tool), ("hello <tool_call>{}</tool_call>", None))
        self.assertEqual(parse("anything", TOOLS), [])

    def test_json_tool_calls_are_held_back_and_parsed(self):
        markers, parse = parsers.pick("auto", "json", dict(AUTO), parsers.parse_xml)
        text = 'Let me check. <tool_call>{"name": "get_weather", "arguments": {"city": "Oslo"}}</tool_call>'
        r, c, tool = run([text[:20], text[20:]], markers, thinking=False)
        self.assertEqual(c.strip(), "Let me check.")
        self.assertEqual(json.loads(parse(tool, TOOLS)[0]["function"]["arguments"]), {"city": "Oslo"})


    def test_odd_tool_definitions_do_not_break_the_xml_parser(self):
        text = "<tool_call><function=f><parameter=a>1</parameter><parameter=b>x</parameter></function></tool_call>"
        for tools in (None, "x", [5, None, "f"], [{"function": "str"}], [{"function": {"name": "f", "parameters": 3}}],
                      [{"function": {"name": "f", "parameters": {"properties": {"a": True, "b": "s"}}}}]):
            calls = parsers.parse_xml(text, tools)
            self.assertEqual(len(calls), 1, tools)
            self.assertEqual(json.loads(calls[0]["function"]["arguments"])["b"], "x")
        typed = [{"function": {"name": "f", "parameters": {"properties": {"a": {"type": "integer"}}}}}]
        self.assertEqual(json.loads(parsers.parse_xml(text, typed)[0]["function"]["arguments"])["a"], 1)


    def test_call_ids_arguments_and_untyped_values(self):
        import re
        text = ("<tool_call><function=f><parameter=s>привет</parameter><parameter=n>7</parameter>"
                "<parameter=l>[1, 2]</parameter><parameter=o>{bad</parameter><parameter=e></parameter></function>"
                "</tool_call>")
        tools = [{"function": {"name": "f", "parameters": {"properties": {"s": {"type": "string"},
                                                                          "n": {"type": "integer"}}}}}]
        call = parsers.parse_xml(text, tools)[0]
        self.assertRegex(call["id"], r"^call_[0-9a-f]{24}$")
        self.assertEqual(call["type"], "function")
        raw = call["function"]["arguments"]
        self.assertIn("привет", raw)                                   # not \u-escaped
        args = json.loads(raw)
        self.assertEqual(args, {"s": "привет", "n": 7, "l": [1, 2], "o": "{bad", "e": ""})
        # a "string" typed value that looks like JSON stays text
        text2 = "<tool_call><function=f><parameter=s>[1]</parameter></function></tool_call>"
        self.assertEqual(json.loads(parsers.parse_xml(text2, tools)[0]["function"]["arguments"]), {"s": "[1]"})

    def test_json_calls_need_a_string_name(self):
        for obj in ({"name": 5, "arguments": {}}, {"arguments": {}}, [1, "x"], {"name": None}):
            text = "<tool_call>" + json.dumps(obj) + "</tool_call>"
            self.assertEqual(parsers.parse_json(text, None), [], obj)
        ok = parsers.parse_json('<tool_call>{"name": "f", "parameters": {"a": 1}}</tool_call>', None)
        self.assertEqual(json.loads(ok[0]["function"]["arguments"]), {"a": 1})
        weird = parsers.parse_json('<tool_call>{"name": "f", "arguments": "not json"}</tool_call>', None)
        self.assertEqual(json.loads(weird[0]["function"]["arguments"]), {"input": "not json"})


if __name__ == "__main__":
    unittest.main()
