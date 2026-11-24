"""Inkling 2's reply format, turned into the one the server already parses (stdlib only).

The model answers in segments, each `<|message_model|>[name]<|content_KIND|>…<|end_message|>`, and ends with
`<|content_model_end_sampling|>` (its EOS):

    KIND = thinking               the reasoning
           text, tool_error,      the answer
           invoke_tool_text
           invoke_tool_json       a tool call: {"name": "f", "args": {...}}

One closing marker ends every kind, so the server's OutputParser (which knows `…</think>` reasoning and
`<tool_call>…</tool_call>`) cannot read it as it is. `StreamFilter` rewrites the stream as it comes: a thinking
segment ends in `</think>`, a tool call is wrapped in `<tool_call>…</tool_call>`, the other markers go. The
parser always starts in reasoning mode (`opens_reasoning`), as for a model whose prompt opens `<think>`: told
`reasoning_effort: "none"`, the template asks for effort 0, and the model may still write a short thinking
segment — reasoning, not the answer, and not to be lost. A reply that skips its reasoning closes it at once.
"""
import json
import re

from engine import parsers

ANSWER = ("<|content_text|>", "<|content_tool_error|>", "<|content_invoke_tool_text|>")
THINK, TOOL, END = "<|content_thinking|>", "<|content_invoke_tool_json|>", "<|end_message|>"
DROP = ("<|content_model_end_sampling|>", "<|endoftext|>")
HEAD = "<|message_model|>"                                   # a message header, then an optional name


class StreamFilter:
    opens_reasoning = True                      # the server starts its parser in reasoning mode

    def __init__(self, thinking_open=True):
        self.reasoning = True                   # the parser is in reasoning mode until it sees </think>
        self.kind = None                        # the segment being written: "think", "text", "tool"
        self.buf = ""

    def _close_reasoning(self):
        if self.reasoning:
            self.reasoning = False
            return "</think>"
        return ""

    def feed(self, text):
        self.buf += text
        out = []
        while self.buf:
            i = self.buf.find("<|")
            if i < 0:
                keep = 1 if self.buf.endswith("<") else 0          # perhaps the start of a marker
                out.append(self._body(self.buf[:len(self.buf) - keep]))
                self.buf = self.buf[len(self.buf) - keep:]
                break
            if i:
                out.append(self._body(self.buf[:i]))
                self.buf = self.buf[i:]
            if self.buf.startswith(HEAD):
                nxt = self.buf.find("<", len(HEAD))                  # the header's name runs to the next marker
                if nxt < 0:
                    break
                self.buf = self.buf[nxt:]
                continue
            j = self.buf.find("|>")
            if j < 0:
                break                                               # a marker cut in two: wait for the rest
            marker, self.buf = self.buf[:j + 2], self.buf[j + 2:]
            out.append(self._marker(marker))
        return "".join(out)

    def _body(self, s):
        if self.kind is None and s.strip():
            return self._close_reasoning() + s          # text outside any segment: take it as the answer
        # a second thinking segment after the answer began has nowhere to go: it is not the answer
        return "" if self.kind == "think" and not self.reasoning else s

    def _marker(self, m):
        if m == THINK:
            self.kind = "think"
            return ""
        if m in ANSWER:
            self.kind = "text"
            return self._close_reasoning()
        if m == TOOL:
            self.kind = "tool"
            return self._close_reasoning() + "<tool_call>"
        if m == END:
            kind, self.kind = self.kind, None
            if kind == "think":
                return self._close_reasoning()
            if kind == "tool":
                return "</tool_call>"
            return ""
        if m in DROP:
            return ""
        return self._body(m)                                    # any other text that looks like a marker stays

    def finish(self):
        rest, self.buf = self.buf, ""
        return rest


def parse_tool_calls(text, tools):
    """Inkling's {"name": …, "args": {…}} (or the usual "arguments") between <tool_call> markers."""
    fixed = []
    for chunk in re.split(r"<tool_call>", text or ""):
        body = chunk.split("</tool_call>")[0].strip()
        if not body:
            continue
        try:
            obj = json.loads(body)
        except ValueError:
            continue                                      # cut off mid-generation
        for o in obj if isinstance(obj, list) else [obj]:
            if isinstance(o, dict) and "args" in o and "arguments" not in o:
                o = dict(o, arguments=o.pop("args"))
            fixed.append(o)
    return parsers.parse_json("".join(f"<tool_call>{json.dumps(o)}</tool_call>" for o in fixed), tools)


EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh", "max")


def effort_for(thinking, effort):
    """The template's reasoning_effort: "none" turns reasoning off; the engine's levels map onto its own."""
    if not thinking:
        return "none"
    return effort if effort in EFFORTS and effort != "none" else "high"
