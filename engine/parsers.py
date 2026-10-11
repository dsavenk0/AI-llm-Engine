"""How model output is split into reasoning, content and tool calls — chosen per model family, with a manual
override (stdlib only).

Automatic (the default) is what the engine always did: the Qwen3.5 `<think>…</think>` and `<tool_call>` XML
for the native engine, and the markers and tool parser of the model's own tokenizer on the mlx-lm backend.
An override replaces one piece without touching the other:

  reasoning:  auto | think | none | tags:<closing marker>      e.g. tags:</reasoning>
  tool calls: auto | qwen_xml | json | mlx-lm | none

`json` is the Hermes/Qwen2.5 style  <tool_call>{"name": …, "arguments": {…}}</tool_call>.
"""
import json
import re
import uuid

REASONING_CHOICES = ("auto", "think", "none")
TOOL_CHOICES = ("auto", "qwen_xml", "json", "mlx-lm", "none")
TAGS_RE = re.compile(r"^tags:(\S{1,64})$")

FUNC_RE = re.compile(r"<function=([^>\n]+)>(.*?)(?:</function>|$)", re.S)
PARAM_RE = re.compile(r"<parameter=([^>\n]+)>\n?(.*?)\n?(?:</parameter>|(?=<parameter=)|$)", re.S)


def _call(name, args):
    return {"id": "call_" + uuid.uuid4().hex[:24], "type": "function",
            "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}}


def parse_xml(text, tools):
    """Qwen3.5's <function=name><parameter=p>v</parameter></function>, typed by the tool's schema."""
    schemas = {}
    for t in tools if isinstance(tools, list) else []:
        f = t.get("function", t) if isinstance(t, dict) else None
        if not isinstance(f, dict):
            continue
        params = f.get("parameters")
        props = params.get("properties") if isinstance(params, dict) else None
        schemas[f.get("name")] = props if isinstance(props, dict) else {}
    calls = []
    for name, body in FUNC_RE.findall(text or ""):
        name = name.strip()
        props = schemas.get(name, {})
        args = {}
        for p, raw in PARAM_RE.findall(body):
            p = p.strip()
            spec = props.get(p)
            typ = spec.get("type") if isinstance(spec, dict) else None
            val = raw
            if typ and typ != "string":
                try:
                    val = json.loads(raw)
                except ValueError:
                    val = raw
            elif not typ and raw.strip()[:1] in "[{":
                try:
                    val = json.loads(raw)
                except ValueError:
                    pass
            args[p] = val
        calls.append(_call(name, args))
    return calls


def parse_json(text, tools):
    """<tool_call>{"name": "f", "arguments": {...}}</tool_call>, one or several (arguments may be a string)."""
    calls = []
    for chunk in re.split(r"<tool_call>", text or ""):
        chunk = chunk.split("</tool_call>")[0].strip()
        if not chunk:
            continue
        try:
            obj = json.loads(chunk)
        except ValueError:
            continue                               # cut off mid-generation: nothing to call
        for o in obj if isinstance(obj, list) else [obj]:
            if not isinstance(o, dict) or not isinstance(o.get("name"), str):
                continue
            args = o.get("arguments", o.get("parameters", {}))
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except ValueError:
                    args = {"input": args}
            calls.append(_call(o["name"], args if isinstance(args, dict) else {"input": args}))
    return calls


def validate(reasoning, tool):
    """Raises ValueError for a value that is not one of the choices (checked at start, not mid-request)."""
    if reasoning not in REASONING_CHOICES and not TAGS_RE.match(reasoning or ""):
        raise ValueError(f"--reasoning-parser: {reasoning!r} is not auto, think, none or tags:<closing marker>")
    if tool not in TOOL_CHOICES:
        raise ValueError(f"--tool-parser: {tool!r} is not one of {', '.join(TOOL_CHOICES)}")


def pick(reasoning, tool, auto_markers, auto_parse):
    """(markers for OutputParser, parse function(text, tools) -> [tool_calls]).

    `auto_markers` / `auto_parse` are what the backend would use on its own. An empty string in the markers
    means "this model has none": the output parser then never enters reasoning mode / holds back tool text."""
    markers = dict(auto_markers or {})
    parse = auto_parse
    m = TAGS_RE.match(reasoning or "")
    if m:
        markers["think_end"] = m.group(1)
    elif reasoning == "think":
        markers["think_end"] = "</think>"
    elif reasoning == "none":
        markers["think_end"] = ""
    if tool == "qwen_xml":
        markers.update(tool_start="<tool_call>", tool_end="</tool_call>")
        parse = parse_xml
    elif tool == "json":
        markers.update(tool_start="<tool_call>", tool_end="</tool_call>")
        parse = parse_json
    elif tool == "none":
        markers.update(tool_start="", tool_end="")
        parse = lambda text, tools: []
    return markers, parse
