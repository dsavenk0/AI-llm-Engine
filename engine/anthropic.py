# Anthropic Messages API adapter (pure translation, no I/O).
#
# A native /v1/messages — the shape the Anthropic SDK and Claude-speaking clients send — on top of the same
# loaded model and the same generation path as the OpenAI and Ollama routes (engine/server.py owns it; this
# module only converts request/response shapes). Not a wrapper around the OpenAI route: the request is mapped
# to the engine's internal chat body, and the reply is emitted as Anthropic content blocks / SSE events.
import json
import uuid


def msg_id():
    return "msg_" + uuid.uuid4().hex[:24]


def _tool_id():
    return "toolu_" + uuid.uuid4().hex[:24]


# ───────────────────────────── request → internal ─────────────────────────────
def _image(block):
    src = block.get("source") or {}
    if src.get("type") == "base64":
        url = f"data:{src.get('media_type', 'image/png')};base64,{src.get('data', '')}"
    else:
        url = src.get("url", "")
    return {"type": "image_url", "image_url": {"url": url}}


def _collapse(parts):
    """A list of content parts → a plain string when it is all text (keeps prompt-cache prefixes stable)."""
    if all(p.get("type") == "text" for p in parts):
        return "".join(p.get("text", "") for p in parts)
    return parts


def _messages(messages):
    """Anthropic messages → OpenAI messages. One Anthropic message can expand into several: tool_result blocks
    in a user turn become their own OpenAI `tool` messages (OpenAI keeps tool results in separate messages)."""
    out = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content")
        if isinstance(content, str):
            out.append({"role": role, "content": content})
            continue
        parts, tool_calls, tool_results = [], [], []
        for b in content or []:
            t = b.get("type")
            if t == "text":
                parts.append({"type": "text", "text": b.get("text", "")})
            elif t == "image":
                parts.append(_image(b))
            elif t == "tool_use":
                tool_calls.append({"id": b.get("id"), "type": "function",
                                   "function": {"name": b.get("name"), "arguments": b.get("input") or {}}})
            elif t == "tool_result":
                tool_results.append(b)
        if role == "assistant":
            msg = {"role": "assistant", "content": _collapse(parts) if parts else ""}
            if tool_calls:
                msg["tool_calls"] = tool_calls
            out.append(msg)
        else:
            for tr in tool_results:                 # tool results come before the user's own text
                c = tr.get("content")
                txt = c if isinstance(c, str) else "".join(
                    x.get("text", "") for x in (c or []) if isinstance(x, dict) and x.get("type") == "text")
                out.append({"role": "tool", "tool_call_id": tr.get("tool_use_id"), "content": txt})
            if parts:
                out.append({"role": "user", "content": _collapse(parts)})
    return out


def _system(system):
    if isinstance(system, list):
        return "".join(b.get("text", "") for b in system if isinstance(b, dict) and b.get("type") == "text")
    return system or ""


def to_internal(body):
    """An Anthropic POST /v1/messages body → the OpenAI-shaped chat body the server's chat core understands."""
    out = {}
    if body.get("model") is not None:
        out["model"] = body["model"]
    msgs = []
    sys = _system(body.get("system"))
    if sys:
        msgs.append({"role": "system", "content": sys})
    msgs += _messages(body.get("messages") or [])
    out["messages"] = msgs
    out["stream"] = bool(body.get("stream", False))
    for k in ("temperature", "top_p", "top_k"):
        if body.get(k) is not None:
            out[k] = body[k]
    if body.get("max_tokens") is not None:
        out["max_tokens"] = body["max_tokens"]
    if body.get("stop_sequences"):
        out["stop"] = body["stop_sequences"]
    if body.get("tools"):
        out["tools"] = [{"type": "function", "function": {
            "name": t.get("name"), "description": t.get("description", ""),
            "parameters": t.get("input_schema") or {}}} for t in body["tools"]]
    tc = body.get("tool_choice")
    if isinstance(tc, dict) and tc.get("type") == "none":
        out["tool_choice"] = "none"
    th = body.get("thinking")
    if isinstance(th, dict):
        out["enable_thinking"] = th.get("type") == "enabled"
    return out


# ───────────────────────────── reply → Anthropic ─────────────────────────────
def stop_reason(finish, has_calls):
    if has_calls:
        return "tool_use"
    return "max_tokens" if finish == "length" else "end_turn"


def _input(call):
    args = (call.get("function") or call).get("arguments")
    if isinstance(args, str):
        try:
            return json.loads(args or "{}")
        except ValueError:
            return {"_raw": args}
    return args or {}


def _tool_block(call):
    return {"type": "tool_use", "id": call.get("id") or _tool_id(),
            "name": (call.get("function") or call).get("name", ""), "input": _input(call)}


def message(model, content, reasoning, calls, finish, usage, rid):
    """A non-streaming /v1/messages response: thinking, text and tool_use content blocks."""
    blocks = []
    if reasoning:
        blocks.append({"type": "thinking", "thinking": reasoning, "signature": ""})
    if content:
        blocks.append({"type": "text", "text": content})
    for c in calls or []:
        blocks.append(_tool_block(c))
    return {"id": rid, "type": "message", "role": "assistant", "model": model, "content": blocks,
            "stop_reason": stop_reason(finish, bool(calls)), "stop_sequence": None,
            "usage": {"input_tokens": usage.get("prompt_tokens", 0),
                      "output_tokens": usage.get("completion_tokens", 0)}}


def error(message, typ="invalid_request_error"):
    return {"type": "error", "error": {"type": typ, "message": str(message)}}


# ───────────────────────────── streaming events ─────────────────────────────
def message_start(model, rid, input_tokens=0):
    return {"type": "message_start", "message": {
        "id": rid, "type": "message", "role": "assistant", "model": model, "content": [],
        "stop_reason": None, "stop_sequence": None,
        "usage": {"input_tokens": input_tokens, "output_tokens": 0}}}


def block_start(index, kind):
    """kind is "thinking" or "text"."""
    cb = {"type": "thinking", "thinking": ""} if kind == "thinking" else {"type": "text", "text": ""}
    return {"type": "content_block_start", "index": index, "content_block": cb}


def block_delta(index, kind, text):
    delta = ({"type": "thinking_delta", "thinking": text} if kind == "thinking"
             else {"type": "text_delta", "text": text})
    return {"type": "content_block_delta", "index": index, "delta": delta}


def tool_block_start(index, call):
    b = _tool_block(call)
    return {"type": "content_block_start", "index": index,
            "content_block": {"type": "tool_use", "id": b["id"], "name": b["name"], "input": {}}}, b["input"]


def input_json_delta(index, inp):
    return {"type": "content_block_delta", "index": index,
            "delta": {"type": "input_json_delta", "partial_json": json.dumps(inp, ensure_ascii=False)}}


def block_stop(index):
    return {"type": "content_block_stop", "index": index}


def message_delta(finish, has_calls, output_tokens):
    return {"type": "message_delta",
            "delta": {"stop_reason": stop_reason(finish, has_calls), "stop_sequence": None},
            "usage": {"output_tokens": output_tokens}}


def message_stop():
    return {"type": "message_stop"}
