# Ollama wire-format adapter (pure translation, no I/O).
#
# Lets an Ollama client (`ollama`-speaking IDE plugins, libraries, the Ollama CLI pointed at us) talk to the
# same loaded model as the OpenAI route, with no separate adapter process. The server (engine/server.py) owns
# the one generation path; this module only converts request/response shapes:
#
#   /api/tags, /api/ps, /api/show, /api/version  — discovery
#   /api/chat                                     — chat (messages); reuses the OpenAI chat core
#   /api/generate                                 — raw prompt → single-turn chat
#
# Everything here is a pure function of plain dicts so it is unit-testable without a model or a socket.
import hashlib
import json
import time

# Ollama clients gate features (tools, thinking) on the server's reported version. We report a recent-enough
# Ollama version so those clients enable them; this is a compatibility shim, not our own version number.
COMPAT_VERSION = "0.5.1"

# Ollama option name -> OpenAI body key. Options Ollama has no OpenAI twin for (num_ctx, mirostat, ...) are
# dropped rather than guessed at.
_OPTIONS = {
    "temperature": "temperature", "top_p": "top_p", "top_k": "top_k", "min_p": "min_p",
    "seed": "seed", "presence_penalty": "presence_penalty", "frequency_penalty": "frequency_penalty",
    "repeat_penalty": "repetition_penalty", "num_predict": "max_tokens", "stop": "stop",
}


def _apply_options(out, options):
    for ok, oak in _OPTIONS.items():
        v = options.get(ok)
        if v is None:
            continue
        if ok == "num_predict" and isinstance(v, int) and v < 0:
            continue                                  # -1 / -2 mean "no limit" in Ollama → leave unset
        out[oak] = v


def _image_url(b):
    """A base64 image (Ollama messages carry raw base64) as an OpenAI data: URL."""
    s = str(b)
    return s if s.startswith("data:") else f"data:image/png;base64,{s}"


def _content(text, images):
    """A message body → an OpenAI content string, or a text+image parts list when images are attached."""
    if not images:
        return text or ""
    parts = []
    if text:
        parts.append({"type": "text", "text": text})
    for b in images:
        parts.append({"type": "image_url", "image_url": {"url": _image_url(b)}})
    return parts


def _message(m):
    """One Ollama chat message → an OpenAI one. tool_calls pass through — engine/server.normalize_messages
    accepts Ollama's object-valued arguments as it does OpenAI's string ones."""
    out = {"role": m.get("role", "user")}
    out["content"] = _content(m.get("content", ""), m.get("images"))
    if m.get("tool_calls"):
        out["tool_calls"] = m["tool_calls"]
    if m.get("tool_name"):                            # a tool result's name
        out["name"] = m["tool_name"]
    return out


def to_openai_chat(body):
    """An Ollama POST /api/chat body → the OpenAI chat body the server's chat core understands."""
    out = {}
    if body.get("model") is not None:
        out["model"] = body["model"]
    out["messages"] = [_message(m) for m in (body.get("messages") or [])]
    out["stream"] = bool(body.get("stream", True))    # Ollama streams by default; OpenAI does not
    _apply_options(out, body.get("options") or {})
    if body.get("tools"):
        out["tools"] = body["tools"]
    if body.get("think") is not None:
        out["enable_thinking"] = bool(body["think"])
    return out


def to_openai_generate(body):
    """An Ollama POST /api/generate body (raw prompt) → an OpenAI chat body (one system + one user turn).
    raw/template/context/suffix are not reproduced — the prompt is sent as a plain user message."""
    msgs = []
    if body.get("system"):
        msgs.append({"role": "system", "content": body["system"]})
    msgs.append({"role": "user", "content": _content(body.get("prompt", ""), body.get("images"))})
    out = {"messages": msgs, "stream": bool(body.get("stream", True))}
    if body.get("model") is not None:
        out["model"] = body["model"]
    _apply_options(out, body.get("options") or {})
    if body.get("think") is not None:
        out["enable_thinking"] = bool(body["think"])
    return out


# ───────────────────────────── responses ─────────────────────────────
def now():
    return time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime())


def _done_reason(finish):
    return "length" if finish == "length" else "stop"


def _counts(usage, timings):
    """The Ollama token/duration counters, from our OpenAI usage + stats. Durations are nanoseconds."""
    out = {}
    if usage:
        out["prompt_eval_count"] = usage.get("prompt_tokens", 0)
        out["eval_count"] = usage.get("completion_tokens", 0)
    if timings:
        ns = lambda s: int(float(s) * 1e9)
        pre, dec = timings.get("prefill_s"), timings.get("decode_s")
        if pre is not None:
            out["prompt_eval_duration"] = ns(pre)
        if dec is not None:
            out["eval_duration"] = ns(dec)
        if pre is not None and dec is not None:
            out["total_duration"] = ns(pre + dec)
    return out


def _tool_calls(calls):
    """OpenAI tool_calls → Ollama shape: {"function": {"name", "arguments": <object>}}."""
    out = []
    for c in calls or []:
        fn = c.get("function", c)
        args = fn.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args or "{}")
            except ValueError:
                args = {"_raw": args}
        out.append({"function": {"name": fn.get("name", ""), "arguments": args or {}}})
    return out


def chat_delta(model, content=None, thinking=None):
    """A streaming /api/chat line (NDJSON): one assistant delta, not yet done."""
    msg = {"role": "assistant", "content": content or ""}
    if thinking:
        msg["thinking"] = thinking
    return {"model": model, "created_at": now(), "message": msg, "done": False}


def chat_tool_delta(model, calls):
    """A streaming /api/chat line carrying tool calls (Ollama puts them on a message, not a delta field)."""
    return {"model": model, "created_at": now(), "done": False,
            "message": {"role": "assistant", "content": "", "tool_calls": _tool_calls(calls)}}


def chat_final(model, finish, usage=None, timings=None):
    """The terminal streaming /api/chat line: empty message, done, with reason and counters."""
    out = {"model": model, "created_at": now(), "message": {"role": "assistant", "content": ""},
           "done": True, "done_reason": _done_reason(finish)}
    out.update(_counts(usage, timings))
    return out


def chat_response(model, content, reasoning, calls, finish, usage=None, timings=None):
    """A non-streaming /api/chat response (one object)."""
    msg = {"role": "assistant", "content": content or ""}
    if reasoning:
        msg["thinking"] = reasoning
    if calls:
        msg["tool_calls"] = _tool_calls(calls)
    out = {"model": model, "created_at": now(), "message": msg,
           "done": True, "done_reason": "stop" if calls else _done_reason(finish)}
    out.update(_counts(usage, timings))
    return out


def generate_delta(model, response):
    """A streaming /api/generate line: a chunk of the raw response."""
    return {"model": model, "created_at": now(), "response": response, "done": False}


def generate_final(model, finish, usage=None, timings=None):
    out = {"model": model, "created_at": now(), "response": "", "done": True,
           "done_reason": _done_reason(finish), "context": []}
    out.update(_counts(usage, timings))
    return out


def generate_response(model, content, finish, usage=None, timings=None):
    out = {"model": model, "created_at": now(), "response": content or "", "done": True,
           "done_reason": _done_reason(finish), "context": []}
    out.update(_counts(usage, timings))
    return out


def error(message):
    """Ollama's error shape — a bare {"error": "..."} object."""
    return {"error": str(message)}


# ───────────────────────────── discovery ─────────────────────────────
def _digest(name):
    return "sha256:" + hashlib.sha256(str(name).encode()).hexdigest()


def model_card(name, size_bytes=0, family=None, params=None, quant=None, fmt="mlx", modified_at=None):
    """One entry for /api/tags and the base of /api/show. Fields clients read for display/selection; the
    details we cannot derive cheaply are left out rather than faked."""
    details = {"format": fmt, "parent_model": ""}
    if family:
        details["family"] = family
        details["families"] = [family]
    if params:
        details["parameter_size"] = params
    if quant:
        details["quantization_level"] = quant
    return {"name": name, "model": name, "modified_at": modified_at or now(),
            "size": int(size_bytes or 0), "digest": _digest(name), "details": details}


def tags(cards):
    return {"models": list(cards)}


def ps(cards):
    """/api/ps — loaded models. Same cards plus an expiry (we don't expire, so far in the future)."""
    out = []
    for c in cards:
        e = dict(c)
        e["expires_at"] = "9999-12-31T23:59:59Z"
        e["size_vram"] = c.get("size", 0)
        out.append(e)
    return {"models": out}


def show(card, context=None, capabilities=None):
    """/api/show — details for one model. We fill what we know; template/modelfile are left empty (the chat
    template lives inside the model, not in a Modelfile here)."""
    out = {"details": card.get("details", {}), "model_info": {}, "capabilities": list(capabilities or ["completion"]),
           "modelfile": "", "parameters": "", "template": ""}
    if context:
        out["model_info"]["general.context_length"] = context
    return out


def version():
    return {"version": COMPAT_VERSION}
