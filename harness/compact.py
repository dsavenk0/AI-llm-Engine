"""Keep a long agent session inside the model's context window.

Two stages, cheapest first:
  1. shrink  — no model call: drop old reasoning, cut old tool output to a stub.
  2. summarize — the model writes a short note about the middle of the conversation, which
     replaces it; the system prompt and the most recent turns stay verbatim.
Pure functions over the OpenAI-style message list, so they are testable without a server.
"""
import json

TRIGGER = 0.70            # compact once the last prompt used this share of the context
KEEP_USER_TURNS = 2       # recent user turns (with everything after them) kept verbatim
OLD_TOOL_KEEP = 300       # chars of an old tool result that survive stage 1
TRANSCRIPT_MAX = 24000   # chars handed to the summarizer

SUMMARY_PROMPT = (
    "Below is the earlier part of a coding session between a user and an AI agent. Write a compact "
    "note that lets the agent continue without the transcript: the user's goal, what has been "
    "done (files changed and why), decisions and constraints, errors met, and what is still open. "
    "Be specific (paths, names, commands). No preamble.\n\n")


def needs_compaction(prompt_tokens, context, trigger=TRIGGER):
    return bool(prompt_tokens and context and prompt_tokens >= trigger * context)


def _tail_start(messages, keep_user_turns):
    """Index where the verbatim tail begins: the Nth user message from the end (never the system)."""
    seen = 0
    for i in range(len(messages) - 1, 0, -1):
        if messages[i].get("role") == "user":
            seen += 1
            if seen >= keep_user_turns:
                return i
    return 1


def shrink(messages, keep_user_turns=KEEP_USER_TURNS):
    """Stage 1, in place on a copy: returns (new_messages, chars_saved)."""
    cut = _tail_start(messages, keep_user_turns)
    out, saved = [], 0
    for i, m in enumerate(messages):
        if 0 < i < cut:
            m = dict(m)
            if m.pop("reasoning_content", None):
                saved += 1
            c = m.get("content")
            if m.get("role") == "tool" and isinstance(c, str) and len(c) > OLD_TOOL_KEEP:
                saved += len(c) - OLD_TOOL_KEEP
                m["content"] = c[:OLD_TOOL_KEEP] + f"\n… [{len(c) - OLD_TOOL_KEEP} chars of old tool output dropped]"
        out.append(m)
    return out, saved


def transcript(messages):
    """Plain-text rendering of messages for the summarizer, newest last, size-capped."""
    lines = []
    for m in messages:
        role, c = m.get("role"), (m.get("content") or "")
        if role == "assistant" and m.get("tool_calls"):
            calls = []
            for t in m["tool_calls"]:
                f = t.get("function", {})
                try:
                    a = json.loads(f.get("arguments") or "{}")
                except ValueError:
                    a = {}
                brief = ", ".join(f"{k}={str(v)[:80]!r}" for k, v in a.items() if k not in ("content", "new", "old"))
                calls.append(f"{f.get('name')}({brief})")
            c = (c + " " if c else "") + "[tool calls: " + "; ".join(calls) + "]"
        elif role == "tool":
            c = c[:OLD_TOOL_KEEP]
        lines.append(f"{role}: {c.strip()}")
    text = "\n".join(lines)
    return text if len(text) <= TRANSCRIPT_MAX else "…\n" + text[-TRANSCRIPT_MAX:]


def compact(messages, summarize, keep_user_turns=KEEP_USER_TURNS):
    """Stage 2. `summarize(prompt) -> str`. Returns the new message list, or the input unchanged
    when there is nothing old enough to fold."""
    cut = _tail_start(messages, keep_user_turns)
    if cut <= 1:
        return messages
    note = (summarize(SUMMARY_PROMPT + transcript(messages[1:cut])) or "").strip()
    if not note:
        return messages
    folded = {"role": "user", "content": "[Summary of the earlier conversation — the transcript was "
                                         "compacted to save context]\n" + note}
    ack = {"role": "assistant", "content": "Understood. I'll continue from this summary."}
    return [messages[0], folded, ack] + messages[cut:]
