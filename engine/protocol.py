"""What the server needs from an engine — written down, so a backend learns its contract here and not
from an AttributeError in the middle of a request.

`engine/server.py` holds one engine and is written against this protocol only. Three classes implement it
today — `generate.Engine` (the native model), `compat.CompatEngine` (mlx-lm models) and
`gguf_engine.GGUFEngine` (the own GGUF decoder) — and `check()` is run on whichever one starts, so a
missing member stops the server at start with its name rather than surfacing later as a 500.

A member is REQUIRED when the server uses it for every backend, OPTIONAL when a backend may honestly not
have the feature; the server asks `can(engine, name)` before using an optional one and answers the client
plainly when it is absent.

The other contracts between modules are written here too, and `check()` holds an engine to all that apply:

    RUNTIME / RUNTIME_BATCH   what the decode loop (engine/session.py) and the batch loop (engine/batch.py)
                              call on `engine.rt` — the seam a second runtime implements
    LOOP_HOST                 what DecodeLoop reads from the engine that mixes it in
    MEMORY                    the memory manager every engine carries (engine/memory.py `Manager`)
    HEALTH                    the /health keys clients rely on (LLMBar, the browser assistant, the CLI)

VERSION is bumped when a required member or a HEALTH / STATS key is added, removed or changes meaning; it is
served in /health as `engine_protocol`, so a client can tell which contract it is talking to.
"""
from typing import Any, Iterator, Protocol, runtime_checkable

VERSION = 2          # 2: the scheduler owns the turn (no `lock`), engines carry `memory`; runtime/host/health contracts

# attributes: identity and limits · state · prompt defaults
ATTRIBUTES = (
    "backend", "max_context", "max_tokens", "model_bytes", "mem_limit",
    "memory", "sleeping", "last_used", "model", "cache_tokens", "stats", "totals", "history", "store",
    "wired_limit_advice_mb", "progress",
    "thinking", "thinking_default", "supports_thinking", "effort", "efforts", "sampling",
    "vision_allowed", "vision_capable",
)
METHODS = ("build_prompt", "encode", "generate", "sleep", "ensure_awake", "persist_prefix", "memory_breakdown")
REQUIRED = ATTRIBUTES + METHODS
OPTIONAL = (
    "embed",              # /v1/embeddings: the normed hidden state as a vector
    "generate_batch",     # several requests in one forward pass per step (--batch-window)
    "parser_markers",     # the reasoning / tool-call tags of this model's template
    "parse_tool_calls",   # a model-specific tool-call parser; the server's own is the fallback
    "moe_stats",          # expert-pool hit rate and traffic, while a MoE model streams its experts
)

# the keys every generate() ends with; a backend that cannot fill one reports None, never omits it
STATS = ("prompt_tokens", "cached_tokens", "completion_tokens", "prefill_s", "decode_s", "prefill_tps",
         "decode_tps", "finish_reason", "think_forced", "loop_stopped", "peak_gb", "draft_accept", "verifies",
         "ram_used")


@runtime_checkable
class EngineBackend(Protocol):
    """The typed form of REQUIRED, for readers and type checkers."""
    backend: str
    max_context: int
    max_tokens: int
    model_bytes: int
    mem_limit: float
    memory: Any                    # engine/memory.py Manager: the order memory is given back in, and the state
    sleeping: bool
    last_used: float
    model: Any                     # None while asleep
    cache_tokens: list
    stats: dict
    totals: dict
    history: Any
    store: Any                     # the SSD prefix store, or None
    wired_limit_advice_mb: Any
    progress: Any                  # {"phase", "done", "total"} during a long prefill, else None
    thinking: bool
    thinking_default: bool
    supports_thinking: bool
    effort: str
    efforts: list
    sampling: dict
    vision_allowed: bool
    vision_capable: bool

    def build_prompt(self, messages, tools=None, thinking=None, effort=None, extra_kwargs=None) -> tuple: ...
    def encode(self, text, images) -> tuple: ...
    def generate(self, ids, media=None, max_tokens=None, sampling=None, stop=None, should_stop=None,
                 seed=None, thinking_open=False, fmt=None) -> Iterator: ...
    def sleep(self) -> None: ...
    def ensure_awake(self) -> None: ...
    def persist_prefix(self) -> None: ...
    def memory_breakdown(self) -> dict: ...


# the arguments the server passes each method by name; a backend whose method cannot take one is as broken
# as one without the method — it just fails later, on the first request
PARAMETERS = {
    "generate": ("ids", "media", "max_tokens", "sampling", "stop", "should_stop", "seed", "thinking_open", "fmt"),
    "build_prompt": ("messages", "tools", "thinking", "effort", "extra_kwargs"),
    "encode": ("text", "images"),
}


# ── the runtime seam (engine/session.py describes each call) ──
RUNTIME = ("make_cache", "settle", "release", "peak_gb", "commit", "can_verify", "rewind",
           "sampler", "sample", "flush", "read_token",
           "prefill", "step", "verify", "greedy_tokens", "verify_token", "sample_at", "embed_media")
RUNTIME_BATCH = ("batch_prefill", "batch_join", "batch_step", "greedy_row", "sample_row")

# ── what DecodeLoop reads from its engine ──
LOOP_HOST = ("rt", "tok", "eos", "vocab", "max_context", "max_tokens", "mem_limit", "think_frac", "speculative",
             "draft_levels", "ngram_min", "cache", "cache_tokens", "snapshots", "store", "im_start", "progress",
             "stats", "totals", "history",
             "log", "ensure_awake", "_reserve", "_check_memory", "_fit_output", "_prefill", "_restore_or_reset",
             "_detokenizer", "_think_closing")

# ── the memory manager ──
MEMORY = ("state", "reclaim", "relieve", "report")

# ── /health: the keys clients read; more may be added, none of these removed or renamed without VERSION ──
HEALTH = ("status", "model", "model_path", "state", "engine_protocol", "backend", "context", "max_tokens",
          "thinking", "thinking_default", "thinking_supported", "effort", "efforts", "vision", "vision_capable",
          "busy", "inflight", "max_queue", "shutting_down", "progress", "memory_gb", "system_ram_used",
          "mem_limit", "memory_breakdown", "memory_manager", "scheduler", "cached_tokens", "requests", "last")


def missing_members(obj, names, prefix=""):
    return [prefix + n for n in names if not hasattr(obj, n)]


def missing(engine):
    """What `engine` owes the protocol: attributes that are absent, methods that are not callable, and
    parameters a method does not accept (as "method(parameter)")."""
    import inspect
    out = [name for name in ATTRIBUTES if not hasattr(engine, name)]
    for name in METHODS:
        fn = getattr(engine, name, None)
        if not callable(fn):
            out.append(name)
            continue
        try:
            params = inspect.signature(fn).parameters
        except (TypeError, ValueError):
            continue
        if any(p.kind is p.VAR_KEYWORD for p in params.values()):
            continue
        out += [f"{name}({p})" for p in PARAMETERS.get(name, ()) if p not in params]
    mem = getattr(engine, "memory", None)
    if mem is not None:
        out += missing_members(mem, MEMORY, "memory.")
    from .session import DecodeLoop
    if isinstance(engine, DecodeLoop):
        out += missing_members(engine, LOOP_HOST)
        rt = getattr(engine, "rt", None)
        if rt is not None:                                  # None only while asleep
            out += missing_members(rt, RUNTIME, "rt.")
            if callable(getattr(engine, "generate_batch", None)):
                out += missing_members(rt, RUNTIME_BATCH, "rt.")
    return out


def check(engine):
    """Raise TypeError naming everything `engine` owes the protocol; return it unchanged when it is whole."""
    lack = missing(engine)
    if lack:
        raise TypeError(f"{type(engine).__name__} does not implement the engine protocol: missing "
                        f"{', '.join(lack)} (see engine/protocol.py)")
    return engine


def can(engine, name):
    """Whether `engine` has the optional capability `name`."""
    if name not in OPTIONAL:
        raise ValueError(f"{name!r} is not an optional capability (one of: {', '.join(OPTIONAL)})")
    return callable(getattr(engine, name, None))
