# Backend interface specification

Three seams, at three grains. Only the first is new code; the second is an interface the code already
obeys without having written it down; the third is a specification for work that has not started.

| Seam | Grain | State |
|---|---|---|
| [1. Kernel](#1-kernel-seam) | one operation (attention over the cache, a quantized matmul) | **implemented** — `engine/kernels/` |
| [2. Engine](#2-engine-seam) | one loaded model serving requests | **in use, undeclared** — four classes obey it by convention |
| [3. Runtime](#3-runtime-seam) | tensors, buffers and a forward pass | **specified here, not built** |

A replacement enters at the finest seam that can hold it. A Metal kernel for one operation is a kernel;
it does not need a new engine. A whole runtime that does not use MLX arrays needs seam 3.

---

## 1. Kernel seam

`engine/kernels/__init__.py`. An *operation* has a name, a reference implementation built from MLX
primitives, and any number of replacements.

### Contract of an implementation

```python
def impl(*args) -> mx.array | None
```

1. **Same signature and result as the reference**, within the tolerance stated in its test
   (`tests/test_kernels.py`). Bit equality is not required and usually not available — a kernel that
   accumulates in float32 where the reference uses float16 is *more* accurate, not equal.
2. **It may decline.** Returning `None` means "not my case"; the caller runs the reference. This is what
   lets a kernel cover only the hot path (one sequence, one new token) and stay a hundred lines.
3. **It never raises at request time for a reason known at start.** Device and build problems are
   reported by `available()`, which is checked once when the operation is resolved.
4. **It has a row in `tests/bench_kernels.py`**, measured against the reference on the same inputs,
   by the method described at the top of that file (distinct buffers so data comes from DRAM,
   interleaved rounds, one eval per round).
5. **It is the default only with that row behind it.** `register(..., default=True)` is a claim that
   the benchmark supports; without it the kernel ships switched off.

### Registry

```python
kernels.register(op, name, fn, available=None, default=False)
kernels.resolve(op)          -> (name, fn)      # honours LLM_KERNELS, falls back to the reference
kernels.get(op)              -> fn
kernels.reference(op)        -> fn
kernels.implementations(op)  -> {name: available?}
kernels.active()             -> {op: name}      # reported by /health
kernels.reset()                                  # after changing LLM_KERNELS in a test or benchmark
```

### Selection

| Mode | How | Behaviour |
|---|---|---|
| Automatic | `LLM_KERNELS` unset or `auto` | each operation's registered default, else the reference |
| Manual, global | `LLM_KERNELS=mlx` | every operation on that implementation where it exists |
| Manual, per operation | `LLM_KERNELS=attn_decode=metal,qmv=tiled` | listed operations pinned, the rest automatic |
| Capability detection | `available()` per implementation | no Metal device, or the kernel does not build → skipped |
| Fallback | always | unknown name, unavailable implementation, or a `None` return → the reference |
| Performance-based | **not automatic, on purpose** | see below |

Performance-based selection is done by a person with `tests/bench_kernels.py`, and recorded as
`default=True`. Timing kernels at start-up was considered and rejected: a fanless Mac changes clocks
within seconds of load, so a 200 ms probe measures the thermal state it happened to start in, and
the engine's cold start (0.5 – 1 s) is a figure users see. If per-machine selection becomes necessary,
the right form is a cached result of an explicit `./llm calibrate --kernels`, not a probe on every start.

### Operations

| Operation | Signature | Reference | Replacements | Default |
|---|---|---|---|---|
| `attn_decode` | `(q[1,H,1,D], keys[1,Hkv,cap,D], values[1,Hkv,cap,D], n, scale) -> [1,H,1,D]` f16 | `mx.fast.scaled_dot_product_attention` on `[:n]` | `metal` — chunked online softmax, GQA rows read once | `mlx` (the Metal kernel measures 0.89 – 1.03× — see the review) |

Operations that already have their own Metal code but predate the registry, and should move behind
it when next touched (not before — they work, and moving them is not an optimization):

| Existing kernel | File | Natural operation name |
|---|---|---|
| gated-delta recurrence | `engine/gated_delta.py` | `gdn_step` |
| ternary lookup-table GEMV, small GEMM, dequantize | `engine/ptq1.py` | `ternary_mv`, `ternary_mm`, `ternary_dequant` |

### Call-site rule

The model code takes the reference path unless a replacement is active, so the default build is the
code it was before the seam existed:

```python
# engine/model.py — Ops.__init__
name, fn = kernels.resolve("attn_decode")
self.attn_decode = None if name == kernels.REFERENCE else fn

# Attention.__call__
o = None
if ops.attn_decode is not None and L == 1 and mask is None:
    o = ops.attn_decode(q, cache.keys, cache.values, cache.offset, self.scale)
if o is None:
    o = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=m)
```

---

## 2. Engine seam

`engine/server.py` holds one global `ENGINE` and is written against whatever it happens to call on it.
Four classes satisfy that today: `Engine` (native), `CompatEngine` (mlx-lm), `GGUFEngine` (own GGUF
decoder, subclass of `CompatEngine`), and the llama.cpp path, which sidesteps the question by being a
separate server process (`engine/gguf_server.py`) behind the same HTTP API.

The interface below was **read off the call sites**, not designed: it is every attribute and method
`server.py` touches. It is declared in `engine/protocol.py` (contract `VERSION` 2, served in `/health` as
`engine_protocol`) and `protocol.check()` holds the engine to it when the server starts. The same file
writes down the other contracts between modules — the runtime seam (`RUNTIME`, `RUNTIME_BATCH`), what the
shared decode loop reads from its engine (`LOOP_HOST`), the memory manager (`MEMORY`) and the `/health`
keys clients rely on (`HEALTH`) — and `tests/test_protocol.py` checks every engine and runtime against them.
That file is authoritative; the listing below is its readable form.

### Required

```python
class EngineBackend(Protocol):
    # identity and limits
    backend: str                          # "native" | "mlxlm" | "gguf-mlx" | …
    max_context: int
    max_tokens: int
    model_bytes: int
    mem_limit: float                      # fraction of system RAM the whole machine may use

    # state
    memory: "memory.Manager"              # the order memory is given back in, and ok / tight / critical;
                                          # whose turn it is belongs to the server's scheduler, not to an engine
    sleeping: bool
    last_used: float
    model: object | None                  # None while asleep
    cache_tokens: list[int]               # tokens whose effects are in the cache
    stats: dict                           # {"requests": int, "last": {...}}
    totals: dict                          # session counters, see generate.summarize
    history: collections.deque
    store: object | None                  # SSD prefix store, or None
    wired_limit_advice_mb: int | None

    # prompt
    thinking: bool
    thinking_default: bool
    effort: str
    sampling: dict[bool, dict]            # presets keyed by "reasoning open?"
    vision_allowed: bool
    def build_prompt(self, messages, tools=None, thinking=None, effort=None,
                     extra_kwargs=None) -> tuple[str, list, bool]: ...
    def encode(self, text, images) -> tuple[list[int], object | None]: ...

    # work
    def generate(self, ids, media=None, max_tokens=None, sampling=None, stop=None,
                 should_stop=None, seed=None, thinking_open=False, fmt=None
                 ) -> Iterator[tuple[int | None, str] | dict]: ...   # pieces, then one stats dict
    def embed(self, texts, normalize=True) -> tuple[list[list[float]], int]: ...

    # lifecycle
    def sleep(self) -> None: ...
    def ensure_awake(self) -> None: ...
    def persist_prefix(self) -> None: ...
    def memory_breakdown(self) -> dict: ...
```

### Optional (the server probes with `hasattr`)

```python
    def generate_batch(self, reqs, on_piece=None) -> list[dict]: ...   # merged decode of several requests
    def parser_markers(self) -> dict: ...                              # reasoning / tool-call tags
    def parse_tool_calls(self, text, tools) -> list: ...
```

### The stats dict every `generate` ends with

`prompt_tokens, cached_tokens, completion_tokens, prefill_s, decode_s, prefill_tps, decode_tps,
finish_reason, think_forced, loop_stopped, peak_gb, draft_accept, verifies, ram_used`. Clients and the
menu-bar app read these; a backend that cannot fill one reports `None`, never omits the key.

### Selection (as implemented)

Two places decide, which is itself a deviation (see below): the CLI picks the *process*, the server
picks the *class*.

| Mode | How |
|---|---|
| Automatic, process | `llm.py`: a `.gguf` the own decoder can run (`gguf.native_support`, setting `gguf_native`) starts `engine.server`; any other `.gguf` starts `engine.gguf_server` (llama.cpp) |
| Automatic, class | `compat.pick_backend`: a `.gguf` file → `GGUFEngine`; a directory whose config is the Qwen3.5 hybrid family (`family.native_capable`) → `Engine`; anything else → `CompatEngine` |
| Manual | `--backend auto\|native\|mlxlm\|gguf` |
| Capability detection | `family.native_capable(config)`, `gguf.native_support(path)`, `mlx_lm_available()` |
| Fallback | a GGUF the own decoder refuses runs on llama.cpp; mlx-lm missing → an install hint, not a crash |

### Known deviations to close before a fifth backend is added

* ~~`Engine` and `CompatEngine` each carry their own copy of the memory guard and the decode loop~~ —
  closed: one guard (`memory.reserve / check / fit_output`), one release order (`memory.Manager`), one
  loop (`session.DecodeLoop`). Sleep/wake is still per engine.
* Backend choice is split between `llm.py` (which server process) and `compat.pick_backend` (which
  class). One function should own it and report *why* it chose, for `./llm models` and `/health` alike.
* ~~Two cache classes with different verbs~~ — closed: `engine/cache.py`.
* `server.py` reads MLX memory counters directly for `/health` (`mx.get_active_memory()` …). Behind
  seam 3 these come from the runtime.

---

## 3. Runtime seam

**Built for the decode loop** (`protocol.RUNTIME`, `protocol.RUNTIME_BATCH`): `MlxHybridRuntime`
(`engine/runtime.py`, the native model) and `MlxLmRuntime` (`engine/compat.py`, mlx-lm and GGUF models)
implement it, and `engine/session.py` and `engine/batch.py` call nothing else — a test fails if they call
a method the contract does not list. The listing below is the **target** a runtime that is not MLX would
grow into (load/unload, opaque cache save/restore, memory reports); the calls actually made today are the
`RUNTIME` names, described one by one in `engine/session.py`.

The design choice that matters: the seam is at the **forward pass and the cache**, not at the tensor.
An interface of `matmul / add / reshape` would re-create MLX badly and force every backend to be a
tensor library. What the Inference Core actually needs from a model is small:

```python
class Runtime(Protocol):
    name: str                                         # "mlx" | "native"
    def capabilities(self) -> set[str]: ...           # {"verify", "batch", "hidden", "vision", "steer", …}

    # model
    def load(self, path, *, layer_range=None, options: dict) -> "ModelHandle": ...
    def unload(self, model) -> None: ...

    # state — opaque to the caller
    def make_cache(self, model) -> "CacheHandle": ...
    def cache_len(self, cache) -> int: ...
    def cache_bytes(self, cache) -> int: ...
    def truncate(self, cache, n) -> None: ...          # forget tokens past n (may raise if not rewindable)
    def snapshot(self, cache) -> "Snapshot": ...       # cheap restore point (recurrent state)
    def restore(self, cache, snap) -> None: ...
    def save(self, cache, file) -> None: ...           # SSD prefix store
    def load_cache(self, model, file) -> "CacheHandle": ...

    # compute
    def forward(self, model, cache, ids, *, last_only=True, verify=False,
                positions=None, embeds=None, mask=None) -> "Logits": ...
    def commit(self, cache, m) -> None: ...            # after verify: keep the first m fed tokens
    def sample(self, logits, sampler_state) -> "Token": ...      # may stay on the device
    def read_token(self, token) -> int: ...            # the only synchronisation point per step
    def flush(self, *handles) -> None: ...             # start work without waiting (async_eval)

    # memory — what the Memory Manager asks
    def memory(self) -> dict: ...                      # active / cached / peak bytes
    def set_limits(self, *, wired: int | None, cache: int | None) -> None: ...
    def trim(self) -> int: ...                         # release reusable buffers; returns bytes freed
```

Notes that the MLX path already dictates, and a native runtime must honour:

* **`sample` returns a handle, `read_token` blocks.** The decode loop builds step *t+1* while *t* runs
  (`mx.async_eval`); that pipelining is worth 15 % of the step (`tests/profile.py`). A runtime that
  forces a synchronous read inside `sample` loses it.
* **`verify` + `commit(m)`** is how prompt-lookup speculation stays exact on a hybrid model: the
  recurrent layers re-run only their cheap recurrence for the accepted tokens. A runtime without it
  reports no `"verify"` capability and the Inference Core decodes plainly.
* **Caches are float16** and their size per token is part of the contract: the Memory Manager budgets
  the context from it. (The defect found in this audit was exactly a cache that was silently larger
  than its budget assumed.)

### Selection, when a second runtime exists

`--runtime auto|mlx|native`. Automatic prefers the runtime whose `capabilities()` cover what the model
and the request need (vision, steering, the quantization in the file), and among those the one the
saved benchmark for that model family favours. Anything the chosen runtime raises `Unsupported` for at
load falls back to MLX — the same shape as today's own-decoder → llama.cpp fallback.
