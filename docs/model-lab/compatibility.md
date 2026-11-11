# Model Lab — compatibility and scope

A pasted spec asked for an 8-phase, 5-milestone "Weight-Space Programming" research subsystem: model
inspection, activation capture, hypothesis-driven direct weight editing with rollback, an agent tool
contract so a Teacher LLM can drive experiments on a Student model, sequential Teacher/Student scheduling,
optional LoRA/QLoRA, a Bonsai-2-ternary research target, and macOS UI integration. This document records
what was actually audited and built: inspection, activation capture, and a full direct-weight-edit
mechanism (single-tensor patches, a multi-layer ablation helper, apply-at-every-load activation, and
baking a patch into a standalone exported model) — and what's still deliberately deferred.

## What exists vs. what's new

| Capability | Status | Where |
|---|---|---|
| Tensor name/shape/dtype, no dequantization | **Already existed** — reused, not duplicated | `engine/loader.py`'s `read_shapes()`, `engine/gguf_tensors.py`'s `GGUFFile`/`TensorInfo` |
| Human-readable grouped model overview | **Already existed**, unrelated command | `./llm inspect` (`engine/modelinfo.py`) — groups tensors by layer pattern for a person to read |
| Exact per-tensor listing, addressable by name | **New** | `./llm lab inspect --model <m> [--tensor <name>]` (`model_lab/inspection.py`) — deliberately *not* grouped, because a patch tool needs to name one exact tensor, not a human-readable summary |
| Activation capture (pre-residual, post-attention/GatedDeltaNet, post-MLP) | **New** | `./llm lab capture --model <m> --prompt <text> [--layers ...]` (`model_lab/capture.py`) |
| Direct reversible weight edits (`W_effective = W_base + ΔW`) on one tensor, native or GGUF | **New** | `./llm lab wsp patches create/apply/rollback` (`engine/weightpatch.py`, `model_lab/patch.py`) |
| Multi-layer ablation (mean-diff direction from two prompt groups, removed from `mlp.down_proj`) | **New**, native only | `./llm lab wsp patches ablate` (`model_lab/ablation.py`) |
| Making a patch "stick" for ordinary use, without touching the checkpoint | **New**, two ways | `wsp patches activate/deactivate` (re-applied every load, like `--steer`) or `wsp patches export` (baked into a separate model folder once) |
| Agent/Teacher tool-calling contract | **Not built yet** — deferred | needs schema validation + a tool allow-list the current chat tool-loop doesn't have (see `llm.py`'s `chat()` tool loop) |
| LoRA/QLoRA training | **Not built yet** — optional, separate track per the spec's own framing | `mlx-lm`'s `mlx_lm.tuner` is reachable only for the optional compat backend, not the native engine |
| Bonsai-2 / ternary weight editing | **Not built yet** — would need a new codec + Metal kernel module from scratch, mirroring `engine/ptq1.py` | out of scope until a real checkpoint's packing is independently verified; `engine/weightpatch.py` explicitly refuses PTQ1/TQ1 tensors rather than guessing |
| macOS UI | **Not built yet** — spec itself says UI comes only after CLI experiments work | |

## Why inspection didn't become a second overview command

`./llm inspect` already reads safetensors/GGUF headers only (no numpy/mlx needed) and groups tensors by
layer pattern (`model.layers.N.mlp.down_proj.weight` for every `N` becomes one row) for a person to read.
`./llm lab inspect` is not a copy of that: it returns every tensor by its **exact** name, ungrouped — the
shape a future `propose_edit`/`activate_patch` tool will need to address one specific tensor, which a
grouped summary cannot do. Both commands stay; they answer different questions.

## Why activation capture needed new engineering, and why it isn't a `set_steer` reuse

`engine/model.py` has exactly one existing hook into the residual stream: `Model.set_steer`, used for
activation steering. It cannot be reused for capture. `steer` is invoked *inside* `Ops.mlp`/`gdn_post`/
`attn_post` — closures that are then wrapped in `mx.compile` (see the `for name, fn in locals().items(): …
mx.compile(fn)` loop at the end of `Ops.__init__`). That's correct for steering, where `steer(y)` is a pure
array transform folded into the traced dataflow and genuinely re-evaluated on every call. A capture hook
instead needs a Python-level *side effect* — stash a copy of the activation — and `mx.compile` traces
Python code only once per input shape, caching the result; a side effect placed inside that trace fires
once, at trace time, not on every subsequent real call. Using `steer`-style injection for capture would
silently record only the first call's activations, not the ones for whatever prompt is actually being
inspected.

`model_lab/capture.py` avoids this by working one level up: `DecoderLayer.__call__` itself is plain,
uncompiled Python that already calls `layer.self_attn`/`layer.linear_attn`, then `ops.mlp`, as two separate
statements. `capture_activations()` makes the exact same two calls in the exact same order — it does not
reimplement any of the real math — and simply captures the Python-level return value between them. This
was verified, not assumed: `tests/test_model_lab.py::TestCapture::test_capturing_does_not_change_the_models_own_output`
compares the last token's captured hidden state against an ordinary `model(ids, cache, hidden=True)` call
and asserts they match (observed: exact bit-for-bit equality on the test fixture, asserted at `atol=1e-4`
for headroom). `test_residual_stream_is_continuous_across_layers` additionally confirms layer *N*'s
captured output is literally layer *N+1*'s captured input, the same tensor the real model passes along.

Captured points, per layer: `pre_residual` (the layer's input), `post_attn_residual` (after the attention
or GatedDeltaNet block, before the MLP), `post_mlp_residual` (the layer's output). This satisfies the
spec's "pre/post-residual" activation request; it does not expose the raw sub-block output *before* its
residual add in isolation (that value only exists inside the still-compiled `attn_post`/`gdn_post`/`mlp`
closures) — if a future experiment needs that specifically, it would mean decomposing those compiled blocks
further, not something this increment attempted.

Native-engine checkpoints only. GGUF models (`engine/dense.py`) and mlx-lm-backed models use entirely
different forward-pass code with no equivalent to `DecoderLayer`/`Ops` — `./llm lab capture` refuses any
model whose route isn't `"native"` with a clear message, rather than silently doing nothing useful.

## The weight-patch mechanism

`engine/weightpatch.py` locates one tensor on either a loaded `engine.model.Model` (native) or
`engine.dense.DenseModel` (GGUF) and computes `W_effective = W_base + ΔW`. Reversibility follows
`tests/test_steering.py::TestInModel::test_off_zero_and_on_behave_as_they_should`'s template (apply →
compare to baseline; reload without the patch → confirm the baseline is reproduced exactly):
`tests/test_model_lab.py`'s `TestPatchNative`/`TestPatchGGUF` reload the checkpoint fresh after patching
and assert the output is bit-for-bit the original, on both routes.

**How the sum is stored decides what the patch does.** A patch names its storage (`manifest.json`'s
`"storage"`, `--storage` on the command line):

| Storage | What is kept | Memory | What survives |
|---|---|---|---|
| `quantized` (default for `create`) | the tensor's own bits; every group keeps its scale and bias, only the codes of weights that moved are re-rounded, a group whose values leave its range gets a new range | unchanged | changes of about a quantization step or more. A weight whose delta is zero keeps its code exactly, so a zero delta is the identity |
| `float16` (default for `ablate`) | the patched tensor, dequantized | that tensor at 16 bits per weight (4× a 4-bit one) — +5 MiB for one `down_proj` of Qwen3.5-0.8B | everything, to float16 rounding |

`apply_delta` returns, and `apply` / the engine's start-up log print, how much of the request is really in
the weights: `realized` (the share of the requested change that arrived) and `unrequested` (the size of
whatever else moved, relative to the request).

This replaced a first version that re-quantized the whole tensor with `mx.quantize`, on the assumption
that it "round-trips". It does not: `mx.quantize` refits every group's range, and refitting an already
quantized group is not the identity. Measured on `mlx-community/Qwen3.5-0.8B-4bit`,
`layers.10.mlp.down_proj` (4-bit, group 64, median step 2.6·10⁻³, weight rms 8.0·10⁻³):

| | First version | Now, `quantized` | Now, `float16` |
|---|---|---|---|
| Zero delta: packed words changed | 69 696 of 458 752 (15 %), up to 3 steps | 0 | — |
| Ablation, α = 1 (delta rms 2.9·10⁻⁴, a ninth of a step): direction component left | 63 % | 72 % | 0.02 % |
| … unrequested change, relative to the request | 1.25× | 0.64× | 0.01× |

So on a 4-bit tensor an ablation kept quantized is mostly not applied, in either version — the first one
only hid it behind noise larger than the edit. That is why `ablate` stores float16 unless told otherwise,
and why the original end-to-end ablation test had to store `down_proj` unquantized to pass. The quantized
case is now tested as what it is (`TestWhatAPatchReallyDoes`).

A tensor that is tied — GGUF Gemma's output head is the input embedding — is patched in both places:
`output.weight` is refused with "patch token_embd.weight instead", and patching `token_embd.weight` now
moves the head with it (it used to leave the head on the old arrays).

The SSD prefix cache of a patched model is its own: the cache key includes the patch (manifest, size and
mtime of the deltas, alpha). Before, a patched engine restored prefixes the unpatched one had stored.

Refused, not silently misapplied: MoE experts, ternary-packed (PTQ1_0/TQ1_G128 — the format the project's
own already-installed `Bonsai-2-27B-CRACK-Ternary-JANG` checkpoint uses) tensors, and `mx`-mode
quantization (mxfp4/mxfp8/nvfp4 — only `affine` round-trips through `mx.quantize`/`mx.dequantize`
cleanly). The native route's fused input projections (`qkv_proj`, `gate_up_proj`, `in_proj_qkvz`,
`in_proj_ba`) are patchable only by the *fused* name the live model actually holds, not the raw per-part
names a GGUF-style listing would show — naming an unfused part is refused with a message explaining why,
never guessed at.

### Ablation (`./llm lab wsp patches ablate`)

`model_lab/ablation.py` computes a unit mean-diff direction from two prompt groups' activations at each
chosen layer (`mean(harmful) - mean(safe)`, via `model_lab/capture.py`) and removes it from that layer's
`mlp.down_proj` rows (`W' = W - alpha·d(d^T W)`, zeroing the output's component along `d` for any input at
alpha=1 — verified analytically in `tests/test_model_lab.py::TestAblation`, and end to end on a quantized
tensor with float16 storage; kept quantized, most of the edit rounds away — see the table above). Native route only, MLP output projection only (every layer has one, and it's never
fused) — attention/GatedDeltaNet output projections are a small, symmetric follow-up, not attempted here.
This computes a *candidate* direction from whatever prompts it's given; it does not judge whether they're
representative, and a direction extracted this way is something to evaluate on held-out prompts, not a
proven fix (the project's own scientific-boundaries framing, not unique to this feature).

### Making a patch stick: `activate`/`deactivate` vs. `export`

Two ways to use a patch without touching the checkpoint, confirmed by direct testing to both work (and to
both have real bugs the first time they were tried — see below):

- **`wsp patches activate --model <m> --patch <id> [--alpha 1.0]`**: stores the patch directory in
  `cfg["models"][path]["weight_patch"]` (mirroring `./llm steer`'s own per-model config), which
  `./llm start`/`chat` thread through to `engine/loader.py`'s `load(..., weight_patch=...)` /
  `engine/dense.py`'s `build(..., weight_patch=...)` — applied fresh every time the engine loads this
  model, the same way `--steer` already is. `deactivate` clears it. Confirmed end-to-end: the engine log
  shows `weight patch on: <tensors>, alpha <n>` on a real `./llm start`.
- **`wsp patches export --model <m> --out <folder> [--patch <id>]`**: `model_lab/export.py` applies the
  patch (if any) in memory, then writes a complete, standalone copy of the model folder with the patched
  weights baked in — the source folder is never touched or used as the destination.

Export looked trivial at first (`mlx.utils.tree_flatten(model.parameters())` straight to a new
`model.safetensors`) and **was not** — two real bugs, found only by reloading the export and comparing
output bit-for-bit, not by a shape check:

1. **Fused quantized tensors must be split back to their checkpoint's original per-part names.**
   `Model.__init__` decides each module's quantization (`engine/quant.py`'s `module_spec`) by reading
   `scales`/`biases` off the checkpoint header *under the unfused per-part hint name*
   (`self_attn.q_proj`, `mlp.gate_proj`, `linear_attn.in_proj_qkv`/`in_proj_b`, ...) —
   `engine/loader.py`'s own `FUSE` table names — never the fused attribute name the live module holds. A
   checkpoint with only the fused name's `scales` makes `module_spec` report "not quantized," so the
   fresh module is built as plain `FLinear` at the wrong (unpacked) shape while the actual tensor is still
   packed — a crash on reload. `mlp.gate_up_proj` exists in every layer of every model, so this is not an
   edge case. `model_lab/export.py` un-fuses each of the four fused tensor types back into real per-part
   tensors by slicing `weight`/`scales`/`biases` along axis 0 (output rows) — the exact inverse of
   `loader.py`'s own `mx.concatenate(slot, axis=0)`, with row counts computed from `model.args` (the same
   config fields the modules were built from), so no precision is lost beyond what fusing already didn't
   introduce.
2. **An unquantized fused tensor needs splitting too, for an unrelated reason.** `loader.py`'s rule for
   which float32/bfloat16 tensors get downcast to float16 explicitly exempts the unfused
   `in_proj_a.weight`/`in_proj_b.weight` (GatedDeltaNet's gate projections, kept at full precision on
   purpose) but has no matching exemption for the fused `in_proj_ba.weight` — leaving it fused, even
   though its *shape* alone reloads without error, silently lost precision that was never meant to be
   lost. `export.py` now always splits every fused tensor, quantized or not, rather than tracking which
   naming exemptions exist where.
3. (Caught and fixed before either of the above, same investigation:) `jang_norms_pre_shifted: false` in
   `loader.py`'s own shift condition (`shift = ... or config.get("jang_norms_pre_shifted") is False`)
   means "not shifted yet, shift it now" — the opposite of what the name suggests. Setting it to `false`
   in the exported config (believing it meant "no further shift needed") doubled every RMSNorm weight's
   +1 shift. The exported config now pins it `true`.

4. **Hadamard sign vectors and the vision tower were left behind.** Neither is in the live parameter
   tree: the signs become `model.had` at load, and the lab loads text-only. A JANG / Prism export did not
   load at all ("missing sign vectors"), and a vision checkpoint's export silently stopped being a vision
   model (no tower, and no index for the loader to find one by). `export.py` now carries both over from
   the source file and writes an index.
5. **`jang_config.json` can ask for the norm shift too** (`layout.language_norms:
   "zero-centered-runtime-plus-one"`), and the loader obeys it whatever `jang_norms_pre_shifted` says — so
   pinning the latter (item 3) was not enough for such a bundle: its export came back with every norm
   shifted twice (logits off by ~4 on the synthetic fixture). The exported `jang_config.json` no longer
   asks.

## Explicitly out of scope, not rejected

Agent/Teacher tool-calling (needs a stricter schema-validated, allow-listed tool contract the current chat
loop doesn't have), LoRA/QLoRA (a separate optional track per the spec's own framing), Bonsai-2 ternary
editing (its own codec+kernel effort), and macOS UI integration (the spec's own ordering: UI only after CLI
experiments work) are all deferred, not attempted, in this increment.
