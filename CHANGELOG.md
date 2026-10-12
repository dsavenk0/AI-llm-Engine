# Changelog

All notable changes to AI-llm-Engine. Dates are 2026; the project has not tagged a release yet — everything below
is part of the 1.0 line. Russian version: [CHANGELOG.ru.md](CHANGELOG.ru.md).

## Unreleased — 12 October

### Changed — one app
- LLMBar, the editor and the voice chat are **one app** (`LLMBar.app`, one process) split into modules: the panel,
  the editor and the voice chat are switched on and off in the settings, create nothing until opened, and share one
  theme (LLMBar's). `build-editor.sh` and `build-voice.sh` are gone: `./build-app.sh` builds everything, the desktop
  widget included, and `./build-app.sh --install` puts it in /Applications. Links are `llmbar://…`.
- LLMBar opens expanded at launch. The wide widget shows GPU load and RAM use on either side of the orb while the
  model is on; the state dot is a filled circle when it is on and an empty ring when it is off.

### Added
- **Widget**: the wide widget stars a favourite model, and with two or more favourites a tap on the name switches to
  the next one (`./llm fav --next`; a running engine restarts on it).
- **Editor: Command Palette** (⇧⌘P, View menu): type a few letters ("zi", "term", "newt") to find any command of the
  File and View menus and the Modules switches — the list is the menus themselves, with their shortcuts, so it
  never offers anything they do not.
- **Editor**: several terminals (tabs in the bottom panel, "+" for another shell, each a real shell with its own
  output; the last one cannot be closed); large files (over 2 MB) open as a read-only head of 256 KB with a banner
  instead of being refused, and are never saved from the editor (the server refuses it too, so a head cannot
  overwrite the file); type errors from `mypy` (Python) and `tsc` (TypeScript) join the Problems list when those
  tools are installed — modules the checker cannot find in a lone buffer are not reported as errors.
- **LLM Voice: dialog mode, the whole conversation, the web.** After an answer the microphone opens again by itself
  (menu: "Dialog mode"; it ends on a long silence, a click on the sphere while nothing is said, or a typed
  message). Each question is sent with the whole exchange so far (newest turns kept within a budget, never a dangling
  answer), and the prompt says follow-ups refer to it. Questions about news, weather, prices, scores or "today"
  are looked up on the web: the model proposes a query (with a keyword fallback in five languages when a small model
  stays silent), `llm web search` fetches DuckDuckGo results and the top pages, and the answer is built from them;
  a failed lookup is said honestly, not guessed. Menu: "Look things up on the web when needed" (only the query
  leaves the Mac).
- **More GGUF on the engine's own decoder**: Q2_K, Q3_K, IQ4_NL and IQ4_XS (decoded exactly — checked tensor by
  tensor against llama.cpp's own dequantization — then re-quantized per 32 weights to 4 / 5 / 6 / 6 bits, so they
  take more memory than in the file), SentencePiece vocabularies with the dummy space prefix (Llama 2 / Mistral /
  MiniCPM; token ids identical to llama.cpp), MiniCPM with LongRoPE (short factors) and its multipliers. IQ1–IQ3 and
  per-expert / shared-expert MoE files still go to llama.cpp.
- **Mixture-of-experts GGUF on the engine's own decoder** (Mixtral-style llama, Qwen3-MoE, Granite-MoE) and dense
  Granite: the experts stay stacked and quantized, a token runs only its top-k with `gather_qmm` (prefill groups
  tokens by expert), Granite's embedding / residual / logit / attention multipliers are read from the file.
  Granite-MoE 1B-A400M matches llama.cpp: identical token ids and, over 32 greedy tokens, identical text on two of
  four prompts (the others differ at near-tied tokens, logprobs within ~0.3). Files with per-expert tensors or a
  shared expert (Qwen2-MoE, DeepSeek) still go to llama.cpp.
- **Gemma 3 on the engine's own decoder** (GGUF): sliding-window layers with their own rope base, norms after the
  attention and feed-forward, GeGLU, tied output, linear rope scaling for the global layers, and a SentencePiece
  tokenizer built from the file (merges ranked by piece score, byte fallback). Checked against llama.cpp: identical
  token ids on a set of English / Russian / CJK / code / emoji strings, identical greedy text over 32 tokens including
  a 1900-token prompt that slides the window. llama.cpp stays the fallback for what the decoder does not cover.
- **A few rows on 1.75-bit models** (2–32: speculative verification, short prompts): a fused decode + 8×8 matrix
  multiply kernel replaces dequantize-then-matmul — 16 rows of a 34816×5120 matrix take 2.3 ms instead of 8.4 ms
  (PTQ1_0 and TQ1_G128). It is compute-bound (about 6 one-row steps for 16 rows), so speculation on these models
  stays off by default; `LLM_PTQ1_SPECULATIVE=1` turns it on for measurements.
- **LLM Voice**: the model menu is always there (with a chevron), also when no model is selected.
- **Readiness check** (`./llm check [--json]`, run by the panel at launch): Apple Silicon and macOS version, Python,
  engine libraries, network (only when something has to be downloaded), free disk, models. Problems appear as a
  banner with "Check again"; a missing connection is named as such instead of a failed install. The "ready" result
  is remembered (requirements + interpreter), so a working setup starts without re-scanning.
- **Offline build**: `./build-app.sh --offline` (and `./build-dmg.sh --offline`) puts Python and the engine's wheels
  inside the app (+~150 MB); the first launch then needs no network. Downloads are cached in `.build-cache`.
- A failed or interrupted first install no longer leaves a half-made environment that looks ready.
- **First launch on a clean Mac**: with no Python 3.10+ (or only an outdated one) the manager downloads a private
  CPython (python-build-standalone, ~35 MB, SHA-256 checked) into the data folder, then creates the engine environment.
  The panel says "FIRST LAUNCH" and shows the install progress instead of a silent wait.
- **Chat history in the terminal, shared with the editor**: `llm chat` saves every conversation per project in the
  same file the editor uses; `/history`, `/open <N>`, `/rename <N> <title>`, `/delete <N>` and `/new` list, continue,
  rename and delete chats, and a chat started in the editor continues in the terminal and back.
- **LLM Voice history**: a clock-arrow button at the top left opens saved conversations — open, rename, delete.
- **GGUF on the engine's own MLX decoder** (llama / Qwen2 / Qwen3 families; Q4_0…Q8_0, Q4_K, Q5_K, Q6_K): weights are
  repacked into MLX quantization on load, tokenizer and chat template come from the file. Anything else still runs
  on llama.cpp behind the same API.
- **Activation edits**: `--steer` / `--steer-alpha` and `./llm steer <repo|folder>` apply a direction edit at the
  residual writers; the base model an edit was made for is fetched with it (pinned revision).
- **LLM Voice**, a separate native app: speak or type, attach files and pictures, hear short answers read aloud;
  a real window with close / minimise, glass in the LLMBar style, its own icon, and a desktop widget
  (`./build-voice.sh --install`) that opens it already listening.
- **Editor**: agent modes (Manual / Edit automatically / Plan / Auto) and Effort, a context ring with Compact,
  chat rename / delete, file and *Discard all* git discard with a confirmation, Open VSX colour themes and snippets
  (opt-in), web search and picture generation tools.
- **Favourite models**: a star in LLMBar's model list, `./llm fav <number|name>`, favourites first everywhere; the
  editor's and Voice's model menus lead with them. A power button on the wide widget turns the model on / off in the
  background, and its voice / editor / terminal icons load the model ahead of use.
- Prompt-lookup speculative decoding for the mlx-lm and GGUF routes (the cache is trimmed back on rejected drafts).
- Tests: mutation tester (`tests/mutate.py`), reference-based tests for every GGUF block format, prefix-store rules,
  MCP / parser / image edge cases. Full suite: 700+ tests.

### Changed
- LLM Voice shows the selected model and its state, with a loading bar while the weights are read into memory;
  it listens the moment it opens, the model starting beside the microphone.
- Pictures sent to a model that cannot read them are dropped with a one-line note instead of a 400.
- Request parameters that cannot be used (`max_tokens`, `temperature`, `stop`…) answer 400 before anything runs.
- `--steer` on the mlx-lm backend is an error at start, not a silent no-op.
- `build-dmg.sh` writes the installer to `docs/` — the file the site serves.

### Fixed
- LLM Voice read LLMBar's settings (theme, transparency, port, key) from the wrong folder and used defaults.
- Editor: the line-number gutter no longer covers the first characters of every line after the window is resized,
  and a file that ends with a newline no longer gets a phantom line number.
- The Hugging Face token is no longer forwarded to the CDN a download redirects to.
- Git: `paths` given as a single string no longer stages the whole repository; huge diffs are cut at 2 MB.
- Shell tool output is capped (4 MB per stream); web fetches check the peer address after connecting.
- MCP client: noise between messages, repeating paging cursors, nameless tools, oversized results, dead servers.
- A stop inside an accepted speculative draft left the cache longer than the fed tokens.
- Image generation: a prompt that starts with `--` is no longer read as an option.

## 11 October

### Added
- **LLM Editor**, a native Swift app (on demand): chat-first and classic layouts, an agent that asks before it
  edits, git and GitHub, problems, terminal, preview, dictation, themes, model source (another OpenAI-compatible
  server), extensions module.
- **1.75-bit routes** on one Metal kernel: PTQ1_0, TQ1_G128, JANG packed (experimental).
- **Optional mlx-lm backend** for architectures the native engine lacks, with routing native / ptq1 / mlxlm.
- **Agent harness**: project memory and automatic compaction, project map and session notes, web search and fetch,
  MCP client (stdio + streamable HTTP, Context7 and Stitch presets), `/fix-tests`, `/workers` delegation.
- **Benchmarks**: 14 suites with confidence intervals (GSM8K, MMLU, MMLU-Pro, ARC, HellaSwag, TruthfulQA,
  WinoGrande, KMMLU, C-Eval, MathQA, BBQ, SafetyBench, HumanEval, MBPP); sandboxed code runs.
- **Optional image generation** through mflux (`./llm setup --images`, `/v1/images/generations`).
- Pluggable reasoning / tool-call parsers with a manual override; GGUF files in folders, LM Studio and Ollama
  stores through llama.cpp; ModelScope as a second model source; pause / resume of downloads.
- LLMBar: model routes and reasons, per-model settings, delete, Liquid Glass, prompt-reading progress, settings.

### Changed
- Any MLX quantization is loaded from tensor shapes; the reasoning default comes from the model's template.
- The SSD prefix cache writes only on sleep and stop; its budget can be `auto`.

### Fixed
- Stale pid files, stuck ports, command timeouts; stop strings are cut out of answers; `/health` answers while the
  engine sleeps; looping answers stop (`repetition_penalty`); a short file is never left under its final name.

## 10 October
- Memory breakdown in LLMBar, clearer settings, rounded panel, own tray icon; `/v1/stats`.
- Landing page (Apple × Cyberpunk HUD) and GitHub Releases links.

## 9 October
- First public version: lean MLX engine for Qwen3.5 / JANG bundles on Apple Silicon, with the LLMBar menu-bar app
  (glass panel anchored under its icon).
