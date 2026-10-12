# Changelog

All notable changes to AI-llm-Engine. Dates are 2026; the project has not tagged a release yet — everything below
is part of the 1.0 line. Russian version: [CHANGELOG.ru.md](CHANGELOG.ru.md).

## Unreleased — 12 October

### Added
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
- Prompt-lookup speculative decoding for the mlx-lm and GGUF routes (the cache is trimmed back on rejected drafts).
- Tests: mutation tester (`tests/mutate.py`), reference-based tests for every GGUF block format, prefix-store rules,
  MCP / parser / image edge cases. Full suite: 700+ tests.

### Changed
- Pictures sent to a model that cannot read them are dropped with a one-line note instead of a 400.
- Request parameters that cannot be used (`max_tokens`, `temperature`, `stop`…) answer 400 before anything runs.
- `--steer` on the mlx-lm backend is an error at start, not a silent no-op.
- `build-dmg.sh` writes the installer to `docs/` — the file the site serves.

### Fixed
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
