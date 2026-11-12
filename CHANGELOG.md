# Changelog

All notable changes to AI-llm-Engine. Dates are 2026. Russian version: [CHANGELOG.ru.md](CHANGELOG.ru.md).

## Unreleased

### Added
- **Tools ▸ Browser Assistant opens Safari on the extension.** The Safari extension now ships inside LLMBar.app;
  the button sets up the service and opens Safari ▸ Settings ▸ Extensions on it to be ticked, and the extension
  takes its token from LLMBar — no Terminal, no folder, nothing to paste. Until the app is signed by an Apple
  developer, Safari needs Develop ▸ Allow Unsigned Extensions first; the button says how.
- **Voice understands Spanish, and any mix of languages.** Spanish (Spain or Latin America) is in the recognition
  menu, "para", "basta" and "nueva conversación" work as voice commands, and *Any language — Whisper detects it*
  lets you switch between Spanish, Russian and English mid-conversation without changing a setting.
- **One queue for everything that uses the model, with priorities.** Voice and the browser assistant go ahead of
  chats, chats ahead of background agents (`X-Priority: interactive | normal | background`); waiting raises a
  request's priority, so nothing starves. While the Mac is past its RAM limit, background requests wait, merged
  passes take fewer requests, and an idle engine gives memory back on its own — the conversation cache is saved
  to the SSD first. `/health` shows the queue and the memory picture.
- **Browser Assistant: a side panel that reads the page and acts on it while you watch.** The toolbar button
  opens a panel beside the page (Chrome's own side panel; docked into the tab in Safari, which has none) with
  each tab's conversation, the model controls and a choice of what the assistant may do: answer about the page,
  act asking before every step, or act on its own and stop before anything that submits a form, pays, orders,
  sends, deletes or leaves the site. It clicks, types, picks options, presses keys, scrolls and navigates; the
  page shows a glowing frame with Stop and an agent cursor that moves onto each element before it acts. With a
  model that can see, 📷 sends a screenshot with a question or with every step. Passwords, card numbers and
  one-time codes are never typed and CAPTCHAs are never touched. Dark and light themes, as in LLMBar. Works in
  Safari, Chrome and other Chromium browsers (`./llm assistant setup` prints the steps). Tested with a simulated
  page and a stand-in browser; not yet driven in a real Safari or Chrome.
- **Browser Assistant (Safari, optional).** An extension that asks the model on your Mac about the page you
  are reading — a question, a summary, or Explain / Translate / Rewrite on selected text — and controls the
  model from the browser: pick it, load it, unload it, stop an answer, with LLMBar and the terminal closed.
  It reads a page only when you ask, says how much of it was sent, and nothing leaves the machine. Install
  it from LLMBar ▸ Tools ▸ Browser Assistant or `./llm assistant setup`. For now Safari loads it as a
  temporary extension (it has to be added again after Safari restarts) — a permanent install needs a signed
  app. See [docs/safari-assistant/README.md](docs/safari-assistant/README.md). Not yet tried in a browser
  by its author: the service and the extension's logic are tested, Safari itself was not driven.
- **`POST /v1/engine/sleep` and `/v1/engine/wake`**: unload the model's weights without stopping the engine,
  and load them again — refused while other requests are in flight unless forced.
- **Chat and voice understand a plain request for a picture or to be read aloud, without a tag.** `./llm chat`
  offers the model `generate_image`/`speak` tool calls once the matching backend (mflux/Kokoro) is installed,
  and `/draw <prompt>` / `/speak <text>` do the same directly, offering to install the backend first if it
  is missing. In LLM Voice, asking for an image by voice draws one and shows it in the conversation; saying
  "stop"/"cancel" or "new conversation" (English or Russian) as the *whole* thing you say acts on it directly
  instead of being sent to the model — a stray word inside a real question is never mistaken for a command.
- **The code editor's Plan mode asks before acting on its own plan, in the same conversation.** The plan is
  now presented as an approval card; approving it continues the same run with edits applied automatically
  (commands still ask), rejecting it keeps the conversation in Plan mode to refine the plan further.
- **`./llm help`, `./llm <command> --help`, and Zsh/Bash tab completion** (`./llm completion zsh|bash`) built
  from one command registry, so the list can't drift from what the CLI actually runs; a misspelled command
  gets a "did you mean" suggestion.
- **Interrupted downloads stay visible in the Models panel**, independent of the current search box, until
  resumed to completion or removed — including one left over from a previous session, not only the current
  app launch.
- **A GGUF Mixture-of-Experts model's experts can be streamed from disk instead of decoded all at once**
  (`stream_experts`, on by default — `LLM_GGUF_STREAM_EXPERTS=0` turns it off): only the experts a token
  actually routes to are decoded, into a small LRU-cached pool, instead of every expert at load time. A
  prompt that would route to more distinct experts at once than the cache holds is processed in smaller
  token groups automatically, rather than refused.
- **`./llm perf --long-context`**: also measures prefill throughput at 64K/128K tokens (off by default — a
  size that does not fit on the machine is reported as skipped, not crashed).
- **Model Lab** (see [docs/model-lab/compatibility.md](docs/model-lab/compatibility.md)): `./llm lab
  inspect` lists every tensor's exact name, shape and dtype on a model (MLX or GGUF), addressable
  individually, not grouped for reading like `./llm inspect`; `./llm lab capture` records pre-residual,
  post-attention/GatedDeltaNet and post-MLP hidden states at chosen decoder layers for one prompt, on the
  native engine route. `./llm lab wsp patches create/apply/rollback` is a direct, reversible weight edit
  (`W_effective = W_base + ΔW`) on one tensor, native or GGUF, quantized or plain — the checkpoint file is
  never touched. `./llm lab wsp patches ablate` computes a multi-layer edit from two prompt groups'
  activations (native only). A patch is made to stick for ordinary use two ways: `wsp patches
  activate`/`deactivate` re-applies it at every engine start (like `--steer`), or `wsp patches export`
  bakes it into a complete, separate model folder once. Also in LLMBar's Tools tab, under Diagnostics.
- **`./llm conduct board`**: the Conductor's subtask routing (who delegated what to which model, and whether
  it worked) is now written to an append-only log as a run goes, not only printed to the terminal and lost —
  inspectable during a run or after it finishes.
- **Merged-pass batching in the server** (`--batch-window <ms>`, `--batch-max`): requests that arrive within the
  window are generated in one forward pass, so the weights are read once for the batch instead of once per
  request. Measured 1.70x at two requests, 2.51x at four, 2.48x at eight (`tests/bench_batch.py`); off by
  default, and a pass that ends up with one request runs the ordinary path so it keeps the prefix cache and
  speculation. Requests with images, a reasoning block or a `response_format` are never merged.
- **Embeddings on the native engine**: `POST /v1/embeddings` (float or base64) and Ollama's `/api/embed` and
  `/api/embeddings`. The vector is the loaded model's final hidden state at the last token, L2-normalized;
  nothing extra is loaded, each input is prefilled on a cache of its own and the conversation's prefix cache is
  left untouched. `dimensions` is refused with the model's real width rather than truncating.
- **Constrained decoding** (`response_format`, Ollama `format`): `engine/structured.py` is a character-level
  pushdown automaton, and the sampler takes the highest-ranked token that keeps the document valid, ending the
  reply when it is whole. The answer parses or the request fails. Enforces `type`, `properties`, `required`,
  `additionalProperties`, `items`, `enum`; any other schema construct is refused with a 400 naming it, before
  the model is woken. About 10 % of decode speed.
- **A stop button for the spoken reply** in LLM Voice: the send button becomes a stop while the answer is being
  written or read aloud, and stops both without opening the microphone.
- **Measurements that did not exist**: `tests/startup.py` (server ready, time to first token, shutdown),
  `tests/context_decode.py` (decode against context length), `tests/layer_profile.py` (per-layer timing inside
  the real forward), `tests/bench_batch.py`.

### Fixed
- The request timeout started when the engine lock was acquired, so waiting for the model ate it. It now starts
  at the first token check.
- `tools/make-tiles.py` trims the dead field around each screenshot and puts them on one canvas, so the README
  row reads as a set rather than a pile; the transparent background is kept.
- **Downloads in LLMBar often stalled or failed outright.** Three independent causes: a transient DNS/
  connection hiccup or Hugging Face rate-limit on the one metadata request that starts a pull crashed the
  whole download before a single byte of the model was fetched — it now retries like the rest of the
  transfer does. A slow connection could go over a minute between progress updates (progress only ticked
  after a full 1 MiB read), long enough for LLMBar's 90-second stall watchdog to bounce a transfer that was
  never actually stuck — the read size is now much smaller, so progress moves far more often. And when a
  repo's listing didn't report a file's size, LLMBar silently dropped every progress line for it instead of
  at least recognizing the transfer was still alive, which could make a healthy download look stalled.
- **Start runs the model you selected, even when another one is stuck.** An engine that held — or was still
  loading — a different model than the selected one used to be waited for; LLMBar sat on "still loading"
  for a model nobody had picked. It is stopped and the selected model starts.
- **A tap on the widget's background no longer starts the voice chat** — only its Voice button does; a tap
  between the buttons opens the panel. The model name on the widget now always switches the model (the next
  favourite, or with fewer than two favourites the next installed model that fits), in the background.
- **Chats on GGUF models (own decoder) and mlx-lm models answered 500.** The server passed the request's
  JSON format to a backend that did not accept the argument. Fixed; `response_format` now constrains the
  answer on those backends too.
- **MLX Mixture-of-Experts models bigger than memory run with their experts on the SSD.** Exact — the same
  answers as with everything loaded — with a per-layer pool of experts filled from the checkpoint as the router
  asks, parallel positional reads, least-used eviction and a hot set saved between runs. On by itself when a MoE
  model does not fit (`LLM_MOE_STREAM`, `LLM_MOE_CACHE`, `LLM_MOE_DIRECT`); `./llm models` lists it as
  "streams experts from the SSD", `/health` reports the pool's hit rate. It makes such a model runnable, not
  fast: on OLMoE-1B-7B it gives 15 – 18 tok/s at half the experts resident (against 108 – 126 with all), set by
  the hit rate and the SSD. Both ways MLX stores experts are read — stacked, and one tensor per expert.
- **Long conversations slowed down far more than they should on mlx-community checkpoints.** These store
  their quantization scales as bfloat16, which made half of the KV cache float32 — twice the size, and
  converted on every token. It is float16 throughout now: on Qwen3.5-0.8B-4bit decoding is about 1.3× faster
  at a short context and 2 – 3× faster at 8K – 32K, the cache takes a third less memory, and the answers are
  the same. JANG / Prism bundles were never affected.
- **A weight patch on a quantized model did not do what it said.** Writing the patched tensor back
  re-quantized all of it, which moved weights nobody asked to move (15 % of a 4-bit tensor's words for a
  delta of zero) and realized only part of a fine-grained edit such as an ablation. A patch now changes only
  the weights its delta touches, reports how much of the requested change is really in the model, and can be
  stored exactly (`--storage float16`, the default for `ablate`). Also fixed: a patched model shared its SSD
  prefix cache with the unpatched one; `wsp patches export` lost the vision tower, produced a Hadamard
  (JANG / Prism) bundle that would not load, and could shift a JANG bundle's norms twice; `activate` accepted
  a patch made for another model.

### Changed
- **The README's efficiency figures were overstated and are corrected**: the weights column counted the
  checkpoint on disk, which for these vision-language models includes a vision tower that is not loaded for text
  work. Against the memory-bus ceiling the three models reach **70 %, 64 % and 64 %**, not 82 / 97 / 77.
- Gatekeeper blocks a downloaded build on first launch (it is signed ad hoc, not notarized). The README, the
  site's install steps and `build-dmg.sh` now say so and name the two ways through; `build-dmg.sh` notarizes and
  staples when `LLMBAR_SIGN_IDENTITY` and `LLMBAR_NOTARY_PROFILE` are set.

### Measured, not changed
- Decode halves between a short context and 32K (76.7 → 36.9 tok/s on the 2B), and all of it is in the six
  layers holding a KV cache; the eighteen GatedDeltaNet layers grow by 0.64 ms between them. Five candidate
  causes were measured and ruled out — see `tests/layer_profile.py`. The whole step reaches 77 GB/s at a short
  context and 51 GB/s at 32K while every operation in it reaches 86-88 alone, which is chain efficiency rather
  than a defect.
- Cold CLI start is 0.06-0.31 s and the server answers 0.50-0.97 s after launch, so neither was optimized.

## 1.0.0 — 28 October

### Changed — one app
- LLMBar, the editor and the voice chat are **one app** (`LLMBar.app`, one process) split into modules: the panel,
  the editor and the voice chat are switched on and off in the settings, create nothing until opened, and share one
  theme (LLMBar's). `build-editor.sh` and `build-voice.sh` are gone: `./build-app.sh` builds everything, the desktop
  widget included, and `./build-app.sh --install` puts it in /Applications. Links are `llmbar://…`.
- LLMBar opens expanded at launch. The wide widget shows GPU load and RAM use on either side of the orb while the
  model is on; the state dot is a filled circle when it is on and an empty ring when it is off.

### Added
- **Batched decode (`Engine.generate_batch`)**: runs several generation requests in one lockstep batch on top of the
  bit-exact kernel in `engine/batch.py`, sharing weight reads across requests. Motivated by reproducible measurement
  (`tests/profile.py`, `tests/concurrency.py`), where ~68% of a decode token is weight reads.
- **Hardware baseline & profiling**: reproducible profiling suites (`tests/profile.py`, `tests/concurrency.py`,
  `tests/perf.py`) documenting Apple M5 baseline metrics in README.
- **On-device multimodal orchestration** (`harness/perception.py` + the Conductor): attachments are recognised
  locally — audio via Whisper, scans/images/PDF via OCR, documents via text extraction — and fed to the optional
  orchestrator, which routes the work across your installed models by role (a fast model reads and extracts, the
  senior model writes the code) and merges one answer. Each subtask receives only the passages of the attachment
  relevant to it (TF·IDF excerpts). `./llm conduct "<task>" --file F …`. Nothing leaves the Mac. The **agents**
  do it too: `read_file` (editor and CLI agent) and a delegated task's attachments recognise a PDF, scan, image
  or audio file to text instead of returning "(binary file)", so the model can read them like any source.
- **Speech-to-text and text-to-speech** (`./llm transcribe <audio>`, `./llm say "<text>"`, `engine/audio.py`):
  on-device Whisper and Kokoro on MLX — optional, like image generation and OCR (`./llm setup --audio` installs
  `mlx-whisper`, `mlx-audio` and `misaki`; the models download on first use). `transcribe` returns text + language
  + timed segments; `say` synthesises a WAV with a Kokoro voice (`--voice`, `--speed`), plays it, and saves it
  (the WAV is written with the stdlib, no soundfile/scipy). The **voice chat** has an opt-in *Transcribe with
  Whisper* setting (off by default): the utterance is recorded while the system recogniser runs live, then
  transcribed with Whisper for the final text, falling back to the system text if Whisper isn't installed or
  returns nothing. Omni is the next piece.
- **Per-request timeout and graceful shutdown**: `--request-timeout N` ends a generation that runs past N seconds
  as a timeout (`stop_reason` `timeout`; the CLI chat says so). On SIGTERM the engine now stops taking new requests
  (a clean 503), lets in-flight ones drain for up to `--shutdown-grace` seconds, then persists the prefix cache and
  exits; `/health` reports `shutting_down`.
- **Request admission cap** (`--max-queue N`, or `advanced.max_queue` in `./llm`): refuses a generation request
  with 503 once this many are in flight, so a coding agent, a chat, the editor and voice all hitting one model
  stay bounded instead of piling up unbounded threads and prompt buffers; `/health` reports `inflight`/`max_queue`.
  (Merging several requests into one forward pass — proven losslessly in `engine/batch.py` — is wired on separately
  with real-model concurrency measurement, since it also has to carry speculation, stop strings and the thinking budget.)
- **CLI chat: a clear reason and retry**: when the engine cuts off a looping answer (`stop_reason` `repetition`),
  the chat now says so and suggests `/retry`; `/retry` re-sends the last question (dropping the previous answer),
  handy after an interrupted or repeating reply.
- **Voice barge-in**: in dialog mode, speaking while the reply is being read aloud stops it and listens, then
  answers the follow-up (ChatGPT-style). A barge-in listener runs during the reply with acoustic echo cancellation
  (`setVoiceProcessingEnabled`) so the mic ignores the assistant's own voice, and only a real couple-of-words
  utterance interrupts (the `BargeIn` policy) — a cough or a stray word won't cut the answer off. The mic button
  (and the orb) now show a stop while a reply is being read and stop it on a press, instead of looking like "Speak".
- **Documents as sources** (`./llm read <file>`, `engine/documents.py`): extracts plain text from PDF, DOCX,
  PPTX, HTML and text/markdown/CSV so a document can be attached to the model as one source — stdlib only, no new
  dependencies (Office files are zipped XML; PDF text is pulled from the content streams). A long document can be
  reduced to the passages relevant to a question — `./llm read <file> --query "…"` scores passages by TF·IDF
  (no model, no embedding) and returns the top ones in document order with char offsets for citations; without a
  query it truncates with a flag.
- **On-device OCR** (Apple Vision): a scanned PDF (no text layer) or an image file is read with `VNRecognizeText`
  through a small local helper (`app/ocr`, bundled with the app; `./llm setup --ocr` builds it for the standalone
  CLI). No network and no third-party dependency; without the helper those files are reported as "needs OCR".
- **More editor languages**: syntax colouring now covers ~40 languages — added Scala, Dart, Haskell, Elixir,
  Erlang, Clojure, R, Julia, Perl, Groovy, Zig, Nim, PowerShell, GraphQL, Protobuf, Solidity, HCL/Terraform and
  CMake (the last two closed `cmake`/`perl` mappings that had no highlighter). Declarative keyword/comment/string
  rules, no heavyweight grammar engine.
- **Ollama-style CLI**: `./llm run <model> [prompt]` (select, downloading the model first if only an `org/name`
  is given, start the engine, then chat — or print one answer for a trailing prompt), `./llm show [model]`
  (format, route, size, context, capabilities) and `./llm ps` (the model loaded now). `--json` on `show`/`ps`.
- **Update check** (`./llm update [--check]`): asks GitHub for the latest release, compares it to the running
  version, and — unless `--check` — downloads the `.dmg` and opens it to drag over the old app. `--json` for tools.
  LLMBar checks in the background (at most every few hours) and shows an "Update available" banner with an
  Update button; it never replaces the app on its own.
- **Optional background service** (`./llm service install | uninstall | status`): a per-user launchd agent that
  keeps the API available after login and restarts the engine on a crash, independent of the menu-bar app (so a
  coding agent can reach the local API with the app closed). Off by default; on-demand with the app/CLI stays the
  norm. The agent runs `./llm serve`, which reads the current config each start (switching model/port needs no
  reinstall), and the model still unloads when idle — an idle service costs little RAM, not the weights.
- **Custom source** (Settings → "Custom source", or `./llm fetch <url|magnet> [name]`): add a model from a source
  **you** provide — a direct http(s) link to a `.gguf` or an archive (`.zip`/`.tar*`, extracted safely), or a
  magnet/`.torrent` via `aria2c` if installed. It only downloads what you point it at, into your models folder.
- **Download manager**: several downloads run at once (a queue, two in parallel); a paused or failed one stays in
  the list with its progress (the `.part` files are kept, resume continues) instead of disappearing; a transfer
  that makes no progress for 90 s is bounced (resumes from disk) up to three times, then paused with a reason.
- **Profiles** (Modelfile-style): `./llm create <name> --from <model> --system "…" --param k=v` saves a named
  preset — a base model with a system prompt and sampling — that `./llm run <name>` then applies. `./llm create`
  lists them, `./llm create <name> --rm` (or `./llm rm <name>`) deletes one, `./llm show <name>` prints it.
- **Anthropic Messages API** (native `/v1/messages`, not an OpenAI wrapper): streaming SSE (`message_start` →
  `content_block_*` → `message_delta` → `message_stop`) or one message object, with `system`, `stop_sequences`,
  `tools`/`tool_use`/`tool_result` and `thinking` translated and the reply returned as Anthropic content blocks.
  Auth accepts `x-api-key`. Same loaded model and generation path as the OpenAI and Ollama routes.
- **Ollama API on the same port**: `POST /api/chat` and `/api/generate` (NDJSON streaming or one object; `options`,
  `tools`, `images`, `think` translated), `GET /api/tags`, `POST /api/show`, `GET /api/ps`, `GET /api/version` — an
  Ollama-speaking client needs no adapter. One generation path feeds both the OpenAI and the Ollama wire formats.
- **GGUF decoder — Gemma 2**: the own MLX decoder now runs Gemma 2 (tanh soft-cap on the attention scores and the
  output logits, sliding window on every other layer, sandwich norms, GeGLU, tied output), checked against an
  independent numpy forward with the same caps.
- **GGUF → MLX converter**: `./llm convert <model.gguf> [out]` exports a GGUF as a plain MLX folder — `config.json`,
  a float16 `model.safetensors` (dequantized by the decoder's own unpacking, renamed to the Hugging Face layout) and
  a `tokenizer.json` rebuilt from the file's vocabulary — so a converted model opens with mlx-lm without the GGUF
  around. llama / Qwen2 / Qwen3 for now.
- **Conductor** (optional, off by default): `./llm conduct` splits a task across models by role (heavy / code / fast
  → a local model switched in turn, or a remote API model run in parallel), with an editor panel to map the roles.
- **Reliability**: an answer cut off for looping says why; a watchdog stops the engine (and any llama-server) if the
  app that launched it crashes; a document that cannot be read stops the send and keeps the text; the chat is saved
  before generation and as it streams, with a Retry for an unfinished reply; Voice stops reading when you speak and
  lets the model, not just keywords, decide when to search the web.
- **Editor remembers its sizes**: the widths of the columns (chats / conversation / files in the chat layout,
  files / editor / outline+assistant in the editor layout), the height of the bottom panel and of the outline, and
  the window's size and place are kept between launches (dividers are plain draggable handles that save what you
  set; a saved place that no longer fits a connected screen is ignored).
- **Assistant tools over MCP** (opt-in *Assistant Tools (MCP)* module): a catalog (context7, Stitch) and your own
  servers — a program over stdio or a streamable-HTTP URL — connected so the editor's assistant can use their tools
  on the open project (they run with the project as their working directory). Every call is approved, a server is
  contacted only on *Test* or use, and the servers live in the workspace `.llm/mcp.json`, trusted per content. This
  replaces the VS Code (Open VSX) extensions feature, which is removed from the app and archived under
  `legacy/vscode-extensions/` (it never shipped in a release).
- **Voice**: a lookup on the web only happens on a clear request ("найди в интернете…", "search the web…") or a clear
  sign that the answer changes (today, weather, price, score…) — never for a greeting, and without an extra model
  call for them.
- **Editor assistant**: an exact echo of the previous answer is never shown (it is asked again, with a note, then
  without the copied conversation); a model under ~1 GB gets a short prompt and no tools, with a note, instead of
  the tool loop it cannot drive.
- **Editor assistant knows what is open**: the system prompt names the file that is open and the cursor line, and a
  message about "this code" / "check", "fix", "explain"… (EN/RU) with nothing attached sends that file along as the
  usual attachment chip (a small writable file, local model only) — so "проверь код" is about the code on screen
  instead of a greeting.
- **Routes say what they can do**: `./llm models` prints under every model which engine runs it (engine,
  engine-ptq1, engine-gguf, mlx-lm, llama.cpp) and its features (vision, thinking, tools, ssd-cache, speculation,
  steering); `models --json` carries `engine` and `features`, and the panel shows them in the model's tooltip.
- **Editor terminal runs full-screen programs** (vim, less, htop, nano): the screen model has the alternate
  screen, scroll regions, insert/delete lines, saved cursor, application cursor keys, bracketed paste and DEC line
  drawing; the pty has a real size that follows the panel (the shell is told with SIGWINCH), F-keys and Page
  Up/Down are sent. Leaving a program restores the scrollback exactly as it was.
- **`./llm inspect [<#|name|path>] [--json]`**: what is inside a model, from its headers only (a 27B in a blink,
  no MLX needed) — architecture, layers, heads, vocabulary, context, experts, parameters and bits per weight (PTQ1 /
  TQ1 counted at 128 weights per 28 bytes), storage types, the largest tensor groups with layer numbers folded, and
  the engine route (for a GGUF: whether the own decoder runs it, and why not).
- **Agent: parallel `delegate`**: one call can ask several workers the same task (`workers`) or hand different
  jobs to different workers (`jobs`, up to 4); they run at the same time, one confirmation shows every destination
  (and flags remote ones and command-line workers that may change files), and the answers come back together, one
  labelled section per worker — a failure of one does not hide the others.
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
