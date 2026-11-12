# Knowledge Engine — Phase 0 audit and architecture

Status: **Phases 1, 2 and 4 implemented** — the core (`engine/knowledge.py`, the OCR helper's page mode,
`./llm knowledge`), the code editor and quick chat integration, and the Knowledge window in LLMBar. Phase 3
(semantic search) and 5 are deferred. Audit date: 2026-11-12, repository state `main` @ `87f27b3`.

**Non-negotiables** (they override any other line in this document):

1. **No indexed documents → no RAG.** Nothing is opened, searched, loaded or added to a prompt.
2. **Minimal external dependencies.** Python stdlib and Apple frameworks already in use; zero new packages.
3. **The app stays light.** The loaded LLM keeps the memory: no secondary model, no daemon, no background
   work that competes with inference. Anything that breaks the budgets in §8 becomes opt-in or is removed.

This document records what the repository already has, what a local knowledge/retrieval feature ("Knowledge
Engine", a small RAG) can reuse, and the minimal design proposed for the following phases. Every claim about
existing code cites the file and function it was verified in.

---

## 1. Starting point and how the master prompt is used

The feature was specified in an external "Master Prompt v2.4". That prompt assumes a Swift-native app with an
`mlx-swift-lm` inference stack. This repository is different:

* the **inference engine and all agent/harness logic are Python** (`engine/`, `harness/`, `ui/`, `assistant/`,
  `llm.py`; the `mlx` Python package, `requirements.txt`);
* **Swift is the shell**: LLMBar (`app/bar`), the code editor (`app/editor`), Quick chat (`app/voice`), the
  widget, the Safari bridge, and a small Vision/PDFKit OCR helper (`app/ocr`).

The prompt is therefore treated as a set of **principles**, not as literal instructions:

| Kept | Adapted or dropped |
|---|---|
| Primary LLM inference has priority over everything else | "No Python runtime": Python *is* the runtime here |
| Dormant when there is nothing to search (no work, no context, no model) | `mlx-swift-lm` / `MLXEmbedders`: not used by this app |
| Lexical retrieval first (SQLite FTS5 / BM25), embeddings optional | SQLite C API: Python's stdlib `sqlite3` is the equivalent |
| No resident secondary model; never unload the main model for RAG | Swift Model Scheduler / RAM Guard: ours are `engine/scheduler.py` and `engine/memory.py` |
| Zero new third-party dependencies | 11-entity schema: reduced to ~5 tables |
| Reuse existing PDF/OCR; no parallel pipelines | 4 retrieval modes × 3 resource profiles: one switch and one profile first |
| Bounded, cancellable work; no daemon; no polling | "Proceed to implementation after the audit": each phase is reviewed first |
| One shared API for every assistant; citations from real evidence only | |

---

## 2. Pluggable and dormant by design

Knowledge is an **optional module**, in line with the app's existing modules (Quick chat and the editor are
already "a module, opens on demand", `app/bar/Bar.swift:1776`):

1. **Not loaded unless used.** `engine/knowledge.py` is imported lazily, the first time a request references a
   library or a temporary knowledge document. Ordinary chat paths do not import it.
2. **Dormant fast path.** Before any retrieval the scope resolver answers one question: *does this request's
   scope contain at least one document in state `ready`?* The answer is kept as a tiny in-memory flag/
   counter per process, refreshed when ingestion or deletion changes it. If the answer is no:
   no database open, no FTS query, no embedding call, no context block, no added latency. The request goes
   down the existing non-RAG path unchanged.
3. **Request must ask for it.** Even with ready documents, retrieval runs only when the chat has Knowledge
   enabled and libraries selected (or temporary documents attached). Having a library somewhere does not
   make every chat search it.
4. **No background presence.** No daemon, no server, no folder polling, no model. Ingestion is a job that
   runs when the user adds a source and stops when it is done.
5. **Release on empty.** When the last ready document is removed, the process drops its connection and
   caches; library metadata and settings remain until the user deletes them.

Acceptance test for this (Phase 1): with an empty knowledge store, an ordinary chat request performs no
`sqlite3` call and its request body/prompt is byte-identical to today's.

---

## 3. Existing architecture relevant to Knowledge

### 3.1 Request paths (where retrieved context would be injected)

| Client | Path | Where text attachments are handled today |
|---|---|---|
| Code editor chat / agent | Swift `ChatStore` (`app/editor/Chat.swift`) → `Backend.agent` (`app/editor/Backend.swift:240`) → `POST /api/agent` on the UI server | `ui/server.py` `Handler.api_agent` (line 710): `P.attachments()` + `P.inline_attachments()` (lines 434–490), joined by `attachment_block()` (line 493), appended to the last user message; limit `MAX_ATTACH_TOTAL = 200_000` bytes |
| Editor plain chat | `POST /api/chat` | `ui/server.py` `Handler.api_chat` (line 923), same attachment code |
| LLMBar Quick chat (voice/typed) | `app/voice/VoiceModule.swift` → engine `/v1/chat/completions` on port 8090 **directly** | `Files.read` / `Files.block` in the same file: text files ≤ 120 KB, ≤ 300 KB total, appended after the message |
| Browser assistant | Safari extension → `assistant/service.py` (stdlib HTTP, token-checked) → engine | page text supplied by the extension |
| Conductor / delegation | `harness/conductor.py` | uses `documents.select_excerpts(srctext, task, budget=4000)` (line 294) |
| CLI | `./llm read <file> [--query]` (`llm.py` ~3385) | `documents.extract_text` + `select_excerpts` |

Today every client inlines whole files or (CLI/Conductor) query-selected excerpts of **one** document. There
is no persistent index, no multi-document search and no citations.

### 3.2 Chat persistence (lifecycle hooks needed for temporary knowledge)

* Editor chats: one JSON file per project in `~/Library/Application Support/LLMBar/editor-chats/`, written by
  Swift `ChatStore.save` (`app/editor/Chat.swift`) and by Python `harness/chats.py` (`save`, `delete` at
  line 129). Deletion today only removes the session from the array.
* Quick chat history: `~/Library/Application Support/LLMBar/voice-chats.json`; `VoiceHistory.delete`
  (`app/voice/VoiceHistory.swift:56`).

Both are the places where a "chat deleted → drop its temporary knowledge bindings" call will go.

### 3.3 Scheduling and memory (reused, not duplicated)

* `engine/scheduler.py` `Scheduler`: one queue for every use of the model, priorities
  `interactive` > `normal` > `background`, aging, admission limit, and deferral of background turns while
  memory is `critical`. Embeddings already go through it (`with SCHED:` in `engine/server.py:691`).
* `engine/memory.py`: `available_bytes`, `pressure_level`, `headroom_bytes`, the RAM guard (`reserve`,
  `check`) and `Manager.state()` → `"ok" | "tight" | "critical"`.

Knowledge ingestion will use `Manager.state()`-equivalent checks (the functions are module-level and
usable outside the engine process) and run at `background` priority whenever it needs the model.

### 3.4 Embeddings that already exist

`POST /v1/embeddings` (and Ollama `/api/embed`), `engine/server.py:657` → `Engine.embed`
(`engine/generate.py:352`): the vector is **the loaded chat model's final hidden state** at the last token,
L2-normalised. Properties:

* no second model, no download, zero extra resident memory — fits the lightweight principle;
* but the vector space changes whenever the user switches models, so any stored vectors are invalidated by
  a model switch; and unless the loaded model is a retrieval-trained one, quality for search is unproven
  (the docstring says so explicitly);
* it is available only from the native backend (`protocol.can(ENGINE, "embed")`).

Conclusion: semantic retrieval is **deferred** (Phase 3) and must be justified by measurements; lexical
retrieval is the product.

---

## 4. PDF and OCR audit

Repository-wide search (`git grep` over Python and Swift, excluding `llama.cpp/`, `third_party/`,
`node_modules/`) found exactly two PDF/OCR implementations.

| Capability | Existing implementation | Verified behaviour / limitations | Knowledge Engine action |
|---|---|---|---|
| PDF preview | **Not found.** `app/editor/Preview.swift` is a WKWebView for HTML/dev servers; no `PDFView`/QuickLook usage in `app/` | — | Defer; open the original in the system viewer |
| Native PDF text | `engine/documents.py` `_pdf()` — stdlib regex parser: inflates FlateDecode streams, collects `(…)` strings near `Tj/TJ/BT` | No page boundaries (only a page **count** via `/Type /Page`); literal strings only — **hex strings and CID/Identity-H fonts are not decoded**, decoding is latin-1, so most non-Latin (e.g. Russian) and many modern PDFs yield garbage or nothing; no reading-order handling; whole file read into memory | **Extend**: page-level extraction via the existing helper (below); keep `_pdf` as fallback |
| Page anchors | None (text is one string) | — | Add (needed for citations) |
| Scanned-page detection | `extract_text`: OCR only when `_pdf` returns **no text at all** | Mixed PDFs (some scanned pages) are never OCR'd | Per-page detection |
| OCR | `app/ocr/main.swift` (built to `bin/ocr` by `./llm setup --ocr`, `llm.py:1626`): Apple Vision `VNRecognizeTextRequest` (`.accurate`, language correction, optional `--lang`), PDF pages rendered with PDFKit at 2×; invoked by `documents._run_ocr` (subprocess, 180 s timeout) | On-device, offline, no model download, no third-party code. Whole file only; pages joined by blank lines (page numbers lost); no per-page cancellation; also offers `--describe` for images | **Reuse and extend in place** |
| Images | `extract_text` → OCR helper | Requires the helper | Reuse |
| DOCX / PPTX / HTML / text | `_docx`, `_pptx`, `_strip_markup`, `_read_text` (stdlib zipfile/regex) | Works for simple documents; no structure anchors | Reuse; DOCX/PPTX not in the first Knowledge release |
| Excerpt selection | `select_excerpts()` (line 238) — TF·IDF over ~1400-char passages | Tokeniser `_TOK = [A-Za-z0-9_]+` (line 213) is **ASCII-only: Cyrillic queries match nothing** | Superseded by FTS5 `unicode61` for Knowledge; worth fixing independently |
| Tests | `tests/test_documents.py` (stdlib-crafted files, OCR cases skip without the helper) | No multi-page, mixed, rotated, CID-font or encrypted PDF fixtures | Extend this file |

**Decision.** Do not add a PDF library and do not write a second OCR path. Extend the existing helper with a
page mode, e.g. `ocr --pages <file.pdf> [--ocr-missing] [--lang …]`, that prints one JSON line per page:
`{"page": n, "text": "…", "source": "text" | "ocr" | "none"}`. Text comes from PDFKit `PDFPage.string`
(handles CID fonts, Unicode, rotation); Vision runs **only** on pages whose native text is empty or below a
threshold, and only when OCR is allowed. Python reads the stream line by line, so a cancelled ingestion kills
one subprocess and keeps the pages already received. Without the helper, `_pdf` remains the fallback and the
document is marked "no page anchors".

This also improves today's attachments (`extract_text` could use the same page mode), but that is a separate
change.

---

## 5. Proposed minimal design (Phase 1)

### 5.1 Module and storage

* `engine/knowledge.py`, **stdlib only** (`sqlite3`, `hashlib`, `re`, `json`).
* One database: `~/Library/Application Support/LLMBar/knowledge/knowledge.db` (WAL, small page cache,
  explicit `user_version` migrations, `foreign_keys=on`).
* FTS5 is available in both the system Python and the project venv (verified: SQLite 3.54.1, `fts5` and the
  `trigram` tokenizer); the module still runs a capability probe and reports clearly if it is missing.

Tables (first release):

| Table | Purpose |
|---|---|
| `libraries` | id, name, created, settings JSON |
| `documents` | id, content hash, original path, display name, kind, state (`importing/indexing/ready/failed/missing`), pages, note, indexed_at |
| `chunks` + `chunks_fts` | chunk id, document id, ordinal, anchor (page / heading path / line range), char offsets, text; FTS5 external-content table over `text` |
| `owners` | (document_id, owner_kind, owner_id): `library:<id>` or `chat:<id>` — a document lives while any owner exists |

Which libraries a chat searches is **not** stored here: it is part of the chat's own state (its JSON in
`editor-chats/` or `voice-chats.json`) and arrives with each request as a `Scope(libraries, chat)`. The index
holds documents; the conversation holds its selection.

Embedding tables (`embedding_spaces`, vector BLOBs) are **not** created until Phase 3.

### 5.2 Ingestion

`add(path, owner)` → hash → if this hash is already `ready`, only add the owner (no re-parse) → extract
(pages for PDF, headings for Markdown, line ranges for code/text) → deterministic chunking (~1–2 KB, split on
headings/paragraphs/pages, no LLM, minimal overlap) → one transaction inserting chunks → `ready`.
Limits: file size cap, chunk count cap per document, bounded subprocess time for OCR, and default exclusion
of secrets (`.env`, keys, `*.pem`, credential files) and `.gitignore`d paths for folders.

Imports are **managed snapshots** of the extracted text (the original file is never modified or deleted);
linked folders with FSEvents are a later phase.

### 5.3 Retrieval and context

* Query → FTS5 `MATCH` with safely quoted terms, `bm25()` ranking, scoped by the chat's owners/bindings in
  the same SQL, `LIMIT` ≤ 50 candidates.
* Exact search for identifiers/paths/phrases uses FTS5 phrase queries; the `trigram` tokenizer is an option
  for substring search over code (to be decided by measurement — it roughly triples index size).
* Russian/other languages: `unicode61 remove_diacritics 2` tokenises Cyrillic correctly; FTS5 has no Russian
  stemmer, so prefix queries (`term*`) are used for word forms. Quality to be evaluated on real documents.
* Context builder: dedupe, keep top passages within a character budget derived from the model's context and
  the space left after history, render each with a citation label (`name.pdf p. 12`, `README.md § Install`,
  `src/x.py:40-88`). Retrieved text is wrapped as untrusted reference material. No evidence → no block.

### 5.4 Access surface (one API for every client)

* Python API in `engine/knowledge.py` — used in-process by `ui/server.py` (and later the Conductor and the
  browser assistant service).
* CLI `./llm knowledge … --json` — the **one** surface for the Swift app: libraries, documents, add/attach,
  detach, forget-chat, promote, search, `--context`. A short process that answers and exits.
* Editor: requests already pass through the UI server, so retrieval happens there (`ui/server.py`
  `knowledge_context`, body field `"knowledge": {"libraries": [...], "chat": "<id>"}`). A request without the
  field is unchanged — the module is not even imported. Sources reach the transcript as a `knowledge` SSE event.
* Quick chat: **decided in Phase 2 — the CLI**, not an engine route. Measured 80 ms per `search --context` call,
  paid only when the quick chat has a library ticked, and the engine server stays untouched (no new endpoint,
  no import). The quick chat already calls the CLI the same way for web search.

### 5.4.1 Swift side (Phase 2 + 4)

* `app/knowledge/KnowledgeLogic.swift` — pure decoding and the per-chat `KnowledgeChoice` (tested in
  `tests/swift/main.swift`).
* `app/knowledge/KnowledgeModule.swift` — `KnowledgeCLI` (runs the CLI), `KnowledgeStore` (shared state; runs
  nothing while no database exists on disk), `KnowledgePicker` (the 📚 menu in the editor composer and the quick
  chat), and the **Knowledge window** (libraries: create, rename, delete; documents: add files/folders, remove,
  re-index changed ones, show in Finder; a "try a question" box showing exactly what a chat would get). Opened
  from LLMBar ▸ Tools ▸ *Knowledge* or the picker's *Manage Knowledge…*; closing it releases its views.
* A module like the editor and the voice chat: `AppHost.Module.knowledge`, switched off in LLMBar settings
  (*Knowledge module*), link `llmbar://knowledge`.
* Editor chats store their choice (`knowledge`, `knowledgeChat`) in the chat JSON; both fields are optional when
  read, so older chats and `llm chat` files still open. Deleting a chat with its own documents runs
  `forget-chat` (Swift `ChatStore.delete`, Python `harness/chats.delete`).

### 5.5 Lifecycle

* Persistent libraries survive chat clear/delete, assistant switches and model switches.
* Temporary documents attached to a chat are owned by `chat:<id>`; deleting the chat (`harness/chats.delete`,
  Swift `ChatStore.delete`, `VoiceHistory.delete`) removes that owner; a document with no owners is
  deleted with its chunks. "Promote to library" adds a library owner.
* Source changed (hash differs) → citation marked stale, re-index on request.

---

## 6. Scheduling and resource policy

* Lexical ingestion and search **do not use the model** and therefore never take a scheduler turn.
* Ingestion runs in a worker thread / subprocess with low priority (`os.nice`), one document at a time, and
  pauses while `memory.pressure_level() >= 2` or `used_fraction()` is near the configured limit.
* OCR is one helper process at a time, page-streamed, cancellable.
* Any future model use (embeddings) goes through `SCHED` at `background` priority and is skipped — with a
  visible reason — when the engine is busy with interactive work or memory is not `ok`. The main model is
  never unloaded for Knowledge.

---

## 7. Dependency delta

**Zero new packages.** Phase 1 uses Python stdlib `sqlite3` (FTS5 built in) and extends the existing Swift
OCR helper with Apple frameworks it already links (Vision, PDFKit). No ORM, vector library, parser framework,
server or second runtime. Binary size: the helper grows by a few hundred lines of Swift; the app bundle gains
one Python module.

---

## 8. Baseline measurements

Measured with a throw-away script (not committed) on an Apple M5, 16 GB, macOS 26 (Darwin 25.4), Python 3.15.2,
SQLite 3.54.1. Synthetic corpus: chunks of ~1.4 KB random words plus a few Russian words; FTS5
`unicode61`, WAL, 2 MB page cache; queries = 2-term, Russian 2-term, 1-term; top-8 by `bm25()`.

| Corpus | DB size | Insert (one transaction) | Query p50 | Query max | Peak RSS of the whole Python process |
|---|---|---|---|---|---|
| 10 000 chunks (~14 M chars) | 28 MB | 0.8 s | 0.08 ms | 0.93 ms | 25.5 MB |
| 100 000 chunks (~140 M chars) | 265 MB | 6.5 s | 0.42 ms | 8.9 ms | 25.5 MB |

`import sqlite3`: 2.5 ms. The process peaked at ~26 MB including the interpreter (~17 MB before the import),
i.e. single-digit MB for the store itself, independent of corpus size (the index is read from disk via the
page cache).

Interpretation: lexical retrieval is effectively free next to LLM inference. Random words are a worst case for
the index size and an easy case for ranking; real-text numbers will differ.

**Not yet measured** (needs the real app and models — to be done in Phase 1 on target hardware):

* engine/UI-server RSS delta with Knowledge imported but idle, and while searching, with a large model loaded;
* end-to-end added latency per chat request with Knowledge on vs off;
* PDFKit page-mode extraction speed and OCR throughput per page;
* retrieval quality on real English and Russian documents and code.

### Phase 1 measurements (real code, real text)

Same machine, project venv (Python 3.14.7). Index: this repository's `engine/`, `docs/` and `README.md` —
63 documents, 780 chunks, 1.7 MB database. Footprint = `memory.process_footprint()` (what Activity Monitor
shows), measured inside one process.

| What | Result | Budget |
|---|---|---|
| `import engine.knowledge` + `Store()` | 11 ms, +4.3 MB | < 5 MB idle |
| `context()` with an empty scope | 2 µs, database not opened | no measurable work |
| `context()` (search + build), 100 queries | p50 0.24 ms, p95 0.33 ms, max 1.0 ms | < 20 ms p95 |
| Footprint after the 100 searches | +6.2 MB total | < 10 MB |

The import cost only applies to a process that actually uses Knowledge: clients import the module lazily
(Phase 2 keeps it that way and tests it).

### Budgets (acceptance for Phase 1)

* Empty store / Knowledge off: **no** measurable change in request latency or prompt bytes (byte-identical).
* Knowledge idle in a process: < 5 MB RSS delta. Search: < 20 ms p95 for ≤ 100 k chunks, < 10 MB RSS delta.
* Ingestion yields under memory pressure and never takes a model turn.
* Any capability that misses these becomes opt-in or is removed.

---

## 9. Phases (adapted to this stack)

| Phase | Content | Review gate |
|---|---|---|
| 0 | This document | ✔ done |
| 1 | `engine/knowledge.py` (schema, ingestion, chunking, FTS5/exact search, owners, context builder with citations), OCR helper page mode, `./llm knowledge` CLI, tests in `tests/test_knowledge.py` + `tests/test_documents.py` | ✔ done — dormant-path tests, budgets met |
| 2 | Integration: UI server `knowledge` field, editor chat and Quick chat selector (📚 menu, empty state), chat-deletion hooks, sources in transcripts | ✔ done (Quick chat via the CLI, see §5.4) |
| 3 | Optional semantic retrieval via existing `/v1/embeddings`, vectors as BLOBs keyed by model fingerprint, streamed exact scan, RRF fusion — only if measured quality gain justifies it | benchmark |
| 4 | Knowledge Center (libraries, documents, progress, diagnostics) as an optional module off the main screen | ✔ done — the Knowledge window |
| 5 | Linked folders (FSEvents + bookmarks), browser page capture, Conductor/agent tool, more formats | per feature |

---

## 10. Open decisions and risks

1. ~~Quick chat access~~ — decided: the CLI (§5.4).
2. **Trigram index for code** — measure size vs. benefit before enabling.
3. **Embedding strategy**: main-model hidden states (free, but per-model and unproven) vs. a small dedicated
   embedding model (costs RAM and a download). Default remains lexical-only.
4. **Russian morphology**: no stemmer in FTS5; prefix queries may be insufficient — evaluate on real data.
5. **PDF fallback quality** without the OCR helper built: `_pdf` cannot decode CID fonts; consider making the
   helper part of the default build since it has no dependencies.
6. **Prompt injection**: retrieved text is untrusted; it is labelled as reference material and never grants
   tool permissions.
