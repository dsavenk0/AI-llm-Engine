# Assistant service protocol (version 1)

`./llm assistant serve` listens on `http://127.0.0.1:8093`. JSON in, JSON out; two endpoints stream
server-sent events.

## Who may call

Every request is checked in this order; the first failure answers.

| Check | Failure |
|---|---|
| `Host` is `127.0.0.1`, `localhost` or `::1` | 403 |
| `Origin`, if present, is a browser extension (`safari-web-extension:`, `chrome-extension:`, `moz-extension:`) — never a web page | 403 |
| `Authorization: Bearer <token>` matches (all paths except `/hello`) | 401 |
| body is a JSON object under 4 MB | 400 / 413 |

CORS is answered for extension origins only, echoing the caller's own origin. The token is created on
first use in `~/Library/Application Support/AI-llm-Engine/browser-assistant.json`, mode 0600.

## Version negotiation

`GET /v1/assistant/hello` → `{"service": "ai-llm-engine-assistant", "protocol": 1}`, without a token. Every
snapshot also carries `protocol`. A client that sees another number says so and does not guess.

## State

`GET /v1/assistant/state` → a snapshot. `GET /v1/assistant/events` → a stream: one
`{"type": "state", "state": <snapshot>}` at once, then one per change, with a `: keep-alive` comment every
15 s. `version` rises by one with every change, so a client that reconnects takes the first event as the
truth and discards what it held.

```json
{
  "version": 12, "protocol": 1,
  "engine": {"state": "ready", "error": null, "operation": null},
  "model": {"state": "ready", "name": "qwen3.5-0.8b-4bit", "path": "/…/Qwen3.5-0.8B-4bit"},
  "default": "/…/Qwen3.5-0.8B-4bit",
  "models": [{"name": "…", "path": "…", "size_gb": 0.58, "format": "MLX 4-bit", "route": "engine",
              "supported": true, "problems": [], "fits": true, "vision": true, "thinking": true, "favorite": false}],
  "in_use": {"inflight": 0, "assistant_generations": 0},
  "generations": [{"id": "…", "session": "…", "since": 1794400000.0}],
  "metrics": {"context_tokens": 45056, "model_gb": 0.39, "system_ram_used": 0.71, "last_decode_tps": 193.0},
  "extension": {"versions": {"safari": "0.1.0"}, "folder": "/…/browser-assistant"},
  "capabilities": {"unload": true, "per_session_model": false, "per_session_model_reason": "…",
                   "hands": {"available": true, "trusted": false, "running": false, "socket": false},
                   "eyes": {"model": "/…/vision-model", "source": "automatic"},
                   "workspace": {"configured": true, "connected": false, "client": "bundled", "scopes": []},
                   "dictation": {"installed": true, "local": true}}
}
```

| Field | Values |
|---|---|
| `engine.state` | `offline` · `starting` · `ready` · `error` (the last start failed; `error` has the engine's own last lines) |
| `engine.operation` | the lifecycle action in progress: `start` · `stop` · `switch` · `load` · `unload`, or null |
| `model.state` | `not_loaded` · `loading` · `ready` · `generating` · `unloading` |
| `default` | the global default model — the CLI's `cfg["model"]`, shared by every client |
| `in_use.inflight` | requests on the engine from **all** clients; `assistant_generations` is this service's share |
| `metrics` | only what the engine reported; a key that is absent was not measured |
| `capabilities.hands` | whether the macOS Automation helper is installed, has Accessibility trust and is running; it starts only for a visual step |
| `capabilities.eyes` | the configured or automatically selected installed vision model; Apple Vision OCR remains the fallback |
| `capabilities.workspace` | OAuth client configuration and local Keychain connection state; never tokens |
| `capabilities.dictation` | whether the optional local Whisper runtime is installed; recordings never leave the Mac |

## Commands

All `POST`, body `{"session": "<id>", …}`. `session` identifies a tab; it decides whose generation a cancel
may stop, and whose requests count as "someone else's".

| Path | Effect | Answers |
|---|---|---|
| `/v1/assistant/engine/start` | start the engine on the default model | 202 + snapshot; progress arrives as events |
| `/v1/assistant/engine/stop` | stop the engine process | 200, or 409 `model_in_use` |
| `/v1/assistant/models/load` | load the weights (starts the engine if it is off) | 202 |
| `/v1/assistant/models/unload` | free the weights, keep the engine | 202, or 409 `model_in_use` |
| `/v1/assistant/models/default` `{path, expected_default?}` | change the global default and restart the engine on it | 202, 404 `unknown_model`, 400 `unsupported_model`, 409 `stale` / `model_in_use` |
| `/v1/assistant/generations/<id>/cancel` | stop one generation | 200 `{"cancelled": true\|false}`, 403 `not_yours` |
| `/v1/assistant/agent/visual/execute` | execute a fresh, service-issued visual action through macOS | 200, 409 `stale_visual_action` |
| `/v1/assistant/accessibility/settings` | open macOS Accessibility settings for the helper | 200 |
| `/v1/assistant/workspace/status` | local Google Workspace connection state | 200, no tokens |
| `/v1/assistant/workspace/configure` `{client_id}` | save a Desktop OAuth client id, mode 0600 | 200 |
| `/v1/assistant/workspace/authorize` `{scopes}` | start PKCE loopback OAuth in the default browser | 202 |
| `/v1/assistant/workspace/disconnect` | delete the refresh token from macOS Keychain | 200 |
| `/v1/assistant/workspace/execute` `{ticket, url}` | consume one model-issued named Workspace operation | 200, optional opaque Undo token |
| `/v1/assistant/workspace/undo` `{undo}` | consume one in-memory Undo token | 200 |
| `/v1/assistant/session/forget` | discard a tab session's visual, Workspace and Undo tickets | 200 |
| `/v1/assistant/vision/configure` `{path}` | choose an installed supported vision model for the Conductor's vision role | 200 |
| `/v1/assistant/vision/install` | search the current Hub, download a compatible fitting vision model and assign its role | 200, 502 `vision_install` |
| `/v1/assistant/audio/setup` | install the optional local Whisper/Kokoro environment after explicit UI confirmation | 200 |
| `/v1/assistant/audio/transcribe` `{audio}` | transcribe one bounded audio data URL locally; the temporary file is always erased | 200, 501 `audio_not_installed` |

* **`model_in_use`** (409): other clients have requests in flight — `others` says how many. Repeat with
  `"confirm": true` to proceed; an unload then waits for the running request rather than cutting it off.
* **`stale`** (409): `expected_default` no longer matches; the error carries the current `default` and
  state. This is what makes two clients switching at once safe.
* **`busy`** (409): another lifecycle action is in progress.
* Cancel is idempotent: a generation that has already ended answers `{"cancelled": false}`.

Errors are `{"error": {"code": "…", "message": "…", …}}`.

## Chat

`POST /v1/assistant/chat`

```json
{"session": "safari-12-ab12cd", "task": "ask", "question": "What does it cost?",
 "page": {"title": "…", "url": "https://…", "text": "…"}, "selection": "", "lang": "English",
 "tone": "clearer", "thinking": false, "max_tokens": 1024}
```

`image` (optional): a screenshot of the tab as a `data:image/jpeg|png|webp;base64,…` URL of at most 3 MB; it is
attached to the question for a model with vision, and refused with 400 `no_vision` for any other. `shared`
then says `"screenshot": true`. A request body may be up to 8 MB.

`task`: `ask` (needs `question`) · `summarize` · `explain` · `translate` (`lang`) · `rewrite` (`tone`:
clearer, shorter, more formal, more friendly) · `key_points`.

The service builds the prompt — the client never sends raw messages. Page text and selection are fenced,
anything inside them that could close the fence is neutralised, the system prompt marks them as untrusted
data, and they are cut to the model's context. A refusal before the stream starts is an ordinary error
(503 `engine_offline`, 400 `bad_request`). Otherwise the answer is an event stream:

```
{"type": "start", "id": "<generation>", "model": "…", "shared": {"url": "…", "page_chars": 5000,
   "page_chars_total": 90000, "page_truncated": true, "selection_chars": 0, …}}
{"type": "reasoning", "text": "…"}        only when thinking is on
{"type": "delta", "text": "…"}
{"type": "done", "finish_reason": "stop", "usage": {…}}   |   {"type": "cancelled"}   |   {"type": "error", "message": "…"}
```

`shared` is exactly what reached the model. A client that disconnects stops its generation on the engine.

## The browser agent

`POST /v1/assistant/agent/step` asks the model for the **next single action** on a page. The service does
not touch a browser and decides nothing about safety: the extension checks the proposal against the live
element and asks the person (`docs/safari-assistant/README.md`, "Doing things on a page").

```json
{"session": "safari-12-ab12cd", "goal": "open the pricing page", "request_id": "…uuid…",
 "snapshot": {"url": "https://…", "title": "…", "text": "…", "captcha": false, "truncated": false,
              "elements": [{"ref": "e1", "tag": "a", "role": "link", "name": "Pricing", "href": "https://…/pricing",
                            "type": "", "value": "", "placeholder": "", "disabled": false, "checked": false,
                            "required": false, "options": [], "dup": 0, "sensitive": false, "blocked": false}]},
 "history": [{"summary": "click link “Home”", "result": "the page changed"}]}
```

The snapshot is cut to a known shape and size before it is used (150 elements, bounded strings; unknown
keys dropped; a sensitive field's value is never put in the prompt). The reply is one validated action:

```json
{"action": {"action": "click", "ref": "e1", "why": "…"}, "attempts": 1, "saw_screenshot": false, "described_screenshot": false}
```

`screenshot` (optional, same form as `image` in a chat) goes with the step when the person turns 📷 on or the
goal asks about visible content. A vision model receives the pixels and sets `saw_screenshot`; otherwise local
Apple Vision text and scene labels are added to the prompt and `described_screenshot` is true.
Whether the action may run unasked is the extension's decision (its policy and the mode the person chose), not
the service's.

| `action` | Fields | Validated as |
|---|---|---|
| `click` | `ref` | the ref was in the list shown to the model |
| `fill` | `ref`, `value` | a ref from the list; text of at most 2000 characters |
| `fill_form` | `fields: [{ref, value}]` | 1 – 6 fields, no ref twice |
| `navigate` | `url` | `http(s)://host…`, no credentials |
| `find` | `query` | non-empty; read-only |
| `scroll` | `direction` (`up` / `down`), or `ref` | a ref from the list when given; changes nothing |
| `press` | `key`, `ref` optional | one of Enter, Tab, Escape, ArrowDown/Up/Left/Right, PageDown/Up, Home, End, Backspace, Space |
| `visual_click` | `x`, `y`, `label`, `ticket` | screenshot-relative coordinate, visible label and a one-use 12-second ticket |
| `visual_type` | `x`, `y`, `label`, `value`, `empty`, `ticket` | the target is visibly empty for automatic mode; otherwise the extension asks first |
| `visual_press` | `key`, `ticket` | a permitted key after a visual step |
| `workspace_sheets_write` | spreadsheet id, one A1 cell, value, overwrite flag, ticket | fixed Sheets read/write operation; overwrite remains high risk |
| `workspace_docs_append` | document id, text, ticket | append text to one document |
| `workspace_drive_create` | name, text, ticket | create one Google document through `drive.file` |
| `workspace_gmail_draft` | recipients, subject, body, ticket | create a draft only; never send |
| `workspace_calendar_create` | title, zoned start/end, ticket | create one event no longer than seven days |
| `wait` | `seconds` | clamped to 1 – 10 |
| `ask_user` | `question` | non-empty |
| `done` | `answer` | text |

The model is asked for a JSON object (`response_format`, so the engine constrains the answer); one that is not
usable gets one retry with the reason. Two unusable replies are `422 bad_action` — the step fails rather than
guesses. `request_id` (lowercase letters, digits and dashes, 8 – 40) lets the caller cancel the step with
`POST /v1/assistant/generations/<request_id>/cancel` before the answer arrives (`409 cancelled` for the step).

### Visual actions and macOS input

The extension captures a fresh visible-tab screenshot immediately before it asks the service to consume a ticket.
The original frame is represented by its SHA-256 identity, byte count, dimensions and a small perceptual signature;
a changed tab, URL, origin, viewport or substantially changed frame invalidates the action. The ticket is bound to
the tab session and URL and is deleted after one use or 12 seconds. The local Automation
helper accepts only click, short Unicode text and a fixed key allowlist over an owner-only Unix socket. It verifies
that one supported browser is frontmost, maps coordinates inside its Accessibility `AXWebArea`, and refuses every
other app. It requires macOS Accessibility permission. Neither a page nor the extension has access to the helper
or an endpoint for arbitrary mouse/keyboard events.

### Google Workspace OAuth

The service uses a bundled public Google Desktop OAuth client id when the release builder supplies one (or a custom id for a source build)
with PKCE and a one-time localhost redirect. The verifier and OAuth state exist only in memory, and the
authorization URL is opened by the service rather than returned to the extension. Refresh tokens are written only
to the macOS Keychain; service state contains only `configured`, `connected`, `client` (`bundled` or `custom`) and
scope names. The pairing token, browser storage, HTTP responses and page never contain Google credentials. The
default scopes are Sheets, Docs, Drive (`drive.file`), Gmail compose and Calendar events; Gmail read is deliberately
not requested because the first connector creates drafts but does not read mail.

Workspace action tickets contain only validated named operations and expire after five minutes. Undo records stay
only in service memory behind separate opaque, one-use tokens. Disconnect deletes the refresh token from Keychain;
deleting a browser conversation calls `session/forget` to remove its remaining tickets. Screenshot bytes are never
put into those records or persisted by the service.

## What the engine gained

`POST /v1/engine/sleep` `{"force": false}` — unload the weights, keep the server; 409 `model_in_use` while
requests are in flight unless forced. `POST /v1/engine/wake` — load them now. Both sit behind the engine's
API key like every other POST. Routes that cannot do this (llama.cpp) answer 404, and the service then
reports `capabilities.unload: false`.
