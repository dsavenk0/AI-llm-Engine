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
  "capabilities": {"unload": true, "per_session_model": false, "per_session_model_reason": "…"}
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

## What the engine gained

`POST /v1/engine/sleep` `{"force": false}` — unload the weights, keep the server; 409 `model_in_use` while
requests are in flight unless forced. `POST /v1/engine/wake` — load them now. Both sit behind the engine's
API key like every other POST. Routes that cannot do this (llama.cpp) answer 404, and the service then
reports `capabilities.unload: false`.
