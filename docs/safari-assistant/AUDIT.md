# Browser Assistant — audit and decisions

Phase 0 of the browser assistant: what the repository and the platform actually offer, what was taken from
the reference project, and the smallest change that delivers a Safari extension controlling the local
engine. Everything below was checked in this repository or against official documentation; what could not
be checked is marked.

## 1. What the engine already offers

| Need | What exists | Where |
|---|---|---|
| Chat with streaming | `POST /v1/chat/completions` (OpenAI wire format, SSE), also Ollama and Anthropic formats | `engine/server.py` |
| Cancel a generation | closing the HTTP connection sets the request's `cancelled` flag; the decode loop checks it every token | `engine/server.py` (`should_stop`) |
| Engine status | `GET /health`: model, `awake`/`sleeping`, `busy`, `inflight`, context size, memory figures, last request's speed | `engine/server.py` |
| Unload weights | only automatically, after `idle_sleep` minutes, or before an image is generated — **no endpoint** | `Engine.sleep` |
| Load weights | implicitly, on the next request after sleep — **no endpoint** | `Engine.ensure_awake` |
| Several models at once | **no** — one engine process serves one model; switching is a restart with another `--model` | `llm.py` `choose_model` → `restart` |
| Installed models | `scan_models(cfg)`: name, path, size, format, quantization, route (native / mlx-lm / GGUF / llama.cpp), whether it is supported and fits | `llm.py` |
| Global default model | `cfg["model"]` in `<data>/.llm/config.json` — already shared by the CLI, LLMBar, the editor and voice, which all go through `./llm` | `llm.py` |
| Start / stop the engine | `start(cfg)`, `stop(cfg)`, `restart(cfg)`; a pid file prevents two copies of a model loading at once | `llm.py` |
| Run without the app or a terminal | `./llm service install` — a LaunchAgent that keeps the engine up after login | `llm.py` |
| Authentication | optional API key (`cfg["api_key"]`, off by default) | `engine/server.py` `_auth` |
| CORS | `Access-Control-Allow-Origin: *` on every response | `engine/server.py` |

Two facts shape the design more than any other:

* **The engine cannot be reached when it is not running, and cannot switch models while it is.** Something
  that outlives the engine has to start it, stop it and restart it with another model. The CLI does that
  today, but only while someone runs it.
* **The engine answers any web page.** CORS is open and there is no key by default, so a script on any site
  can already call `http://127.0.0.1:8090`. That is the engine's existing posture and is not changed here;
  the assistant's own service does not inherit it (section 4).

## 2. What the platform offers (Safari 26.4, macOS 26.4.1)

| Question | Answer | Source |
|---|---|---|
| Load an extension without Xcode | yes — Safari ▸ Settings ▸ Developer ▸ *Add Temporary Extension…* takes a folder holding `manifest.json`; it is removed when Safari quits, and needs *Allow unsigned extensions* | [Running your Safari web extension](https://developer.apple.com/documentation/safariservices/running-your-safari-web-extension) |
| Install permanently, distribute | needs a containing app built with Xcode and signed | same |
| Native messaging (extension ↔ app) | needs the Xcode project; the handler runs in a **sandboxed** app extension, which cannot start processes or read the engine's config | [Messaging between the app and JavaScript](https://developer.apple.com/documentation/safariservices/messaging-between-the-app-and-javascript-in-a-safari-web-extension) |
| This machine | Command Line Tools only (no Xcode), no signing identity, Node 22 and npm | checked |

So the first deliverable is an extension loaded as a temporary extension, talking to a local service over
loopback HTTP. A containing app, permanent installation and native messaging wait for Xcode and a
Developer ID — they are packaging, not function: the extension does not depend on them.

## 3. The reference project: jose-donato/ollama-reply

Read at its current `HEAD` (MIT licence). No code is copied; the table records what was learned.

| Pattern | Verdict | Why |
|---|---|---|
| Model calls from the background script, not the content script | **adopted** | the page cannot see the request, and one place holds the connection |
| UI injected into pages inside a shadow root | **adopted** (closed root) | the page's CSS cannot restyle it and its scripts cannot read it |
| `browser.*` through a polyfill over `chrome.*` | **adopted as a facade** | one small module instead of a dependency |
| `host_permissions: ["*://*/*"]` | rejected | full access to every site; `activeTab` on a user gesture is enough for this phase |
| `OLLAMA_ORIGINS=*` so the extension may call the server | rejected | it lets every web page call the model too |
| No authentication | rejected | any local process or page could drive the model |
| Non-streaming request, model name hard-coded | rejected | the user waits blind, and only one model works |
| Content script on fixed sites (twitter.com) | deferred | site adapters are Phase 2+ |

## 4. Design

```text
Safari extension (TypeScript)                 browser-assistant/
  popup: model picker, engine controls, ask about the page
  context menu → in-page panel (closed shadow root): selection actions
  background: the only code that talks to the service; streams to the UI over a port
        │  loopback HTTP, bearer token, extension origin only
        ▼
Assistant service (Python, stdlib)            assistant/service.py — ./llm assistant serve
  state snapshot + versioned event stream (models, default, engine, model, own generations)
  start / stop the engine, switch the global default, load / unload weights
  chat proxy with per-session cancellation
        │  the engine's existing API, plus two new endpoints
        ▼
AI-llm-Engine (unchanged except: POST /v1/engine/sleep, POST /v1/engine/wake)
```

Decisions, each with its reason:

1. **A separate service, not more routes on the engine.** Only something that outlives the engine can
   start it or switch its model. It reuses `llm.py`'s own `scan_models`, `start`, `stop` and config — the
   same code the CLI and LLMBar run — so there is one model registry and one global default. It can run
   from a LaunchAgent (`./llm assistant install`), so neither LLMBar nor a terminal is needed at run time.
2. **Loopback HTTP now, native messaging later.** Native messaging needs Xcode and runs sandboxed. HTTP on
   `127.0.0.1` works from a temporary extension today, and from Chrome unchanged.
3. **The service trusts nothing a web page can send.** It listens on `127.0.0.1` only; refuses a `Host`
   that is not loopback (DNS rebinding); refuses any request whose `Origin` is a web origin; answers CORS
   only for extension origins; and requires a bearer token generated on first run and stored `0600`. The
   user pastes the token into the extension once (`./llm assistant token` copies it).
4. **Three different stops.** *Stop generation* cancels only the caller's own generation. *Unload* frees
   the weights but keeps the engine up; it is refused while other clients have requests in flight, unless
   the user confirms. *Stop engine* ends the process, with the same conflict rule.
5. **One model at a time, said plainly.** The engine serves one model, so a per-session model override
   cannot run beside the global default. The service reports `per_session_model: false` with the reason;
   switching the model is a global action, refused while others are generating unless confirmed.
6. **Least privilege in the browser.** Permissions: `activeTab`, `scripting`, `contextMenus`, `storage`, and
   host access to `http://127.0.0.1/*` only. Page content is read only on a user gesture, only from the
   active tab, and the UI shows how much of it was sent.
7. **Page text is data.** It is sent inside delimiters, under a system prompt that tells the model it is
   untrusted content, never instructions.
8. **TypeScript, no UI framework.** Two dev dependencies — `typescript` for checking, `esbuild` for
   bundling each entry into one file. The popup is small enough for plain DOM code.

## 5. What this phase does not do

Browser actions (click, fill), Google Sheets, Excel Online, the email assistant, OAuth, a containing app and
Chrome testing are later phases (`docs/safari-assistant/README.md` keeps the matrix). Each needs something
this phase does not have: action tools need the approval and element-reference machinery; Sheets, Excel and
mailbox access need OAuth client registrations from Google and Microsoft, which only the project owner can
create; the containing app needs Xcode and a signing identity.

## 6. Risks

| Risk | Handling |
|---|---|
| Safari's handling of extension `fetch` to loopback (CORS, preflight) differs from Chrome's | the service answers preflight for extension origins; verified with requests that carry those headers, not in Safari itself |
| A temporary extension disappears when Safari quits | documented; permanent install needs the containing app |
| The token is visible to anything that can read the user's files | the same trust as the engine's own config; the token only reaches the model and the engine controls, never the file system |
| `start` quits vMLX / Ollama when `auto_free` is on | existing behaviour of every client; the extension says the engine is starting and the setting is the user's |
| Two clients switch the model at once | switching takes a version (`if_version`); a stale one gets 409 with the current state |
