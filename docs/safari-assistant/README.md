# Browser Assistant

An optional Safari extension that asks the model running on your Mac about the page you are reading, and
controls that model — pick it, load it, unload it, stop an answer — without LLMBar or a terminal open.
Nothing is sent to a cloud. The CLI and the engine work exactly as before without it.

Why it is built the way it is: [AUDIT.md](AUDIT.md). The wire format: [PROTOCOL.md](PROTOCOL.md).

## Install

From LLMBar: **Tools ▸ Browser Assistant**. From a terminal: `./llm assistant setup`. Both do the same
thing, and can be run again at any time:

1. put the extension's files in `~/Library/Application Support/AI-llm-Engine/browser-assistant/safari`;
2. install the assistant service as a LaunchAgent, so it is up after login;
3. copy the pairing token to the clipboard.

Then three steps that Safari reserves for you — no app can take them on your behalf:

1. Safari ▸ Settings ▸ Advanced: turn on *Show features for web developers*.
2. Safari ▸ Settings ▸ Developer: *Add Temporary Extension…* and choose the folder from step 1 above.
3. The extension's settings open: paste the token and press **Connect**.

**Safari forgets a temporary extension when it quits**; repeat step 2 after restarting Safari. The token
and settings are kept. A permanent install needs the extension inside a signed app built with Xcode — see
[Packaging](#packaging-and-updates).

`./llm assistant status` says what is installed and running. `./llm assistant uninstall` removes the
service; deleting the folder removes the files.

## Use

**The toolbar popup** shows the engine and the model, with separate controls:

| Control | What it does | What it does not do |
|---|---|---|
| Model picker | sets the default model — the same one the CLI and LLMBar use — and restarts the engine on it | it is not a per-tab choice: the engine serves one model at a time |
| Start engine | starts the engine and loads the default model | |
| Load model / Unload model | loads the weights, or frees their memory | unloading keeps the engine running; the next request loads the model again |
| Stop generation | stops the answer you are reading | never another tab's, the CLI's or LLMBar's |
| Stop engine | quits the engine process, for every client | |

Unloading, switching and stopping are refused while another client has a request in flight, with a
**Do it anyway** button. Memory, context size and speed are shown only when the engine reported them.

Ask a question about the page, or press **Summarize**, **Key points** or **Explain**. The page is read when
you ask, not before, and the answer says how much of it was sent to the model.

**The context menu** on selected text — Explain, Summarize, Translate, Rewrite — answers in a small panel
on the page. *Summarize this page* is in the page's own context menu.

## Doing things on a page: click, fill, navigate

Type a goal in the popup (“open the pricing page and find the cheapest plan”) and press **Do it**. The agent
works one step at a time: it reads the page, the model proposes one action, and a card in the panel on the page
says exactly what will happen. **Nothing is clicked, typed or opened until you press Approve** — every step asks.

| The agent can | Before it does |
|---|---|
| click a button, link, checkbox… | shows the element as the *page* names it, where a link goes, and why the click is risky |
| fill a field (up to six at once) | shows each field and the exact text; says when it replaces text that is there |
| open an address | shows the address and whether it leaves this site, is unencrypted or is on your network |
| look for an element that is not in the list | reads only — no question asked |
| ask you something | stops until you answer (also after you decline a step) |

What it will never do, whatever the model says: type into a **password, card number or one-time-code** field;
touch a **CAPTCHA**; open `javascript:`, `data:`, `file:` or an address with a login in it; steer the browser
to this assistant or the engine; run code, a selector or markup supplied by the model or the page — it passes
the page agent short references (“e12”) and nothing else.

How a step is kept honest:

* The card is built from the **live element**, not from the model's description, so a model (or a page)
  cannot call a “Delete account” button “Details”.
* A reference works only for the page reading that issued it. If the page was read again, the address changed,
  the element left, changed its name or role, is hidden, disabled or covered by something else, the step is
  refused and the agent looks again.
* A step runs only on the tab and the address it was proposed for; if the page moves while you decide,
  nothing happens.
* At most 8 steps per task, and two failed steps in a row end it. **Stop** (or closing the panel, or closing
  the tab) ends it at once, including a question the model is still being asked.
* Text from the page — names, labels, what is on screen — reaches the model as data between fences, with an
  instruction to ignore anything in it that looks like an order. This lowers the risk of a page steering the
  model; it does not remove it, which is why every step is approved by you.

The site must be one the extension may read: the page you started on (the extension's toolbar button grants
it for that tab), and any site the task moves to once you press **Allow** in the popup. **What the agent did**,
in the extension's settings, keeps the last 200 steps — what was proposed, approved, refused and done, and on
which site, never page text and never a typed value.

## What it can and cannot see

* It reads a page only when you ask about it, and only the tab you are looking at (`activeTab`). It has no
  standing access to any site, and runs nothing on pages you have not asked about.
* Page text goes to the service on `127.0.0.1` and from there to the local model. Nothing else.
* The service refuses web pages, requests not addressed to loopback, and anything without the token.
* Page content is given to the model as data, with an instruction not to obey text found inside it. That
  lowers the risk of a page steering the model; it does not remove it — and in this version the model can
  only answer, not act, so a misled answer is the worst case.
* Nothing is logged: not page text, not questions, not answers.

## Feature matrix

| Feature | State |
|---|---|
| Assistant service: state, events, model and engine controls, chat proxy, cancellation | implemented, tested (43 tests, fake engine over real HTTP) |
| Engine: `POST /v1/engine/sleep`, `/v1/engine/wake` | implemented, tested on the real server with a synthetic model |
| Extension: popup, settings, context menu, in-page panel | implemented; type-checked and bundled; logic unit-tested (14 tests) |
| **Extension running inside Safari** | **not tested** — no browser was driven; first run is on your Mac |
| Selected-text actions, page Q&A, streaming, cancel | implemented (see the line above) |
| Browser actions: click, fill, fill form, navigate, find; approval card; audit log | implemented; **tested without a browser** — the page agent in jsdom (fake layout), the conductor with a fake page, tab and service, the policy rules, the service endpoint over HTTP (51 + 18 tests). **Not driven in Safari or Chrome** |
| Per-site access for the agent after it leaves the page it started on | implemented through `permissions.request` from the popup; **whether Safari grants it that way is unverified** — if not, the popup says to allow the site in Safari ▸ Settings ▸ Websites |
| Sites with the page in iframes, shadow DOM, canvas | **not covered**: the agent sees the top document's ordinary elements only |
| Model picker, load / unload / stop, conflict confirmation | implemented (same) |
| LLMBar ▸ Tools ▸ Browser Assistant | implemented; the app type-checks, was not rebuilt or run |
| Update of the extension's files when LLMBar updates | implemented, tested |
| Chrome / Chromium build | builds from the same source (`dist/chromium`); **never loaded in Chrome** |
| Permanent Safari install (containing app, signing) | planned — needs Xcode and a Developer ID |
| Native messaging | planned, with the containing app |
| Google Sheets, Excel Online | planned (Phases 3 – 4) — need OAuth client registrations |
| Email assistant (Gmail, Outlook Web) | planned (Phase 5) |
| Mailbox access through Gmail API / Microsoft Graph | planned (Phase 6) — needs OAuth client registrations |
| Per-session model override | not possible on this engine: one model at a time. The service says so (`per_session_model: false`) |

## Packaging and updates

**Today (temporary extension).** The extension lives in Application Support, not inside LLMBar.app, so an
update that replaces or moves the app cannot leave Safari pointing at files that are gone. When a new
LLMBar — or a `git pull` and `./llm assistant setup` — brings a newer extension:

1. the running service notices its own code changed on disk and exits when nobody is mid-answer; launchd
   starts the new one;
2. on start the service compares the shipped extension with the installed copy and replaces the copy;
3. the next time the popup opens, the old extension — still loaded in Safari — is told a newer version is
   installed and where; restarting Safari and adding the folder again loads it.

The extension and the service also compare protocol versions, and each says plainly when the other is
too old, rather than failing on a missing field.

**With a signed app (planned).** A Safari web extension is distributed inside a macOS app. The natural home
is LLMBar.app itself: the extension becomes an app extension in `Contents/PlugIns`, next to the widget, and
is enabled once in Safari ▸ Settings ▸ Extensions. Then Safari updates the extension whenever the app is
updated — nothing to re-add, no developer setting. The app does not have to be *running* for the extension
to work. This needs Xcode (to build the extension target) and a Developer ID (Safari loads only signed
extensions outside developer mode); neither is on the machine this was written on.

## Build from source

```sh
cd browser-assistant
npm install          # two tools: typescript, esbuild
npm run check        # type-check
npm test             # unit tests
npm run build        # dist/safari and dist/chromium
```

`./build-app.sh` builds the extension into the app when Node.js is installed, and skips it otherwise.
