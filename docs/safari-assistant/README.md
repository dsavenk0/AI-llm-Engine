# Browser Assistant

An optional browser extension — Safari, Firefox, Chrome and other Chromium browsers — with a side panel that asks the
model running on your Mac about the page you are reading, or lets it read, click, type and navigate on the page
while you watch, asking before anything that submits, pays or sends. It also controls that model — pick it, load
it, unload it, stop an answer — without LLMBar or a terminal open.
Nothing is sent to a cloud. The CLI and the engine work exactly as before without it.

Why it is built the way it is: [AUDIT.md](AUDIT.md). The wire format: [PROTOCOL.md](PROTOCOL.md).

## Install

**Safari, from LLMBar: Tools ▸ Browser Assistant.** The extension ships inside LLMBar.app, so Safari lists it
like any app's extension. The button sets up the service (a LaunchAgent, so the extension works with LLMBar
closed) and the pairing token in the background, then opens **Safari ▸ Settings ▸ Extensions on the extension**:
tick *LLMBar Assistant*. That tick, and the per-site access Safari asks for the first time the panel opens on a
site, are the only steps — Apple lets no app take them on your behalf. The extension then fetches the token from
LLMBar itself; nothing to paste.

Until LLMBar is signed by an Apple developer, Safari lists the extension only while **Develop ▸ Allow Unsigned
Extensions** is on (the button says so and opens Safari): turn on *Show features for web developers* in Safari ▸
Settings ▸ Advanced, then that menu item, then press the button again. Safari turns it off when it quits. Building
with a free *Apple Development* certificate (sign in to Xcode with your Apple ID, then
`LLMBAR_SIGN_IDENTITY="Apple Development: …" ./build-app.sh --install`) keeps the extension enabled for good.

**Without LLMBar** (a terminal): `./llm assistant setup` does the same setup and prints how to load the files as
a temporary extension (Safari ▸ Settings ▸ Developer ▸ *Add Temporary Extension…*), which Safari forgets when it
quits; paste the token it copied into the extension's settings.

**Chrome, Edge, Brave, Arc, Vivaldi: Tools ▸ Browser Assistant · Chrome** (the row names the browser found on
this Mac). Chrome installs an extension that is not from its store only by your hand, so the button prepares
everything around that: the files in `…/browser-assistant/chromium`, the service, and a native-messaging host that
lets the extension fetch the token by itself (`assistant/native_host.py`, registered for each installed Chromium
browser). It opens the browser's extensions page, shows the folder in Finder and puts its path on the clipboard:
turn on *Developer mode*, *Load unpacked*, ⌘⇧G, paste, Open. The extension's ID is fixed by a key in its manifest,
so the host recognises it wherever the folder is. Each Tools row shows the version it installs (`0.1.0 (c51b127)`:
the release, then the commit; the panel's foot shows the same).

**Firefox: Tools ▸ Browser Assistant · Firefox.** LLMBar prepares `…/browser-assistant/firefox`, registers the
pairing host in `~/Library/Application Support/Mozilla/NativeMessagingHosts`, and opens Firefox's
`about:debugging` page. Choose *This Firefox* → *Load Temporary Add-on…* and select the `manifest.json` in that
folder (its path is copied to the clipboard). The extension opens in Firefox's native sidebar and gets its token
from the host. Firefox removes temporary add-ons when it quits, so load it again after restarting Firefox. A
permanent install requires publishing and signing the add-on through Mozilla Add-ons.



`./llm assistant status` says what is installed and running. `./llm assistant uninstall` removes the
service; deleting the folder removes the files.

## Use

**The side panel.** The extension's toolbar button opens a panel beside the page — Chrome's own side panel in
Chrome and Edge; in Safari (which gives extensions no side panel) and in a Chromium browser without one, the
same panel is docked into the right edge of the tab, the page shifted left to make room, and closed with its ✕.
The panel follows the tab in front: every tab keeps its own conversation.

At the top, the **model chip** opens the engine and model controls:

| Control | What it does | What it does not do |
|---|---|---|
| Model picker | sets the default model — the same one the CLI and LLMBar use — and restarts the engine on it | it is not a per-tab choice: the engine serves one model at a time |
| Start engine | starts the engine and loads the default model | |
| Load model / Unload model | loads the weights, or frees their memory | unloading keeps the engine running; the next request loads the model again |
| ■ (beside the message box) | stops the answer or the task in this tab | never another tab's, the CLI's or LLMBar's |
| Stop engine | quits the engine process, for every client | |

Unloading, switching and stopping are refused while another client has a request in flight, with a
**Do it anyway** button. Memory, context size and speed are shown only when the engine reported them.

Below it, **what the assistant may do**:

| Mode | What happens |
|---|---|
| Ask about the page | a question, or *Summarize*, *Key points*, *Explain the selection*. The page is read when you ask, and the answer says how much of it reached the model |
| Act on the page — ask before every step | the agent proposes one step at a time; nothing is clicked, typed or opened until you approve it |
| Act on the page — ask before anything risky | the agent works while you watch; steps the policy marks low-risk go ahead on their own, and anything that **submits a form, pays, orders, sends, deletes, leaves the site, replaces text you typed, or touches a form with a password or card** stops and waits for your approval — every time |

**📷 Screenshot.** In *Ask*, a question about what is visible automatically attaches a picture of the tab; the
button can attach one to any question. In *Act*, the same button or a goal about visual content sends a picture
with each step. Vision models receive the pixels. A text-only model gets a local Apple Vision description (scene
labels and recognized text), which is useful for text and controls but less reliable for layout and photographs.
Screenshots are off for other requests because each one adds prefill time. The first capture asks the browser for
permission to photograph sites.

**Conversations.** A follow-up question carries the tab's conversation so far (its last exchanges, never the page
text twice). Each tab's conversation is kept in the extension — it comes back when the panel reopens, after a
reload or a browser restart — and **◷** lists past ones to open, rename or delete; **＋** starts a new one. While a
panel is open the service keeps the model loaded.

**Pictures.** A question about images or how the page looks brings a screenshot by itself, with the page's largest
visible pictures cut out of a full-resolution capture. A model with vision sees them; a model without gets them
described in words by Apple Vision (scene labels and the text it reads — good for screenshots and charts, rough
for photographs).

**The context menu** on selected text — Explain, Summarize, Translate, Rewrite — answers in a small panel on the
page. *Summarize this page* is in the page's own context menu.

## Doing things on a page

Describe the goal (“find a vegetarian place nearby rated 4.5+ and order their bowl for pickup at 19:00”). The
agent reads the page, the model proposes one action, and the policy decides whether it may go ahead or must ask.
While it works, **the page shows it**: a glowing frame around the tab, a pill at the top — *LLMBar is working
here* with a **Stop** button — a line at the bottom naming the current step, and an **agent cursor** (a cyan
arrow, never your own pointer) that moves onto each element before it is clicked or typed into, with a ripple on
a click and an outline on a filled field. The frame turns yellow while it waits for you.

| The agent can | Asks first? |
|---|---|
| click a button, link, checkbox… | when it is risky: its label says buy / pay / order / send / delete…, it submits a non-search form, it leaves the site, several elements look alike |
| fill a field (up to six at once); pick an option in a list | when it would replace text already there |
| press a key: Enter, Tab, Escape, arrows, Page Up/Down, Home, End, Backspace, Space | Enter in a form that is not a search box; any key in a form with a password or card |
| open an address | when it leaves the site, is unencrypted or is on your network |
| scroll the page (or the app's scrolling area), bring an element into view, wait for the page | never — these change nothing |
| look for an element that is not in the list | never — reads only |
| ask you something | stops until you answer (also after you decline a step) |

In *ask before every step* every one of these that changes something asks. A search form (a form marked as
search, holding a search field, or posting to a search address) only looks something up, so submitting it is not
counted as sending.

What it will never do, whatever the model says: type into a **password, card number or one-time-code** field;
touch a **CAPTCHA**; open `javascript:`, `data:`, `file:` or an address with a login in it; steer the browser
to this assistant or the engine; run code, a selector or markup supplied by the model or the page — it passes
the page agent short references (“e12”) and nothing else.

How a step is kept honest:

* The approval card is built from the **live element**, not from the model's description, so a model (or a
  page) cannot call a “Delete account” button “Details”.
* A reference works only for the page reading that issued it. If the page was read again, the address changed,
  the element left, changed its name or role, is hidden, disabled or covered by something else, the step is
  refused and the agent looks again.
* A step runs only on the tab and the address it was proposed for; if the page moves while you decide,
  nothing happens.
* At most 8 steps per task when every step asks, 20 when the agent acts on its own; two failed steps in a row
  end it. **Stop** — in the panel or on the page — or closing the tab ends it at once, including a question the
  model is still being asked. The page's Stop works only on a real click and can only stop.
* Text from the page — names, labels, what is on screen — reaches the model as data between fences, with an
  instruction to ignore anything in it that looks like an order. This lowers the risk of a page steering the
  model; it does not remove it, which is why the risky steps always wait for you.

The site must be one the extension may read: the page you started on (the toolbar button grants it for that
tab), and any site the task moves to once you press **Allow** in the panel. **What the agent did**, in the
extension's settings, keeps the last 200 steps — what was proposed, approved (by you or on its own), refused and
done, and on which site, never page text and never a typed value.

**Apps that draw on a canvas** (Google Sheets and Docs, Google Keep, Figma, maps) have no elements to list where
their content is. With a vision model and 📷 on, the agent can see their controls. When it cannot find a safe DOM
target it proposes a visual action: LLMBar uses macOS Accessibility to move the real pointer, click or type in the
active browser only. The agent card identifies the current route as **DOM**, **vision**, **macOS** or **Workspace**.
The Hands banner reports whether Accessibility is enabled and opens the correct System Settings page. A visible
empty selected Sheets cell may be filled automatically in the low-risk mode;
an existing or uncertain cell waits for approval. Keep notes follow the same path: create, type, then verify the
new card on a fresh screenshot.

**Dictation.** Focus an ordinary text field on the page, then press the microphone in the assistant composer.
The extension remembers that exact field before recording, sends a bounded clip only to the loopback service,
and local Whisper inserts the recognised text at the saved caret. The temporary recording is always removed.
Password, card, one-time-code, CAPTCHA, disabled and read-only fields are refused. If the target disappears while
speech is recognised, the text is placed in the assistant composer instead of being lost. The first click can
install the optional audio environment after confirmation; Whisper weights download on first use.

When no compatible vision model is installed, **Find and install eyes** searches current Hub metadata, chooses a
supported model that fits the Mac, downloads it after confirmation and assigns the Conductor's `vision` role.
Until then, Apple Vision OCR remains available. Image requests follow the same on-demand pattern: the model can
express the intent before mflux exists, while installation and model download still require visible approval.

**Google Workspace.** Release builders can embed the public Desktop OAuth client ID with
`LLMBAR_GOOGLE_CLIENT_ID`. Until a build includes that ID, the OAuth controls stay hidden and the agent continues
through DOM, vision and Accessibility without asking for Auth. The connector code remains in place and appears
automatically when configured. Sign-in uses a one-time localhost callback and PKCE. The refresh token stays in macOS Keychain; the
extension, web pages and model never receive it, and the settings page automatically confirms when Keychain has
the connection. The first connector release can write one Sheets cell, append Docs text, create a Drive document,
create a Gmail draft (never send it) and create a Calendar event. Each is a fixed named operation rather than an
arbitrary Google request and offers an in-memory, one-use Undo. Gmail read access is not requested. Google Keep
continues through the browser route because it has no equivalent public Workspace API.

## What it can and cannot see

* It reads a page only when you ask about it, and only the tab you are looking at (`activeTab`). It has no
  standing access to any site, and runs nothing on pages you have not asked about.
* Page text goes to the service on `127.0.0.1` and from there to the local model. Nothing else.
* The service refuses web pages, requests not addressed to loopback, and anything without the token.
* Page content is given to the model as data, with an instruction not to obey text found inside it. That
  lowers the risk of a page steering the model; it does not remove it — which is why a risky step always
  waits for you, in either acting mode.
* A screenshot is taken only with 📷 on, of the visible part of the tab in front, scaled to at most 1280 pixels
  wide, and goes only to a model that can see — the service drops it for any other.
* Nothing is logged: not page text, not questions, not answers, not screenshots. Conversation storage contains
  only question/answer text. Screenshot bytes live for one request; deleting a conversation also tells the local
  service to discard that tab session's remaining visual, Workspace and undo tickets.

## Feature matrix

| Feature | State |
|---|---|
| Assistant service: state, events, model and engine controls, chat proxy, cancellation | implemented, tested (43 tests, fake engine over real HTTP) |
| Engine: `POST /v1/engine/sleep`, `/v1/engine/wake` | implemented, tested on the real server with a synthetic model |
| Extension: side panel (Chrome's own; docked into the tab in Safari and elsewhere), settings, context menu | implemented; type-checked and bundled; the panel rendered headless with a stand-in browser API, light and dark |
| **Extension running inside Safari** | **not tested** — no browser was driven; first run is on your Mac |
| Selected-text actions, page Q&A, streaming, cancel | implemented (see the line above) |
| Browser actions: click, fill, fill form, navigate, find, scroll, key presses, wait; approval card; audit log; Undo for field fills | implemented; type-checked and covered by page-agent, runner, policy and service tests. **Not driven in a real browser in this checkout** |
| Acting on its own with approval before risky steps | implemented, tested as above |
| Working-here frame, Stop on the page, agent cursor | implemented; rendered headless on a test page |
| Screenshots for vision models (questions and agent steps) | implemented for Safari, Firefox and Chromium; text-only models get a local Apple Vision description |
| Microphone dictation into a selected page field | implemented with local Whisper, sensitive-field guards, 30-second/5-MB bounds, immediate temporary-file cleanup and composer fallback |
| On-demand image / vision preparation | implemented with explicit approval; image requests can prepare mflux, and settings can discover a compatible fitting vision model |
| Per-site access for the agent after it leaves the page it started on | implemented through `permissions.request` from the panel; **whether Safari grants it that way is unverified** — if not, allow the site in Safari ▸ Settings ▸ Websites |
| Sites with the page in iframes, shadow DOM, canvas | not covered by the top-document element list; the vision route can instead ground a short-lived native macOS action in the fresh frame |
| Model picker, load / unload / stop, conflict confirmation | implemented (same) |
| LLMBar ▸ Tools ▸ Browser Assistant | implemented; the complete app bundle builds successfully |
| Update of the extension's files when LLMBar updates | implemented, tested |
| Chrome / Chromium build | builds from the same source (`dist/chromium`), side panel included; **never loaded in Chrome** |
| Firefox | built from the shared source with a native `sidebar_action`, Gecko extension ID and Firefox native-messaging registration; temporary installation through `about:debugging`; **not yet exercised in a live Firefox** |
| Safari extension inside LLMBar.app; Tools ▸ Browser Assistant opens Safari on it | implemented; built and registered with macOS (`pluginkit` lists it); ad-hoc signed, so Safari needs *Allow Unsigned Extensions* until it is signed with an Apple Development / Developer ID certificate. **Not yet enabled in Safari by its author** |
| Native messaging: the extension gets the token from LLMBar | implemented (reads the token file only); **untested in Safari** |
| Vision-guided canvas apps and real macOS actions | implemented: one-use visual tickets, fresh screenshots, active-browser check and Accessibility helper; requires manual verification in Safari, Chrome and Firefox |
| Google Keep / Sheets / Docs / Drive / Gmail / Calendar web UI | implemented through the hybrid DOM + vision + macOS route; high-risk changes still ask first |
| Google Workspace OAuth | implemented but dormant until the release embeds its public client ID; Desktop OAuth PKCE, one-time state and macOS Keychain token storage remain in place |
| Google Workspace named operations and Undo | implemented for Sheets cell write, Docs append, Drive document create, Gmail draft and Calendar event; mocked API tests cover routing, validation and reversal |
| Per-session model override | not possible on this engine: one model at a time. The service says so (`per_session_model: false`) |

## Packaging and updates

**Inside LLMBar.app (Safari).** The extension is part of the app: a new LLMBar brings a new extension, and Safari
picks it up with the app — nothing to re-add. The service still updates itself: when it notices its own code
changed on disk it exits when nobody is mid-answer, and launchd starts the new one.

**A temporary or unpacked extension (Safari without LLMBar, Firefox, Chrome).** Its files live in Application Support,
so an update that replaces the app cannot leave the browser pointing at files that are gone. On start the service
compares the shipped extension with that copy and replaces it; the side panel then says a newer version is
installed and where — reload it in the browser (Safari: add the folder again; Chrome: the reload button in
chrome://extensions).

The extension and the service also compare protocol versions, and each says plainly when the other is
too old, rather than failing on a missing field.

**Signing.** Safari keeps an extension from an app signed by an Apple developer; an ad-hoc signed one (this
build) only while *Allow Unsigned Extensions* is on. A free Apple Development certificate is enough on your own
Mac; distributing the app to others needs a Developer ID and notarization (`LLMBAR_SIGN_IDENTITY`, see
`build-app.sh`).

## Build from source

```sh
cd browser-assistant
npm install          # two tools: typescript, esbuild
npm run check        # type-check
npm test             # unit tests
npm run build        # dist/safari, dist/chromium and dist/firefox
```

`./build-app.sh` builds the extension into the app when Node.js is installed, and skips it otherwise.
