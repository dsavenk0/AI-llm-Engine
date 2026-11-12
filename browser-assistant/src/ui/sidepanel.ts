// The side panel (Chrome): one place beside the page to ask about it, or to let the model act on it while you
// watch. It follows the active tab of its window — each tab keeps its own conversation — and, like every other
// page of the extension, holds no token: everything goes through the background script.

import { api, browserName } from "../browser-adapters/api";
import { AgentView, Autonomy } from "../core/agent-types";
import { activityLabel } from "../core/activity";
import { originOf } from "../core/policy";
import { Command, PROTOCOL, ServiceError, Snapshot, Task, UiEvent, UiRequest } from "../core/protocol";
import { isActive, renderAgent } from "./agent-view";
import { Answer, el, metricsText, modelLabel, problemText, statusLine, updateNotice } from "./view";

// The background script is a service worker that Chrome puts to sleep after ~30 s without events — while you read,
// switch tabs or minimise the window. That breaks this port; the panel then reconnects on its own and asks for
// the state and the tab again, keeping the conversation on screen, instead of falling silent.
let port = connect();
function connect() {
  const p = api.runtime.connect({ name: "sidepanel" });
  p.onMessage.addListener((ev: UiEvent) => onEvent(ev));
  p.onDisconnect.addListener(() => {
    setTimeout(() => {
      port = connect();
      port.postMessage({ kind: "subscribe" });
      if (tabId !== null) port.postMessage({ kind: "watch", tabId });
    }, 250);
  });
  return p;
}
const post = (m: UiRequest): void => {
  try {
    port.postMessage(m);
  } catch {
    port = connect();                         // disconnected in between: once more on a fresh port
    port.postMessage({ kind: "subscribe" });
    port.postMessage(m);
  }
};
const $ = <T extends HTMLElement>(id: string): T => document.getElementById(id) as T;

const threads = $("threads"), input = $<HTMLTextAreaElement>("input"), send = $<HTMLButtonElement>("send");
const stop = $<HTMLButtonElement>("stop"), mode = $<HTMLSelectElement>("mode"), tabLine = $("tab");
const chip = $<HTMLButtonElement>("model-chip"), sheet = $("model-sheet"), picker = $<HTMLSelectElement>("model");
const controls = $("controls"), note = $("note"), see = $<HTMLButtonElement>("see");
// in Safari (and a Chromium browser without a side panel) this page is docked into the tab in a frame
const EMBEDDED = new URLSearchParams(location.search).has("docked");
let lastBubble: HTMLElement | null = null;
const MANIFEST = (api.runtime as unknown as { getManifest(): { version: string; version_name?: string } }).getManifest();
const MINE = MANIFEST.version;

let state: Snapshot | null = null;
let tabId: number | null = null;
let windowId: number | undefined;

// ───────────── one conversation per tab ─────────────
interface Thread { root: HTMLElement; empty: HTMLElement; agents: Map<string, HTMLElement>; view: AgentView | null }
const byTab = new Map<number, Thread>();
let answer: { a: Answer; tab: number } | null = null;           // the one answer streaming now, and whose it is

const DEMOS: Record<"ask" | "act", { label: string; text: string; task?: Task }[]> = {
  ask: [
    { label: "Summarize this page", text: "", task: "summarize" },
    { label: "Key points", text: "", task: "key_points" },
    { label: "Explain the selection", text: "", task: "explain" },
  ],
  act: [
    { label: "Find a restaurant", text: "Search this site for a well-rated restaurant near me and open its page" },
    { label: "Fill this form", text: "Fill this form with the details I give you, but do not submit it" },
    { label: "Compare prices", text: "Open the pricing page and tell me which plan is cheapest per month" },
  ],
};

function thread(id: number): Thread {
  let t = byTab.get(id);
  if (t) return t;
  const empty = el("div", { class: "empty" });
  const root = el("div", { class: "thread" }, empty);
  t = { root, empty, agents: new Map(), view: null };
  byTab.set(id, t);
  threads.append(root);
  renderEmpty(t);
  return t;
}

function renderEmpty(t: Thread): void {
  const acting = mode.value !== "ask";
  t.empty.replaceChildren(
    el("div", { class: "empty-mark" }, "AI>_"),
    el("h1", {}, acting ? "What should I do on this page?" : "What do you want to know about this page?"),
    el("p", { class: "dim" }, acting
      ? mode.value === "act-auto"
        ? "I read the page and act on it while you watch. Anything that submits, pays, orders, sends, deletes or leaves the site waits for your approval."
        : "I read the page and propose one step at a time; nothing happens until you approve it."
      : "The page text goes to the model on this Mac, nowhere else."),
  );
  const list = el("div", { class: "demos" });
  for (const d of DEMOS[acting ? "act" : "ask"]) {
    const b = el("button", { type: "button", class: "demo", textContent: d.label });
    b.addEventListener("click", () => {
      if (d.task) withAccess(() => whenReady(() => void ask(d.task!)));
      else { input.value = d.text; input.focus(); autosize(); }
    });
    list.append(b);
  }
  t.empty.append(list);
}

function show(id: number): void {
  tabId = id;
  learnUrl(id);
  const t = thread(id);
  for (const x of byTab.values()) x.root.hidden = x !== t;
  post({ kind: "watch", tabId: id });
  syncButtons();
}

function bubble(t: Thread, text: string, kind: string): void {
  t.empty.hidden = true;
  lastBubble = el("div", { class: "msg user" }, el("div", { class: "kind dim small" }, kind), text);
  t.root.append(lastBubble);
  scroll();
}

function scroll(): void {
  threads.scrollTop = threads.scrollHeight;
}

// ───────────── the engine starts itself ─────────────
// A question or a task sent while the engine is off starts it, then runs as soon as the model is ready — as the
// voice chat and the editor do. ■ cancels the wait.
let pendingRun: (() => void) | null = null;
let pendingNote: HTMLElement | null = null;

function whenReady(run: () => void): void {
  const e = state?.engine.state;
  if (e === "ready") { run(); return; }
  pendingRun = run;
  if (tabId !== null) {
    const t = thread(tabId);
    t.empty.hidden = true;
    pendingNote = el("div", { class: "msg note dim" }, "Starting the model…");
    t.root.append(pendingNote);
    scroll();
  }
  if (e === "offline" || e === "error") post({ kind: "command", command: { op: "start" }, confirm: false });
  syncButtons();
}

function settlePending(): void {
  if (!pendingRun || !state) return;
  if (state.engine.state === "ready") {
    const run = pendingRun;
    pendingRun = null;
    pendingNote?.remove();
    pendingNote = null;
    run();
  } else if (state.engine.state === "error") {
    pendingRun = null;
    if (pendingNote) { pendingNote.textContent = `The model did not start: ${state.engine.error || "see the model menu"}`; pendingNote.className = "msg note bad"; }
    pendingNote = null;
    syncButtons();
  }
}

// ───────────── access to the site ─────────────
// Chrome grants nothing when the side panel opens (unlike a popup, opening it is not "activeTab"), so the panel
// asks for the site in front at the moment the person sends — a click, the gesture the browser requires — and
// the browser shows its own Allow prompt once per site. Called synchronously from the click, before any await.
let currentUrl = "";

/** The panel asks the browser itself which address the tab shows (it needs it before the click that asks for
 *  access, and an answer relayed through the background could arrive late or empty). */
function learnUrl(id: number): void {
  api.tabs.get(id).then((t) => { if (id === tabId) currentUrl = t.url || ""; }).catch(() => undefined);
}

function withAccess(go: () => void): void {
  const origin = originOf(currentUrl);
  if (!origin) { go(); return; }                          // a browser page: the background says it cannot be read
  let request: Promise<boolean>;
  try {
    request = api.permissions.request({ origins: [origin + "/*"] });
  } catch {
    go();                                                 // no permissions API here (Safari decides by itself)
    return;
  }
  request.then((ok) => {
    if (ok) go();
    else if (tabId !== null) {
      const t = thread(tabId);
      t.empty.hidden = true;
      t.root.append(el("div", { class: "msg note bad" },
        `LLMBar reads ${origin.replace(/^https?:\/\//, "")} only with your permission. Send again and choose Allow.`));
      scroll();
    }
  }).catch(() => go());
}

// ───────────── asking ─────────────
async function ask(task: Task, question?: string): Promise<void> {
  if (tabId === null || busy()) return;
  const t = thread(tabId);
  const labels: Partial<Record<Task, string>> = { summarize: "Summarize this page", key_points: "Key points",
    explain: "Explain the selection" };
  bubble(t, question || labels[task] || task, "Ask");
  const a = new Answer();
  a.begin();
  t.root.append(el("div", { class: "msg assistant" }, a.root));
  answer = { a, tab: tabId };
  const lang = navigator.language.startsWith("ru") ? "Russian" : "English";
  post({ kind: "ask", tabId, task, question, lang, thinking: $<HTMLInputElement>("thinking").checked,
         screenshot: seeing() });
  syncButtons();
  scroll();
}

// ───────────── acting ─────────────
function act(goal: string): void {
  if (tabId === null || busy()) return;
  const autonomy: Autonomy = mode.value === "act-auto" ? "auto" : "ask";
  bubble(thread(tabId), goal, (autonomy === "auto" ? "Act · asks before anything risky" : "Act · asks before every step")
    + (seeing() ? " · sees the page" : ""));
  post({ kind: "agent-start", tabId, goal, autonomy, see: seeing() });
}

function renderTask(v: AgentView): void {
  const t = thread(v.tabId);
  t.view = v;
  let card = t.agents.get(v.id);
  if (!card) {
    card = el("div", { class: "msg agent-card" });
    t.agents.set(v.id, card);
    t.empty.hidden = true;
    t.root.append(card);
  }
  const body = el("div", { class: "agent" });
  renderAgent(body, v, {
    approve: () => post({ kind: "agent-approve" }), reject: () => post({ kind: "agent-reject" }),
    cancel: () => post({ kind: "agent-cancel" }), reply: (text) => post({ kind: "agent-reply", text }),
  });
  body.querySelector(".agent-goal")?.remove();                     // the goal is the bubble just above
  body.querySelector(".agent-state")?.remove();                    // the live line above the card says it
  body.querySelector(".agent-stop")?.remove();                     // the composer's ■ stops it
  const now = activityLabel(v);
  const head = el("div", { class: "activity-now" + (isActive(v) ? " live" : "") },
                  isActive(v) ? now || "Working" : v.state === "completed" ? "Done" : "Stopped");
  card.replaceChildren(head, body);
  if (v.state === "waiting_for_access" && v.needsAccess) card.append(allowButton(v.needsAccess));
  scroll();
}

function allowButton(origin: string): HTMLElement {
  const b = el("button", { type: "button", class: "primary", textContent: `Allow ${origin.replace(/^https?:\/\//, "")}` });
  b.addEventListener("click", async (e) => {
    if (!e.isTrusted) return;
    let granted = false;
    try {
      granted = await api.permissions.request({ origins: [origin + "/*"] });     // this click is the user gesture
    } catch {
      granted = false;
    }
    if (granted) post({ kind: "agent-resume" });
    else b.textContent = "Not allowed — allow the site in the browser's extension settings, then click again";
  });
  return el("div", { class: "row" }, b);
}

// ───────────── screenshots ─────────────
function canSee(): boolean {
  const m = state?.models.find((x) => x.path === state?.model.path);
  return !!m?.vision;
}

function seeing(): boolean {
  return see.getAttribute("aria-pressed") === "true" && canSee();
}

see.addEventListener("click", async (e) => {
  if (!e.isTrusted) return;
  const on = see.getAttribute("aria-pressed") !== "true";
  if (on) {
    // a screenshot needs access to every site the tab may show; asked for once, on this click
    let ok = false;
    try {
      ok = await api.permissions.contains({ origins: ["<all_urls>"] }) || await api.permissions.request({ origins: ["<all_urls>"] });
    } catch {
      ok = false;
    }
    if (!ok) { see.title = "The browser did not allow screenshots"; return; }
  }
  see.setAttribute("aria-pressed", String(on));
});

// ───────────── the composer ─────────────
function busy(): boolean {
  return !!pendingRun || !!answer?.a.active || (tabId !== null && isActive(byTab.get(tabId)?.view ?? null));
}

function syncButtons(): void {
  const working = busy();
  const ready = state?.engine.state === "ready";
  stop.hidden = !working;
  see.disabled = !canSee();
  see.title = canSee() ? (mode.value === "ask" ? "Attach a screenshot of the page to the question"
    : "Let the model see a screenshot of the page at every step") : "The loaded model cannot see images";
  send.hidden = working;
  send.disabled = !state;                       // off is fine: sending starts the model
  input.placeholder = !state ? "Not connected — open ⚙︎ settings"
    : !ready && !pendingRun ? (mode.value === "ask" ? "Ask about this page — the model starts by itself"
                                                    : "Describe what to do — the model starts by itself")
    : mode.value === "ask" ? "Ask about this page — Enter to send" : "Describe what to do on this page — Enter to start";
}

function submit(): void {
  const text = input.value.trim();
  if (send.disabled || busy()) return;
  if (mode.value === "ask") withAccess(() => whenReady(() => void ask(text ? "ask" : "summarize", text || undefined)));
  else if (text) withAccess(() => whenReady(() => act(text)));
  else return;
  input.value = "";
  autosize();
}

function autosize(): void {
  input.style.height = "auto";
  input.style.height = Math.min(input.scrollHeight, 180) + "px";
}

send.addEventListener("click", submit);
input.addEventListener("input", autosize);
input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); submit(); }
});
stop.addEventListener("click", () => {
  if (pendingRun) {
    pendingRun = null;
    if (pendingNote) pendingNote.textContent = "Cancelled.";
    pendingNote = null;
    syncButtons();
    return;
  }
  if (answer?.a.active) post({ kind: "cancel" });
  else post({ kind: "agent-cancel" });
});
mode.addEventListener("change", () => {
  try { localStorage.setItem("mode", mode.value); } catch { /* private storage: keep the default */ }
  for (const t of byTab.values()) renderEmpty(t);
  syncButtons();
});
$("clear").addEventListener("click", () => {
  if (tabId === null || busy()) return;
  byTab.get(tabId)?.root.remove();
  byTab.delete(tabId);
  show(tabId);
});
$("settings").addEventListener("click", () => void api.runtime.openOptionsPage());

// ───────────── theme: dark and light, as in LLMBar, or the system's ─────────────
const THEMES = ["auto", "dark", "light"] as const;
type Theme = typeof THEMES[number];
const THEME_ICON: Record<Theme, string> = { auto: "◐", dark: "☾", light: "☀" };
const THEME_NAME: Record<Theme, string> = { auto: "follows the system", dark: "dark", light: "light" };
function applyTheme(t: Theme): void {
  if (t === "auto") delete document.documentElement.dataset.theme;
  else document.documentElement.dataset.theme = t;
  const b = $("theme");
  b.textContent = THEME_ICON[t];
  b.title = `Theme: ${THEME_NAME[t]} — click to change`;
}
let theme: Theme = "auto";
try {
  const saved = localStorage.getItem("theme") as Theme | null;
  if (saved && THEMES.includes(saved)) theme = saved;
} catch {
  // storage blocked: follow the system
}
applyTheme(theme);
$("theme").addEventListener("click", () => {
  theme = THEMES[(THEMES.indexOf(theme) + 1) % THEMES.length];
  applyTheme(theme);
  try { localStorage.setItem("theme", theme); } catch { /* not remembered: fine */ }
});
// which build this is, at the foot of the panel: "v0.1.0.264", the commit in its tooltip
const foot = document.querySelector(".foot");
if (foot) {
  const v = el("span", { class: "ver", textContent: ` · v${MINE}`, title: MANIFEST.version_name || MINE });
  foot.append(v);
}
// Chrome's own side panel already shows the extension's icon and name above this page: ours would repeat it
if (!EMBEDDED) document.body.classList.add("browser-titled");
if (EMBEDDED) {
  $("undock").hidden = false;
  $("undock").addEventListener("click", () => post({ kind: "undock" }));
}

// ───────────── the model menu ─────────────
chip.addEventListener("click", () => {
  sheet.hidden = !sheet.hidden;
  chip.setAttribute("aria-expanded", String(!sheet.hidden));
});

function command(cmd: Command, confirm = false): void {
  note.textContent = "";
  post({ kind: "command", command: cmd, confirm });
}

function controlButton(label: string, cmd: Command, danger = false): HTMLButtonElement {
  const b = el("button", { type: "button", textContent: label, class: danger ? "danger" : "" });
  b.addEventListener("click", () => command(cmd));
  return b;
}

function renderState(ev: Extract<UiEvent, { kind: "state" }>): void {
  state = ev.state;
  setTimeout(settlePending, 0);                // after the rest of the state is drawn
  const problem = $("problem");
  problem.hidden = !!(ev.connected && state);
  problem.textContent = problem.hidden ? "" : problemText(ev.problem);
  if (!state) {
    $("dot").className = "dot off";
    $("model-name").textContent = "Not connected";
    syncButtons();
    return;
  }
  const line = statusLine(state);
  $("dot").className = "dot " + line.tone;
  $("model-name").textContent = state.model.name || (state.engine.state === "ready" ? "No model" : line.text);
  chip.title = line.text;
  $("metrics").textContent = metricsText(state);
  const notice = updateNotice(state, MINE, browserName === "chromium" ? "chromium" : "safari", PROTOCOL, ev.app);
  $("update").hidden = !notice;
  $("update").textContent = notice;
  if (notice && browserName === "chromium" && !ev.app) {
    // the update waits for this click: reloading on its own would close the panel mid-conversation
    const now = el("button", { type: "button", class: "primary", textContent: "Update now" });
    now.style.marginLeft = "8px";
    now.addEventListener("click", () => { if (!busy()) post({ kind: "reload" }); });
    $("update").append(now);
  }
  picker.replaceChildren(...state.models.map((m) => el("option", {
    value: m.path, textContent: modelLabel(m), disabled: !m.supported, selected: m.path === state!.default })));
  const opBusy = state.engine.operation !== null;
  picker.disabled = opBusy || !state.models.length;
  controls.replaceChildren();
  const e = state.engine.state, m = state.model.state;
  if (e === "offline" || e === "error") controls.append(controlButton("Start engine", { op: "start" }));
  else if (e === "ready") {
    if (m === "not_loaded") controls.append(controlButton("Load model", { op: "load" }));
    if ((m === "ready" || m === "generating") && state.capabilities.unload) controls.append(controlButton("Unload model", { op: "unload" }));
    controls.append(controlButton("Stop engine", { op: "stop" }, true));
  }
  for (const b of Array.from(controls.querySelectorAll("button"))) b.disabled = opBusy;
  if (state.engine.error) note.textContent = state.engine.error;
  syncButtons();
}

picker.addEventListener("change", () => {
  if (state && picker.value !== state.default) command({ op: "default", path: picker.value, expected: state.default });
});

function conflict(error: ServiceError, cmd: Command): void {
  note.replaceChildren(error.message + " ");
  if (error.code === "model_in_use") {
    const yes = el("button", { type: "button", textContent: "Do it anyway", class: "danger" });
    yes.addEventListener("click", () => command(cmd, true));
    note.append(yes);
  }
}

// ───────────── the background's side ─────────────
function onEvent(ev: UiEvent): void {
  if (ev.kind === "state") renderState(ev);
  else if (ev.kind === "agent") {
    if (ev.view) renderTask(ev.view);
    else if (tabId !== null) thread(tabId).view = null;
    syncButtons();
  } else if (ev.kind === "tab") {
    if (ev.tabId !== tabId) return;
    if (ev.url) currentUrl = ev.url;
    let host = "";
    try { host = new URL(ev.url).host; } catch { host = ""; }
    tabLine.textContent = host ? `${host} — ${ev.title}` : ev.title || ev.url;
  } else if (ev.kind === "shot") {
    lastBubble?.append(el("img", { class: "shot", src: ev.dataUrl, alt: "The screenshot sent with this question" }));
    scroll();
  } else if (ev.kind === "chat" && answer) {
    answer.a.on(ev.event);
    syncButtons();
    scroll();
  } else if (ev.kind === "fatal") {
    if (answer?.a.active) answer.a.fail(ev.message);
    else if (tabId !== null) thread(tabId).root.append(el("div", { class: "msg note bad" }, ev.message));
    syncButtons();
  } else if (ev.kind === "command-result" && !ev.ok && ev.error) conflict(ev.error, ev.command);
}

api.tabs.onUpdated.addListener((id, change) => {
  if (id === tabId && change.url) currentUrl = change.url;     // a new page in the same tab
});
api.tabs.onActivated.addListener((info) => {
  if (windowId === undefined || info.windowId === windowId) show(info.tabId);
});

try {
  const saved = localStorage.getItem("mode");
  if (saved && [...mode.options].some((o) => o.value === saved)) mode.value = saved;
} catch {
  // storage blocked: the default mode
}
post({ kind: "subscribe" });
void api.tabs.query({ active: true, currentWindow: true }).then(([tab]) => {
  if (!tab || tab.id === undefined) return;
  windowId = tab.windowId;
  show(tab.id);
});
autosize();
