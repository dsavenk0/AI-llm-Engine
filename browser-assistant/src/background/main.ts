// The background script: the only code that holds the token and talks to the assistant service. The side panel
// (Chrome's own, or docked into the tab) and the in-page answer panel connect to it over runtime ports; it reads the page they ask about, streams the
// answer back, and keeps one state stream open while any of them is listening.

import { api, browserName, Port } from "../browser-adapters/api";
import { Client, ServiceFailure, Settings } from "../core/client";
import { ChatEvent, Command, DEFAULT_SERVICE, Snapshot, Task, UiEvent, UiRequest } from "../core/protocol";
import { extractPage, Extracted } from "../content/extract";
import { AgentRunner } from "./agent";
import { captureTab, captureWithPictures, CaptureError } from "./capture";
import { bridge, overlay } from "./agent-bridge";
import { overlayFor } from "../core/activity";
import { originOf } from "../core/policy";
import { AgentView, AuditEntry } from "../core/agent-types";

const MAX_PAGE_CHARS = 200_000;      // what is read from a page; the service cuts it to the model's context
const BOOT = Math.random().toString(36).slice(2, 8);

async function settings(): Promise<Settings> {
  const s = await api.storage.local.get(["service", "token"]);
  let token = (s.token as string) || "";
  if (!token) token = await pairFromApp();
  return { service: (s.service as string) || DEFAULT_SERVICE, token };
}

/** The pairing token without pasting: in Safari from the handler inside LLMBar.app, in a Chromium browser from
 *  the native-messaging host `./llm assistant setup` registers (assistant/native_host.py). Where neither exists —
 *  a temporary Safari extension — the settings page asks for it as before. */
let appInfo: { version: string } | undefined;          // set once LLMBar's native handler has answered

async function pairFromApp(): Promise<string> {
  if (!api.runtime.sendNativeMessage) return "";
  try {
    const r = await api.runtime.sendNativeMessage("local.llm_engine.assistant", { kind: "token" });
    if (r && typeof r.version === "string" && r.version) appInfo = { version: r.version };
    if (r && r.ok && typeof r.token === "string" && r.token.length >= 24) {
      await api.storage.local.set({ token: r.token });
      return r.token;
    }
  } catch {
    // not inside the app, or the app is not installed: pair by hand
  }
  return "";
}

// ───────────── state: one stream, shared by every open side panel and in-page panel ─────────────
const listeners = new Set<Port>();
type StateEvent = Extract<UiEvent, { kind: "state" }>;
let last: StateEvent = { kind: "state", state: null, connected: false, problem: "connecting" };
let stream: AbortController | null = null;

// ───────────── updating itself ─────────────
// A browser loads an unpacked (Chrome) or temporary (Safari) extension from its folder only when it is loaded.
// LLMBar's update puts newer files in that folder, and the service reports their version: then the extension
// reloads itself from the folder — once per version, and never while an answer or a task is running. Inside
// LLMBar.app (Safari) the extension comes with the app, so this does not apply.
const MINE = api.runtime.getManifest().version;
let reloading = false;

async function maybeReload(s: Snapshot | null): Promise<void> {
  if (!s || appInfo || reloading) return;
  const folder = s.extension.versions[browserName === "chromium" ? "chromium" : "safari"];
  if (!folder || !newerVersion(folder, MINE)) return;
  if (running.size || runner.anyActive()) return;            // after the current answer or task, not in the middle
  // A reload closes the side panel and its conversation: while one is open, wait — it shows that an update is
  // ready and reloads on the person's click (kind "reload"); with no panel open the update happens by itself.
  if (sidePanels().length) return;
  const done = (await api.storage.local.get("reloadedFor")).reloadedFor;
  if (done === folder) return;                               // tried once for this version: do not loop
  reloading = true;
  await api.storage.local.set({ reloadedFor: folder });
  api.runtime.reload();
}

function newerVersion(a: string, b: string): boolean {
  const x = a.split(".").map(Number), y = b.split(".").map(Number);
  for (let i = 0; i < Math.max(x.length, y.length); i++) {
    if ((x[i] || 0) !== (y[i] || 0)) return (x[i] || 0) > (y[i] || 0);
  }
  return false;
}

function broadcast(ev: StateEvent): void {
  void maybeReload(ev.state).catch(() => undefined);
  if (appInfo) ev = { ...ev, app: appInfo };
  last = ev;
  for (const p of listeners) send(p, ev);
}

function send(port: Port, ev: UiEvent): void {
  try {
    port.postMessage(ev);
  } catch {
    listeners.delete(port);
  }
}

async function follow(): Promise<void> {
  if (stream) return;
  const ctl = new AbortController();
  stream = ctl;
  let delay = 500;
  void pairFromApp();                  // also learns which extension version LLMBar ships now (after an update)
  while (!ctl.signal.aborted && listeners.size) {
    const s = await settings();
    if (!s.token) {
      broadcast({ kind: "state", state: null, connected: false, problem: "not_paired" });
      break;
    }
    try {
      await new Client(s).events((state: Snapshot) => {
        delay = 500;
        broadcast({ kind: "state", state, connected: true });
      }, ctl.signal);
    } catch (e) {
      if (ctl.signal.aborted) break;
      const problem = e instanceof ServiceFailure && e.status === 401 ? "bad_token" : "unreachable";
      broadcast({ kind: "state", state: null, connected: false, problem });
      if (problem === "bad_token") break;
    }
    await new Promise((r) => setTimeout(r, delay));
    delay = Math.min(delay * 2, 8000);
  }
  if (stream === ctl) stream = null;
}

function restartStream(): void {
  stream?.abort();
  stream = null;
  void follow();
}

// ───────────── commands ─────────────
async function command(port: Port, cmd: Command, confirm: boolean, session: string): Promise<void> {
  const c = new Client(await settings());
  const paths: Record<Command["op"], string> = {
    start: "/v1/assistant/engine/start", stop: "/v1/assistant/engine/stop", load: "/v1/assistant/models/load",
    unload: "/v1/assistant/models/unload", default: "/v1/assistant/models/default",
  };
  const body: Record<string, unknown> = { session, confirm };
  if (cmd.op === "default") {
    body.path = cmd.path;
    body.expected_default = cmd.expected;
  }
  try {
    await c.post(paths[cmd.op], body);
    send(port, { kind: "command-result", ok: true, command: cmd });
  } catch (e) {
    const error = e instanceof ServiceFailure ? e.error : { code: "unreachable", message: String(e) };
    send(port, { kind: "command-result", ok: false, error, command: cmd });
  }
}

// ───────────── asking about a page ─────────────
interface Running { id: string | null; abort: AbortController; session: string }
const running = new Map<Port, Running>();

async function readPage(tabId: number): Promise<Extracted> {
  const res = await api.scripting.executeScript({ target: { tabId }, func: extractPage, args: [MAX_PAGE_CHARS] });
  const out = res && res[0] && (res[0].result as Extracted | undefined);
  if (!out) throw new Error("the page did not answer");
  return out;
}

async function ask(port: Port, req: Extract<UiRequest, { kind: "ask" }>): Promise<void> {
  if (running.has(port)) await cancel(port);
  const session = `${browserName}-${req.tabId}-${BOOT}`;
  let page: Extracted;
  try {
    page = await readPage(req.tabId);
  } catch {
    const url = (await tabInfo(req.tabId))?.url || "";
    const origin = originOf(url);
    send(port, { kind: "fatal", message: origin
      ? `LLMBar has no access to ${origin.replace(/^https?:\/\//, "")} yet. Send again and choose Allow when the browser asks.`
      : `This is one of the browser's own pages${url ? ` (${url.split("?")[0]})` : ""}, which no extension may read. `
        + "Open a website and ask there." });
    return;
  }
  if (req.selection && !page.selection) page.selection = req.selection;
  if (req.selectionOnly) page.text = "";
  let image: string | undefined;
  let images: string[] = [];
  if (req.screenshot) {
    try {
      const shot = await captureWithPictures(req.tabId, page.pictures || [], page.dpr || 1);
      image = shot.screenshot;
      images = shot.pictures;                                  // the page's own pictures, at full resolution
      send(port, { kind: "shot", dataUrl: image });
    } catch (e) {
      send(port, { kind: "fatal", message: e instanceof CaptureError ? e.message : `No screenshot: ${String(e)}` });
      return;
    }
  }
  send(port, { kind: "page", title: page.title, url: page.url, chars: page.text.length + page.selection.length });
  const run: Running = { id: null, abort: new AbortController(), session };
  running.set(port, run);
  try {
    await new Client(await settings()).chat({
      session, task: req.task, question: req.question, lang: req.lang, tone: req.tone, thinking: req.thinking, image, images,
      history: Array.isArray(req.history) ? req.history.slice(-8) : undefined,
      page: { title: page.title, url: page.url, text: page.text, selection: page.selection },
    }, (ev: ChatEvent) => {
      if (ev.type === "start") run.id = ev.id;
      send(port, { kind: "chat", event: ev });
    }, run.abort.signal);
  } catch (e) {
    if (!run.abort.signal.aborted) send(port, { kind: "chat", event: { type: "error", message: String(e) } });
    else send(port, { kind: "chat", event: { type: "cancelled" } });
  } finally {
    if (running.get(port) === run) running.delete(port);
  }
}

async function cancel(port: Port): Promise<void> {
  const run = running.get(port);
  if (!run) return;
  running.delete(port);
  if (run.id) {
    try {
      await new Client(await settings()).cancel(run.id, run.session);
      return;                          // the stream ends on its own with a "cancelled" event
    } catch {
      // fall through: dropping the connection also stops the engine
    }
  }
  run.abort.abort();
}

// ───────────── the browser agent ─────────────
const portTab = new Map<Port, number>();           // which tab a side panel or in-page panel is about

async function audit(entry: AuditEntry): Promise<void> {
  try {
    const cur = ((await api.storage.local.get("audit")).audit as AuditEntry[] | undefined) || [];
    cur.push(entry);
    await api.storage.local.set({ audit: cur.slice(-200) });          // metadata only: never page text, never a typed value
  } catch {
    // a log that cannot be written must not stop the task
  }
}

async function tabInfo(id: number): Promise<{ url: string; loading: boolean } | null> {
  try {
    const t = await api.tabs.get(id);
    return { url: t.url || "", loading: t.status === "loading" };
  } catch {
    return null;
  }
}

// With a side panel open the approval card lives there; the in-page panel is only for Safari and context menus.
const sidePanels = (): Port[] => [...listeners].filter((p) => p.name === "sidepanel");
const pageBridge = { ...bridge, panelShown: async (tabId: number) => sidePanels().length > 0 || bridge.panelShown(tabId) };

function sendTab(port: Port, tabId: number): void {
  void api.tabs.get(tabId).then((t) => send(port, { kind: "tab", tabId, title: t.title || "", url: t.url || "" }))
    .catch(() => undefined);
}

const runner = new AgentRunner({
  page: pageBridge,
  step: async (req, signal) => new Client(await settings()).agentStep(req, signal),
  cancelStep: async (id, session) => new Client(await settings()).cancel(id, session),
  tab: tabInfo,
  navigate: async (tabId, url) => { await api.tabs.update(tabId, { url }); },
  settle: async (tabId) => {
    await new Promise((r) => setTimeout(r, 250));                      // the status may not have flipped to loading yet
    for (let i = 0; i < 50; i++) {
      const t = await tabInfo(tabId);
      if (!t || !t.loading) return;
      await new Promise((r) => setTimeout(r, 200));
    }
  },
  notify: (tabId: number, view: AgentView | null) => {
    for (const [port, tab] of portTab) if (tab === tabId) send(port, { kind: "agent", view });
    void overlay(tabId, overlayFor(view)).catch(() => undefined);     // the frame on the page itself
  },
  audit: (e) => { void audit(e); },
  session: (tabId) => `${browserName}-${tabId}-${BOOT}`,
  capture: (tabId) => captureTab(tabId),
  id: () => crypto.randomUUID(),
});

api.tabs.onRemoved.addListener((id) => runner.cancel(id));             // a closed tab ends its task
api.tabs.onUpdated.addListener((id, change) => {
  if (change.title === undefined && change.url === undefined && change.status !== "complete") return;
  for (const [port, tab] of portTab) if (tab === id && port.name === "sidepanel") sendTab(port, id);
  // a new page in a tab the agent is driving: put the frame back once the page agent is there again
  if (change.status === "complete" && runner.active(id)) void overlay(id, overlayFor(runner.view(id))).catch(() => undefined);
});

// The toolbar button opens the side panel: Chrome's own where the browser has one, otherwise the same page docked
// into the tab (Safari has no side panel API for extensions, and some Chromium browsers leave it out).
const DOCK_ID = "ai-llm-engine-assistant-dock";
async function dock(tabId: number): Promise<void> {
  await api.scripting.executeScript({ target: { tabId }, files: ["dock.js"] });
}
async function undock(tabId: number): Promise<void> {
  await api.scripting.executeScript({ target: { tabId }, args: [DOCK_ID], func: (id: string) => {
    document.getElementById(id)?.remove();
    document.documentElement.style.removeProperty("margin-right");
  } });
}
if (api.sidePanel) void api.sidePanel.setPanelBehavior({ openPanelOnActionClick: true }).catch(() => undefined);
api.action?.onClicked.addListener((tab) => {
  if (tab.id !== undefined) void dock(tab.id).catch(() => undefined);    // a browser page: nothing can be docked there
});

// ───────────── the context menu and the in-page panel ─────────────
interface Pending { task: Task; selection?: string; selectionOnly: boolean }
const pending = new Map<number, Pending>();

const MENU: { id: string; title: string; contexts: string[]; task: Task; selectionOnly: boolean }[] = [
  { id: "explain", title: "Explain the selection", contexts: ["selection"], task: "explain", selectionOnly: false },
  { id: "summarize-sel", title: "Summarize the selection", contexts: ["selection"], task: "summarize", selectionOnly: true },
  { id: "translate", title: "Translate the selection", contexts: ["selection"], task: "translate", selectionOnly: true },
  { id: "rewrite", title: "Rewrite the selection more clearly", contexts: ["selection"], task: "rewrite", selectionOnly: true },
  { id: "summarize-page", title: "Summarize this page", contexts: ["page"], task: "summarize", selectionOnly: false },
];

api.runtime.onInstalled.addListener(async () => {
  await api.contextMenus.removeAll();
  for (const m of MENU) api.contextMenus.create({ id: m.id, title: m.title, contexts: m.contexts });
  const s = await api.storage.local.get(["token"]);
  if (!s.token && !(await pairFromApp())) await api.runtime.openOptionsPage();
});

api.contextMenus.onClicked.addListener(async (info, tab) => {
  const item = MENU.find((m) => m.id === info.menuItemId);
  if (!item || !tab || tab.id === undefined) return;
  pending.set(tab.id, { task: item.task, selection: info.selectionText, selectionOnly: item.selectionOnly });
  try {
    await api.scripting.executeScript({ target: { tabId: tab.id }, files: ["panel.js"] });
  } catch {
    pending.delete(tab.id);
  }
});

// ───────────── ports ─────────────
api.runtime.onConnect.addListener((port: Port) => {
  if (port.name === "overlay") {
    // the Stop on the page's working-here pill: it can stop the agent on its own tab and do nothing else
    const own = port.sender?.tab?.id;
    port.onMessage.addListener((msg: UiRequest) => { if (msg.kind === "agent-cancel" && own !== undefined) runner.cancel(own); });
    return;
  }
  if (port.name !== "panel" && port.name !== "options" && port.name !== "sidepanel") return;
  const tabId = port.sender?.tab?.id;
  port.onMessage.addListener(async (msg: UiRequest) => {
    // which tab the agent messages of this port are about: an in-page panel lives in one; the side panel follows the active one
    const agentTab = async (): Promise<number | undefined> =>
      tabId ?? portTab.get(port) ?? (await api.tabs.query({ active: true, currentWindow: true }))[0]?.id;
    if (msg.kind === "agent-start") {
      portTab.set(port, msg.tabId);
      try {
        send(port, { kind: "agent", view: await runner.start(msg.tabId, msg.goal, msg.autonomy === "auto" ? "auto" : "ask",
                                                                msg.see === true) });
        if (port.name !== "sidepanel") await bridge.showPanel(msg.tabId).catch(() => undefined);
      } catch (e) {
        send(port, { kind: "fatal", message: e instanceof Error ? e.message : String(e) });
      }
      return;
    }
    if (msg.kind === "agent-approve" || msg.kind === "agent-reject" || msg.kind === "agent-cancel" || msg.kind === "agent-resume" || msg.kind === "agent-reply") {
      const t = await agentTab();
      if (t === undefined) return;
      if (msg.kind === "agent-approve") runner.approve(t);
      else if (msg.kind === "agent-reject") runner.reject(t);
      else if (msg.kind === "agent-cancel") runner.cancel(t);
      else if (msg.kind === "agent-resume") runner.resume(t);
      else runner.reply(t, msg.text);
      return;
    }
    if (msg.kind === "reload") {
      api.runtime.reload();                                    // the person pressed "Update now" in the panel
      return;
    }
    if (msg.kind === "undock") {
      if (tabId !== undefined) await undock(tabId).catch(() => undefined);
      return;
    }
    if (msg.kind === "watch") {
      // the side panel follows the active tab of its window: its agent card and page line switch with it
      portTab.set(port, msg.tabId);
      send(port, { kind: "agent", view: runner.view(msg.tabId) });
      sendTab(port, msg.tabId);
      return;
    }
    if (msg.kind === "subscribe") {
      listeners.add(port);
      send(port, last);
      const t = tabId ?? (await api.tabs.query({ active: true, currentWindow: true }))[0]?.id;
      if (t !== undefined) {
        portTab.set(port, t);
        send(port, { kind: "agent", view: runner.view(t) });
      }
      if (!stream) void follow();
      else if (!last.connected) restartStream();
      if (port.name === "panel" && tabId !== undefined && pending.has(tabId)) {
        const p = pending.get(tabId)!;
        pending.delete(tabId);
        const lang = navigator.language.startsWith("ru") ? "Russian" : "English";
        void ask(port, { kind: "ask", tabId, task: p.task, selection: p.selection, selectionOnly: p.selectionOnly, lang });
      }
    } else if (msg.kind === "command") {
      const tab = port.name === "panel" ? tabId : (await api.tabs.query({ active: true, currentWindow: true }))[0]?.id;
      await command(port, msg.command, msg.confirm === true, `${browserName}-${tab ?? "x"}-${BOOT}`);
    } else if (msg.kind === "ask") {
      // a panel may only ask about the tab it lives in
      const target = port.name === "panel" ? tabId : msg.tabId;
      if (target !== undefined) await ask(port, { ...msg, tabId: target });
    } else if (msg.kind === "cancel") {
      await cancel(port);
    }
  });
  port.onDisconnect.addListener(() => {
    listeners.delete(port);
    portTab.delete(port);                // a page navigating away drops its panel's port: that does not stop the agent
    void cancel(port);                 // a closed panel must not leave the model generating for nobody
    if (!listeners.size && stream) {
      stream.abort();
      stream = null;
    }
  });
});
