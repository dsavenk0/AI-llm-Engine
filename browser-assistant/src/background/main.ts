// The background script: the only code that holds the token and talks to the assistant service. The popup
// and the in-page panel connect to it over runtime ports; it reads the page they ask about, streams the
// answer back, and keeps one state stream open while any of them is listening.

import { api, browserName, Port } from "../browser-adapters/api";
import { Client, ServiceFailure, Settings } from "../core/client";
import { ChatEvent, Command, DEFAULT_SERVICE, Snapshot, Task, UiEvent, UiRequest } from "../core/protocol";
import { extractPage, Extracted } from "../content/extract";
import { AgentRunner } from "./agent";
import { bridge } from "./agent-bridge";
import { AgentView, AuditEntry } from "../core/agent-types";

const MAX_PAGE_CHARS = 200_000;      // what is read from a page; the service cuts it to the model's context
const BOOT = Math.random().toString(36).slice(2, 8);

async function settings(): Promise<Settings> {
  const s = await api.storage.local.get(["service", "token"]);
  return { service: (s.service as string) || DEFAULT_SERVICE, token: (s.token as string) || "" };
}

// ───────────── state: one stream, shared by every open popup and panel ─────────────
const listeners = new Set<Port>();
type StateEvent = Extract<UiEvent, { kind: "state" }>;
let last: StateEvent = { kind: "state", state: null, connected: false, problem: "connecting" };
let stream: AbortController | null = null;

function broadcast(ev: StateEvent): void {
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
    send(port, { kind: "fatal", message: "This page cannot be read by extensions (a browser page, the App Store, "
      + "or a site where access was not granted). Select text and use the context menu instead." });
    return;
  }
  if (req.selection && !page.selection) page.selection = req.selection;
  if (req.selectionOnly) page.text = "";
  send(port, { kind: "page", title: page.title, url: page.url, chars: page.text.length + page.selection.length });
  const run: Running = { id: null, abort: new AbortController(), session };
  running.set(port, run);
  try {
    await new Client(await settings()).chat({
      session, task: req.task, question: req.question, lang: req.lang, tone: req.tone, thinking: req.thinking,
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
const portTab = new Map<Port, number>();           // which tab a popup or panel is about

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

const runner = new AgentRunner({
  page: bridge,
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
  },
  audit: (e) => { void audit(e); },
  session: (tabId) => `${browserName}-${tabId}-${BOOT}`,
  id: () => crypto.randomUUID(),
});

api.tabs.onRemoved.addListener((id) => runner.cancel(id));             // a closed tab ends its task

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
  if (!s.token) await api.runtime.openOptionsPage();
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
  if (port.name !== "popup" && port.name !== "panel" && port.name !== "options") return;
  const tabId = port.sender?.tab?.id;
  port.onMessage.addListener(async (msg: UiRequest) => {
    // which tab the agent messages of this port are about: a panel lives in one; a popup is about the active one
    const agentTab = async (): Promise<number | undefined> =>
      tabId ?? portTab.get(port) ?? (await api.tabs.query({ active: true, currentWindow: true }))[0]?.id;
    if (msg.kind === "agent-start") {
      portTab.set(port, msg.tabId);
      try {
        send(port, { kind: "agent", view: await runner.start(msg.tabId, msg.goal) });
        await bridge.showPanel(msg.tabId).catch(() => undefined);
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
    void cancel(port);                 // a closed popup must not leave the model generating for nobody
    if (!listeners.size && stream) {
      stream.abort();
      stream = null;
    }
  });
});
