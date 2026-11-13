// The assistant service's client. Used by the background script only: no page and no content script ever
// holds the token or talks to the service.

import { Action, PageSnapshot } from "./agent-types";
import { ChatEvent, PageContext, ServiceError, Snapshot, Task } from "./protocol";
import { readSse } from "./sse";

export interface Settings {
  service: string;
  token: string;
}

export class ServiceFailure extends Error {
  constructor(public status: number, public error: ServiceError) {
    super(error.message);
  }
}

export class Client {
  constructor(private settings: Settings) {}

  private headers(): Record<string, string> {
    return { "Content-Type": "application/json", Authorization: `Bearer ${this.settings.token}` };
  }

  private url(path: string): string {
    return this.settings.service.replace(/\/+$/, "") + path;
  }

  async hello(): Promise<{ service: string; protocol: number }> {
    const r = await fetch(this.url("/v1/assistant/hello"));
    if (!r.ok) throw new ServiceFailure(r.status, { code: "unreachable", message: `service answered ${r.status}` });
    return r.json();
  }

  async state(): Promise<Snapshot> {
    return this.json("GET", "/v1/assistant/state");
  }

  async post<T>(path: string, body: unknown): Promise<T> {
    return this.json("POST", path, body);
  }

  async get<T>(path: string): Promise<T> {
    return this.json("GET", path);
  }

  private async json<T>(method: string, path: string, body?: unknown): Promise<T> {
    const r = await fetch(this.url(path), {
      method, headers: this.headers(), body: body === undefined ? undefined : JSON.stringify(body),
    });
    const data = await r.json().catch(() => ({}));
    if (!r.ok) {
      const err = (data && data.error) || { code: "http_" + r.status, message: `the service answered ${r.status}` };
      throw new ServiceFailure(r.status, err);
    }
    return data as T;
  }

  /** Follow the state stream until `signal` aborts or the connection drops. */
  async events(onState: (s: Snapshot) => void, signal: AbortSignal): Promise<void> {
    const r = await fetch(this.url("/v1/assistant/events"), { headers: this.headers(), signal });
    if (!r.ok || !r.body) {
      const data = await r.json().catch(() => ({}));
      throw new ServiceFailure(r.status, data.error || { code: "http_" + r.status, message: `events: ${r.status}` });
    }
    await readSse(r.body, (ev) => {
      const e = ev as { type?: string; state?: Snapshot };
      if (e.type === "state" && e.state) onState(e.state);
    });
  }

  /** One task about a page, streamed. Resolves when the answer ends, however it ends. */
  async chat(req: {
    session: string; task: Task; question?: string; page: PageContext; lang?: string; tone?: string;
    thinking?: boolean; image?: string; images?: string[]; history?: { q: string; a: string }[];
  }, onEvent: (e: ChatEvent) => void, signal: AbortSignal): Promise<void> {
    const body = {
      session: req.session, task: req.task, question: req.question, lang: req.lang, tone: req.tone,
      thinking: req.thinking, selection: req.page.selection, image: req.image,
      images: req.images && req.images.length ? req.images : undefined,
      history: req.history && req.history.length ? req.history : undefined,
      page: { title: req.page.title, url: req.page.url, text: req.page.text },
    };
    const r = await fetch(this.url("/v1/assistant/chat"), {
      method: "POST", headers: this.headers(), body: JSON.stringify(body), signal,
    });
    if (!r.ok || !r.body) {
      const data = await r.json().catch(() => ({}));
      const err: ServiceError = data.error || { code: "http_" + r.status, message: `the service answered ${r.status}` };
      onEvent({ type: "error", message: err.message, code: err.code });
      return;
    }
    await readSse(r.body, (ev) => onEvent(ev as ChatEvent));
  }

  /** The model's next action for this page and goal. `requestId` names the call so it can be cancelled. */
  async agentStep(req: { session: string; goal: string; snapshot: PageSnapshot; history: { summary: string; result: string }[];
    requestId: string; tabId: number; screenshot?: string | null; frame?: { width: number; height: number; signature: string } }, signal: AbortSignal): Promise<{ action: Action; attempts: number }> {
    const r = await fetch(this.url("/v1/assistant/agent/step"), {
      method: "POST", headers: this.headers(), signal,
      body: JSON.stringify({ session: req.session, goal: req.goal, snapshot: req.snapshot, history: req.history,
                             request_id: req.requestId, tab_id: req.tabId, screenshot: req.screenshot || undefined,
                             frame: req.frame }),
    });
    const data = await r.json().catch(() => ({}));
    if (!r.ok) throw new ServiceFailure(r.status, data.error || { code: "http_" + r.status, message: `the service answered ${r.status}` });
    return data;
  }

  /** Run a service-issued, short-lived visual action through the private macOS helper. */
  async visualExecute(req: { session: string; ticket: string; tabId: number; url: string; screenshot: string;
    frame?: { width: number; height: number; signature: string }; browser: string }): Promise<{ ok: boolean; error?: string; route?: string }> {
    return this.post("/v1/assistant/agent/visual/execute", req);
  }

  async workspaceExecute(req: { session: string; ticket: string; url: string }): Promise<{ ok: boolean; service?: string; undo?: string; [k: string]: unknown }> {
    return this.post("/v1/assistant/workspace/execute", req);
  }

  async workspaceUndo(req: { session: string; undo: string }): Promise<{ ok: boolean }> {
    return this.post("/v1/assistant/workspace/undo", req);
  }

  async forgetSession(session: string): Promise<void> {
    await this.post("/v1/assistant/session/forget", { session });
  }

  async setupDictation(): Promise<{ ready: boolean }> {
    return this.post("/v1/assistant/audio/setup", {});
  }

  async transcribe(audio: string): Promise<{ text: string; language?: string }> {
    return this.post("/v1/assistant/audio/transcribe", { audio });
  }

  async cancel(id: string, session: string): Promise<void> {
    await this.post(`/v1/assistant/generations/${encodeURIComponent(id)}/cancel`, { session });
  }
}
