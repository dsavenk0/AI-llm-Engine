// What the side panel and the in-page panel share: turning a snapshot into words and buttons, and rendering a
// streamed answer. Plain DOM — small enough that a framework would be most of the bundle.

import { ChatEvent, ModelInfo, Shared, Snapshot } from "../core/protocol";

export function el<K extends keyof HTMLElementTagNameMap>(tag: K, props: Partial<HTMLElementTagNameMap[K]> & {
  class?: string } = {}, ...children: (Node | string)[]): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag);
  const { class: cls, ...rest } = props as { class?: string } & Record<string, unknown>;
  if (cls) node.className = cls;
  Object.assign(node, rest);
  for (const c of children) node.append(c);
  return node;
}

const ENGINE: Record<string, string> = { offline: "Engine is off", starting: "Engine is starting…",
  ready: "Engine is running", error: "Engine failed to start" };
const MODEL: Record<string, string> = { not_loaded: "model not loaded", loading: "loading the model…",
  ready: "model ready", generating: "generating", unloading: "unloading…", error: "model error" };

export function statusLine(s: Snapshot): { text: string; tone: "ok" | "busy" | "off" | "bad" } {
  const tone = s.engine.state === "error" ? "bad" : s.engine.state === "offline" ? "off"
    : s.engine.state === "starting" || s.model.state !== "ready" && s.model.state !== "not_loaded" ? "busy" : "ok";
  const text = s.engine.state === "ready" ? `${s.model.name ?? "Engine"} — ${MODEL[s.model.state]}` : ENGINE[s.engine.state];
  return { text, tone };
}

export function problemText(problem: string | undefined): string {
  return problem === "not_paired" ? "Not connected yet: open Settings and paste the token."
    : problem === "bad_token" ? "The token was not accepted. Open Settings and paste it again."
    : problem === "connecting" ? "Connecting…"
    : "The assistant service is not running. Start it with ./llm assistant install (or LLMBar ▸ Tools ▸ Browser Assistant).";
}

export function modelLabel(m: ModelInfo): string {
  const bits = [m.format, `${m.size_gb.toFixed(1)} GB`, m.route].filter(Boolean).join(" · ");
  return `${m.favorite ? "★ " : ""}${m.name} — ${bits}${m.supported ? (m.fits ? "" : " · may not fit") : " · unsupported"}`;
}

/** True when `latest` is a higher dotted version than `mine`. */
export function newer(latest: string | undefined, mine: string): boolean {
  if (!latest) return false;
  const a = latest.split(".").map(Number), b = mine.split(".").map(Number);
  for (let i = 0; i < Math.max(a.length, b.length); i++) {
    const x = a[i] || 0, y = b[i] || 0;
    if (x !== y) return x > y;
  }
  return false;
}

/** What to tell the user when the service or the installed files are ahead of, or behind, this extension. */
export function updateNotice(s: Snapshot, mine: string, browser: string, protocol: number,
                             app?: { version: string }): string {
  if (s.protocol > protocol) {
    return `The engine was updated and this extension is too old for it. Add it again from ${s.extension.folder}/${browser}.`;
  }
  if (s.protocol < protocol) return "This extension is newer than the installed engine. Update LLMBar or the engine.";
  if (app) {
    // inside LLMBar the extension comes with the app: the only stale case is Safari still running the old one
    return newer(app.version, mine)
      ? `LLMBar was updated and brings version ${app.version} of this extension (this is ${mine}). Quit and reopen Safari to load it.`
      : "";
  }
  const latest = s.extension.versions[browser];
  if (!newer(latest, mine)) return "";
  return browser === "chromium"
    ? `Updating to version ${latest} (this is ${mine})… If it does not reload by itself, press ↻ on the extension in chrome://extensions.`
    : `Version ${latest} of this extension is installed on this Mac (this is ${mine}). Restart Safari and add it again from ${s.extension.folder}/${browser}.`;
}

/** Only what the engine measured; a figure that is absent is not shown. */
export function metricsText(s: Snapshot): string {
  const m = s.metrics;
  const out: string[] = [];
  if (m.model_gb !== undefined) out.push(`model ${m.model_gb.toFixed(1)} GB`);
  if (m.system_ram_used !== undefined) out.push(`RAM ${(m.system_ram_used * 100).toFixed(0)}%`);
  if (m.context_tokens !== undefined) out.push(`context ${Math.round(m.context_tokens / 1024)}K`);
  if (m.last_decode_tps !== undefined) out.push(`${m.last_decode_tps.toFixed(0)} tok/s`);
  if (s.in_use.inflight) out.push(`${s.in_use.inflight} in flight`);
  return out.join(" · ");
}

export function sharedText(sh: Shared): string {
  const parts: string[] = [];
  if (sh.selection_chars) parts.push(`selection, ${sh.selection_chars.toLocaleString()} characters`
    + (sh.selection_truncated ? ` of ${sh.selection_chars_total.toLocaleString()}` : ""));
  if (sh.page_chars) parts.push(`page text, ${sh.page_chars.toLocaleString()} characters`
    + (sh.page_truncated ? ` of ${sh.page_chars_total.toLocaleString()} (cut to fit the model)` : ""));
  let host = "";
  try {
    host = new URL(sh.url).host;
  } catch {
    host = "";
  }
  return `Shared with the local model${host ? ` from ${host}` : ""}: ${parts.join("; ") || "nothing"}.`;
}

/** An answer area that follows a chat stream. Text is always set as text, never as HTML. */
export class Answer {
  readonly root = el("div", { class: "answer" });
  private shared = el("div", { class: "shared" });
  private think = el("details", { class: "think" }, el("summary", {}, "Reasoning"));
  private thinkBody = el("div", { class: "think-body" });
  private body = el("div", { class: "body" });
  private note = el("div", { class: "note" });
  active = false;

  constructor() {
    this.think.append(this.thinkBody);
    this.think.hidden = true;
    this.root.append(this.shared, this.think, this.body, this.note);
    this.root.setAttribute("aria-live", "polite");
    this.root.hidden = true;
  }

  begin(): void {
    this.root.hidden = false;
    this.active = true;
    this.shared.textContent = "Reading the page…";
    this.body.textContent = "";
    this.thinkBody.textContent = "";
    this.think.hidden = true;
    this.note.textContent = "";
    this.note.className = "note";
  }

  fail(message: string): void {
    this.root.hidden = false;
    this.active = false;
    this.note.textContent = message;
    this.note.className = "note bad";
  }

  on(ev: ChatEvent): void {
    if (ev.type === "start") this.shared.textContent = sharedText(ev.shared);
    else if (ev.type === "reasoning") {
      this.think.hidden = false;
      this.thinkBody.textContent += ev.text;
    } else if (ev.type === "delta") {
      this.body.textContent += ev.text;
      this.root.scrollTop = this.root.scrollHeight;
    } else {
      this.active = false;
      if (ev.type === "cancelled") this.note.textContent = "Stopped.";
      else if (ev.type === "error") this.fail(ev.message);
      else if (ev.finish_reason === "length") this.note.textContent = "The answer reached its length limit.";
    }
  }

  text(): string {
    return this.body.textContent || "";
  }
}
