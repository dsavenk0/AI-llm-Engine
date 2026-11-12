// What the page shows while the agent works on it: a glowing frame around the viewport, a pill at the top that says
// so with a Stop button, and a line at the bottom naming what it is doing right now. The person can always see
// that the tab is being driven, and stop it from the page itself.
//
// It lives in a closed shadow root (the page's styles cannot restyle it, its scripts cannot read it), lets every
// click through except on the pill, and Stop only acts on a real click — a page script cannot press it, and it
// can only ever *stop* the agent, never start or approve anything.

import { api } from "../browser-adapters/api";

export interface OverlayState {
  active: boolean;
  label: string;           // what it is doing: "Reading the page", "Click “Send”"…
  waiting: boolean;        // waiting for the person: the frame stops pulsing and the pill says where to answer
}

const HOST = "ai-llm-engine-assistant-overlay";

const CSS = `
:host { all: initial; }
.frame { position: fixed; inset: 0; pointer-events: none; border: 2px solid rgba(0, 229, 255, .85);
  box-shadow: inset 0 0 28px rgba(0, 229, 255, .45), inset 0 0 4px rgba(0, 229, 255, .9);
  animation: pulse 2.4s ease-in-out infinite; }
.frame.waiting { animation: none; border-color: rgba(255, 240, 0, .9);
  box-shadow: inset 0 0 26px rgba(255, 240, 0, .35); }
@keyframes pulse { 50% { box-shadow: inset 0 0 44px rgba(0, 229, 255, .6), inset 0 0 6px rgba(0, 229, 255, 1); } }
@media (prefers-reduced-motion: reduce) { .frame { animation: none; } }
.pill { position: fixed; top: 12px; left: 50%; transform: translateX(-50%); pointer-events: auto;
  display: flex; align-items: center; gap: 12px; padding: 7px 7px 7px 16px; border-radius: 999px;
  background: rgba(4, 10, 16, .92); border: 1px solid rgba(0, 229, 255, .5); color: #dcf2fa;
  box-shadow: 0 8px 28px rgba(0, 0, 0, .45), 0 0 18px rgba(0, 229, 255, .25);
  font: 600 13px/1.2 -apple-system, BlinkMacSystemFont, "Helvetica Neue", sans-serif; max-width: calc(100vw - 32px); }
.mark { color: #00e5ff; font: 700 12px/1 ui-monospace, Menlo, monospace; letter-spacing: .02em; }
.text { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.stop { all: unset; cursor: pointer; display: inline-flex; align-items: center; gap: 6px; padding: 6px 12px;
  border-radius: 999px; background: #f2f7fa; color: #05080b; font: 600 12px/1 -apple-system, BlinkMacSystemFont, sans-serif; }
.stop:hover { background: #ffffff; }
.stop:focus-visible { outline: 2px solid #00e5ff; outline-offset: 2px; }
.stop i { width: 9px; height: 9px; border-radius: 2px; background: currentColor; display: inline-block; }
.label { position: fixed; left: 14px; bottom: 14px; pointer-events: none; padding: 5px 10px; border-radius: 8px;
  background: rgba(0, 229, 255, .92); color: #031014; font: 600 12px/1.2 -apple-system, BlinkMacSystemFont, sans-serif;
  max-width: calc(100vw - 28px); white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
  box-shadow: 0 4px 14px rgba(0, 0, 0, .35); }
.waiting ~ .label { background: rgba(255, 240, 0, .95); }
.cursor { position: fixed; left: 0; top: 0; width: 26px; height: 30px; pointer-events: none; opacity: 0;
  transition: transform .45s cubic-bezier(.3, .7, .2, 1), opacity .2s; filter: drop-shadow(0 0 6px rgba(0, 229, 255, .8)); }
.cursor.on { opacity: 1; }
.ring { position: fixed; pointer-events: none; border-radius: 50%; border: 2px solid #00e5ff; width: 14px; height: 14px;
  margin: -7px 0 0 -7px; animation: ring .55s ease-out forwards; }
@keyframes ring { to { width: 54px; height: 54px; margin: -27px 0 0 -27px; opacity: 0; } }
.focus { position: fixed; pointer-events: none; border: 2px solid #00e5ff; border-radius: 6px;
  box-shadow: 0 0 0 4px rgba(0, 229, 255, .18), 0 0 18px rgba(0, 229, 255, .45); animation: fade 1.1s ease-out forwards; }
@keyframes fade { 0%, 60% { opacity: 1; } 100% { opacity: 0; } }
@media (prefers-reduced-motion: reduce) { .cursor { transition: opacity .2s; } .ring { animation-duration: .01s; } }
`;

export function createOverlay(doc: Document) {
  let host: HTMLElement | null = null;
  let frame: HTMLElement, text: HTMLElement, label: HTMLElement, cursor: HTMLElement, root: ShadowRoot;
  let at = { x: -1, y: -1 };

  function build(): void {
    host = doc.createElement("div");
    host.id = HOST;
    host.style.cssText = "all: initial; position: fixed; inset: 0; z-index: 2147483646; pointer-events: none;";
    root = host.attachShadow({ mode: "closed" });
    const style = doc.createElement("style");
    style.textContent = CSS;
    frame = doc.createElement("div");
    frame.className = "frame";
    const pill = doc.createElement("div");
    pill.className = "pill";
    pill.setAttribute("role", "status");
    const mark = doc.createElement("span");
    mark.className = "mark";
    mark.textContent = "AI>_";
    text = doc.createElement("span");
    text.className = "text";
    const stop = doc.createElement("button");
    stop.className = "stop";
    stop.type = "button";
    stop.append(doc.createElement("i"), "Stop");
    stop.addEventListener("click", (e) => {
      if (!e.isTrusted) return;                               // only a person stops it from here
      try {
        const port = api.runtime.connect({ name: "overlay" });
        port.postMessage({ kind: "agent-cancel" });
        setTimeout(() => port.disconnect(), 500);
      } catch {
        // the extension was reloaded: there is nothing left to stop
      }
      hide();
    });
    pill.append(mark, text, stop);
    label = doc.createElement("div");
    label.className = "label";
    cursor = doc.createElement("div");
    cursor.className = "cursor";
    // an arrow pointer drawn in the frame's colour, so it is never mistaken for the person's own
    cursor.innerHTML = '<svg viewBox="0 0 26 30" width="26" height="30" aria-hidden="true"><path d="M3 2 L3 24 L9 18.5 ' +
      'L13 27.5 L17 25.8 L13 17 L21 17 Z" fill="#00e5ff" stroke="#031014" stroke-width="1.6" stroke-linejoin="round"/></svg>';
    root.append(style, frame, pill, label, cursor);
    doc.documentElement.append(host);
  }

  function hide(): void {
    host?.remove();
    host = null;
    at = { x: -1, y: -1 };
  }

  function show(s: OverlayState): boolean {
    if (!s.active) {
      hide();
      return true;
    }
    if (!host || !host.isConnected) build();
    frame.className = "frame" + (s.waiting ? " waiting" : "");
    text.textContent = s.waiting ? "LLMBar is waiting for you in the side panel" : "LLMBar is working here";
    label.textContent = s.label;
    label.hidden = !s.label;
    return true;
  }

  const still = (): boolean => doc.defaultView?.matchMedia?.("(prefers-reduced-motion: reduce)").matches === true;
  const pause = (ms: number) => new Promise((r) => setTimeout(r, ms));

  /** Move the visible cursor onto an element before the agent acts on it, so the person sees where it goes. */
  async function pointTo(r: { x: number; y: number; w: number; h: number }): Promise<void> {
    if (!host || !host.isConnected) return;                   // the frame is not up: nothing is being shown
    const x = Math.round(r.x + Math.min(r.w / 2, 40)), y = Math.round(r.y + r.h / 2);
    const win = doc.defaultView!;
    if (at.x < 0) {                                            // first appearance: come in from the bottom centre
      cursor.style.transition = "none";
      cursor.style.transform = `translate(${win.innerWidth / 2}px, ${win.innerHeight - 40}px)`;
      void cursor.offsetWidth;
      cursor.style.transition = "";
    }
    cursor.classList.add("on");
    cursor.style.transform = `translate(${x - 3}px, ${y - 2}px)`;
    at = { x, y };
    await pause(still() ? 60 : 480);
  }

  /** What the person sees as the action lands: a ripple for a click or a key, an outline round a filled field. */
  function mark(kind: "click" | "type", r?: { x: number; y: number; w: number; h: number }): void {
    if (!host || !host.isConnected) return;
    const n = doc.createElement("div");
    if (kind === "type" && r) {
      n.className = "focus";
      Object.assign(n.style, { left: `${r.x - 3}px`, top: `${r.y - 3}px`, width: `${r.w + 6}px`, height: `${r.h + 6}px` });
    } else {
      n.className = "ring";
      Object.assign(n.style, { left: `${at.x}px`, top: `${at.y}px` });
    }
    root.append(n);
    setTimeout(() => n.remove(), 1200);
  }

  return { show, pointTo, mark };
}
