// The in-page panel: where an answer to a context-menu action appears. Injected into the active tab on that
// click only. It lives in a closed shadow root — the page's styles cannot reach it and the page's scripts
// cannot read the answer — and it holds no token: everything goes through the background script.

import { api } from "../browser-adapters/api";
import { UiEvent, UiRequest } from "../core/protocol";
import { Answer, el } from "../ui/view";
import { AgentView } from "../core/agent-types";
import { isActive, renderAgent } from "../ui/agent-view";
import base from "../ui/base.css";
import css from "../ui/panel.css";
import agentCss from "../ui/agent.css";

const HOST = "ai-llm-engine-assistant-panel";

function open(): void {
  document.getElementById(HOST)?.remove();
  const host = el("div", { id: HOST });
  host.style.cssText = "all: initial; position: fixed; z-index: 2147483647; right: 16px; bottom: 16px;";
  const root = host.attachShadow({ mode: "closed" });
  const style = el("style");
  // the shared variables are declared on :root for the extension's own pages; in a shadow tree that is :host
  style.textContent = base.replace(/:root/g, ":host") + css + agentCss;
  const answer = new Answer();
  const stop = el("button", { type: "button", textContent: "Stop", title: "Stop generating" });
  const copy = el("button", { type: "button", textContent: "Copy" });
  const close = el("button", { type: "button", textContent: "✕", title: "Close" });
  close.setAttribute("aria-label", "Close");
  const bar = el("div", { class: "bar" }, el("span", { class: "title" }, "Local assistant"), stop, copy, close);
  const agentRoot = el("div", { class: "agent" });
  agentRoot.hidden = true;
  const panel = el("div", { class: "panel" }, bar, agentRoot, answer.root);
  panel.setAttribute("role", "dialog");
  panel.setAttribute("aria-label", "Local assistant answer");
  root.append(style, panel);
  document.documentElement.append(host);

  const port = api.runtime.connect({ name: "panel" });
  const post = (m: UiRequest): void => port.postMessage(m);
  let agentView: AgentView | null = null;
  port.onMessage.addListener((ev: UiEvent) => {
    if (ev.kind === "agent") {
      agentView = ev.view;
      renderAgent(agentRoot, ev.view, {
        approve: () => post({ kind: "agent-approve" }), reject: () => post({ kind: "agent-reject" }),
        cancel: () => post({ kind: "agent-cancel" }), reply: (text) => post({ kind: "agent-reply", text }),
      });
      stop.hidden = true; copy.hidden = !!ev.view;
      if (ev.view) answer.root.hidden = true;
      else if (answer.root.hidden) answer.begin();                    // no agent task here: this panel is for an answer
      return;
    }
    if (ev.kind === "chat") {
      if (answer.root.hidden) answer.begin();
      answer.on(ev.event);
      stop.hidden = !answer.active;
    } else if (ev.kind === "fatal") {
      answer.fail(ev.message);
      stop.hidden = true;
    } else if (ev.kind === "state" && !ev.connected && ev.problem && ev.problem !== "connecting") {
      answer.fail("The assistant is not connected. Open the side panel (the toolbar button) to set it up.");
      stop.hidden = true;
    }
  });
  stop.addEventListener("click", () => post({ kind: "cancel" }));
  copy.addEventListener("click", () => void navigator.clipboard.writeText(answer.text()));
  const shut = (): void => {
    port.disconnect();                 // the background cancels a generation nobody is reading
    host.remove();
    document.removeEventListener("keydown", onKey, true);
  };
  const onKey = (e: KeyboardEvent): void => {
    if (e.key === "Escape") shut();
  };
  close.addEventListener("click", () => {
    if (isActive(agentView)) post({ kind: "agent-cancel" });           // closing the panel on purpose stops the agent
    shut();
  });
  document.addEventListener("keydown", onKey, true);
  post({ kind: "subscribe" });
}

open();
