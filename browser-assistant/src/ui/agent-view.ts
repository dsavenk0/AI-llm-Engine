// The agent's face: what it is doing, and — above all — the card that asks before it does it. Plain DOM; every
// piece of text from a page or a model is set as text, never as markup.

import { AgentView, TaskState } from "../core/agent-types";
import { el } from "./view";

export interface AgentHandlers {
  approve(): void;
  reject(): void;
  cancel(): void;
  reply(text: string): void;
}

const LINE: Partial<Record<TaskState, string>> = {
  requested: "Starting…", planning: "Looking at the page and deciding the next step…", executing: "Doing it…",
  verifying: "Checking what happened…", completed: "Done.", failed: "Stopped.", cancelled: "Stopped.",
};

export function stateLine(v: AgentView): string {
  if (v.state === "approval_required") return "Waiting for your answer";
  if (v.state === "waiting_for_user") return "Waiting for you";
  if (v.state === "waiting_for_access") return "Needs access to a site";
  return LINE[v.state] || v.state;
}

export function isActive(v: AgentView | null): boolean {
  return !!v && v.state !== "completed" && v.state !== "failed" && v.state !== "cancelled";
}

/** Rebuild `root` from `v`. `trusted` is the check that a click came from a person, not from a script. */
export function renderAgent(root: HTMLElement, v: AgentView | null, h: AgentHandlers): void {
  root.replaceChildren();
  root.hidden = !v;
  if (!v) return;
  const trusted = (e: Event, fn: () => void): void => { if (e.isTrusted) fn(); };     // a page script cannot click for you

  root.append(el("div", { class: "agent-goal" }, el("strong", {}, "Task: "), v.goal));
  const status = el("div", { class: "agent-state", textContent: stateLine(v) });
  status.setAttribute("role", "status");
  root.append(status);

  if (v.steps.length) {
    const list = el("ol", { class: "agent-steps" });
    for (const s of v.steps) list.append(el("li", { class: s.ok ? "" : "bad" }, `${s.summary} — ${s.result}`));
    root.append(list);
  }

  if (v.state === "approval_required" && v.proposal) {
    const p = v.proposal.preview;
    const card = el("div", { class: "approval" + (p.risk === "high" ? " high" : "") });
    card.setAttribute("role", "group");
    card.setAttribute("aria-label", "Approve this step?");
    card.append(el("div", { class: "approval-title" }, p.title));
    for (const line of p.lines) card.append(el("div", { class: "approval-line" }, line));
    if (p.reasons.length) {
      const ul = el("ul", { class: "approval-reasons" });
      for (const r of p.reasons) ul.append(el("li", {}, "⚠ " + r));
      card.append(ul);
    }
    const yes = el("button", { type: "button", class: "primary", textContent: p.risk === "high" ? "Approve anyway" : "Approve" });
    const no = el("button", { type: "button", textContent: "Decline" });
    yes.addEventListener("click", (e) => trusted(e, h.approve));
    no.addEventListener("click", (e) => trusted(e, h.reject));
    card.append(el("div", { class: "approval-buttons" }, yes, no));
    root.append(card);
    yes.focus({ preventScroll: true });
  } else if (v.state === "waiting_for_user") {
    root.append(el("div", { class: "agent-ask" }, v.message || "What should I do?"));
    const box = el("input", { type: "text", placeholder: "Your answer…" });
    box.setAttribute("aria-label", "Your answer");
    const send = el("button", { type: "button", class: "primary", textContent: "Send" });
    const go = (): void => { if (box.value.trim()) h.reply(box.value.trim()); };
    send.addEventListener("click", (e) => trusted(e, go));
    box.addEventListener("keydown", (e) => { if (e.key === "Enter" && e.isTrusted) go(); });
    root.append(el("div", { class: "agent-reply" }, box, send));
  } else if (v.state === "waiting_for_access") {
    root.append(el("div", { class: "agent-ask" }, v.message || "The extension needs access to this site."),
                el("div", { class: "note" }, "Press Allow in the side panel."));
  } else if (v.message) {
    root.append(el("div", { class: v.state === "failed" ? "note bad" : "agent-ask" }, v.message));
  }

  if (isActive(v)) {
    const stop = el("button", { type: "button", class: "danger", textContent: "Stop" });
    stop.addEventListener("click", (e) => trusted(e, h.cancel));
    root.append(el("div", { class: "agent-stop" }, stop));
  }
}
