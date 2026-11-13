// The agent's conductor: one task per tab, from a goal to a finished job, one approved step at a time.
//
//   snapshot the page → ask the model for the next action → check it against the live element → show the person
//   exactly what it will do → on approval do it → look at what happened → repeat
//
// It owns the rules that make this safe and nothing else does: every click, fill and navigation waits for a
// person; an action is carried out only on the tab, address and snapshot it was proposed for; a step that fails
// is reported, never retried blindly; a task ends after a handful of steps or two failures in a row. The page,
// the service, the tabs and the clock are handed in, so all of it runs under test without a browser.

import { Action, AgentView, AuditEntry, Autonomy, ElementInfo, PageSnapshot, Preview, StepView, TaskState } from "../core/agent-types";
import { MAX_FAILURES, MAX_STEPS, MAX_STEPS_AUTO, originOf, previewFor, refusal, summarize } from "../core/policy";
import { Binding, TERMINAL, transition } from "../core/task";
import type { Expect, Outcome, Verdict } from "../content/agent-core";

export interface CapturedFrame { screenshot: string; width: number; height: number; signature: string }

export interface PageBridge {
  ensure(tabId: number): Promise<void>;                       // put the page agent in the tab; throws if it may not
  panelShown(tabId: number): Promise<boolean>;
  showPanel(tabId: number): Promise<void>;
  snapshot(tabId: number): Promise<PageSnapshot>;
  find(tabId: number, query: string): Promise<{ generation: number; elements: ElementInfo[] }>;
  inspect(tabId: number, ref: string, expect: Expect): Promise<Verdict>;
  click(tabId: number, ref: string, expect: Expect): Promise<Outcome>;
  fill(tabId: number, ref: string, value: string, expect: Expect): Promise<Outcome>;
  undo?(tabId: number, token: string): Promise<Outcome>;
  rememberDictationTarget?(tabId: number): Promise<Outcome>;
  dictate?(tabId: number, text: string): Promise<Outcome>;
  scroll(tabId: number, direction: "up" | "down", ref?: string): Promise<Outcome>;
  press(tabId: number, key: string, ref: string | undefined, expect: Expect): Promise<Outcome>;
}

export interface Deps {
  page: PageBridge;
  step(req: { session: string; goal: string; snapshot: PageSnapshot; history: { summary: string; result: string }[];
    requestId: string; tabId: number; screenshot?: string | null; frame?: Omit<CapturedFrame, "screenshot"> }, signal: AbortSignal):
    Promise<{ action: Action; saw_screenshot?: boolean; described_screenshot?: boolean }>;
  cancelStep(requestId: string, session: string): Promise<void>;
  tab(tabId: number): Promise<{ url: string; loading: boolean } | null>;      // null: the tab is gone
  navigate(tabId: number, url: string): Promise<void>;
  settle(tabId: number): Promise<void>;                                        // wait for a tab to finish loading
  notify(tabId: number, view: AgentView | null): void;
  audit(entry: AuditEntry): void;
  session(tabId: number): string;
  id(): string;
  capture?(tabId: number): Promise<string | CapturedFrame>;                    // a picture of the tab, for a model that sees
  visualExecute?(req: { session: string; ticket: string; tabId: number; url: string; screenshot: string;
    frame?: Omit<CapturedFrame, "screenshot"> }): Promise<{ ok: boolean; error?: string; route?: string }>;
  workspaceExecute?(req: { session: string; ticket: string; url: string }): Promise<{ ok: boolean; service?: string; error?: string;
    undo?: string; undo_label?: string }>;
  workspaceUndo?(req: { session: string; undo: string }): Promise<{ ok: boolean }>;
}

interface Proposal { action: Action; preview: Preview; live: Record<string, ElementInfo>; snapshot: PageSnapshot }
type Decision = "approve" | "reject" | "cancel";

interface Task {
  id: string; bind: Binding; goal: string; state: TaskState;
  steps: StepView[]; history: { summary: string; result: string }[];
  stepsLeft: number; failures: number; autonomy: Autonomy; see: boolean;
  proposal?: Proposal; message?: string; needsAccess?: string;
  abort: AbortController; requestId: string;
  waiter: ((d: Decision | { reply: string } | "resume") => void) | null;
  undo?: { kind: "workspace"; token: string; label: string } | { kind: "page"; tokens: string[]; label: string };
  route?: "DOM" | "vision" | "macOS" | "Workspace";
}

export class AgentRunner {
  private tasks = new Map<number, Task>();
  constructor(private d: Deps) {}

  view(tabId: number): AgentView | null {
    const t = this.tasks.get(tabId);
    if (!t) return null;
    return {
      id: t.id, tabId: t.bind.tabId, goal: t.goal, state: t.state, origin: t.bind.origin, steps: t.steps.slice(-8),
      stepsLeft: t.stepsLeft, autonomy: t.autonomy, proposal: t.state === "approval_required" && t.proposal ? { action: t.proposal.action, preview: t.proposal.preview } : undefined,
      message: t.message, needsAccess: t.needsAccess,
      doing: t.proposal && (t.state === "executing" || t.state === "verifying" || t.state === "approval_required")
        ? t.proposal.preview.title : undefined,
      undo: t.undo ? { label: t.undo.label } : undefined,
      route: t.route,
    };
  }

  /** Whether a task is running on any tab (an update waits for it). */
  anyActive(): boolean {
    for (const t of this.tasks.values()) if (!TERMINAL.has(t.state)) return true;
    return false;
  }

  active(tabId: number): boolean {
    const t = this.tasks.get(tabId);
    return !!t && !TERMINAL.has(t.state);
  }

  private go(t: Task, state: TaskState, message?: string): void {
    t.state = transition(t.state, state);
    t.message = message;
    if (state !== "waiting_for_access") t.needsAccess = undefined;
    this.d.notify(t.bind.tabId, this.view(t.bind.tabId));
  }

  private log(t: Task, kind: AuditEntry["kind"], what: string, ok?: boolean): void {
    this.d.audit({ t: Date.now(), task: t.id, origin: t.bind.origin, kind, what, ok });
  }

  private record(t: Task, summary: string, result: string, ok: boolean): void {
    t.steps.push({ summary, result, ok });
    t.history.push({ summary, result });
    t.failures = ok ? 0 : t.failures + 1;
  }

  /** Begin a task on a tab. Resolves once it is running; the task goes on in the background. */
  async start(tabId: number, goal: string, autonomy: Autonomy = "ask", see = false): Promise<AgentView> {
    if (this.active(tabId)) throw new Error("a task is already running on this tab");
    const tab = await this.d.tab(tabId);
    if (tab && !tab.url) throw new Error("I cannot see this tab's address yet — allow LLMBar on this site (send again "
      + "and choose Allow), or reload the extension if it was just updated");
    const origin = tab ? originOf(tab.url) : "";
    if (!origin) throw new Error("the agent works only on web pages (http and https), not on the browser's own pages");
    const t: Task = { id: this.d.id(), bind: { tabId, origin }, goal: goal.trim().slice(0, 2000), state: "requested", steps: [],
                      history: [], stepsLeft: autonomy === "auto" ? MAX_STEPS_AUTO : MAX_STEPS, failures: 0, autonomy, see, abort: new AbortController(), requestId: this.d.id(), waiter: null };
    if (!t.goal) throw new Error("say what to do");
    this.tasks.set(tabId, t);
    this.log(t, "start", t.goal.slice(0, 80));
    void this.run(t).catch((e) => this.fail(t, `something went wrong: ${e instanceof Error ? e.message : String(e)}`));
    return this.view(tabId)!;
  }

  private fail(t: Task, why: string): void {
    if (TERMINAL.has(t.state)) return;
    this.log(t, "stop", why.slice(0, 120), false);
    this.go(t, "failed", why);
  }

  approve(tabId: number): void { this.decide(tabId, "approve"); }
  reject(tabId: number): void { this.decide(tabId, "reject"); }
  reply(tabId: number, text: string): void {
    const t = this.tasks.get(tabId);
    if (t && t.state === "waiting_for_user" && t.waiter) { const w = t.waiter; t.waiter = null; w({ reply: text.slice(0, 1000) }); }
  }
  resume(tabId: number): void {
    const t = this.tasks.get(tabId);
    if (t && t.state === "waiting_for_access" && t.waiter) { const w = t.waiter; t.waiter = null; w("resume"); }
  }

  private decide(tabId: number, d: Decision): void {
    const t = this.tasks.get(tabId);
    if (t && t.state === "approval_required" && t.waiter) { const w = t.waiter; t.waiter = null; w(d); }
  }

  /** Stop, whatever it is doing. Safe to call twice, or on a task that has finished. */
  cancel(tabId: number): void {
    const t = this.tasks.get(tabId);
    if (!t || TERMINAL.has(t.state)) return;
    t.abort.abort();
    void this.d.cancelStep(t.requestId, this.d.session(tabId)).catch(() => undefined);
    if (t.waiter) { const w = t.waiter; t.waiter = null; w("cancel"); }
    this.log(t, "stop", "cancelled by the user");
    this.go(t, "cancelled", "Stopped.");
  }

  async undo(tabId: number): Promise<void> {
    const t = this.tasks.get(tabId), item = t?.undo;
    if (!t || !item) return;
    t.undo = undefined;                         // one click, even if the network result is uncertain
    try {
      let ok = false;
      if (item.kind === "workspace") {
        if (!this.d.workspaceUndo) throw new Error("Workspace undo is not available");
        ok = (await this.d.workspaceUndo({ session: this.d.session(tabId), undo: item.token })).ok;
      } else {
        if (!this.d.page.undo) throw new Error("page undo is not available");
        ok = true;
        for (const token of item.tokens.slice().reverse()) {
          if (!(await this.d.page.undo(tabId, token)).ok) { ok = false; break; }
        }
      }
      this.record(t, `undo ${item.label}`, ok ? "undone" : "could not undo", ok);
    } catch (e) {
      this.record(t, `undo ${item.label}`, `could not undo: ${e instanceof Error ? e.message : String(e)}`, false);
    }
    this.d.notify(tabId, this.view(tabId));
  }

  private waitFor<T>(t: Task): Promise<T> {
    return new Promise((resolve) => { t.waiter = resolve as never; });
  }

  private stopped(t: Task): boolean {
    return TERMINAL.has(t.state);
  }

  private async run(t: Task): Promise<void> {
    const tabId = t.bind.tabId;
    this.go(t, "planning");
    let snapshot: PageSnapshot | null = null;
    while (!this.stopped(t)) {
      if (t.stepsLeft <= 0) {
        const n = t.autonomy === "auto" ? MAX_STEPS_AUTO : MAX_STEPS;
        return this.fail(t, `I stopped after ${n} steps without finishing. Tell me where to go on from.`);
      }
      if (t.failures >= MAX_FAILURES) return this.fail(t, "Two steps in a row did not work, so I stopped rather than guess. " + (t.steps.at(-1)?.result || ""));
      t.proposal = undefined;                              // what it was doing is over: the next look begins
      if (t.state !== "planning") this.go(t, "planning");

      // where is the tab? (a click or the page itself may have moved it)
      const tab = await this.d.tab(tabId);
      if (this.stopped(t)) return;
      if (!tab) return this.fail(t, "The tab was closed.");
      const now = originOf(tab.url);
      if (!now) return this.fail(t, "The tab is no longer on an http or https page.");
      if (now !== t.bind.origin) { t.bind = { tabId, origin: now }; snapshot = null; this.log(t, "navigate", `now on ${now}`); }

      if (!snapshot) {
        try {
          await this.d.page.ensure(tabId);
          snapshot = await this.d.page.snapshot(tabId);
          if (!(await this.d.page.panelShown(tabId))) await this.d.page.showPanel(tabId).catch(() => undefined);
        } catch {
          t.needsAccess = t.bind.origin;
          this.go(t, "waiting_for_access", `I need permission to read ${t.bind.origin}. Allow the extension on that site, then continue.`);
          t.needsAccess = t.bind.origin;
          this.d.notify(tabId, this.view(tabId));
          const r = await this.waitFor<string>(t);
          if (r === "cancel" || this.stopped(t)) return;
          continue;
        }
      }

      // with "let the model see the page" on, a picture of the tab goes with the element list; a picture that
      // cannot be taken (the tab is behind another, the browser refused) is skipped, never a reason to stop
      let screenshot: string | null = null;
      let frame: Omit<CapturedFrame, "screenshot"> | undefined;
      if (t.see && this.d.capture) {
        const captured = await this.d.capture(tabId).catch(() => null);
        if (typeof captured === "string") screenshot = captured;
        else if (captured) ({ screenshot, ...frame } = captured);
      }
      if (this.stopped(t)) return;
      let action: Action;
      try {
        const decision = await this.d.step({ session: this.d.session(tabId), goal: t.goal, snapshot, history: t.history.slice(-6),
                                             requestId: t.requestId, tabId, screenshot, frame }, t.abort.signal);
        action = decision.action;
        t.route = action.action.startsWith("visual_") ? "macOS" : action.action.startsWith("workspace_") ? "Workspace"
          : decision.saw_screenshot || decision.described_screenshot ? "vision" : "DOM";
      } catch (e) {
        if (this.stopped(t) || t.abort.signal.aborted) return;
        return this.fail(t, `The model could not decide the next step: ${e instanceof Error ? e.message : String(e)}`);
      }
      if (this.stopped(t)) return;
      t.requestId = this.d.id();

      if (action.action === "done") {
        this.record(t, "finish", action.answer || "done", true);
        this.log(t, "stop", "done", true);
        return this.go(t, "completed", action.answer || "Done.");
      }
      if (action.action === "ask_user") {
        this.go(t, "waiting_for_user", action.question);
        const r = await this.waitFor<{ reply: string } | "cancel">(t);
        if (r === "cancel" || this.stopped(t)) return;
        this.record(t, `asked: ${action.question}`, `the user said: ${(r as { reply: string }).reply}`, true);
        t.stepsLeft -= 1;
        continue;
      }
      if (action.action === "find") {
        const found = await this.d.page.find(tabId, action.query);
        const names = found.elements.map((e) => `${e.ref} ${e.role} “${e.name}”`).join("; ");
        this.record(t, summarize(action, {}), found.elements.length ? `found: ${names}` : "nothing matched", true);
        t.stepsLeft -= 1;
        const have = new Set(snapshot.elements.map((e) => e.ref));
        snapshot = { ...snapshot, elements: [...snapshot.elements, ...found.elements.filter((e) => !have.has(e.ref))] };
        continue;                                          // same page, same references: ask again with the find results in
      }
      if (action.action === "scroll" || action.action === "wait") {
        // looking, not changing: no question to ask. The next step reads the page afresh.
        let result = "done";
        if (action.action === "scroll") {
          const r = await this.d.page.scroll(tabId, action.direction, action.ref).catch((e) => ({ ok: false, error: String(e) } as Outcome));
          result = r.ok ? (r.readback || "scrolled") : `could not scroll: ${r.error}`;
          this.record(t, summarize(action, {}), result, r.ok);
        } else {
          await new Promise((r) => setTimeout(r, Math.max(1, Math.min(10, action.seconds)) * 1000));
          if (this.stopped(t)) return;
          this.record(t, summarize(action, {}), "waited", true);
        }
        t.stepsLeft -= 1;
        snapshot = null;
        continue;
      }

      // an action that changes something: look at the live elements, then ask
      const expect: Expect = { generation: snapshot.generation, url: snapshot.url };
      const refs = action.action === "click" || action.action === "fill" ? [action.ref]
        : action.action === "press" ? (action.ref ? [action.ref] : [])
        : action.action === "fill_form" ? action.fields.map((f) => f.ref) : [];
      const live: Record<string, ElementInfo> = {};
      let problem = "";
      for (const ref of refs) {
        const v = await this.d.page.inspect(tabId, ref, expect);
        if (v.ok) live[ref] = v.info; else { problem = v.error; break; }
      }
      const why = problem || refusal(action, live, snapshot.url);
      if (why) {
        this.log(t, "refuse", `${action.action}: ${why}`.slice(0, 120), false);
        this.record(t, summarize(action, live), `not done: ${why}`, false);
        t.stepsLeft -= 1;
        snapshot = null;
        continue;
      }
      const preview = previewFor(action, live, snapshot.url);
      t.proposal = { action, preview, live, snapshot };
      this.log(t, "propose", `${summarize(action, live)} (${preview.risk})`);
      let decision: Decision;
      if (t.autonomy === "auto" && preview.risk === "low") {
        decision = "approve";                              // low risk, and the person chose to let those go ahead
      } else {
        this.go(t, "approval_required");
        decision = await this.waitFor<Decision>(t);
      }
      if (decision === "cancel" || this.stopped(t)) return;
      if (decision === "reject") {
        this.log(t, "reject", summarize(action, live));
        this.record(t, summarize(action, live), "declined by the user", true);
        t.failures = 0;
        this.go(t, "waiting_for_user", "You declined that. Tell me what to do instead, or stop.");
        const r = await this.waitFor<{ reply: string } | "cancel">(t);
        if (r === "cancel" || this.stopped(t)) return;
        this.record(t, "asked what to do instead", `the user said: ${(r as { reply: string }).reply}`, true);
        snapshot = null;
        continue;
      }
      this.log(t, "approve", (t.state === "approval_required" ? "" : "on its own: ") + summarize(action, live));
      this.go(t, "executing");
      const outcome = await this.execute(t, action, live, snapshot, expect);
      if (this.stopped(t)) return;
      this.go(t, "verifying");
      this.log(t, "execute", `${summarize(action, live)} → ${outcome.text}`.slice(0, 160), outcome.ok);
      this.record(t, summarize(action, live), outcome.text, outcome.ok);
      t.stepsLeft -= 1;
      snapshot = null;                                     // whatever happened, the next step starts from a fresh look
      if (outcome.access) {
        t.needsAccess = outcome.access;
        this.go(t, "waiting_for_access", `The page moved to ${outcome.access}, which the extension may not read yet. Allow it, then continue.`);
        t.needsAccess = outcome.access;
        this.d.notify(tabId, this.view(tabId));
        const r = await this.waitFor<string>(t);
        if (r === "cancel" || this.stopped(t)) return;
      }
    }
  }

  /** Do one approved action — after checking, once more, that nothing it depends on has moved. */
  private async execute(t: Task, action: Action, live: Record<string, ElementInfo>, snap: PageSnapshot, expect: Expect):
    Promise<{ ok: boolean; text: string; access?: string; undo?: { token: string; label: string } }> {
    const tabId = t.bind.tabId;
    const tab = await this.d.tab(tabId);
    if (!tab) return { ok: false, text: "the tab was closed" };
    if (tab.url !== snap.url || originOf(tab.url) !== t.bind.origin) {
      return { ok: false, text: "the page changed while waiting for your answer, so nothing was done" };
    }
    try {
      if (action.action === "navigate") {
        await this.d.navigate(tabId, action.url);
        await this.d.settle(tabId);
        const after = await this.d.tab(tabId);
        const origin = after ? originOf(after.url) : "";
        this.log(t, "navigate", action.url.slice(0, 100));
        try { if (origin) await this.d.page.ensure(tabId); } catch { return { ok: true, text: `opened ${after?.url}`, access: origin || undefined }; }
        return { ok: true, text: `opened ${after?.url ?? action.url}` };
      }
      if (action.action === "visual_click" || action.action === "visual_type" || action.action === "visual_press") {
        if (!this.d.capture || !this.d.visualExecute) return { ok: false, text: "macOS visual actions are not available in this browser" };
        const captured = await this.d.capture(tabId).catch(() => "");
        const screenshot = typeof captured === "string" ? captured : captured?.screenshot || "";
        const frame = typeof captured === "string" || !captured ? undefined
          : { width: captured.width, height: captured.height, signature: captured.signature };
        if (!screenshot) return { ok: false, text: "I could not take a fresh screenshot, so I did not act" };
        const r = await this.d.visualExecute({ session: this.d.session(tabId), ticket: action.ticket, tabId,
                                                url: snap.url, screenshot, frame });
        return r.ok ? { ok: true, text: action.action === "visual_type" ? "typed through macOS" : "acted through macOS" }
                    : { ok: false, text: r.error || "the macOS action did not go through" };
      }
      if (action.action.startsWith("workspace_")) {
        if (!this.d.workspaceExecute || !("ticket" in action)) return { ok: false, text: "Google Workspace is not connected" };
        const r = await this.d.workspaceExecute({ session: this.d.session(tabId), ticket: action.ticket, url: snap.url });
        if (r.ok && r.undo) t.undo = { kind: "workspace", token: r.undo, label: r.undo_label || "Workspace change" };
        return r.ok ? { ok: true, text: `completed through Google ${r.service || "Workspace"}`,
                        undo: r.undo ? { token: r.undo, label: r.undo_label || "Workspace change" } : undefined }
                    : { ok: false, text: r.error || "the Workspace operation did not go through" };
      }
      if (action.action === "click") {
        let r: Outcome;
        try {
          r = await this.d.page.click(tabId, action.ref, expect);
        } catch {
          // the click made the page unload: the script's answer never came. That is a navigation, not a failure.
          await this.d.settle(tabId);
          return { ok: true, text: "the page navigated" };
        }
        if (!r.ok) return { ok: false, text: r.error || "the click did not go through" };
        if (r.urlChanged) await this.d.settle(tabId);
        return { ok: true, text: r.urlChanged ? "the page address changed" : (r.mutations || 0) > 0 ? "the page changed" : "nothing visibly changed" };
      }
      if (action.action === "press") {
        let r: Outcome;
        try {
          r = await this.d.page.press(tabId, action.key, action.ref, expect);
        } catch {
          await this.d.settle(tabId);                      // Enter sent a form and the page unloaded
          return { ok: true, text: "the page navigated" };
        }
        if (!r.ok) return { ok: false, text: r.error || "the key did not go through" };
        if (r.urlChanged) await this.d.settle(tabId);
        return { ok: true, text: r.urlChanged ? "the page address changed" : (r.mutations || 0) > 0 ? "the page changed" : "nothing visibly changed" };
      }
      const fields = action.action === "fill" ? [{ ref: action.ref, value: action.value }] : action.action === "fill_form" ? action.fields : [];
      const done: string[] = [];
      const undoTokens: string[] = [];
      for (const f of fields) {
        const r = await this.d.page.fill(tabId, f.ref, f.value, expect);
        if (r.undo) undoTokens.push(r.undo);
        if (!r.ok) {
          if (undoTokens.length) t.undo = { kind: "page", tokens: undoTokens, label: `fields ${done.join(", ")}` };
          return { ok: false, text: `${done.length ? `filled ${done.join(", ")}, then ` : ""}could not fill ${live[f.ref]?.name || f.ref}: ${r.error}` };
        }
        done.push(live[f.ref]?.name || f.ref);
      }
      if (undoTokens.length) t.undo = { kind: "page", tokens: undoTokens, label: `fields ${done.join(", ")}` };
      return { ok: true, text: `filled ${done.join(", ")}` };
    } catch (e) {
      return { ok: false, text: `it did not work: ${e instanceof Error ? e.message : String(e)}` };
    }
  }
}
