import assert from "node:assert/strict";
import { test } from "node:test";
import { Action, AuditEntry, ElementInfo, PageSnapshot } from "../src/core/agent-types";
import { MAX_STEPS } from "../src/core/policy";
import { AgentRunner, Deps } from "../src/background/agent";

const el = (ref: string, over: Partial<ElementInfo> = {}): ElementInfo => ({
  ref, tag: "button", role: "button", name: ref, type: "", value: "", placeholder: "", href: "", disabled: false, checked: false,
  required: false, options: [], dup: 0, sensitive: false, blocked: false, inForm: false, formAction: "", formSensitive: false, ...over });
const URL0 = "https://shop.example/cart";
const snap = (els: ElementInfo[], over: Partial<PageSnapshot> = {}): PageSnapshot =>
  ({ generation: 1, url: URL0, title: "Cart", text: "", elements: els, captcha: false, truncated: false, ...over });
const tick = (ms = 0) => new Promise((r) => setTimeout(r, ms));
async function until(cond: () => boolean, what: string): Promise<void> {
  for (let i = 0; i < 200 && !cond(); i++) await tick(2);
  assert.ok(cond(), `timed out waiting for: ${what}`);
}

interface World { r: AgentRunner; log: string[]; audit: AuditEntry[]; views: any[]; tab: { url: string; loading: boolean } | null;
  actions: Action[]; elements: ElementInfo[]; clicks: string[]; fills: [string, string][]; cancelled: string[];
  failInspect?: string; clickOutcome: () => any; ensureFails: boolean; steps: number; delay: number }

function world(script: Action[], elements: ElementInfo[] = [el("e1", { name: "Add to cart" }), el("e2", { role: "textbox", tag: "input", name: "Search" })]): World {
  const w: World = { log: [], audit: [], views: [], tab: { url: URL0, loading: false }, actions: [...script], elements, clicks: [], fills: [],
                     cancelled: [], clickOutcome: () => ({ ok: true, urlChanged: false, mutations: 3 }), ensureFails: false, steps: 0, delay: 0 } as unknown as World;
  let n = 0;
  const d: Deps = {
    page: {
      ensure: async () => { if (w.ensureFails) throw new Error("no access"); },
      panelShown: async () => true, showPanel: async () => undefined,
      snapshot: async () => snap(w.elements),
      find: async (_t, q) => ({ generation: 1, elements: w.elements.filter((e) => e.name.includes(q)) }),
      inspect: async (_t, ref) => w.failInspect ? { ok: false, error: w.failInspect } : { ok: true, info: w.elements.find((e) => e.ref === ref)! },
      click: async (_t, ref) => { w.clicks.push(ref); return w.clickOutcome(); },
      fill: async (_t, ref, value) => { w.fills.push([ref, value]); return { ok: true, readback: value }; },
    },
    step: async (_req, signal) => {
      w.steps++;
      if (w.delay) await new Promise((res, rej) => { const t = setTimeout(res, w.delay); signal.addEventListener("abort", () => { clearTimeout(t); rej(new Error("aborted")); }); });
      return { action: w.actions.shift() ?? { action: "done", answer: "all done" } };
    },
    cancelStep: async (id) => { w.cancelled.push(id); },
    tab: async () => w.tab, navigate: async (_t, url) => { w.log.push("navigate " + url); w.tab = { url, loading: false }; },
    settle: async () => undefined,
    notify: (_t, view) => { w.views.push(view); }, audit: (e) => { w.audit.push(e); },
    session: (t) => `s${t}`, id: () => `00000000-0000-4000-8000-${String(++n).padStart(12, "0")}`,
  };
  w.r = new AgentRunner(d);
  return w;
}
const click = (ref: string): Action => ({ action: "click", ref, why: "needed" });
const state = (w: World) => w.r.view(1)?.state;

test("a click is proposed, waits for a person, and happens only after the approval", async () => {
  const w = world([click("e1")]);
  await w.r.start(1, "Add it");
  await until(() => state(w) === "approval_required", "the proposal");
  assert.deepEqual(w.clicks, []);                                              // nothing has happened yet
  const v = w.r.view(1)!;
  assert.equal(v.proposal!.preview.title, "Click “Add to cart”");
  await tick(30);
  assert.deepEqual(w.clicks, []);                                              // and waiting does not approve it
  w.r.approve(1);
  await until(() => state(w) === "completed", "completion");
  assert.deepEqual(w.clicks, ["e1"]);
  assert.equal(w.r.view(1)!.message, "all done");
  assert.deepEqual(w.audit.map((a) => a.kind), ["start", "propose", "approve", "execute", "stop"]);
});

test("a declined action is never carried out, and the task asks what to do instead", async () => {
  const w = world([click("e1"), { action: "done", answer: "ok, left it" }]);
  await w.r.start(1, "Add it");
  await until(() => state(w) === "approval_required", "the proposal");
  w.r.reject(1);
  await until(() => state(w) === "waiting_for_user", "the question");
  assert.deepEqual(w.clicks, []);
  assert.match(w.r.view(1)!.message!, /declined/);
  w.r.reply(1, "leave it");
  await until(() => state(w) === "completed", "completion");
  assert.deepEqual(w.clicks, []);
  assert.ok(w.r.view(1)!.steps.some((s) => s.result === "declined by the user"));
});

test("approve, reject and reply do nothing at any other moment", async () => {
  const w = world([click("e1")]);
  w.delay = 40;
  await w.r.start(1, "Add it");
  w.r.approve(1); w.r.reject(1); w.r.reply(1, "x");                           // the model is still thinking
  await until(() => state(w) === "approval_required", "the proposal");
  w.r.reply(1, "x"); w.r.resume(1);                                            // wrong kind of answer for this state
  await tick(20);
  assert.equal(state(w), "approval_required");
  assert.deepEqual(w.clicks, []);
  w.r.approve(1); w.r.approve(1);                                              // a double click is one approval
  await until(() => state(w) === "completed", "completion");
  assert.deepEqual(w.clicks, ["e1"]);
});

test("a fill is shown with its value and is typed only on approval; the log keeps the field, never the value", async () => {
  const w = world([{ action: "fill", ref: "e2", value: "red shoes", why: "search" }]);
  await w.r.start(1, "Search");
  await until(() => state(w) === "approval_required", "the proposal");
  assert.ok(w.r.view(1)!.proposal!.preview.lines.includes("“red shoes”"));
  assert.deepEqual(w.fills, []);
  w.r.approve(1);
  await until(() => state(w) === "completed", "completion");
  assert.deepEqual(w.fills, [["e2", "red shoes"]]);
  assert.ok(!JSON.stringify(w.audit).includes("red shoes"));
  assert.ok(!JSON.stringify(w.r.view(1)!.steps).includes("red shoes"));
});

test("a secret field is refused without ever asking the person, and two refusals in a row end the task", async () => {
  const w = world([{ action: "fill", ref: "e2", value: "hunter2", why: "" }, { action: "fill", ref: "e2", value: "hunter2", why: "" }],
                  [el("e2", { role: "textbox", tag: "input", name: "Password", sensitive: true })]);
  await w.r.start(1, "Log in");
  await until(() => state(w) === "failed", "failure");
  assert.deepEqual(w.fills, []);
  assert.ok(!w.views.some((v) => v && v.state === "approval_required"));      // the person was never bothered
  assert.match(w.r.view(1)!.message!, /Two steps in a row/);
  assert.ok(w.audit.filter((a) => a.kind === "refuse").length === 2);
  assert.ok(!JSON.stringify(w.audit).includes("hunter2"));
});

test("a click on a CAPTCHA, a disabled element or something that left the page is refused", async () => {
  for (const [els, why] of [[[el("e1", { blocked: true })], /CAPTCHA/], [[el("e1", { disabled: true })], /disabled/]] as const) {
    const w = world([click("e1"), click("e1")], [...els]);
    await w.r.start(1, "x");
    await until(() => state(w) === "failed", "failure");
    assert.deepEqual(w.clicks, []);
    assert.match(w.r.view(1)!.steps[0].result, why);
  }
  const w = world([click("e1"), click("e1")]);
  w.failInspect = "that element is no longer on the page";
  await w.r.start(1, "x");
  await until(() => state(w) === "failed", "failure");
  assert.deepEqual(w.clicks, []);
});

test("if the page changes while the person decides, nothing is done", async () => {
  const w = world([click("e1")]);
  await w.r.start(1, "Add it");
  await until(() => state(w) === "approval_required", "the proposal");
  w.tab = { url: "https://shop.example/somewhere-else", loading: false };      // the page moved on
  w.r.approve(1);
  await until(() => w.r.view(1)!.steps.length > 0, "the result");
  assert.deepEqual(w.clicks, []);
  assert.match(w.r.view(1)!.steps[0].result, /changed while waiting/);
});

test("a tab that closes between steps ends the task; one that is not an http page cannot start one", async () => {
  const w = world([{ action: "find", query: "x" }, click("e1")]);
  (w.r as any).d.page.find = async () => { w.tab = null; return { generation: 1, elements: [] }; };      // closed during a step
  await w.r.start(1, "Add it");
  await until(() => state(w) === "failed", "failure");
  assert.match(w.r.view(1)!.message!, /tab was closed/);
  assert.deepEqual(w.clicks, []);
  const w2 = world([]);
  w2.tab = { url: "chrome://settings", loading: false };
  await assert.rejects(w2.r.start(1, "x"), /http and https/);
});

test("the model is asked for at most eight steps, then the task stops and says so", async () => {
  const w = world(Array.from({ length: 20 }, () => ({ action: "find", query: "zzz" }) as Action));
  await w.r.start(1, "Look around");
  await until(() => state(w) === "failed", "the limit");
  assert.equal(w.steps, MAX_STEPS);
  assert.match(w.r.view(1)!.message!, new RegExp(`stopped after ${MAX_STEPS} steps`));
});

test("find adds what it found to the page the model is looking at, so its references work", async () => {
  const w = world([{ action: "find", query: "Pay" }, click("e9")],
                  [el("e1", { name: "Home" }), el("e9", { name: "Pay later" })]);
  w.elements = [el("e1", { name: "Home" })];                                   // the list the snapshot returned was cut
  const all = [el("e1", { name: "Home" }), el("e9", { name: "Pay later" })];
  let looked = false;
  (w.r as any).d.page.find = async () => { looked = true; w.elements = all; return { generation: 1, elements: [all[1]] }; };
  (w.r as any).d.page.snapshot = async () => snap([all[0]]);
  await w.r.start(1, "Pay");
  await until(() => state(w) === "approval_required", "the proposal for the found element");
  assert.ok(looked);
  assert.equal(w.r.view(1)!.proposal!.action.action, "click");
});

test("ask_user pauses the task until the person answers, and the answer goes back to the model", async () => {
  const w = world([{ action: "ask_user", question: "Which size?" }, { action: "done", answer: "noted" }]);
  await w.r.start(1, "Buy shoes");
  await until(() => state(w) === "waiting_for_user", "the question");
  assert.equal(w.r.view(1)!.message, "Which size?");
  w.r.reply(1, "42");
  await until(() => state(w) === "completed", "completion");
  assert.ok(w.r.view(1)!.steps.some((s) => s.result === "the user said: 42"));
});

test("cancel stops a task that is thinking, and one that is waiting for approval, and cancels the model call", async () => {
  const w = world([click("e1")]);
  w.delay = 200;
  await w.r.start(1, "x");
  await tick(10);
  w.r.cancel(1);
  assert.equal(state(w), "cancelled");
  assert.equal(w.cancelled.length, 1);
  await tick(250);
  assert.deepEqual(w.clicks, []);
  const w2 = world([click("e1")]);
  await w2.r.start(1, "x");
  await until(() => state(w2) === "approval_required", "the proposal");
  w2.r.cancel(1);
  w2.r.cancel(1);                                                              // twice is fine
  w2.r.approve(1);                                                             // and a late approval does nothing
  await tick(20);
  assert.equal(state(w2), "cancelled");
  assert.deepEqual(w2.clicks, []);
  assert.equal(w2.r.active(1), false);
  await w2.r.start(1, "again");                                                // the tab is free for a new task
});

test("one task per tab", async () => {
  const w = world([click("e1")]);
  await w.r.start(1, "x");
  await assert.rejects(w.r.start(1, "y"), /already running/);
});

test("navigation: a blocked address is refused unasked; an allowed one waits for approval; leaving the site is flagged", async () => {
  const w = world([{ action: "navigate", url: "javascript:alert(1)", why: "" }, { action: "navigate", url: "http://127.0.0.1:8093/v1/assistant/state", why: "" }]);
  await w.r.start(1, "go");
  await until(() => state(w) === "failed", "failure");
  assert.deepEqual(w.log, []);
  const w2 = world([{ action: "navigate", url: "https://other.example/x", why: "needed" }]);
  await w2.r.start(1, "go");
  await until(() => state(w2) === "approval_required", "the proposal");
  assert.equal(w2.r.view(1)!.proposal!.preview.risk, "high");
  assert.deepEqual(w2.log, []);
  w2.r.approve(1);
  await until(() => state(w2) === "completed", "completion");
  assert.deepEqual(w2.log, ["navigate https://other.example/x"]);
  assert.equal(w2.r.view(1)!.origin, "https://other.example");                // the task now lives on the new origin
});

test("a page the extension may not read pauses the task for access instead of failing it, and resumes", async () => {
  const w = world([click("e1")]);
  w.ensureFails = true;
  await w.r.start(1, "x");
  await until(() => state(w) === "waiting_for_access", "the access request");
  assert.equal(w.r.view(1)!.needsAccess, "https://shop.example");
  w.ensureFails = false;
  w.r.resume(1);
  await until(() => state(w) === "approval_required", "the proposal");
});

test("a click that unloads the page is a navigation, not a failure", async () => {
  const w = world([click("e1")]);
  w.clickOutcome = () => { throw new Error("the page went away"); };
  await w.r.start(1, "x");
  await until(() => state(w) === "approval_required", "the proposal");
  w.r.approve(1);
  await until(() => state(w) === "completed", "completion");
  assert.equal(w.r.view(1)!.steps[0].ok, true);
  assert.match(w.r.view(1)!.steps[0].result, /navigated/);
});

test("a failed fill reports which fields were done and which was not", async () => {
  const w = world([{ action: "fill_form", fields: [{ ref: "e2", value: "a" }, { ref: "e3", value: "b" }], why: "" }, { action: "done", answer: "" }],
                  [el("e2", { role: "textbox", name: "Name" }), el("e3", { role: "textbox", name: "City" })]);
  (w.r as any).d.page.fill = async (_t: number, ref: string, value: string) => { w.fills.push([ref, value]); return ref === "e3" ? { ok: false, error: "read-only" } : { ok: true }; };
  await w.r.start(1, "x");
  await until(() => state(w) === "approval_required", "the proposal");
  w.r.approve(1);
  await until(() => w.r.view(1)!.steps.length > 0, "the result");
  assert.equal(w.r.view(1)!.steps[0].ok, false);
  assert.match(w.r.view(1)!.steps[0].result, /filled Name, then could not fill City: read-only/);
});
