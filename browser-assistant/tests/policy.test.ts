import assert from "node:assert/strict";
import { test } from "node:test";
import { Action, ElementInfo } from "../src/core/agent-types";
import { clickRisk, navigationRisk, previewFor, refusal, sensitiveReason, summarize } from "../src/core/policy";
import { NEXT, sameBinding, TERMINAL, transition, TransitionError } from "../src/core/task";

const el = (over: Partial<ElementInfo> = {}): ElementInfo => ({
  ref: "e1", tag: "button", role: "button", name: "Details", type: "", value: "", placeholder: "", href: "", disabled: false,
  checked: false, required: false, options: [], dup: 0, sensitive: false, blocked: false, inForm: false, formAction: "",
  formSensitive: false, formSearch: false, ...over });
const PAGE = "https://shop.example/cart";

test("secrets are recognised by type, autocomplete and by what a person would call the field", () => {
  assert.equal(sensitiveReason({ type: "password" }), "a password field");
  assert.match(sensitiveReason({ autocomplete: "cc-number" }) || "", /payment/);
  assert.match(sensitiveReason({ autocomplete: "one-time-code" }) || "", /payment/);
  assert.match(sensitiveReason({ autocomplete: "new-password" }) || "", /payment|one-time/);
  for (const hint of ["Card number", "CVV", "Security code", "Enter your PIN code", "OTP", "IBAN", "Пароль", "Номер карты"]) {
    assert.ok(sensitiveReason({ type: "text", hint }), hint);
  }
  for (const hint of ["Search", "Email address", "First name", "Passenger count", "Compass"]) {
    assert.equal(sensitiveReason({ type: "text", hint }), null, hint);
  }
});

test("a click is high-risk for what it does, and says why", () => {
  assert.equal(clickRisk(el({ name: "Show details" }), PAGE).risk, "low");
  for (const name of ["Buy now", "Place order", "Delete account", "Send", "Оплатить", "Remove from cart", "Confirm"]) {
    assert.equal(clickRisk(el({ name }), PAGE).risk, "high", name);
  }
  const submit = clickRisk(el({ type: "submit", inForm: true, formAction: "https://pay.example/charge" }), PAGE);
  assert.equal(submit.risk, "high");
  assert.ok(submit.reasons.some((r) => r.includes("submits a form to https://pay.example")));
  assert.ok(submit.reasons.some((r) => r.includes("another site")));
  assert.ok(clickRisk(el({ formSensitive: true }), PAGE).reasons[0].includes("password, card"));
  const away = clickRisk(el({ role: "link", tag: "a", href: "https://elsewhere.example/x" }), PAGE);
  assert.ok(away.reasons[0].includes("leaves this site for https://elsewhere.example"));
  assert.equal(clickRisk(el({ role: "link", tag: "a", href: "https://shop.example/other" }), PAGE).risk, "low");
  assert.ok(clickRisk(el({ dup: 3 }), PAGE).reasons[0].includes("3 elements look the same"));
});

test("navigation: only http(s), never credentials, never the assistant itself, flagged when it leaves", () => {
  for (const url of ["javascript:alert(1)", "data:text/html,x", "file:///etc/hosts", "about:blank", "blob:https://x/y", "ftp://x"]) {
    assert.equal(navigationRisk(url, PAGE).allowed, false, url);
  }
  assert.equal(navigationRisk("https://user:pw@evil.example/", PAGE).allowed, false);
  assert.equal(navigationRisk("not a url", PAGE).allowed, false);
  for (const url of ["http://127.0.0.1:8093/v1/assistant/state", "http://localhost:8090/health", "http://[::1]:8093/"]) {
    const v = navigationRisk(url, PAGE);
    assert.equal(v.allowed, false, url);
    assert.match(v.error || "", /local assistant or engine/);
  }
  const same = navigationRisk("https://shop.example/checkout", PAGE);
  assert.deepEqual([same.allowed, same.risk, same.reasons], [true, "low", []]);
  const out = navigationRisk("https://other.example/", PAGE);
  assert.deepEqual([out.allowed, out.risk], [true, "high"]);
  assert.ok(out.reasons[0].includes("leaves https://shop.example for https://other.example"));
  assert.ok(navigationRisk("http://192.168.1.5/admin", PAGE).reasons.some((r) => r.includes("local network")));
  assert.ok(navigationRisk("http://shop.example/", PAGE).reasons.some((r) => r.includes("not encrypted")));
  assert.equal(navigationRisk("http://localhost:3000/", PAGE).allowed, true);       // a dev server is fine
});

test("the agent never fills a secret, a disabled field, a CAPTCHA or anything not on the page", () => {
  const live = { e1: el({ ref: "e1", role: "textbox", name: "Search" }), e2: el({ ref: "e2", role: "textbox", name: "Password", sensitive: true }),
                 e3: el({ ref: "e3", disabled: true }), e4: el({ ref: "e4", blocked: true, name: "I'm not a robot" }) };
  const fill = (ref: string, value = "x"): Action => ({ action: "fill", ref, value, why: "" });
  assert.equal(refusal(fill("e1"), live, PAGE), null);
  assert.match(refusal(fill("e2"), live, PAGE) || "", /never fills/);
  assert.match(refusal({ action: "fill_form", fields: [{ ref: "e1", value: "a" }, { ref: "e2", value: "b" }], why: "" }, live, PAGE) || "", /never fills/);
  assert.match(refusal(fill("e3"), live, PAGE) || "", /disabled/);
  assert.match(refusal({ action: "click", ref: "e4", why: "" }, live, PAGE) || "", /CAPTCHA/);
  assert.match(refusal({ action: "click", ref: "e9", why: "" }, live, PAGE) || "", /not on the page/);
  assert.match(refusal(fill("e1", "x".repeat(2001)), live, PAGE) || "", /longer than/);
  assert.match(refusal({ action: "navigate", url: "javascript:1", why: "" }, live, PAGE) || "", /never opened/);
  assert.equal(refusal({ action: "click", ref: "e2", why: "" }, live, PAGE), null);   // clicking a field is not filling it
});

test("the preview is built from the live element, so the model cannot relabel what will happen", () => {
  const live = { e1: el({ name: "Add to cart", href: "https://shop.example/cart/add" }) };
  const p = previewFor({ action: "click", ref: "e1", why: "the user wants it" }, live, PAGE);
  assert.equal(p.title, "Click “Add to cart”");
  assert.ok(p.lines[0].includes("On https://shop.example: click the button “Add to cart”"));
  assert.ok(p.lines.some((l) => l.includes("goes to https://shop.example/cart/add")));
  assert.ok(p.lines.some((l) => l.startsWith("Why: the user wants it")));
  assert.equal(p.destination, "https://shop.example/cart/add");
  const fill = previewFor({ action: "fill", ref: "e1", value: "red shoes", why: "" }, { e1: el({ role: "textbox", name: "Search", value: "old" }) }, PAGE);
  assert.ok(fill.lines.includes("“red shoes”"));
  assert.equal(fill.risk, "high");                                           // it replaces text that is there
  const form = previewFor({ action: "fill_form", fields: [{ ref: "e1", value: "a" }, { ref: "e2", value: "b" }], why: "" },
    { e1: el({ role: "textbox", name: "Name" }), e2: el({ ref: "e2", role: "textbox", name: "City" }) }, PAGE);
  assert.equal(form.title, "Fill 2 fields");
  assert.deepEqual(form.lines.slice(1, 3), ["• textbox “Name”: “a”", "• textbox “City”: “b”"]);
  const nav = previewFor({ action: "navigate", url: "https://other.example/x", why: "" }, {}, PAGE);
  assert.equal(nav.risk, "high");
  assert.ok(nav.lines[0].includes("https://other.example/x"));
});

test("the log of a step names the element and never a typed value", () => {
  const live = { e1: el({ role: "textbox", name: "Card" }) };
  const text = summarize({ action: "fill", ref: "e1", value: "4111111111111111", why: "" }, live);
  assert.equal(text, "fill textbox “Card”");
  assert.ok(!text.includes("4111"));
  assert.equal(summarize({ action: "navigate", url: "https://a.example/", why: "" }, {}), "go to https://a.example/");
});

test("the task table: every state's exits are written down, and nothing leaves a final state", () => {
  assert.equal(transition("requested", "planning"), "planning");
  assert.equal(transition("planning", "approval_required"), "approval_required");
  assert.equal(transition("approval_required", "executing"), "executing");
  assert.equal(transition("executing", "verifying"), "verifying");
  assert.equal(transition("verifying", "planning"), "planning");
  for (const bad of [["requested", "executing"], ["planning", "verifying"], ["approval_required", "completed"], ["completed", "planning"],
                     ["cancelled", "planning"], ["failed", "executing"], ["executing", "approval_required"]] as const) {
    assert.throws(() => transition(bad[0], bad[1]), TransitionError, bad.join(" → "));
  }
  for (const s of Object.keys(NEXT) as (keyof typeof NEXT)[]) {
    if (!TERMINAL.has(s)) {
      assert.ok(NEXT[s].includes("cancelled"), `${s} can be cancelled`);
      assert.ok(NEXT[s].includes("failed") || s === "verifying", `${s} can fail`);
    } else assert.deepEqual(NEXT[s], [], `${s} is final`);
  }
  // an action runs in the tab and on the origin where the task began, nowhere else
  assert.equal(sameBinding({ tabId: 4, origin: "https://a.example" }, 4, "https://a.example"), true);
  assert.equal(sameBinding({ tabId: 4, origin: "https://a.example" }, 5, "https://a.example"), false);
  assert.equal(sameBinding({ tabId: 4, origin: "https://a.example" }, 4, "https://b.example"), false);
});

test("submitting a search form is not flagged; submitting any other form is", () => {
  const search = el({ type: "submit", inForm: true, formSearch: true, name: "Go", formAction: "https://shop.example/s" });
  const order = el({ type: "submit", inForm: true, formSearch: false, name: "Go", formAction: "https://shop.example/o" });
  assert.equal(clickRisk(search, PAGE).risk, "low");
  assert.equal(clickRisk(order, PAGE).risk, "high");
  const pay = el({ type: "submit", inForm: true, formSearch: true, name: "Pay now" });
  assert.equal(clickRisk(pay, PAGE).risk, "high");                       // the words still count
});

test("key presses: only listed keys, never into a secret field, Enter in a form is risky", () => {
  const field = el({ ref: "e2", role: "textbox", tag: "input", name: "Message", inForm: true });
  const live = { e2: field };
  assert.equal(refusal({ action: "press", key: "F5" as any, why: "" }, {}, PAGE), "the key F5 is not allowed");
  assert.equal(previewFor({ action: "press", key: "Enter", ref: "e2", why: "" }, live, PAGE).risk, "high");
  assert.equal(previewFor({ action: "press", key: "Escape", ref: "e2", why: "" }, live, PAGE).risk, "low");
  const secret = { e2: { ...field, sensitive: true } };
  assert.match(refusal({ action: "press", key: "Enter", ref: "e2", why: "" }, secret, PAGE)!, /never fills/);
  assert.equal(summarize({ action: "scroll", direction: "down" }, {}), "scroll down");
});
