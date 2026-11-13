// What the agent may do to a page, decided here and nowhere else — against the live element, not the model's
// description of it. Pure functions: no DOM, no browser API, so each rule has a test.
//
// The posture: reading is free; every action that changes something (click, fill, navigate) is shown to the
// person first, described from what the page really contains. Beyond that, some things are never done at all:
// passwords, card numbers and one-time codes are never typed, CAPTCHAs are never touched, and the extension
// will not steer the browser to the local assistant or engine.

import { Action, ElementInfo, KEYS, Preview, Risk } from "./agent-types";

export const MAX_STEPS = 8;
export const MAX_STEPS_AUTO = 20;         // acting on its own takes more, smaller steps between two questions
export const MAX_FAILURES = 2;
export const MAX_VALUE = 2000;

/** A whole-word, case-insensitive match for any of `alternatives` (regex source). JavaScript's \b knows only
 *  ASCII letters, so "Оплатить" would never match a \b…\b pattern: the boundary is "not next to a letter". */
const words = (alternatives: string): RegExp =>
  new RegExp(`(?<![\\p{L}\\p{N}_])(?:${alternatives})(?![\\p{L}\\p{N}_])`, "iu");

// what makes a field one the agent must not fill
const SENSITIVE_AUTOCOMPLETE = /(^|\s)(cc-[a-z-]+|one-time-code|current-password|new-password)(\s|$)/i;
const SENSITIVE_WORDS = words(
  "pass(?:word|code|phrase)?|passwd|pwd|cvv|cvc|cvn|card\\s*(?:number|no)|cc-?num|security\\s*code|one[- ]?time|otp|" +
  "iban|ssn|social\\s*security|pin\\s*code|парол[ьяеюи]*|код\\s*(?:из\\s*)?смс|номер\\s*карты|пин-?код");

/** Why this field may not be filled, or null. `hint` is the text a person would read as its label. */
export function sensitiveReason(f: { type?: string; autocomplete?: string; name?: string; id?: string; hint?: string }): string | null {
  if ((f.type || "").toLowerCase() === "password") return "a password field";
  if (SENSITIVE_AUTOCOMPLETE.test(f.autocomplete || "")) return "a payment or one-time-code field";
  if (SENSITIVE_WORDS.test([f.name, f.id, f.hint].filter(Boolean).join(" "))) return "a field that looks like a secret";
  return null;
}

// things whose click does something that is hard to undo
const CONSEQUENTIAL = words(
  "buy|purchase|pay|checkout|place\\s+order|order\\s+now|confirm|donate|subscribe|delete|remove|erase|send|" +
  "submit|post|publish|transfer|withdraw|sign\\s*out|log\\s*out|unsubscribe|accept|agree|authori[sz]e|allow|grant|" +
  "book|reserve|apply|register|sign\\s*up|куп\\p{L}*|оплат\\p{L}*|заказ\\p{L}*|оформ\\p{L}*|удал\\p{L}*|отправ\\p{L}*|" +
  "подтверд\\p{L}*|подписа\\p{L}*|выйти|выход|разреш\\p{L}*|перевест\\p{L}*|перевод|согласен|согласна");

export function sameOrigin(a: string, b: string): boolean {
  try {
    return new URL(a).origin === new URL(b).origin;
  } catch {
    return false;
  }
}

export function originOf(url: string): string {
  try {
    const u = new URL(url);
    return u.protocol === "http:" || u.protocol === "https:" ? u.origin : "";
  } catch {
    return "";
  }
}

/** Reasons a click on `el` (on a page at `pageUrl`) deserves a second look. Never blocks: the person decides. */
export function clickRisk(el: ElementInfo, pageUrl: string): { risk: Risk; reasons: string[] } {
  const reasons: string[] = [];
  if (CONSEQUENTIAL.test(el.name)) reasons.push(`its label says “${el.name.slice(0, 40)}”, which usually has a lasting effect`);
  const submits = el.type === "submit" || (el.inForm && el.role === "button" && el.type !== "button" && el.type !== "reset");
  if (submits && !el.formSearch) {          // a search box only looks something up; any other form sends what is in it
    reasons.push("it submits a form" + (el.formAction ? ` to ${originOf(el.formAction) || el.formAction}` : ""));
  }
  if (el.formSensitive) reasons.push("its form holds a password, card or one-time-code field");
  if (el.href && originOf(el.href) && !sameOrigin(el.href, pageUrl)) reasons.push(`it leaves this site for ${originOf(el.href)}`);
  if (el.formAction && originOf(el.formAction) && !sameOrigin(el.formAction, pageUrl)) {
    reasons.push(`its form posts to another site, ${originOf(el.formAction)}`);
  }
  if (el.dup > 1) reasons.push(`${el.dup} elements look the same; make sure this is the one you mean`);
  return { risk: reasons.length ? "high" : "low", reasons };
}

const PRIVATE_HOST = /^(localhost|.*\.local|.*\.internal|127\.\d+\.\d+\.\d+|10\.\d+\.\d+\.\d+|192\.168\.\d+\.\d+|169\.254\.\d+\.\d+|172\.(1[6-9]|2\d|3[01])\.\d+\.\d+|\[?::1\]?)$/i;
// the assistant's own service and the engine: never somewhere the agent steers a browser
const CONTROL_PORTS = new Set(["8090", "8093", "8091", "8092"]);

export interface NavVerdict { allowed: boolean; risk: Risk; reasons: string[]; error?: string; url?: string }

export function navigationRisk(target: string, currentUrl: string): NavVerdict {
  let u: URL;
  try {
    u = new URL(target);
  } catch {
    return { allowed: false, risk: "high", reasons: [], error: "that is not a valid address" };
  }
  if (u.protocol !== "http:" && u.protocol !== "https:") {
    return { allowed: false, risk: "high", reasons: [], error: `${u.protocol} addresses are never opened` };
  }
  if (u.username || u.password) return { allowed: false, risk: "high", reasons: [], error: "addresses with a login in them are never opened" };
  if (PRIVATE_HOST.test(u.hostname) && CONTROL_PORTS.has(u.port)) {
    return { allowed: false, risk: "high", reasons: [], error: "this is the local assistant or engine: it is not opened from a page" };
  }
  const reasons: string[] = [];
  if (!sameOrigin(u.href, currentUrl)) reasons.push(`it leaves ${originOf(currentUrl) || "this page"} for ${u.origin}`);
  if (PRIVATE_HOST.test(u.hostname)) reasons.push("it is an address on this Mac or the local network");
  if (u.protocol === "http:") reasons.push("the connection is not encrypted");
  return { allowed: true, risk: reasons.length ? "high" : "low", reasons, url: u.href };
}

/** Why a model-proposed value may not be typed, or null. */
export function valueProblem(v: unknown): string | null {
  if (typeof v !== "string") return "a value must be text";
  if (v.length > MAX_VALUE) return `a value is longer than ${MAX_VALUE} characters`;
  return null;
}

const short = (s: string, n: number): string => (s.length > n ? s.slice(0, n - 1) + "…" : s);

export function describeElement(el: ElementInfo): string {
  const what = el.role || el.tag;
  return el.name ? `${what} “${short(el.name, 60)}”` : what;
}

/** The text of a step, for the log of what was done (a fill's value is not in it). */
export function summarize(action: Action, live: Record<string, ElementInfo>): string {
  const d = (ref: string): string => (live[ref] ? describeElement(live[ref]) : ref);
  switch (action.action) {
    case "click": return `click ${d(action.ref)}`;
    case "fill": return `fill ${d(action.ref)}`;
    case "fill_form": return `fill ${action.fields.map((f) => d(f.ref)).join(", ")}`;
    case "navigate": return `go to ${action.url}`;
    case "find": return `look for “${action.query}”`;
    case "scroll": return action.ref ? `scroll to ${d(action.ref)}` : `scroll ${action.direction}`;
    case "press": return `press ${action.key}${action.ref ? ` in ${d(action.ref)}` : ""}`;
    case "visual_click": return `click visible “${short(action.label, 60)}”`;
    case "visual_type": return `type into visible “${short(action.label, 60)}”`;
    case "visual_press": return `press ${action.key} in the visible app`;
    case "workspace_sheets_write": return `write Google Sheets ${action.cell}`;
    case "workspace_docs_append": return "append to Google Docs";
    case "workspace_drive_create": return `create Drive document “${short(action.name, 50)}”`;
    case "workspace_gmail_draft": return `create Gmail draft to ${short(action.to, 60)}`;
    case "workspace_calendar_create": return `create Calendar event “${short(action.summary, 50)}”`;
    case "wait": return `wait ${action.seconds}s`;
    case "ask_user": return `ask: ${action.question}`;
    case "done": return "finish";
  }
}

/** What the person is asked to approve, from the live elements. */
export function previewFor(action: Action, live: Record<string, ElementInfo>, pageUrl: string): Preview {
  const host = originOf(pageUrl) || pageUrl;
  if (action.action === "click") {
    const el = live[action.ref];
    const { risk, reasons } = clickRisk(el, pageUrl);
    const lines = [`On ${host}: click the ${describeElement(el)}.`];
    if (el.href) lines.push(`It goes to ${short(el.href, 120)}.`);
    if (action.why) lines.push(`Why: ${short(action.why, 160)}`);
    return { title: `Click “${short(el.name || el.role || el.tag, 40)}”`, lines, risk, reasons, destination: el.href || undefined };
  }
  if (action.action === "fill") {
    const el = live[action.ref];
    const lines = [`On ${host}: type into the ${describeElement(el)}:`, `“${short(action.value, 200)}”`];
    if (el.value) lines.push("It already contains text, which will be replaced.");
    if (action.why) lines.push(`Why: ${short(action.why, 160)}`);
    return { title: `Fill “${short(el.name || el.placeholder || el.tag, 40)}”`, lines, risk: el.value ? "high" : "low",
             reasons: el.value ? ["the field already has text that will be replaced"] : [] };
  }
  if (action.action === "fill_form") {
    const lines = [`On ${host}: fill ${action.fields.length} field(s):`];
    let replaced = false;
    for (const f of action.fields) {
      const el = live[f.ref];
      lines.push(`• ${describeElement(el)}: “${short(f.value, 120)}”${el.value ? " (replaces existing text)" : ""}`);
      replaced = replaced || !!el.value;
    }
    if (action.why) lines.push(`Why: ${short(action.why, 160)}`);
    return { title: `Fill ${action.fields.length} fields`, lines, risk: replaced ? "high" : "low",
             reasons: replaced ? ["some fields already have text that will be replaced"] : [] };
  }
  if (action.action === "navigate") {
    const v = navigationRisk(action.url, pageUrl);
    const lines = [`Open ${v.url || action.url}`, `from ${host}.`];
    if (action.why) lines.push(`Why: ${short(action.why, 160)}`);
    return { title: "Open a page", lines, risk: v.risk, reasons: v.reasons, destination: v.url };
  }
  if (action.action === "press") {
    const el = action.ref ? live[action.ref] : undefined;
    const lines = [`On ${host}: press ${action.key}${el ? ` in the ${describeElement(el)}` : ""}.`];
    if (action.why) lines.push(`Why: ${short(action.why, 160)}`);
    const reasons: string[] = [];
    if (action.key === "Enter" && (!el || (el.inForm && !el.formSearch))) reasons.push("Enter may send what is in the form");
    if (el?.formSensitive) reasons.push("its form holds a password, card or one-time-code field");
    return { title: `Press ${action.key}`, lines, risk: reasons.length ? "high" : "low", reasons };
  }
  if (action.action === "visual_click") {
    const lines = [`On ${host}: click the visible “${short(action.label, 80)}”.`, `Route: macOS Accessibility.`];
    if (action.why) lines.push(`Why: ${short(action.why, 160)}`);
    const high = CONSEQUENTIAL.test(action.label);
    return { title: `Click “${short(action.label, 40)}”`, lines, risk: high ? "high" : "low",
             reasons: high ? ["the visible label suggests a lasting effect"] : [] };
  }
  if (action.action === "visual_type") {
    const lines = [`On ${host}: type into the visible “${short(action.label, 80)}”:`, `“${short(action.value, 200)}”`,
                   "Route: macOS Accessibility."];
    if (action.why) lines.push(`Why: ${short(action.why, 160)}`);
    return { title: `Fill “${short(action.label, 40)}”`, lines, risk: action.empty ? "low" : "high",
             reasons: action.empty ? [] : ["the target may already contain data that would be replaced"] };
  }
  if (action.action === "visual_press") {
    const high = action.key === "Enter";
    return { title: `Press ${action.key}`, lines: [`On ${host}: press ${action.key}.`, "Route: macOS Accessibility."],
             risk: high ? "high" : "low", reasons: high ? ["Enter may submit the current form"] : [] };
  }
  if (action.action === "workspace_sheets_write") {
    const high = action.overwrite;
    return { title: `Write ${action.cell}`, lines: [`In Google Sheets: write “${short(action.value, 200)}”.`,
      action.overwrite ? "An existing value may be replaced." : "The service will refuse unless the cell is empty."],
      risk: high ? "high" : "low", reasons: high ? ["an existing cell may be overwritten"] : [] };
  }
  if (action.action === "workspace_docs_append")
    return { title: "Append to Google Docs", lines: [`Append: “${short(action.text, 240)}”`, "Route: Google Docs API."],
      risk: "high", reasons: ["this changes a cloud document"] };
  if (action.action === "workspace_drive_create")
    return { title: `Create “${short(action.name, 50)}”`, lines: ["Create a Google Docs file in Drive.",
      `Initial text: “${short(action.text, 200)}”`], risk: "high", reasons: ["this creates a cloud file"] };
  if (action.action === "workspace_gmail_draft")
    return { title: "Create Gmail draft", lines: [`To: ${short(action.to, 120)}`, `Subject: ${short(action.subject, 120)}`,
      `Body: “${short(action.body, 240)}”`, "The message is saved as a draft and is not sent."], risk: "high",
      reasons: ["this creates email content in Gmail"] };
  if (action.action === "workspace_calendar_create")
    return { title: "Create Calendar event", lines: [`${short(action.summary, 120)}`, `${action.start} → ${action.end}`],
      risk: "high", reasons: ["this adds an event to the primary calendar"] };
  return { title: action.action, lines: [], risk: "low", reasons: [] };
}

/** Why the agent may not do this to these live elements, or null. Checked at proposal and again at execution. */
export function refusal(action: Action, live: Record<string, ElementInfo>, pageUrl: string): string | null {
  if (action.action === "navigate") {
    const v = navigationRisk(action.url, pageUrl);
    return v.allowed ? null : v.error || "not allowed";
  }
  if ((action.action === "press" || action.action === "visual_press") && !(KEYS as readonly string[]).includes(action.key)) return `the key ${action.key} is not allowed`;
  if (action.action === "visual_type") {
    if (!action.ticket) return "the visual action has no service ticket";
    return valueProblem(action.value);
  }
  if (action.action === "visual_click" || action.action === "visual_press") return action.ticket ? null : "the visual action has no service ticket";
  if (action.action.startsWith("workspace_")) return "ticket" in action && action.ticket ? null : "the Workspace action has no service ticket";
  const refs = action.action === "click" || action.action === "fill" ? [action.ref]
    : action.action === "press" ? (action.ref ? [action.ref] : [])
    : action.action === "fill_form" ? action.fields.map((f) => f.ref) : [];
  for (const ref of refs) {
    const el = live[ref];
    if (!el) return `${ref} is not on the page any more`;
    if (el.blocked) return "that is part of a CAPTCHA: only you can do that";
    if (el.disabled) return `${describeElement(el)} is disabled`;
    if (action.action !== "click" && el.sensitive) return `${describeElement(el)} is a field the agent never fills (a password, card or one-time code)`;
  }
  if (action.action === "fill") return valueProblem(action.value);
  if (action.action === "fill_form") {
    for (const f of action.fields) {
      const p = valueProblem(f.value);
      if (p) return p;
    }
  }
  return null;
}
