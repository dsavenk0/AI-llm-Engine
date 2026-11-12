// The part of the agent that lives in the page: it describes what a person could click or fill, hands out
// short-lived references to those elements, and — only when asked, only for a reference it issued — checks that
// the element is still what it was and acts on it.
//
// It runs in the extension's isolated world: the page's own scripts cannot see it, call it or replace it. It
// offers a fixed set of methods, and takes element references, never selectors, code or markup — so nothing the
// model or the page says can make it run anything else.
//
// A reference ("e12") belongs to one *snapshot generation*. Taking a new snapshot invalidates every earlier
// reference; so does a change of address. An action carries the generation and the address it was proposed for,
// and is refused when either is not the current one, when the element has left the page, when its role, name or
// type changed, when it is hidden, disabled, or covered by something else.

import { ElementInfo, PageSnapshot } from "../core/agent-types";
import { sensitiveReason } from "../core/policy";

const CANDIDATES = [
  "a[href]", "button", "input", "select", "textarea", "summary", "[role=button]", "[role=link]", "[role=checkbox]",
  "[role=radio]", "[role=tab]", "[role=menuitem]", "[role=switch]", "[role=option]", "[role=combobox]",
  "[role=textbox]", "[contenteditable='']", "[contenteditable=true]", "[onclick]",
].join(",");
const CAPTCHA = "iframe[src*='recaptcha'],iframe[src*='hcaptcha'],iframe[src*='turnstile'],iframe[src*='challenges.cloudflare']," +
  ".g-recaptcha,.h-captcha,.cf-turnstile,[data-sitekey]";
const TEXT_INPUTS = new Set(["text", "search", "email", "url", "tel", "number", "date", "datetime-local", "month", "time", "week", ""]);
const NOT_FILLABLE = new Set(["password", "file", "hidden", "checkbox", "radio", "submit", "button", "image", "reset", "range", "color"]);

interface Fingerprint { tag: string; role: string; name: string; type: string }
interface Entry { el: WeakRef<Element>; fp: Fingerprint }

export interface Expect { generation: number; url: string }
export type Verdict = { ok: true; info: ElementInfo } | { ok: false; error: string };
export type Outcome = { ok: boolean; error?: string; urlChanged?: boolean; mutations?: number; readback?: string };

const collapse = (s: string | null | undefined): string => (s || "").replace(/\s+/g, " ").trim();

export function createAgent(win: Window & typeof globalThis) {
  const doc = win.document;
  let generation = 0;
  let counter = 0;
  let refs = new Map<string, Entry>();

  // ───────────── what an element is ─────────────
  function roleOf(el: Element): string {
    const explicit = el.getAttribute("role");
    if (explicit) return explicit.split(/\s+/)[0];
    const tag = el.tagName.toLowerCase();
    if (tag === "a") return "link";
    if (tag === "button" || tag === "summary") return "button";
    if (tag === "select") return "combobox";
    if (tag === "textarea") return "textbox";
    if (tag === "input") {
      const t = (el.getAttribute("type") || "text").toLowerCase();
      if (t === "checkbox" || t === "radio") return t;
      if (t === "submit" || t === "button" || t === "reset" || t === "image") return "button";
      if (t === "range") return "slider";
      return "textbox";
    }
    return el.hasAttribute("contenteditable") ? "textbox" : tag;
  }

  function labelText(el: Element): string {
    const labelled = el.getAttribute("aria-labelledby");
    if (labelled) {
      const t = labelled.split(/\s+/).map((id) => collapse(doc.getElementById(id)?.textContent)).filter(Boolean).join(" ");
      if (t) return t;
    }
    const id = el.getAttribute("id");
    if (id) {
      const l = doc.querySelector(`label[for="${win.CSS?.escape ? win.CSS.escape(id) : id.replace(/"/g, "")}"]`);
      if (l) return collapse(l.textContent);
    }
    const wrap = el.closest("label");
    return wrap ? collapse(wrap.textContent) : "";
  }

  function nameOf(el: Element): string {
    const aria = collapse(el.getAttribute("aria-label"));
    if (aria) return aria.slice(0, 100);
    const label = labelText(el);
    if (label) return label.slice(0, 100);
    const tag = el.tagName.toLowerCase();
    if (tag === "input") {
      const t = (el.getAttribute("type") || "text").toLowerCase();
      if (t === "submit" || t === "button" || t === "reset") return collapse((el as HTMLInputElement).value || t).slice(0, 100);
      if (t === "image") return collapse(el.getAttribute("alt")).slice(0, 100);
    }
    const text = collapse(el.textContent);
    if (text && tag !== "select" && tag !== "textarea") return text.slice(0, 100);
    return collapse(el.getAttribute("title") || el.getAttribute("alt") || el.getAttribute("placeholder")).slice(0, 100);
  }

  function visible(el: Element): boolean {
    if (el.closest("[aria-hidden=true],[hidden]")) return false;
    const check = (el as Element & { checkVisibility?: (o: object) => boolean }).checkVisibility;
    if (typeof check === "function") {
      if (!check.call(el, { checkOpacity: true, checkVisibilityCSS: true })) return false;
    } else {
      // no checkVisibility (an older browser, or a test): a display:none or hidden ancestor hides everything inside it
      for (let n: Element | null = el; n; n = n.parentElement) {
        const st = win.getComputedStyle(n);
        if (st.display === "none" || st.visibility === "hidden" || st.visibility === "collapse") return false;
      }
    }
    if (parseFloat(win.getComputedStyle(el).opacity || "1") === 0) return false;
    const r = el.getBoundingClientRect();
    return r.width > 0 && r.height > 0;
  }

  function absolute(href: string | null): string {
    if (!href) return "";
    try {
      const u = new URL(href, win.location.href);
      if (u.protocol !== "http:" && u.protocol !== "https:") return "";
      u.username = u.password = u.hash = "";
      return u.href.slice(0, 500);
    } catch {
      return "";
    }
  }

  function describe(el: Element, ref: string): ElementInfo {
    const tag = el.tagName.toLowerCase();
    const type = tag === "input" ? (el.getAttribute("type") || "text").toLowerCase() : "";
    const name = nameOf(el);
    const placeholder = collapse(el.getAttribute("placeholder")).slice(0, 60);
    const secret = sensitiveReason({ type, autocomplete: el.getAttribute("autocomplete") || "", name: el.getAttribute("name") || "",
                                     id: el.getAttribute("id") || "", hint: `${name} ${placeholder}` });
    const form = (el as HTMLInputElement).form || el.closest("form");
    let formSensitive = false;
    if (form) {
      for (const f of Array.from(form.querySelectorAll("input,textarea"))) {
        if (sensitiveReason({ type: (f.getAttribute("type") || "").toLowerCase(), autocomplete: f.getAttribute("autocomplete") || "",
                              name: f.getAttribute("name") || "", id: f.getAttribute("id") || "" })) { formSensitive = true; break; }
      }
    }
    const rawValue = tag === "input" || tag === "textarea" || tag === "select" ? (el as HTMLInputElement).value || "" : "";
    return {
      ref, tag, role: roleOf(el), name, type,
      value: secret ? "" : collapse(rawValue).slice(0, 80), placeholder,
      href: tag === "a" ? absolute(el.getAttribute("href")) : "",
      disabled: (el as HTMLButtonElement).disabled === true || el.getAttribute("aria-disabled") === "true",
      checked: (el as HTMLInputElement).checked === true || el.getAttribute("aria-checked") === "true",
      required: (el as HTMLInputElement).required === true,
      options: tag === "select" ? Array.from((el as HTMLSelectElement).options).slice(0, 10).map((o) => collapse(o.text).slice(0, 40)) : [],
      dup: 0, sensitive: secret !== null, blocked: !!el.closest(CAPTCHA),
      inForm: !!form, formAction: form ? absolute(form.getAttribute("action") || win.location.href) : "", formSensitive,
    };
  }

  const fingerprint = (info: ElementInfo): Fingerprint => ({ tag: info.tag, role: info.role, name: info.name, type: info.type });

  function register(el: Element): ElementInfo {
    const info = describe(el, `e${++counter}`);
    refs.set(info.ref, { el: new WeakRef(el), fp: fingerprint(info) });
    return info;
  }

  // ───────────── reading ─────────────
  function candidates(): Element[] {
    return Array.from(doc.querySelectorAll(CANDIDATES)).filter((el) => {
      if (el.tagName === "INPUT" && (el.getAttribute("type") || "").toLowerCase() === "hidden") return false;
      return visible(el);
    });
  }

  function snapshot(max = 120): PageSnapshot {
    generation += 1;
    counter = 0;
    refs = new Map();
    const all = candidates();
    const picked = all.slice(0, max);
    const elements = picked.map(register);
    const seen = new Map<string, number>();
    for (const e of elements) if (e.name) seen.set(`${e.role}|${e.name}`, (seen.get(`${e.role}|${e.name}`) || 0) + 1);
    for (const e of elements) {
      const n = seen.get(`${e.role}|${e.name}`) || 0;
      e.dup = n > 1 ? n : 0;
    }
    const body = doc.body as HTMLElement | null;
    const text = collapse(body ? (body as HTMLElement).innerText ?? body.textContent : "").slice(0, 1500);
    return { generation, url: win.location.href, title: collapse(doc.title).slice(0, 200), text, elements,
             captcha: !!doc.querySelector(CAPTCHA), truncated: all.length > picked.length };
  }

  /** Elements whose name, text or placeholder contains `query`, registered in the current generation. */
  function find(query: string, max = 20): { generation: number; elements: ElementInfo[] } {
    const q = collapse(query).toLowerCase();
    const out: ElementInfo[] = [];
    if (q) {
      for (const el of candidates()) {
        const hay = `${nameOf(el)} ${collapse(el.getAttribute("placeholder"))} ${collapse(el.textContent).slice(0, 200)}`.toLowerCase();
        if (!hay.includes(q)) continue;
        const known = Array.from(refs.entries()).find(([, v]) => v.el.deref() === el);
        out.push(known ? describe(el, known[0]) : register(el));
        if (out.length >= max) break;
      }
    }
    return { generation, elements: out };
  }

  // ───────────── trusting a reference ─────────────
  function resolve(ref: string, expect: Expect): { el: Element; info: ElementInfo } | { error: string } {
    if (expect.generation !== generation) return { error: "the page was read again since this was proposed" };
    if (win.location.href !== expect.url) return { error: "the page address changed since this was proposed" };
    const entry = refs.get(ref);
    if (!entry) return { error: `${ref} was not issued for this page` };
    const el = entry.el.deref();
    if (!el || !el.isConnected) return { error: "that element is no longer on the page" };
    const info = describe(el, ref);
    const fp = entry.fp;
    if (info.tag !== fp.tag || info.role !== fp.role || info.name !== fp.name || info.type !== fp.type) {
      return { error: "that element has changed since it was listed" };
    }
    if (!visible(el)) return { error: "that element is hidden" };
    if (typeof doc.elementFromPoint === "function") {
      const r = el.getBoundingClientRect();
      const top = doc.elementFromPoint(r.left + r.width / 2, r.top + r.height / 2);
      if (top && top !== el && !el.contains(top)) return { error: "something else is covering that element" };
    }
    return { el, info };
  }

  /** The element as it is *now*, for the preview — or why it cannot be used. */
  function inspect(ref: string, expect: Expect): Verdict {
    const r = resolve(ref, expect);
    return "error" in r ? { ok: false, error: r.error } : { ok: true, info: r.info };
  }

  // ───────────── acting ─────────────
  function settle(quiet = 300, max = 2000): Promise<number> {
    return new Promise((done) => {
      let changes = 0;
      let timer = win.setTimeout(finish, quiet);
      const obs = new win.MutationObserver((m) => {
        changes += m.length;
        win.clearTimeout(timer);
        timer = win.setTimeout(finish, quiet);
      });
      obs.observe(doc, { subtree: true, childList: true, attributes: true, characterData: true });
      const cap = win.setTimeout(finish, max);
      function finish(): void {
        obs.disconnect();
        win.clearTimeout(timer);
        win.clearTimeout(cap);
        done(changes);
      }
    });
  }

  async function click(ref: string, expect: Expect): Promise<Outcome> {
    const r = resolve(ref, expect);
    if ("error" in r) return { ok: false, error: r.error };
    if (r.info.disabled) return { ok: false, error: "that element is disabled" };
    if (r.info.blocked) return { ok: false, error: "that is part of a CAPTCHA" };
    const before = win.location.href;
    const done = settle();
    (r.el as HTMLElement).scrollIntoView?.({ block: "center" });
    (r.el as HTMLElement).click();
    const mutations = await done;
    return { ok: true, urlChanged: win.location.href !== before, mutations };
  }

  const nativeSetter = (el: Element): ((v: string) => void) | null => {
    const proto = el.tagName === "TEXTAREA" ? win.HTMLTextAreaElement.prototype
      : el.tagName === "SELECT" ? win.HTMLSelectElement.prototype : win.HTMLInputElement.prototype;
    const d = Object.getOwnPropertyDescriptor(proto, "value");
    return d && d.set ? (v: string) => d.set!.call(el, v) : null;
  };

  async function fill(ref: string, value: string, expect: Expect): Promise<Outcome> {
    const r = resolve(ref, expect);
    if ("error" in r) return { ok: false, error: r.error };
    const { el, info } = r;
    if (info.sensitive) return { ok: false, error: "that is a password, card or one-time-code field: it is never filled" };
    if (info.blocked) return { ok: false, error: "that is part of a CAPTCHA" };
    if (info.disabled) return { ok: false, error: "that field is disabled" };
    if ((el as HTMLInputElement).readOnly) return { ok: false, error: "that field is read-only" };
    const done = settle(150, 800);
    (el as HTMLElement).focus?.();
    let readback = "";
    if (el.tagName === "SELECT") {
      const sel = el as HTMLSelectElement;
      const want = collapse(value).toLowerCase();
      const opt = Array.from(sel.options).find((o) => o.value === value || collapse(o.text).toLowerCase() === want);
      if (!opt) return { ok: false, error: `the list has no option “${value.slice(0, 40)}”` };
      nativeSetter(el)?.(opt.value);
      sel.value = opt.value;
      readback = collapse(sel.options[sel.selectedIndex]?.text);
      if (readback.toLowerCase() !== collapse(opt.text).toLowerCase()) return { ok: false, error: "the list did not take that option" };
    } else if (el.tagName === "TEXTAREA" || (el.tagName === "INPUT" && TEXT_INPUTS.has(info.type) && !NOT_FILLABLE.has(info.type))) {
      nativeSetter(el)?.(value);                         // the page's framework listens to the native setter's events
      readback = (el as HTMLInputElement).value;
    } else if ((el as HTMLElement).isContentEditable || el.getAttribute("contenteditable") !== null) {
      el.textContent = value;
      readback = el.textContent || "";
    } else {
      return { ok: false, error: `a ${info.role || info.tag} cannot be filled` };
    }
    el.dispatchEvent(new win.Event("input", { bubbles: true }));
    el.dispatchEvent(new win.Event("change", { bubbles: true }));
    await done;
    if (el.tagName !== "SELECT" && readback !== value) {
      // a number or date field may normalise what it was given; only a different value is a failure
      if (!(el.tagName === "INPUT" && collapse(readback) === collapse(value))) {
        return { ok: false, error: "the field did not keep the value", readback: readback.slice(0, 80) };
      }
    }
    return { ok: true, readback: readback.slice(0, 80) };
  }

  return { snapshot, find, inspect, click, fill, generation: () => generation };
}

export type PageAgent = ReturnType<typeof createAgent>;
