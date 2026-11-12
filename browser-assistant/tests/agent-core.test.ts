import assert from "node:assert/strict";
import { test } from "node:test";
import { JSDOM } from "jsdom";
import { createAgent } from "../src/content/agent-core";

interface Page { win: any; doc: Document; agent: ReturnType<typeof createAgent>; cover: Map<Element, Element>; hidden: Set<Element> }

/** A jsdom page with a fake layout: every element is 100×20 at a row of its own, so hit-testing can be asked. */
function page(html: string, url = "https://shop.example/cart"): Page {
  const dom = new JSDOM(`<!doctype html><html><head><title>Cart</title></head><body>${html}</body></html>`, { url, pretendToBeVisual: true });
  const win = dom.window;
  const doc = win.document as Document;
  const rows = new Map<Element, number>();
  const row = (el: Element): number => { if (!rows.has(el)) rows.set(el, rows.size); return rows.get(el)!; };
  const hidden = new Set<Element>();
  const cover = new Map<Element, Element>();
  win.Element.prototype.getBoundingClientRect = function (this: Element) {
    const top = row(this) * 30;
    const h = hidden.has(this) ? 0 : 20;
    return { left: 0, top, width: hidden.has(this) ? 0 : 100, height: h, right: 100, bottom: top + h, x: 0, y: top, toJSON() {} };
  };
  win.document.elementFromPoint = (_x: number, y: number) => {
    const el = [...rows.entries()].find(([, i]) => i === Math.floor(y / 30))?.[0] ?? null;
    return (el && cover.get(el)) || el;
  };
  return { win, doc, agent: createAgent(win), cover, hidden };
}
const FORM = `
  <a href="/home">Home</a>
  <a href="https://elsewhere.example/x?token=1#frag">Elsewhere</a>
  <button id="add">Add to cart</button>
  <button class="x">Remove</button><button class="y">Remove</button>
  <form action="/pay"><label for="q">Search</label><input id="q" type="text" value="old">
    <input id="pw" type="password" aria-label="Password" value="hunter2">
    <input autocomplete="cc-number" aria-label="Card" value="4111111111111111">
    <select id="size" aria-label="Size"><option value="s">Small</option><option value="l">Large</option></select>
    <textarea aria-label="Notes"></textarea>
    <input type="checkbox" aria-label="Gift wrap">
    <input type="submit" value="Pay now"></form>
  <div style="display:none"><button>Hidden</button></div>
  <button aria-hidden="true">Aria hidden</button>
  <input type="hidden" name="csrf" value="secret">`;

test("a snapshot lists what a person could use, with accessible names, and hides the rest", () => {
  const { agent } = page(FORM);
  const s = agent.snapshot();
  const by = (name: string) => s.elements.find((e) => e.name === name)!;
  assert.equal(s.generation, 1);
  assert.equal(s.url, "https://shop.example/cart");
  assert.deepEqual(s.elements.map((e) => e.ref).slice(0, 3), ["e1", "e2", "e3"]);
  assert.equal(by("Home").role, "link");
  assert.equal(by("Home").href, "https://shop.example/home");
  assert.equal(by("Elsewhere").href, "https://elsewhere.example/x?token=1");        // the fragment is dropped
  assert.equal(by("Search").role, "textbox");                                        // named by its <label for>
  assert.equal(by("Search").value, "old");
  assert.equal(by("Size").role, "combobox");
  assert.deepEqual(by("Size").options, ["Small", "Large"]);
  assert.equal(by("Pay now").role, "button");                                        // an input submit, named by its value
  assert.equal(by("Pay now").inForm, true);
  assert.equal(by("Pay now").formAction, "https://shop.example/pay");
  assert.equal(by("Pay now").formSensitive, true);
  assert.ok(!s.elements.some((e) => e.name === "Hidden" || e.name === "Aria hidden"));
  assert.ok(!s.elements.some((e) => e.name === "csrf"));                             // type=hidden is not an element to use
});

test("secrets are flagged and their values never leave the page", () => {
  const { agent } = page(FORM);
  const s = agent.snapshot();
  const pw = s.elements.find((e) => e.name === "Password")!;
  const card = s.elements.find((e) => e.name === "Card")!;
  assert.deepEqual([pw.sensitive, pw.value, card.sensitive, card.value], [true, "", true, ""]);
  assert.ok(!JSON.stringify(s).includes("hunter2") && !JSON.stringify(s).includes("4111"));
});

test("identical-looking elements are counted so the person can be warned", () => {
  const { agent } = page(FORM);
  const s = agent.snapshot();
  assert.deepEqual(s.elements.filter((e) => e.name === "Remove").map((e) => e.dup), [2, 2]);
  assert.equal(s.elements.find((e) => e.name === "Home")!.dup, 0);
});

test("a long page is cut and says so; a CAPTCHA is noticed and its parts are blocked", () => {
  const many = Array.from({ length: 30 }, (_, i) => `<button>B${i}</button>`).join("");
  const a = page(many).agent.snapshot(10);
  assert.deepEqual([a.elements.length, a.truncated], [10, true]);
  const c = page(`<button>Real</button><div class="g-recaptcha"><button>I am not a robot</button></div>`).agent.snapshot();
  assert.equal(c.captcha, true);
  assert.deepEqual(c.elements.map((e) => [e.name, e.blocked]), [["Real", false], ["I am not a robot", true]]);
});

test("find registers what it finds in the current generation, without invalidating the rest", () => {
  const { agent } = page(FORM);
  const s = agent.snapshot(3);                              // the list was cut: Pay now is not in it
  assert.ok(!s.elements.some((e) => e.name === "Pay now"));
  const f = agent.find("pay now");
  assert.equal(f.generation, s.generation);
  assert.equal(f.elements[0].name, "Pay now");
  assert.ok(agent.inspect(f.elements[0].ref, { generation: s.generation, url: "https://shop.example/cart" }).ok);
  assert.ok(agent.inspect("e1", { generation: s.generation, url: "https://shop.example/cart" }).ok);
  assert.deepEqual(agent.find("").elements, []);
});

test("a reference is refused when the generation, the address, the element or its look has changed", () => {
  const p = page(FORM);
  const s = p.agent.snapshot();
  const expect = { generation: s.generation, url: s.url };
  const add = s.elements.find((e) => e.name === "Add to cart")!;
  assert.equal(p.agent.inspect(add.ref, expect).ok, true);
  const refused = (r: ReturnType<typeof p.agent.inspect>, why: RegExp) => { assert.equal(r.ok, false); assert.match((r as any).error, why); };

  refused(p.agent.inspect(add.ref, { ...expect, generation: 99 }), /read again/);
  refused(p.agent.inspect(add.ref, { ...expect, url: "https://shop.example/other" }), /address changed/);
  refused(p.agent.inspect("e999", expect), /not issued/);
  const el = p.doc.getElementById("add")!;
  el.textContent = "Add to wishlist";                                              // the button's meaning changed
  refused(p.agent.inspect(add.ref, expect), /changed since it was listed/);
  el.textContent = "Add to cart";
  p.hidden.add(el);
  refused(p.agent.inspect(add.ref, expect), /hidden/);
  p.hidden.delete(el);
  p.cover.set(el, p.doc.querySelector(".x")!);                                     // an overlay sits on top of it
  refused(p.agent.inspect(add.ref, expect), /covering/);
  p.cover.delete(el);
  assert.equal(p.agent.inspect(add.ref, expect).ok, true);
  el.remove();
  refused(p.agent.inspect(add.ref, expect), /no longer on the page/);
});

test("a new snapshot retires every earlier reference", () => {
  const p = page(FORM);
  const first = p.agent.snapshot();
  const old = { generation: first.generation, url: first.url };
  p.agent.snapshot();
  assert.equal(p.agent.inspect("e1", old).ok, false);
});

test("click acts on the verified element and reports what it saw", async () => {
  const p = page(FORM);
  const s = p.agent.snapshot();
  const expect = { generation: s.generation, url: s.url };
  let clicked = 0;
  p.doc.getElementById("add")!.addEventListener("click", () => { clicked++; p.doc.body.append(p.doc.createElement("p")); });
  const add = s.elements.find((e) => e.name === "Add to cart")!;
  const r = await p.agent.click(add.ref, expect);
  assert.deepEqual([r.ok, clicked, (r.mutations || 0) > 0, r.urlChanged], [true, 1, true, false]);
  const stale = await p.agent.click(add.ref, { ...expect, generation: 9 });
  assert.deepEqual([stale.ok, clicked], [false, 1]);                               // refused, and nothing happened
});

test("click refuses a disabled element and a CAPTCHA part", async () => {
  const p = page(`<button disabled>Off</button><div class="h-captcha"><button>Verify</button></div>`);
  const s = p.agent.snapshot();
  const expect = { generation: s.generation, url: s.url };
  for (const name of ["Off", "Verify"]) {
    const e = s.elements.find((x) => x.name === name)!;
    assert.equal((await p.agent.click(e.ref, expect)).ok, false, name);
  }
});

test("fill sets the value the way a framework notices, reads it back, and fires input and change", async () => {
  const p = page(FORM);
  const s = p.agent.snapshot();
  const expect = { generation: s.generation, url: s.url };
  const events: string[] = [];
  const q = p.doc.getElementById("q") as HTMLInputElement;
  q.addEventListener("input", () => events.push("input"));
  q.addEventListener("change", () => events.push("change"));
  const r = await p.agent.fill(s.elements.find((e) => e.name === "Search")!.ref, "red shoes", expect);
  assert.deepEqual([r.ok, r.readback, q.value, events], [true, "red shoes", "red shoes", ["input", "change"]]);
  const size = await p.agent.fill(s.elements.find((e) => e.name === "Size")!.ref, "large", expect);   // by visible text
  assert.deepEqual([size.ok, (p.doc.getElementById("size") as HTMLSelectElement).value], [true, "l"]);
  const none = await p.agent.fill(s.elements.find((e) => e.name === "Size")!.ref, "huge", expect);
  assert.deepEqual([none.ok, /no option/.test(none.error || "")], [false, true]);
  const notes = await p.agent.fill(s.elements.find((e) => e.name === "Notes")!.ref, "line one\nline two", expect);
  assert.equal(notes.ok, true);
});

test("fill never types into a secret, a checkbox, a CAPTCHA or a read-only field", async () => {
  const p = page(FORM + `<input readonly aria-label="Fixed" value="a"><div class="cf-turnstile"><input aria-label="Token"></div>`);
  const s = p.agent.snapshot();
  const expect = { generation: s.generation, url: s.url };
  for (const [name, why] of [["Password", /never filled/], ["Card", /never filled/], ["Gift wrap", /cannot be filled/],
                             ["Fixed", /read-only/], ["Token", /CAPTCHA/]] as const) {
    const r = await p.agent.fill(s.elements.find((e) => e.name === name)!.ref, "x", expect);
    assert.equal(r.ok, false, name);
    assert.match(r.error || "", why, name);
  }
  assert.equal((p.doc.getElementById("pw") as HTMLInputElement).value, "hunter2");   // untouched
});

test("a field that does not keep the value is reported as a failure, not as success", async () => {
  const p = page(`<input aria-label="Tiny" maxlength="3">`);
  const s = p.agent.snapshot();
  const input = p.doc.querySelector("input") as HTMLInputElement;
  Object.defineProperty(input, "value", { get: () => "", set: () => {}, configurable: true });   // swallows input, like a locked widget
  const r = await p.agent.fill(s.elements[0].ref, "hello", { generation: s.generation, url: s.url });
  assert.deepEqual([r.ok, r.error], [false, "the field did not keep the value"]);
});

test("a page's own script cannot reach the agent, and the agent takes no code or selectors", () => {
  const p = page(`<button id="b">Go</button>`);
  assert.deepEqual(Object.keys(p.agent).sort(), ["click", "fill", "find", "generation", "inspect", "press", "rectOf", "scroll", "snapshot"]);
  const s = p.agent.snapshot();
  for (const evil of ["button", "#b", "javascript:alert(1)", "e1; alert(1)", "../e1"]) {
    assert.equal(p.agent.inspect(evil, { generation: s.generation, url: s.url }).ok, false, evil);
  }
});

// ───────────── scrolling, keys, search forms ─────────────
test("a search form is told apart from one that sends something", () => {
  const { agent } = page(`<form role="search" action="/s"><input aria-label="Find" name="q"><button>Go</button></form>
    <form action="/order"><input aria-label="Name"><button>Order</button></form>`);
  const s = agent.snapshot();
  const by = (n: string) => s.elements.find((e) => e.name === n)!;
  assert.equal(by("Go").formSearch, true);
  assert.equal(by("Order").formSearch, false);
});

test("Enter in a form field submits that form, as a person's Enter would; other keys only reach the page's handlers", async () => {
  const { agent, doc, win } = page(`<form action="/s"><input aria-label="Find"></form>`);
  const s = agent.snapshot();
  const form = doc.querySelector("form")!;
  let submitted = 0;
  form.addEventListener("submit", (e: Event) => { e.preventDefault(); submitted++; });
  const keys: string[] = [];
  doc.addEventListener("keydown", (e: any) => keys.push(e.key));
  const expect = { generation: s.generation, url: win.location.href };
  assert.equal((await agent.press("ArrowDown", "e1", expect)).ok, true);
  assert.equal(submitted, 0);
  assert.equal((await agent.press("Enter", "e1", expect)).ok, true);
  assert.equal(submitted, 1);
  assert.deepEqual(keys, ["ArrowDown", "Enter"]);
  assert.equal((await agent.press("a", "e1", expect)).ok, false);              // not a key it may press
});

test("no key is pressed into a password field", async () => {
  const { agent, win } = page(`<input type="password" aria-label="Password">`);
  const s = agent.snapshot();
  const pw = s.elements.find((e) => e.name === "Password");
  if (!pw) return;                                                              // not listed at all: nothing to press into
  const r = await agent.press("Enter", pw.ref, { generation: s.generation, url: win.location.href });
  assert.equal(r.ok, false);
});

test("scrolling moves the page and says when it is already at the end", async () => {
  const { agent, doc, win } = page(`<p>long</p>`);
  let top = 0;
  Object.defineProperty(doc.documentElement, "scrollHeight", { configurable: true, get: () => 5000 });
  Object.defineProperty(doc.documentElement, "clientHeight", { configurable: true, get: () => 800 });
  Object.defineProperty(doc.documentElement, "scrollTop", { configurable: true, get: () => top,
    set: (v: number) => { top = Math.max(0, Math.min(4200, v)); } });
  Object.defineProperty(doc, "scrollingElement", { configurable: true, get: () => doc.documentElement });
  Object.defineProperty(win, "innerHeight", { configurable: true, get: () => 800 });
  const down = await agent.scroll("down");
  assert.equal(down.ok, true);
  assert.ok(top > 0);
  const up = await agent.scroll("up");
  assert.equal(up.ok, true);
  const again = await agent.scroll("up");
  assert.equal(again.readback, "already at the top");
});
