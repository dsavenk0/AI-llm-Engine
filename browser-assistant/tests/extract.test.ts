import assert from "node:assert/strict";
import { test } from "node:test";
import { JSDOM } from "jsdom";
import { extractPage } from "../src/content/extract";

/** Run extractPage in a jsdom page, with checkVisibility behaving as in Chrome: false for display:none and for
 *  a display:contents box (which draws nothing of its own although its children are on screen). */
function read(html: string, max = 10000) {
  const dom = new JSDOM(`<!doctype html><html><head><title>T</title></head><body>${html}</body></html>`,
    { url: "https://site.example/page", pretendToBeVisual: true });
  const g = globalThis as any;
  const saved = { document: g.document, window: g.window, Node: g.Node, getComputedStyle: g.getComputedStyle, location: g.location };
  const win = dom.window as any;
  win.Element.prototype.checkVisibility = function (this: Element) {
    const d = win.getComputedStyle(this).display;
    return d !== "none" && d !== "contents";
  };
  Object.assign(g, { document: win.document, window: win, Node: win.Node, getComputedStyle: win.getComputedStyle.bind(win),
                     location: win.location });
  try {
    return extractPage(max);
  } finally {
    Object.assign(g, saved);
  }
}

test("text inside display:contents wrappers is read, not skipped as invisible", () => {
  const r = read(`<div style="display:contents"><main style="display:contents"><h1>Menu</h1><p>Harvest bowl 12.50</p></main></div>`);
  assert.match(r.text, /# Menu/);
  assert.match(r.text, /Harvest bowl 12\.50/);
});

test("an almost empty <main> falls back to the whole body", () => {
  const r = read(`<main><span>Loading</span></main><section><p>${"The real article text. ".repeat(20)}</p></section>`);
  assert.match(r.text, /The real article text/);
});

test("hidden parts stay out, and scripts and navigation are skipped", () => {
  const r = read(`<nav>Home About</nav><p>Visible</p><p style="display:none">Secret</p><script>var x=1</script>`);
  assert.match(r.text, /Visible/);
  assert.doesNotMatch(r.text, /Secret|Home About|var x/);
});
