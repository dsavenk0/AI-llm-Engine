// What the assistant reads from a page. Injected with scripting.executeScript({ func }) into the active tab,
// only after a user gesture (the popup opened, a context-menu click) — so the function is serialised and
// must not use anything from outside its own body.
//
// It reads what a person can see: the selection, and the text of the main content (the <main> / <article>
// element when there is one, else the body), skipping scripts, styles, navigation, hidden elements and form
// field values. Tables keep their rows; headings and list items keep a line of their own.

export interface Extracted {
  title: string;
  url: string;
  selection: string;
  text: string;
  truncated: boolean;
}

export function extractPage(maxChars: number): Extracted {
  const SKIP = new Set(["SCRIPT", "STYLE", "NOSCRIPT", "TEMPLATE", "SVG", "CANVAS", "IFRAME", "OBJECT", "EMBED",
    "NAV", "FOOTER", "ASIDE", "FORM", "BUTTON", "SELECT", "TEXTAREA", "INPUT", "DIALOG"]);
  const BLOCK = /^(P|DIV|SECTION|ARTICLE|MAIN|HEADER|H[1-6]|LI|UL|OL|DL|DT|DD|TR|TABLE|BLOCKQUOTE|PRE|FIGCAPTION|BR|HR)$/;
  const hidden = (el: Element): boolean => {
    if (el.getAttribute("aria-hidden") === "true" || (el as HTMLElement).hidden) return true;
    const check = (el as Element & { checkVisibility?: () => boolean }).checkVisibility;
    if (typeof check === "function") return !check.call(el);
    const st = getComputedStyle(el);
    return st.display === "none" || st.visibility === "hidden";
  };
  const root = document.querySelector("main, article, [role='main']") || document.body;
  const out: string[] = [];
  let size = 0;
  let truncated = false;
  const add = (s: string): void => {
    if (truncated) return;
    if (size + s.length > maxChars) {
      out.push(s.slice(0, Math.max(0, maxChars - size)));
      truncated = true;
      size = maxChars;
      return;
    }
    out.push(s);
    size += s.length;
  };
  const walk = (node: Node): void => {
    if (truncated) return;
    if (node.nodeType === Node.TEXT_NODE) {
      const t = (node.nodeValue || "").replace(/\s+/g, " ");
      if (t.trim()) add(t);
      return;
    }
    if (node.nodeType !== Node.ELEMENT_NODE) return;
    const el = node as Element;
    if (SKIP.has(el.tagName.toUpperCase()) || hidden(el)) return;
    const tag = el.tagName.toUpperCase();
    const block = BLOCK.test(tag);
    if (block) add("\n");
    if (/^H[1-6]$/.test(tag)) add("#".repeat(Number(tag[1])) + " ");
    if (tag === "LI") add("- ");
    if (tag === "TD" || tag === "TH") add(" | ");
    if (tag === "IMG") {
      const alt = el.getAttribute("alt");
      if (alt) add(`[image: ${alt}]`);
      return;
    }
    for (const child of Array.from(el.childNodes)) walk(child);
    if (block) add("\n");
  };
  if (root) walk(root);
  const text = out.join("").replace(/[ \t]+\n/g, "\n").replace(/\n[ \t]+/g, "\n").replace(/\n{3,}/g, "\n\n").trim();
  const sel = (window.getSelection()?.toString() || "").trim();
  return { title: document.title || "", url: location.href, selection: sel.slice(0, maxChars), text, truncated };
}
