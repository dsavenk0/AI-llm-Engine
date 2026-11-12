// The side panel where the browser has no side panel API (Safari; a Chromium browser without chrome.sidePanel):
// the same page, sidepanel.html, docked to the right edge of the tab in a frame. The frame is an extension page —
// the site cannot read or script it — and the page is shifted left so the panel does not cover it.
// Injected on a click of the toolbar button; a second click, or the panel's own ✕, takes it away again.

import { api } from "../browser-adapters/api";

const ID = "ai-llm-engine-assistant-dock";
const WIDTH = 400;

function toggle(): void {
  const old = document.getElementById(ID);
  const html = document.documentElement;
  if (old) {
    old.remove();
    html.style.removeProperty("margin-right");
    return;
  }
  const host = document.createElement("div");
  host.id = ID;
  host.style.cssText = `all: initial; position: fixed; top: 0; right: 0; bottom: 0; width: ${WIDTH}px; max-width: 92vw;
    z-index: 2147483647; box-shadow: -10px 0 30px rgba(0, 0, 0, .35); background: #05090d;`;
  const root = host.attachShadow({ mode: "closed" });
  const frame = document.createElement("iframe");
  frame.src = api.runtime.getURL("sidepanel.html?docked=1");
  frame.title = "LLMBar Assistant";
  frame.setAttribute("allow", "clipboard-write");
  frame.style.cssText = "border: 0; width: 100%; height: 100%; display: block; color-scheme: normal;";
  root.append(frame);
  html.append(host);
  html.style.setProperty("margin-right", `${Math.min(WIDTH, innerWidth * 0.92)}px`, "important");
}

toggle();
