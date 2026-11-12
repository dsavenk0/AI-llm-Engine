// The real page bridge: the only way the background script reaches into a page for the agent.
//
// Everything goes through one fixed dispatcher that names a method from a short list and passes plain data.
// There is no way to hand it code, a selector or markup — and the page's own scripts live in another world and
// cannot see what the extension installed.

import { api } from "../browser-adapters/api";
import type { PageBridge } from "./agent";
import type { OverlayState } from "../content/overlay";

export const PANEL_ID = "ai-llm-engine-assistant-panel";

// Runs inside the page (it is serialised by executeScript): it must use nothing from this file.
function dispatch(method: string, args: unknown[]): unknown {
  const agent = (globalThis as unknown as Record<string, Record<string, (...a: unknown[]) => unknown>>).__llmAssistantAgent;
  const allowed = ["snapshot", "find", "inspect", "click", "fill", "scroll", "press", "generation", "overlay"];
  if (!agent || allowed.indexOf(method) < 0) return { __missing: true };
  return agent[method](...args);
}

async function call<T>(tabId: number, method: string, args: unknown[] = []): Promise<T> {
  const res = await api.scripting.executeScript({ target: { tabId }, func: dispatch, args: [method, args] });
  const out = res && res[0] ? res[0].result : undefined;
  if (out && typeof out === "object" && (out as { __missing?: boolean }).__missing) throw new Error("the page agent is not installed in this tab");
  return out as T;
}

export const bridge: PageBridge = {
  async ensure(tabId) {
    try {
      await call(tabId, "generation");
    } catch {
      await api.scripting.executeScript({ target: { tabId }, files: ["agent-page.js"] });
      await call(tabId, "generation");
    }
  },
  async panelShown(tabId) {
    const res = await api.scripting.executeScript({ target: { tabId }, func: (id: string) => !!document.getElementById(id), args: [PANEL_ID] });
    return res[0]?.result === true;
  },
  async showPanel(tabId) {
    await api.scripting.executeScript({ target: { tabId }, files: ["panel.js"] });
  },
  snapshot: (tabId) => call(tabId, "snapshot"),
  find: (tabId, query) => call(tabId, "find", [query]),
  inspect: (tabId, ref, expect) => call(tabId, "inspect", [ref, expect]),
  click: (tabId, ref, expect) => call(tabId, "click", [ref, expect]),
  fill: (tabId, ref, value, expect) => call(tabId, "fill", [ref, value, expect]),
  scroll: (tabId, direction, ref) => call(tabId, "scroll", [direction, ref]),
  press: (tabId, key, ref, expect) => call(tabId, "press", [key, ref, expect]),
};

/** Show or clear the working-here frame on a tab. Only on a tab whose page agent is already installed — the
 *  overlay never installs anything by itself, so a tab the extension may not read simply shows nothing. */
export async function overlay(tabId: number, state: OverlayState): Promise<void> {
  await call(tabId, "overlay", [state]);
}
