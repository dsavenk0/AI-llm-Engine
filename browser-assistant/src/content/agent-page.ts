// Installs the page agent in the extension's isolated world, once per page load. The background script calls it
// through one fixed dispatcher (see background/agent-bridge.ts): the methods below are the whole surface.
//
// Every action that touches an element first moves the visible cursor onto it (content/overlay.ts), so the person
// watching sees where the agent is about to click or type before it does.
import { createAgent, Expect, PageAgent } from "./agent-core";
import { createOverlay, OverlayState } from "./overlay";

type Installed = PageAgent & { overlay(s: OverlayState): boolean };
const g = globalThis as unknown as { __llmAssistantAgent?: Installed };
if (!g.__llmAssistantAgent) {
  const overlay = createOverlay(document);
  const agent = createAgent(window);
  const shown = async (ref: string | undefined) => {
    const r = ref ? agent.rectOf(ref) : null;
    if (r) await overlay.pointTo(r);
    return r;
  };
  g.__llmAssistantAgent = {
    ...agent,
    overlay: (s: OverlayState) => overlay.show(s),
    click: async (ref: string, expect: Expect) => {
      await shown(ref);
      const out = await agent.click(ref, expect);
      if (out.ok) overlay.mark("click");
      return out;
    },
    fill: async (ref: string, value: string, expect: Expect) => {
      const r = await shown(ref);
      const out = await agent.fill(ref, value, expect);
      if (out.ok && r) overlay.mark("type", agent.rectOf(ref) || r);
      return out;
    },
    press: async (key: string, ref: string | undefined, expect: Expect) => {
      await shown(ref);
      const out = await agent.press(key, ref, expect);
      if (out.ok) overlay.mark("click");
      return out;
    },
  };
}
