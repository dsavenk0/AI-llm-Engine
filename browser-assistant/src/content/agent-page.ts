// Installs the page agent in the extension's isolated world, once per page load. The background script calls it
// through one fixed dispatcher (see background/agent.ts): the methods below are the whole surface.
import { createAgent, PageAgent } from "./agent-core";

const g = globalThis as unknown as { __llmAssistantAgent?: PageAgent };
if (!g.__llmAssistantAgent) g.__llmAssistantAgent = createAgent(window);
