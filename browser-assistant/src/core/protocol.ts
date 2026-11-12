// The assistant service's wire format (docs/safari-assistant/PROTOCOL.md), as types.

export const PROTOCOL = 1;
export const DEFAULT_SERVICE = "http://127.0.0.1:8093";

export type EngineState = "offline" | "starting" | "ready" | "error";
export type ModelState = "not_loaded" | "loading" | "ready" | "generating" | "unloading" | "error";

export interface ModelInfo {
  name: string;
  path: string;
  size_gb: number;
  format: string | null;
  route: string | null;
  supported: boolean;
  problems: string[];
  fits: boolean;
  vision: boolean;
  thinking: boolean;
  favorite: boolean;
}

export interface Snapshot {
  version: number;
  protocol: number;
  engine: { state: EngineState; error: string | null; operation: string | null };
  model: { state: ModelState; name: string | null; path: string | null };
  default: string;
  models: ModelInfo[];
  in_use: { inflight: number; assistant_generations: number };
  generations: { id: string; session: string; since: number }[];
  metrics: Partial<Record<"context_tokens" | "model_gb" | "active_gb" | "system_ram_used" | "last_decode_tps" |
    "last_prefill_tps", number>>;
  extension: { versions: Record<string, string>; folder: string };
  capabilities: { unload: boolean; per_session_model: boolean; per_session_model_reason?: string };
}

export type Task = "ask" | "summarize" | "explain" | "translate" | "rewrite" | "key_points";

export interface PageContext {
  title: string;
  url: string;
  text: string;
  selection: string;
}

export interface Shared {
  title: boolean;
  url: string;
  page_chars: number;
  page_chars_total: number;
  page_truncated: boolean;
  selection_chars: number;
  selection_chars_total: number;
  selection_truncated: boolean;
  screenshot?: boolean;
  pictures?: number;
  described?: boolean;
}

export type ChatEvent =
  | { type: "start"; id: string; model: string; shared: Shared }
  | { type: "reasoning"; text: string }
  | { type: "delta"; text: string }
  | { type: "done"; finish_reason: string | null; usage: Record<string, number> | null }
  | { type: "cancelled" }
  | { type: "error"; message: string; code?: string };

export interface ServiceError {
  code: string;
  message: string;
  others?: number;
  default?: string;
  state?: Snapshot;
}

// What the UI pages and the in-page panel say to the background, and what it says back.
import { AgentView, Autonomy } from "./agent-types";

export type UiRequest =
  | { kind: "subscribe" }
  | { kind: "agent-start"; tabId: number; goal: string; autonomy?: Autonomy; see?: boolean }
  | { kind: "agent-approve" } | { kind: "agent-reject" } | { kind: "agent-cancel" } | { kind: "agent-resume" }
  | { kind: "agent-reply"; text: string }
  | { kind: "command"; command: Command; confirm?: boolean }
  | { kind: "ask"; tabId: number; task: Task; question?: string; lang?: string; tone?: string; thinking?: boolean;
      selectionOnly?: boolean; selection?: string; screenshot?: boolean; history?: { q: string; a: string }[] }
  | { kind: "cancel" }
  | { kind: "watch"; tabId: number }
  | { kind: "undock" }
  | { kind: "reload" };                        // the person chose to load the newer version now                        // the docked panel (Safari) closes itself          // the side panel follows the active tab of its window

export type Command =
  | { op: "start" } | { op: "stop" } | { op: "load" } | { op: "unload" }
  | { op: "default"; path: string; expected: string };

export type UiEvent =
  | { kind: "state"; state: Snapshot | null; connected: boolean; problem?: string;
      app?: { version: string } }                // inside LLMBar (Safari): the extension version the app now ships
  | { kind: "command-result"; ok: boolean; error?: ServiceError; command: Command }
  | { kind: "chat"; event: ChatEvent }
  | { kind: "page"; title: string; url: string; chars: number }
  | { kind: "shot"; dataUrl: string }                   // the screenshot that went with a question, to show it
  | { kind: "agent"; view: AgentView | null }
  | { kind: "tab"; tabId: number; title: string; url: string }
  | { kind: "fatal"; message: string };
