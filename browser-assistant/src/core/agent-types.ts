// The browser agent's shared shapes: what the page script reports, what the model proposes, what the person is
// shown, and where a task is in its life.

export interface ElementInfo {
  ref: string;               // "e12": valid only for the snapshot generation that issued it
  tag: string;
  role: string;
  name: string;              // the accessible name: what a person or a screen reader would call it
  type: string;
  value: string;             // never set for a sensitive field
  placeholder: string;
  href: string;              // absolute, credentials and fragment removed
  disabled: boolean;
  checked: boolean;
  required: boolean;
  options: string[];
  dup: number;               // how many elements share this role and name, when more than one
  sensitive: boolean;        // a password, card number, one-time code…: the agent may not fill it
  blocked: boolean;          // inside a CAPTCHA: the agent may not touch it
  inForm: boolean;
  formAction: string;        // the form's destination, absolute
  formSensitive: boolean;    // the form it sits in holds a sensitive field
  formSearch: boolean;       // the form is a search box: submitting it only looks something up
}

export interface PageSnapshot {
  generation: number;
  url: string;
  title: string;
  text: string;
  elements: ElementInfo[];
  captcha: boolean;
  truncated: boolean;
}

export type Action =
  | { action: "click"; ref: string; why: string }
  | { action: "fill"; ref: string; value: string; why: string }
  | { action: "fill_form"; fields: { ref: string; value: string }[]; why: string }
  | { action: "navigate"; url: string; why: string }
  | { action: "find"; query: string }
  | { action: "scroll"; direction: "up" | "down"; ref?: string }
  | { action: "press"; key: Key; ref?: string; why: string }
  | { action: "wait"; seconds: number }
  | { action: "ask_user"; question: string }
  | { action: "done"; answer: string };

export type Risk = "low" | "high";

/** The keys the agent may press — navigation and confirmation keys, never text or shortcuts. */
export const KEYS = ["Enter", "Tab", "Escape", "ArrowDown", "ArrowUp", "ArrowLeft", "ArrowRight", "PageDown", "PageUp",
  "Home", "End", "Backspace", "Space"] as const;
export type Key = typeof KEYS[number];

/** How much the agent may do without asking. "ask": every step that changes something waits for approval.
 *  "auto": low-risk steps (scrolling, finding, typing into ordinary fields, links within the site) go ahead on
 *  their own; anything the policy marks high-risk — submitting, paying, ordering, sending, deleting, leaving
 *  the site, a form holding a password or card — still stops and asks, every time. */
export type Autonomy = "ask" | "auto";

/** What the person is asked to approve. Built from the live element, never from the model's words. */
export interface Preview {
  title: string;
  lines: string[];
  risk: Risk;
  reasons: string[];
  destination?: string;
}

export type TaskState =
  | "requested" | "planning" | "approval_required" | "executing" | "verifying"
  | "waiting_for_user" | "waiting_for_access" | "completed" | "failed" | "cancelled";

export interface StepView {
  summary: string;
  result: string;
  ok: boolean;
}

export interface AgentView {
  id: string;
  tabId: number;
  goal: string;
  state: TaskState;
  origin: string;
  steps: StepView[];
  stepsLeft: number;
  autonomy: Autonomy;
  proposal?: { action: Action; preview: Preview };
  message?: string;           // the final answer, a question for the person, or what went wrong
  needsAccess?: string;       // an origin the extension may not read yet
  doing?: string;             // the step being approved, done or checked right now, as the person was shown it
}

export interface AuditEntry {
  t: number;
  task: string;
  origin: string;
  kind: "start" | "propose" | "approve" | "reject" | "execute" | "result" | "refuse" | "navigate" | "stop";
  what: string;
  ok?: boolean;
}
