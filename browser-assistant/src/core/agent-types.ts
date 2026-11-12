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
  | { action: "ask_user"; question: string }
  | { action: "done"; answer: string };

export type Risk = "low" | "high";

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
  proposal?: { action: Action; preview: Preview };
  message?: string;           // the final answer, a question for the person, or what went wrong
  needsAccess?: string;       // an origin the extension may not read yet
}

export interface AuditEntry {
  t: number;
  task: string;
  origin: string;
  kind: "start" | "propose" | "approve" | "reject" | "execute" | "result" | "refuse" | "navigate" | "stop";
  what: string;
  ok?: boolean;
}
