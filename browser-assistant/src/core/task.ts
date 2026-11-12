// The life of an agent task, as a table: every state, and the only states it may go to next. A transition that
// is not here is a bug and throws — nothing wedges silently in a state nothing clears.
//
//   requested → planning → approval_required → executing → verifying → planning …
//                 │              │                                   └→ completed
//                 ├→ waiting_for_user (a question, or after a decline) → planning
//                 ├→ waiting_for_access (a site the extension may not read yet) → planning
//                 └→ completed | failed | cancelled   (any state may end in failed or cancelled)

import { TaskState } from "./agent-types";

export const NEXT: Record<TaskState, TaskState[]> = {
  requested: ["planning", "failed", "cancelled"],
  planning: ["approval_required", "waiting_for_user", "waiting_for_access", "executing", "completed", "failed", "cancelled"],
  approval_required: ["executing", "planning", "waiting_for_user", "failed", "cancelled"],
  executing: ["verifying", "waiting_for_access", "failed", "cancelled"],
  verifying: ["planning", "completed", "waiting_for_access", "failed", "cancelled"],
  waiting_for_user: ["planning", "failed", "cancelled"],
  waiting_for_access: ["planning", "failed", "cancelled"],
  completed: [],
  failed: [],
  cancelled: [],
};

export const TERMINAL: ReadonlySet<TaskState> = new Set(["completed", "failed", "cancelled"]);

export class TransitionError extends Error {}

export function transition(from: TaskState, to: TaskState): TaskState {
  if (from === to) return to;
  if (!NEXT[from].includes(to)) throw new TransitionError(`a task cannot go from ${from} to ${to}`);
  return to;
}

/** A tab id and the origin it was started on: an action is only ever carried out where the task began. */
export interface Binding { tabId: number; origin: string }

export function sameBinding(a: Binding, tabId: number, origin: string): boolean {
  return a.tabId === tabId && a.origin === origin;
}
