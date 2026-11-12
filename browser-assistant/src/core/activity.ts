// One sentence for what the agent is doing right now — shown on the page under the working-here frame and in the
// side panel. Built from the task's state and the step the person was shown, never from the model's own words.

import { AgentView } from "./agent-types";

const TERMINAL = new Set(["completed", "failed", "cancelled"]);

export function activityLabel(v: AgentView): string {
  switch (v.state) {
    case "requested":
    case "planning": return "Reading the page";
    case "approval_required": return v.doing ? `${v.doing} — waiting for your approval` : "Waiting for your approval";
    case "executing": return v.doing || "Acting";
    case "verifying": return v.doing ? `${v.doing} — checking what happened` : "Checking what happened";
    case "waiting_for_user": return "Waiting for your answer";
    case "waiting_for_access": return "Waiting for access to this site";
    default: return "";
  }
}

/** What the page overlay should show for this task (or nothing, once it is over). */
export function overlayFor(v: AgentView | null): { active: boolean; label: string; waiting: boolean } {
  if (!v || TERMINAL.has(v.state)) return { active: false, label: "", waiting: false };
  const waiting = v.state === "approval_required" || v.state === "waiting_for_user" || v.state === "waiting_for_access";
  return { active: true, label: activityLabel(v), waiting };
}
