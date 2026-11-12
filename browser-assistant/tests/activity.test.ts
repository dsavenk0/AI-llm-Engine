import assert from "node:assert/strict";
import { test } from "node:test";
import { activityLabel, overlayFor } from "../src/core/activity";
import { AgentView } from "../src/core/agent-types";

const view = (over: Partial<AgentView>): AgentView => ({ id: "t", tabId: 1, goal: "g", state: "planning", origin: "https://a.example",
  steps: [], stepsLeft: 8, autonomy: "ask", ...over });

test("the page frame says what the agent is doing, from its state and the step the person saw", () => {
  assert.equal(activityLabel(view({ state: "planning" })), "Reading the page");
  assert.equal(activityLabel(view({ state: "executing", doing: "Click “Send”" })), "Click “Send”");
  assert.equal(activityLabel(view({ state: "approval_required", doing: "Click “Pay”" })), "Click “Pay” — waiting for your approval");
  assert.deepEqual(overlayFor(view({ state: "approval_required" })).waiting, true);
  assert.deepEqual(overlayFor(view({ state: "executing" })), { active: true, label: "Acting", waiting: false });
  for (const s of ["completed", "failed", "cancelled"] as const) assert.equal(overlayFor(view({ state: s })).active, false);
  assert.equal(overlayFor(null).active, false);
});
