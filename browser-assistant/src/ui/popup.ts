// The toolbar popup: engine and model controls, and questions about the active page.

import { api } from "../browser-adapters/api";
import { Command, ServiceError, Snapshot, Task, UiEvent, UiRequest } from "../core/protocol";
import { browserName } from "../browser-adapters/api";
import { PROTOCOL } from "../core/protocol";
import { Answer, el, metricsText, modelLabel, problemText, statusLine, updateNotice } from "./view";

const port = api.runtime.connect({ name: "popup" });
const post = (m: UiRequest): void => port.postMessage(m);
const $ = <T extends HTMLElement>(id: string): T => document.getElementById(id) as T;

const dot = $("dot"), status = $("status"), metrics = $("metrics"), problem = $("problem");
const picker = $<HTMLSelectElement>("model"), controls = $("controls"), note = $("note");
const question = $<HTMLTextAreaElement>("question"), askBtn = $<HTMLButtonElement>("ask");
const stopBtn = $<HTMLButtonElement>("stop-generation"), pageLine = $("page"), version = $("version");
const answer = new Answer();
$("answer-slot").append(answer.root);

const MINE = (api.runtime as unknown as { getManifest(): { version: string } }).getManifest().version;
let state: Snapshot | null = null;
let confirming: Command | null = null;

function button(label: string, title: string, cmd: Command, danger = false): HTMLButtonElement {
  const b = el("button", { type: "button", textContent: label, title, class: danger ? "danger" : "" });
  b.addEventListener("click", () => run(cmd, false));
  return b;
}

function run(command: Command, confirm: boolean): void {
  note.textContent = "";
  confirming = null;
  post({ kind: "command", command, confirm });
}

function render(ev: Extract<UiEvent, { kind: "state" }>): void {
  state = ev.state;
  const connected = ev.connected && state !== null;
  problem.hidden = connected;
  problem.textContent = connected ? "" : problemText(ev.problem);
  $("main").hidden = !connected;
  if (!state) {
    dot.className = "dot off";
    status.textContent = "Not connected";
    metrics.textContent = "";
    return;
  }
  const line = statusLine(state);
  dot.className = "dot " + line.tone;
  status.textContent = line.text;
  metrics.textContent = metricsText(state);
  if (state.engine.error) note.textContent = state.engine.error;
  const notice = updateNotice(state, MINE, browserName === "chromium" ? "chromium" : "safari", PROTOCOL);
  $("update").hidden = !notice;
  $("update").textContent = notice;

  // the picker shows the installed models; the selected one is the global default, shared with the CLI and LLMBar
  const current = state.default;
  picker.replaceChildren(...state.models.map((m) => el("option", {
    value: m.path, textContent: modelLabel(m), disabled: !m.supported, selected: m.path === current })));
  if (!state.models.length) picker.append(el("option", { textContent: "No models installed — ./llm pull <org/name>" }));
  const busy = state.engine.operation !== null;
  picker.disabled = busy || !state.models.length;

  controls.replaceChildren();
  const e = state.engine.state, m = state.model.state;
  if (e === "offline" || e === "error") {
    controls.append(button("Start engine", "Start the engine and load the selected model", { op: "start" }));
  } else if (e === "ready") {
    if (m === "not_loaded") controls.append(button("Load model", "Load the weights into memory now", { op: "load" }));
    if ((m === "ready" || m === "generating") && state.capabilities.unload) {
      controls.append(button("Unload model", "Free the model's memory; the engine keeps running", { op: "unload" }));
    }
    controls.append(button("Stop engine", "Quit the engine process — for every client", { op: "stop" }, true));
  }
  for (const b of Array.from(controls.querySelectorAll("button"))) b.disabled = busy;
  askBtn.disabled = e !== "ready" || answer.active;
  stopBtn.hidden = !answer.active;
}

function conflict(error: ServiceError, command: Command): void {
  if (error.code === "model_in_use") {
    confirming = command;
    note.replaceChildren(error.message + " ");
    const yes = el("button", { type: "button", textContent: "Do it anyway", class: "danger" });
    yes.addEventListener("click", () => confirming && run(confirming, true));
    note.append(yes);
  } else {
    note.textContent = error.message;
  }
}

port.onMessage.addListener((ev: UiEvent) => {
  if (ev.kind === "state") render(ev);
  else if (ev.kind === "command-result" && !ev.ok && ev.error) conflict(ev.error, ev.command);
  else if (ev.kind === "page") pageLine.textContent = `${ev.title || ev.url}`;
  else if (ev.kind === "fatal") {
    answer.fail(ev.message);
    stopBtn.hidden = true;
    askBtn.disabled = false;
  } else if (ev.kind === "chat") {
    answer.on(ev.event);
    stopBtn.hidden = !answer.active;
    askBtn.disabled = answer.active || state?.engine.state !== "ready";
  }
});

picker.addEventListener("change", () => {
  if (state && picker.value !== state.default) run({ op: "default", path: picker.value, expected: state.default }, false);
});

async function ask(task: Task, text?: string): Promise<void> {
  const [tab] = await api.tabs.query({ active: true, currentWindow: true });
  if (!tab || tab.id === undefined) {
    answer.fail("No active tab.");
    return;
  }
  answer.begin();
  stopBtn.hidden = false;
  askBtn.disabled = true;
  const lang = navigator.language.startsWith("ru") ? "Russian" : "English";
  post({ kind: "ask", tabId: tab.id, task, question: text, lang, thinking: $<HTMLInputElement>("thinking").checked });
}

askBtn.addEventListener("click", () => {
  const q = question.value.trim();
  void ask(q ? "ask" : "summarize", q || undefined);
});
question.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey && !askBtn.disabled) {
    e.preventDefault();
    askBtn.click();
  }
});
for (const b of Array.from(document.querySelectorAll<HTMLButtonElement>("[data-task]"))) {
  b.addEventListener("click", () => void ask(b.dataset.task as Task));
}
stopBtn.addEventListener("click", () => post({ kind: "cancel" }));
$("settings").addEventListener("click", () => void api.runtime.openOptionsPage());
$("copy").addEventListener("click", () => void navigator.clipboard.writeText(answer.text()));

version.textContent = "v" + MINE;
void api.tabs.query({ active: true, currentWindow: true }).then(([tab]) => {
  pageLine.textContent = tab?.title || tab?.url || "";
});
post({ kind: "subscribe" });
