// Settings: where the assistant service is, and the token that pairs this browser with it.

import { api } from "../browser-adapters/api";
import { Client, ServiceFailure } from "../core/client";
import { DEFAULT_SERVICE, PROTOCOL } from "../core/protocol";

const $ = <T extends HTMLElement>(id: string): T => document.getElementById(id) as T;
const service = $<HTMLInputElement>("service"), token = $<HTMLInputElement>("token"), result = $("result");
const googleClient = $<HTMLInputElement>("google-client-id"), workspaceResult = $("workspace-result");
const visionModel = $<HTMLSelectElement>("vision-model"), visionResult = $("vision-result");
const visionInstall = $<HTMLButtonElement>("vision-install");
const workspaceConnect = $<HTMLButtonElement>("workspace-connect");
const workspaceDisconnect = $<HTMLButtonElement>("workspace-disconnect");
const workspaceAdvanced = $<HTMLDetailsElement>("workspace-advanced");
const workspaceSettings = $("workspace-settings");

async function pairFromApp(): Promise<string> {
  if (!api.runtime.sendNativeMessage) return "";
  try {
    const reply = await api.runtime.sendNativeMessage("local.llm_engine.assistant", { kind: "token" });
    if (reply?.ok && typeof reply.token === "string" && reply.token.length >= 24) {
      await api.storage.local.set({ token: reply.token });
      return reply.token;
    }
  } catch {
    // The extension may be running outside LLMBar; the manual clipboard flow remains available.
  }
  return "";
}

async function load(): Promise<void> {
  const s = await api.storage.local.get(["service", "token"]);
  service.value = (s.service as string) || DEFAULT_SERVICE;
  token.value = (s.token as string) || await pairFromApp();
  if (token.value) say("Paired with LLMBar. Press Connect.", true);
  void loadVision();
  void refreshWorkspace();
}

function say(text: string, ok: boolean): void {
  result.textContent = text;
  result.className = ok ? "ok" : "bad";
}

async function save(): Promise<void> {
  let url: URL;
  try {
    url = new URL(service.value.trim() || DEFAULT_SERVICE);
  } catch {
    return say("That is not a URL.", false);
  }
  // the service is local by design: the token and page text must never be sent to another machine
  if (url.protocol !== "http:" || !["127.0.0.1", "localhost"].includes(url.hostname)) {
    return say("The service must be on this Mac: http://127.0.0.1:<port>.", false);
  }
  if (!token.value.trim()) token.value = await pairFromApp();
  if (!token.value.trim()) {
    return say("LLMBar did not provide a token. In LLMBar, open Tools → Browser Assistant · Safari, then paste the copied token here.", false);
  }
  const settings = { service: url.origin, token: token.value.trim() };
  const c = new Client(settings);
  try {
    const hello = await c.hello();
    if (hello.protocol !== PROTOCOL) {
      return say(`The service speaks protocol ${hello.protocol}, this extension ${PROTOCOL}. Update the older one.`, false);
    }
    const st = await c.state();
    await api.storage.local.set(settings);
    say(`Connected. ${st.models.length} model(s) installed; engine ${st.engine.state}.`, true);
    void loadVision();
    void refreshWorkspace(c);
  } catch (e) {
    if (e instanceof ServiceFailure && e.status === 401) say("The service refused the token.", false);
    else say("No answer from the service. Start it with: ./llm assistant install", false);
  }
}

function workspaceSay(text: string, ok: boolean): void { workspaceResult.textContent = text; workspaceResult.className = ok ? "ok" : "bad"; }
async function loadVision(): Promise<void> {
  const c = await workspaceClient(); if (!c) return;
  try {
    const st = await c.state();
    const models = st.models.filter((m) => m.vision && m.supported);
    visionModel.replaceChildren(...models.map((m) => {
      const o = document.createElement("option"); o.value = m.path;
      o.textContent = `${m.name} · ${m.size_gb.toFixed(1)} GB${m.fits ? "" : " · may not fit"}`;
      o.selected = m.path === st.capabilities.eyes?.model; return o;
    }));
    visionResult.textContent = models.length ? `Current: ${st.capabilities.eyes?.source || "automatic"}.`
      : "No installed vision model; Apple Vision OCR remains available.";
  } catch { visionResult.textContent = "Connect LLMBar first."; }
}
$<HTMLButtonElement>("vision-save").addEventListener("click", async () => {
  const c = await workspaceClient(); if (!c || !visionModel.value) return;
  try { await c.post("/v1/assistant/vision/configure", { path: visionModel.value });
    visionResult.textContent = "Vision model saved. It will be loaded only when a page needs eyes.";
  } catch (e) { visionResult.textContent = e instanceof Error ? e.message : "Could not save the vision model."; }
});
visionInstall.addEventListener("click", async () => {
  const c = await workspaceClient(); if (!c) return;
  if (!confirm("Find and download a compatible local vision model? Model weights can use several GB of disk space.")) return;
  visionInstall.disabled = true;
  visionResult.className = "";
  visionResult.textContent = "Finding a compatible model and downloading it… You can keep using the browser.";
  try {
    await c.post("/v1/assistant/vision/install", {});
    await loadVision();
    visionResult.className = "ok";
    visionResult.textContent = "Eyes are installed. The vision model will load only when a page or attachment needs it.";
  } catch (e) {
    visionResult.className = "bad";
    visionResult.textContent = e instanceof ServiceFailure ? e.error.message
      : e instanceof Error ? e.message : "Could not install a vision model.";
  } finally { visionInstall.disabled = false; }
});
interface WorkspaceStatus { configured: boolean; connected: boolean; client: "bundled" | "custom" | null; scopes: string[] }
async function workspaceClient(): Promise<Client | null> {
  const s = await api.storage.local.get(["service", "token"]);
  if (!(s.token as string)) { workspaceSay("Connect LLMBar first.", false); return null; }
  return new Client({ service: (s.service as string) || DEFAULT_SERVICE, token: s.token as string });
}
async function refreshWorkspace(c?: Client): Promise<WorkspaceStatus | null> {
  const client = c || await workspaceClient(); if (!client) return null;
  try {
    const st = await client.get<WorkspaceStatus>("/v1/assistant/workspace/status");
    // Keep the complete connector dormant until a release includes its public Desktop client ID (or a person
    // already configured one). The browser agent itself continues through DOM/vision and asks for no Google auth.
    workspaceSettings.hidden = !st.configured && !st.connected;
    if (workspaceSettings.hidden) {
      workspaceAdvanced.open = false;
      workspaceResult.textContent = "";
      return st;
    }
    workspaceConnect.textContent = st.connected ? "Reconnect or change permissions" : "Connect Google Workspace";
    workspaceDisconnect.disabled = !st.connected;
    if (st.connected) workspaceSay("Google Workspace is connected. Credentials are stored in macOS Keychain.", true);
    else if (st.configured) workspaceSay("Ready to connect. Google sign-in will open in this browser.", true);
    return st;
  } catch { workspaceSay("Connect LLMBar first to check Google Workspace.", false); return null; }
}
workspaceConnect.addEventListener("click", async () => {
  const c = await workspaceClient(); if (!c) return;
  const before = await refreshWorkspace(c);
  if (!before?.configured && !googleClient.value.trim()) {
    workspaceAdvanced.open = true; googleClient.focus();
    return workspaceSay("Paste a Google Desktop OAuth client ID, then press Connect again.", false);
  }
  const button = workspaceConnect; button.disabled = true;
  try {
    if (googleClient.value.trim()) await c.post("/v1/assistant/workspace/configure", { client_id: googleClient.value.trim() });
    const r = await c.post<{ started: boolean }>("/v1/assistant/workspace/authorize",
      // Least privilege: the first Gmail action creates drafts and never reads or sends mail.
      { scopes: ["sheets", "docs", "drive", "gmail_compose", "calendar"] });
    workspaceSay(r.started ? "Finish Google sign-in in the opened tab…" : "Could not start Google sign-in.", !!r.started);
    if (r.started) for (let i = 0; i < 60; i += 1) {
      await new Promise((resolve) => setTimeout(resolve, 2000));
      const st = await refreshWorkspace(c); if (st?.connected) break;
    }
  } catch (e) { workspaceSay(e instanceof Error ? e.message : "Could not connect Google Workspace.", false); }
  finally { button.disabled = false; }
});
workspaceDisconnect.addEventListener("click", async () => {
  const c = await workspaceClient(); if (!c) return;
  try {
    await c.post("/v1/assistant/workspace/disconnect", {});
    await refreshWorkspace(c);
    workspaceSay("Google session removed from macOS Keychain.", true);
  }
  catch (e) { workspaceSay(e instanceof Error ? e.message : "Could not remove Google session.", false); }
});

$("save").addEventListener("click", () => void save());
$("forget").addEventListener("click", async () => {
  await api.storage.local.set({ token: "" });
  token.value = "";
  say("Token removed from this browser.", true);
});
async function showActivity(): Promise<void> {
  const list = $("activity");
  const log = ((await api.storage.local.get("audit")).audit as { t: number; origin: string; kind: string; what: string; ok?: boolean }[] | undefined) || [];
  list.replaceChildren();
  if (!log.length) {
    const li = document.createElement("li");
    li.textContent = "Nothing yet.";
    list.append(li);
    return;
  }
  for (const e of log.slice(-60).reverse()) {
    const li = document.createElement("li");
    li.className = e.ok === false ? "bad" : "";
    li.textContent = `${new Date(e.t).toLocaleTimeString()} · ${e.origin.replace(/^https?:\/\//, "")} · ${e.kind}: ${e.what}`;
    list.append(li);
  }
}
$("clear-log").addEventListener("click", async () => { await api.storage.local.set({ audit: [] }); await showActivity(); });
void showActivity();
void load();
