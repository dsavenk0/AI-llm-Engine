// Settings: where the assistant service is, and the token that pairs this browser with it.

import { api } from "../browser-adapters/api";
import { Client, ServiceFailure } from "../core/client";
import { DEFAULT_SERVICE, PROTOCOL } from "../core/protocol";

const $ = <T extends HTMLElement>(id: string): T => document.getElementById(id) as T;
const service = $<HTMLInputElement>("service"), token = $<HTMLInputElement>("token"), result = $("result");

async function load(): Promise<void> {
  const s = await api.storage.local.get(["service", "token"]);
  service.value = (s.service as string) || DEFAULT_SERVICE;
  token.value = (s.token as string) || "";
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
  } catch (e) {
    if (e instanceof ServiceFailure && e.status === 401) say("The service refused the token.", false);
    else say("No answer from the service. Start it with: ./llm assistant install", false);
  }
}

$("save").addEventListener("click", () => void save());
$("forget").addEventListener("click", async () => {
  await api.storage.local.set({ token: "" });
  token.value = "";
  say("Token removed from this browser.", true);
});
void load();
