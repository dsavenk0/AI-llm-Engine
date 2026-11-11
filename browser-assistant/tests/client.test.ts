import assert from "node:assert/strict";
import { test } from "node:test";
import { Client, ServiceFailure } from "../src/core/client";
import { ChatEvent } from "../src/core/protocol";

type Call = { url: string; init: RequestInit };
function fake(reply: (c: Call) => Response): Call[] {
  const calls: Call[] = [];
  globalThis.fetch = (async (url: string, init: RequestInit = {}) => {
    const c = { url: String(url), init };
    calls.push(c);
    return reply(c);
  }) as typeof fetch;
  return calls;
}
const sse = (events: unknown[]): Response => new Response(events.map((e) => `data: ${JSON.stringify(e)}\n\n`).join(""),
  { headers: { "Content-Type": "text/event-stream" } });
const client = new Client({ service: "http://127.0.0.1:8093/", token: "tok" });
const page = { title: "T", url: "https://example.com/a", text: "body", selection: "sel" };

test("every authenticated call carries the token, and the service address has no doubled slash", async () => {
  const calls = fake(() => Response.json({ version: 1 }));
  await client.state();
  assert.equal(calls[0].url, "http://127.0.0.1:8093/v1/assistant/state");
  assert.equal((calls[0].init.headers as Record<string, string>).Authorization, "Bearer tok");
});

test("an error body becomes a ServiceFailure with the service's own code", async () => {
  fake(() => Response.json({ error: { code: "model_in_use", message: "busy", others: 2 } }, { status: 409 }));
  await assert.rejects(client.post("/v1/assistant/models/unload", {}), (e: unknown) =>
    e instanceof ServiceFailure && e.status === 409 && e.error.code === "model_in_use" && e.error.others === 2);
});

test("chat sends the page and the selection separately and relays the stream in order", async () => {
  const calls = fake(() => sse([{ type: "start", id: "g1", model: "m", shared: {} }, { type: "delta", text: "Hi" },
    { type: "done", finish_reason: "stop", usage: null }]));
  const got: ChatEvent[] = [];
  await client.chat({ session: "s", task: "ask", question: "q?", page }, (e) => got.push(e), new AbortController().signal);
  assert.deepEqual(got.map((e) => e.type), ["start", "delta", "done"]);
  const body = JSON.parse(calls[0].init.body as string);
  assert.deepEqual(body.page, { title: "T", url: "https://example.com/a", text: "body" });
  assert.equal(body.selection, "sel");
  assert.equal(body.session, "s");
});

test("a refusal before the stream starts is reported as one error event", async () => {
  fake(() => Response.json({ error: { code: "engine_offline", message: "the engine is not running" } }, { status: 503 }));
  const got: ChatEvent[] = [];
  await client.chat({ session: "s", task: "summarize", page }, (e) => got.push(e), new AbortController().signal);
  assert.deepEqual(got, [{ type: "error", message: "the engine is not running", code: "engine_offline" }]);
});

test("cancel names the generation and the session that owns it", async () => {
  const calls = fake(() => Response.json({ cancelled: true }));
  await client.cancel("g/1", "s");
  assert.equal(calls[0].url, "http://127.0.0.1:8093/v1/assistant/generations/g%2F1/cancel");
  assert.deepEqual(JSON.parse(calls[0].init.body as string), { session: "s" });
});
