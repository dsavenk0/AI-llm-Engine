import assert from "node:assert/strict";
import { test } from "node:test";
import { Conversations, historyFor, MAX_CONVERSATIONS, Store } from "../src/ui/conversations";

function memory(): Store & { data: Record<string, unknown> } {
  const data: Record<string, unknown> = {};
  return { data, get: async (keys) => Object.fromEntries(keys.map((k) => [k, data[k]])),
           set: async (items) => { Object.assign(data, JSON.parse(JSON.stringify(items))); } };
}

test("an exchange starts a conversation named after its question, and later ones join it", async () => {
  let t = 1000;
  const c = new Conversations(memory(), () => t++);
  const id = await c.add("c1", { q: "What is on this page?", a: "A menu." }, { url: "https://eats.example/menu" });
  await c.add(id, { q: "And the cheapest?", a: "Lemonade." }, { url: "https://eats.example/menu" });
  const got = await c.get(id);
  assert.equal(got!.title, "What is on this page?");
  assert.equal(got!.host, "eats.example");
  assert.deepEqual(got!.turns.map((x) => x.q), ["What is on this page?", "And the cheapest?"]);
  assert.deepEqual(historyFor(got, 1), [{ q: "And the cheapest?", a: "Lemonade." }]);
});

test("a tab remembers its conversation; rename and delete work, and delete unbinds the tab", async () => {
  const c = new Conversations(memory());
  await c.add("c1", { q: "hi", a: "hello" }, { url: "https://a.example/" });
  await c.bind(7, "c1");
  assert.equal(await c.forTab(7), "c1");
  await c.rename("c1", "  Greetings  ");
  assert.equal((await c.get("c1"))!.title, "Greetings");
  await c.remove("c1");
  assert.equal(await c.get("c1"), undefined);
  assert.equal(await c.forTab(7), undefined);
});

test("the newest conversations are kept, the oldest give way", async () => {
  let t = 0;
  const c = new Conversations(memory(), () => t++);
  for (let i = 0; i < MAX_CONVERSATIONS + 5; i++) await c.add(`c${i}`, { q: `q${i}`, a: "a" }, { url: "https://x.example/" });
  const all = await c.list();
  assert.equal(all.length, MAX_CONVERSATIONS);
  assert.equal(all[0].id, `c${MAX_CONVERSATIONS + 4}`);
  assert.equal(await c.get("c0"), undefined);
});
