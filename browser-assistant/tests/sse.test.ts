import assert from "node:assert/strict";
import { test } from "node:test";
import { readSse, SseParser } from "../src/core/sse";

test("events split across chunks are joined, comments and bad JSON skipped", () => {
  const p = new SseParser();
  assert.deepEqual(p.push('data: {"a"'), []);
  assert.deepEqual(p.push(':1}\n'), []);
  assert.deepEqual(p.push('\n: keep-alive\n\ndata: not json\n\ndata: {"b":2}\r\n\r\n'), [{ a: 1 }, { b: 2 }]);
});

test("a multi-line data field is one event", () => {
  const p = new SseParser();
  assert.deepEqual(p.push('data: {"t":\ndata: "x"}\n\n'), [{ t: "x" }]);
});

test("readSse decodes UTF-8 cut in the middle of a character and flushes the tail", async () => {
  const bytes = new TextEncoder().encode('data: {"text":"привет"}\n\ndata: {"end":true}');
  const cut = 20;                                   // inside a two-byte character
  const stream = new ReadableStream<Uint8Array>({
    start(c) {
      c.enqueue(bytes.slice(0, cut));
      c.enqueue(bytes.slice(cut));
      c.close();
    },
  });
  const got: unknown[] = [];
  await readSse(stream, (e) => got.push(e));
  assert.deepEqual(got, [{ text: "привет" }, { end: true }]);
});
