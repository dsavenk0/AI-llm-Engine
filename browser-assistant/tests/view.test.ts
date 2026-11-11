import assert from "node:assert/strict";
import { test } from "node:test";
import { Snapshot } from "../src/core/protocol";
import { metricsText, modelLabel, newer, sharedText, statusLine, updateNotice } from "../src/ui/view";

const snap = (over: Partial<Snapshot> = {}): Snapshot => ({
  version: 1, protocol: 1, engine: { state: "ready", error: null, operation: null },
  model: { state: "ready", name: "small", path: "/m/small" }, default: "/m/small", models: [],
  in_use: { inflight: 0, assistant_generations: 0 }, generations: [], metrics: {},
  extension: { versions: { safari: "0.1.0" }, folder: "/x/browser-assistant" },
  capabilities: { unload: true, per_session_model: false }, ...over,
});

test("versions compare by number, not by text", () => {
  assert.equal(newer("0.10.0", "0.9.0"), true);
  assert.equal(newer("0.1.0", "0.1.0"), false);
  assert.equal(newer("0.1", "0.1.1"), false);
  assert.equal(newer(undefined, "0.1.0"), false);
});

test("an update is announced only when the installed files are ahead, with the folder to add again", () => {
  assert.equal(updateNotice(snap(), "0.1.0", "safari", 1), "");
  const ahead = snap({ extension: { versions: { safari: "0.2.0" }, folder: "/x/ba" } });
  assert.match(updateNotice(ahead, "0.1.0", "safari", 1), /Version 0\.2\.0.*this is 0\.1\.0.*\/x\/ba\/safari/);
  assert.equal(updateNotice(ahead, "0.1.0", "chromium", 1), "");          // nothing installed for that browser
  assert.match(updateNotice(snap({ protocol: 2 }), "0.1.0", "safari", 1), /too old/);
  assert.match(updateNotice(snap({ protocol: 1 }), "0.1.0", "safari", 2), /newer than the installed engine/);
});

test("the status line tells the three states apart", () => {
  assert.deepEqual(statusLine(snap()), { text: "small — model ready", tone: "ok" });
  assert.equal(statusLine(snap({ model: { state: "generating", name: "small", path: null } })).tone, "busy");
  assert.deepEqual(statusLine(snap({ engine: { state: "offline", error: null, operation: null } })),
    { text: "Engine is off", tone: "off" });
  assert.equal(statusLine(snap({ engine: { state: "error", error: "x", operation: null } })).tone, "bad");
});

test("only measured figures are shown", () => {
  assert.equal(metricsText(snap()), "");
  assert.equal(metricsText(snap({ metrics: { model_gb: 0.41, last_decode_tps: 152.4, context_tokens: 8192 } })),
    "model 0.4 GB · context 8K · 152 tok/s");
});

test("what was shared is said in numbers, with the cut", () => {
  const text = sharedText({ title: true, url: "https://example.com/a", page_chars: 5000, page_chars_total: 90000,
    page_truncated: true, selection_chars: 0, selection_chars_total: 0, selection_truncated: false });
  assert.match(text, /from example\.com: page text, 5,000 characters of 90,000 \(cut to fit the model\)/);
});

test("a model that is unsupported or may not fit says so in the picker", () => {
  const m = { name: "big", path: "/m/big", size_gb: 7.5, format: "MLX 8-bit", route: "mlx-lm", supported: true,
    problems: [], fits: false, vision: false, thinking: false, favorite: true };
  assert.equal(modelLabel(m), "★ big — MLX 8-bit · 7.5 GB · mlx-lm · may not fit");
  assert.match(modelLabel({ ...m, supported: false }), /unsupported$/);
});
