import assert from "node:assert/strict";
import { test } from "node:test";
import { wantsToSee } from "../src/core/vision";

test("a question about how the page looks asks for a picture, in English, Russian and Spanish", () => {
  for (const q of ["ты видишь изображения?", "что на картинке", "What is in this image?", "how does the chart look",
                   "¿qué hay en la foto?", "опиши скриншот"]) assert.equal(wantsToSee(q), true, q);
});

test("an ordinary question does not, and a word inside another word does not count", () => {
  for (const q of ["summarize the article", "сколько стоит доставка", "imagine a plan", "seed the database", undefined, ""])
    assert.equal(wantsToSee(q as string), false, String(q));
});
