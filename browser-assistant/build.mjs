// Builds the extension for each browser into dist/<browser>/ — one bundled file per entry, a merged manifest,
// the static pages and the icons. `--tests` bundles the unit tests for `node --test` instead.
import { build } from "esbuild";
import { cpSync, mkdirSync, readFileSync, readdirSync, rmSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = dirname(fileURLToPath(import.meta.url));
const read = (p) => JSON.parse(readFileSync(join(root, p), "utf8"));
const common = { bundle: true, format: "esm", target: "es2022", logLevel: "warning", loader: { ".css": "text" } };

if (process.argv.includes("--tests")) {
  const out = join(root, "dist", "tests");
  rmSync(out, { recursive: true, force: true });
  await build({ ...common, platform: "node", outdir: out, outExtension: { ".js": ".mjs" },
    entryPoints: readdirSync(join(root, "tests")).filter((f) => f.endsWith(".test.ts")).map((f) => join(root, "tests", f)) });
} else {
  const version = read("package.json").version;
  for (const browser of ["safari", "chromium"]) {
    const out = join(root, "dist", browser);
    rmSync(out, { recursive: true, force: true });
    mkdirSync(out, { recursive: true });
    await build({ ...common, outdir: out, entryPoints: {
      background: join(root, "src/background/main.ts"), popup: join(root, "src/ui/popup.ts"),
      options: join(root, "src/ui/options.ts") } });
    // the panel is injected with scripting.executeScript({files}), which runs classic scripts
    await build({ ...common, format: "iife", outdir: out, entryPoints: { panel: join(root, "src/content/panel.ts") } });
    writeFileSync(join(out, "manifest.json"),
      JSON.stringify({ ...read("manifests/base.json"), ...read(`manifests/${browser}.json`), version }, null, 2) + "\n");
    for (const page of ["popup.html", "options.html"]) cpSync(join(root, "src/ui", page), join(out, page));
    writeFileSync(join(out, "popup.css"), readFileSync(join(root, "src/ui/base.css"), "utf8")
      + readFileSync(join(root, "src/ui/popup.css"), "utf8"));
    cpSync(join(root, "icons"), join(out, "icons"), { recursive: true });
    console.log(`built dist/${browser} (v${version})`);
  }
}
