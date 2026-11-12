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
  await build({ ...common, platform: "node", outdir: out, outExtension: { ".js": ".mjs" }, external: ["jsdom"],
    entryPoints: readdirSync(join(root, "tests")).filter((f) => f.endsWith(".test.ts")).map((f) => join(root, "tests", f)) });
} else {
  const version = read("package.json").version;
  for (const browser of ["safari", "chromium"]) {
    const out = join(root, "dist", browser);
    rmSync(out, { recursive: true, force: true });
    mkdirSync(out, { recursive: true });
    await build({ ...common, outdir: out, entryPoints: {
      background: join(root, "src/background/main.ts"),
      options: join(root, "src/ui/options.ts"), sidepanel: join(root, "src/ui/sidepanel.ts") } });
    // the panel is injected with scripting.executeScript({files}), which runs classic scripts
    await build({ ...common, format: "iife", outdir: out, entryPoints: {
      panel: join(root, "src/content/panel.ts"), "agent-page": join(root, "src/content/agent-page.ts"),
      dock: join(root, "src/content/dock.ts") } });
    // the browser's file overrides the base, except `permissions`, which it adds to
    const base = read("manifests/base.json"), own = read(`manifests/${browser}.json`);
    const permissions = [...base.permissions, ...(own.extra_permissions || [])];
    delete own.extra_permissions;
    writeFileSync(join(out, "manifest.json"),
      JSON.stringify({ ...base, ...own, permissions, version }, null, 2) + "\n");
    cpSync(join(root, "src/ui/options.html"), join(out, "options.html"));
    writeFileSync(join(out, "options.css"), readFileSync(join(root, "src/ui/base.css"), "utf8")
      + readFileSync(join(root, "src/ui/options.css"), "utf8"));
    cpSync(join(root, "src/ui/sidepanel.html"), join(out, "sidepanel.html"));
    writeFileSync(join(out, "sidepanel.css"), readFileSync(join(root, "src/ui/base.css"), "utf8")
      + readFileSync(join(root, "src/ui/agent.css"), "utf8") + readFileSync(join(root, "src/ui/sidepanel.css"), "utf8"));
    cpSync(join(root, "icons"), join(out, "icons"), { recursive: true });
    console.log(`built dist/${browser} (v${version})`);
  }
}
