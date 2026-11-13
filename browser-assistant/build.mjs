// Builds the extension for each browser into dist/<browser>/ — one bundled file per entry, a merged manifest,
// the static pages and the icons. `--tests` bundles the unit tests for `node --test` instead.
import { build } from "esbuild";
import { execSync } from "node:child_process";
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
  // Every build gets its own version: package.json's x.y.z plus the commit count as a fourth part (browsers take
  // up to four dotted numbers), and the short commit in version_name. Safari and Chrome see a newer extension,
  // and the side panel, the service and LLMBar can tell builds apart.
  const git = (cmd) => { try { return execSync(`git ${cmd}`, { cwd: root, stdio: ["ignore", "pipe", "ignore"] }).toString().trim(); } catch { return ""; } };
  const base = read("package.json").version;
  const count = Math.min(65535, parseInt(git("rev-list --count HEAD") || "0", 10) || 0);
  const sha = git("rev-parse --short HEAD");
  const dirty = git("status --porcelain -- .") ? "+" : "";
  const version = `${base}.${count}`;
  const versionName = sha ? `${base} (${sha}${dirty})` : base;
  for (const browser of ["safari", "chromium", "firefox"]) {
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
    const baseManifest = read("manifests/base.json"), own = read(`manifests/${browser}.json`);
    const permissions = [...baseManifest.permissions, ...(own.extra_permissions || [])];
    delete own.extra_permissions;
    const manifest = { ...baseManifest, ...own, permissions, version, version_name: versionName };
    // Chrome-specific WAR hardening is not needed on Firefox and is not part of its manifest schema.
    if (browser === "firefox") {
      for (const resource of manifest.web_accessible_resources || []) delete resource.use_dynamic_url;
    }
    writeFileSync(join(out, "manifest.json"), JSON.stringify(manifest, null, 2) + "\n");
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
