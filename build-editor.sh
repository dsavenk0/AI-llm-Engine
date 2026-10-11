#!/usr/bin/env bash
# Builds LLMEditor.app — the native, on-demand code editor (app/editor/*.swift).
#   ./build-editor.sh [output-dir]      default: this folder (LLMEditor.app is git-ignored)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
OUT="${1:-$ROOT}"
APP="$OUT/LLMEditor.app"
VERSION="${VERSION:-1.0}"

echo "▸ Compiling the editor…"
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
swiftc -O -target "$(uname -m)-apple-macos13.0" -o "$APP/Contents/MacOS/LLMEditor" "$ROOT"/app/editor/*.swift

cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key><string>LLM Editor</string>
  <key>CFBundleDisplayName</key><string>LLM Editor</string>
  <key>CFBundleIdentifier</key><string>local.llm-engine.editor</string>
  <key>CFBundleVersion</key><string>$VERSION</string>
  <key>CFBundleShortVersionString</key><string>$VERSION</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleExecutable</key><string>LLMEditor</string>
  <key>CFBundleIconFile</key><string>AppIcon</string>
  <key>LSMinimumSystemVersion</key><string>13.0</string>
  <key>NSHighResolutionCapable</key><true/>
  <key>LLMRoot</key><string>$ROOT</string>
  <key>NSHumanReadableCopyright</key><string>© 2026 AI-llm-Engine · Apache-2.0</string>
</dict>
</plist>
PLIST

# reuse the engine's icon when it has been built
for icon in "$ROOT/LLMBar.app/Contents/Resources/AppIcon.icns" "$OUT/../Resources/AppIcon.icns"; do
  [ -f "$icon" ] && cp "$icon" "$APP/Contents/Resources/AppIcon.icns" && break
done
codesign --force --sign - "$APP" >/dev/null 2>&1 || true
echo "✓ Done: $APP ($(du -sh "$APP" | cut -f1))"
