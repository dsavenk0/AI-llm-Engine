#!/usr/bin/env bash
# Builds LLMBar.app — menu-bar launcher with an icon; double-click, no terminal needed.
# The engine and CLI are bundled inside, so the app also works without this checkout:
# it then keeps config, logs and its .venv in ~/Library/Application Support/LLMBar.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
APP="$ROOT/LLMBar.app"
VERSION="${VERSION:-1.0}"

echo "▸ Compiling…"
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources/llm"
swiftc -O -o "$APP/Contents/MacOS/LLMBar" "$ROOT/app/LLMBar.swift"

echo "▸ Bundling engine + CLI…"
for f in llm llm.py agent.py requirements.txt LICENSE NOTICE; do cp "$ROOT/$f" "$APP/Contents/Resources/llm/"; done
for d in engine harness locales third_party; do ditto "$ROOT/$d" "$APP/Contents/Resources/llm/$d"; done
find "$APP/Contents/Resources/llm" -name "__pycache__" -type d -prune -exec rm -rf {} +

cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key><string>LLMBar</string>
  <key>CFBundleDisplayName</key><string>LLMBar</string>
  <key>CFBundleIdentifier</key><string>local.llm-engine.llmbar</string>
  <key>NSHumanReadableCopyright</key><string>© 2026 AI-llm-Engine · Apache-2.0</string>
  <key>CFBundleVersion</key><string>$VERSION</string>
  <key>CFBundleShortVersionString</key><string>$VERSION</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleExecutable</key><string>LLMBar</string>
  <key>CFBundleIconFile</key><string>AppIcon</string>
  <key>LSMinimumSystemVersion</key><string>13.0</string>
  <key>LSUIElement</key><true/>
  <key>NSHighResolutionCapable</key><true/>
  <key>LSApplicationCategoryType</key><string>public.app-category.developer-tools</string>
  <key>LLMRoot</key><string>$ROOT</string>
  <key>NSAppleEventsUsageDescription</key><string>Opens chat, agent and logs in Terminal.</string>
</dict>
</plist>
PLIST

echo "▸ Icon…"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
PNG="$TMP/icon-1024.png"
swift "$ROOT/tools/make-icon.swift" "$PNG"
ICONSET="$TMP/AppIcon.iconset"
mkdir -p "$ICONSET"
for s in 16 32 128 256 512; do
  sips -z $s $s "$PNG" --out "$ICONSET/icon_${s}x${s}.png" >/dev/null
  sips -z $((s * 2)) $((s * 2)) "$PNG" --out "$ICONSET/icon_${s}x${s}@2x.png" >/dev/null
done
iconutil -c icns "$ICONSET" -o "$APP/Contents/Resources/AppIcon.icns"

echo "▸ Ad-hoc signing…"
codesign --force --deep --sign - "$APP"
xattr -dr com.apple.quarantine "$APP" 2>/dev/null || true

# Refresh the icon cache: Finder/Launchpad keep the old icon in Launch Services
# until the bundle mtime changes and it is re-registered.
echo "▸ Refreshing icon cache…"
touch "$APP"
LSREGISTER="/System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister"
[ -x "$LSREGISTER" ] && "$LSREGISTER" -f "$APP" >/dev/null 2>&1 || true

echo ""
echo "✓ Done: $APP ($(du -sh "$APP" | cut -f1))"
echo "  Double-click it or drag it to /Applications. It lives in the menu bar."
