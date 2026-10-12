#!/usr/bin/env bash
# Builds LLMBar.app — ONE app, several modules: the menu-bar panel (engine control), the code editor and the voice
# chat, with a desktop widget inside. Double-click, no terminal needed.
#   ./build-app.sh              builds LLMBar.app in this folder (git-ignored)
#   ./build-app.sh --install    also copies it to /Applications and registers the widget with the system
# The engine and CLI are bundled inside, so the app also works without this checkout: it then keeps config, logs
# and its .venv in ~/Library/Application Support/LLMBar.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
FINAL="$ROOT/LLMBar.app"
VERSION="${VERSION:-1.0}"
INSTALL=0
[[ "${1:-}" == "--install" ]] && INSTALL=1

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
# built beside, swapped in at the end: while it compiles the app in this folder stays whole (a half-built bundle
# shows Finder and the Dock the "prohibited" placeholder instead of the icon)
APP="$TMP/LLMBar.app"

echo "▸ Compiling (panel + editor + voice in one binary)…"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources/llm"
swiftc -O -target "$(uname -m)-apple-macos13.0" -o "$APP/Contents/MacOS/LLMBar" \
  "$ROOT/app/main.swift" "$ROOT"/app/core/*.swift "$ROOT"/app/bar/*.swift "$ROOT"/app/editor/*.swift "$ROOT"/app/voice/*.swift

echo "▸ Bundling engine + CLI…"
for f in llm llm.py agent.py requirements.txt LICENSE NOTICE; do cp "$ROOT/$f" "$APP/Contents/Resources/llm/"; done
for d in engine harness ui locales third_party; do ditto "$ROOT/$d" "$APP/Contents/Resources/llm/$d"; done
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
  <key>CFBundleURLTypes</key><array><dict>
    <key>CFBundleURLName</key><string>LLMBar</string>
    <key>CFBundleURLSchemes</key><array><string>llmbar</string></array>
  </dict></array>
  <key>LSMinimumSystemVersion</key><string>13.0</string>
  <key>LSUIElement</key><true/>
  <key>NSHighResolutionCapable</key><true/>
  <key>LSApplicationCategoryType</key><string>public.app-category.developer-tools</string>
  <key>LLMRoot</key><string>$ROOT</string>
  <key>NSAppTransportSecurity</key><dict><key>NSAllowsLocalNetworking</key><true/></dict>
  <key>NSAppleEventsUsageDescription</key><string>Opens chat, agent and logs in Terminal.</string>
  <key>NSMicrophoneUsageDescription</key><string>You speak to the assistant; the audio is turned into text on this Mac.</string>
  <key>NSSpeechRecognitionUsageDescription</key><string>Speech recognition turns what you say into text for the local model.</string>
</dict>
</plist>
PLIST

echo "▸ Desktop widget (macOS 14+)…"
# a WidgetKit extension: it draws the model name and the orb, and every tap is a llmbar:// link into this app.
# It needs the App Sandbox entitlement and a signature (ad hoc is enough on this Mac); skipped if it cannot build.
WIDGET="$APP/Contents/PlugIns/LLMBarWidget.appex"
if [[ -f "$ROOT/llm.py" ]]; then LLM_HOME_DIR="$ROOT"; else LLM_HOME_DIR="$HOME/Library/Application Support/LLMBar"; fi
mkdir -p "$WIDGET/Contents/MacOS"
if swiftc -O -parse-as-library -application-extension -target "$(uname -m)-apple-macos14.0" \
     -framework WidgetKit -framework SwiftUI -Xlinker -e -Xlinker _NSExtensionMain \
     -o "$WIDGET/Contents/MacOS/LLMBarWidget" "$ROOT/app/widget/LLMBarWidget.swift" 2>"$TMP/widget.log"; then
  cat > "$WIDGET/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key><string>LLMBar Widget</string>
  <key>CFBundleDisplayName</key><string>LLMBar</string>
  <key>CFBundleIdentifier</key><string>local.llm-engine.llmbar.widget</string>
  <key>CFBundleVersion</key><string>$VERSION</string>
  <key>CFBundleShortVersionString</key><string>$VERSION</string>
  <key>CFBundlePackageType</key><string>XPC!</string>
  <key>CFBundleExecutable</key><string>LLMBarWidget</string>
  <key>LSMinimumSystemVersion</key><string>14.0</string>
  <key>LLMHome</key><string>$LLM_HOME_DIR</string>
  <key>NSExtension</key><dict>
    <key>NSExtensionPointIdentifier</key><string>com.apple.widgetkit-extension</string>
  </dict>
</dict>
</plist>
PLIST
  cat > "$TMP/widget.entitlements" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>com.apple.security.app-sandbox</key><true/>
  <key>com.apple.security.network.client</key><true/>
  <key>com.apple.security.temporary-exception.files.absolute-path.read-only</key>
  <array><string>$LLM_HOME_DIR/.llm/</string></array>
</dict></plist>
PLIST
  codesign --force --sign - --entitlements "$TMP/widget.entitlements" "$WIDGET" >/dev/null 2>&1 || true
else
  echo "  (the desktop widget was not built: $(tail -1 "$TMP/widget.log"))"
  rm -rf "$APP/Contents/PlugIns"
fi

echo "▸ Icon…"
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
codesign --force --sign - "$APP"
xattr -dr com.apple.quarantine "$APP" 2>/dev/null || true

rm -rf "$FINAL"
ditto "$APP" "$FINAL"
APP="$FINAL"

# Refresh the icon cache: Finder/Launchpad keep the old icon in Launch Services
# until the bundle mtime changes and it is re-registered.
echo "▸ Refreshing icon cache…"
touch "$APP"
LSREGISTER="/System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister"
[ -x "$LSREGISTER" ] && "$LSREGISTER" -f "$APP" >/dev/null 2>&1 || true

if [[ "$INSTALL" == 1 ]]; then
  # the widget gallery only lists extensions of apps that live in /Applications (or ~/Applications)
  DEST="/Applications/LLMBar.app"
  rm -rf "$DEST"
  ditto "$APP" "$DEST"
  [ -x "$LSREGISTER" ] && "$LSREGISTER" -f "$DEST" >/dev/null 2>&1 || true
  [ -d "$DEST/Contents/PlugIns/LLMBarWidget.appex" ] && pluginkit -a "$DEST/Contents/PlugIns/LLMBarWidget.appex" >/dev/null 2>&1 || true
  echo "✓ Installed: $DEST — add the widget: right-click the desktop → Edit Widgets → LLMBar"
fi

echo ""
echo "✓ Done: $APP ($(du -sh "$APP" | cut -f1))"
echo "  Double-click it or drag it to /Applications. It lives in the menu bar; the editor and the voice chat open from it."
