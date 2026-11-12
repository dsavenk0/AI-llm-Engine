#!/usr/bin/env bash
# Builds LLMBar.app — ONE app, several modules: the menu-bar panel (engine control), the code editor and the voice
# chat, with a desktop widget inside. Double-click, no terminal needed.
#   ./build-app.sh              builds LLMBar.app in this folder (git-ignored)
#   ./build-app.sh --install    also copies it to /Applications and registers the widget with the system
#   ./build-app.sh --offline    also puts Python and the engine's wheels inside (+~150 MB): works with no network
#                               at first launch; flags combine
# The engine and CLI are bundled inside, so the app also works without this checkout: it then keeps config, logs
# and its .venv in ~/Library/Application Support/LLMBar.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
FINAL="$ROOT/LLMBar.app"
VERSION="${VERSION:-1.0}"
INSTALL=0
OFFLINE=0
for arg in "$@"; do
  case "$arg" in
    --install) INSTALL=1 ;;
    --offline) OFFLINE=1 ;;
    *) echo "unknown option: $arg (use --install and/or --offline)" >&2; exit 2 ;;
  esac
done

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
# built beside, swapped in at the end: while it compiles the app in this folder stays whole (a half-built bundle
# shows Finder and the Dock the "prohibited" placeholder instead of the icon)
APP="$TMP/LLMBar.app"

echo "▸ Compiling (panel + editor + voice in one binary)…"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources/llm"
swiftc -O -target "$(uname -m)-apple-macos13.0" -o "$APP/Contents/MacOS/LLMBar" \
  "$ROOT/app/main.swift" "$ROOT"/app/core/*.swift "$ROOT"/app/bar/*.swift "$ROOT"/app/editor/*.swift "$ROOT"/app/voice/*.swift

echo "▸ OCR helper (Apple Vision)…"
mkdir -p "$APP/Contents/Resources/llm/bin"
swiftc -O -target "$(uname -m)-apple-macos13.0" -o "$APP/Contents/Resources/llm/bin/ocr" "$ROOT/app/ocr/main.swift"

echo "▸ Bundling engine + CLI…"
for f in llm llm.py agent.py requirements.txt LICENSE NOTICE; do cp "$ROOT/$f" "$APP/Contents/Resources/llm/"; done
for d in engine harness ui locales third_party model_lab assistant; do ditto "$ROOT/$d" "$APP/Contents/Resources/llm/$d"; done
# The browser extension ships built, so a Mac without Node.js can still install it (./llm assistant setup
# copies it to Application Support, and copies it again when a later build of this app brings a newer one).
if [ -f "$ROOT/browser-assistant/package.json" ] && command -v npm >/dev/null 2>&1; then
  echo "▸ Browser extension…"
  if (cd "$ROOT/browser-assistant" && npm install --no-audit --no-fund >/dev/null 2>&1 && npm run -s build >/dev/null); then
    mkdir -p "$APP/Contents/Resources/llm/browser-assistant"
    ditto "$ROOT/browser-assistant/dist/safari" "$APP/Contents/Resources/llm/browser-assistant/dist/safari"
    ditto "$ROOT/browser-assistant/dist/chromium" "$APP/Contents/Resources/llm/browser-assistant/dist/chromium"
  else
    echo "  (the extension did not build — the app is made without it)"
  fi
else
  echo "▸ Browser extension: skipped (Node.js is not installed)"
fi
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
  codesign --force --sign "${LLMBAR_SIGN_IDENTITY:--}" --entitlements "$TMP/widget.entitlements" "$WIDGET" >/dev/null 2>&1 || true
else
  echo "  (the desktop widget was not built: $(tail -1 "$TMP/widget.log"))"
  rm -rf "$APP/Contents/PlugIns"
fi

echo "▸ Safari extension (inside the app)…"
# The browser assistant as a real Safari extension: Safari lists extensions that live inside an app, so LLMBar can
# open Safari ▸ Settings ▸ Extensions on it for the person to tick (an app may not enable one by itself). The web
# extension is dist/safari; the small native handler only hands the extension the service's pairing token.
# Safari keeps an ad-hoc signed one only while Develop ▸ Allow Unsigned Extensions is on; sign with an Apple
# Development or Developer ID identity (LLMBAR_SIGN_IDENTITY) and it stays enabled across restarts.
SAFARI_EXT="$APP/Contents/PlugIns/LLMBar Assistant.appex"
ASSISTANT_HOME="$HOME/Library/Application Support/AI-llm-Engine"
if [ -f "$ROOT/browser-assistant/dist/safari/manifest.json" ]; then
  mkdir -p "$SAFARI_EXT/Contents/MacOS" "$SAFARI_EXT/Contents/Resources"
  # the extension's own build version (x.y.z.<commits>), so Safari and LLMBar can tell one build from the next
  EXT_VERSION="$(python3 -c 'import json,sys; m=json.load(open(sys.argv[1])); print(m["version"])' "$ROOT/browser-assistant/dist/safari/manifest.json")"
  EXT_SHORT="$(python3 -c 'import json,sys; m=json.load(open(sys.argv[1])); print(m.get("version_name") or m["version"])' "$ROOT/browser-assistant/dist/safari/manifest.json")"
  if swiftc -O -parse-as-library -application-extension -target "$(uname -m)-apple-macos13.0" \
       -framework SafariServices -Xlinker -e -Xlinker _NSExtensionMain \
       -o "$SAFARI_EXT/Contents/MacOS/LLMBarAssistant" "$ROOT/app/safari/SafariExtensionHandler.swift" 2>"$TMP/safari.log"; then
    ditto "$ROOT/browser-assistant/dist/safari" "$SAFARI_EXT/Contents/Resources"
    cat > "$SAFARI_EXT/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key><string>LLMBar Assistant</string>
  <key>CFBundleDisplayName</key><string>LLMBar Assistant</string>
  <key>CFBundleIdentifier</key><string>local.llm-engine.llmbar.safari</string>
  <key>CFBundleVersion</key><string>$EXT_VERSION</string>
  <key>CFBundleShortVersionString</key><string>${EXT_VERSION%.*}</string>
  <key>LLMExtensionBuild</key><string>$EXT_SHORT</string>
  <key>CFBundlePackageType</key><string>XPC!</string>
  <key>CFBundleExecutable</key><string>LLMBarAssistant</string>
  <key>LSMinimumSystemVersion</key><string>13.0</string>
  <key>LLMAssistantHome</key><string>$ASSISTANT_HOME</string>
  <key>NSHumanReadableDescription</key><string>Ask the model on this Mac about the page, or let it act on the page while you watch.</string>
  <key>NSExtension</key><dict>
    <key>NSExtensionPointIdentifier</key><string>com.apple.Safari.web-extension</string>
    <key>NSExtensionPrincipalClass</key><string>SafariWebExtensionHandler</string>
  </dict>
</dict>
</plist>
PLIST
    cat > "$TMP/safari.entitlements" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>com.apple.security.app-sandbox</key><true/>
  <key>com.apple.security.temporary-exception.files.absolute-path.read-only</key>
  <array><string>$ASSISTANT_HOME/</string></array>
</dict></plist>
PLIST
    codesign --force --sign "${LLMBAR_SIGN_IDENTITY:--}" --entitlements "$TMP/safari.entitlements" "$SAFARI_EXT" >/dev/null 2>&1 || true
  else
    echo "  (the Safari extension was not built: $(tail -1 "$TMP/safari.log"))"
    rm -rf "$SAFARI_EXT"
  fi
else
  echo "▸ Safari extension: skipped (the browser extension was not built)"
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

if [[ "$OFFLINE" == 1 ]]; then
  echo "▸ Bundling Python and the engine's wheels (offline build)…"
  python3 "$ROOT/tools/bundle-deps.py" "$APP"
fi

# Ad hoc by default; a stable identity also keeps the microphone permission across rebuilds. With a real
# Developer ID, sign the way notarization requires — Apple rejects a submission that lacks the hardened
# runtime or a secure timestamp, and neither can be applied to an ad-hoc signature.
#
# The hardened runtime then needs telling that this app legitimately loads code it did not sign: the
# bundled Python and MLX are .so files from pip, and MLX compiles Metal kernels at runtime. Without these
# three entitlements the app signs and notarizes but refuses to start.
if [ -n "${LLMBAR_SIGN_IDENTITY:-}" ]; then
  echo "▸ Signing with ${LLMBAR_SIGN_IDENTITY} (hardened runtime, timestamped)…"
  cat > "$TMP/app.entitlements" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>com.apple.security.cs.disable-library-validation</key><true/>
  <key>com.apple.security.cs.allow-jit</key><true/>
  <key>com.apple.security.cs.allow-unsigned-executable-memory</key><true/>
</dict></plist>
PLIST
  # nested code first, outside in: a bundle is only valid once everything inside it is signed
  find "$APP/Contents" \( -name "*.dylib" -o -name "*.so" \) -print0 2>/dev/null \
    | xargs -0 -r codesign --force --timestamp --options runtime --sign "$LLMBAR_SIGN_IDENTITY" 2>/dev/null || true
  for nested in "$APP/Contents/Resources/llm/bin/ocr" "$APP/Contents/PlugIns/"*.appex; do
    [ -e "$nested" ] && codesign --force --timestamp --options runtime \
      --entitlements "$TMP/app.entitlements" --sign "$LLMBAR_SIGN_IDENTITY" "$nested" || true
  done
  codesign --force --timestamp --options runtime --entitlements "$TMP/app.entitlements" \
    --sign "$LLMBAR_SIGN_IDENTITY" "$APP"
  codesign --verify --strict --verbose=2 "$APP" 2>&1 | tail -2
else
  echo "▸ Ad-hoc signing…"
  codesign --force --sign - "$APP"
fi
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
  [ -d "$DEST/Contents/PlugIns/LLMBar Assistant.appex" ] && pluginkit -a "$DEST/Contents/PlugIns/LLMBar Assistant.appex" >/dev/null 2>&1 || true
  echo "✓ Installed: $DEST — add the widget: right-click the desktop → Edit Widgets → LLMBar"
fi

echo ""
echo "✓ Done: $APP ($(du -sh "$APP" | cut -f1))"
echo "  Double-click it or drag it to /Applications. It lives in the menu bar; the editor and the voice chat open from it."
