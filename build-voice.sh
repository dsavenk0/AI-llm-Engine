#!/usr/bin/env bash
# Builds LLMVoice.app — the small voice assistant (app/voice/main.swift): speak, hear the local model answer.
#   ./build-voice.sh [output-dir]      default: this folder (LLMVoice.app is git-ignored)
#   ./build-voice.sh --install         also copy it to /Applications and register its desktop widget
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
INSTALL=0
if [[ "${1:-}" == "--install" ]]; then INSTALL=1; shift; fi
OUT="${1:-$ROOT}"
APP="$OUT/LLMVoice.app"
VERSION="${VERSION:-1.0}"

echo "▸ Compiling the voice assistant…"
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
swiftc -O -target "$(uname -m)-apple-macos13.0" -o "$APP/Contents/MacOS/LLMVoice" "$ROOT"/app/voice/*.swift

cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key><string>LLM Voice</string>
  <key>CFBundleDisplayName</key><string>LLM Voice</string>
  <key>CFBundleIdentifier</key><string>local.llm-engine.voice</string>
  <key>CFBundleVersion</key><string>$VERSION</string>
  <key>CFBundleShortVersionString</key><string>$VERSION</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleExecutable</key><string>LLMVoice</string>
  <key>CFBundleIconFile</key><string>AppIcon</string>
  <key>LSMinimumSystemVersion</key><string>13.0</string>
  <key>NSHighResolutionCapable</key><true/>
  <key>CFBundleURLTypes</key><array><dict>
    <key>CFBundleURLName</key><string>LLM Voice</string>
    <key>CFBundleURLSchemes</key><array><string>llmvoice</string></array>
  </dict></array>
  <key>LLMRoot</key><string>$ROOT</string>
  <key>NSAppTransportSecurity</key><dict><key>NSAllowsLocalNetworking</key><true/></dict>
  <key>NSMicrophoneUsageDescription</key><string>You speak to the assistant; the audio is turned into text on this Mac.</string>
  <key>NSSpeechRecognitionUsageDescription</key><string>Speech recognition turns what you say into text for the local model.</string>
  <key>NSHumanReadableCopyright</key><string>© 2026 AI-llm-Engine · Apache-2.0</string>
</dict>
</plist>
PLIST

# its own icon (app/voice/icon-1024.png): the orb on the family's dark tile
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
cp "$ROOT/app/voice/icon-1024.png" "$TMP/icon-1024.png"
ICONSET="$TMP/AppIcon.iconset"
mkdir -p "$ICONSET"
for s in 16 32 128 256 512; do
  sips -z $s $s "$TMP/icon-1024.png" --out "$ICONSET/icon_${s}x${s}.png" >/dev/null
  sips -z $((s * 2)) $((s * 2)) "$TMP/icon-1024.png" --out "$ICONSET/icon_${s}x${s}@2x.png" >/dev/null
done
iconutil -c icns "$ICONSET" -o "$APP/Contents/Resources/AppIcon.icns"
# the desktop widget (macOS 14+): a WidgetKit extension that only draws the orb and opens the app on a tap.
# It needs the App Sandbox entitlement and a signature (ad hoc is enough on this Mac); skipped if it cannot build.
WIDGET="$APP/Contents/PlugIns/LLMVoiceWidget.appex"
# where the config LLMBar and ./llm share lives (the widget reads the selected model from it)
if [[ -f "$ROOT/llm.py" ]]; then LLM_HOME_DIR="$ROOT"; else LLM_HOME_DIR="$HOME/Library/Application Support/LLMBar"; fi
mkdir -p "$WIDGET/Contents/MacOS"
if swiftc -O -parse-as-library -application-extension -target "$(uname -m)-apple-macos14.0" \
     -framework WidgetKit -framework SwiftUI -Xlinker -e -Xlinker _NSExtensionMain \
     -o "$WIDGET/Contents/MacOS/LLMVoiceWidget" "$ROOT/app/voice/widget/LLMVoiceWidget.swift" 2>"$TMP/widget.log"; then
  cat > "$WIDGET/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key><string>LLM Voice Widget</string>
  <key>CFBundleDisplayName</key><string>LLM Voice</string>
  <key>CFBundleIdentifier</key><string>local.llm-engine.voice.widget</string>
  <key>CFBundleVersion</key><string>$VERSION</string>
  <key>CFBundleShortVersionString</key><string>$VERSION</string>
  <key>CFBundlePackageType</key><string>XPC!</string>
  <key>CFBundleExecutable</key><string>LLMVoiceWidget</string>
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
codesign --force --sign - "$APP" >/dev/null 2>&1 || true
echo "✓ Done: $APP ($(du -sh "$APP" | cut -f1))"
if [[ "$INSTALL" == 1 ]]; then
  # the widget gallery only lists extensions of apps that live in /Applications (or ~/Applications)
  DEST="/Applications/LLMVoice.app"
  rm -rf "$DEST"
  ditto "$APP" "$DEST"
  /System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister -f "$DEST" >/dev/null 2>&1 || true
  pluginkit -a "$DEST/Contents/PlugIns/LLMVoiceWidget.appex" >/dev/null 2>&1 || true
  echo "✓ Installed: $DEST — add the widget: right-click the desktop → Edit Widgets → LLM Voice"
fi
