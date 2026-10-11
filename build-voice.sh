#!/usr/bin/env bash
# Builds LLMVoice.app — the small voice assistant (app/voice/main.swift): speak, hear the local model answer.
#   ./build-voice.sh [output-dir]      default: this folder (LLMVoice.app is git-ignored)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
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
  <key>LSUIElement</key><true/>
  <key>NSHighResolutionCapable</key><true/>
  <key>LLMRoot</key><string>$ROOT</string>
  <key>NSAppTransportSecurity</key><dict><key>NSAllowsLocalNetworking</key><true/></dict>
  <key>NSMicrophoneUsageDescription</key><string>You speak to the assistant; the audio is turned into text on this Mac.</string>
  <key>NSSpeechRecognitionUsageDescription</key><string>Speech recognition turns what you say into text for the local model.</string>
  <key>NSHumanReadableCopyright</key><string>© 2026 AI-llm-Engine · Apache-2.0</string>
</dict>
</plist>
PLIST

# its own icon: LLMBar's little terminal with a "))" — sound waves — in the accent yellow
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
swift "$ROOT/tools/make-icon.swift" "$TMP/icon-1024.png" "o))" FFE94D
ICONSET="$TMP/AppIcon.iconset"
mkdir -p "$ICONSET"
for s in 16 32 128 256 512; do
  sips -z $s $s "$TMP/icon-1024.png" --out "$ICONSET/icon_${s}x${s}.png" >/dev/null
  sips -z $((s * 2)) $((s * 2)) "$TMP/icon-1024.png" --out "$ICONSET/icon_${s}x${s}@2x.png" >/dev/null
done
iconutil -c icns "$ICONSET" -o "$APP/Contents/Resources/AppIcon.icns"
codesign --force --sign - "$APP" >/dev/null 2>&1 || true
echo "✓ Done: $APP ($(du -sh "$APP" | cut -f1))"
