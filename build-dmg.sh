#!/usr/bin/env bash
# Builds a styled installer DMG with LLMBar.app and an Applications shortcut.
# Uses only stock macOS tools: swift, hdiutil, Finder and SetFile.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
APP="$ROOT/LLMBar.app"
OUT_DIR="$ROOT/docs"            # the site serves this file (vercel.json: outputDirectory docs)
VOL_NAME="LLMBar"
DMG_WIDTH=680
DMG_HEIGHT=440
STAGE="$(mktemp -d "$ROOT/.dmg-stage.XXXXXX")"
RW_DMG="$ROOT/.LLMBar-rw.dmg"
DEVICE=""

cleanup() {
  if [[ -n "$DEVICE" ]]; then hdiutil detach "$DEVICE" -quiet 2>/dev/null || true; fi
  rm -rf "$STAGE" "$RW_DMG"
}
trap cleanup EXIT

SUFFIX=""
OFFLINE_ARG=()
if [[ "${1:-}" == "--offline" ]]; then SUFFIX="-offline"; OFFLINE_ARG=(--offline); fi   # Python + wheels inside

echo "▸ Building the app…"
"$ROOT/build-app.sh" ${OFFLINE_ARG[@]+"${OFFLINE_ARG[@]}"}

VERSION="$(/usr/libexec/PlistBuddy -c 'Print :CFBundleShortVersionString' "$APP/Contents/Info.plist")"
FINAL_DMG="$OUT_DIR/LLMBar-$VERSION$SUFFIX.dmg"
mkdir -p "$STAGE/.background" "$OUT_DIR"

echo "▸ Staging DMG contents…"
ditto "$APP" "$STAGE/LLMBar.app"
ln -s /Applications "$STAGE/Applications"
swift "$ROOT/tools/make-dmg-background.swift" "$STAGE/.background/background.png"
SetFile -a V "$STAGE/.background"

echo "▸ Creating image…"
rm -f "$FINAL_DMG" "$RW_DMG"
APP_MB="$(du -sm "$APP" | cut -f1)"
hdiutil create -quiet -size "$((APP_MB + 60))m" -fs HFS+ -volname "$VOL_NAME" -format UDRW "$RW_DMG" -srcfolder "$STAGE"

echo "▸ Arranging icons in Finder…"
DEVICE="$(hdiutil attach -readwrite -noverify -noautoopen "$RW_DMG" | awk '/\/dev\/disk/ {print $1; exit}')"
[[ -n "$DEVICE" ]] || { echo "Could not mount the DMG." >&2; exit 1; }

osascript <<APPLESCRIPT
tell application "Finder"
  tell disk "$VOL_NAME"
    open
    set current view of container window to icon view
    set toolbar visible of container window to false
    set statusbar visible of container window to false
    set bounds of container window to {100, 100, $((100 + DMG_WIDTH)), $((100 + DMG_HEIGHT))}
    set viewOptions to the icon view options of container window
    set arrangement of viewOptions to not arranged
    set icon size of viewOptions to 96
    set text size of viewOptions to 13
    set background picture of viewOptions to file ".background:background.png"
    set position of item "LLMBar.app" to {155, 245}
    set position of item "Applications" to {525, 245}
    close
    open
    update without registering applications
    delay 2
  end tell
end tell
APPLESCRIPT

sync
hdiutil detach "$DEVICE" -quiet
DEVICE=""

echo "▸ Compressing final DMG…"
hdiutil convert "$RW_DMG" -quiet -format UDZO -imagekey zlib-level=9 -o "$FINAL_DMG"

echo "✓ Done: $FINAL_DMG ($(du -h "$FINAL_DMG" | cut -f1))"
echo "  Open it and drag LLMBar to Applications."

# A stable-named copy for the website's one-click button. GitHub serves
# releases/latest/download/LLMBar.dmg as an attachment, so the page downloads straight away
# instead of bouncing to the releases list — but only if the release carries this exact name.
# Upload BOTH assets (the versioned one for humans, LLMBar.dmg for the link).
STABLE_DMG="$OUT_DIR/LLMBar.dmg"
cp -f "$FINAL_DMG" "$STABLE_DMG"
echo "✓ Stable copy for the site's download button: $STABLE_DMG"
