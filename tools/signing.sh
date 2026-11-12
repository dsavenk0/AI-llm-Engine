# LLMBar's one signing policy — sourced by build-app.sh and build-dmg.sh, never run on its own.
#
# Two modes, one switch:
#   adhoc         the default. Signed with "-": runs on the Mac that built it; a downloaded copy is stopped once by
#                 Gatekeeper. Needs nothing — no Apple account, no certificate, works on CI.
#   developer-id  signed with a "Developer ID Application" certificate, the hardened runtime and a secure
#                 timestamp — what notarization requires. With LLMBAR_NOTARY_PROFILE set, build-dmg.sh also
#                 notarizes and staples the DMG, and a download opens with no warning.
#
# The mode, first match wins:   --sign <mode> on the command line  ·  LLMBAR_SIGN=<mode>
#                               ·  developer-id when LLMBAR_SIGN_IDENTITY is set (the older switch)  ·  adhoc
# The identity (developer-id):  LLMBAR_SIGN_IDENTITY (a name or SHA-1)  ·  else the single Developer ID
#                               Application identity in the keychain — none or several is an error, not a guess
# LLMBAR_SIGN_KEYCHAIN          a keychain file to take the identity from (CI imports its certificate into one)
#
#   ./build-app.sh --sign status   what would be used, and what is missing — changes nothing

SIGN_MODE=""
SIGN_ID=""
SIGN_KEYCHAIN_ARGS=()
SIGN_NESTED=()            # "path|entitlements" of nested bundles, signed with their own entitlements (sandbox)

signing_identities() {
  # the Developer ID Application identities that can sign, one "SHA1 name" per line
  local kc=()
  [ -n "${LLMBAR_SIGN_KEYCHAIN:-}" ] && kc=("$LLMBAR_SIGN_KEYCHAIN")
  security find-identity -v -p codesigning ${kc[@]+"${kc[@]}"} 2>/dev/null \
    | sed -nE 's/^ *[0-9]+\) ([0-9A-F]{40}) "(Developer ID Application: [^"]*)".*/\1 \2/p'
}

# signing_resolve [mode-from-command-line] — sets SIGN_MODE and SIGN_ID, or exits with what to do.
signing_resolve() {
  local asked="${1:-}"
  SIGN_MODE="${asked:-${LLMBAR_SIGN:-}}"
  if [ -z "$SIGN_MODE" ]; then
    if [ -n "${LLMBAR_SIGN_IDENTITY:-}" ]; then SIGN_MODE="developer-id"; else SIGN_MODE="adhoc"; fi
  fi
  if [ -n "${LLMBAR_SIGN_KEYCHAIN:-}" ] && [ "$SIGN_MODE" != "status" ]; then
    # codesign only finds identities in keychains on the search list; --keychain then narrows it to this one
    if ! security list-keychains -d user 2>/dev/null | grep -qF "$(basename "$LLMBAR_SIGN_KEYCHAIN")"; then
      echo "✗ $LLMBAR_SIGN_KEYCHAIN is not on the keychain search list; add it for the build:" >&2
      echo "  security list-keychains -d user -s \"$LLMBAR_SIGN_KEYCHAIN\" \$(security list-keychains -d user | tr -d '\"')" >&2
      exit 2
    fi
    SIGN_KEYCHAIN_ARGS=(--keychain "$LLMBAR_SIGN_KEYCHAIN")
  fi
  case "$SIGN_MODE" in
    adhoc|ad-hoc)
      SIGN_MODE="adhoc"; SIGN_ID="-" ;;
    developer-id|devid|developer)
      SIGN_MODE="developer-id"
      if [ -n "${LLMBAR_SIGN_IDENTITY:-}" ]; then
        SIGN_ID="$LLMBAR_SIGN_IDENTITY"
      else
        local found count
        found="$(signing_identities)"
        count="$(printf '%s' "$found" | grep -c . || true)"
        if [ "$count" = "1" ]; then
          SIGN_ID="${found%% *}"
        elif [ "$count" = "0" ]; then
          echo "✗ developer-id signing: no \"Developer ID Application\" certificate in the keychain." >&2
          echo "  Create one at developer.apple.com (Certificates ▸ +), install it, or set LLMBAR_SIGN_IDENTITY." >&2
          echo "  Or build with --sign adhoc." >&2
          exit 2
        else
          echo "✗ developer-id signing: $count Developer ID certificates — say which one:" >&2
          printf '%s\n' "$found" | sed 's/^/    /' >&2
          echo "  LLMBAR_SIGN_IDENTITY=<SHA-1 or full name> ./build-app.sh --sign developer-id" >&2
          exit 2
        fi
      fi ;;
    status) ;;
    *)
      echo "✗ unknown signing mode: $SIGN_MODE (adhoc | developer-id | status)" >&2
      exit 2 ;;
  esac
}

signing_status() {
  echo "Signing"
  local mode="${LLMBAR_SIGN:-}"
  [ -z "$mode" ] && { [ -n "${LLMBAR_SIGN_IDENTITY:-}" ] && mode="developer-id" || mode="adhoc"; }
  echo "  default mode : $mode  (change: --sign adhoc|developer-id, or LLMBAR_SIGN)"
  local ids
  ids="$(signing_identities)"
  if [ -n "$ids" ]; then
    echo "  Developer ID : $(printf '%s\n' "$ids" | grep -c .) certificate(s) in the keychain"
    printf '%s\n' "$ids" | sed -E 's/^([0-9A-F]{8})[0-9A-F]+ /    \1… /'
  else
    echo "  Developer ID : none in the keychain — developer-id builds need one (developer.apple.com)"
  fi
  [ -n "${LLMBAR_SIGN_IDENTITY:-}" ] && echo "  identity     : LLMBAR_SIGN_IDENTITY is set"
  if [ -n "${LLMBAR_NOTARY_PROFILE:-}" ]; then
    echo "  notarization : LLMBAR_NOTARY_PROFILE is set — build-dmg.sh notarizes developer-id builds"
  else
    echo "  notarization : off (xcrun notarytool store-credentials <name>, then LLMBAR_NOTARY_PROFILE=<name>)"
  fi
  if [ -d "${1:-}" ]; then
    echo "  last build   : $(signing_describe "$1")"
  fi
}

# signing_describe <bundle> — one line: how an existing bundle is signed.
signing_describe() {
  local info
  info="$(codesign -dv "$1" 2>&1)" || { echo "not signed"; return; }
  if printf '%s' "$info" | grep -q "Signature=adhoc"; then
    echo "ad hoc"
  else
    local team auth
    team="$(printf '%s' "$info" | sed -nE 's/^TeamIdentifier=(.*)/\1/p')"
    auth="$(codesign -dvv "$1" 2>&1 | sed -nE 's/^Authority=(Developer ID Application[^(]*).*/\1/p' | head -1)"
    echo "${auth:-signed} (team ${team:-?})$(printf '%s' "$info" | grep -q 'runtime' && echo ', hardened runtime')"
  fi
}

# signing_code <path> [entitlements] — one item, the way the mode requires; a failure stops the build.
signing_code() {
  local path="$1" ent="${2:-}" args=(--force --sign "$SIGN_ID")
  if [ "$SIGN_MODE" = "developer-id" ]; then
    args+=(--timestamp --options runtime ${SIGN_KEYCHAIN_ARGS[@]+"${SIGN_KEYCHAIN_ARGS[@]}"})
  fi
  [ -n "$ent" ] && args+=(--entitlements "$ent")
  codesign "${args[@]}" "$path"
}

# signing_add_nested <bundle> <entitlements> — remember an extension to sign with its own entitlements.
signing_add_nested() { SIGN_NESTED+=("$1|$2"); }

# signing_app <app> <app-entitlements> — inside out: loose libraries, helpers, extensions (each with its own
# entitlements — an extension must keep its sandbox), then the app; then verify.
signing_app() {
  local app="$1" app_ent="$2"
  echo "▸ Signing ($SIGN_MODE)…"
  if [ "$SIGN_MODE" = "developer-id" ]; then
    # loose Mach-O inside Resources (bundled Python and wheels): notarization rejects any unsigned one
    while IFS= read -r -d '' lib; do
      signing_code "$lib" >/dev/null
    done < <(find "$app/Contents" \( -name "*.dylib" -o -name "*.so" \) -print0 2>/dev/null)
  fi
  [ -e "$app/Contents/Resources/llm/bin/ocr" ] && signing_code "$app/Contents/Resources/llm/bin/ocr" \
    "$([ "$SIGN_MODE" = "developer-id" ] && echo "$app_ent")"
  local item path ent
  for item in ${SIGN_NESTED[@]+"${SIGN_NESTED[@]}"}; do
    path="${item%%|*}"; ent="${item#*|}"
    [ -e "$path" ] && signing_code "$path" "$ent"
  done
  signing_code "$app" "$([ "$SIGN_MODE" = "developer-id" ] && echo "$app_ent")"
  signing_verify "$app"
}

signing_verify() {
  local app="$1"
  codesign --verify --deep --strict "$app" || { echo "✗ the signature does not verify" >&2; exit 1; }
  local item path
  for item in ${SIGN_NESTED[@]+"${SIGN_NESTED[@]}"}; do       # an extension that lost its sandbox will not load
    path="${item%%|*}"
    if [ -e "$path" ] && ! codesign -d --entitlements - "$path" 2>/dev/null | grep -q "app-sandbox"; then
      echo "✗ $(basename "$path") is signed without its sandbox entitlement" >&2
      exit 1
    fi
  done
  echo "  signature: $(signing_describe "$app") ✓"
}

# signing_dmg <dmg> — a developer-id DMG is signed too (Gatekeeper checks the image before mounting it).
signing_dmg() {
  if [ "$SIGN_MODE" = "developer-id" ]; then
    codesign --force --timestamp --sign "$SIGN_ID" ${SIGN_KEYCHAIN_ARGS[@]+"${SIGN_KEYCHAIN_ARGS[@]}"} "$1"
    codesign --verify --strict "$1"
    echo "  DMG signed: $(signing_describe "$1")"
  fi
}
