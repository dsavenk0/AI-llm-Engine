"""tools/signing.sh — the one signing policy of build-app.sh and build-dmg.sh, both modes.

    .venv/bin/python -m unittest tests.test_signing -v

`codesign` and `security` are replaced by recording stand-ins on PATH, so the developer-id mode is checked without
a certificate (and nothing touches the keychain): the order (inside out), the hardened runtime and timestamp, and
that an extension keeps its own sandbox entitlements instead of the app's.
"""
import os
import shutil
import subprocess
import tempfile
import textwrap
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIGNING = os.path.join(ROOT, "tools", "signing.sh")

FAKE_CODESIGN = r'''#!/bin/bash
# records every call; remembers which entitlements each path was signed with, and answers -d like codesign
echo "$*" >> "$STATE/calls"
target="${@: -1}"
key="$(printf '%s' "$target" | shasum | cut -c1-16)"
if [[ " $* " == *" --sign "* ]]; then
  ent=""
  args=("$@")
  for ((i=0; i<${#args[@]}; i++)); do [ "${args[$i]}" = "--entitlements" ] && ent="${args[$((i+1))]}"; done
  if [ -n "$ent" ]; then cp "$ent" "$STATE/$key.ent"; else rm -f "$STATE/$key.ent"; fi
  [[ " $* " == *" --sign - "* ]] && echo adhoc > "$STATE/$key.kind" || echo devid > "$STATE/$key.kind"
  exit 0
fi
if [[ " $* " == *" --entitlements - "* ]]; then [ -f "$STATE/$key.ent" ] && cat "$STATE/$key.ent"; exit 0; fi
if [[ "$1" == -dv* ]]; then
  if [ "$(cat "$STATE/$key.kind" 2>/dev/null)" = "adhoc" ]; then echo "Signature=adhoc" >&2
  else printf 'TeamIdentifier=TEST123456\nCodeDirectory flags=0x10000(runtime)\n' >&2; fi
  exit 0
fi
exit 0
'''

FAKE_SECURITY = r'''#!/bin/bash
if [ "$1" = "find-identity" ]; then
  [ -n "$FAKE_IDENTITIES" ] && printf '%b' "$FAKE_IDENTITIES"
  printf '     %s valid identities found\n' "${FAKE_COUNT:-0}"
fi
exit 0
'''


@unittest.skipUnless(shutil.which("bash"), "bash is not available")
class TestSigning(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.state = os.path.join(self.d, "state")
        bin_ = os.path.join(self.d, "bin")
        os.makedirs(self.state)
        os.makedirs(bin_)
        for name, body in (("codesign", FAKE_CODESIGN), ("security", FAKE_SECURITY)):
            p = os.path.join(bin_, name)
            with open(p, "w") as f:
                f.write(body)
            os.chmod(p, 0o755)
        # a bundle shaped like LLMBar.app: a library, the OCR helper, two sandboxed extensions
        self.app = os.path.join(self.d, "LLMBar.app")
        for rel in ("Contents/MacOS/LLMBar", "Contents/Resources/llm/bin/ocr",
                    "Contents/Resources/python/lib/_mlx.so", "Contents/PlugIns/W.appex/Contents/MacOS/W",
                    "Contents/PlugIns/S.appex/Contents/MacOS/S"):
            os.makedirs(os.path.dirname(os.path.join(self.app, rel)), exist_ok=True)
            open(os.path.join(self.app, rel), "w").close()
        for name, body in (("app.ent", "allow-jit"), ("w.ent", "app-sandbox widget"), ("s.ent", "app-sandbox safari")):
            with open(os.path.join(self.d, name), "w") as f:
                f.write(body)
        self.env = dict(os.environ, PATH=bin_ + os.pathsep + os.environ["PATH"], STATE=self.state)
        for k in ("LLMBAR_SIGN", "LLMBAR_SIGN_IDENTITY", "LLMBAR_SIGN_KEYCHAIN", "LLMBAR_NOTARY_PROFILE"):
            self.env.pop(k, None)

    def tearDown(self):
        shutil.rmtree(self.d, ignore_errors=True)

    def run_sh(self, script, **env):
        full = f"set -euo pipefail\nsource {SIGNING!r}\n" + textwrap.dedent(script)
        return subprocess.run(["bash", "-c", full], capture_output=True, text=True, env=dict(self.env, **env),
                              cwd=self.d, timeout=60)

    def calls(self):
        with open(os.path.join(self.state, "calls")) as f:
            return [line.split() for line in f.read().splitlines()]

    def sign_app(self, mode, **env):
        return self.run_sh(f"""
            signing_resolve {mode}
            signing_add_nested "$PWD/LLMBar.app/Contents/PlugIns/W.appex" "$PWD/w.ent"
            signing_add_nested "$PWD/LLMBar.app/Contents/PlugIns/S.appex" "$PWD/s.ent"
            signing_app "$PWD/LLMBar.app" "$PWD/app.ent"
            echo "MODE=$SIGN_MODE ID=$SIGN_ID"
        """, **env)

    def signed(self):
        """[(target basename, has runtime, has timestamp, entitlements basename or None)] in signing order."""
        out = []
        for c in self.calls():
            if "--sign" not in c:
                continue
            ent = c[c.index("--entitlements") + 1] if "--entitlements" in c else None
            out.append((os.path.basename(c[-1]), "runtime" in c, "--timestamp" in c,
                        os.path.basename(ent) if ent else None))
        return out

    def test_adhoc_is_the_default_and_keeps_the_extension_sandbox(self):
        r = self.sign_app("")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("MODE=adhoc ID=-", r.stdout)
        s = self.signed()
        self.assertEqual([t for t, *_ in s], ["ocr", "W.appex", "S.appex", "LLMBar.app"])
        self.assertTrue(all(not rt and not ts for _, rt, ts, _ in s), "no runtime or timestamp ad hoc")
        self.assertEqual({t: e for t, _, _, e in s}["W.appex"], "w.ent")
        self.assertIn("ad hoc ✓", r.stdout)

    def test_developer_id_signs_inside_out_with_the_runtime_and_own_entitlements(self):
        r = self.sign_app("developer-id", LLMBAR_SIGN_IDENTITY="ABCDEF0123456789ABCDEF0123456789ABCDEF01")
        self.assertEqual(r.returncode, 0, r.stderr)
        s = self.signed()
        self.assertEqual([t for t, *_ in s], ["_mlx.so", "ocr", "W.appex", "S.appex", "LLMBar.app"])
        self.assertTrue(all(rt and ts for _, rt, ts, _ in s), "hardened runtime and timestamp everywhere")
        ents = {t: e for t, _, _, e in s}
        self.assertEqual((ents["W.appex"], ents["S.appex"]), ("w.ent", "s.ent"), "extensions keep their sandbox")
        self.assertEqual((ents["LLMBar.app"], ents["ocr"]), ("app.ent", "app.ent"))
        self.assertIn("team TEST123456", r.stdout)

    def test_the_older_identity_variable_still_means_developer_id(self):
        r = self.sign_app("", LLMBAR_SIGN_IDENTITY="X" * 40)
        self.assertIn("MODE=developer-id", r.stdout, r.stderr)

    def test_one_certificate_in_the_keychain_is_found(self):
        ids = '  1) 0123456789ABCDEF0123456789ABCDEF01234567 "Developer ID Application: A B (TEAM000001)"\\n'
        r = self.run_sh('signing_resolve developer-id; echo "ID=$SIGN_ID"', FAKE_IDENTITIES=ids, FAKE_COUNT="1")
        self.assertIn("ID=0123456789ABCDEF0123456789ABCDEF01234567", r.stdout, r.stderr)

    def test_none_or_several_certificates_is_an_error_not_a_guess(self):
        r = self.run_sh("signing_resolve developer-id")
        self.assertEqual(r.returncode, 2)
        self.assertIn("no \"Developer ID Application\" certificate", r.stderr)
        two = ('  1) 0123456789ABCDEF0123456789ABCDEF01234567 "Developer ID Application: A (T1)"\\n'
               '  2) 89ABCDEF0123456789ABCDEF0123456789ABCDEF "Developer ID Application: B (T2)"\\n')
        r = self.run_sh("signing_resolve developer-id", FAKE_IDENTITIES=two, FAKE_COUNT="2")
        self.assertEqual(r.returncode, 2)
        self.assertIn("2 Developer ID certificates", r.stderr)

    def test_an_unknown_mode_is_refused(self):
        r = self.run_sh("signing_resolve sideways")
        self.assertEqual(r.returncode, 2)
        self.assertIn("unknown signing mode", r.stderr)

    def test_an_extension_signed_without_its_sandbox_fails_the_build(self):
        r = self.run_sh("""
            signing_resolve adhoc
            signing_add_nested "$PWD/LLMBar.app/Contents/PlugIns/W.appex" "$PWD/app.ent"
            signing_app "$PWD/LLMBar.app" "$PWD/app.ent"
        """)
        self.assertEqual(r.returncode, 1)
        self.assertIn("without its sandbox", r.stderr)

    def test_a_developer_id_dmg_is_signed_an_adhoc_one_is_not(self):
        dmg = os.path.join(self.d, "x.dmg")
        open(dmg, "w").close()
        self.run_sh(f'signing_resolve adhoc; signing_dmg "{dmg}"')
        self.assertFalse(os.path.exists(os.path.join(self.state, "calls")))
        self.run_sh(f'signing_resolve developer-id; signing_dmg "{dmg}"', LLMBAR_SIGN_IDENTITY="Y" * 40)
        self.assertEqual(self.signed(), [("x.dmg", False, True, None)])

    def test_status_changes_nothing(self):
        r = subprocess.run([os.path.join(ROOT, "build-app.sh"), "--sign", "status"], capture_output=True, text=True,
                           env=self.env, timeout=60)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("default mode : adhoc", r.stdout)
        self.assertFalse(any("--sign" in c for c in (self.calls() if os.path.exists(
            os.path.join(self.state, "calls")) else [])))


if __name__ == "__main__":
    unittest.main()
