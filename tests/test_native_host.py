"""The Chromium native-messaging host that pairs the extension (assistant/native_host.py)."""
import io
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from assistant import native_host  # noqa: E402


def framed(obj):
    data = json.dumps(obj).encode()
    return struct.pack("=I", len(data)) + data


class TestNativeHost(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.tok = os.path.join(self.dir, "browser-assistant.json")

    def test_it_answers_the_token_and_nothing_else(self):
        with open(self.tok, "w") as f:
            json.dump({"token": "t" * 40, "port": 8093}, f)
        self.assertEqual(native_host.answer({"kind": "token"}, self.tok), {"ok": True, "token": "t" * 40})
        self.assertFalse(native_host.answer({"kind": "shell", "cmd": "ls"}, self.tok)["ok"])
        self.assertFalse(native_host.answer("token", self.tok)["ok"])
        self.assertFalse(native_host.answer({"kind": "token"}, os.path.join(self.dir, "missing.json"))["ok"])

    def test_the_wire_format_round_trips_and_refuses_huge_messages(self):
        self.assertEqual(native_host.read_message(io.BytesIO(framed({"kind": "token"}))), {"kind": "token"})
        self.assertIsNone(native_host.read_message(io.BytesIO(b"")))
        self.assertIsNone(native_host.read_message(io.BytesIO(struct.pack("=I", 10**7))))
        out = io.BytesIO()
        native_host.write_message(out, {"ok": False})
        self.assertEqual(native_host.read_message(io.BytesIO(out.getvalue())), {"ok": False})

    def test_install_writes_a_manifest_only_for_installed_browsers_and_the_launcher_works(self):
        support = os.path.join(self.dir, "support")
        os.makedirs(os.path.join(support, "Google", "Chrome"))
        home = os.path.join(self.dir, "data", "browser-assistant")
        os.makedirs(os.path.dirname(home))
        with open(os.path.join(self.dir, "data", "browser-assistant.json"), "w") as f:
            json.dump({"token": "x" * 32}, f)
        done = native_host.install(home, python=sys.executable, support=support)
        self.assertEqual(done, ["Chrome"])
        m = json.load(open(os.path.join(support, "Google", "Chrome", "NativeMessagingHosts", native_host.NAME + ".json")))
        self.assertEqual(m["allowed_origins"], [f"chrome-extension://{native_host.EXTENSION_ID}/"])
        self.assertFalse(os.path.exists(os.path.join(support, "Chromium")))          # not installed: left alone
        r = subprocess.run([m["path"]], input=framed({"kind": "token"}), capture_output=True, timeout=30)
        self.assertEqual(native_host.read_message(io.BytesIO(r.stdout)), {"ok": True, "token": "x" * 32})

    def test_the_id_matches_the_key_in_the_chromium_manifest(self):
        import base64
        import hashlib
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        key = json.load(open(os.path.join(root, "browser-assistant", "manifests", "chromium.json")))["key"]
        h = hashlib.sha256(base64.b64decode(key)).hexdigest()[:32]
        self.assertEqual("".join(chr(ord("a") + int(c, 16)) for c in h), native_host.EXTENSION_ID)

    def test_firefox_install_registers_its_gecko_extension_id(self):
        support = os.path.join(self.dir, "support")
        home = os.path.join(self.dir, "data", "browser-assistant")
        done = native_host.install(home, python=sys.executable, support=support, firefox=True)
        self.assertEqual(done, ["Firefox"])

        host = os.path.join(support, "Mozilla", "NativeMessagingHosts", native_host.NAME + ".json")
        with open(host) as f:
            manifest = json.load(f)
        self.assertEqual(manifest["allowed_extensions"], [native_host.FIREFOX_EXTENSION_ID])
        self.assertNotIn("allowed_origins", manifest)

        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(root, "browser-assistant", "manifests", "firefox.json")) as f:
            addon = json.load(f)
        self.assertEqual(addon["browser_specific_settings"]["gecko"]["id"], native_host.FIREFOX_EXTENSION_ID)


if __name__ == "__main__":
    unittest.main()
