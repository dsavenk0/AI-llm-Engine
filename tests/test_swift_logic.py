"""The editor app's pure logic (app/editor/Logic.swift), compiled with swiftc and run.

    .venv/bin/python -m unittest tests.test_swift_logic -v
"""
import os
import shutil
import subprocess
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@unittest.skipUnless(shutil.which("swiftc"), "swiftc is not available")
class TestSwiftLogic(unittest.TestCase):
    def test_logic_checks_pass(self):
        out = os.path.join(tempfile.mkdtemp(), "logic_tests")
        build = subprocess.run(["swiftc", "-O", os.path.join(ROOT, "app/editor/Logic.swift"),
                                os.path.join(ROOT, "tests/swift/main.swift"), "-o", out],
                               capture_output=True, text=True, timeout=300)
        self.assertEqual(build.returncode, 0, build.stderr[-2000:])
        run = subprocess.run([out], capture_output=True, text=True, timeout=60)
        self.assertEqual(run.returncode, 0, run.stdout[-3000:])
        self.assertIn("checks passed", run.stdout)


if __name__ == "__main__":
    unittest.main()
