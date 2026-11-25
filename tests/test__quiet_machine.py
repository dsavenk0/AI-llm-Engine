"""Loaded first by `unittest discover -s tests` (the name sorts before every other test module), before any test
runs: the RAM guard sees a quiet machine for the whole run — and so do the servers the tests start, which inherit
the environment. See tests/__init__.py and engine/memory.py (LLM_TEST_FREE_RAM)."""
import os
import sys
import unittest
import unittest.mock

os.environ.setdefault("LLM_TEST_FREE_RAM", "0.6")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine import memory  # noqa: E402


class TestQuietMachine(unittest.TestCase):
    def test_the_suite_sees_a_quiet_machine(self):
        self.assertAlmostEqual(memory.used_fraction(), 0.4, places=2)
        self.assertEqual(memory.pressure_level(), 1)

    def test_a_bad_value_means_the_real_readings(self):
        with unittest.mock.patch.dict(os.environ, {"LLM_TEST_FREE_RAM": "lots"}):
            self.assertIsNone(memory.simulated_free())


if __name__ == "__main__":
    unittest.main()
