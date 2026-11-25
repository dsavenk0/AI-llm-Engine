"""The test suite. Everything that imports tests.* (a module run as `-m unittest tests.test_x`, or one that imports
tests.synth) gets a quiet machine: the RAM guard sees 60% free and no memory pressure, so the results do not depend
on what else this Mac is running (engine/memory.py: LLM_TEST_FREE_RAM). `discover -s tests` imports the modules
without this package: tests/test__quiet_machine.py, loaded first, does the same there."""
import os

os.environ.setdefault("LLM_TEST_FREE_RAM", "0.6")
