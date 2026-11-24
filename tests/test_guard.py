"""harness/guard.py: noticing a reply or a tool call that loops, without cutting off real output.

    .venv/bin/python -m unittest tests.test_guard -v
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from harness import guard  # noqa: E402


class TestLooping(unittest.TestCase):
    def test_one_line_over_and_over(self):
        self.assertTrue(guard.looping("Intro.\n" + "Let me look at the file again now.\n" * 30))

    def test_a_block_repeated_without_newlines(self):
        block = "The answer is that we should check the settings and then restart the server process. " * 3
        self.assertTrue(guard.looping("Start. " + block * 8))

    def test_ordinary_text_and_tables_pass(self):
        prose = " ".join(f"Sentence number {i} says something new about item {i * 7}." for i in range(200))
        self.assertFalse(guard.looping(prose))
        table = "| file | lines |\n|---|---|\n" + "".join(f"| src/mod{i}.py | {i * 13} |\n" for i in range(150))
        self.assertFalse(guard.looping(table))
        self.assertFalse(guard.looping("short"))

    def test_short_repeated_markup_is_not_a_loop(self):
        self.assertFalse(guard.looping("Done.\n" + "---\n" * 100 + "Next part of a real answer here."))


class TestSameCalls(unittest.TestCase):
    def test_third_identical_call_in_a_row(self):
        k = guard.call_key("read_file", {"path": "a.py"})
        self.assertFalse(guard.same_calls([], k))
        self.assertFalse(guard.same_calls([k], k))
        self.assertTrue(guard.same_calls([k, k], k))

    def test_argument_order_does_not_matter_and_a_change_resets(self):
        a = guard.call_key("grep", {"pattern": "x", "path": "."})
        b = guard.call_key("grep", {"path": ".", "pattern": "x"})
        self.assertEqual(a, b)
        other = guard.call_key("grep", {"pattern": "y", "path": "."})
        self.assertFalse(guard.same_calls([a, other], a))


if __name__ == "__main__":
    unittest.main()
