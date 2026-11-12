"""Pictures in words for a model without vision (assistant/describe.py), and how they reach the prompt."""
import base64
import os
import struct
import sys
import unittest
import zlib
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from assistant import describe, prompts  # noqa: E402


def png(w=64, h=48):
    raw = b"".join(b"\x00" + b"\xff\xff\xff" * w for _ in range(h))
    chunk = lambda t, d: struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xffffffff)
    data = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) + \
        chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")
    return "data:image/png;base64," + base64.b64encode(data).decode()


class TestDescribe(unittest.TestCase):
    def test_without_the_helper_every_picture_gets_an_empty_description(self):
        with mock.patch.object(describe, "ocr_tool", lambda: None):
            self.assertEqual(describe.describe([png(), png()]), [{"text": "", "labels": []}] * 2)

    @unittest.skipUnless(describe.ocr_tool(), "the OCR helper is not built (./llm setup --ocr): Apple Vision was not exercised")
    def test_the_real_helper_answers_one_description_per_picture(self):
        out = describe.describe([png(), png(80, 60)])
        self.assertEqual(len(out), 2)
        for d in out:
            self.assertIsInstance(d["labels"], list)
            self.assertIsInstance(d["text"], str)

    def test_the_prompt_names_each_picture_and_says_it_is_a_description(self):
        words = describe.as_text([{"text": "COPILOT CLI", "labels": ["screenshot", "text"]},
                                  {"text": "", "labels": []}], True)
        self.assertIn("You cannot see pictures", words)
        self.assertIn("The screenshot of the page — shows: screenshot, text; text in it:\nCOPILOT CLI", words)
        self.assertIn("Picture 1 from the page — nothing recognisable", words)
        msgs, shared = prompts.build({"task": "ask", "question": "what is on the image?", "page": {"text": "x"},
                                      "described": words}, 8192, 512)
        self.assertIsInstance(msgs[1]["content"], str)                 # words, not image parts
        self.assertIn("COPILOT CLI", msgs[1]["content"])
        self.assertTrue(shared["described"])


if __name__ == "__main__":
    unittest.main()
