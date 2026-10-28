"""Document text extraction (engine/documents.py) — stdlib-crafted files, no external samples or libs.

    .venv/bin/python -m unittest tests.test_documents -v
"""
import os
import sys
import tempfile
import unittest
import zipfile
import zlib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from engine import documents as D  # noqa: E402


def write(name, data, d):
    p = os.path.join(d, name)
    with open(p, "wb" if isinstance(data, bytes) else "w") as f:
        f.write(data)
    return p


class TestDocuments(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()

    def test_plain_text_and_markdown(self):
        self.assertEqual(D.extract_text(write("a.md", "# T\n\nhello", self.d))["text"], "# T\n\nhello")
        self.assertEqual(D.extract_text(write("a.txt", "plain", self.d))["kind"], "text")

    def test_empty_file(self):
        r = D.extract_text(write("empty.txt", "", self.d))
        self.assertEqual(r["text"], "")

    def _docx(self, body):
        p = os.path.join(self.d, "doc.docx")
        with zipfile.ZipFile(p, "w") as z:
            z.writestr("word/document.xml", "<w:document><w:body>" + body + "</w:body></w:document>")
        return p

    def test_docx_joins_runs_and_keeps_paragraphs(self):
        r = D.extract_text(self._docx("<w:p><w:r><w:t>Hello</w:t></w:r><w:r><w:t> World</w:t></w:r></w:p>"
                                      "<w:p><w:r><w:t>Second</w:t></w:r></w:p>"))
        self.assertEqual(r["text"], "Hello World\nSecond")
        self.assertEqual(r["kind"], "docx")

    def test_docx_unescapes_entities(self):
        r = D.extract_text(self._docx("<w:p><w:r><w:t>a &amp; b &lt; c</w:t></w:r></w:p>"))
        self.assertEqual(r["text"], "a & b < c")

    def test_pptx_reads_every_slide(self):
        p = os.path.join(self.d, "deck.pptx")
        with zipfile.ZipFile(p, "w") as z:
            z.writestr("ppt/slides/slide1.xml", "<p><a:p><a:r><a:t>One</a:t></a:r></a:p></p>")
            z.writestr("ppt/slides/slide2.xml", "<p><a:p><a:r><a:t>Two</a:t></a:r></a:p></p>")
        self.assertIn("One", D.extract_text(p)["text"])
        self.assertIn("Two", D.extract_text(p)["text"])

    def test_bad_office_file(self):
        with self.assertRaises(D.DocError):
            D.extract_text(write("x.docx", b"not a zip", self.d))

    def test_html_strips_tags_scripts_and_unescapes(self):
        p = write("a.html", "<html><body><h1>T</h1><p>Hi &amp; bye</p><script>x=1</script></body></html>", self.d)
        text = D.extract_text(p)["text"]
        self.assertIn("Hi & bye", text)
        self.assertNotIn("x=1", text)
        self.assertNotIn("<", text)

    def test_pdf_uncompressed_stream(self):
        content = b"BT /F1 12 Tf (Hello PDF) Tj ET"
        pdf = b"%PDF-1.4\n1 0 obj<</Type/Page>>endobj\n2 0 obj<</Length 30>>stream\n" + content + b"\nendstream endobj\n"
        r = D.extract_text(write("a.pdf", pdf, self.d))
        self.assertEqual(r["text"], "Hello PDF")
        self.assertEqual(r["pages"], 1)

    def test_pdf_flate_compressed_stream(self):
        content = b"BT (Compressed text here) Tj ET"
        pdf = b"%PDF-1.5\n/Type /Page\nstream\n" + zlib.compress(content) + b"\nendstream"
        r = D.extract_text(write("c.pdf", pdf, self.d))
        self.assertIn("Compressed text here", r["text"])

    def test_pdf_escapes(self):
        pdf = b"%PDF-1.4\nstream\nBT (A\\(B\\) \\\\ C\\n) Tj ET\nendstream"
        self.assertEqual(D.extract_text(write("e.pdf", pdf, self.d))["text"].strip(), "A(B) \\ C")

    def test_scanned_pdf_reports_no_text_layer(self):
        r = D.extract_text(write("s.pdf", b"%PDF-1.4\n/Type /Page\n", self.d))
        self.assertEqual(r["text"], "")
        self.assertIn("scanned", r["note"])

    def test_pdf_detected_by_magic_without_extension(self):
        pdf = b"%PDF-1.4\nstream\nBT (By magic) Tj ET\nendstream"
        self.assertEqual(D.extract_text(write("noext", pdf, self.d))["kind"], "pdf")

    def test_binary_is_refused(self):
        with self.assertRaises(D.DocError):
            D.extract_text(write("x.bin", b"\x00\x01\x02binary", self.d))

    def test_missing_file(self):
        with self.assertRaises(D.DocError):
            D.extract_text(os.path.join(self.d, "nope.txt"))

    def _fake_ocr(self, says="FAKE OCR"):
        p = os.path.join(self.d, "ocr")
        with open(p, "w") as f:
            f.write('#!/bin/sh\nprintf "%s"\n' % says)
        os.chmod(p, 0o755)
        return p

    def test_image_is_ocr_d_when_the_helper_is_present(self):
        png = write("pic.png", b"\x89PNG\r\n\x1a\n" + b"\x00" * 20, self.d)   # PNG magic, body irrelevant to the fake
        os.environ["LLM_OCR"] = self._fake_ocr("Recognised text")
        try:
            r = D.extract_text(png)
        finally:
            del os.environ["LLM_OCR"]
        self.assertEqual(r["text"], "Recognised text")
        self.assertEqual(r["kind"], "image")

    def test_scanned_pdf_falls_back_to_ocr(self):
        scan = write("s.pdf", b"%PDF-1.4\n/Type /Page\n", self.d)             # no text layer
        os.environ["LLM_OCR"] = self._fake_ocr("From the scan")
        try:
            r = D.extract_text(scan)
        finally:
            del os.environ["LLM_OCR"]
        self.assertEqual(r["kind"], "pdf-ocr")
        self.assertEqual(r["text"], "From the scan")

    def test_image_without_helper_is_a_clear_error(self):
        from unittest import mock
        png = write("pic.png", b"\x89PNG\r\n\x1a\n" + b"\x00" * 20, self.d)
        with mock.patch.object(D, "ocr_tool", lambda: None):
            with self.assertRaises(D.DocError):
                D.extract_text(png)

    def test_scanned_pdf_without_helper_keeps_the_note(self):
        from unittest import mock
        scan = write("s.pdf", b"%PDF-1.4\n/Type /Page\n", self.d)
        with mock.patch.object(D, "ocr_tool", lambda: None):
            r = D.extract_text(scan)
        self.assertEqual(r["text"], "")
        self.assertIn("scanned", r["note"])

    def test_excerpts_pick_the_relevant_passage(self):
        doc = ("Intro about gardens.\n\n" + "A section on cooking pasta with tomato sauce. " * 40 + "\n\n"
               + "A section on quantum entanglement and qubits. " * 40 + "\n\n" + "Closing weather notes.")
        ex = D.select_excerpts(doc, "how do qubits entangle", budget=2000, size=400)
        self.assertGreater(ex["selected"], 0)
        self.assertLess(ex["selected"], ex["total"])        # not everything — a selection
        self.assertIn("qubit", ex["text"].lower())
        self.assertNotIn("pasta", ex["text"].lower())
        self.assertTrue(all("start" in p and "end" in p for p in ex["passages"]))   # offsets for citations

    def test_excerpts_fall_back_to_head_without_a_query(self):
        doc = "x" * 500
        self.assertEqual(D.select_excerpts(doc, "", budget=50)["text"], doc[:50])
        self.assertEqual(D.select_excerpts(doc, "", budget=50)["selected"], 0)

    def test_excerpts_fall_back_when_nothing_matches(self):
        doc = "alpha beta gamma. " * 50
        ex = D.select_excerpts(doc, "zzzznomatch", budget=100)
        self.assertEqual(ex["selected"], 0)
        self.assertEqual(ex["text"], doc[:100])

    def test_excerpts_respect_the_budget(self):
        doc = ("topic one here. " * 30 + "\n\n") * 10 + "the special keyword appears here once"
        ex = D.select_excerpts(doc, "special keyword", budget=300, size=200)
        self.assertLessEqual(len(ex["text"]), 400)          # budget plus a little joining slack

    def test_truncation_flag(self):
        orig = D.MAX_CHARS
        try:
            D.MAX_CHARS = 10
            r = D.extract_text(write("long.txt", "x" * 100, self.d))
            self.assertTrue(r["truncated"])
            self.assertEqual(len(r["text"]), 10)
        finally:
            D.MAX_CHARS = orig


if __name__ == "__main__":
    unittest.main()
