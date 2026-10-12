"""`llm web search`: the JSON line the voice chat reads, with the network stubbed.

    .venv/bin/python -m unittest tests.test_web_cmd -v
"""
import contextlib
import io
import json
import os
import sys
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import llm  # noqa: E402
from harness import web  # noqa: E402

HITS = [{"title": "Lisbon weather", "url": "https://example.org/a", "snippet": "Sunny, 22 °C"},
        {"title": "Forecast", "url": "https://example.org/b", "snippet": "Light wind"},
        {"title": "Third", "url": "https://example.org/c", "snippet": "…"}]


def run(args):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ok = llm.web_cmd(args)
    return ok, json.loads(buf.getvalue())


class TestWebCmd(unittest.TestCase):
    def test_results_and_the_top_pages(self):
        with mock.patch.object(web, "search", return_value=HITS) as s, \
                mock.patch.object(web, "fetch", side_effect=lambda url, **k: f"text of {url}") as f:
            ok, out = run(["search", "weather", "in", "Lisbon", "--n", "3", "--read", "2"])
        self.assertTrue(ok)
        s.assert_called_once()
        self.assertEqual(s.call_args[0][0], "weather in Lisbon")
        self.assertEqual(s.call_args[1]["n"], 3)
        self.assertEqual(out["query"], "weather in Lisbon")
        self.assertEqual(out["results"], HITS)
        self.assertEqual([p["url"] for p in out["pages"]], ["https://example.org/a", "https://example.org/b"])
        self.assertEqual(f.call_count, 2)

    def test_a_page_that_cannot_be_read_is_just_missing(self):
        def fetch(url, **k):
            if url.endswith("/a"):
                raise ValueError("refusing to fetch a private/local address")
            return "ok"
        with mock.patch.object(web, "search", return_value=HITS), mock.patch.object(web, "fetch", side_effect=fetch):
            ok, out = run(["search", "x", "--read", "2"])
        self.assertTrue(ok)
        self.assertEqual([p["url"] for p in out["pages"]], ["https://example.org/b"])

    def test_offline_is_an_error_field_not_a_traceback(self):
        with mock.patch.object(web, "search", side_effect=OSError("[Errno 8] nodename nor servname provided")):
            ok, out = run(["search", "anything"])
        self.assertFalse(ok)
        self.assertEqual(out["results"], [])
        self.assertIn("nodename", out["error"])

    def test_usage_and_limits(self):
        ok, out = run([])
        self.assertFalse(ok)
        self.assertIn("usage", out["error"])
        with mock.patch.object(web, "search", return_value=[]) as s:
            run(["search", "q", "--n", "99", "--read", "9"])
            self.assertEqual(s.call_args[1]["n"], 8)               # capped
            ok, out = run(["search", "q", "--n", "x"])
            self.assertEqual(s.call_args[1]["n"], 4)               # not a number: the default
        self.assertEqual(out["pages"], [])

    def test_a_long_query_is_cut(self):
        with mock.patch.object(web, "search", return_value=[]) as s:
            run(["search", "w" * 1000])
        self.assertEqual(len(s.call_args[0][0]), 300)


if __name__ == "__main__":
    unittest.main()
