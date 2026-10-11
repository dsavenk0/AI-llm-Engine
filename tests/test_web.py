"""harness.web: search parsing, page → text, and the guards (no internet needed).

    .venv/bin/python -m unittest tests.test_web -v
"""
import json
import os
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from harness import web  # noqa: E402

DDG = """<html><body>
<div class="result"><h2><a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fdocs.example.org%2Fapi%2Fv2&amp;rut=abc">
  Example <b>API</b> v2 reference</a></h2>
  <a class="result__snippet" href="//duckduckgo.com/l/?uddg=x">Changed in v2: <b>auth</b> now uses tokens.</a></div>
<div class="result"><h2><a class="result__a" href="https://other.example.com/page">Other page</a></h2>
  <a class="result__snippet" href="https://other.example.com/page">A second result.</a></div>
<div class="result"><h2><a class="result__a" href="//duckduckgo.com/y.js?ad_provider=x">Sponsored</a></h2></div>
</body></html>"""

PAGE = """<!doctype html><html><head><title>Guide &amp; Notes</title><meta charset="utf-8">
<style>.x{color:red}</style><script>var tracking = 1;</script></head>
<body><nav><a href="/">Home</a><a href="/docs">Docs</a></nav>
<div class="sidebar"><ul><li>Sidebar link</li></ul></div>
<main><h1>Install</h1><p>Run <code>pip install foo</code> first.</p>
<ul><li>step one</li><li>step two</li></ul>
<pre>def f():
    return 1</pre>
<div role="navigation"><a href="#">Next page</a></div>
<p>Done.</p></main><footer>Copyright junk</footer></body></html>"""


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        routes = {
            "/page": ("text/html; charset=utf-8", PAGE.encode()),
            "/plain": ("text/plain", b"just text"),
            "/bin": ("application/octet-stream", b"\x00\x01"),
            "/big": ("text/plain", b"x" * (web.MAX_BYTES + 5000)),
            "/json": ("application/json", json.dumps({"a": 1}).encode()),
        }
        if self.path == "/redirect-private":
            self.send_response(302)
            self.send_header("Location", "http://127.0.0.1:1/secret")
            self.end_headers()
            return
        if self.path not in routes:
            self.send_response(404)
            self.end_headers()
            return
        ctype, body = routes[self.path]
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class TestWeb(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        cls.base = f"http://127.0.0.1:{cls.srv.server_address[1]}"
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        cls.srv.server_close()

    def test_search_results_are_parsed_and_unwrapped(self):
        r = web.parse_results(DDG)
        self.assertEqual([x["url"] for x in r], ["https://docs.example.org/api/v2", "https://other.example.com/page"])
        self.assertEqual(r[0]["title"], "Example API v2 reference")
        self.assertEqual(r[0]["snippet"], "Changed in v2: auth now uses tokens.")
        self.assertIn("1. Example API v2 reference", web.format_results(r))
        self.assertEqual(web.format_results([]), "(no results)")

    def test_page_becomes_clean_text(self):
        out = web.fetch(self.base + "/page", allow_private=True)
        self.assertTrue(out.startswith("Guide & Notes\n" + self.base + "/page"))
        for kept in ("# Install", "`pip install foo`", "- step one", "```\ndef f():\n    return 1\n```", "Done."):
            self.assertIn(kept, out)
        for dropped in ("tracking", "color:red", "Home", "Sidebar link", "Next page", "Copyright junk"):
            self.assertNotIn(dropped, out)

    def test_plain_and_json_pass_through(self):
        self.assertTrue(web.fetch(self.base + "/plain", allow_private=True).endswith("just text"))
        self.assertIn('"a": 1', web.fetch(self.base + "/json", allow_private=True))

    def test_binary_and_missing_pages_report_errors(self):
        self.assertIn("not a text page", web.fetch(self.base + "/bin", allow_private=True))
        self.assertIn("HTTP 404", web.fetch(self.base + "/nope", allow_private=True))

    def test_long_text_is_cut_with_a_note(self):
        out = web.fetch(self.base + "/big", max_chars=1000, allow_private=True)
        self.assertIn("more characters", out)
        self.assertLess(len(out), 1400)

    def test_private_addresses_are_refused_by_default(self):
        self.assertIn("private/local address", web.fetch(self.base + "/page"))
        self.assertIn("private/local address", web.fetch("http://169.254.169.254/latest/meta-data/"))
        self.assertIn("only http", web.fetch("file:///etc/passwd"))

    def test_only_global_addresses_pass_including_masked_and_shared_ones(self):
        import ipaddress
        for bad in ("127.0.0.1", "10.1.2.3", "192.168.0.9", "172.16.5.5", "169.254.169.254", "100.64.1.1",   # CGNAT
                    "0.0.0.0", "224.0.0.1", "::1", "fe80::1", "fc00::1", "::ffff:127.0.0.1", "::ffff:10.0.0.5",
                    "::"):
            with self.assertRaises(ValueError, msg=bad):
                web._require_public(ipaddress.ip_address(bad), "h")
        for good in ("93.184.216.34", "8.8.8.8", "2606:4700:4700::1111", "::ffff:8.8.8.8"):
            web._require_public(ipaddress.ip_address(good), "h")

    def test_a_name_that_resolves_to_a_private_address_is_refused(self):
        from unittest import mock
        fake = lambda host, port, *a, **k: [(2, 1, 6, "", ("::ffff:127.0.0.1", port, 0, 0))]
        with mock.patch.object(web.socket, "getaddrinfo", fake):
            with self.assertRaises(ValueError):
                web.check_url("http://rebind.example/x")

    def test_a_redirect_into_the_lan_is_refused(self):
        # the first hop is allowed (allow_private), the redirect target is checked by the same rule
        class Strict(web._Redirects):
            allow_private = False
        out = web.fetch(self.base + "/redirect-private", allow_private=False)
        self.assertIn("private/local address", out)  # even the first hop is local here
        with self.assertRaises(ValueError):
            Strict().redirect_request(web.urllib.request.Request(self.base + "/x"), None, 302, "Found", {},
                                      "http://10.0.0.5/admin")

    def test_tools_ask_before_going_out_and_respect_no(self):
        asked = []
        t = web.tools(lambda what: asked.append(what) or False)
        self.assertEqual(set(t), {"web_search", "web_fetch"})
        self.assertIn("declined", t["web_search"][1](query="anything"))
        self.assertIn("declined", t["web_fetch"][1](url=self.base + "/page"))
        self.assertEqual(len(asked), 2)
        schema = t["web_fetch"][0]["function"]
        self.assertEqual(schema["name"], "web_fetch")
        self.assertEqual(schema["parameters"]["required"], ["url"])

    def test_tools_fetch_when_allowed(self):
        t = web.tools(lambda what: True, allow_private=True)
        self.assertIn("# Install", t["web_fetch"][1](url=self.base + "/page"))


class TestAgentWeb(unittest.TestCase):
    def agent(self):
        import tempfile
        import agent
        import llm
        cfg = llm.load_cfg()
        cfg["port"] = 1
        return agent.Agent(cfg, tempfile.mkdtemp())

    def test_web_tools_are_opt_in_and_one_question_per_session(self):
        a = self.agent()
        names = lambda: {t["function"]["name"] for t in a.tool_schemas()}
        self.assertNotIn("web_search", names())          # nothing extra in the prompt by default
        a.set_web(True)
        self.assertTrue({"web_search", "web_fetch"} <= names())
        a.auto = True                                     # /auto answers the one question
        self.assertTrue(a.allow_web("fetch x"))
        a.set_web(False)
        self.assertNotIn("web_search", names())
        self.assertIn("unknown tool", a.call_tool("web_search", {"query": "x"}))


if __name__ == "__main__":
    unittest.main()
