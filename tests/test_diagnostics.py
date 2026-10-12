"""ui/diagnostics.py: the problems an editor underlines — built in, and via the language's own parser.

    .venv/bin/python -m unittest tests.test_diagnostics -v
"""
import json
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from ui import diagnostics as D  # noqa: E402


def lines(diags):
    return [(d["line"], d["source"]) for d in diags]


class TestBuiltIn(unittest.TestCase):
    def test_valid_code_has_no_problems(self):
        self.assertEqual(D.diagnose("def f(x):\n    return [x, {'a': (1, 2)}]\n", "python"), [])
        self.assertEqual(D.diagnose('{"a": [1, 2, {"b": null}]}', "json"), [])
        self.assertEqual(D.diagnose("fn main() {\n    let v = vec![1, 2];\n}\n", "rust"), [])
        self.assertEqual(D.diagnose("", "python"), [])
        self.assertEqual(D.diagnose("", "json"), [])

    def test_python_syntax_errors_carry_a_position(self):
        d = D.diagnose("x = 1\ndef f(:\n    pass\n", "python")
        self.assertEqual(lines(d), [(2, "python")])
        self.assertGreaterEqual(d[0]["col"], 1)
        self.assertEqual(d[0]["severity"], "error")
        self.assertIn("syntax", d[0]["message"].lower())

    def test_python_indentation_and_unterminated_string(self):
        self.assertEqual(lines(D.diagnose("def f():\nreturn 1\n", "python")), [(2, "python")])
        self.assertEqual(lines(D.diagnose("s = 'oops\nx = 2\n", "python")), [(1, "python")])

    def test_json_errors(self):
        d = D.diagnose('{\n  "a": 1,\n  "b": ,\n}', "json")
        self.assertEqual(lines(d), [(3, "json")])
        self.assertEqual(D.diagnose("{'single': 1}", "json")[0]["line"], 1)

    def test_toml_and_xml(self):
        self.assertEqual(D.diagnose("[a]\nkey = 1\n", "toml"), [])
        bad = D.diagnose("[a]\nkey = \n", "toml")
        self.assertEqual(lines(bad), [(2, "toml")])
        self.assertEqual(D.diagnose("<a><b></b></a>", "xml"), [])
        self.assertEqual(lines(D.diagnose("<a>\n<b></a>", "xml")), [(2, "xml")])

    def test_yaml_tabs(self):
        d = D.diagnose("a:\n\tb: 1\nc: 2\n", "yaml")
        self.assertEqual(lines(d), [(2, "yaml")])

    def test_brackets_unclosed_unexpected_and_mismatched(self):
        d = D.diagnose("function f() {\n  if (x) {\n    return 1;\n}\n", "javascript")
        self.assertEqual([(x["line"], x["message"]) for x in d], [(1, "'{' is never closed")])
        d = D.diagnose("int main() {\n  return 0;\n}\n}\n", "c")
        self.assertEqual([(x["line"], x["message"]) for x in d], [(4, "unexpected '}'")])
        d = D.diagnose("let a = (1, 2];\n", "swift")
        self.assertEqual(len(d), 1)
        self.assertIn("does not match", d[0]["message"])

    def test_brackets_in_strings_and_comments_are_ignored(self):
        src = 'let s = "}}}"  // ( [ {\n/* ) ] } */\nfunc f() {}\n'
        self.assertEqual(D.diagnose(src, "swift"), [])
        self.assertEqual(D.diagnose("a = '))' # (((\n", "python"), [])

    def test_many_problems_are_capped(self):
        d = D.diagnose(")" * 500, "javascript")
        self.assertLessEqual(len(d), 21)

    def test_never_raises_on_garbage(self):
        import random
        rng = random.Random(1)
        for lang in ("python", "json", "toml", "xml", "yaml", "javascript", "swift", "css", "nonsense"):
            for _ in range(20):
                junk = "".join(rng.choice("(){}[]'\"\\\n\t#/*<>:=a1") for _ in range(rng.randint(0, 120)))
                self.assertIsInstance(D.diagnose(junk, lang), list)

    def test_our_own_sources_are_clean(self):
        for rel in ("llm.py", "agent.py", "ui/server.py", "ui/codeinfo.py", "app/bar/Bar.swift", "app/editor/Chat.swift"):
            text = open(os.path.join(ROOT, rel), encoding="utf-8").read()
            lang = "swift" if rel.endswith(".swift") else "python"
            self.assertEqual(D.diagnose(text, lang), [], rel)


class TestDeepChecks(unittest.TestCase):
    @unittest.skipUnless(D.available("swift"), "swiftc is not installed")
    def test_swiftc_parses_the_buffer(self):
        self.assertEqual(D.deep_check("func f() -> Int { return 1 }\n", "swift"), [])
        d = D.deep_check("func f( {\n}\n", "swift")
        self.assertEqual(d[0]["line"], 1)
        self.assertEqual(d[0]["source"], "swiftc")
        self.assertNotIn("buffer.swift", d[0]["message"])

    @unittest.skipUnless(D.available("shell"), "bash is not installed")
    def test_bash_syntax(self):
        self.assertEqual(D.deep_check("echo hi\n", "shell"), [])
        d = D.deep_check("if true; then\n  echo hi\n", "shell")
        self.assertEqual(d[0]["source"], "bash")
        self.assertEqual(d[0]["line"], 3)

    @unittest.skipUnless(D.available("javascript"), "node is not installed")
    def test_node_syntax(self):
        self.assertEqual(D.deep_check("const a = 1;\n", "javascript"), [])
        d = D.deep_check("const = 5;\n", "javascript")
        self.assertEqual((d[0]["line"], d[0]["source"]), (1, "node"))
        self.assertIn("SyntaxError", d[0]["message"])

    def test_a_language_without_a_tool_gives_nothing(self):
        self.assertEqual(D.deep_check("anything", "markdown"), [])
        self.assertFalse(D.available("markdown"))


class TestEndpoints(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import tempfile
        import threading
        import urllib.request
        from ui.server import Project, UIServer
        cls.srv = UIServer(Project(tempfile.mkdtemp()), 0, 1, token="t")
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()
        cls.url = f"http://127.0.0.1:{cls.srv.server_address[1]}"
        cls.urlopen = staticmethod(urllib.request.urlopen)
        cls.Request = urllib.request.Request

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        cls.srv.server_close()

    def post(self, path, body):
        req = self.Request(self.url + path, method="POST", data=json.dumps(body).encode(),
                           headers={"X-UI-Token": "t", "Content-Type": "application/json"})
        with self.urlopen(req, timeout=30) as r:
            return json.load(r)

    def test_analyze_includes_the_builtin_diagnostics(self):
        r = self.post("/api/analyze", {"path": "a.py", "text": "def f(:\n  pass\n"})
        self.assertEqual(lines(r["diagnostics"]), [(1, "python")])
        self.assertEqual(self.post("/api/analyze", {"path": "a.py", "text": "x = 1\n"})["diagnostics"], [])

    def test_the_deep_endpoint_merges_tiers_without_duplicates(self):
        r = self.post("/api/diagnostics", {"path": "x.sh", "text": "if true; then\n echo\n"})
        self.assertEqual(r["language"], "shell")
        self.assertTrue(r["diagnostics"])
        self.assertEqual(len({(d["line"], d["message"]) for d in r["diagnostics"]}), len(r["diagnostics"]))
        quick = self.post("/api/diagnostics", {"path": "x.py", "text": "def f(:\n", "deep": False})
        self.assertEqual(lines(quick["diagnostics"]), [(1, "python")])


if __name__ == "__main__":
    unittest.main()
