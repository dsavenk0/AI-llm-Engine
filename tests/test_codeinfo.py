"""ui/codeinfo.py: language detection, syntax spans and outlines — what an editor shows for a file.

    .venv/bin/python -m unittest tests.test_codeinfo -v
"""
import os
import random
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from ui import codeinfo as C  # noqa: E402


def names(symbols):
    return [(s["kind"], s["name"]) for s in symbols]


def find(symbols, name):
    for d, s in C.flatten(symbols):
        if s["name"] == name:
            return s
    raise AssertionError(f"{name} not in outline: {[s['name'] for _, s in C.flatten(symbols)]}")


def kinds_at(text, lang):
    return [(text[a:b], k) for a, b, k in C.highlight(text, lang)]


class TestDetect(unittest.TestCase):
    def test_extensions_and_special_names(self):
        cases = {"a.py": "python", "x/y/App.swift": "swift", "m.tsx": "typescript", "q.mjs": "javascript",
                 "main.go": "go", "lib.rs": "rust", "Foo.java": "java", "x.kt": "kotlin", "a.cpp": "cpp",
                 "style.css": "css", "data.json": "json", "c.yml": "yaml", "pyproject.toml": "toml",
                 "README.md": "markdown", "Dockerfile": "dockerfile", "Makefile": "makefile",
                 "run.command": "shell", "q.sql": "sql", "index.html": "html", "notes.txt": "plaintext"}
        for path, want in cases.items():
            with self.subTest(path):
                self.assertEqual(C.detect_language(path), want)

    def test_shebang_decides_when_there_is_no_extension(self):
        self.assertEqual(C.detect_language("tool", "#!/usr/bin/env python3\nprint(1)\n"), "python")
        self.assertEqual(C.detect_language("tool", "#!/bin/bash\necho hi\n"), "shell")
        self.assertEqual(C.detect_language("tool", "#!/usr/bin/env node\nconsole.log(1)\n"), "javascript")

    def test_content_sniffing_for_unnamed_buffers(self):
        self.assertEqual(C.detect_language("", '{"a": [1, 2], "b": null}'), "json")
        self.assertEqual(C.detect_language("", "<!DOCTYPE html>\n<html><body></body></html>"), "html")
        self.assertEqual(C.detect_language("", "import os\n\ndef main():\n    pass\n"), "python")
        self.assertEqual(C.detect_language("", "package main\n\nfunc main() {\n}\n"), "go")
        self.assertEqual(C.detect_language("", "import SwiftUI\n\nstruct V: View {\n    var body: some View { Text(\"x\") }\n}\n"), "swift")
        self.assertEqual(C.detect_language("", "# Title\n\n- one\n- two\n"), "markdown")
        self.assertEqual(C.detect_language("", "just some words\nand more words\n"), "plaintext")
        self.assertEqual(C.detect_language("", ""), "plaintext")

    def test_the_extension_beats_the_content(self):
        self.assertEqual(C.detect_language("notes.txt", "def f():\n    pass\n"), "plaintext")


class TestHighlight(unittest.TestCase):
    def test_python_basics(self):
        got = dict((t, k) for t, k in kinds_at("def run(x):  # go\n    return 'a' + str(42)\n", "python"))
        self.assertEqual(got["def"], "keyword")
        self.assertEqual(got["run"], "function")
        self.assertEqual(got["# go"], "comment")
        self.assertEqual(got["'a'"], "string")
        self.assertEqual(got["42"], "number")
        self.assertEqual(got["str"], "function")

    def test_keywords_inside_strings_and_comments_are_not_keywords(self):
        spans = kinds_at('s = "if else return"  # def class\n', "python")
        self.assertFalse([t for t, k in spans if k == "keyword"])
        self.assertIn(('"if else return"', "string"), spans)

    def test_python_triple_quotes_and_escapes(self):
        text = 'x = """a "quoted" line\nsecond"""\ny = "esc \\" still"\n'
        spans = kinds_at(text, "python")
        self.assertIn(('"""a "quoted" line\nsecond"""', "string"), spans)
        self.assertIn(('"esc \\" still"', "string"), spans)

    def test_decorators_and_types_and_constants(self):
        spans = dict(kinds_at("@dataclass\nclass Point:\n    MAX_SIZE = 3\n", "python"))
        self.assertEqual(spans["@dataclass"], "decorator")
        self.assertEqual(spans["Point"], "type")
        self.assertEqual(spans["MAX_SIZE"], "constant")

    def test_c_family_comments_preprocessor_and_unterminated(self):
        text = '#include <stdio.h>\n/* block\ncomment */ int main() { return 0; } // end\nchar *s = "oops\n'
        spans = kinds_at(text, "c")
        self.assertIn(("#include <stdio.h>", "decorator"), spans)
        self.assertIn(("/* block\ncomment */", "comment"), spans)
        self.assertIn(("// end", "comment"), spans)
        self.assertEqual([k for t, k in spans if t == "main"], ["function"])
        self.assertTrue(any(k == "string" and t.startswith('"oops') for t, k in spans))   # stops at the line end

    def test_js_template_literals_span_lines(self):
        spans = kinds_at("const a = `one\ntwo ${x}`;\nlet b = 1;", "javascript")
        self.assertIn(("`one\ntwo ${x}`", "string"), spans)
        self.assertIn(("const", "keyword"), spans)

    def test_json_keys_are_properties_and_values_are_not(self):
        spans = kinds_at('{"name": "x", "n": 3, "ok": true}', "json")
        self.assertIn(('"name"', "property"), spans)
        self.assertIn(('"x"', "string"), spans)
        self.assertIn(("true", "keyword"), spans)

    def test_yaml_keys(self):
        spans = kinds_at("name: demo  # c\nitems:\n  - a: 1\n", "yaml")
        self.assertIn(("name", "property"), spans)
        self.assertIn(("# c", "comment"), spans)
        self.assertIn(("a", "property"), spans)

    def test_markdown_and_markup(self):
        md = kinds_at("# Title\n\nuse `code` and [link](http://x)\n```py\nx = 1\n```\n", "markdown")
        self.assertIn(("# Title", "heading"), md)
        self.assertIn(("`code`", "string"), md)
        self.assertIn(("[link](http://x)", "function"), md)
        self.assertIn(("x = 1", "string"), md)                    # inside a fence
        html = kinds_at('<div class="a"><!-- c --></div>', "html")
        self.assertIn(("div", "tag"), html)
        self.assertIn(('"a"', "string"), html)
        self.assertIn(("<!-- c -->", "comment"), html)

    def test_spans_are_ordered_non_overlapping_and_in_bounds(self):
        text = open(os.path.join(ROOT, "agent.py")).read()
        spans = C.highlight(text, "python")
        last = 0
        for a, b, k in spans:
            self.assertTrue(last <= a < b <= len(text), (a, b, k))
            last = b

    def test_unknown_language_and_garbage_never_raise(self):
        self.assertEqual(C.highlight("anything", "klingon"), [])
        rng = random.Random(3)
        for lang in list(C.LANGS) + ["markdown", "html"]:
            for _ in range(15):
                junk = "".join(rng.choice("abc 123\n\"'`/*#{}()[]<>\\=:;@") for _ in range(rng.randint(0, 200)))
                C.highlight(junk, lang)
                C.outline(junk, lang)


class TestOutline(unittest.TestCase):
    def test_python_nesting_kinds_and_ranges(self):
        src = ("import os\nMAX = 3\n\nclass A:\n    '''doc'''\n    def m(self):\n        def inner():\n            pass\n"
               "        return 1\n    async def n(self): ...\n\ndef top():\n    pass\n")
        o = C.outline(src, "python")
        self.assertEqual(names(o), [("constant", "MAX"), ("class", "A"), ("function", "top")])
        a = o[1]
        self.assertEqual((a["line"], a["end"]), (4, 10))
        self.assertEqual(names(a["children"]), [("method", "m"), ("method", "n")])
        self.assertEqual(names(a["children"][0]["children"]), [("function", "inner")])

    def test_python_with_a_syntax_error_still_gets_an_outline(self):
        o = C.outline("class A:\n    def ok(self):\n        return (\n\ndef later():\n    pass\n", "python")
        flat = [s["name"] for _, s in C.flatten(o)]
        self.assertIn("A", flat)
        self.assertIn("later", flat)

    def test_python_conditional_definitions_are_found(self):
        o = C.outline("import sys\nif sys.platform == 'darwin':\n    def mac(): pass\nelse:\n    def other(): pass\n", "python")
        self.assertEqual(names(o), [("function", "mac"), ("function", "other")])

    def test_markdown_hierarchy_and_fences(self):
        md = "# One\n\n## A\n```\n# not a heading\n```\n## B\n### B1\n# Two\n"
        o = C.outline(md, "markdown")
        self.assertEqual([s["name"] for s in o], ["One", "Two"])
        self.assertEqual([s["name"] for s in o[0]["children"]], ["A", "B"])
        self.assertEqual(o[0]["children"][1]["children"][0]["name"], "B1")
        self.assertEqual(o[0]["end"], 8)

    def test_json_top_level_keys_with_lines(self):
        o = C.outline('{\n  "name": "x",\n  "deps": {\n    "a": 1\n  },\n  "list": [1, 2]\n}\n', "json")
        self.assertEqual(names(o), [("key", "name"), ("object", "deps"), ("array", "list")])
        self.assertEqual([s["line"] for s in o], [2, 3, 6])
        self.assertEqual(C.outline("not json", "json"), [])

    def test_yaml_and_toml_and_ini(self):
        y = C.outline("services:\n  web:\n    image: x\n  db:\n    image: y\nversion: 3\n", "yaml")
        self.assertEqual([s["name"] for s in y], ["services", "version"])
        self.assertEqual([s["name"] for s in y[0]["children"]], ["web", "db"])
        t = C.outline("[package]\nname = 'x'\n[[bin]]\nname = 'y'\n", "toml")
        self.assertEqual([s["name"] for s in t], ["package", "bin"])

    def test_html_headings_ids_and_blocks(self):
        o = C.outline('<h1>Title</h1>\n<div id="app"></div>\n<script src="a.js"></script>\n<h2><b>Sub</b></h2>', "html")
        self.assertEqual(names(o), [("heading", "Title"), ("id", "#app"), ("block", "script"), ("heading", "Sub")])

    def test_typescript_classes_methods_arrows_and_types(self):
        src = ("export interface Opts { a: number }\ntype Id = string;\nexport class Svc extends Base {\n"
               "  private cache = new Map();\n  async load(id: Id): Promise<void> {\n    if (id) { return; }\n  }\n"
               "  static of(): Svc { return new Svc(); }\n}\nexport const run = async (x: number) => {\n  return x;\n};\n"
               "function plain() {}\nenum Color { Red, Green }\n")
        o = C.outline(src, "typescript")
        self.assertEqual([s["name"] for s in o], ["Opts", "Id", "Svc", "run", "plain", "Color"])
        svc = find(o, "Svc")
        self.assertEqual(names(svc["children"]), [("method", "load"), ("method", "of")])   # `if` is not a method
        self.assertEqual((svc["line"], svc["end"]), (3, 9))

    def test_javascript_braces_in_strings_and_comments_do_not_break_nesting(self):
        src = ("class A {\n  m() {\n    const s = '}}}';  // }\n    /* { */\n    return 1;\n  }\n  n() {}\n}\nfunction after() {}\n")
        o = C.outline(src, "javascript")
        self.assertEqual([s["name"] for s in o], ["A", "after"])
        self.assertEqual([s["name"] for s in o[0]["children"]], ["m", "n"])

    def test_swift_types_extensions_and_funcs(self):
        src = ("import SwiftUI\n\nstruct Card: View {\n    var body: some View { Text(\"x\") }\n    func tap() {\n    }\n}\n"
               "extension Card {\n    private func helper() {}\n}\nfinal class Model: ObservableObject {\n"
               "    init() {}\n    @MainActor func load() async {}\n}\nfunc free() {}\nprotocol P { func req() }\n")
        o = C.outline(src, "swift")
        self.assertEqual([(s["kind"], s["name"]) for s in o],
                         [("struct", "Card"), ("extension", "Card"), ("class", "Model"), ("function", "free"), ("protocol", "P")])
        self.assertEqual([s["name"] for s in o[0]["children"]], ["tap"])
        self.assertEqual([s["name"] for s in o[2]["children"]], ["init", "load"])

    def test_go_receivers_and_types(self):
        src = "package x\n\ntype S struct {\n  a int\n}\n\nfunc (s *S) Run() error {\n  return nil\n}\n\nfunc New() *S { return &S{} }\n"
        o = C.outline(src, "go")
        self.assertEqual(names(o), [("type", "S"), ("method", "Run"), ("function", "New")])

    def test_rust_impl_and_fns(self):
        src = "pub struct P { x: i32 }\nimpl P {\n    pub fn new() -> Self { P { x: 0 } }\n    fn get(&self) -> i32 { self.x }\n}\nfn main() {}\n"
        o = C.outline(src, "rust")
        self.assertEqual([s["name"] for s in o], ["P", "P", "main"])
        self.assertEqual([s["name"] for s in o[1]["children"]], ["new", "get"])

    def test_java_class_and_methods(self):
        src = "package a;\npublic class Svc {\n    private int x;\n    public static void main(String[] args) {\n    }\n    int get() { return x; }\n}\n"
        o = C.outline(src, "java")
        self.assertEqual(names(o), [("class", "Svc")])
        self.assertEqual([s["name"] for s in o[0]["children"]], ["main", "get"])

    def test_c_functions_and_structs(self):
        src = "#include <stdio.h>\nstruct node {\n  int v;\n};\nstatic int add(int a, int b) {\n  return a + b;\n}\nint main(void) {\n  if (1) { }\n  return 0;\n}\n"
        o = C.outline(src, "c")
        self.assertEqual([s["name"] for s in o], ["node", "add", "main"])

    def test_shell_ruby_makefile_dockerfile_sql(self):
        self.assertEqual([s["name"] for s in C.outline("build() {\n  echo\n}\nfunction clean {\n}\n", "shell")], ["build", "clean"])
        rb = C.outline("module M\n  class A\n    def go\n    end\n  end\nend\n", "ruby")
        self.assertEqual(names(rb), [("class", "M")])
        self.assertEqual(names(rb[0]["children"][0]["children"]), [("method", "go")])
        self.assertEqual([s["name"] for s in C.outline("all: build\n\nbuild:\n\techo\nVAR := 1\n", "makefile")], ["all", "build"])
        self.assertEqual([s["name"] for s in C.outline("FROM python:3 AS base\nRUN x\nFROM base AS final\n", "dockerfile")], ["base", "final"])
        sql = C.outline("create table users (id int);\nCREATE OR REPLACE VIEW v AS select 1;\n", "sql")
        self.assertEqual([s["name"] for s in sql], ["users", "v"])

    def test_flatten_gives_document_order_with_depth(self):
        o = C.outline("class A:\n    def m(self): pass\n\ndef f(): pass\n", "python")
        self.assertEqual([(d, s["name"]) for d, s in C.flatten(o)], [(0, "A"), (1, "m"), (0, "f")])


class TestPerformance(unittest.TestCase):
    """Editors must not stall on big files: these were minutes (JSON) before the keys pass was made linear."""

    def timed(self, fn, *a):
        import time
        t = time.perf_counter()
        r = fn(*a)
        return r, time.perf_counter() - t

    def test_large_json_highlights_in_linear_time(self):
        import json
        text = json.dumps([{"id": i, "name": f"item {i}", "tags": ["a", "b"]} for i in range(6000)], indent=1)
        spans, secs = self.timed(C.highlight, text, "json")
        self.assertLess(secs, 3.0, f"{len(text)} chars took {secs:.1f}s")
        props = [(a, b) for a, b, k in spans if k == "property"]
        self.assertEqual(len(props), 18000)
        self.assertTrue(all(text[a:b].startswith('"') and text[a:b].endswith('"') for a, b in props[:200]))

    def test_large_yaml_and_toml(self):
        yaml = "\n".join(f"key{i}: value {i}  # c" for i in range(20000))
        _, secs = self.timed(C.highlight, yaml, "yaml")
        self.assertLess(secs, 3.0)
        toml = "\n".join(f"k{i} = {i}" for i in range(20000))
        _, secs = self.timed(C.highlight, toml, "toml")
        self.assertLess(secs, 3.0)

    def test_comment_lines_are_not_keys(self):
        spans = kinds_at("# a: b\nreal: 1\n", "yaml")
        self.assertEqual([t for t, k in spans if k == "property"], ["real"])

    def test_a_huge_buffer_is_left_unhighlighted(self):
        big = "x = 1\n" * 400_000
        self.assertGreater(len(big), C.MAX_HIGHLIGHT)
        self.assertEqual(C.highlight(big, "python"), [])

    def test_code_files_of_a_megabyte_stay_interactive(self):
        src = open(os.path.join(ROOT, "llm.py")).read() * 12              # ~700 KB
        _, th = self.timed(C.highlight, src, "python")
        _, to = self.timed(C.outline, src, "python")
        self.assertLess(th, 2.0)
        self.assertLess(to, 2.0)


class TestOnTheRepository(unittest.TestCase):
    def test_every_source_file_here_is_handled_without_error(self):
        count = 0
        for base in ("engine", "harness", "tests", "ui", "app", "docs", "."):
            for name in sorted(os.listdir(os.path.join(ROOT, base))):
                path = os.path.join(ROOT, base, name)
                if not os.path.isfile(path) or os.path.getsize(path) > 600_000:
                    continue
                try:
                    text = open(path, encoding="utf-8").read()
                except (UnicodeDecodeError, OSError):
                    continue
                lang = C.detect_language(path, text)
                spans = C.highlight(text, lang)
                self.assertTrue(all(0 <= a < b <= len(text) for a, b, _ in spans), path)
                C.outline(text, lang)
                count += 1
        self.assertGreater(count, 40)

    def test_known_symbols_in_our_own_code(self):
        def outline_of(rel, lang=None):
            text = open(os.path.join(ROOT, rel), encoding="utf-8").read()
            return C.outline(text, lang or C.detect_language(rel, text))
        self.assertIn("Agent", [s["name"] for s in outline_of("agent.py")])
        agent = find(outline_of("agent.py"), "Agent")
        self.assertIn("fix_tests", [s["name"] for s in agent["children"]])
        self.assertEqual(agent["kind"], "class")
        llm = [s["name"] for s in outline_of("llm.py")]
        for fn in ("load_cfg", "inspect_model", "hf_pull", "main"):
            self.assertIn(fn, llm)
        swift = outline_of("app/LLMBar.swift")
        top = {s["name"]: s for s in swift}
        for t in ("ContentView", "EngineState", "AppDelegate", "SettingsTab", "ModelsTab", "AboutView"):
            self.assertIn(t, top, f"{t} missing from Swift top level: {sorted(top)[:25]}")
        self.assertIn("showAbout", [s["name"] for s in top["AppDelegate"]["children"]])
        md = outline_of("README.md")
        self.assertTrue(md and md[0]["kind"] == "heading")


if __name__ == "__main__":
    unittest.main()
