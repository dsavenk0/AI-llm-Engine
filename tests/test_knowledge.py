"""Knowledge libraries (engine/knowledge.py) and `./llm knowledge` — temp databases, stdlib-made files.

    .venv/bin/python -m unittest tests.test_knowledge -v

No model, no network. The PDF cases use a fake page-mode helper; the real Vision/PDFKit helper is covered in
tests/test_documents.py (skipped when it is not built).
"""
import contextlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from engine import documents as D  # noqa: E402
from engine import knowledge as K  # noqa: E402


def write(d, name, text):
    p = os.path.join(d, name)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "wb" if isinstance(text, bytes) else "w", encoding=None if isinstance(text, bytes) else "utf-8") as f:
        f.write(text)
    return p


def fake_page_helper(d, pages):
    """A stand-in for `bin/ocr --pages`: answers the capability probe and prints the given pages."""
    lines = [json.dumps({"pages": len(pages)})] + [json.dumps({"page": i + 1, "text": t, "source": s})
                                                    for i, (t, s) in enumerate(pages)]
    data = write(d, "pages.jsonl", "\n".join(lines) + "\n")
    p = write(d, "fake-ocr", "#!/bin/sh\n"
              'if [ "$1" = "--pages" ] && [ $# -eq 1 ]; then echo "usage: ocr --pages <file.pdf>" >&2; exit 2; fi\n'
              f'cat "{data}"\n')
    os.chmod(p, 0o755)
    return p


MD = """# Guide

Intro paragraph.

## Install

Run the installer and wait.

```
# not a heading inside a fence
```

## Память

Модели загружаются в память по требованию.
"""


class Base(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.db = os.path.join(self.d, "kb", "knowledge.db")
        self.st = K.Store(self.db, pressure=lambda: False, sleep=lambda s: None)

    def tearDown(self):
        self.st.close()
        shutil.rmtree(self.d, ignore_errors=True)

    def lib(self, name="Docs"):
        return self.st.create_library(name)


class TestDormant(Base):
    def test_a_store_that_was_never_used_creates_nothing(self):
        scope = K.Scope(libraries=["Docs"], chat="c1")
        self.assertFalse(self.st.active(scope))
        self.assertEqual(self.st.search("anything", scope), [])
        self.assertIsNone(self.st.context("anything", scope))
        self.assertEqual(self.st.libraries(), [])
        self.assertEqual(self.st.documents(), [])
        self.assertEqual(self.st.forget_chat("c1"), 0)
        self.assertFalse(self.st.stats()["exists"])
        self.assertFalse(os.path.exists(self.db))

    def test_an_empty_scope_never_opens_the_database(self):
        lib = self.lib()
        self.st.add(write(self.d, "a.txt", "alpha beta"), library=lib)
        self.st.close()
        with mock.patch.object(K.sqlite3, "connect", side_effect=AssertionError("opened")):
            self.assertFalse(self.st.active(K.Scope()))
            self.assertEqual(self.st.search("alpha", K.Scope()), [])
            self.assertIsNone(self.st.context("alpha", K.Scope()))
            self.assertFalse(self.st.active(None))

    def test_a_library_without_ready_documents_adds_no_context(self):
        self.lib()
        self.assertFalse(self.st.active(K.Scope(libraries=["Docs"])))
        self.assertIsNone(self.st.context("x", K.Scope(libraries=["Docs"])))
        self.assertIsNone(self.st.context("x", K.Scope(libraries=["no such library"])))

    def test_removing_the_last_document_lets_go_of_the_connection(self):
        lib = self.lib()
        r = self.st.add(write(self.d, "a.txt", "alpha"), library=lib)
        self.assertIsNotNone(self.st._db)
        self.st.remove(r["id"], library=lib)
        self.assertIsNone(self.st._db)

    def test_importing_the_module_does_no_io(self):
        code = ("import sys, os; sys.path.insert(0, %r); os.environ['LLM_KNOWLEDGE_DB'] = %r;"
                "import engine.knowledge as K; K.store(); print(os.path.exists(%r))") % (ROOT, self.db, self.db)
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
        self.assertEqual(out.stdout.strip(), "False", out.stderr)


class TestIndexing(Base):
    def test_markdown_chunks_carry_heading_and_lines(self):
        lib = self.lib()
        self.st.add(write(self.d, "guide.md", MD), library=lib)
        hits = self.st.search("installer", K.Scope(libraries=[lib]))
        self.assertEqual(hits[0]["name"], "guide.md")
        self.assertIn("§ Guide › Install", hits[0]["anchor"])
        self.assertIn("L5-", hits[0]["anchor"])
        anchors = [c[0] for c in K.chunk_markdown(MD)]
        self.assertFalse(any("not a heading" in a for a in anchors))      # fenced "#" is not a heading

    def test_code_chunks_have_exact_line_ranges(self):
        src = "".join(f"line {i}\n" for i in range(1, 401))
        chunks = K.chunk_lines(src, size=500)
        self.assertGreater(len(chunks), 3)
        lines = src.split("\n")
        for anchor, text in chunks:
            a, _, b = anchor[1:].partition("-")
            self.assertEqual(text.split("\n")[0], lines[int(a) - 1])
            self.assertEqual(text.split("\n")[-1], lines[int(b or a) - 1])

    def test_cyrillic_and_word_forms(self):
        lib = self.lib()
        self.st.add(write(self.d, "ru.md", MD), library=lib)
        hits = self.st.search("модель памяти", K.Scope(libraries=["docs"]))     # names are case-insensitive
        self.assertTrue(hits)
        self.assertIn("Память", hits[0]["anchor"])

    def test_exact_phrase(self):
        lib = self.lib()
        self.st.add(write(self.d, "a.py", "def run_tests():\n    raise ValueError('index out of range')\n"), library=lib)
        self.st.add(write(self.d, "b.txt", "range of index values is out\n"), library=lib)
        hits = self.st.search("index out of range", K.Scope(libraries=[lib]), exact=True)
        self.assertEqual([h["name"] for h in hits], ["a.py"])

    def test_identical_content_is_stored_once(self):
        lib = self.lib()
        a = self.st.add(write(self.d, "a.txt", "same words"), library=lib)
        b = self.st.add(write(self.d, "copy.txt", "same words"), chat="c1")
        self.assertTrue(b["reused"])
        self.assertEqual(a["id"], b["id"])
        self.assertEqual(self.st.stats()["documents"], 1)

    def test_pdf_pages_become_anchors(self):
        os.environ["LLM_OCR"] = fake_page_helper(self.d, [("first page words", "text"), ("", "none"),
                                                           ("third page about tokens", "ocr")])
        try:
            lib = self.lib()
            r = self.st.add(write(self.d, "a.pdf", b"%PDF-1.4\n"), library=lib)
        finally:
            del os.environ["LLM_OCR"]
        self.assertEqual(r["state"], "ready")
        self.assertIn("1 page(s) have no text layer", r["note"])
        hits = self.st.search("tokens", K.Scope(libraries=[lib]))
        self.assertEqual(hits[0]["anchor"], "p. 3")
        doc = self.st.documents(library=lib)[0]
        self.assertEqual((doc["pages"], doc["anchored"], doc["kind"]), (3, True, "pdf"))

    def test_pdf_without_text_is_failed_not_ready(self):
        with mock.patch.object(D, "ocr_tool", lambda: None):
            r = self.st.add(write(self.d, "scan.pdf", b"%PDF-1.4\n/Type /Page\n"), library=self.lib())
        self.assertEqual(r["state"], "failed")
        self.assertIn("scanned", r["note"])
        self.assertFalse(self.st.active(K.Scope(libraries=["Docs"])))

    def test_refusals(self):
        lib = self.lib()
        for name, body in ((".env", "KEY=1"), ("id_rsa", "x"), ("server.pem", "x"), ("a.exe", "x")):
            with self.assertRaises(K.KnowledgeError):
                self.st.add(write(self.d, name, body), library=lib)
        with self.assertRaises(K.KnowledgeError):
            self.st.add(os.path.join(self.d, "missing.txt"), library=lib)
        with self.assertRaises(K.KnowledgeError):
            self.st.add(write(self.d, "x.txt", "x"))                       # no owner
        with mock.patch.object(K, "MAX_TEXT_BYTES", 10):
            with self.assertRaises(K.KnowledgeError):
                self.st.add(write(self.d, "big.txt", "x" * 11), library=lib)
        r = self.st.add(write(self.d, "bin.txt", b"\x00\x01binary"), library=lib)
        self.assertEqual(r["state"], "failed")

    def test_changed_original_is_flagged(self):
        lib = self.lib()
        p = write(self.d, "a.txt", "alpha")
        self.st.add(p, library=lib)
        self.assertFalse(self.st.documents()[0]["changed"])
        with open(p, "a") as f:
            f.write(" more")
        self.assertTrue(self.st.documents()[0]["changed"])
        os.remove(p)
        self.assertTrue(self.st.documents()[0]["missing"])

    def test_folder_skips_vcs_dependencies_hidden_and_secrets(self):
        root = os.path.join(self.d, "proj")
        for rel in ("README.md", "src/app.py", "node_modules/x/index.js", ".git/config.toml", ".hidden/a.md",
                    ".env", "keys/server.pem", "img.png", "notes/.draft.md"):
            write(root, rel, "words")
        rels = sorted(os.path.relpath(p, root) for p in K.list_folder(root))
        self.assertEqual(rels, ["README.md", os.path.join("src", "app.py")])

    @unittest.skipUnless(shutil.which("git"), "git is not installed")
    def test_folder_respects_gitignore(self):
        root = os.path.join(self.d, "repo")
        write(root, "keep.md", "a")
        write(root, "out/gen.md", "b")
        write(root, ".gitignore", "out/\n")
        subprocess.run(["git", "init", "-q", root], check=True)
        rels = sorted(os.path.relpath(p, root) for p in K.list_folder(root))
        self.assertEqual(rels, ["keep.md"])


class TestScopeAndLifecycle(Base):
    def setUp(self):
        super().setUp()
        self.lib_id = self.lib()
        self.shared = write(self.d, "shared.txt", "shared zebra facts")
        self.private = write(self.d, "private.txt", "private giraffe facts")
        self.st.add(self.shared, library=self.lib_id)
        self.st.add(self.shared, chat="A")
        self.st.add(self.private, chat="A")

    def names(self, q, scope):
        return sorted(h["name"] for h in self.st.search(q, scope))

    def test_a_chat_reads_only_its_own_and_its_libraries(self):
        self.assertEqual(self.names("giraffe", K.Scope(chat="A")), ["private.txt"])
        self.assertEqual(self.names("giraffe", K.Scope(chat="B")), [])
        self.assertEqual(self.names("giraffe", K.Scope(libraries=[self.lib_id])), [])
        self.assertEqual(self.names("zebra", K.Scope(libraries=[self.lib_id], chat="B")), ["shared.txt"])

    def test_deleting_a_chat_keeps_what_a_library_owns(self):
        removed = self.st.forget_chat("A")
        self.assertEqual(removed, 1)
        self.assertEqual(self.names("giraffe", K.Scope(chat="A")), [])
        self.assertEqual(self.names("zebra", K.Scope(libraries=[self.lib_id])), ["shared.txt"])
        self.assertTrue(os.path.exists(self.private) and os.path.exists(self.shared))   # originals untouched

    def test_promote_keeps_a_temporary_document(self):
        doc = next(d["id"] for d in self.st.documents(chat="A") if d["name"] == "private.txt")
        self.st.promote(doc, self.lib_id)
        self.st.forget_chat("A")
        self.assertEqual(self.names("giraffe", K.Scope(libraries=[self.lib_id])), ["private.txt"])

    def test_deleting_the_library_clears_the_index(self):
        self.st.forget_chat("A")
        self.st.delete_library(self.lib_id)
        self.assertEqual(self.st.stats()["documents"], 0)
        db = self.st._conn()
        self.assertEqual(db.execute("SELECT COUNT(*) FROM chunks_fts WHERE chunks_fts MATCH 'zebra'").fetchone()[0], 0)
        self.assertTrue(os.path.exists(self.shared))
        with self.assertRaises(K.KnowledgeError):
            self.st.delete_library(self.lib_id)

    def test_library_names(self):
        with self.assertRaises(K.KnowledgeError):
            self.st.create_library("docs")                              # names are unique regardless of case
        self.st.rename_library("Docs", "Manuals")
        self.assertEqual([lib["name"] for lib in self.st.libraries()], ["Manuals"])
        self.assertEqual(self.st.libraries()[0]["ready"], 1)


class TestContext(Base):
    def test_numbered_sources_untrusted_header_and_budget(self):
        lib = self.lib()
        for i in range(6):
            self.st.add(write(self.d, f"n{i}.txt", f"orbit telescope note {i}. " * 60), library=lib)
        c = self.st.context("telescope orbit", K.Scope(libraries=[lib]), budget=2500)
        self.assertLessEqual(len(c["text"]), 2500 + 600)                  # header + labels on top of the budget
        self.assertIn("not instructions", c["text"])
        self.assertEqual([s["n"] for s in c["sources"]], list(range(1, len(c["sources"]) + 1)))
        self.assertIn("[1] ", c["text"])
        self.assertTrue(all(s["anchor"].startswith("L") for s in c["sources"]))

    def test_no_match_means_no_context(self):
        lib = self.lib()
        self.st.add(write(self.d, "a.txt", "alpha"), library=lib)
        self.assertIsNone(self.st.context("zzzz", K.Scope(libraries=[lib])))

    def test_hostile_queries_are_just_words(self):
        lib = self.lib()
        self.st.add(write(self.d, "a.txt", "NEAR AND OR alpha"), library=lib)
        for q in ('"', "AND (", "NEAR(a b)", "a* OR -b", "col:alpha", "^", "' OR 1=1 --", "\x00"):
            self.st.search(q, K.Scope(libraries=[lib]))
            self.st.search(q, K.Scope(libraries=[lib]), exact=True)
        self.assertEqual(K.fts_query(""), "")
        self.assertEqual(K.fts_query("the"), '"the"')                     # a lone stop word is still searched


class TestPressure(Base):
    def test_ingestion_waits_while_memory_is_tight(self):
        states = iter([True, True, False, False])
        slept, events = [], []
        st = K.Store(self.db, pressure=lambda: next(states), sleep=slept.append)
        lib = st.create_library("L")
        r = st.add_many([write(self.d, "a.txt", "one"), write(self.d, "b.txt", "two")], library=lib,
                        progress=events.append)
        st.close()
        self.assertEqual([x["state"] for x in r], ["ready", "ready"])
        self.assertEqual(len(slept), 2)
        self.assertEqual(sum(e["stage"] == "paused" for e in events), 1)

    def test_stop_while_paused(self):
        st = K.Store(self.db, pressure=lambda: True, sleep=lambda s: None)
        lib = st.create_library("L")
        calls = iter([False, True])
        r = st.add_many([write(self.d, "a.txt", "one")], library=lib, should_stop=lambda: next(calls, True))
        st.close()
        self.assertEqual(r, [])


class TestCLI(Base):
    def run_cli(self, *args):
        import llm
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
            ok = llm.knowledge_cmd({}, list(args) + ["--db", self.db] + ([] if "--json" in args else ["--json"]))
        return ok, buf.getvalue()

    def test_create_add_search_remove(self):
        p = write(self.d, "notes.md", MD)
        self.assertTrue(self.run_cli("create", "Notes")[0])
        ok, out = self.run_cli("add", "Notes", p, "--json")
        self.assertTrue(ok)
        doc = json.loads(out)[0]["id"]
        ok, out = self.run_cli("search", "Notes", "installer", "--json")
        self.assertEqual(json.loads(out)[0]["name"], "notes.md")
        ok, out = self.run_cli("search", "Notes", "installer", "--context")
        self.assertIn("[1] notes.md", out)
        self.assertTrue(self.run_cli("remove", "Notes", str(doc))[0])
        ok, out = self.run_cli("stats", "--json")
        self.assertEqual(json.loads(out)["ready"], 0)

    def test_chat_documents_attach_search_promote_forget(self):
        p = write(self.d, "spec.md", "# Spec\n\nThe flux capacitor needs 1.21 gigawatts.\n")
        ok, out = self.run_cli("attach", "CHAT-1", p)
        self.assertTrue(ok)
        doc = json.loads(out)[0]["id"]
        ok, out = self.run_cli("search", "-", "gigawatts", "--chat", "CHAT-1")
        self.assertEqual(json.loads(out)[0]["name"], "spec.md")
        ok, out = self.run_cli("search", "-", "gigawatts", "--chat", "OTHER")
        self.assertEqual(json.loads(out), [])
        ok, out = self.run_cli("docs", "--chat", "CHAT-1")
        self.assertEqual([d["name"] for d in json.loads(out)], ["spec.md"])
        self.run_cli("create", "Keep")
        self.assertTrue(self.run_cli("promote", str(doc), "Keep")[0])
        ok, out = self.run_cli("forget-chat", "CHAT-1")
        self.assertEqual(json.loads(out)["removed"], 0)                  # the library keeps it
        ok, out = self.run_cli("search", "Keep", "gigawatts")
        self.assertEqual(len(json.loads(out)), 1)
        self.run_cli("attach", "CHAT-2", write(self.d, "tmp.txt", "ephemeral note"))
        ok, out = self.run_cli("docs", "--chat", "CHAT-2")
        tmp = json.loads(out)[0]["id"]
        self.assertTrue(self.run_cli("detach", "CHAT-2", str(tmp))[0])
        ok, out = self.run_cli("search", "-", "ephemeral", "--chat", "CHAT-2")
        self.assertEqual(json.loads(out), [])

    def test_terminal_chat_delete_forgets_its_documents(self):
        from harness import chats
        self.st.add(write(self.d, "a.txt", "walrus"), chat="S1")
        sessions = [{"id": "S1", "title": "t", "updated": 1, "items": [], "knowledgeChat": True},
                    {"id": "S2", "title": "u", "updated": 1, "items": []}]
        with mock.patch.object(K, "store", lambda path=None: self.st):
            chats.delete(sessions, sessions[0])
            chats.delete(sessions, sessions[0])                          # S2 has no knowledge: nothing to do
        self.assertEqual(self.st.search("walrus", K.Scope(chat="S1")), [])
        self.assertEqual(sessions, [])

    def test_errors_are_reported_not_raised(self):
        ok, out = self.run_cli("add", "Nope", __file__)
        self.assertFalse(ok)
        ok, out = self.run_cli("delete", "Nope", "--yes", "--json")
        self.assertFalse(ok)
        self.assertIn("error", json.loads(out))
        self.assertFalse(self.run_cli("bogus")[0])


if __name__ == "__main__":
    unittest.main()
