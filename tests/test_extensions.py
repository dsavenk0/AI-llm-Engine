"""Open VSX support (ui/extensions.py): offline — the network is a function that is passed in."""
import io
import json
import os
import tempfile
import unittest
import zipfile

from ui import extensions as X

THEME = """{
  // a comment, and a trailing comma below — both legal in VS Code theme files
  "name": "Night",
  "type": "dark",
  "colors": {"editor.background": "#112233", "editor.foreground": "#EEEEEE"},
  "tokenColors": [
    {"settings": {"foreground": "#CCCCCC"}},
    {"scope": ["comment", "punctuation.definition.comment"], "settings": {"foreground": "#667788", "fontStyle": "italic"}},
    {"scope": "keyword", "settings": {"foreground": "#FF0000"}},
    {"scope": "keyword.control", "settings": {"foreground": "#00FF00"}},
    {"scope": "string, string.quoted.double", "settings": {"foreground": "#0000FF"}},
    {"scope": "entity.name.function", "settings": {"foreground": "#ABCDEF"}},
  ],
}"""
SNIPPETS = '{"Print": {"prefix": ["pr", "print"], "body": ["print($1)", "$0"], "description": "print"}}'


def make_vsix(files, pkg=None):
    pkg = pkg or {"name": "night", "publisher": "acme", "version": "1.2.0", "displayName": "Night",
                  "contributes": {"themes": [{"label": "Night", "uiTheme": "vs-dark", "path": "./themes/night.json"}],
                                  "snippets": [{"language": "python", "path": "./snippets/py.json"}],
                                  "languages": [{"id": "python", "extensions": [".py"]}],
                                  "commands": [{"command": "x"}]},
                  "main": "./out/extension.js"}
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("extension/package.json", json.dumps(pkg))
        for name, text in files.items():
            z.writestr("extension/" + name, text)
        z.writestr("extension/out/extension.js", "throw new Error('must never run')")
    return buf.getvalue()


class TestExtensions(unittest.TestCase):
    def setUp(self):
        self.base = tempfile.mkdtemp()
        self.vsix = make_vsix({"themes/night.json": THEME, "snippets/py.json": SNIPPETS})

    def test_jsonc_handles_comments_and_trailing_commas_but_not_inside_strings(self):
        self.assertEqual(X.jsonc('{"a": "// not a comment", /* x */ "b": [1,2,],}'), {"a": "// not a comment", "b": [1, 2]})

    def test_install_keeps_only_declarative_files_and_reports_what_it_skips(self):
        m = X.install_vsix(self.vsix, dest_root=self.base)
        self.assertEqual(m["id"], "acme.night")
        self.assertTrue(m["has_code"])
        self.assertIn("commands", m["skipped"])
        folder = os.path.join(self.base, "acme.night-1.2.0")
        self.assertEqual(sorted(os.listdir(folder)), ["manifest.json", "package.json", "snippets", "themes"])
        self.assertFalse(os.path.exists(os.path.join(folder, "out")))          # the extension's code is never unpacked

    def test_theme_includes_may_form_a_cycle_without_hanging(self):
        a = json.dumps({"include": "./b.json", "colors": {"editor.background": "#101010"}})
        b = json.dumps({"include": "./a.json", "tokenColors": []})
        pkg = {"name": "loop", "publisher": "acme", "version": "1",
               "contributes": {"themes": [{"label": "L", "uiTheme": "vs-dark", "path": "./themes/a.json"}]}}
        vsix = make_vsix({"themes/a.json": a, "themes/b.json": b}, pkg)
        m = X.install_vsix(vsix, dest_root=self.base)            # would spin forever on a cycle
        self.assertEqual([t["label"] for t in m["themes"]], ["L"])
        self.assertEqual(X.theme("acme.loop", "L", base=self.base)["background"], "#101010")

    def test_checksum_is_verified_when_given(self):
        import hashlib
        good = hashlib.sha256(self.vsix).hexdigest()
        X.install_vsix(self.vsix, expected_sha256=good + "  file.vsix", dest_root=self.base)
        with self.assertRaises(X.ExtensionError):
            X.install_vsix(self.vsix, expected_sha256="0" * 64, dest_root=self.base)

    def test_a_zip_with_path_tricks_or_a_bad_id_is_refused(self):
        evil = make_vsix({"themes/night.json": THEME}, {
            "name": "../x", "publisher": "acme", "version": "1", "contributes": {}})
        with self.assertRaises(X.ExtensionError):
            X.install_vsix(evil, dest_root=self.base)
        tricky = make_vsix({}, {"name": "ok", "publisher": "acme", "version": "1",
                                "contributes": {"themes": [{"label": "T", "path": "../../etc/passwd"}]}})
        m = X.install_vsix(tricky, dest_root=self.base)
        self.assertEqual(m["themes"], [])                                       # the escaping path is ignored
        with self.assertRaises(X.ExtensionError):
            X.install_vsix(b"not a zip", dest_root=self.base)

    def test_theme_conversion_prefers_the_most_specific_scope(self):
        X.install_vsix(self.vsix, dest_root=self.base)
        t = X.theme("acme.night", "Night", base=self.base)
        self.assertTrue(t["dark"])
        self.assertEqual((t["background"], t["foreground"]), ("#112233", "#eeeeee"))
        self.assertEqual(t["kinds"]["keyword"], "#00ff00")                      # keyword.control beats keyword
        self.assertEqual(t["kinds"]["comment"], "#667788")
        self.assertEqual(t["kinds"]["string"], "#0000ff")
        self.assertEqual(t["kinds"]["function"], "#abcdef")
        with self.assertRaises(X.ExtensionError):
            X.theme("acme.night", "Nope", base=self.base)

    def test_snippets_come_back_per_language(self):
        X.install_vsix(self.vsix, dest_root=self.base)
        got = X.snippets("python", base=self.base)
        self.assertEqual(sorted(s["prefix"] for s in got), ["pr", "print"])
        self.assertEqual(got[0]["body"], ["print($1)", "$0"])
        self.assertEqual(X.snippets("rust", base=self.base), [])

    def test_a_new_version_replaces_the_old_and_remove_deletes(self):
        X.install_vsix(self.vsix, dest_root=self.base)
        newer = make_vsix({"themes/night.json": THEME, "snippets/py.json": SNIPPETS},
                          {"name": "night", "publisher": "acme", "version": "2.0.0", "contributes": {}})
        X.install_vsix(newer, dest_root=self.base)
        self.assertEqual([m["version"] for m in X.installed(self.base)], ["2.0.0"])
        self.assertEqual(X.remove("acme.night", base=self.base), 1)
        self.assertEqual(X.installed(self.base), [])
        with self.assertRaises(X.ExtensionError):
            X.remove("../etc", base=self.base)

    def test_removing_one_extension_never_takes_a_neighbour_whose_id_starts_the_same(self):
        long_pkg = {"name": "night-extra", "publisher": "acme", "version": "1.0.0", "contributes": {}}
        X.install_vsix(self.vsix, dest_root=self.base)
        X.install_vsix(make_vsix({}, long_pkg), dest_root=self.base)
        self.assertEqual(sorted(m["id"] for m in X.installed(self.base)), ["acme.night", "acme.night-extra"])
        self.assertEqual(X.remove("acme.night", base=self.base), 1)
        self.assertEqual([m["id"] for m in X.installed(self.base)], ["acme.night-extra"])
        # updating one does not replace the other either
        X.install_vsix(self.vsix, dest_root=self.base)
        X.install_vsix(make_vsix({}, dict(long_pkg, version="2.0.0")), dest_root=self.base)
        self.assertEqual(sorted((m["id"], m["version"]) for m in X.installed(self.base)),
                         [("acme.night", "1.2.0"), ("acme.night-extra", "2.0.0")])

    def test_search_skips_incomplete_rows_and_reports_garbage(self):
        good = {"namespace": "a", "name": "b", "version": "1", "description": None, "downloadCount": None}
        rows = json.dumps({"extensions": [good, {"name": "orphan"}, "text", {"namespace": "a"}]}).encode()
        got = X.search("x", get=lambda url, limit=0: rows)
        self.assertEqual([r["id"] for r in got], ["a.b"])
        self.assertEqual((got[0]["description"], got[0]["downloads"]), ("", 0))
        with self.assertRaises(X.ExtensionError):
            X.search("x", get=lambda url, limit=0: b"<html>")
        with self.assertRaises(X.ExtensionError):
            X.search("x", get=lambda url, limit=0: b'["not", "an", "object"]')

    def test_the_checksum_is_asked_for_before_the_big_download(self):
        calls = []

        def get(url, limit=0):
            calls.append(url)
            if url.endswith("/api/acme/night"):
                return json.dumps({"files": {"download": "https://open-vsx.org/x.vsix"}}).encode()   # no sha256
            return self.vsix
        with self.assertRaises(X.ExtensionError):
            X.install("acme", "night", get=get)
        self.assertEqual(len(calls), 1)                                  # the package itself was never fetched

    def test_only_open_vsx_over_https_is_ever_fetched(self):
        for url in ("http://open-vsx.org/api/x", "https://evil.example/open-vsx.org", "https://open-vsx.org.evil.io/x",
                    "file:///etc/passwd"):
            with self.assertRaises(X.ExtensionError):
                X.fetch(url)

    def test_install_from_the_api_with_a_fake_network(self):
        import hashlib
        calls = []
        sha = hashlib.sha256(self.vsix).hexdigest()

        def get(url, limit=0):
            calls.append(url)
            if url.endswith("/api/acme/night"):
                return json.dumps({"files": {"download": "https://open-vsx.org/api/acme/night/1.2.0/file/x.vsix",
                                             "sha256": "https://open-vsx.org/api/acme/night/1.2.0/file/x.sha256"}}).encode()
            if url.endswith(".vsix"):
                return self.vsix
            return sha.encode()
        os.environ["LLM_EXTENSIONS_DIR"] = self.base
        self.addCleanup(os.environ.pop, "LLM_EXTENSIONS_DIR", None)
        m = X.install("acme", "night", get=get)
        self.assertEqual(m["version"], "1.2.0")
        self.assertEqual(len(calls), 3)
        with self.assertRaises(X.ExtensionError):
            X.install("acme/../x", "night", get=get)                             # an id is a plain name
        with self.assertRaises(X.ExtensionError):
            X.install("acme", "night", get=lambda u, limit=0: json.dumps(
                {"files": {"download": "https://evil.example/x.vsix"}}).encode())
        with self.assertRaises(X.ExtensionError):                                 # no published checksum: not installed
            X.install("acme", "night", get=lambda u, limit=0: json.dumps(
                {"files": {"download": "https://open-vsx.org/x.vsix"}}).encode() if "/api/" in u and ".vsix" not in u
                else self.vsix)
        X._check_url("https://openvsx.eclipsecontent.org/a/b.vsix")               # the CDN open-vsx.org redirects to


class TestExtensionsMore(unittest.TestCase):
    def setUp(self):
        self.base = tempfile.mkdtemp()

    def test_limits_are_what_the_docs_say(self):
        self.assertEqual((X.MAX_VSIX, X.MAX_MEMBER, X.MAX_UNPACKED), (40 * 2**20, 6 * 2**20, 60 * 2**20))

    def test_search_clamps_its_size_and_survives_odd_answers(self):
        seen = []
        get = lambda url, **kw: (seen.append(url), b'{"extensions": []}')[1]
        X.search("night", size=0, get=get)
        X.search("night", size=999, get=get)
        X.search("night", size=7, get=get)
        self.assertTrue(seen[0].endswith("size=1&sortBy=relevance") or "size=1&" in seen[0])
        self.assertIn("size=50&", seen[1])
        self.assertIn("size=7&", seen[2])
        for body in (b'{"extensions": null}', b'{}', b'[]', b'{"extensions": "x"}', b'{"extensions": {"a": 1}}'):
            try:
                self.assertIsInstance(X.search("n", get=lambda url, **kw: body), list)
            except X.ExtensionError:
                pass                                                  # a clean error is fine; a crash is not

    def test_an_answer_that_is_not_an_object_or_not_json_is_a_clean_error(self):
        for body in (b"<html>", b"[1, 2]", b'"x"'):
            with self.assertRaises(X.ExtensionError, msg=body):
                X.install("acme", "night", get=lambda url, **kw: body)

    def test_a_blank_checksum_is_a_mismatch_not_a_crash(self):
        vsix = make_vsix({"themes/night.json": THEME, "snippets/py.json": SNIPPETS})
        for bad in ("   ", "\n", "deadbeef"):
            with self.assertRaises(X.ExtensionError, msg=repr(bad)):
                X.install_vsix(vsix, bad, dest_root=self.base)
        import hashlib
        good = hashlib.sha256(vsix).hexdigest()
        X.install_vsix(vsix, good.upper() + "  night.vsix", dest_root=self.base)      # sha256sum's format, any case

    def test_a_package_that_claims_too_much_or_holds_a_huge_file_is_refused(self):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            z.writestr("extension/package.json", json.dumps({"name": "n", "publisher": "p",
                       "contributes": {"themes": [{"label": "L", "path": "./t.json"}]}}))
            z.writestr("extension/t.json", "x" * (X.MAX_MEMBER + 1))
        with self.assertRaises(X.ExtensionError):
            X.install_vsix(buf.getvalue(), dest_root=self.base)
        self.assertEqual(os.listdir(self.base), [])                                  # nothing half-installed

    def test_jsonc_keeps_escaped_quotes_comment_markers_in_strings_and_a_bom(self):
        self.assertEqual(X.jsonc('{"a": "he said \\"//hi\\" /* x */", "b": 1 // tail\n}'),
                         {"a": 'he said "//hi" /* x */', "b": 1})
        self.assertEqual(X.jsonc("\ufeff{\"a\": 1 /* c */}"), {"a": 1})
        self.assertEqual(X.jsonc('{"a": [1, {"b": 2,},],}'), {"a": [1, {"b": 2}]})

    def test_installed_skips_manifests_that_are_not_objects(self):
        for name, body in (("ok.ext-1", '{"id": "ok.ext"}'), ("bad.ext-1", "[1]"), ("worse.ext-1", "{")):
            os.makedirs(os.path.join(self.base, name))
            with open(os.path.join(self.base, name, "manifest.json"), "w") as f:
                f.write(body)
        got = X.installed(self.base)
        self.assertEqual([m["id"] for m in got], ["ok.ext"])
        self.assertEqual(got[0]["folder"], "ok.ext-1")

    def test_an_extension_with_odd_contributions_still_installs_what_is_valid(self):
        pkg = {"name": "odd", "publisher": "acme", "version": "0.1.0",
               "contributes": {"themes": ["x", {"path": "./missing.json"}, {"label": "Ok", "path": "./t.json"}],
                               "snippets": [{"path": "./s.json"}, {"language": "go", "path": "./s.json"}],
                               "languages": [{"id": "go", "extensions": [".go"], "aliases": ["Go"]}, "x", {}],
                               "iconThemes": [{"id": "i"}], "debuggers": [{"type": "x"}]}}
        vsix = make_vsix({"t.json": '{"name": "n", "colors": {}}', "s.json": "{}"}, pkg)
        m = X.install_vsix(vsix, dest_root=self.base)
        self.assertEqual([t["label"] for t in m["themes"]], ["Ok"])
        self.assertEqual([s["language"] for s in m["snippets"]], ["go"])
        self.assertEqual(m["languages"], [{"id": "go", "extensions": [".go"], "aliases": ["Go"]}])
        self.assertEqual(m["skipped"], ["debuggers"])                              # iconThemes is declarative, not skipped
        self.assertFalse(m["has_code"] and "main" not in pkg)
        self.assertEqual(m["version"], "0.1.0")


if __name__ == "__main__":
    unittest.main()
