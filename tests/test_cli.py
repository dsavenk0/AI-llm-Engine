"""CLI tests (stdlib only): client config writers and translations.

    .venv/bin/python -m unittest tests.test_cli -v
"""
import ast
import json
import os
import re
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import llm  # noqa: E402

MODEL = {"name": "Demo-Model", "path": "/tmp/demo", "size": 4.0, "vision": True, "thinking": True}


def cfg():
    c = llm.load_cfg()
    c.update(port=8090, api_key="", mem_limit=88, models={})
    return c


class TestContinueConfig(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.path = os.path.join(self.dir, "config.yaml")

    def write(self):
        llm._write_continue(self.path, cfg(), MODEL, 40960, 8192)
        with open(self.path) as f:
            return f.read()

    def test_new_file(self):
        text = self.write()
        self.assertIn("schema: v1", text)
        self.assertIn("apiBase: http://localhost:8090/v1", text)
        self.assertIn("contextLength: 40960", text)
        self.assertIn("capabilities: [tool_use, image_input]", text)

    def test_keeps_other_models_and_is_idempotent(self):
        with open(self.path, "w") as f:
            f.write("name: Mine\nversion: 1\nschema: v1\nmodels:\n  - name: Other\n    provider: ollama\n"
                    "    model: qwen\nrules:\n  - be nice\n")
        first = self.write()
        second = self.write()
        self.assertEqual(first, second)
        self.assertEqual(first.count(llm.BEGIN), 1)
        self.assertIn("  - name: Other", first)
        self.assertIn("rules:\n  - be nice", first)
        self.assertTrue(os.path.exists(self.path + ".bak"))
        # our block sits inside the models list (indented under "models:")
        self.assertLess(first.index("models:"), first.index(llm.BEGIN))
        self.assertLess(first.index(llm.END), first.index("rules:"))

    def test_empty_models_list(self):
        with open(self.path, "w") as f:
            f.write("name: x\nversion: 1\nschema: v1\nmodels: []\n")
        text = self.write()
        self.assertNotIn("models: []", text)
        self.assertIn("models:\n  " + llm.BEGIN, text)


class TestOpencodeStyleConfig(unittest.TestCase):
    def test_kilo_replaces_legacy_provider_and_keeps_others(self):
        d = tempfile.mkdtemp()
        p = os.path.join(d, "kilo.jsonc")
        with open(p, "w") as f:
            f.write('{\n  // comment\n  "provider": {"local_qwen": {}, "other": {"x": 1},},\n  "theme": "dark"\n}\n')
        llm._write_opencode_style(p, cfg(), MODEL, 40960, 8192, drop=("local_qwen",))
        with open(p) as f:
            data = json.load(f)
        self.assertNotIn("local_qwen", data["provider"])
        self.assertEqual(data["provider"]["other"], {"x": 1})
        self.assertEqual(data["theme"], "dark")
        m = data["provider"]["local_mlx"]["models"]["demo-model"]
        self.assertEqual(m["limit"], {"context": 40960, "output": 8192})
        self.assertEqual(data["model"], "local_mlx/demo-model")


class TestTranslations(unittest.TestCase):
    def test_every_ui_string_has_russian(self):
        with open(os.path.join(ROOT, "locales", "ru.json"), encoding="utf-8") as f:
            ru = json.load(f)
        keys = set()
        for name in ("llm.py", "agent.py"):
            with open(os.path.join(ROOT, name)) as f:
                src = f.read()
            for n in ast.walk(ast.parse(src)):
                if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "T" and n.args \
                        and isinstance(n.args[0], ast.Constant):
                    keys.add(n.args[0].value)
            keys |= set(re.findall(r'\("([A-Z][^"]+)", (?:lambda|logs|run_tests)', src))
        self.assertEqual(sorted(keys - set(ru)), [])
        for k, v in ru.items():
            self.assertEqual(set(re.findall(r"\{(\w+)", k)), set(re.findall(r"\{(\w+)", v)), k)


if __name__ == "__main__":
    unittest.main()
