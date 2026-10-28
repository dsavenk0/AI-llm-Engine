"""The Conductor: decomposition, routing (remote parallel vs local switching), synthesis, and config.

    .venv/bin/python -m unittest tests.test_conductor -v
"""
import json
import os
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness import conductor  # noqa: E402


class FakeLLM(BaseHTTPRequestHandler):
    """A deterministic OpenAI-compatible endpoint: it answers by what the prompt asks for."""
    prompts = []

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n))
        content = body["messages"][0]["content"]
        FakeLLM.prompts.append(content)
        if "conductor of a team" in content:
            out = json.dumps({"subtasks": [
                {"id": "s1", "role": "fast", "task": "find the files", "depends_on": []},
                {"id": "s2", "role": "code", "task": "write the function", "depends_on": ["s1"]}]})
        elif "Combine their results" in content:
            out = "FINAL ANSWER"
        else:
            out = "done: " + content.split("\n")[0][:40]
        data = json.dumps({"choices": [{"message": {"content": out}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


class Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = ThreadingHTTPServer(("127.0.0.1", 0), FakeLLM)
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()
        cls.url = f"http://127.0.0.1:{cls.srv.server_address[1]}/v1"

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        cls.srv.server_close()

    def setUp(self):
        FakeLLM.prompts.clear()
        self.root = tempfile.mkdtemp()

    def remote(self, model="m"):
        return {"type": "openai", "url": self.url, "model": model}


class TestExtractJSON(unittest.TestCase):
    def test_finds_object_in_prose_and_fences(self):
        self.assertEqual(conductor._extract_json('sure:\n```json\n{"a": 1}\n```'), {"a": 1})
        self.assertEqual(conductor._extract_json('{"a": {"b": 2}} trailing'), {"a": {"b": 2}})
        self.assertEqual(conductor._extract_json('a string with a brace "}" {"ok": true}'), {"ok": True})
        self.assertIsNone(conductor._extract_json("no json here"))


class TestDecompose(Base):
    def test_parses_plan_and_validates_roles_and_deps(self):
        eng = conductor._Engine(None, lambda _l: None)
        subs = conductor.decompose("do a thing", self.remote(), eng, self.root, {"heavy", "code", "fast"})
        self.assertEqual([s["id"] for s in subs], ["s1", "s2"])
        self.assertEqual(subs[1]["depends_on"], ["s1"])

    def test_unknown_role_falls_back_to_an_available_one(self):
        # only 'heavy' is available, so the planner's fast/code map onto it
        eng = conductor._Engine(None, lambda _l: None)
        subs = conductor.decompose("x", self.remote(), eng, self.root, {"heavy"})
        self.assertTrue(all(s["role"] == "heavy" for s in subs))


class TestRouting(Base):
    def test_remote_roles_run_and_return_results(self):
        roles = {"heavy": self.remote(), "fast": self.remote(), "code": self.remote()}
        eng = conductor._Engine(None, lambda _l: None)
        subs = [{"id": "s1", "role": "fast", "task": "A", "depends_on": []},
                {"id": "s2", "role": "code", "task": "B", "depends_on": ["s1"]}]
        res = conductor.run_subtasks(subs, roles, eng, self.root)
        self.assertTrue(res["s1"].startswith("done: A"))
        self.assertTrue(res["s2"].startswith("done: B"))

    def test_local_roles_switch_the_engine_and_are_grouped(self):
        switches = []

        def switch_local(model):
            switches.append(model)
            return self.remote(model)              # the "loaded" model answers from the fake endpoint

        roles = {"heavy": self.remote("big"), "fast": {"local": "ModelA"}, "code": {"local": "ModelB"}}
        eng = conductor._Engine(switch_local, lambda _l: None)
        subs = [{"id": "s1", "role": "fast", "task": "A", "depends_on": []},     # ModelA
                {"id": "s3", "role": "fast", "task": "C", "depends_on": []},     # ModelA (same level, same model)
                {"id": "s2", "role": "code", "task": "B", "depends_on": ["s1"]}]  # ModelB, later level
        res = conductor.run_subtasks(subs, roles, eng, self.root)
        self.assertEqual(len(res), 3)
        # s1 and s3 share ModelA (one switch), then one switch to ModelB: three subtasks, two switches
        self.assertEqual(switches, ["ModelA", "ModelB"])

    def test_a_failed_subtask_is_reported_not_raised(self):
        roles = {"heavy": self.remote(), "fast": {"type": "openai", "url": "http://127.0.0.1:0/v1", "model": "m"}}
        eng = conductor._Engine(None, lambda _l: None)
        res = conductor.run_subtasks([{"id": "s1", "role": "fast", "task": "A", "depends_on": []}], roles, eng, self.root)
        self.assertTrue(res["s1"].startswith("error:"))


class TestConductEndToEnd(Base):
    def test_decompose_route_synthesize(self):
        conductor.set_role(self.root, "heavy", self.remote("big"))
        conductor.set_role(self.root, "code", self.remote("coder"))
        conductor.set_role(self.root, "fast", self.remote("quick"))
        trace = []
        answer = conductor.conduct("build a feature", self.root, emit=trace.append)
        self.assertEqual(answer, "FINAL ANSWER")
        self.assertTrue(any(line.startswith("[plan]") for line in trace))
        self.assertTrue(any("[route] s1" in line for line in trace))
        self.assertTrue(any(line.startswith("[synth]") for line in trace))

    def test_missing_heavy_is_a_clear_error(self):
        with self.assertRaises(RuntimeError) as cm:
            conductor.conduct("x", self.root)       # no config, no local fallback
        self.assertIn("heavy", str(cm.exception))


class TestConfig(Base):
    def test_set_role_writes_and_resolve_reads(self):
        conductor.set_role(self.root, "heavy", self.remote("big"))
        conductor.set_role(self.root, "code", {"local": "Coder-7B"})
        # a role given as a worker name resolves against .llm/workers.json
        os.makedirs(os.path.join(self.root, ".llm"), exist_ok=True)
        with open(os.path.join(self.root, ".llm", "workers.json"), "w") as f:
            json.dump({"workers": {"quick": {"type": "openai", "url": self.url, "model": "q"}}}, f)
        conductor.set_role(self.root, "fast", "quick")
        roles, dropped = conductor.resolve_roles(self.root)
        self.assertTrue(conductor.is_local(roles["code"]))
        self.assertEqual(roles["fast"]["model"], "q")
        self.assertEqual(roles["heavy"]["model"], "big")
        self.assertEqual(dropped, {})

    def test_a_missing_worker_is_dropped_with_a_reason(self):
        conductor.set_role(self.root, "heavy", "ghost")
        roles, dropped = conductor.resolve_roles(self.root)
        self.assertIn("heavy", dropped)
        self.assertIn("ghost", dropped["heavy"])

    def test_unmapped_roles_fall_back_to_local(self):
        conductor.set_role(self.root, "heavy", self.remote())
        roles, _ = conductor.resolve_roles(self.root, local={"local": "Selected"})
        self.assertEqual(roles["fast"], {"local": "Selected"})


class TestPerception(unittest.TestCase):
    """Attachments → text sources, and those sources reaching the planner and the subtasks."""

    class Stub:                       # a duck-typed engine that records the prompts it is asked to run
        def __init__(self, reply):
            self.reply, self.prompts = reply, []

        def run(self, spec, prompt, root):
            self.prompts.append(prompt)
            return self.reply

    def test_recognise_text_and_error(self):
        from harness import perception
        d = tempfile.mkdtemp()
        p = os.path.join(d, "spec.md")
        with open(p, "w") as f:
            f.write("The API returns JSON with a status field.")
        src = perception.recognize([p])
        self.assertEqual(src[0]["kind"], "text")
        self.assertIn("status field", src[0]["text"])
        bad = perception.recognize([os.path.join(d, "missing.pdf")])      # a bad file is a source with a note
        self.assertEqual(bad[0]["kind"], "error")
        self.assertTrue(bad[0]["note"])

    def test_sources_reach_the_planner(self):
        eng = self.Stub('{"subtasks":[{"id":"s1","role":"heavy","task":"implement it"}]}')
        conductor.decompose("build the client", {"type": "openai"}, eng, "/tmp", {"heavy"},
                            srctext="### api.md\nthe endpoint returns a status field")
        self.assertIn("status field", eng.prompts[0])       # the planner saw the recognised source

    def test_each_subtask_gets_the_relevant_excerpt(self):
        eng = self.Stub("ok")
        subs = [{"id": "s1", "role": "fast", "task": "what status field does the API return", "depends_on": []}]
        res = conductor.run_subtasks(subs, {"fast": {"type": "openai", "url": "http://x", "model": "m"}},
                                     eng, "/tmp", srctext="the API must return JSON with a status field")
        self.assertEqual(res["s1"], "ok")
        self.assertIn("status field", eng.prompts[0])       # the relevant passage reached the subtask


if __name__ == "__main__":
    unittest.main()
