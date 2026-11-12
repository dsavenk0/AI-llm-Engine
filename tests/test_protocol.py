"""The engine protocol (engine/protocol.py): every backend the server can start is held to it, and the
server is asked over HTTP for the things the protocol exists to guarantee.

    .venv/bin/python -m unittest tests.test_protocol -v
"""
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request

from engine import protocol
from tests import gguf_synth as G
from tests.synth import CHATML, make_bundle
from tests.test_server_native import ROOT, free_port


def quiet(*_a):
    pass


class FakeMemory:
    def state(self): ...
    def reclaim(self, need=0, upto="restore_points"): ...
    def relieve(self): ...
    def report(self): ...


class Whole:
    """A stand-in with every required member."""
    backend = "fake"

    def __init__(self):
        for name in protocol.ATTRIBUTES:
            if name != "backend":
                setattr(self, name, None)
        self.memory = FakeMemory()

    def build_prompt(self, messages, tools=None, thinking=None, effort=None, extra_kwargs=None): ...
    def encode(self, text, images): ...
    def generate(self, ids, media=None, max_tokens=None, sampling=None, stop=None, should_stop=None, seed=None,
                 thinking_open=False, fmt=None): ...
    def sleep(self): ...
    def ensure_awake(self): ...
    def persist_prefix(self): ...
    def memory_breakdown(self): ...


class TestCheck(unittest.TestCase):
    def test_a_whole_backend_passes_and_is_returned(self):
        e = Whole()
        self.assertEqual(protocol.missing(e), [])
        self.assertIs(protocol.check(e), e)
        self.assertIsInstance(e, protocol.EngineBackend)

    def test_everything_owed_is_named_at_once(self):
        e = Whole()
        del e.store
        e.sleep = None
        with self.assertRaises(TypeError) as ctx:
            protocol.check(e)
        self.assertIn("missing store, sleep", str(ctx.exception))

    def test_a_method_that_cannot_take_what_the_server_passes_is_caught(self):
        # the defect this check was written for: generate() without `fmt` answered every chat with a 500
        class Old(Whole):
            def generate(self, ids, media=None, max_tokens=None, sampling=None, stop=None, should_stop=None,
                         seed=None, thinking_open=False): ...
        self.assertEqual(protocol.missing(Old()), ["generate(fmt)"])

        class Open(Whole):
            def generate(self, ids, **kw): ...
        self.assertEqual(protocol.missing(Open()), [])                           # **kw takes anything

    def test_the_memory_manager_is_held_to_its_contract(self):
        e = Whole()
        del FakeMemory.relieve
        try:
            self.assertEqual(protocol.missing(e), ["memory.relieve"])
        finally:
            FakeMemory.relieve = lambda self: None

    def test_a_runtime_missing_a_seam_method_is_named(self):
        from engine.session import DecodeLoop
        from engine.runtime import MlxHybridRuntime

        class Loop(Whole, DecodeLoop):
            pass
        e = Loop()
        for name in protocol.LOOP_HOST:
            if not hasattr(e, name):
                setattr(e, name, None)
        e.rt = object.__new__(MlxHybridRuntime)
        self.assertEqual(protocol.missing(e), [])
        e.rt = type("Partial", (), {n: None for n in protocol.RUNTIME if n != "rewind"})()
        self.assertEqual(protocol.missing(e), ["rt.rewind"])
        e.generate_batch = lambda reqs, on_piece=None: []           # batching needs the batch half too
        self.assertEqual(protocol.missing(e), ["rt.rewind"] + [f"rt.{n}" for n in protocol.RUNTIME_BATCH])

    def test_every_runtime_implements_the_seam(self):
        from engine.compat import MlxLmRuntime
        from engine.runtime import MlxHybridRuntime
        for cls in (MlxHybridRuntime, MlxLmRuntime):
            self.assertEqual([n for n in protocol.RUNTIME + protocol.RUNTIME_BATCH
                              if not callable(getattr(cls, n, None))], [], cls.__name__)

    def test_the_loop_documents_every_runtime_call_it_makes(self):
        # every rt.<name> engine/session.py and engine/batch.py call is in RUNTIME / RUNTIME_BATCH
        import re
        calls = set()
        for f in ("session.py", "batch.py"):
            calls |= set(re.findall(r"\brt\.([a-z_]+)\(", open(os.path.join(ROOT, "engine", f)).read()))
        self.assertEqual(calls - set(protocol.RUNTIME) - set(protocol.RUNTIME_BATCH), set())

    def test_optional_capabilities_are_asked_for_by_name(self):
        e = Whole()
        self.assertFalse(protocol.can(e, "embed"))
        e.embed = lambda texts, normalize=True: ([], 0)
        self.assertTrue(protocol.can(e, "embed"))
        with self.assertRaises(ValueError):
            protocol.can(e, "generate")                       # required: asking is a mistake

    def test_the_lists_are_consistent(self):
        self.assertEqual(set(protocol.EngineBackend.__annotations__), set(protocol.ATTRIBUTES))
        for m in protocol.METHODS:
            self.assertTrue(callable(getattr(protocol.EngineBackend, m)), m)
        self.assertFalse(set(protocol.REQUIRED) & set(protocol.OPTIONAL))
        self.assertTrue(set(protocol.PARAMETERS) <= set(protocol.METHODS))


class TestBackends(unittest.TestCase):
    """Each engine class, constructed on a synthetic model, owes the protocol nothing."""

    def test_the_native_engine(self):
        from engine.generate import Engine
        d = tempfile.mkdtemp()
        make_bundle(d, CHATML, seed=31)
        e = Engine(d, vision=False, max_context=256, max_tokens=8, disk_cache_gb=0, log=quiet)
        self.assertEqual(protocol.missing(e), [])
        self.assertEqual(e.backend, "native")
        self.assertTrue(protocol.can(e, "embed") and protocol.can(e, "generate_batch"))
        self.assertFalse(protocol.can(e, "parser_markers"))

    def test_the_own_gguf_decoder(self):
        from engine.gguf_engine import GGUFEngine
        path = os.path.join(tempfile.mkdtemp(), "llama-Q8_0.gguf")
        G.write(path, seed=5, quant="Q8_0", arch="llama", moe=False)
        e = GGUFEngine(path, max_context=256, max_tokens=8, log=quiet)
        self.assertEqual(protocol.missing(e), [])
        self.assertEqual(e.backend, "gguf-mlx")
        self.assertFalse(protocol.can(e, "generate_batch"))
        self.assertTrue(protocol.can(e, "parser_markers"))

    def test_the_mlx_lm_engine_class(self):
        # no mlx-lm model is needed to hold the class to the method half of the protocol
        from engine.compat import CompatEngine
        for name in protocol.METHODS:
            self.assertTrue(callable(getattr(CompatEngine, name, None)), name)
        import inspect
        for name, params in protocol.PARAMETERS.items():
            have = inspect.signature(getattr(CompatEngine, name)).parameters
            self.assertEqual([p for p in params if p not in have], [], name)

    def test_every_generate_ends_with_the_same_statistics(self):
        from engine.generate import Engine
        from engine.gguf_engine import GGUFEngine
        d = tempfile.mkdtemp()
        make_bundle(d, CHATML, seed=31)
        path = os.path.join(tempfile.mkdtemp(), "llama-Q8_0.gguf")
        G.write(path, seed=5, quant="Q8_0", arch="llama", moe=False)
        for e in (Engine(d, vision=False, max_context=256, max_tokens=8, disk_cache_gb=0, log=quiet),
                  GGUFEngine(path, max_context=256, max_tokens=8, log=quiet)):
            *_, st = e.generate([3, 4, 5, 6], max_tokens=4, sampling={"temperature": 0})
            self.assertEqual([k for k in protocol.STATS if k not in st], [], type(e).__name__)


class GGUFServer:
    def __init__(self, *extra):
        self.path = os.path.join(tempfile.mkdtemp(), "llama-Q8_0.gguf")
        G.write(self.path, seed=5, quant="Q8_0", arch="llama", moe=False)
        self.port = free_port()
        self.log = tempfile.NamedTemporaryFile("w+", suffix=".log", delete=False)
        self.proc = subprocess.Popen(
            [sys.executable, "-B", "-m", "engine.server", "--model", self.path, "--port", str(self.port),
             "--context", "512", "--max-tokens", "16", "--backend", "gguf", *extra],
            cwd=ROOT, stdout=self.log, stderr=subprocess.STDOUT)
        deadline = time.time() + 120
        try:
            while time.time() < deadline:
                if self.proc.poll() is not None:
                    raise RuntimeError("server exited:\n" + open(self.log.name).read()[-2000:])
                try:
                    urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=1).read()
                    return
                except urllib.error.HTTPError as e:      # answering, but with an error: say which
                    raise RuntimeError(f"/health answered {e.code}:\n" + open(self.log.name).read()[-1500:]) from None
                except OSError:
                    time.sleep(0.3)
            raise RuntimeError("server did not come up")
        except BaseException:
            self.stop()                                    # a server that failed to start must not be left running
            raise

    def post(self, path, body):
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}{path}", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            try:
                return e.code, json.loads(e.read() or b"{}")
            finally:
                e.close()

    def stop(self):
        self.proc.terminate()
        try:
            self.proc.wait(10)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        self.log.close()


class TestGGUFOverHTTP(unittest.TestCase):
    """The own GGUF decoder behind the real server. Nothing tested this route over HTTP before, and every
    chat on it answered 500: the server passed `fmt`, which CompatEngine.generate did not take."""

    @classmethod
    def setUpClass(cls):
        cls.srv = GGUFServer("--batch-window", "20")

    @classmethod
    def tearDownClass(cls):
        cls.srv.stop()

    def chat(self, **kw):
        return self.srv.post("/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}],
                                                      "max_tokens": 6, "temperature": 0, **kw})

    def test_a_chat_is_answered(self):
        code, body = self.chat()
        self.assertEqual(code, 200, body)
        self.assertEqual(body["choices"][0]["message"]["role"], "assistant")
        self.assertGreater(body["usage"]["completion_tokens"], 0)

    def test_health_names_the_backend(self):
        h = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{self.srv.port}/health", timeout=5).read())
        self.assertEqual(h["backend"], "gguf-mlx")
        self.assertEqual([k for k in protocol.HEALTH if k not in h], [])          # the same contract as native
        self.assertEqual(h["engine_protocol"], protocol.VERSION)

    def test_a_json_answer_is_constrained_on_this_backend_too(self):
        # random weights never close a JSON object in 16 tokens, so the honest outcome here is the server's
        # own "format incomplete" refusal — which only exists when the constraint was really applied; without
        # it the same request returns 200 with noise
        code, body = self.chat(response_format={"type": "json_object"}, max_tokens=16)
        if code == 200:
            self.assertTrue(body["choices"][0]["message"]["content"].lstrip().startswith("{"), body)
        else:
            self.assertEqual(code, 400, body)
            self.assertIn("before the requested format was complete", body["error"]["message"])

    def test_overlapping_requests_are_served_in_turn_when_the_backend_cannot_batch(self):
        # --batch-window is on, and this backend has no generate_batch: merging used to be attempted anyway
        results = []
        ts = [threading.Thread(target=lambda: results.append(self.chat()[0])) for _ in range(3)]
        [t.start() for t in ts]
        [t.join(60) for t in ts]
        self.assertEqual(results, [200, 200, 200])

    def test_embeddings_say_they_are_not_available_here(self):
        code, body = self.srv.post("/v1/embeddings", {"input": "x"})
        self.assertIn(code, (400, 501))
        self.assertIn("error", body)


if __name__ == "__main__":
    unittest.main()
