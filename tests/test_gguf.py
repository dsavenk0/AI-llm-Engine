"""GGUF support: the header reader, file discovery (folders, LM Studio, Ollama), repo file picking, the
llama.cpp installer, and the gateway in front of llama-server — with a fake llama-server (no model, no GPU).

    .venv/bin/python -m unittest tests.test_gguf -v
"""
import io
import json
import os
import socket
import stat
import struct
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from engine import gguf  # noqa: E402


def _s(x):
    b = x.encode()
    return struct.pack("<Q", len(b)) + b


def make_gguf(path, arch="llama", name="Tiny", ftype=15, ctx=4096, template="{{ messages }}", extra_tokens=3):
    kv = [_s("general.architecture") + struct.pack("<I", 8) + _s(arch),
          _s("general.name") + struct.pack("<I", 8) + _s(name),
          _s("general.file_type") + struct.pack("<I", 4) + struct.pack("<I", ftype),
          _s(f"{arch}.context_length") + struct.pack("<I", 4) + struct.pack("<I", ctx),
          _s("tokenizer.ggml.tokens") + struct.pack("<I", 9) + struct.pack("<I", 8) + struct.pack("<Q", extra_tokens)
          + b"".join(_s("t" * (i + 1)) for i in range(extra_tokens)),
          _s("tokenizer.ggml.scores") + struct.pack("<I", 9) + struct.pack("<I", 6) + struct.pack("<Q", 2)
          + struct.pack("<ff", 0.0, 1.0)]
    if template:
        kv.append(_s("tokenizer.chat_template") + struct.pack("<I", 8) + _s(template))
    with open(path, "wb") as f:
        f.write(b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0) + struct.pack("<Q", len(kv)) + b"".join(kv))
    return path


class TestHeader(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()

    def test_the_fields_the_engine_needs(self):
        p = make_gguf(os.path.join(self.dir, "m-Q4_K_M.gguf"), arch="qwen2", name="Qwen Tiny", ftype=15, ctx=32768)
        h = gguf.read_header(p)
        self.assertEqual((h["architecture"], h["name"], h["quant"], h["context_length"]),
                         ("qwen2", "Qwen Tiny", "Q4_K_M", 32768))
        self.assertEqual(h["chat_template"], "{{ messages }}")

    def test_arrays_of_tokens_and_scores_are_skipped_over(self):
        p = make_gguf(os.path.join(self.dir, "m.gguf"), template="the template comes after the arrays", extra_tokens=500)
        self.assertEqual(gguf.read_header(p)["chat_template"], "the template comes after the arrays")

    def test_not_a_gguf_or_truncated(self):
        bad = os.path.join(self.dir, "bad.gguf")
        with open(bad, "wb") as f:
            f.write(b"NOPE" + b"\0" * 64)
        with self.assertRaises(gguf.GGUFError):
            gguf.read_header(bad)
        good = make_gguf(os.path.join(self.dir, "g.gguf"))
        cut = os.path.join(self.dir, "cut.gguf")
        with open(good, "rb") as f, open(cut, "wb") as g:
            g.write(f.read()[:60])
        with self.assertRaises(gguf.GGUFError):
            gguf.read_header(cut)

    def test_quant_from_the_file_name_when_the_header_has_none(self):
        self.assertEqual(gguf.guess_quant("llama-3.2-1b-instruct-Q6_K.gguf"), "Q6_K")
        self.assertEqual(gguf.guess_quant("model-IQ4_XS.gguf"), "IQ4_XS")
        self.assertEqual(gguf.guess_quant("mystery.gguf"), "?")


class TestPickFiles(unittest.TestCase):
    def sib(self, *names):
        return [{"rfilename": n, "size": 100 + i} for i, n in enumerate(names)]

    def test_the_default_is_the_balanced_quantization(self):
        files, chosen, offered = gguf.pick_files(self.sib("m-Q8_0.gguf", "m-Q4_K_M.gguf", "m-Q2_K.gguf", "README.md"))
        self.assertEqual((chosen, [f["rfilename"] for f in files]), ("Q4_K_M", ["m-Q4_K_M.gguf"]))
        self.assertEqual(offered[0], "Q4_K_M")

    def test_a_named_quantization_and_a_wrong_name(self):
        files, chosen, _ = gguf.pick_files(self.sib("m-Q8_0.gguf", "m-Q4_K_M.gguf"), "q8_0")
        self.assertEqual(chosen, "Q8_0")
        with self.assertRaises(gguf.GGUFError) as ctx:
            gguf.pick_files(self.sib("m-Q8_0.gguf"), "Q3_K_S")
        self.assertIn("Q8_0", str(ctx.exception))

    def test_split_models_come_whole_and_a_projector_comes_along(self):
        sib = self.sib("big-Q4_K_M-00001-of-00002.gguf", "big-Q4_K_M-00002-of-00002.gguf", "big-Q8_0.gguf",
                       "mmproj-model-f32.gguf", "mmproj-model-F16.gguf")
        files, chosen, _ = gguf.pick_files(sib)
        names = [f["rfilename"] for f in files]
        self.assertEqual(chosen, "Q4_K_M")
        self.assertEqual(names[:2], ["big-Q4_K_M-00001-of-00002.gguf", "big-Q4_K_M-00002-of-00002.gguf"])
        self.assertEqual(names[2], "mmproj-model-F16.gguf")

    def test_repo_suffix(self):
        self.assertEqual(gguf.split_repo_quant("org/name:Q4_K_M"), ("org/name", "Q4_K_M"))
        self.assertEqual(gguf.split_repo_quant("org/name"), ("org/name", None))
        self.assertEqual(gguf.split_repo_quant("ms:org/name:Q8_0"), ("ms:org/name", "Q8_0"))


class TestOllamaStore(unittest.TestCase):
    def test_a_manifest_points_at_the_gguf_blob(self):
        root = tempfile.mkdtemp()
        blob = os.path.join(root, "blobs", "sha256-abc123")
        os.makedirs(os.path.dirname(blob))
        make_gguf(blob)
        man = os.path.join(root, "manifests", "registry.ollama.ai", "library", "llama3.2")
        os.makedirs(man)
        with open(os.path.join(man, "latest"), "w") as f:
            json.dump({"layers": [{"mediaType": "application/vnd.ollama.image.model", "digest": "sha256:abc123"},
                                  {"mediaType": "application/vnd.ollama.image.template", "digest": "sha256:zzz"}]}, f)
        self.assertEqual(gguf.ollama_models(root), {"llama3.2:latest": blob})

    def test_a_missing_store_is_empty(self):
        self.assertEqual(gguf.ollama_models(os.path.join(tempfile.mkdtemp(), "nope")), {})


class TestInstaller(unittest.TestCase):
    def archive(self, members):
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as t:
            for name, data in members.items():
                ti = tarfile.TarInfo(name)
                ti.size = len(data)
                ti.mode = 0o755
                t.addfile(ti, io.BytesIO(data))
        return buf.getvalue()

    def opener(self, tarball):
        releases = [{"tag_name": "b1", "assets": [
            {"name": "llama-b1-bin-macos-arm64.tar.gz", "browser_download_url": "https://x/a.tar.gz"},
            {"name": "llama-b1-bin-macos-arm64-kleidiai.tar.gz", "browser_download_url": "https://x/k.tar.gz"}]}]

        def open_(req, timeout=0):
            url = req.full_url if hasattr(req, "full_url") else req
            body = json.dumps(releases).encode() if "api.github" in url else tarball
            return io.BytesIO(body)
        return open_

    def test_it_picks_the_plain_macos_build_and_puts_the_server_in_the_data_folder(self):
        tag, name, url = gguf.pick_asset([{"tag_name": "b1", "assets": [
            {"name": "llama-b1-bin-macos-arm64-kleidiai.tar.gz", "browser_download_url": "k"},
            {"name": "llama-b1-bin-macos-arm64.tar.gz", "browser_download_url": "a"},
            {"name": "llama-b1-bin-ubuntu-x64.tar.gz", "browser_download_url": "u"}]}])
        self.assertEqual((tag, url), ("b1", "a"))
        data = tempfile.mkdtemp()
        tb = self.archive({"build/bin/llama-server": b"#!/bin/sh\necho hi\n", "build/bin/libggml.dylib": b"x"})
        with mock.patch("platform.machine", lambda: "arm64"):
            path = gguf.install(data, log=lambda s: None, opener=self.opener(tb))
        self.assertEqual(path, os.path.join(data, "llama.cpp", "llama-server"))
        self.assertTrue(os.access(path, os.X_OK))
        self.assertTrue(os.path.exists(os.path.join(data, "llama.cpp", "libggml.dylib")))     # its libraries come along
        self.assertEqual(gguf.server_path(data), path)

    def test_an_archive_that_escapes_is_refused(self):
        data = tempfile.mkdtemp()
        tb = self.archive({"../evil": b"x", "bin/llama-server": b"x"})
        with mock.patch("platform.machine", lambda: "arm64"), self.assertRaises(gguf.GGUFError):
            gguf.install(data, log=lambda s: None, opener=self.opener(tb))
        self.assertFalse(os.path.exists(os.path.join(os.path.dirname(data), "evil")))

    def test_an_archive_without_the_server_is_refused(self):
        data = tempfile.mkdtemp()
        with mock.patch("platform.machine", lambda: "arm64"), self.assertRaises(gguf.GGUFError):
            gguf.install(data, log=lambda s: None, opener=self.opener(self.archive({"readme.txt": b"x"})))


FAKE_SERVER = r'''#!/usr/bin/env python3
import json, sys, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
args = sys.argv[1:]
port = int(args[args.index("--port") + 1])
open(sys.argv[0] + ".args", "w").write(" ".join(args))
LOAD = float(__import__("os").environ.get("FAKE_LOAD", "0"))
time.sleep(LOAD)
class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_GET(self):
        body = b'{"status":"ok"}'
        self.send_response(200); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        open(sys.argv[0] + ".last", "w").write(json.dumps(req))
        if req.get("stream"):
            self.send_response(200); self.send_header("Content-Type", "text/event-stream"); self.end_headers()
            for piece in ("Hel", "lo", " wor", "ld"):
                self.wfile.write(("data: " + json.dumps({"choices": [{"delta": {"content": piece}}]}) + "\n\n").encode())
                self.wfile.flush(); time.sleep(0.02)
            self.wfile.write(b"data: [DONE]\n\n")
        else:
            body = json.dumps({"choices": [{"message": {"role": "assistant", "content": "Hello world"}, "finish_reason": "stop"}],
                               "usage": {"prompt_tokens": 7, "completion_tokens": 2}}).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
'''


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Gateway:
    def __init__(self, tmp, *extra, env=None):
        self.tmp = tmp
        self.model = make_gguf(os.path.join(tmp, "tiny-Q4_K_M.gguf"), arch="qwen2", name="Tiny Qwen", ctx=8192)
        self.fake = os.path.join(tmp, "llama-server")
        with open(self.fake, "w") as f:
            f.write(FAKE_SERVER)
        os.chmod(self.fake, os.stat(self.fake).st_mode | stat.S_IEXEC)
        self.port = free_port()
        self.base = f"http://127.0.0.1:{self.port}"
        self.log = open(os.path.join(tmp, "gw.log"), "w+")
        self.proc = subprocess.Popen(
            [sys.executable, "-B", "-m", "engine.gguf_server", "--model", self.model, "--port", str(self.port),
             "--llama-server", self.fake, "--ram-gb", "16", *extra],
            cwd=ROOT, stdout=self.log, stderr=subprocess.STDOUT, env=dict(os.environ, **(env or {})))
        deadline = time.time() + 60
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError("gateway exited:\n" + open(self.log.name).read())
            try:
                self.get("/health")
                return
            except OSError:
                time.sleep(0.2)
        raise RuntimeError("gateway did not come up")

    def stop(self):
        self.proc.terminate()
        try:
            self.proc.wait(10)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        self.log.close()

    def req(self, method, path, body=None, headers=None):
        r = urllib.request.Request(self.base + path, method=method, data=json.dumps(body).encode() if body is not None else None,
                                   headers={"Content-Type": "application/json", **(headers or {})})
        try:
            with urllib.request.urlopen(r, timeout=30) as resp:
                return resp.status, dict(resp.headers), resp.read()
        except urllib.error.HTTPError as e:
            try:
                return e.code, dict(e.headers), e.read()
            finally:
                e.close()

    def get(self, path, headers=None):
        return json.loads(self.req("GET", path, headers=headers)[2])

    def chat(self, **kw):
        body = {"model": "tiny", "messages": [{"role": "user", "content": "hi"}], **kw}
        code, h, raw = self.req("POST", "/v1/chat/completions", body)
        return code, raw

    def last_request(self):
        return json.load(open(self.fake + ".last"))


class TestGateway(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def gw(self, *extra, env=None):
        g = Gateway(self.tmp, *extra, env=env)
        self.addCleanup(g.stop)
        return g

    def test_health_and_models_look_like_the_engine(self):
        g = self.gw()
        h = g.get("/health")
        self.assertEqual((h["status"], h["backend"], h["state"], h["model"]), ("ok", "llama.cpp", "awake", "tiny-q4_k_m"))
        self.assertEqual(h["context"], 8192)                       # the model's own, under the RAM cap
        self.assertEqual(h["format"], {"architecture": "qwen2", "quant": "Q4_K_M"})
        m = g.get("/v1/models")
        self.assertEqual(m["data"][0]["id"], "tiny-q4_k_m")

    def test_llama_server_gets_the_flags_that_matter(self):
        g = self.gw()
        g.chat()
        args = open(g.fake + ".args").read().split()
        for flag in ("-m", "-c", "-ngl", "-np", "--jinja"):
            self.assertIn(flag, args)
        self.assertEqual(args[args.index("-c") + 1], "8192")
        self.assertEqual(args[args.index("--host") + 1], "127.0.0.1")        # never exposed directly

    def test_prompt_lookup_speculation_is_on_when_the_server_knows_it(self):
        with open(os.path.join(self.tmp, "llama-server"), "w") as f:
            f.write(FAKE_SERVER)
        # a fake that answers --help with the option list of a recent build
        helper = os.path.join(self.tmp, "helped")
        os.makedirs(helper)
        g = self.gw()
        g.chat()
        args = open(g.fake + ".args").read().split()
        self.assertNotIn("--spec-type", args)            # this fake does not list the option: not passed

    def test_the_speculation_flag_follows_the_help_text_and_the_switch(self):
        import importlib
        sys.path.insert(0, ROOT)
        from engine import gguf_server
        importlib.reload(gguf_server)
        new = os.path.join(self.tmp, "new-server")
        with open(new, "w") as f:
            f.write("#!/bin/sh\necho '--spec-type none,draft-simple,ngram-simple,ngram-mod'\n")
        old = os.path.join(self.tmp, "old-server")
        with open(old, "w") as f:
            f.write("#!/bin/sh\necho 'usage: llama-server'\n")
        for p in (new, old):
            os.chmod(p, 0o755)
        on = gguf_server.Inner(new, "m.gguf", 4096, "m").command(1234)
        self.assertEqual(on[on.index("--spec-type") + 1], "ngram-mod")
        self.assertNotIn("--spec-type", gguf_server.Inner(new, "m.gguf", 4096, "m", speculative=False).command(1234))
        self.assertNotIn("--spec-type", gguf_server.Inner(old, "m.gguf", 4096, "m").command(1234))

    def test_a_chat_is_forwarded_and_max_tokens_is_clamped(self):
        g = self.gw("--max-tokens", "256")
        code, raw = g.chat(max_tokens=100000)
        self.assertEqual(code, 200)
        self.assertEqual(json.loads(raw)["choices"][0]["message"]["content"], "Hello world")
        self.assertEqual(g.last_request()["max_tokens"], 256)
        self.assertEqual(g.last_request()["model"], "tiny-q4_k_m")

    def test_bad_bodies_are_refused_cleanly(self):
        import http.client
        g = self.gw()
        conn = http.client.HTTPConnection("127.0.0.1", g.port, timeout=10)
        conn.putrequest("POST", "/v1/chat/completions")
        conn.putheader("Content-Length", str(10 ** 11))           # announced, never sent
        conn.endheaders()
        self.assertEqual(conn.getresponse().status, 413)
        conn.close()
        self.assertEqual(g.req("POST", "/v1/chat/completions", [1, 2, 3])[0], 400)           # not an object
        self.assertEqual(g.req("POST", "/v1/chat/completions",
                               {"messages": [{"role": "user", "content": "x"}], "max_tokens": "lots"})[0], 400)
        code, raw = g.chat(max_tokens=0)                                                      # 0 = "no preference"
        self.assertEqual(code, 200)

    def test_streaming_is_passed_through_event_by_event(self):
        g = self.gw()
        r = urllib.request.Request(g.base + "/v1/chat/completions", method="POST",
                                   data=json.dumps({"messages": [{"role": "user", "content": "x"}], "stream": True}).encode(),
                                   headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(r, timeout=30) as resp:
            self.assertIn("text/event-stream", resp.headers["Content-Type"])
            text = resp.read().decode()
        pieces = [json.loads(l[6:])["choices"][0]["delta"]["content"] for l in text.split("\n")
                  if l.startswith("data: ") and l != "data: [DONE]"]
        self.assertEqual("".join(pieces), "Hello world")
        self.assertTrue(text.rstrip().endswith("data: [DONE]"))

    def test_api_key_and_fallback(self):
        g = self.gw("--api-key", "secret", "--fallback-model", "off")
        self.assertEqual(g.req("GET", "/health")[0], 401)
        auth = {"Authorization": "Bearer secret"}
        code, _, raw = g.req("POST", "/v1/chat/completions", {"model": "other", "messages": []}, auth)
        self.assertEqual((code, json.loads(raw)["error"]["code"]), (404, "model_not_found"))
        for ok in ("tiny-q4_k_m", "TINY-Q4_K_M", "org/tiny-q4_k_m"):
            code, _, _ = g.req("POST", "/v1/chat/completions", {"model": ok, "messages": [{"role": "user", "content": "x"}]}, auth)
            self.assertEqual(code, 200, ok)

    def test_it_sleeps_when_idle_and_wakes_on_demand(self):
        g = self.gw("--idle-sleep", "0.03")                         # ~2 s
        self.assertEqual(g.get("/health")["state"], "awake")
        deadline = time.time() + 40
        state = "awake"
        while time.time() < deadline and state != "sleeping":
            time.sleep(1)
            state = g.get("/health")["state"]
        self.assertEqual(state, "sleeping")
        self.assertEqual(g.chat()[0], 200)                          # the next request loads it again
        self.assertEqual(g.get("/health")["state"], "awake")

    def test_a_server_that_cannot_start_is_a_clean_error(self):
        broken = os.path.join(self.tmp, "broken-server")
        with open(broken, "w") as f:
            f.write("#!/bin/sh\necho 'unknown model architecture' >&2\nexit 7\n")
        os.chmod(broken, 0o755)
        model = make_gguf(os.path.join(self.tmp, "b.gguf"))
        p = subprocess.run([sys.executable, "-B", "-m", "engine.gguf_server", "--model", model, "--port", str(free_port()),
                            "--llama-server", broken, "--ram-gb", "16"], cwd=ROOT, capture_output=True, text=True, timeout=60)
        self.assertEqual(p.returncode, 3)
        self.assertIn("llama-server exited", p.stderr)

    def test_a_missing_llama_cpp_says_how_to_install(self):
        model = make_gguf(os.path.join(self.tmp, "c.gguf"))
        env = dict(os.environ, LLM_HOME=self.tmp, PATH="/usr/bin:/bin")
        env.pop("LLM_LLAMA_SERVER", None)
        p = subprocess.run([sys.executable, "-B", "-m", "engine.gguf_server", "--model", model, "--port", str(free_port())],
                           cwd=ROOT, capture_output=True, text=True, timeout=60, env=env)
        self.assertEqual(p.returncode, 2)
        self.assertIn("setup --gguf", p.stderr)


class TestLlmIntegration(unittest.TestCase):
    def setUp(self):
        import llm
        self.llm = llm
        self.dir = tempfile.mkdtemp()
        self.cfg = dict(llm.load_cfg(), dirs=[self.dir], models={}, model="")
        self.m = make_gguf(os.path.join(self.dir, "org", "x", "tiny-Q4_K_M.gguf"), arch="llama", name="Tiny Llama",
                           template="{% if enable_thinking %}<think>{% endif %}") if os.makedirs(
            os.path.join(self.dir, "org", "x")) is None else None

    def test_scan_finds_the_file_with_a_gguf_route(self):
        found = [m for m in self.llm.scan_gguf(self.cfg)]
        self.assertEqual(len(found), 1)
        m = found[0]
        self.assertEqual((m["route"], m["format"], m["type"], m["supported"]), ("gguf", "GGUF Q4_K_M", "llama", True))
        self.assertTrue(m["thinking"])
        self.assertIn("Tiny Llama", m["name"])

    def test_projectors_and_other_split_parts_are_not_models(self):
        d = os.path.join(self.dir, "org", "x")
        make_gguf(os.path.join(d, "mmproj-tiny-F16.gguf"))
        make_gguf(os.path.join(d, "big-Q4_K_M-00001-of-00002.gguf"))
        make_gguf(os.path.join(d, "big-Q4_K_M-00002-of-00002.gguf"))
        names = sorted(os.path.basename(m["path"]) for m in self.llm.scan_gguf(self.cfg))
        self.assertEqual(names, ["big-Q4_K_M-00001-of-00002.gguf", "tiny-Q4_K_M.gguf"])
        self.assertTrue(next(m for m in self.llm.scan_gguf(self.cfg) if m["path"] == self.m)["vision"])   # the projector is next to it

    def test_the_engine_command_runs_the_gateway(self):
        m = self.llm.inspect_gguf(self.m)
        cmd = self.llm.engine_cmd(self.cfg, m)
        self.assertEqual(cmd[1:3], ["-m", "engine.gguf_server"])
        self.assertEqual(cmd[cmd.index("--model") + 1], self.m)
        self.assertNotIn("engine.server", cmd)

    def test_without_llama_cpp_a_gguf_model_is_blocked_with_the_command_to_run(self):
        m = self.llm.inspect_gguf(self.m)
        with mock.patch.object(self.llm.gguf, "server_path", lambda d=None: None), \
                mock.patch.object(self.llm, "setup_gguf", lambda: False):
            why = self.llm.route_blocker(self.cfg, m, False)
        self.assertIn("setup --gguf", why)

    def test_a_gguf_selects_and_deletes_like_any_model(self):
        self.cfg["model"] = self.m
        self.assertEqual(self.llm.current_model(self.cfg)["route"], "gguf")
        with mock.patch.object(self.llm, "save_cfg", lambda c: None), \
                mock.patch.object(self.llm, "read_pid", lambda p: None):      # not the engine that runs on this Mac
            ok = self.llm.delete_model(self.cfg, self.llm.inspect_gguf(self.m), assume_yes=True)
        self.assertTrue(ok)
        self.assertFalse(os.path.exists(self.m))


class TestHeaderAndSupportDetails(unittest.TestCase):
    """The reader's limits and every reason the own decoder gives for refusing a file (real GGUF bytes, via the
    synthetic writer used for the decoder tests)."""

    @classmethod
    def setUpClass(cls):
        from tests import gguf_synth
        cls.S = gguf_synth
        cls.dir = tempfile.mkdtemp()

    def write(self, name, **kw):
        path = os.path.join(self.dir, name)
        self.S.write(path, **kw)
        return path

    def test_the_quantization_names_follow_llama_cpp_ftype_ids(self):
        want = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 7: "Q8_0", 8: "Q5_0", 9: "Q5_1", 10: "Q2_K", 11: "Q3_K_S",
                12: "Q3_K_M", 13: "Q3_K_L", 14: "Q4_K_S", 15: "Q4_K_M", 16: "Q5_K_S", 17: "Q5_K_M", 18: "Q6_K",
                19: "IQ2_XXS", 20: "IQ2_XS", 21: "Q2_K_S", 22: "IQ3_XS", 23: "IQ3_XXS", 24: "IQ1_S", 25: "IQ4_NL",
                26: "IQ3_S", 27: "IQ3_M", 28: "IQ2_S", 29: "IQ2_M", 30: "IQ4_XS", 31: "IQ1_M", 32: "BF16"}
        self.assertEqual(gguf.FTYPE, want)

    def test_quant_is_guessed_from_a_file_name(self):
        for name, q in (("m-Q4_K_M.gguf", "Q4_K_M"), ("m-q5_0.gguf", "Q5_0"), ("m-IQ3_XS.gguf", "IQ3_XS"),
                        ("m-BF16.gguf", "BF16"), ("m-f16.gguf", "F16"), ("m-Q8_0.gguf", "Q8_0"),
                        ("m-Q6_K.gguf", "Q6_K"), ("model.gguf", "?")):
            self.assertEqual(gguf.guess_quant(name), q, name)

    def test_projectors_and_plain_paths_are_told_apart(self):
        self.assertTrue(gguf.is_projector("/x/mmproj-F16.gguf"))
        self.assertTrue(gguf.is_projector("/x/MMPROJ-model.gguf"))
        self.assertFalse(gguf.is_projector("/x/model.gguf"))
        f = self.write("plain.gguf")
        self.assertTrue(gguf.is_gguf_path(f))
        self.assertFalse(gguf.is_gguf_path(self.dir))
        self.assertFalse(gguf.is_gguf_path(os.path.join(self.dir, "none.gguf")))
        other = os.path.join(self.dir, "x.bin")
        open(other, "wb").close()
        self.assertFalse(gguf.is_gguf_path(other))

    def test_a_truncated_or_foreign_file_is_a_clean_error(self):
        f = self.write("ok.gguf")
        data = open(f, "rb").read()
        bad = os.path.join(self.dir, "cut.gguf")
        with open(bad, "wb") as fh:
            fh.write(data[:40])
        for fn in (gguf.read_header_full, lambda p: gguf.read_array(p, "tokenizer.ggml.tokens")):
            with self.assertRaises(gguf.GGUFError):
                fn(bad)
        notgguf = os.path.join(self.dir, "not.gguf")
        with open(notgguf, "wb") as fh:
            fh.write(b"NOPE" + bytes(64))
        with self.assertRaises(gguf.GGUFError):
            gguf.read_header_full(notgguf)

    def test_arrays_come_back_whole_and_a_missing_key_is_none(self):
        f = self.write("arr.gguf")
        toks = gguf.read_array(f, "tokenizer.ggml.tokens")
        self.assertEqual(len(toks), len(self.S.vocab()[0]))
        self.assertEqual(toks[:3], self.S.vocab()[0][:3])
        self.assertEqual(gguf.read_array(f, "tokenizer.ggml.token_type")[:2], self.S.vocab()[1][:2])
        self.assertIsNone(gguf.read_array(f, "no.such.key"))
        with self.assertRaises(gguf.GGUFError):
            gguf.read_array(f, "tokenizer.ggml.tokens", limit=3)             # larger than allowed

    def test_directory_of_tensors_lists_types_and_names(self):
        f = self.write("t.gguf", quant="Q8_0")
        types = gguf.tensor_types(f)
        self.assertIn(8, types)                                              # Q8_0 matrices
        self.assertIn(0, types)                                              # F32 norms
        names = gguf.tensor_names(f)
        self.assertIn("token_embd.weight", names)
        self.assertIn("blk.0.attn_q.weight", names)
        self.assertIn("output.weight", names)

    def test_native_support_names_why_not(self):
        self.assertEqual(gguf.native_support(self.write("fine.gguf")), (True, ""))
        ok, why = gguf.native_support(self.write("moe.gguf", extra_meta={"llama.expert_count": 8}))
        self.assertEqual((ok, why), (False, "mixture of experts"))
        ok, why = gguf.native_support(self.write("spm.gguf", extra_meta={"tokenizer.ggml.model": "llama"}))
        self.assertEqual((ok, why), (False, "tokenizer 'llama'"))
        ok, why = gguf.native_support(self.write("pre.gguf", extra_meta={"tokenizer.ggml.pre": "weird"}))
        self.assertEqual((ok, why), (False, "pre-tokenizer 'weird'"))
        ok, why = gguf.native_support(self.write("yarn.gguf", extra_meta={"llama.rope.scaling.type": "yarn"}))
        self.assertEqual((ok, why), (False, "rope scaling 'yarn'"))
        ok, why = gguf.native_support(self.write("sw.gguf", extra_meta={"llama.attention.sliding_window": 4096}))
        self.assertEqual((ok, why), (False, "sliding-window attention"))
        ok, why = gguf.native_support(self.write("l3.gguf", extra_meta={"llama.rope.scaling.type": "llama3"}))
        self.assertEqual((ok, why), (False, "llama3 rope scaling without its frequency table"))
        ok, why = gguf.native_support(self.write("l3ok.gguf", rope_factors=[1.0] * (self.S.HEAD_DIM // 2)))
        self.assertEqual((ok, why), (True, ""))
        ok, why = gguf.native_support(self.write("q.gguf", tensor_type_override=("blk.0.ffn_up.weight", 10)))
        self.assertEqual((ok, why), (False, "quantization type(s) [10]"))
        ok, why = gguf.native_support(self.write("gemma.gguf", arch="llama", extra_meta={"general.architecture": "gemma2"}))
        self.assertEqual((ok, why), (False, "architecture 'gemma2'"))
        ok, why = gguf.native_support(os.path.join(self.dir, "missing.gguf"))
        self.assertFalse(ok)
        self.assertTrue(why)


if __name__ == "__main__":
    unittest.main()
