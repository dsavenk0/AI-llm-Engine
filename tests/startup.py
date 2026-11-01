"""What the first second costs: server start, time to first token, shutdown.

The decode and prefill numbers in tests/perf.py say nothing about the part a user actually waits through —
launching the engine and getting the first character back. This measures that end to end, on a real model,
through the HTTP path the app and every editor integration use.

    .venv/bin/python tests/startup.py [model path ...]

Reported per model:

    boot        process spawn → the port accepts a connection
    ready       → /health answers 200 (the model is loaded and the worker is up)
    ttft cold   first streamed token of the first request (kernels not warmed by a real prompt yet)
    ttft warm   first streamed token of a later request with a different prompt
    stop        SIGTERM → the process is gone (it drains, then persists the prefix cache)

Note on `boot`: the server binds its port only after the model has loaded (engine/server.py waits on READY
before constructing the HTTP server), so `boot` and `ready` are two views of the same wait — a client that
connects earlier gets connection-refused rather than an answer. The measurement below shows them equal to
the hundredth of a second, which is that waiting proved: the gap is one /health round trip, not a second
phase of loading.

Measured on an M5 (16 GB, macOS 26.4.1, MLX 0.32.3), machine otherwise idle:

    model                         boot   ready  ttft cold  ttft warm    stop
    Qwen3.5-0.8B-8bit            0.57s   0.58s      0.05s      0.08s   0.09s
    Qwen3.5-2B-MLX-4bit          0.50s   0.50s      0.08s      0.13s   0.09s
    Qwen3.5-4B-MLX-4bit          0.97s   0.97s      0.14s      0.23s   0.20s
"""
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

DEFAULT = [os.path.expanduser("~/.cache/huggingface/hub/mlx-community/Qwen3.5-0.8B-8bit"),
           os.path.expanduser("~/.cache/huggingface/hub/mlx-community/Qwen3.5-2B-MLX-4bit")]
PROMPTS = ["Say hello in one short sentence.", "Name one colour, nothing else."]


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def connects(port):
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.2):
            return True
    except OSError:
        return False


def ttft(port, prompt):
    """Seconds from sending the request to the first streamed character of the answer."""
    body = json.dumps({"model": "bench", "stream": True, "max_tokens": 16, "temperature": 0,
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=120) as r:
        for line in r:
            if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
                continue
            delta = json.loads(line[6:])["choices"][0].get("delta") or {}
            if delta.get("content") or delta.get("reasoning_content"):
                return time.time() - t0
    return float("nan")


def measure(model):
    port = free_port()
    log = open(os.devnull, "w")
    t0 = time.time()
    proc = subprocess.Popen([sys.executable, "-B", "-m", "engine.server", "--model", model, "--port", str(port),
                             "--name", "bench", "--disk-cache-gb", "0"],
                            cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    try:
        while not connects(port):
            if proc.poll() is not None:
                raise RuntimeError("the server exited during start")
            time.sleep(0.01)
        boot = time.time() - t0
        while True:
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5).read()
                break
            except OSError:
                time.sleep(0.01)
        ready = time.time() - t0
        cold = ttft(port, PROMPTS[0])
        warm = ttft(port, PROMPTS[1])
        t1 = time.time()
        proc.send_signal(signal.SIGTERM)
        proc.wait(60)
        stop = time.time() - t1
    finally:
        if proc.poll() is None:
            proc.kill()
        log.close()
    return boot, ready, cold, warm, stop


models = sys.argv[1:] or [m for m in DEFAULT if os.path.isdir(m)]
if not models:
    sys.exit("no model to measure: pass a path")
print(f"{'model':<26} {'boot':>7} {'ready':>7} {'ttft cold':>10} {'ttft warm':>10} {'stop':>7}")
for m in models:
    b, r, c, w, s = measure(m)
    print(f"{os.path.basename(m):<26} {b:>6.2f}s {r:>6.2f}s {c:>9.2f}s {w:>9.2f}s {s:>6.2f}s")
