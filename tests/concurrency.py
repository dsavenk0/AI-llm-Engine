"""What several clients on one loaded model actually cost.

Generation is serialised on one worker thread (MLX streams are thread-local), so concurrent
requests queue. This measures the consequence: per-request latency and aggregate throughput as
the number of clients grows. It is the evidence for whether merging requests into a single
forward pass (engine/batch.py, proven bit-exact but not wired) is worth the hot-path work:
with perfect batching the weights are read once for the whole batch, so aggregate throughput
would scale with N instead of staying flat.

Start the engine first (./llm start), then:

    .venv/bin/python -m tests.concurrency            # N = 1, 2, 4
    .venv/bin/python -m tests.concurrency 1 2 4 8
"""
import json
import os
import statistics
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAX_TOKENS = 64
PROMPT = "Write a short paragraph about a stone garden."


def config():
    home = os.environ.get("LLM_HOME") or ROOT
    try:
        with open(os.path.join(home, ".llm", "config.json")) as f:
            cfg = json.load(f)
    except (OSError, ValueError):
        cfg = {}
    return cfg.get("port", 8090), cfg.get("api_key", "")


def ask(port, key):
    body = {"model": "x", "stream": False, "temperature": 0, "max_tokens": MAX_TOKENS,
            "messages": [{"role": "user", "content": PROMPT}]}
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = "Bearer " + key
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                 data=json.dumps(body).encode(), headers=headers)
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=600) as r:
        out = json.loads(r.read())
    return time.perf_counter() - t0, (out.get("usage") or {}).get("completion_tokens", 0)


def run(port, key, n):
    with ThreadPoolExecutor(max_workers=n) as pool:
        pool.submit(ask, port, key).result()                 # warm the prompt cache / wake the model
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=n) as pool:
        res = [f.result() for f in [pool.submit(ask, port, key) for _ in range(n)]]
    wall = time.perf_counter() - t0
    lat = [r[0] for r in res]
    toks = sum(r[1] for r in res)
    return wall, statistics.mean(lat), max(lat), toks / wall if wall else 0


def main(counts):
    port, key = config()
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5).read()
    except OSError:
        sys.exit(f"no engine answering on port {port} — start it with ./llm start")
    print(f"{MAX_TOKENS} tokens per request, temperature 0, port {port}\n")
    print(f"{'clients':>7}  {'wall':>7}  {'latency mean':>13}  {'latency max':>12}  {'aggregate':>12}")
    base = None
    for n in counts:
        wall, mean, worst, tps = run(port, key, n)
        base = base or tps
        print(f"{n:>7}  {wall:6.2f}s  {mean:12.2f}s  {worst:11.2f}s  {tps:8.1f} tok/s"
              f"   ({tps / base:.2f}x of N=1)")


if __name__ == "__main__":
    main([int(a) for a in sys.argv[1:]] or [1, 2, 4])
