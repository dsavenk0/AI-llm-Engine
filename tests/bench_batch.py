"""Does one merged pass beat N requests served one at a time?

The motivation is in tests/concurrency.py: with requests served one at a time the aggregate throughput is
flat (four clients buy 1.01x), because most of a decode step is reading the weights and every request reads
them again. This measures the other side — Engine.generate_batch against Engine.generate run N times.

Both sides use the same engine, the same prompts and greedy sampling, with speculation and the prefix cache
off, so the only difference is whether the weights are read once per step for the whole batch or once per
step per request.

    .venv/bin/python tests/bench_batch.py [model path]          # NEW=48 tokens per request by default

Measured on an M5 (16 GB, macOS 26.4.1, MLX 0.32.3) with mlx-community/Qwen3.5-2B-MLX-4bit, 48 tokens,
two runs an hour apart that agreed within 0.07x (the second is shown):

        N   serial s  serial tok/s   batch s  batch tok/s  speedup
        1       0.68          70.2      0.71         67.4    0.96x
        2       1.37          69.9      0.81        118.6    1.70x
        4       2.75          69.9      1.09        175.4    2.51x
        8       5.63          68.2      2.27        169.1    2.48x

Four rows is where it stops paying: eight rows produce the same ~170 tok/s, so the pass has left the
weight-read floor behind and is bounded by the kernels themselves.
"""
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

MODEL = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser(
    "~/.cache/huggingface/hub/mlx-community/Qwen3.5-2B-MLX-4bit")
NEW = int(os.environ.get("NEW", "48"))

PROMPTS = [
    "Explain in two sentences why reading model weights dominates a decode step.",
    "List three ways to cut the latency of a single token on Apple Silicon.",
    "Write a short paragraph about why a 16 GB laptop needs a memory ceiling.",
    "Describe what a KV cache stores and why it grows with the context.",
    "Name two reasons speculative decoding helps on code edits but not on prose.",
    "Summarise what a Hadamard rotation does to a weight matrix before quantization.",
    "In two sentences, what is the difference between prefill and decode?",
    "Why does a batch of requests read the weights fewer times in total?",
]

import mlx.core as mx                                            # noqa: E402
from engine.generate import Engine                               # noqa: E402

eng = Engine(MODEL, vision=False, speculative=False, log=lambda *_: None)
eng.cache = None


def chat_ids(text):
    return eng.tok.encode(eng.tok.render([{"role": "user", "content": text}]))


def serial(ids_list):
    t0 = time.time()
    produced = 0
    for ids in ids_list:
        eng.cache = None
        for item in eng.generate(ids, max_tokens=NEW, sampling={"temperature": 0}):
            if isinstance(item, dict):
                produced += item["completion_tokens"]
    return time.time() - t0, produced


def batched(ids_list):
    reqs = [{"ids": ids, "max_tokens": NEW, "sampling": {"temperature": 0}} for ids in ids_list]
    t0 = time.time()
    stats = eng.generate_batch(reqs)
    return time.time() - t0, sum(s["completion_tokens"] for s in stats)


print(f"model {os.path.basename(MODEL)}  weights {eng.model_bytes / 2**30:.2f} GB  {NEW} new tokens/request")
print(f"{'N':>2}  {'serial s':>9} {'serial tok/s':>13}  {'batch s':>8} {'batch tok/s':>12}  {'speedup':>7}")

ids_all = [chat_ids(p) for p in PROMPTS]
for n in (1, 2, 4, 8):
    ids_list = ids_all[:n]
    mx.clear_cache()
    serial(ids_list[:1])                                         # warm-up, not timed
    st, stok = serial(ids_list)
    mx.clear_cache()
    bt, btok = batched(ids_list)
    print(f"{n:>2}  {st:>9.2f} {stok / st:>13.1f}  {bt:>8.2f} {btok / bt:>12.1f}  {st / bt:>6.2f}x"
          f"   ({stok} vs {btok} tokens)")
