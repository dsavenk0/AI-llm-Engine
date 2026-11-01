"""Decode speed as the context grows — where the KV cache starts costing more than the weights.

tests/perf.py measures a decode step on a 512-token context only (it trims the cache back after every step),
so how decode scales with context was never measured. It matters because a decode step reads two things off
the memory bus: the weights, once, and the whole KV cache, once. The weights are fixed; the cache grows with
every token, so past some context length the cache is the larger read and the engine is no longer limited by
model size at all.

This measures the real step time at several context lengths and, beside it, the real size of the cache at
that length (summed from the cache arrays themselves, not from a formula), so the two reads can be compared
against this machine's measured bandwidth.

    .venv/bin/python tests/context_decode.py [model path] [max context]

`floor` is (weights + cache) / measured bandwidth: what the bus alone owes at that context. `of floor` is how
much of that the engine actually achieves — above 100 % would mean the step is faster than streaming both
reads once, which the KV cache can manage when it is small enough to stay in cache hierarchies.

Measured on an M5 (16 GB, macOS 26.4.1, MLX 0.32.3), mlx-community/Qwen3.5-2B-MLX-4bit:

     context     cache  bytes/token      step    tok/s    floor  of floor
         512     0.03G        1.02G    13.03ms    76.7    8.86ms       68%
        2048     0.06G        1.05G    13.76ms    72.7    9.09ms       66%
        8192     0.16G        1.15G    16.38ms    61.1   10.01ms       61%
       16384     0.31G        1.29G    20.03ms    49.9   11.24ms       56%
       32768     0.59G        1.57G    27.11ms    36.9   13.69ms       51%

Decode halves between a short context and 32K. Only part of that is the extra bytes: the step grows twice as
fast as the read does, so efficiency falls from 68 % to 51 %. Attention is not where it goes — on these
shapes mx.fast.scaled_dot_product_attention reads the cache at 89 GB/s against 93 GB/s for a plain
contiguous sum of the same bytes, slicing the cache out of its over-allocated buffer costs 2-4 %, and the
cache arrays keep their identity across steps (so there is no hidden copy). What the rest is has not been
established yet; attributing it needs per-layer timing inside the real forward.

Worth knowing when reading these numbers: this is a hybrid model, and only 6 of its 24 layers hold a KV
cache at all — the other 18 are GatedDeltaNet, whose state is a fixed size. A pure transformer of the same
size would degrade considerably faster.
"""
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import mlx.core as mx                                           # noqa: E402
from engine.generate import Engine                              # noqa: E402
from engine.model import Positions                              # noqa: E402

SIZES = (512, 2048, 8192, 16384, 32768)


def bandwidth_gbs():
    """Measured sequential read bandwidth, so the floor uses this machine's real number."""
    n = int(1.0 * 2**30 / 2)
    x = mx.random.normal((n,)).astype(mx.float16)
    mx.eval(x)
    for _ in range(3):
        mx.eval(mx.sum(x))
    ts = []
    for _ in range(5):
        t = time.perf_counter()
        mx.eval(mx.sum(x))
        ts.append(time.perf_counter() - t)
    return x.nbytes / min(ts) / 1e9


def cache_bytes(cache):
    """What the cache really occupies right now, counted off the arrays the layers hold."""
    total = 0
    for c in cache:
        for a in (c.arrays if hasattr(c, "arrays") else c.state):
            if a is not None:
                total += a.nbytes
    return total


def main(path, limit):
    bw = bandwidth_gbs()
    sizes = [s for s in SIZES if s <= limit]
    eng = Engine(path, vision=False, max_context=max(sizes) + 512, max_tokens=64, ptq1=True, log=lambda *a: None)
    model, wb = eng.model, eng.model_bytes

    cache = model.make_cache()
    mx.eval(model(mx.array([list(range(1000, 1512))]), cache))    # a context to start from, and a warm-up

    def step(n):
        """One decode step, pipelined the way the engine's loop runs it."""
        out = None
        t = time.perf_counter()
        for _ in range(n):
            out = model(mx.array([[1000]]), cache, Positions(), last_only=False)
            mx.async_eval(out)
        mx.eval(out)
        return (time.perf_counter() - t) / n

    print(f"\n=== {os.path.basename(path.rstrip('/'))} ===")
    print(f"weights {wb / 2**30:.2f} GB · measured bandwidth {bw:.0f} GB/s")
    print(f"{'context':>8} {'cache':>9} {'bytes/token':>12} {'step':>9} {'tok/s':>8} {'floor':>8} {'of floor':>9}")
    grown = 512
    for size in sizes:
        while grown < size:                                       # grow the cache to this context, cheaply
            n = min(512, size - grown)
            mx.eval(model(mx.array([list(range(1000, 1000 + n))]), cache, Positions()))
            grown += n
        step(3)                                                   # warm-up at this size
        dt = min(step(10) for _ in range(3))
        kv = cache_bytes(cache)
        floor = (wb + kv) / (bw * 1e9)
        print(f"{size:>8} {kv / 2**30:>8.2f}G {(wb + kv) / 2**30:>11.2f}G {dt * 1e3:>8.2f}ms "
              f"{1 / dt:>7.1f} {floor * 1e3:>7.2f}ms {floor / dt * 100:>8.0f}%")


if __name__ == "__main__":
    model = os.path.expanduser(sys.argv[1]) if len(sys.argv) > 1 else os.path.expanduser(
        "~/.cache/huggingface/hub/mlx-community/Qwen3.5-2B-MLX-4bit")
    main(model, int(sys.argv[2]) if len(sys.argv) > 2 else 32768)
