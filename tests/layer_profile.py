"""Where a decode step's time sits, layer by layer, inside the real forward.

Timing one layer at a time means an eval per layer, and that serializes what the real loop overlaps — the
mistake that made several micro-benchmarks around this question useless, because the per-call
synchronization (0.17-0.3 ms) was larger than the thing being measured. This times the forward *truncated*
after N layers instead: one eval at the end, so everything inside stays pipelined, and layer N's cost is
the difference between N+1 layers and N. The cache is saved and restored around every run, so each
measurement starts from the same state.

    .venv/bin/python tests/layer_profile.py [model path] [context ...]

Measured on an M5 (16 GB, macOS 26.4.1, MLX 0.32.3), mlx-community/Qwen3.5-2B-MLX-4bit, at 512 and 32768:

    attention layers (6)     2.47 ms  ->  20.80 ms     +18.33 ms
    GatedDeltaNet (18)       7.38 ms  ->   8.01 ms      +0.64 ms
    whole step incl. head   14.17 ms  ->  33.26 ms

So the halving of decode between a short context and 32K sits entirely in the six layers that hold a KV
cache; the eighteen recurrent layers do not grow at all, their state being a fixed size. That is why this
hybrid degrades more gently with context than a pure transformer of the same size would.

What it is not: the attention layers read their cache at roughly 22 GB/s in situ, while the identical
mx.fast.scaled_dot_product_attention call on identical shapes reaches 86-88 GB/s on its own. Five
explanations were measured and ruled out — slicing the cache out of its over-allocated buffer (2-4 %), other
resident memory (no effect with up to 2.2 GB of ballast), the write-then-read of one lazy graph forcing a
copy (1.0x the read alone), hidden copies between steps (the arrays keep their identity), and the allocator
cache limit (5-6 %, and the threshold does not scale between models). What is left is not a defect but the
efficiency of the chain: the whole step reaches 77 GB/s at a short context and 51 GB/s at 32K, while every
operation in it reaches 86-88 GB/s alone. Narrowing that further wants Metal's own profiler, not another
Python timing loop.
"""
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import mlx.core as mx                                            # noqa: E402
from engine.generate import Engine                               # noqa: E402
from engine.model import Positions                               # noqa: E402

REPS = 12


def build(path, ctx):
    """An engine with its cache filled to `ctx` tokens, the way a long conversation leaves it."""
    eng = Engine(path, vision=False, max_context=ctx + 512, max_tokens=32, log=lambda *a: None)
    cache = eng.model.make_cache()
    grown = 0
    while grown < ctx:
        n = min(512, ctx - grown)
        mx.eval(eng.model(mx.array([list(range(1000, 1000 + n))]), cache, Positions()))
        grown += n
    return eng, cache


def prefix_times(eng, cache):
    """Cumulative forward time for the first N layers, N = 0..all, plus the whole step for comparison."""
    model = eng.model
    layers = model.language_model.model.layers
    saves = [(c.snapshot() if hasattr(c, "snapshot") else c.offset) for c in cache]

    def restore():
        for c, s in zip(cache, saves):
            if hasattr(c, "restore"):
                c.restore(s)
            else:
                c.offset = s

    tok = mx.array([[1000]])

    def run(n):
        h = model.embed(tok)
        pos = Positions()
        for layer, c in list(zip(layers, cache))[:n]:
            h = layer(h, model.ops, c, pos, None)
        return h

    def best(fn):
        restore()
        mx.eval(fn())                                            # warm this shape
        ts = []
        for _ in range(REPS):
            restore()
            t = time.perf_counter()
            mx.eval(fn())
            ts.append(time.perf_counter() - t)
        return min(ts)

    out = [best(lambda n=n: run(n)) for n in range(len(layers) + 1)]
    full = best(lambda: model(tok, cache, Positions(), last_only=False))
    restore()
    return out, full


def main(path, contexts):
    res = {}
    for ctx in contexts:
        eng, cache = build(path, ctx)
        kinds = ["attn" if not l.is_linear else "gdn" for l in eng.model.language_model.model.layers]
        times, full = prefix_times(eng, cache)
        kv = sum(a.nbytes for c in cache for a in (c.arrays if hasattr(c, "arrays") else c.state)
                 if a is not None)
        res[ctx] = {"kinds": kinds, "per": [(times[i + 1] - times[i]) * 1e3 for i in range(len(times) - 1)],
                    "embed": times[0] * 1e3, "full": full * 1e3, "kv": kv, "weights": eng.model_bytes}
        print(f"ctx {ctx}: embed {times[0] * 1e3:.3f} ms · all layers {times[-1] * 1e3:.2f} ms · "
              f"whole step {full * 1e3:.2f} ms · cache {kv / 2**20:.0f} MB", flush=True)
        del cache, eng
        mx.clear_cache()

    a, b = contexts[0], contexts[-1]
    kinds = res[a]["kinds"]
    print(f"\n{'layer':>5} {'kind':>5} {a:>11} {b:>11} {'growth':>10}")
    for i, k in enumerate(kinds):
        x, y = res[a]["per"][i], res[b]["per"][i]
        print(f"{i:>5} {k:>5} {x:>10.3f}m {y:>10.3f}m {y - x:>+9.3f}m")
    for name, want in (("attn layers", "attn"), ("gdn layers", "gdn")):
        idx = [i for i, k in enumerate(kinds) if k == want]
        x = sum(res[a]["per"][i] for i in idx)
        y = sum(res[b]["per"][i] for i in idx)
        print(f"{name:>11} {x:>10.2f}m {y:>10.2f}m {y - x:>+9.2f}m   ({len(idx)} layers)")
    print()
    for ctx in contexts:
        r = res[ctx]
        gb = (r["weights"] + r["kv"]) / 1e9
        print(f"ctx {ctx:>6}: {gb:5.2f} GB read per step / {r['full']:5.2f} ms = {gb / r['full'] * 1000:3.0f} GB/s")


if __name__ == "__main__":
    args = sys.argv[1:]
    model = os.path.expanduser(args[0]) if args and not args[0].isdigit() else os.path.expanduser(
        "~/.cache/huggingface/hub/mlx-community/Qwen3.5-2B-MLX-4bit")
    sizes = [int(a) for a in args if a.isdigit()] or [512, 32768]
    main(model, sizes)
