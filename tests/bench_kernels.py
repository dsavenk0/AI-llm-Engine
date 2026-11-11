"""Every kernel of engine/kernels against its MLX reference, on the same inputs.

    .venv/bin/python tests/bench_kernels.py                    # attention decode, 512 … 32768 rows
    .venv/bin/python tests/bench_kernels.py --heads 16 --kv-heads 8 --dim 128 --ctx 4096 16384
    .venv/bin/python tests/bench_kernels.py --json out.json

A replacement kernel is only worth keeping where this says so, so the method matters more than the code:

  * `--layers` distinct caches (default 6, like the model) are walked in turn, so the rows come from
    DRAM the way a real decode step reads them. One cache called in a loop sits in the system cache and
    flatters whichever implementation is measured second;
  * the implementations are interleaved round by round, and the figure is the median of the rounds —
    a fanless Mac drops its clocks under load, and back-to-back blocks measure the thermal state;
  * each round is one eval of all the layers' outputs: timing a layer at a time adds a 0.2-0.3 ms
    synchronisation per call, more than the thing being measured at a short context;
  * GB/s is K and V bytes actually filled, over the time — comparable with the bus figure of
    tests/profile.py.

The error column is the largest absolute difference from the reference on these inputs (float16 outputs
of magnitude ~1), there to catch a wrong kernel, not to certify one: tests/test_kernels.py does that.
"""
import argparse
import json
import os
import statistics
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import mlx.core as mx                                            # noqa: E402
from engine import kernels                                       # noqa: E402

STEP = 256


def make_caches(layers, hkv, d, n, seed=0):
    """`layers` caches filled to `n` rows inside a buffer over-allocated the way KVCache grows it."""
    mx.random.seed(seed)
    cap = ((n + STEP - 1) // STEP) * STEP
    out = []
    for _ in range(layers):
        k = (mx.random.normal((1, hkv, cap, d)) * 0.5).astype(mx.float16)
        v = mx.random.normal((1, hkv, cap, d)).astype(mx.float16)
        out.append((k, v))
    mx.eval(out)
    return out


def bench_attn_decode(args):
    op = "attn_decode"
    impls = [name for name, ok in kernels.implementations(op).items() if ok]
    scale = args.dim ** -0.5
    rows = []
    for n in args.ctx:
        caches = make_caches(args.layers, args.kv_heads, args.dim, n)
        mx.random.seed(1)
        qs = [mx.random.normal((1, args.heads, 1, args.dim)).astype(mx.float16) for _ in caches]
        mx.eval(qs)
        fns = {name: kernels._ops[op][name][0] for name in impls}

        def run(fn):
            return [fn(q, k, v, n, scale) for q, (k, v) in zip(qs, caches)]

        ref = run(fns[kernels.REFERENCE])
        mx.eval(ref)
        err = {}
        for name, fn in fns.items():
            out = run(fn)
            if any(o is None for o in out):
                err[name] = None
                continue
            mx.eval(out)                                         # also the warm-up of this shape
            err[name] = max(mx.abs(o.astype(mx.float32) - r.astype(mx.float32)).max().item()
                            for o, r in zip(out, ref))
        times = {name: [] for name in impls if err[name] is not None}
        for _ in range(args.rounds):
            for name in times:
                t = time.perf_counter()
                mx.eval(run(fns[name]))
                times[name].append(time.perf_counter() - t)
        nbytes = args.layers * 2 * args.kv_heads * n * args.dim * 2
        base = statistics.median(times[kernels.REFERENCE])
        for name, ts in times.items():
            med = statistics.median(ts)
            rows.append({"op": op, "impl": name, "rows": n, "ms_per_layer": med / args.layers * 1e3,
                         "best_ms_per_layer": min(ts) / args.layers * 1e3, "gb_s": nbytes / med / 1e9,
                         "vs_mlx": base / med, "max_abs_err": err[name]})
        del caches, qs, ref
        mx.clear_cache()
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ctx", type=int, nargs="+", default=[512, 2048, 8192, 32768])
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--kv-heads", type=int, default=2)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--rounds", type=int, default=30)
    ap.add_argument("--json", help="also write the rows to this file")
    args = ap.parse_args()
    rows = bench_attn_decode(args)
    print(f"attn_decode · {args.heads} heads / {args.kv_heads} kv heads / dim {args.dim} · "
          f"{args.layers} caches · median of {args.rounds} interleaved rounds · MLX {mx.__version__}")
    print(f"{'rows':>7} {'impl':>6} {'ms/layer':>9} {'best':>8} {'GB/s':>7} {'vs mlx':>7} {'max err':>9}")
    for r in rows:
        print(f"{r['rows']:>7} {r['impl']:>6} {r['ms_per_layer']:>9.3f} {r['best_ms_per_layer']:>8.3f} "
              f"{r['gb_s']:>7.1f} {r['vs_mlx']:>6.2f}x {r['max_abs_err']:>9.2e}")
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"mlx": mx.__version__, "args": vars(args), "rows": rows}, f, indent=1)


if __name__ == "__main__":
    main()
