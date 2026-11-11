"""Expert streaming on a real MoE model, against the same model with every expert resident.

    .venv/bin/python tests/bench_moe.py [model folder] [capacities ...]

For each pool size: resident memory, load time, decode speed on a first pass (cold pools) and a second (warm),
the hit rate and how much was read from disk. Reads go around the page cache (LLM_MOE_DIRECT=1) so the disk is
measured, not RAM that happens to hold the file; run it with LLM_MOE_DIRECT=0 to see the cached case.
Stop the engine first — the model is loaded here, several times, one after another.
"""
import gc
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("LLM_MOE_DIRECT", "1")

import mlx.core as mx                                            # noqa: E402
from engine import memory                                        # noqa: E402
from engine.compat import CompatEngine                           # noqa: E402

PROMPTS = ["Explain how a hash table handles collisions, in one paragraph.",
           "Write a Python function that merges two sorted lists, with a short explanation.",
           "What are the main causes of the French Revolution? Answer in a few sentences."]


def run(path, capacity, tokens=64):
    env = {"LLM_MOE_STREAM": "off"} if capacity == 0 else {"LLM_MOE_STREAM": "on", "LLM_MOE_CACHE": str(capacity),
                                                           "LLM_MOE_USAGE_DIR": "/tmp/moe-bench-none"}
    os.environ.update(env)
    t0 = time.time()
    # the RAM guard is raised to 95 % here: a benchmark on a busy Mac would otherwise be refused before it starts
    e = CompatEngine(path, max_context=4096, max_tokens=tokens, mem_limit=0.95, log=lambda *a: None)
    load_s = time.time() - t0
    rows = []
    for rnd in range(2):
        tps = []
        for p in PROMPTS:
            text, _, _ = e.build_prompt([{"role": "user", "content": p}], thinking=False)
            ids, _ = e.encode(text, [])
            e.cache, e.cache_tokens = None, []
            *_, st = e.generate(ids, sampling={"temperature": 0}, max_tokens=tokens)
            tps.append(st["decode_tps"])
        rows.append(sum(tps) / len(tps))
    s = e.moe_stats()
    out = {"capacity": capacity, "resident_gb": e.model_bytes / 2**30, "load_s": load_s, "cold_tps": rows[0],
           "warm_tps": rows[1], "hit_rate": s and s["hit_rate"], "gb_read": s and s["gb_read"],
           "peak_gb": mx.get_peak_memory() / 2**30}
    if e.moe:
        e.moe.close()
    del e
    gc.collect()
    mx.clear_cache()
    mx.reset_peak_memory()
    return out


def main():
    args = sys.argv[1:]
    path = os.path.expanduser(args[0]) if args and not args[0].isdigit() else os.path.expanduser(
        "~/.cache/huggingface/hub/mlx-community/OLMoE-1B-7B-0125-Instruct-4bit")
    caps = [int(a) for a in args if a.isdigit()] or [0, 64, 32, 16, 8]
    print(f"{os.path.basename(path)} · direct reads: {os.environ['LLM_MOE_DIRECT']} · RAM in use {memory.used_fraction():.0%}")
    print(f"{'experts/layer':>13} {'resident GB':>11} {'load s':>7} {'cold tok/s':>10} {'warm tok/s':>10} {'hit rate':>8} {'GB read':>8}")
    for c in caps:
        r = run(path, c)
        print(f"{('all' if c == 0 else c):>13} {r['resident_gb']:>11.2f} {r['load_s']:>7.1f} {r['cold_tps']:>10.1f} "
              f"{r['warm_tps']:>10.1f} {('-' if r['hit_rate'] is None else format(r['hit_rate'], '.1%')):>8} "
              f"{('-' if r['gb_read'] is None else format(r['gb_read'], '.2f')):>8}", flush=True)


if __name__ == "__main__":
    main()
