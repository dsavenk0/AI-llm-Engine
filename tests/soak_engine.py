"""Does the engine's memory grow with the number of requests? Mixed traffic on a real model — prose, code
edits, JSON-constrained answers, stop strings, cancelled requests, reasoning, embeddings, sleep and wake —
sampled every 40 requests: process footprint, MLX's active memory, live Python objects, Python heap, threads,
file descriptors. Not part of the default test run (it takes minutes and needs a model).

    .venv/bin/python tests/soak_engine.py [requests] [model path]

Measured on Qwen3.5-0.8B-4bit, 400 requests: footprint 623 – 692 MB with no trend (+0.21 MB per 100, the
dips are the sleep/wake cycles), MLX active 484.9 MB constant, Python objects constant, heap +0.002 MB per
100, 1 thread, 9 descriptors throughout.
"""
import gc, os, sys, threading, time, tracemalloc
sys.path.insert(0, ".")
import mlx.core as mx
from engine import memory
from engine.generate import Engine
P = os.path.expanduser(sys.argv[2] if len(sys.argv) > 2 else "~/.cache/huggingface/hub/mlx-community/Qwen3.5-0.8B-4bit")
eng = Engine(P, vision=False, max_context=8192, max_tokens=64, disk_cache_gb=0, log=lambda *a: None)
code = open("engine/quant.py").read()[:1200]
def ids_for(i):
    kinds = [f"Question {i}: explain item {i % 7} briefly.", "Rewrite this renaming bits to nbits:\n" + code + f"\n# v{i}",
             f"List {i % 5 + 2} colours.", "Continue the story. " * (i % 20 + 1)]
    msgs = [{"role": "system", "content": "You are terse."}, {"role": "user", "content": kinds[i % 4]}]
    text, _, topen = eng.build_prompt(msgs, thinking=(i % 9 == 0))
    return eng.encode(text, [])[0], topen
def one(i):
    ids, topen = ids_for(i)
    kw = {"sampling": {"temperature": 0.7 if i % 3 else 0, "top_p": 0.9}, "max_tokens": 24 + i % 40, "thinking_open": topen}
    if i % 11 == 0: kw["fmt"] = (None, "object")
    if i % 5 == 0: kw["stop"] = ["\n\n"]
    n = [0]
    if i % 6 == 0: kw["should_stop"] = lambda: (n.__setitem__(0, n[0] + 1) or n[0] > 5)      # a cancelled request
    eng.speculative = i % 2 == 0
    try:
        for item in eng.generate(ids, **kw): pass
    except Exception as e:
        if "format" not in str(e): raise
    if i % 13 == 0: eng.embed(["a short text to embed", "another"])
    if i % 50 == 49: eng.sleep(); eng.ensure_awake()                                          # unload and reload
for i in range(40): one(i)                       # warm-up: compiled graphs, allocator pools
gc.collect(); tracemalloc.start()
base = tracemalloc.take_snapshot()
rows = []
N = int(sys.argv[1]) if len(sys.argv) > 1 else 400
for i in range(40, 40 + N):
    one(i)
    if (i - 39) % 40 == 0:
        gc.collect(); mx.clear_cache()
        rows.append((i - 39, memory.process_footprint() / 2**20, mx.get_active_memory() / 2**20, len(gc.get_objects()),
                     tracemalloc.get_traced_memory()[0] / 2**20, threading.active_count(), len(os.listdir("/dev/fd"))))
        print("req %4d  footprint %7.1f MB  mlx active %7.1f MB  py objects %7d  py heap %6.2f MB  threads %d  fds %d" % rows[-1], flush=True)
top = tracemalloc.take_snapshot().compare_to(base, "lineno")[:6]
print("largest Python allocations since the warm-up:")
for t in top: print("   %+8.1f KB  %s" % (t.size_diff / 1024, str(t.traceback[0]).replace(os.getcwd() + "/", "")))
first, last = rows[1], rows[-1]
span = last[0] - first[0]
print("growth per 100 requests (after the first sample): footprint %+.2f MB · mlx active %+.2f MB · py objects %+d · py heap %+.3f MB"
      % ((last[1] - first[1]) / span * 100, (last[2] - first[2]) / span * 100, (last[3] - first[3]) / span * 100, (last[4] - first[4]) / span * 100))
