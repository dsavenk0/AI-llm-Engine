"""Performance suite with a saved baseline — run before/after every optimization.

    .venv/bin/python -m tests.perf                 # measure and compare to tests/perf_baseline.json
    .venv/bin/python -m tests.perf --save          # measure and store as the new baseline
    .venv/bin/python -m tests.perf --tolerance 0.1 # fail on >10 % regressions (default 0.15)
    ./llm perf [--save]

Uses the model from ENGINE_TEST_MODEL or the one selected in ./llm. Stop the
server first (two copies of a 27B model will not fit in 16 GB).

A fanless MacBook lowers clocks after a few seconds of full load, so each
metric is measured after a warm-up and reported as the best of several runs;
compare numbers taken on the same machine in the same state (on power, idle).
"""
import argparse
import json
import os
import platform
import statistics
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
BASELINE = os.path.join(ROOT, "tests", "perf_baseline.json")

# name -> (unit, direction): +1 higher is better, -1 lower is better
METRICS = {
    "load_s": ("s", -1),
    "decode_step_ms": ("ms", -1),
    "verify16_cost": ("× step", -1),
    "prefill_tps": ("tok/s", +1),
    "prose_tps": ("tok/s", +1),
    "code_edit_tps": ("tok/s", +1),
    "code_edit_accept": ("ratio", +1),
    "peak_gb": ("GB", -1),
    "sys_ram_peak": ("fraction", -1),
}


def model_path():
    p = os.environ.get("ENGINE_TEST_MODEL")
    if p:
        return os.path.expanduser(p)
    try:
        with open(os.path.join(os.environ.get("LLM_HOME", ROOT), ".llm", "config.json")) as f:
            return json.load(f)["model"]
    except (OSError, ValueError, KeyError):
        sys.exit("set ENGINE_TEST_MODEL or select a model with ./llm use")


SYS_PEAK = [0.0]


def generate(eng, msgs, max_tokens, speculative):
    eng.speculative = speculative
    eng.cache = None
    text, _, _ = eng.build_prompt(msgs, thinking=False)
    ids, _ = eng.encode(text, [])
    for item in eng.generate(ids, sampling={"temperature": 0}, max_tokens=max_tokens):
        if isinstance(item, dict):
            SYS_PEAK[0] = max(SYS_PEAK[0], item.get("ram_used") or 0)
            return item


def measure():
    import mlx.core as mx
    from engine import memory
    from engine.generate import Engine, MemoryLimitError

    used = memory.used_fraction()
    print(f"system RAM in use before loading: {used:.0%}")
    if used > 0.45:
        print("warning: other apps already use a lot of memory; numbers may be throttled or skipped")
    from engine.model import Positions

    t0 = time.time()
    eng = Engine(model_path(), vision=False, max_context=8192, max_tokens=512, log=lambda *a: None)
    res = {"load_s": round(time.time() - t0, 2)}
    model = eng.model

    # decode step and the cost of a 16-token verify pass, on a 512-token context
    cache = model.make_cache()
    mx.eval(model(mx.array([list(range(1000, 1512))]), cache))

    def step(T, n):
        ts = []
        for _ in range(n):
            t = time.time()
            mx.eval(model(mx.array([[1000] * T]), cache, Positions(), last_only=False, verify=T > 1))
            ts.append(time.time() - t)
            model.commit(cache, 0) if T > 1 else None
            for c in cache:
                if hasattr(c, "trim_to"):
                    c.trim_to(512)
        return statistics.median(ts) * 1000
    step(1, 3), step(16, 2)
    d1 = step(1, 15)
    res["decode_step_ms"] = round(d1, 1)
    res["verify16_cost"] = round(step(16, 5) / d1, 2)

    def guarded(name, fn):
        try:
            fn()
        except MemoryLimitError as e:
            res[name] = None
            print(f"{name}: skipped — {e}")

    # prefill throughput on a ~2K-token prompt
    para = ("A Japanese garden balances stone, water, moss and trees; every element is a symbol. " * 120)
    msgs = [{"role": "user", "content": para + "\nSummarize in one sentence."}]
    def prefill_run():
        generate(eng, msgs, 4, False)
        res["prefill_tps"] = generate(eng, msgs, 4, False)["prefill_tps"]
    guarded("prefill_tps", prefill_run)

    prose = [{"role": "user", "content": "Write a detailed story about a bonsai garden in Kyoto."}]
    def prose_run():
        generate(eng, prose, 32, True)
        res["prose_tps"] = max(generate(eng, prose, 200, True)["decode_tps"] for _ in range(2))
    guarded("prose_tps", prose_run)

    with open(os.path.join(ROOT, "engine", "tokenizer.py")) as f:
        src = f.read()
    edit = [{"role": "user", "content": "Rewrite this file in full, renaming StreamDecoder to IncrementalDecoder. "
                                        "Output only the code.\n\n" + src}]
    def edit_run():
        best = max((generate(eng, edit, 300, True) for _ in range(2)), key=lambda s: s["decode_tps"])
        res["code_edit_tps"] = best["decode_tps"]
        res["code_edit_accept"] = best["draft_accept"] or 0.0
    guarded("code_edit_tps", edit_run)
    res["peak_gb"] = round(mx.get_peak_memory() / 2**30, 2)
    res["sys_ram_peak"] = round(SYS_PEAK[0], 3)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--save", action="store_true", help="store results as the new baseline")
    ap.add_argument("--tolerance", type=float, default=0.15, help="allowed relative regression (default 0.15)")
    args = ap.parse_args()

    res = measure()
    base = {}
    if os.path.exists(BASELINE):
        with open(BASELINE) as f:
            base = json.load(f).get("metrics", {})

    failed = []
    print(f"\n{'metric':18s} {'value':>10s} {'baseline':>10s} {'change':>9s}")
    for k, (unit, sign) in METRICS.items():
        v, b = res.get(k), base.get(k)
        change, flag = "", ""
        if isinstance(v, (int, float)) and isinstance(b, (int, float)) and b:
            rel = (v - b) / b
            change = f"{rel * 100:+.1f}%"
            if sign * rel < -args.tolerance:
                flag = "  ✗ regression"
                failed.append(k)
            elif sign * rel > args.tolerance:
                flag = "  ✓ better"
        print(f"{k:18s} {v!s:>10s} {b if b is not None else '—'!s:>10s} {change:>9s} {unit}{flag}")

    if args.save:
        with open(BASELINE, "w") as f:
            json.dump({"machine": platform.machine(), "platform": platform.platform(),
                       "date": time.strftime("%Y-%m-%d"), "metrics": res}, f, indent=2)
        print(f"\nbaseline saved: {BASELINE}")
    if failed and not args.save:
        print(f"\nregressions beyond {args.tolerance:.0%}: {', '.join(failed)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
