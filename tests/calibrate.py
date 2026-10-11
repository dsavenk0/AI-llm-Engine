"""Empirically find the largest prompt this machine can actually prefill for a model,
instead of only trusting Engine's analytic auto-context formula (RAM x limit - weights,
divided by kv_per_token). That formula is on paper — it can be too optimistic, e.g. the
Metal wired-limit case: the working set macOS actually hands out can be smaller than the
formula assumes, and the auto-computed context then doesn't really fit.

    .venv/bin/python -m tests.calibrate                   # binary search, print the result
    .venv/bin/python -m tests.calibrate --apply            # also set it as the context (not "auto")
    .venv/bin/python -m tests.calibrate --model A --model B
    ./llm calibrate [--apply] [--model PATH ...]

Each candidate size runs in its own subprocess: a fresh model load plus one real prefill
attempt. A hard crash or OOM at one size only writes that candidate off as "too big" (same
as a clean MemoryLimitError) instead of taking down the whole search. This can take a few
minutes — each candidate reloads the model. Stop the server first (two copies of a model
will not fit in 16 GB).
"""
import argparse
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tests.perf import model_key, model_paths, prompt_of_size  # noqa: E402

FLOOR = 4096


def native_context(path):
    """The model's own max_position_embeddings, as an upper bound for the search —
    no point probing past what the model supports regardless of memory."""
    try:
        with open(os.path.join(path, "config.json")) as f:
            cfg = json.load(f)
        return (cfg.get("text_config") or cfg).get("max_position_embeddings", 32768)
    except (OSError, ValueError, KeyError):
        return 32768


def _probe(path, n):
    """Child process body (--probe n): load the model with max_context=n and try to
    prefill a real ~n-token prompt. True = it fits."""
    from engine.generate import Engine, MemoryLimitError
    try:
        eng = Engine(path, vision=False, max_context=n + 256, max_tokens=1, log=lambda *a: None)
        text, _, _ = eng.build_prompt([{"role": "user", "content": prompt_of_size(eng, n)}], thinking=False)
        ids, _ = eng.encode(text, [])
        # truncating to exactly n tokens can cut off the chat template's closing tags — harmless
        # here, since this only exercises the prefill memory path, not the model's actual answer.
        for _ in eng.generate(ids[:n], sampling={"temperature": 0}, max_tokens=1):
            pass
        return True
    except MemoryLimitError:
        return False
    except Exception:
        return False


def probe_subprocess(path, n, timeout=600):
    try:
        r = subprocess.run([sys.executable, "-m", "tests.calibrate", "--probe", str(n), "--model", path],
                           cwd=ROOT, capture_output=True, timeout=timeout)
        return r.returncode == 0
    except subprocess.TimeoutExpired:
        return False


def calibrate(path, hi, log=print):
    """Largest N in [FLOOR, hi] for which probe_subprocess(path, N) is True, or None if
    not even FLOOR fits. Classic "find the largest value satisfying a predicate" binary
    search — success at a given size implies success at any smaller size under the same
    conditions, so this converges to the true boundary."""
    lo = FLOOR
    if not probe_subprocess(path, lo):
        return None
    best = lo
    while lo <= hi:
        mid = (lo + hi) // 2
        log(f"  probing {mid} tokens…")
        if probe_subprocess(path, mid):
            best, lo = mid, mid + 1
        else:
            hi = mid - 1
    return (best // 1024) * 1024  # round down to a tidy number


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", action="append",
                    help="model path to calibrate (repeatable); default: ENGINE_TEST_MODEL or ./llm's selection")
    ap.add_argument("--max", type=int, default=131072, help="don't search above this many tokens (default 131072)")
    ap.add_argument("--apply", action="store_true",
                    help="write the result as that model's context setting (only when it's the one "
                         "currently selected in ./llm — context stops being \"auto\")")
    ap.add_argument("--probe", type=int, help=argparse.SUPPRESS)  # internal: single-trial child mode
    args = ap.parse_args()

    if args.probe is not None:
        sys.exit(0 if _probe(model_paths(args)[0], args.probe) else 1)

    for path in model_paths(args):
        key = model_key(path)
        print(f"\n=== {key} ===")
        n = calibrate(path, min(args.max, native_context(path)))
        if n is None:
            print(f"result: nothing fits even {FLOOR} tokens — close other apps or lower the RAM limit")
            continue
        print(f"result: {n} tokens")
        if args.apply:
            import llm
            cfg = llm.load_cfg()
            if os.path.realpath(path) != os.path.realpath(cfg.get("model", "")):
                print(f"not applying — {key} isn't the model currently selected in ./llm (switch to it first)")
                continue
            llm.set_mcfg(cfg, "context", n)
            llm.save_cfg(cfg)
            print(f"applied: context set to {n} tokens (was auto)")


if __name__ == "__main__":
    main()
