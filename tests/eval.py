"""Lightweight accuracy suite — complements tests/perf.py (speed) by checking that
quantization and engine changes don't quietly break answer quality.

    .venv/bin/python -m tests.eval                       # measure and compare to tests/eval_baseline.json
    .venv/bin/python -m tests.eval --save                 # measure and store as the new baseline
    .venv/bin/python -m tests.eval --model A --model B    # compare several models in one run
    .venv/bin/python -m tests.eval -v                     # also print every failing problem
    ./llm eval [--save] [--model PATH ...]

Three small, hand-written suites — no dataset downloads, no new dependencies:
- math: grade-school word problems (GSM8K-style), scored by extracting the final number.
- code: short HumanEval-style function-completion problems, scored by actually running the
  model's code against a few assert statements (pass/fail, like HumanEval's pass@1).
- sanity: trivial direct questions, in the model's default reasoning mode. These
  must finish with finish_reason == "stop", not "length" — this is exactly what the
  think-budget / repetition-loop guard (Engine.generate's think_budget, is_repeating) is
  for: short, simple prompts are what a reasoning model gets stuck looping on instead of
  answering, so this suite is a direct regression check for that fix, not just a content check.

Uses the model from ENGINE_TEST_MODEL or the one selected in ./llm, unless --model is given
(repeatable). Stop the server first (two copies of a model will not fit in 16 GB).
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
import unicodedata

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
BASELINE = os.path.join(ROOT, "tests", "eval_baseline.json")

from tests.perf import model_key, model_paths  # noqa: E402 — same --model/env resolution as perf.py

# Token budget per reply. Deliberately generous: with thinking on, a small model spends a few
# hundred tokens reasoning even about a one-line sum, and a tight limit makes the engine's
# think-budget guard cut it off mid-calculation — the eval would then be scoring the cutoff,
# not the model. (A 0.8B model answered 34 instead of 48 at 300 tokens and got it right at 1500.)
MAX_REPLY = 1536

MATH_INSTRUCTION = " Think if you need to, then end your reply with a line formatted exactly: Answer: <number>"
MATH_PROBLEMS = [
    ("Tom has 3 boxes of pencils with 12 pencils each. He gives away 9 pencils. "
     "How many pencils does he have left?", 27),
    ("A train travels 60 miles per hour for 3 hours, then 40 miles per hour for 2 hours. "
     "What is the total distance traveled, in miles?", 260),
    ("Sarah had $85. She spent $23 on a book and $14 on lunch. "
     "How much money does she have left, in dollars?", 48),
    ("A rectangular garden is 12 meters long and 7 meters wide. "
     "What is its area in square meters?", 84),
    ("There are 24 students in a class. If each student needs 4 sheets of paper, "
     "how many sheets are needed in total?", 96),
    ("A factory produces 450 widgets per day. How many widgets does it produce in 9 days?", 4050),
    ("Emma read 23 pages on Monday, 31 pages on Tuesday, and 18 pages on Wednesday. "
     "How many pages did she read in total?", 72),
    ("A shop had 200 apples. It sold three quarters of them. How many apples are left?", 50),
    ("A car uses 1 gallon of gas every 25 miles. How many gallons does it need to travel 325 miles?", 13),
    ("Liam has twice as many marbles as Noah. Noah has 17 marbles. "
     "How many marbles do they have together?", 51),
    ("A water tank holds 500 liters and is currently two fifths full. "
     "How many more liters are needed to fill it completely?", 300),
    ("Five friends split a restaurant bill of $142.50 equally, then each adds a $3.50 tip. "
     "How much does each friend pay in total, in dollars?", 32),
    ("A school ordered 18 boxes of 24 pencils each, but 36 pencils were damaged and discarded. "
     "How many usable pencils remain?", 396),
    ("A runner's pace is 8 minutes per mile. How many minutes will it take to run 10.5 miles?", 84),
    ("A warehouse has 1240 boxes. Workers load 85 boxes per hour onto trucks. "
     "How many full hours of loading will it take to load all the boxes (round down)?", 14),
]

CODE_PROBLEMS = [
    {"name": "is_prime", "prompt":
        "Write a complete Python function:\n\ndef is_prime(n: int) -> bool:\n"
        "    \"\"\"n is a positive integer. Return True if n is prime, False otherwise "
        "(1 is not prime).\"\"\"\n\nOutput only the function in a single Python code block.",
     "tests": "assert is_prime(2) is True\nassert is_prime(3) is True\nassert is_prime(4) is False\n"
              "assert is_prime(17) is True\nassert is_prime(1) is False\nassert is_prime(9) is False\n"},
    {"name": "reverse_words", "prompt":
        "Write a complete Python function:\n\ndef reverse_words(s: str) -> str:\n"
        "    \"\"\"Given a string of words separated by single spaces, return a string with "
        "the words in reverse order, still separated by single spaces.\"\"\"\n\n"
        "Output only the function in a single Python code block.",
     "tests": "assert reverse_words('hello world') == 'world hello'\n"
              "assert reverse_words('a b c') == 'c b a'\nassert reverse_words('single') == 'single'\n"},
    {"name": "sum_of_digits", "prompt":
        "Write a complete Python function:\n\ndef sum_of_digits(n: int) -> int:\n"
        "    \"\"\"Return the sum of the decimal digits of a non-negative integer n.\"\"\"\n\n"
        "Output only the function in a single Python code block.",
     "tests": "assert sum_of_digits(0) == 0\nassert sum_of_digits(123) == 6\nassert sum_of_digits(9999) == 36\n"},
    {"name": "is_palindrome", "prompt":
        "Write a complete Python function:\n\ndef is_palindrome(s: str) -> bool:\n"
        "    \"\"\"Return True if s reads the same forwards and backwards exactly as given "
        "(no case changes, no whitespace removal), False otherwise.\"\"\"\n\n"
        "Output only the function in a single Python code block.",
     "tests": "assert is_palindrome('level') is True\nassert is_palindrome('hello') is False\n"
              "assert is_palindrome('') is True\nassert is_palindrome('a') is True\n"},
    {"name": "flatten", "prompt":
        "Write a complete Python function:\n\ndef flatten(nested: list) -> list:\n"
        "    \"\"\"Given a list of lists of ints, return a single flat list with all the inner "
        "elements in original order.\"\"\"\n\nOutput only the function in a single Python code block.",
     "tests": "assert flatten([[1, 2], [3], [], [4, 5, 6]]) == [1, 2, 3, 4, 5, 6]\n"
              "assert flatten([[1]]) == [1]\nassert flatten([]) == []\n"},
    {"name": "count_vowels", "prompt":
        "Write a complete Python function:\n\ndef count_vowels(s: str) -> int:\n"
        "    \"\"\"Count the number of vowel characters (a, e, i, o, u, case-insensitive) in s.\"\"\"\n\n"
        "Output only the function in a single Python code block.",
     "tests": "assert count_vowels('Hello World') == 3\nassert count_vowels('xyz') == 0\n"
              "assert count_vowels('AEIOUaeiou') == 10\n"},
    {"name": "fibonacci", "prompt":
        "Write a complete Python function:\n\ndef fibonacci(n: int) -> int:\n"
        "    \"\"\"Return the nth Fibonacci number: fibonacci(0) = 0, fibonacci(1) = 1.\"\"\"\n\n"
        "Output only the function in a single Python code block.",
     "tests": "assert fibonacci(0) == 0\nassert fibonacci(1) == 1\nassert fibonacci(2) == 1\n"
              "assert fibonacci(10) == 55\n"},
    {"name": "unique_sorted", "prompt":
        "Write a complete Python function:\n\ndef unique_sorted(lst: list) -> list:\n"
        "    \"\"\"Return a sorted list of the unique integers in lst, ascending, duplicates removed.\"\"\"\n\n"
        "Output only the function in a single Python code block.",
     "tests": "assert unique_sorted([3, 1, 2, 3, 1]) == [1, 2, 3]\nassert unique_sorted([]) == []\n"
              "assert unique_sorted([5]) == [5]\n"},
]

# (prompt, expected substring — matched case-insensitively against the whole reply)
SANITY = [
    ("What is 7 + 5? Reply with just the number.", "12"),
    ("What is the capital of Japan? Reply with just the city name.", "tokyo"),
    ("Reply with exactly the single word: PONG", "pong"),
    ("What color do you get when you mix blue and yellow paint? One word.", "green"),
    ("Is the sky blue on a clear day? Answer yes or no.", "yes"),
    ("What is the chemical symbol for water? Reply with just the symbol.", "h2o"),
]


def extract_number(text):
    """The number on the requested "Answer: <number>" line; failing that, the last number on
    the reply's last non-empty line (models that bold it or add a unit). Deliberately NOT
    "the last number anywhere": on a reply cut off mid-reasoning that grabs list numbering
    or an intermediate figure and scores a truncation as a wrong answer."""
    m = re.findall(r"Answer:\s*\$?(-?\d+(?:\.\d+)?)", text)
    if m:
        return float(m[-1])
    lines = [l for l in text.strip().splitlines() if l.strip()]
    m = re.findall(r"-?\d+(?:\.\d+)?", lines[-1]) if lines else []
    return float(m[-1]) if m else None


def extract_code(text):
    m = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.S)
    return (m.group(1) if m else text).strip()


def run_code_problem(problem, completion):
    # the model's code runs in the same sandbox as the standard benchmarks (no network, writes only
    # inside a throw-away folder, killed with all its children on timeout)
    from tests.benchmarks import run_python
    return run_python(extract_code(completion) + "\n\n" + problem["tests"])


GREEDY = False  # --greedy: temperature 0 instead of the model's own sampling presets
THINK = None    # --think on|off: force the reasoning mode; auto (None) follows the model's template


def ask(eng, prompt, max_tokens, thinking):
    eng.cache = None
    text, _, thinking_open = eng.build_prompt([{"role": "user", "content": prompt}], thinking=thinking)
    ids, _ = eng.encode(text, [])
    out, st = "", None
    # The model's own presets (e.g. temperature 1.0 / top_p 0.95 for Qwen3.5 thinking) with a
    # fixed seed: reproducible, but not greedy — Qwen's model cards warn greedy decoding sends
    # reasoning models into endless repetition, and a benchmark that provokes that on purpose
    # measures the decoding mode rather than the model.
    sampling = {"temperature": 0} if GREEDY else {
        k: v for k, v in (eng.sampling[bool(thinking_open)] or {}).items()
        if k in ("temperature", "top_p", "top_k", "min_p", "presence_penalty")}
    for item in eng.generate(ids, sampling=sampling, seed=0, max_tokens=max_tokens, thinking_open=thinking_open):
        if isinstance(item, dict):
            st = item
        else:
            out += item[1]
    return out, st


def measure_one(path):
    from engine.generate import Engine

    eng = Engine(path, vision=False, max_context=4096, max_tokens=MAX_REPLY, ptq1=True, log=lambda *a: None)
    think = THINK  # None = the model's own default (what a client that says nothing gets)
    res = {"math_correct": 0, "math_total": len(MATH_PROBLEMS),
           "code_correct": 0, "code_total": len(CODE_PROBLEMS),
           "sanity_correct": 0, "sanity_total": len(SANITY), "sanity_finished_cleanly": 0,
           "think_forced": 0, "truncated": 0}
    fails = []

    def note(st, what):
        """Count replies where the engine force-closed the reasoning or ran out of tokens, so a
        score drop can be told apart from a model that simply got the answer wrong."""
        if st and st.get("think_forced"):
            res["think_forced"] += 1
            fails.append(f"{what}: reasoning force-closed ({st['think_forced']})")
        if st and st.get("finish_reason") == "length":
            res["truncated"] += 1
            fails.append(f"{what}: hit the token limit")

    for q, expected in MATH_PROBLEMS:
        out, st = ask(eng, q + MATH_INSTRUCTION, MAX_REPLY, think)
        got = extract_number(out)
        ok = got is not None and abs(got - expected) < 1e-6
        res["math_correct"] += int(ok)
        note(st, f"math {expected}")
        if not ok:
            fails.append(f"math: expected {expected}, got {got!r} — {out[-120:]!r}")

    for prob in CODE_PROBLEMS:
        out, st = ask(eng, prob["prompt"], MAX_REPLY, think)
        ok = run_code_problem(prob, out)
        res["code_correct"] += int(ok)
        note(st, f"code {prob['name']}")
        if not ok:
            fails.append(f"code: {prob['name']} failed")

    for q, expected in SANITY:
        out, st = ask(eng, q, MAX_REPLY, think)
        # NFKD turns subscript/superscript digits (H₂O) into plain ones before matching
        folded = unicodedata.normalize("NFKD", out).lower()
        res["sanity_correct"] += int(expected in folded)
        res["sanity_finished_cleanly"] += int(bool(st) and st.get("finish_reason") == "stop")
        note(st, f"sanity {expected!r}")
        if expected not in folded:
            fails.append(f"sanity: {q!r} -> {out[:80]!r}")

    res["fails"] = fails
    return res


BENCH_BASELINE = os.path.join(ROOT, "tests", "bench_baseline.json")


def run_benchmarks(args):
    """--suite: the standard benchmarks (tests/benchmarks.py) on each requested model."""
    from tests import benchmarks as B
    if args.suite == "list":
        for k, v in B.SUITES.items():
            print(f"{k:12s} {v['title']:18s} {v['area']:10s} {'runs model code' if v['kind'] == 'code' else ''}")
        return
    names = B.parse_suites(args.suite)
    from engine.generate import Engine
    base_all = {}
    if os.path.exists(BENCH_BASELINE):
        with open(BENCH_BASELINE) as f:
            base_all = json.load(f).get("models", {})
    think = {"auto": None, "on": True, "off": False}[args.think]
    global GREEDY
    GREEDY = args.greedy
    out = {}
    for path in model_paths(args):
        key = model_key(path)
        print(f"\n=== {key} · {', '.join(names)} · limit {args.limit or 'full'} · seed {args.seed} ===")
        eng = Engine(path, vision=False, max_context=4096, max_tokens=2048, ptq1=True, log=lambda *a: None)
        out[key] = {}
        for name in names:
            res = B.run_suite(eng, name, ask, limit=args.limit, seed=args.seed, think=think, allow_code=args.allow_code)
            out[key][f"{name}@{args.limit}/{args.seed}"] = res
            line = B.format_result(name, res)
            b = base_all.get(key, {}).get(f"{name}@{args.limit}/{args.seed}")
            if b and "correct" in res and b.get("total") == res["total"]:
                line += f"   baseline {b['correct']}/{b['total']}  {res['correct'] - b['correct']:+d}"
            print(line)
            if args.verbose and res.get("fails"):
                print("    " + "\n    ".join(res["fails"]))
        del eng
    if args.save:
        merged = base_all
        for k, v in out.items():
            merged.setdefault(k, {}).update({s: {kk: vv for kk, vv in r.items() if kk != "fails"} for s, r in v.items()
                                           if "correct" in r})
        with open(BENCH_BASELINE, "w") as f:
            json.dump({"date": time.strftime("%Y-%m-%d"), "models": merged}, f, indent=2)
        print(f"\nbaseline saved: {BENCH_BASELINE}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", action="append",
                    help="model path to evaluate (repeat to compare several models in one run); "
                         "default: ENGINE_TEST_MODEL or the model selected in ./llm")
    ap.add_argument("--save", action="store_true", help="store results as the new baseline")
    ap.add_argument("--tolerance", type=int, default=1,
                    help="allowed drop in correct answers per suite before it's a regression (default 1)")
    ap.add_argument("-v", "--verbose", action="store_true", help="print every failing problem")
    ap.add_argument("--greedy", action="store_true",
                    help="temperature 0 instead of the model's sampling presets (reasoning models loop on it)")
    ap.add_argument("--think", choices=("auto", "on", "off"), default="auto",
                    help="reasoning mode: auto = the model's own template default (small Qwen3.5 models: off)")
    ap.add_argument("--suite", help="standard benchmarks instead of the built-in suites: comma list or 'all' "
                                    "(see tests/benchmarks.py; 'list' shows them)")
    ap.add_argument("--limit", type=int, default=50, help="questions per benchmark suite (0 = the full dataset)")
    ap.add_argument("--seed", type=int, default=0, help="which random subset of each dataset")
    ap.add_argument("--allow-code", action="store_true",
                    help="let HumanEval/MBPP run the model's code on this machine")
    args = ap.parse_args()
    global GREEDY, THINK
    GREEDY = args.greedy
    THINK = {"auto": None, "on": True, "off": False}[args.think]

    if args.suite:
        return run_benchmarks(args)
    paths = model_paths(args)
    base_all = {}
    if os.path.exists(BASELINE):
        with open(BASELINE) as f:
            base_all = json.load(f).get("models", {})

    results, failed = {}, []
    for path in paths:
        key = model_key(path)
        print(f"\n=== {key} ===")
        res = measure_one(path)
        results[key] = res
        base = base_all.get(key, {})
        for suite in ("math", "code", "sanity"):
            correct, total = res[f"{suite}_correct"], res[f"{suite}_total"]
            line = f"{suite:8s} {correct}/{total}"
            b = base.get(f"{suite}_correct")
            if isinstance(b, int):
                delta = correct - b
                line += f"   baseline {b}/{total}   {delta:+d}"
                if delta < -args.tolerance:
                    line += "  ✗ regression"
                    failed.append(f"{key}/{suite}")
            print(line)
        print(f"sanity finished cleanly (not cut off mid-thought): "
              f"{res['sanity_finished_cleanly']}/{res['sanity_total']}")
        total = res["math_total"] + res["code_total"] + res["sanity_total"]
        print(f"replies where reasoning was force-closed: {res['think_forced']}/{total}"
              f" · hit the token limit: {res['truncated']}/{total}")
        if args.verbose and res["fails"]:
            print("  " + "\n  ".join(res["fails"]))

    if args.save:
        clean = {k: {kk: vv for kk, vv in v.items() if kk != "fails"} for k, v in results.items()}
        with open(BASELINE, "w") as f:
            json.dump({"date": time.strftime("%Y-%m-%d"), "models": clean}, f, indent=2)
        print(f"\nbaseline saved: {BASELINE}")
    if failed and not args.save:
        print(f"\nregressions beyond {args.tolerance}: {', '.join(failed)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
