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
- sanity: trivial direct questions, run with thinking on when the model supports it. These
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

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
BASELINE = os.path.join(ROOT, "tests", "eval_baseline.json")

from tests.perf import model_key, model_paths  # noqa: E402 — same --model/env resolution as perf.py

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
    m = re.findall(r"Answer:\s*\$?(-?\d+(?:\.\d+)?)", text)
    if not m:
        m = re.findall(r"-?\d+(?:\.\d+)?", text)
    return float(m[-1]) if m else None


def extract_code(text):
    m = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.S)
    return (m.group(1) if m else text).strip()


def run_code_problem(problem, completion):
    script = extract_code(completion) + "\n\n" + problem["tests"] + "\nprint('OK')\n"
    try:
        p = subprocess.run([sys.executable, "-I", "-c", script], capture_output=True, text=True, timeout=10)
        return p.returncode == 0 and "OK" in p.stdout
    except subprocess.TimeoutExpired:
        return False


def ask(eng, prompt, max_tokens, thinking):
    eng.cache = None
    text, _, thinking_open = eng.build_prompt([{"role": "user", "content": prompt}], thinking=thinking)
    ids, _ = eng.encode(text, [])
    out, st = "", None
    for item in eng.generate(ids, sampling={"temperature": 0}, max_tokens=max_tokens, thinking_open=thinking_open):
        if isinstance(item, dict):
            st = item
        else:
            out += item[1]
    return out, st


def measure_one(path):
    from engine.generate import Engine

    eng = Engine(path, vision=False, max_context=4096, max_tokens=512, log=lambda *a: None)
    think = eng.supports_thinking
    res = {"math_correct": 0, "math_total": len(MATH_PROBLEMS),
           "code_correct": 0, "code_total": len(CODE_PROBLEMS),
           "sanity_correct": 0, "sanity_total": len(SANITY), "sanity_finished_cleanly": 0}
    fails = []

    for q, expected in MATH_PROBLEMS:
        out, _ = ask(eng, q + MATH_INSTRUCTION, 300, think)
        got = extract_number(out)
        ok = got is not None and abs(got - expected) < 1e-6
        res["math_correct"] += int(ok)
        if not ok:
            fails.append(f"math: expected {expected}, got {got!r} — {out[-120:]!r}")

    for prob in CODE_PROBLEMS:
        out, _ = ask(eng, prob["prompt"], 400, think)
        ok = run_code_problem(prob, out)
        res["code_correct"] += int(ok)
        if not ok:
            fails.append(f"code: {prob['name']} failed")

    for q, expected in SANITY:
        out, st = ask(eng, q, 300, think)
        res["sanity_correct"] += int(expected in out.lower())
        res["sanity_finished_cleanly"] += int(bool(st) and st.get("finish_reason") == "stop")
        if expected not in out.lower():
            fails.append(f"sanity: {q!r} -> {out[:80]!r}")

    res["fails"] = fails
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", action="append",
                    help="model path to evaluate (repeat to compare several models in one run); "
                         "default: ENGINE_TEST_MODEL or the model selected in ./llm")
    ap.add_argument("--save", action="store_true", help="store results as the new baseline")
    ap.add_argument("--tolerance", type=int, default=1,
                    help="allowed drop in correct answers per suite before it's a regression (default 1)")
    ap.add_argument("-v", "--verbose", action="store_true", help="print every failing problem")
    args = ap.parse_args()

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
