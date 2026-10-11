"""Standard benchmark suites for the local engine, run zero-shot through the model's chat template.

    ./llm eval --suite list
    ./llm eval --suite gsm8k,mmlu,arc --limit 100          # 100 random questions each (seeded)
    ./llm eval --suite humaneval,mbpp --allow-code         # runs the model's code locally
    ./llm eval --suite mmlu --limit 0                      # 0 = the full dataset

Data comes from the Hugging Face datasets-server (plain HTTPS + JSON, nothing to install) and is
cached under ~/.cache/ai-llm-engine/bench. The score is generative zero-shot accuracy — the model
answers with a letter/number/function — so it is comparable between models and between runs of this
tool, NOT with published log-likelihood or few-shot numbers.
"""
import json
import os
import random
import re
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

LETTERS = "ABCDEFGHIJKLMNOP"
SERVER = os.environ.get("BENCH_DATASETS_URL", "https://datasets-server.huggingface.co")
CACHE = os.path.join(os.path.expanduser("~"), ".cache", "ai-llm-engine", "bench")
MCQ_INSTRUCTION = "Answer with only the letter of the correct option."


# ───────────────────────────── formatting: row -> (prompt, gold) ─────────────────────────────
def _options(choices):
    return "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(choices))


def _mcq(stem, choices, gold_index):
    return f"{stem}\n\n{_options(choices)}\n\n{MCQ_INSTRUCTION}", LETTERS[gold_index]


def fmt_mmlu(row, rng=None):
    subject = str(row.get("subject", "")).replace("_", " ")
    stem = (f"The following is a multiple choice question about {subject}." if subject else
            "The following is a multiple choice question.") + "\n\n" + row["question"]
    return _mcq(stem, row["choices"], int(row["answer"]))


def fmt_mmlu_pro(row, rng=None):
    return _mcq(row["question"], row["options"], int(row["answer_index"]))


def fmt_arc(row, rng=None):
    labels = [str(l) for l in row["choices"]["label"]]
    return _mcq(row["question"], row["choices"]["text"], labels.index(str(row["answerKey"])))


def fmt_hellaswag(row, rng=None):
    ctx = (row.get("activity_label") + ": " if row.get("activity_label") else "") + row["ctx"]
    stem = "Choose the most plausible continuation of the text.\n\n" + ctx
    return _mcq(stem, row["endings"], int(row["label"]))


def fmt_truthfulqa(row, rng):
    """mc1: exactly one correct choice, listed first in the data — shuffled so position carries no hint."""
    t = row["mc1_targets"]
    order = list(range(len(t["choices"])))
    rng.shuffle(order)
    choices = [t["choices"][i] for i in order]
    gold = next(k for k, i in enumerate(order) if t["labels"][i] == 1)
    return _mcq(row["question"], choices, gold)


def fmt_winogrande(row, rng=None):
    stem = ("Fill in the blank (_) in the sentence with the option that makes it correct.\n\n" + row["sentence"])
    return _mcq(stem, [row["option1"], row["option2"]], int(row["answer"]) - 1)


def fmt_gsm8k(row, rng=None):
    from tests.eval import MATH_INSTRUCTION
    gold = float(str(row["answer"]).split("####")[-1].replace(",", "").strip())
    return row["question"] + MATH_INSTRUCTION, gold


def fmt_humaneval(row, rng=None):
    prompt = ("Complete the following Python function. Reply with the complete function (including its "
              "signature) in a single ```python code block.\n\n```python\n" + row["prompt"].rstrip() + "\n```")
    return prompt, {"prelude": _imports(row["prompt"]), "tests": row["test"] + f"\n\ncheck({row['entry_point']})"}


def fmt_mbpp(row, rng=None):
    tests = "\n".join(row["test_list"])
    prompt = (f"{row['text'].strip()}\nYour code should pass these tests:\n{tests}\n\n"
              "Reply with the function in a single ```python code block.")
    return prompt, {"prelude": row.get("test_setup_code") or "", "tests": tests}


def _imports(src):
    return "\n".join(l for l in src.splitlines() if l.startswith(("import ", "from ")))


# ───────────────────────────── scoring ─────────────────────────────
def strip_reasoning(text):
    """What follows the last </think>: a reply that is still inside its reasoning has no answer."""
    return text.rsplit("</think>", 1)[-1] if "</think>" in text else text


def extract_choice(text, n):
    """The option letter the reply commits to, or None. Looks for an explicit 'Answer: X' first, then a
    reply that opens with the letter ('B', 'B.', '(B)', 'B) ...'), then a lone letter early on."""
    text = strip_reasoning(text).strip()
    letters = LETTERS[:n]
    cls = f"[{letters}]"
    m = re.findall(rf"(?:answer|option|choice)\s*(?:is|:)?\s*\**\(?({cls})\)?\b", text, re.I)
    if m:
        return m[-1].upper()
    m = re.match(rf"^[\s\*\(\[`\"']*({cls})(?![A-Za-z])", text)
    if m:
        return m.group(1)
    m = re.search(rf"(?<![A-Za-z])\(?({cls})\)?(?![A-Za-z])", text[:60])
    return m.group(1) if m else None


def run_python(script, timeout=10):
    """True when `script` runs to completion printing OK. Isolated mode, throw-away directory."""
    with tempfile.TemporaryDirectory() as d:
        try:
            p = subprocess.run([sys.executable, "-I", "-B", "-c", script + "\nprint('OK')\n"], cwd=d,
                               capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL)
        except subprocess.TimeoutExpired:
            return False
    return p.returncode == 0 and p.stdout.rstrip().endswith("OK")


def score_code(reply, gold):
    from tests.eval import extract_code
    code = extract_code(strip_reasoning(reply))
    return run_python("\n".join(x for x in (gold["prelude"], code, gold["tests"]) if x))


def score_number(reply, gold):
    from tests.eval import extract_number
    got = extract_number(strip_reasoning(reply))
    return got is not None and abs(got - gold) < 1e-6


# ───────────────────────────── registry ─────────────────────────────
SUITES = {
    "gsm8k": dict(title="GSM8K", area="math", dataset="openai/gsm8k", config="main", split="test",
                  fmt=fmt_gsm8k, kind="number", max_tokens=1536),
    "mmlu": dict(title="MMLU", area="knowledge", dataset="cais/mmlu", config="all", split="test",
                 fmt=fmt_mmlu, kind="mcq"),
    "mmlu-pro": dict(title="MMLU-Pro", area="knowledge", dataset="TIGER-Lab/MMLU-Pro", config="default",
                     split="test", fmt=fmt_mmlu_pro, kind="mcq"),
    "arc": dict(title="ARC-Challenge", area="reasoning", dataset="allenai/ai2_arc", config="ARC-Challenge",
                split="test", fmt=fmt_arc, kind="mcq"),
    "hellaswag": dict(title="HellaSwag", area="reasoning", dataset="Rowan/hellaswag", config="default",
                      split="validation", fmt=fmt_hellaswag, kind="mcq"),
    "truthfulqa": dict(title="TruthfulQA (MC1)", area="reasoning", dataset="truthfulqa/truthful_qa",
                       config="multiple_choice", split="validation", fmt=fmt_truthfulqa, kind="mcq"),
    "winogrande": dict(title="WinoGrande", area="reasoning", dataset="allenai/winogrande",
                       config="winogrande_xl", split="validation", fmt=fmt_winogrande, kind="mcq"),
    "humaneval": dict(title="HumanEval", area="code", dataset="openai/openai_humaneval", config="openai_humaneval",
                      split="test", fmt=fmt_humaneval, kind="code", max_tokens=1536),
    "mbpp": dict(title="MBPP", area="code", dataset="google-research-datasets/mbpp", config="full", split="test",
                 fmt=fmt_mbpp, kind="code", max_tokens=1536),
}
DEFAULT_LIMIT = 50
# MBPP tasks whose own reference solution fails its own tests (checked: 497 of 500 pass) — scoring a
# model against them would only punish it for the dataset's defects
BROKEN = {"mbpp": {180, 367, 493}}


# ───────────────────────────── data ─────────────────────────────
def _get(url, retries=8):
    """GET JSON; the datasets server rate-limits (429): honour Retry-After, back off, and keep going."""
    delay = 1.0
    for attempt in range(retries):
        wait = delay
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "ai-llm-engine-bench"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            retry_after = e.headers.get("Retry-After") if e.headers else None
            e.close()
            if e.code not in (429, 500, 502, 503, 504) or attempt == retries - 1:
                raise
            if retry_after and retry_after.isdigit():
                wait = min(60.0, float(retry_after))
        except (urllib.error.URLError, OSError):
            if attempt == retries - 1:
                raise
        time.sleep(wait)
        delay = min(delay * 2, 30)


def _rows_url(spec, offset, length):
    q = urllib.parse.urlencode({"dataset": spec["dataset"], "config": spec["config"], "split": spec["split"],
                                "offset": offset, "length": length})
    return f"{SERVER}/rows?{q}"


def load_rows(name, limit=DEFAULT_LIMIT, seed=0, cache=True):
    """`limit` random rows (0 = all), reproducible for a seed, cached on disk."""
    spec = SUITES[name]
    path = os.path.join(CACHE, f"{name}-{limit}-{seed}.json")
    if cache and os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    broken = BROKEN.get(name, set())
    first = _get(_rows_url(spec, 0, 1))
    total = first.get("num_rows_total") or 0
    if not total:
        raise RuntimeError(f"{spec['dataset']}: the datasets server returned no rows")
    if limit and limit < total:
        idx = sorted(random.Random(seed).sample(range(total), limit))
        with ThreadPoolExecutor(4) as ex:
            rows = list(ex.map(lambda i: _get(_rows_url(spec, i, 1))["rows"][0]["row"], idx))
    else:
        rows = []
        for off in range(0, total, 100):
            rows += [r["row"] for r in _get(_rows_url(spec, off, min(100, total - off)))["rows"]]
    rows = [r for r in rows if r.get("task_id") not in broken]
    if cache:
        os.makedirs(CACHE, exist_ok=True)
        with open(path + ".tmp", "w") as f:
            json.dump(rows, f)
        os.replace(path + ".tmp", path)
    return rows


# ───────────────────────────── running ─────────────────────────────
def interval(correct, total):
    """Half-width of the 95 % normal-approximation interval — small samples are noisy; say so."""
    if not total:
        return 0.0
    p = correct / total
    return 1.96 * (p * (1 - p) / total) ** 0.5


def run_suite(eng, name, ask, limit=DEFAULT_LIMIT, seed=0, think=None, allow_code=False, rows=None,
              default_max_tokens=64, log=print):
    """Score one suite. `ask(eng, prompt, max_tokens, thinking) -> (reply, stats)` is tests.eval.ask."""
    spec = SUITES[name]
    if spec["kind"] == "code" and not allow_code:
        return {"skipped": "runs the model's code locally; pass --allow-code"}
    rows = rows if rows is not None else load_rows(name, limit, seed)
    rng = random.Random(seed)
    res = {"correct": 0, "total": 0, "unparsed": 0, "truncated": 0, "think_forced": 0, "tokens": 0, "seconds": 0.0}
    fails = []
    # multiple-choice answers are one letter: a reasoning model need not think for them unless asked to
    thinking = think if think is not None else (False if spec["kind"] == "mcq" else None)
    t0 = time.time()
    for i, row in enumerate(rows):
        prompt, gold = spec["fmt"](row, rng)
        max_tokens = spec.get("max_tokens", default_max_tokens if thinking is False else 1024)
        reply, st = ask(eng, prompt, max_tokens, thinking)
        res["total"] += 1
        res["tokens"] += (st or {}).get("completion_tokens", 0)
        if st and st.get("finish_reason") == "length":
            res["truncated"] += 1
        if st and st.get("think_forced"):
            res["think_forced"] += 1
        if spec["kind"] == "mcq":
            letter = extract_choice(reply, len(LETTERS[:len(spec_options(row, spec))]))
            ok = letter == gold
            res["unparsed"] += int(letter is None)
        elif spec["kind"] == "number":
            ok = score_number(reply, gold)
        else:
            ok = score_code(reply, gold)
        res["correct"] += int(ok)
        if not ok and len(fails) < 20:
            fails.append(f"#{i}: gold {gold if not isinstance(gold, dict) else '(tests)'} — {reply[-100:]!r}")
        if (i + 1) % 10 == 0:
            log(f"  {name}: {i + 1}/{len(rows)}  acc {res['correct'] / res['total']:.1%}")
    res["seconds"] = round(time.time() - t0, 1)
    res["fails"] = fails
    return res


def spec_options(row, spec):
    """How many options a multiple-choice row has (to restrict which letters count as an answer)."""
    for key in ("choices", "options", "endings"):
        v = row.get(key)
        if isinstance(v, list):
            return v
        if isinstance(v, dict) and "text" in v:
            return v["text"]
    if "mc1_targets" in row:
        return row["mc1_targets"]["choices"]
    if "option1" in row:
        return [row["option1"], row["option2"]]
    return list("ABCD")


def format_result(name, res):
    if "skipped" in res:
        return f"{SUITES[name]['title']:18s} skipped — {res['skipped']}"
    c, t = res["correct"], res["total"]
    line = (f"{SUITES[name]['title']:18s} {c}/{t}  {c / t:6.1%} ± {interval(c, t):.1%}"
            f"   {res['seconds']:.0f}s  {res['tokens']} tok")
    extra = []
    if res.get("unparsed"):
        extra.append(f"{res['unparsed']} unparsed")
    if res.get("truncated"):
        extra.append(f"{res['truncated']} cut off")
    if res.get("think_forced"):
        extra.append(f"{res['think_forced']} reasoning force-closed")
    return line + ("   (" + ", ".join(extra) + ")" if extra else "")


def parse_suites(arg):
    if not arg or arg == "all":
        return list(SUITES)
    names = [s.strip().lower() for s in arg.split(",") if s.strip()]
    bad = [n for n in names if n not in SUITES]
    if bad:
        raise SystemExit(f"unknown suite(s): {', '.join(bad)}; available: {', '.join(SUITES)}")
    return names
