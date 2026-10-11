"""tests/benchmarks.py without a network or a model: row formatting, answer extraction, scoring, the
dataset loader against a fake datasets-server, and the run loop with a stub model.

    .venv/bin/python -m unittest tests.test_benchmarks -v
"""
import json
import os
import random
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest import mock
from urllib.parse import parse_qs, urlparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from tests import benchmarks as B  # noqa: E402

MMLU = {"question": "What is 2+2?", "subject": "elementary_mathematics", "choices": ["3", "4", "5", "6"], "answer": 1}
PRO = {"question": "Pick the prime.", "options": ["4", "6", "7", "8", "9"], "answer_index": 2}
ARC = {"question": "Which is a mammal?", "choices": {"text": ["trout", "whale", "frog"], "label": ["A", "B", "C"]},
       "answerKey": "B"}
ARC_NUM = {"question": "Q", "choices": {"text": ["x", "y", "z", "w"], "label": ["1", "2", "3", "4"]}, "answerKey": "3"}
HELLA = {"activity_label": "Cooking", "ctx": "A man stirs a pot.", "endings": ["he flies", "he tastes it", "he sleeps", "he dies"],
         "label": "1"}
TRUTH = {"question": "Can you see the Great Wall from space?",
         "mc1_targets": {"choices": ["No, not with the naked eye", "Yes, clearly", "Yes, at night", "Only in winter"],
                         "labels": [1, 0, 0, 0]}}
WINO = {"sentence": "The trophy didn't fit in the suitcase because the _ was too big.", "option1": "trophy",
        "option2": "suitcase", "answer": "1"}
GSM = {"question": "Tom has 3 apples and buys 4 more. How many?", "answer": "3 + 4 = 7\n#### 7"}
GSM_COMMA = {"question": "Q", "answer": "so much\n#### 1,250"}
HUMAN = {"prompt": "from typing import List\n\n\ndef add(a: int, b: int) -> int:\n    \"\"\"Add.\"\"\"\n",
         "canonical_solution": "    return a + b\n",
         "test": "def check(candidate):\n    assert candidate(1, 2) == 3\n    assert candidate(0, 0) == 0\n",
         "entry_point": "add", "task_id": "HumanEval/0"}
MBPP = {"task_id": 1, "text": "Write a function to multiply two numbers.", "code": "def mul(a, b):\n    return a * b",
        "test_list": ["assert mul(2, 3) == 6", "assert mul(0, 5) == 0"], "test_setup_code": ""}


class TestFormatting(unittest.TestCase):
    def test_mmlu_style_multiple_choice(self):
        prompt, gold = B.fmt_mmlu(MMLU)
        self.assertEqual(gold, "B")
        self.assertIn("about elementary mathematics", prompt)
        self.assertIn("A. 3\nB. 4\nC. 5\nD. 6", prompt)
        self.assertTrue(prompt.rstrip().endswith(B.MCQ_INSTRUCTION))

    def test_ten_options(self):
        prompt, gold = B.fmt_mmlu_pro(PRO)
        self.assertEqual(gold, "C")
        self.assertIn("E. 9", prompt)

    def test_arc_with_letter_and_numeric_labels(self):
        self.assertEqual(B.fmt_arc(ARC)[1], "B")
        self.assertEqual(B.fmt_arc(ARC_NUM)[1], "C")             # label "3" is the third option

    def test_hellaswag(self):
        prompt, gold = B.fmt_hellaswag(HELLA)
        self.assertEqual(gold, "B")
        self.assertIn("Cooking: A man stirs a pot.", prompt)

    def test_truthfulqa_is_shuffled_but_the_gold_follows_the_right_answer(self):
        seen = set()
        for seed in range(12):
            prompt, gold = B.fmt_truthfulqa(TRUTH, random.Random(seed))
            line = [l for l in prompt.splitlines() if l.startswith(f"{gold}. ")][0]
            self.assertIn("No, not with the naked eye", line)
            seen.add(gold)
        self.assertGreater(len(seen), 1)                          # position varies, as it should

    def test_winogrande_maps_one_two_to_letters(self):
        prompt, gold = B.fmt_winogrande(WINO)
        self.assertEqual(gold, "A")
        self.assertIn("A. trophy\nB. suitcase", prompt)
        self.assertEqual(B.fmt_winogrande(dict(WINO, answer="2"))[1], "B")

    def test_gsm8k_gold_number(self):
        prompt, gold = B.fmt_gsm8k(GSM)
        self.assertEqual(gold, 7.0)
        self.assertIn("Answer:", prompt)
        self.assertEqual(B.fmt_gsm8k(GSM_COMMA)[1], 1250.0)

    def test_code_tasks_carry_their_tests(self):
        prompt, gold = B.fmt_humaneval(HUMAN)
        self.assertIn("def add(a: int, b: int)", prompt)
        self.assertIn("from typing import List", gold["prelude"])
        self.assertTrue(gold["tests"].rstrip().endswith("check(add)"))
        prompt, gold = B.fmt_mbpp(MBPP)
        self.assertIn("assert mul(2, 3) == 6", prompt)
        self.assertEqual(gold["tests"], "\n".join(MBPP["test_list"]))


class TestExtraction(unittest.TestCase):
    def test_the_ways_models_actually_answer(self):
        cases = {
            "B": "B", "B.": "B", "(C)": "C", "**D**": "D", "A) because it is": "A", "b": None,
            "The answer is (C).": "C", "Answer: B": "B", "answer: **A**": "A",
            "Let me think... the correct option is D": "D",
            "</think>\nC": "C", "<think>maybe A, maybe B</think>\n\nB": "B",
            "I am not sure.": None, "": None,
        }
        for text, want in cases.items():
            with self.subTest(text):
                self.assertEqual(B.extract_choice(text, 4), want)

    def test_letters_beyond_the_option_count_do_not_count(self):
        self.assertIsNone(B.extract_choice("E", 4))
        self.assertEqual(B.extract_choice("E", 5), "E")

    def test_a_reply_still_inside_its_reasoning_has_no_answer(self):
        self.assertEqual(B.strip_reasoning("<think>A or B"), "<think>A or B")   # no </think>: nothing to strip
        self.assertEqual(B.strip_reasoning("x</think>y</think>final"), "final")


class TestScoring(unittest.TestCase):
    def test_numbers(self):
        self.assertTrue(B.score_number("so 3+4\nAnswer: 7", 7.0))
        self.assertFalse(B.score_number("Answer: 8", 7.0))
        self.assertTrue(B.score_number("<think>Answer: 5</think>\nAnswer: 7", 7.0))   # only the final answer counts
        self.assertFalse(B.score_number("", 7.0))

    def test_code_passes_fails_and_times_out(self):
        _, gold = B.fmt_humaneval(HUMAN)
        good = "```python\ndef add(a: int, b: int) -> int:\n    return a + b\n```"
        bad = "```python\ndef add(a, b):\n    return a - b\n```"
        loop = "```python\ndef add(a, b):\n    while True: pass\n```"
        self.assertTrue(B.score_code(good, gold))
        self.assertTrue(B.score_code("Sure!\n\n" + good + "\nHope that helps.", gold))
        self.assertFalse(B.score_code(bad, gold))
        self.assertFalse(B.score_code("no code at all", gold))
        self.assertFalse(B.run_python("while True: pass", timeout=1))             # a runaway is a fail, not a hang

    def test_a_script_that_exits_early_or_prints_nothing_is_not_a_pass(self):
        self.assertTrue(B.run_python("x = 1"))
        self.assertFalse(B.run_python("import sys; sys.exit(3)"))
        self.assertFalse(B.run_python("raise SystemExit"))
        self.assertTrue(B.run_python("def f(): return 1\nassert f() == 1"))
        # a program that merely PRINTS "OK" and then dies must not pass
        self.assertFalse(B.run_python("print('OK')\nraise ValueError('boom')"))

    def test_mbpp_prelude_runs_first(self):
        gold = {"prelude": "import math", "tests": "assert math.floor(f(2.5)) == 5"}
        self.assertTrue(B.score_code("```python\ndef f(x):\n    return x * 2\n```", gold))

    def test_the_confidence_interval(self):
        self.assertAlmostEqual(B.interval(50, 100), 0.098, places=3)
        self.assertEqual(B.interval(0, 0), 0.0)
        self.assertEqual(B.interval(10, 10), 0.0)


class FakeServer(BaseHTTPRequestHandler):
    total = 25
    rate_limit_first = 0
    calls = []

    def log_message(self, *a):
        pass

    def do_GET(self):
        q = {k: v[0] for k, v in parse_qs(urlparse(self.path).query).items()}
        FakeServer.calls.append((int(q["offset"]), int(q["length"])))
        if FakeServer.rate_limit_first > 0:
            FakeServer.rate_limit_first -= 1
            self.send_response(429)
            self.send_header("Retry-After", "0")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        off, n = int(q["offset"]), int(q["length"])
        rows = [{"row_idx": i, "row": {"question": f"q{i}", "subject": "s", "choices": ["a", "b", "c", "d"],
                                       "answer": i % 4}} for i in range(off, min(off + n, FakeServer.total))]
        body = json.dumps({"rows": rows, "num_rows_total": FakeServer.total}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class TestLoader(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = ThreadingHTTPServer(("127.0.0.1", 0), FakeServer)
        cls.url = f"http://127.0.0.1:{cls.srv.server_address[1]}"
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        cls.srv.server_close()

    def setUp(self):
        FakeServer.total, FakeServer.rate_limit_first, FakeServer.calls = 25, 0, []
        self.cache = tempfile.mkdtemp()
        for name, val in (("SERVER", self.url), ("CACHE", self.cache)):
            p = mock.patch.object(B, name, val)
            p.start()
            self.addCleanup(p.stop)
        p = mock.patch.object(B.time, "sleep", lambda s: None)
        p.start()
        self.addCleanup(p.stop)

    def test_a_seeded_subset_is_reproducible_and_distinct_per_seed(self):
        a = [r["question"] for r in B.load_rows("mmlu", 6, seed=1, cache=False)]
        b = [r["question"] for r in B.load_rows("mmlu", 6, seed=1, cache=False)]
        c = [r["question"] for r in B.load_rows("mmlu", 6, seed=2, cache=False)]
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)
        self.assertEqual(len(set(a)), 6)                          # no duplicates

    def test_zero_means_the_whole_dataset_in_pages(self):
        FakeServer.total = 230
        rows = B.load_rows("mmlu", 0, cache=False)
        self.assertEqual(len(rows), 230)
        self.assertEqual([r["question"] for r in rows], [f"q{i}" for i in range(230)])
        pages = [c for c in FakeServer.calls if c[1] > 1]
        self.assertEqual(pages, [(0, 100), (100, 100), (200, 30)])

    def test_a_limit_above_the_size_returns_everything(self):
        self.assertEqual(len(B.load_rows("mmlu", 1000, cache=False)), 25)

    def test_the_second_call_comes_from_the_disk_cache(self):
        B.load_rows("mmlu", 5, seed=3)
        n = len(FakeServer.calls)
        again = B.load_rows("mmlu", 5, seed=3)
        self.assertEqual(len(FakeServer.calls), n)                # no new requests
        self.assertEqual(len(again), 5)
        self.assertTrue(os.listdir(self.cache))

    def test_rate_limiting_is_waited_out(self):
        FakeServer.rate_limit_first = 3
        rows = B.load_rows("mmlu", 3, seed=0, cache=False)
        self.assertEqual(len(rows), 3)

    def test_an_empty_dataset_is_an_error_not_a_zero_score(self):
        FakeServer.total = 0
        with self.assertRaises(RuntimeError):
            B.load_rows("mmlu", 5, cache=False)

    def test_known_broken_mbpp_tasks_are_dropped(self):
        def fake_get(url):
            if "length=1" in url and "offset=0" in url:
                return {"rows": [], "num_rows_total": 4}
            return {"rows": [{"row": {"task_id": t, "text": "", "code": "", "test_list": []}} for t in (1, 180, 2, 367)]}
        with mock.patch.object(B, "_get", fake_get):
            ids = [r["task_id"] for r in B.load_rows("mbpp", 0, cache=False)]
        self.assertEqual(ids, [1, 2])


class TestRunLoop(unittest.TestCase):
    def rows(self):
        return [dict(MMLU, answer=i % 4, question=f"q{i}") for i in range(8)]

    def test_scores_a_perfect_a_hopeless_and_a_rambling_model(self):
        rows = self.rows()
        gold = [B.fmt_mmlu(r)[1] for r in rows]
        it = iter(gold)
        perfect = lambda eng, prompt, mt, think: (f"Answer: {next(it)}", {"completion_tokens": 3, "finish_reason": "stop"})
        r = B.run_suite(None, "mmlu", perfect, rows=rows)
        self.assertEqual((r["correct"], r["total"], r["unparsed"]), (8, 8, 0))
        wrong = lambda eng, prompt, mt, think: ("Z", {"completion_tokens": 1, "finish_reason": "stop"})
        r = B.run_suite(None, "mmlu", wrong, rows=rows)
        self.assertEqual((r["correct"], r["unparsed"]), (0, 8))
        cut = lambda eng, prompt, mt, think: ("hmm let me think", {"completion_tokens": 64, "finish_reason": "length",
                                                                   "think_forced": "budget"})
        r = B.run_suite(None, "mmlu", cut, rows=rows)
        self.assertEqual((r["truncated"], r["think_forced"], r["tokens"]), (8, 8, 512))

    def test_multiple_choice_runs_without_reasoning_unless_asked(self):
        seen = []
        ask = lambda eng, prompt, mt, think: (seen.append((think, mt)) or "A", {})
        B.run_suite(None, "mmlu", ask, rows=self.rows()[:1])
        B.run_suite(None, "mmlu", ask, rows=self.rows()[:1], think=True)
        self.assertEqual(seen[0][0], False)
        self.assertLess(seen[0][1], 256)                           # a letter needs no big budget
        self.assertEqual(seen[1][0], True)
        seen.clear()
        B.run_suite(None, "gsm8k", ask, rows=[GSM], think=None)
        self.assertIsNone(seen[0][0])                              # math/code follow the model's own default

    def test_gsm8k_and_code_end_to_end(self):
        ask = lambda eng, prompt, mt, think: ("so\nAnswer: 7", {})
        self.assertEqual(B.run_suite(None, "gsm8k", ask, rows=[GSM])["correct"], 1)
        ok = lambda eng, prompt, mt, think: ("```python\ndef mul(a, b):\n    return a * b\n```", {})
        self.assertEqual(B.run_suite(None, "mbpp", ok, rows=[MBPP], allow_code=True)["correct"], 1)

    def test_code_is_not_run_without_permission(self):
        called = []
        r = B.run_suite(None, "humaneval", lambda *a: called.append(1) or ("", {}), rows=[HUMAN])
        self.assertIn("skipped", r)
        self.assertEqual(called, [])
        self.assertIn("--allow-code", B.format_result("humaneval", r))

    def test_result_line_names_what_went_wrong(self):
        line = B.format_result("mmlu", {"correct": 30, "total": 50, "unparsed": 4, "truncated": 2, "think_forced": 0,
                                        "tokens": 400, "seconds": 12.0})
        self.assertIn("30/50", line)
        self.assertIn("60.0%", line)
        self.assertIn("4 unparsed", line)
        self.assertIn("2 cut off", line)

    def test_suite_selection(self):
        self.assertEqual(B.parse_suites("gsm8k, MMLU"), ["gsm8k", "mmlu"])
        self.assertEqual(B.parse_suites("all"), list(B.SUITES))
        with self.assertRaises(SystemExit):
            B.parse_suites("gsm8k,nope")


if __name__ == "__main__":
    unittest.main()
