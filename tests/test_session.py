"""The decode loop (engine/session.py) on a runtime that is not MLX: a toy model in plain Python.

If the loop runs here — prefix reuse, pipelined decode, speculation with exact acceptance, stop strings —
then it does not depend on what computes a forward pass, which is the point of the runtime seam.

    .venv/bin/python -m unittest tests.test_session -v
"""
import os
import re
import threading
import unittest
from collections import deque

from engine import session

V = 50
EOS = 0


def follow(tok):
    """The toy model: every token has one successor, and 49 is followed by end-of-sequence."""
    return EOS if tok == V - 1 else tok + 1


class ToyCache:
    rewindable = True

    def __init__(self):
        self.toks = []

    def snapshot(self):
        return len(self.toks)

    def restore(self, n):
        del self.toks[n:]

    arrays = ()


class ToySampler:
    greedy = True
    allow = on_pick = None


class ToyRuntime:
    """Logits are just "the token that follows"; a handle is a one-element list read by read_token."""
    name = "toy"

    def __init__(self):
        self.model = self
        self.steps = self.verifies = self.flushed = 0
        self.prefilled = []

    def make_cache(self):
        return [ToyCache()]

    def sampler(self, vocab, seed, sampling):
        return ToySampler()

    def prefill(self, cache, ids, embeds, pos3, delta, s, e, hidden=False):
        cache[0].toks += ids[s:e]
        self.prefilled.append((s, e))
        return [follow(ids[e - 1])]

    def step(self, token, cache, delta):
        tok = token if isinstance(token, int) else token[0]
        cache[0].toks.append(tok)
        self.steps += 1
        return [follow(tok)]

    def sample(self, sampler, logits):
        return [logits[0]]

    def flush(self, *handles):
        self.flushed += 1

    def read_token(self, handle):
        return handle if isinstance(handle, int) else handle[0]

    def verify(self, token, draft, cache, delta):
        self.verifies += 1
        cache[0].base = len(cache[0].toks)
        cache[0].toks += [token] + draft
        return [follow(t) for t in [token] + draft]

    def greedy_tokens(self, logits):
        return list(logits)

    def commit(self, cache, m):
        del cache[0].toks[cache[0].base + m:]

    def settle(self, cache):
        pass

    def release(self):
        pass

    def peak_gb(self):
        return 0.0


class ToyTok:
    def decode(self, ids):
        return "".join(f"<{t}>" for t in ids)

    def encode(self, text):
        return [int(x) for x in re.findall(r"<(\d+)>", text)]


class ToyEngine(session.DecodeLoop):
    def __init__(self, speculative=True):
        self.rt = ToyRuntime()
        self.tok, self.eos, self.vocab = ToyTok(), {EOS}, V
        self.max_context, self.max_tokens, self.prefill_step, self.step_now = 4096, 256, 8, 8
        self.mem_limit, self.think_frac = 0.99, 0.6
        self.speculative, self.draft_levels, self.ngram_min = speculative, (1, 2, 3, 4, 15), 2
        self.cache, self.cache_tokens, self.snapshots, self.store = None, [], [], None
        self.im_start, self.progress = 40, None
        self.stats = {"requests": 0, "last": {}}
        self.totals = {k: 0 for k in ("requests", "prompt_tokens", "cached_tokens", "completion_tokens", "prefill_s",
                                      "decode_s", "drafted", "accepted")}
        self.history, self.lock, self.logs = deque(maxlen=8), threading.Lock(), []

    def log(self, msg):
        self.logs.append(msg)

    def ensure_awake(self):
        pass

    def _reserve(self, n):
        self.step_now = self.prefill_step

    def _fit_output(self, n):
        return n

    def _check_memory(self, stage):
        return True

    def run(self, ids, **kw):
        out = list(self.generate(ids, **kw))
        return [t for t, _ in out[:-1] if t is not None], "".join(p for _, p in out[:-1]), out[-1]


class TestTheLoopWithoutMlx(unittest.TestCase):
    def test_the_core_names_no_array_library(self):
        src = open(os.path.join(os.path.dirname(session.__file__), "session.py")).read()
        code = "\n".join(l for l in src.split('"""')[2].splitlines() if not l.strip().startswith("#"))
        for word in ("mlx", "mx.", "numpy", "np.", "torch"):
            self.assertNotIn(word, code)
        self.assertEqual(re.findall(r"^(?:import|from) (\S+)", src, re.M),
                         ["time", ".", ".stopfilter", ".tokenizer"])

    def test_plain_decode_follows_the_model_to_its_end(self):
        e = ToyEngine(speculative=False)
        toks, text, st = e.run([5, 6, 7], max_tokens=100)
        self.assertEqual(toks, list(range(8, V)))                   # 8..49; the end-of-sequence token is not text
        self.assertEqual((st["finish_reason"], st["completion_tokens"], st["verifies"]), ("stop", V - 8 + 1, 0))
        self.assertEqual(text, "".join(f"<{t}>" for t in range(8, V)))
        self.assertEqual(e.rt.prefilled, [(0, 3)])
        self.assertGreater(e.rt.flushed, 10)                         # the loop stayed pipelined

    def test_long_prompts_are_cut_into_chunks(self):
        e = ToyEngine(speculative=False)
        e.run(list(range(1, 21)), max_tokens=2)
        self.assertEqual(e.rt.prefilled, [(0, 8), (8, 16), (16, 20)])

    def test_speculation_gives_the_same_tokens_in_fewer_passes(self):
        prompt = list(range(10, 40)) + [10, 11]        # the answer will copy 12..39 from the prompt
        plain, spec = ToyEngine(speculative=False), ToyEngine(speculative=True)
        a, _, sa = plain.run(prompt, max_tokens=24)
        b, _, sb = spec.run(prompt, max_tokens=24)
        self.assertEqual(a, b)
        self.assertEqual(a, list(range(12, 36)))
        self.assertGreater(sb["verifies"], 0)
        self.assertEqual(sb["draft_accept"], 1.0)
        self.assertLess(spec.rt.steps + spec.rt.verifies, plain.rt.steps / 2)
        self.assertEqual(spec.cache[0].toks, prompt + b)             # the cache holds exactly what was emitted

    def test_a_wrong_draft_is_rolled_back(self):
        prompt = [10, 11, 30, 31, 10, 11]              # the n-gram 10,11 was followed by 30 — the model says 12
        e = ToyEngine(speculative=True)
        toks, _, st = e.run(prompt, max_tokens=5)
        self.assertEqual(toks, [12, 13, 14, 15, 16])
        self.assertLess(st["draft_accept"] or 0, 1.0)
        self.assertEqual(e.cache[0].toks, prompt + toks)

    def test_the_next_turn_reuses_the_prefix(self):
        e = ToyEngine(speculative=False)
        first = [40, 1, 2, 3, 40, 4, 5]
        toks, _, _ = e.run(first, max_tokens=3)
        nxt = first + toks + [40, 9]
        _, _, st = e.run(nxt, max_tokens=2)
        self.assertEqual(st["cached_tokens"], len(first) + len(toks))
        _, _, st = e.run(first + [20], max_tokens=2)   # diverges after the last restore point (the final 40)
        self.assertEqual(st["cached_tokens"], 4)

    def test_a_stop_string_ends_the_answer_before_it(self):
        e = ToyEngine(speculative=False)
        toks, text, st = e.run([5], max_tokens=40, stop=["<9>"])
        self.assertEqual((text, st["finish_reason"]), ("<6><7><8>", "stop"))

    def test_cancellation_and_limits(self):
        e = ToyEngine(speculative=False)
        n = []
        toks, _, st = e.run([5], max_tokens=40, should_stop=lambda: n.append(1) or len(n) >= 4)
        self.assertEqual((len(toks), st["finish_reason"]), (4, "cancelled"))
        with self.assertRaises(session.ContextError):
            e.run(list(range(1, 40)) * 200)


if __name__ == "__main__":
    unittest.main()
