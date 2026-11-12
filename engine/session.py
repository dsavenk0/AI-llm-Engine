"""The Inference Core: one request, from a list of token ids to a stream of text.

Everything that decides *what happens* during generation lives here and nowhere else — reusing a prefix,
cutting the prompt into chunks under a memory limit, the pipelined decode loop, prompt-lookup speculation
and its exactness rule, stop strings, the reasoning budget, loop detection, the statistics. Nothing here
knows what computes a forward pass: every such step is a call on `self.rt`, a runtime (engine/runtime.py
is the MLX one). A test holds this file to that — it imports no array library and names none.

What the loop asks of a runtime:

    sampler(vocab, seed, sampling)        an object carrying the sampling policy and its state
    prefill(cache, ids, …)                run a chunk of the prompt; logits for its last position
    step(token, cache, delta)             run one token; logits
    sample(sampler, logits) -> handle     pick the next token — a handle, not a value: nothing has waited yet
    flush(handle)                         start computing it without waiting
    read_token(handle) -> int             wait for it: the one synchronisation per token
    verify(token, draft, cache, delta)    run a token and a drafted continuation in one pass; logits per position
    greedy_tokens / verify_token / sample_at    read that pass: the model's own choices, and the acceptance test
    commit(cache, m)                      keep the first m tokens of a verify pass, undo the rest
    can_verify(cache)                     whether this cache can undo a verify pass (else no speculation)
    rewind(cache, n) -> bool              cut the cache back to n tokens; False when it cannot (recurrent state)
    make_cache, settle, release, peak_gb, embed_media

`sample` returning a handle is what keeps the loop pipelined: the graph for token t+1 is built while t
is still running, and the loop only blocks in `read_token`. A runtime that computed inside `sample` would
give that up (15 % of a step on MLX).

The class is a mixin: an engine supplies the state (cache, tokenizer, limits, the prefix store) and a
runtime, and gets `generate`. Both engines use it — the native one (engine/generate.py) and the mlx-lm /
GGUF one (engine/compat.py), which replaces only the prefix policy (`_restore_or_reset`), the prefill
schedule (`_prefill`) and what its tokenizer calls the reasoning end (`_think_closing`, `_detokenizer`).
"""
import time

from . import memory
from .stopfilter import StopFilter
from .tokenizer import StreamDecoder

GiB = 2 ** 30


class ContextError(ValueError):
    pass


MemoryLimitError = memory.MemoryLimitError      # raised by the RAM guard (engine/memory.py)


class FormatError(ValueError):
    """The reply cannot be continued without breaking the format the request asked for."""


class Drafter:
    """Prompt-lookup drafts: continue the most recent earlier occurrence of the
    current n-gram suffix (n = nmax..nmin) found in prompt + output."""

    def __init__(self, tokens, nmin=2, nmax=4):
        self.nmin, self.nmax = nmin, nmax
        self.toks = []
        self.index = {n: {} for n in range(nmin, nmax + 1)}
        for t in tokens:
            self.add(t)

    def add(self, t):
        L = len(self.toks)  # t lands at position L: every n-gram ending at L-1 now has a continuation
        for n in range(self.nmin, self.nmax + 1):
            if L >= n:
                self.index[n][tuple(self.toks[L - n:L])] = L
        self.toks.append(t)

    def propose(self, k):
        T = self.toks
        for n in range(self.nmax, self.nmin - 1, -1):
            if len(T) >= n:
                p = self.index[n].get(tuple(T[-n:]))
                if p is not None and p < len(T):
                    return T[p:p + k]
        return []


# Draft lengths to try. Verifying 1..4 extra tokens costs ~1.4..2.3 decode
# steps; 5..15 all cost the same (~5 steps, tiled matmul) — so jump to 15.
DRAFT_LEVELS = (1, 2, 3, 4, 15)


def is_repeating(tail, min_n=20, max_n=300, long_n=80):
    """True if the tail of `tail` is consecutive copies of some chunk — a reasoning
    model stuck repeating one phrase ("Okay, let's process the message again" x N,
    `n` short, needs 3 copies to rule out coincidence) or cycling through a few
    phrases ("A, then B, then A, then B...", `n` covers the whole A+B cycle, so 2
    exact copies of that longer chunk are already a strong loop signal)."""
    L = len(tail)
    for n in range(min(max_n, L // 2), min_n - 1, -1):
        reps = 2 if n >= long_n else 3
        if n * reps > L:
            continue
        chunks = [tail[L - (k + 1) * n: L - k * n] for k in range(reps)]
        if chunks[0].strip() and all(c == chunks[0] for c in chunks[1:]):
            return True
    return False


def is_looping(tail, min_n=6, max_n=400, window=3000):
    """True when the answer's end is one chunk repeated back to back — a small model cycling through the
    same list or sentence until the token limit. Separator lines ("-----", "=====") and other chunks made
    of a few distinct characters need far more repeats before they count."""
    tail = tail[-window:]
    L = len(tail)
    for n in range(min_n, min(max_n, L // 3) + 1):
        chunk = tail[L - n:]
        if not chunk.strip():
            continue
        k = 1
        while (k + 1) * n <= L and tail[L - (k + 1) * n: L - k * n] == chunk:
            k += 1
        plain = len(set(chunk)) <= 3
        if k >= 3 and k * n >= (1800 if plain else 300):
            return True
    return False


class DecodeLoop:
    """`generate()` and what it needs, over `self.rt`."""

    # ───────────────────────── what the engine's tokenizer says ─────────────────────────
    def _detokenizer(self):
        """What turns ids back into text, special tokens kept (the output parser reads the markers)."""
        return self.tok

    def _think_closing(self):
        """(the reasoning end marker, the ids teacher-forced to close reasoning); (None, []) for a model
        that does not reason — then the reasoning budget never applies."""
        return "</think>", self.tok.encode("\n</think>\n\n")

    # ───────────────────────── prefill ─────────────────────────
    def _prefill(self, cache, ids, embeds=None, pos3=None, delta=0, start=0, hidden=False):
        """Run ids[start:] through the model in chunks; return last-position logits, or with `hidden` the
        normed hidden state the output head reads (what `embed` turns into a vector).

        Near the RAM limit the chunk shrinks (1024 → 256 → 128): peak activation
        memory scales with it, so long prompts still go through instead of failing.
        """
        logits, n, s, step = None, len(ids), start, self.step_now
        try:
            while s < n:
                self.progress = {"phase": "prefill", "done": s, "total": n}   # /health: a long prompt is not "hung"
                s, logits, step = self._prefill_chunk(cache, ids, embeds, pos3, delta, s, n, step, logits, hidden)
        finally:
            self.progress = None
        return logits

    def _prefill_chunk(self, cache, ids, embeds, pos3, delta, s, n, step, logits, hidden=False):
        """One chunk of the prefill; returns (next start, logits, chunk size for the next one)."""
        e = min(n, s + step)
        logits = self.rt.prefill(cache, ids, embeds, pos3, delta, s, e, hidden)
        s = e
        if s < n:
            self.rt.release()
            used = memory.used_fraction()
            if used > self.mem_limit and step > 128:
                step = max(128, step // 4)
                self.log(f"memory: {used:.0%} RAM in use, prefill chunk reduced to {step}")
            elif used > self.mem_limit:
                self._check_memory("prefill")
        return s, logits, step

    def _restore_or_reset(self, ids):
        """Reuse the cache when the new prompt extends a known prefix."""
        if self.cache is not None:
            full = self.cache_tokens
            if len(ids) > len(full) and ids[:len(full)] == full:
                return len(full)
            for toks, snaps in sorted(self.snapshots, key=lambda x: -len(x[0])):
                if len(ids) > len(toks) and ids[:len(toks)] == toks:
                    for c, s in zip(self.cache, snaps):
                        c.restore(s)
                    # longer snapshots referred to KV positions that will now be overwritten
                    self.snapshots = [x for x in self.snapshots if len(x[0]) <= len(toks)]
                    return len(toks)
        if self.store:
            hit = self.store.lookup(ids)
            if hit:
                try:
                    self.cache, snaps, toks = self.store.load(hit, self.rt.model)
                except Exception as e:      # a half-written or damaged file: serve from scratch, drop the entry
                    self.log(f"prefix cache: an entry could not be read and was dropped ({e!r})")
                    self.store.discard(hit[0])
                else:
                    self.snapshots, self.cache_tokens = [(toks, snaps)], list(toks)
                    return len(toks)
        self.cache = self.rt.make_cache()
        self.cache_tokens, self.snapshots = [], []
        return 0

    # ───────────────────────── generate ─────────────────────────
    def generate(self, ids, media=None, max_tokens=None, sampling=None, stop=None, should_stop=None, seed=None,
                 thinking_open=False, fmt=None):
        """Yield (token_id, text) pairs; final item is a dict with stats.

        `fmt` turns on constrained decoding: (schema, root), where schema is a JSON Schema node or None and
        root is "object" / "any". The sampler then only ever picks a token that keeps the answer a valid
        JSON prefix, and generation ends the moment the document is whole — so the caller gets something
        parseable or an error, never prose.
        """
        self.ensure_awake()
        if len(ids) >= self.max_context:
            raise ContextError(f"prompt is {len(ids)} tokens, context limit is {self.max_context}")
        max_tokens = min(max_tokens or self.max_tokens, self.max_tokens, self.max_context - len(ids))
        rt = self.rt
        sampler = rt.sampler(self.vocab, seed, sampling or {})
        from .structured import constrain
        detok = self._detokenizer()
        fmt_done = constrain(sampler, fmt, lambda tok: detok.decode([tok]), self.eos)
        t0 = time.time()
        delta = 0
        if media is not None:
            self._drop_prefix_cache()
            self._reserve(len(ids))
            emb, pos3, delta = rt.embed_media(ids, media)
            self.cache = rt.make_cache()
            self.cache_tokens, self.snapshots = [], []
            cache, reused = self.cache, 0
            logits = self._prefill(cache, ids, embeds=emb, pos3=pos3)
        else:
            reused = self._restore_or_reset(ids)
            self._reserve(len(ids) - reused)
            cache = self.cache
            # Restore points: the end of the first message (usually the system
            # prompt — shared by every conversation of an agent) and the final
            # <|im_start|> (the generation prompt — shared by the next turn).
            starts = [i for i, t in enumerate(ids) if t == self.im_start]
            cuts = {c for c in (starts[1] if len(starts) > 1 else None, starts[-1] if starts else None)
                    if c is not None and reused < c < len(ids)}
            # A prompt that parts from a stored one in the middle of its system message (a date, a file list,
            # a session id that an agent puts there) would otherwise be prefilled whole every time: also take
            # a restore point exactly where the two diverge, so the shared part is restored next time.
            if self.store and len(ids) > self.store.min_tokens:
                shared, key = self.store.overlap(ids)
                if shared >= self.store.min_tokens and reused + 64 < shared < len(ids):
                    # a changing value sits in the middle of a line: back off to the start of that line, so
                    # the next prompt that changes the same value (a minute later) still shares the point
                    point = shared
                    for j in range(shared - 1, max(self.store.min_tokens, shared - 200), -1):
                        if "\n" in self.tok.decode([ids[j]]):
                            point = j + 1
                            break
                    if point not in cuts and reused < point:
                        cuts.add(point)
                        self.log(f"prefix cache: this prompt shares {shared} tokens with a stored one and then "
                                 f"differs — restore point added at {point}")
            cuts = sorted(cuts)
            pos = reused
            for cut in cuts:
                self._prefill(cache, ids[:cut], start=pos)
                self.snapshots.append((ids[:cut], [c.snapshot() for c in cache]))
                pos = cut
            self.snapshots = self.snapshots[-3:]
            logits = self._prefill(cache, ids, start=pos)
        self.cache_tokens = list(ids)
        t_prefill = time.time() - t0
        max_tokens = self._fit_output(max_tokens)
        think_budget = int(max_tokens * self.think_frac) if thinking_open and self.think_frac else None

        dec = StreamDecoder(detok)
        out_tokens, text, finish = [], "", "length"
        stops = StopFilter(stop)
        spec = {"verifies": 0, "drafted": 0, "accepted": 0}
        looped = [False]
        in_think = False        # (set below once the budget is known; emit() reads it when it runs)
        cache_stale = False   # set when a stop cut off tokens the state already contains
        t1 = time.time()

        def emit(tok):
            """Returns (stop, piece). Handles eos / stop strings / cancellation.

            A stop string is never part of the answer (OpenAI semantics): the text is cut where it
            begins, and while the tail of the text could still turn into a stop string it is held back
            instead of being streamed."""
            nonlocal text, finish
            out_tokens.append(tok)
            if tok in self.eos:
                finish = "stop"
                return True, None
            piece = dec.add(tok)
            text += piece
            hit, piece = stops.feed(piece)
            if hit:
                finish = "stop"
                return True, piece
            if fmt_done and fmt_done.pop(0):
                finish = "stop"
                return True, piece
            if should_stop and should_stop():
                finish = "cancelled"
                return True, piece
            if len(out_tokens) % 32 == 0 and not self._check_memory("decode"):
                finish = "length"
                self.log(f"memory: stopped generation at {memory.used_fraction():.0%} RAM")
                return True, piece
            if len(out_tokens) % 16 == 0 and not in_think and is_looping(
                    text[text.rfind(think_end) + len(think_end):] if think_end and think_end in text else text):
                finish, looped[0] = "stop", True
                self.log(f"repetition: the answer was looping; stopped after {len(out_tokens)} tokens")
                return True, piece
            return len(out_tokens) >= max_tokens, piece

        # Reasoning models can fail to ever emit </think> — budget how many tokens the
        # reasoning phase may use, and bail out of an exact repeating loop even sooner.
        think_end, closing = self._think_closing()
        in_think = think_budget is not None
        forced = None   # why the reasoning was force-closed: "budget" | "repetition" (None = never)

        def _think_check():
            """True once the reasoning phase has run long enough (budget) or is visibly
            stuck repeating itself, and should be force-closed."""
            nonlocal in_think, forced
            if not in_think:
                return False
            if think_end in text:
                in_think = False
                return False
            if think_budget and len(out_tokens) >= think_budget:
                forced = "budget"
                return True
            if is_repeating(text[-700:]):
                forced = "repetition"
                return True
            return False

        def _force_close_think():
            """Teacher-force a `</think>` and hand back fresh logits to resume sampling from,
            so the model actually sees that reasoning ended instead of being cut off cold."""
            logits_ = None
            for cid in closing:
                logits_ = rt.step(cid, cache, delta)
                fed.append(cid)
                if drafter is not None:
                    drafter.add(cid)
                done_, piece_ = emit(cid)
                if piece_:
                    yield cid, piece_
                if done_:
                    return None
            return logits_

        drafter = None
        if not (self.speculative and media is None and fmt_done is None and rt.can_verify(cache)):
            # plain pipelined decode: graph for token t+1 is built while t runs
            y = rt.sample(sampler, logits)
            rt.flush(y)
            fed = list(ids)
            for _ in range(max_tokens):
                nxt = rt.sample(sampler, rt.step(y, cache, delta))
                rt.flush(nxt)
                tok = rt.read_token(y)
                fed.append(tok)
                done, piece = emit(tok)
                if piece:
                    yield tok, piece
                if done:
                    break
                if closing and _think_check():
                    in_think = False
                    logits = yield from _force_close_think()
                    if logits is None:
                        break
                    y = rt.sample(sampler, logits)
                    rt.flush(y)
                else:
                    y = nxt
        else:
            # Prompt-lookup speculative decode with exact acceptance. Plain steps
            # stay pipelined; verification kicks in only while the output copies
            # text that already exists (code edits, quotes, tool arguments).
            drafter = Drafter(ids, nmin=self.ngram_min, nmax=max(4, self.ngram_min))
            fed = list(ids)
            y = rt.sample(sampler, logits)
            rt.flush(y)
            lvl, spec_mode, done = 1, False, False
            while not done:
                if not spec_mode:
                    nxt = rt.sample(sampler, rt.step(y, cache, delta))
                    rt.flush(nxt)
                    tok = rt.read_token(y)
                    drafter.add(tok)
                    fed.append(tok)
                    done, piece = emit(tok)
                    if piece:
                        yield tok, piece
                    if not done and closing and _think_check():
                        in_think = False
                        logits = yield from _force_close_think()
                        if logits is None:
                            done = True
                        else:
                            y = rt.sample(sampler, logits)
                            rt.flush(y)
                    else:
                        y = nxt
                    spec_mode = bool(drafter.propose(1))  # current suffix seen before → likely copying
                    continue

                tok = rt.read_token(y)
                if tok in self.eos:
                    out_tokens.append(tok)
                    finish = "stop"
                    break
                drafter.add(tok)
                room = max_tokens - len(out_tokens) - 1
                draft = drafter.propose(self.draft_levels[lvl])[:max(0, room)]
                if not draft:  # back to pipelined plain decode
                    y = rt.sample(sampler, rt.step(tok, cache, delta))
                    rt.flush(y)
                    fed.append(tok)
                    done, piece = emit(tok)
                    if piece:
                        yield tok, piece
                    if not done and closing and _think_check():
                        in_think = False
                        logits = yield from _force_close_think()
                        if logits is None:
                            done = True
                        else:
                            y = rt.sample(sampler, logits)
                            rt.flush(y)
                    spec_mode = False
                    continue
                logits = rt.verify(tok, draft, cache, delta)
                acc, nxt = 0, None
                if sampler.greedy:
                    best = rt.greedy_tokens(logits)
                    while acc < len(draft) and best[acc] == draft[acc]:
                        acc += 1
                    nxt = best[acc]
                else:
                    for i, d in enumerate(draft):
                        ok, t_ = rt.verify_token(sampler, logits, i, d)
                        if not ok:
                            nxt = t_
                            break
                        acc += 1
                    if nxt is None:
                        nxt = rt.sample_at(sampler, logits, len(draft))
                rt.commit(cache, 1 + acc)
                spec["verifies"] += 1
                spec["drafted"] += len(draft)
                spec["accepted"] += acc
                full = acc == len(draft) == self.draft_levels[lvl]
                lvl = min(lvl + 1, len(self.draft_levels) - 1) if full else max(0, min(lvl, acc) - (acc == 0))
                for j, t_ in enumerate([tok] + draft[:acc]):
                    if j:
                        drafter.add(t_)
                    fed.append(t_)
                    done, piece = emit(t_)
                    if piece:
                        yield t_, piece
                    if done:
                        if j < acc:       # accepted tokens after the stop are in the state but not in `fed`
                            cache_stale = True
                        break
                if not done and closing and _think_check():
                    in_think = False
                    logits = yield from _force_close_think()
                    if logits is None:
                        done = True
                    else:
                        y = rt.sample(sampler, logits)
                        rt.flush(y)
                else:
                    y = nxt
        rest = stops.flush()                  # held back as a possible stop string that never completed
        if rest:
            yield None, rest
        tail = dec.flush()
        if tail and finish == "length":
            yield None, tail
        rt.settle(cache)
        self.cache_tokens = fed
        if cache_stale and not rt.rewind(cache, len(fed)):
            self.cache, self.cache_tokens = None, []     # the recurrent state ran past `fed` and cannot be trimmed
        if media is not None:
            self.cache = None  # 3-D positions: not reusable as a text prefix
        t_dec = time.time() - t1
        n_out = len(out_tokens)
        st = {
            "prompt_tokens": len(ids), "cached_tokens": reused, "completion_tokens": n_out,
            "prefill_s": round(t_prefill, 3), "decode_s": round(t_dec, 3),
            "prefill_tps": round((len(ids) - reused) / t_prefill, 1) if t_prefill > 0 else 0,
            "decode_tps": round(n_out / t_dec, 2) if t_dec > 0 else 0,
            "finish_reason": finish, "think_forced": forced, "loop_stopped": looped[0],
            "peak_gb": rt.peak_gb(),
            "draft_accept": round(spec["accepted"] / spec["drafted"], 2) if spec["drafted"] else None,
            "verifies": spec["verifies"],
            "ram_used": round(memory.used_fraction(), 3),
        }
        self.stats["requests"] += 1
        self.stats["last"] = st
        T_ = self.totals
        T_["requests"] += 1
        T_["prompt_tokens"] += st["prompt_tokens"]
        T_["cached_tokens"] += st["cached_tokens"]
        T_["completion_tokens"] += st["completion_tokens"]
        T_["prefill_s"] += st["prefill_s"]
        T_["decode_s"] += st["decode_s"]
        T_["drafted"] += spec["drafted"]
        T_["accepted"] += spec["accepted"]
        self.history.append({"t": round(time.time()), "prompt": st["prompt_tokens"], "cached": st["cached_tokens"],
                             "out": st["completion_tokens"], "prefill_tps": st["prefill_tps"],
                             "decode_tps": st["decode_tps"], "accept": st["draft_accept"]})
        rt.release()
        yield st
