"""Batched decoding: run several requests through one loaded model in a single forward pass per step.

This is the Stage-1 core of concurrent serving. It is used only when more than one request is in flight;
a lone request keeps the engine's tuned single-sequence path (prefix cache, speculation) untouched.

Each request is prefilled on its own (correct positions, no padding), then the per-layer KV caches are
assembled, left-padded, into one batched cache. Decoding then runs in lockstep: every step is one forward
over `[B, 1]`, with a per-row rope offset (each row's position is its own length) and an additive mask that
hides the padding at the front of shorter rows — so the result is bit-for-bit what each request would get
alone (no cross-talk, no loss). Finished rows are carried as padding until the whole batch is done.

Works on attention and hybrid (GatedDeltaNet/SSM) native models: an attention layer's KV is left-padded and
stacked, while an SSM layer's recurrent state is position-independent, so its conv tail and state are simply
stacked over the batch. Memory is bounded by the caller (how many requests it admits at once): the batched KV
is B × max_len, so the caller caps B.

The loop below never touches an array: prefill, cache assembly, the masked lockstep step and reading a row
are runtime methods (engine/runtime.py, `batch_*`), so a second runtime batches by implementing them.
"""


def batched_generate(rt, prompts, max_new=64, eos=(), sampler=None,
                     samplers=None, should_stop=None, on_token=None):
    """Greedy/sampled continuation of several prompts at once.

    `rt`: the engine's runtime (engine/runtime.py). `prompts`: a list of token-id lists.
    `sampler`: a callable (logits_row) -> token id applied to every row, greedy by default; `samplers`: a
    per-row list that overrides it (so each request keeps its own temperature/top-p). `should_stop`: an
    optional per-row list of callables () -> bool, checked each step, to stop a row early (a client
    disconnected, a stop string matched). `on_token(i, tok)`: optional streaming callback. Returns a list of
    generated-token lists, one per prompt, in order (stop tokens excluded)."""
    B = len(prompts)
    if B == 0:
        return []
    eos = set(eos)
    samp = samplers if samplers is not None else [sampler or rt.greedy_row] * B
    stops = should_stop if should_stop is not None else [None] * B
    # 1) prefill each request on its own (B=1): correct positions, and it tells us the per-request length
    per_caches, first = [], []
    for i, ids in enumerate(prompts):
        cache, row = rt.batch_prefill(ids)
        per_caches.append(cache)
        first.append(samp[i](row))
    # 2) one left-padded batched cache; each row keeps its own positions
    state = rt.batch_join(per_caches, [len(p) for p in prompts])

    outputs = [[] for _ in range(B)]
    done = [bool(stops[i] and stops[i]()) for i in range(B)]
    cur = list(first)
    for i, t in enumerate(cur):
        if done[i] or t in eos:
            done[i] = True
        else:
            outputs[i].append(t)
            if on_token:
                on_token(i, t)

    for _ in range(max_new - 1):
        if all(done):
            break
        rows = rt.batch_step(state, cur)
        nxt = [samp[b](rows[b]) for b in range(B)]
        for b in range(B):
            if done[b]:
                continue
            if stops[b] and stops[b]():
                done[b] = True
                continue
            t = nxt[b]
            if t in eos or len(outputs[b]) >= max_new:
                done[b] = True
            else:
                outputs[b].append(t)
                if on_token:
                    on_token(b, t)
        cur = nxt
    return outputs
