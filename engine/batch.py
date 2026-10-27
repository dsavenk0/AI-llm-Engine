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
"""
import mlx.core as mx

from .model import KVCache, Positions, SSMCache

NEG = -30000.0          # additive mask value that zeroes a key in float16 softmax


def _greedy(logits_row):
    return int(mx.argmax(logits_row).item())


def _assemble(caches, layer_idx, lmax):
    """Combine one layer's per-request caches into a batched one. Attention: left-pad KV to `lmax` and stack.
    SSM: stack the (position-independent) conv tail and recurrent state over the batch."""
    sample = caches[0][layer_idx]
    if isinstance(sample, KVCache):
        ks, vs = [], []
        for c in caches:
            kv = c[layer_idx]
            k, v = kv.keys[..., :kv.offset, :], kv.values[..., :kv.offset, :]
            pad = lmax - k.shape[2]
            if pad:
                kz = mx.zeros((1, k.shape[1], pad, k.shape[3]), k.dtype)
                vz = mx.zeros((1, v.shape[1], pad, v.shape[3]), v.dtype)
                k, v = mx.concatenate([kz, k], axis=2), mx.concatenate([vz, v], axis=2)
            ks.append(k)
            vs.append(v)
        merged = KVCache()
        merged.keys, merged.values = mx.concatenate(ks, 0), mx.concatenate(vs, 0)
        merged.offset = lmax
        return merged
    merged = SSMCache()
    merged.conv = mx.concatenate([c[layer_idx].conv for c in caches], 0) if sample.conv is not None else None
    merged.state = mx.concatenate([c[layer_idx].state for c in caches], 0) if sample.state is not None else None
    merged.offset = lmax
    return merged


def batched_generate(model, prompts, max_new=64, eos=(), sampler=_greedy, on_token=None):
    """Greedy/sampled continuation of several prompts at once.

    `model`: a loaded engine model (engine/model.py). `prompts`: a list of token-id lists. `sampler`: a
    callable (logits_row [V]) -> token id, applied per row. `on_token(i, tok)`: optional streaming callback.
    Returns a list of generated-token lists, one per prompt, in the same order (stop tokens excluded)."""
    B = len(prompts)
    if B == 0:
        return []
    eos = set(eos)
    # 1) prefill each request on its own (B=1): correct positions, and it tells us the per-request length
    per_caches, first = [], []
    for ids in prompts:
        cache = model.make_cache()
        logits = model(mx.array([list(ids)]), cache, Positions(delta=0))
        per_caches.append(cache)
        first.append(sampler(logits[0, -1]))

    lens = [len(p) for p in prompts]
    lmax = max(lens)
    n_layers = len(per_caches[0])
    cache = [_assemble(per_caches, li, lmax) for li in range(n_layers)]
    pad = mx.array([lmax - n for n in lens])                 # padding columns at the front of each row
    delta = mx.array([n - lmax for n in lens])               # per-row rope offset: cache.offset + delta == len

    outputs = [[] for _ in range(B)]
    done = [False] * B
    cur = list(first)
    for i, t in enumerate(cur):
        if t in eos:
            done[i] = True
        else:
            outputs[i].append(t)
            if on_token:
                on_token(i, t)

    for _ in range(max_new - 1):
        if all(done):
            break
        lk = cache[0].offset + 1                              # keys after this step's update
        cols = mx.arange(lk)
        mask = mx.where(cols[None, None, None, :] < pad[:, None, None, None], NEG, 0.0).astype(mx.float16)
        ids = mx.array([[t] for t in cur])                   # [B, 1]
        logits = model(ids, cache, Positions(delta=delta), mask=mask)
        nxt = [sampler(logits[b, -1]) for b in range(B)]
        for b in range(B):
            if done[b]:
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
