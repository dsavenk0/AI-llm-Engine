"""Pipeline-parallel inference: the model's layers split across hosts, the hidden state passed along.

Why a pipeline and not tensor parallelism. Between hosts a pipeline sends one hidden state per token —
hidden_size × 2 bytes, a few kilobytes — while tensor parallelism needs an all-reduce inside every layer.
Thunderbolt 5 is about 10 GB/s against 120+ GB/s of local memory here (far more on a Studio), so the link
is ten to forty times slower than memory: kilobytes per token do not notice that, per-layer all-reduces do.
The recurrent layers help too — GatedDeltaNet state is per layer, so it shards with the layers.

What this buys, and what it does not. A pipeline does not make one request faster: its latency is the sum of
the hosts plus the link. It makes a model *fit* that does not fit in one machine, which is the whole point
for 120B-class weights. Said plainly because the opposite is easy to assume.

Running it (MLX's ring backend, one process per rank):

    MLX_HOSTFILE=hosts.json MLX_RANK=0 ./llm serve …        # on the first Mac
    MLX_HOSTFILE=hosts.json MLX_RANK=1 ./llm serve …        # on the second

where hosts.json is a JSON array of "host:port" entries, one per rank. On one machine the same thing works
over loopback, which is how the split is tested without a second Mac: correctness is provable here, the
memory benefit is not, because both ranks then share the same RAM.
"""
import mlx.core as mx


def init(backend="any"):
    """The communication group, or None when this process is on its own.

    MLX decides from the environment: the ring backend wants MLX_RANK and MLX_HOSTFILE, jaccl wants
    JACCL_RANK and a device file. With neither set there is no cluster and every caller falls back to the
    ordinary single-process path, which is what makes this safe to call unconditionally.
    """
    if not mx.distributed.is_available():
        return None
    try:
        group = mx.distributed.init(strict=True, backend=backend)
    except Exception:
        return None                      # nothing configured: not an error, just not a cluster
    return group if group.size() > 1 else None


def shard(n_layers, rank, size):
    """The half-open range of layers this rank owns.

    Contiguous, because a pipeline passes the hidden state forward and a rank that owned scattered layers
    would have to pass it back and forth. The remainder goes to the earliest ranks, so with 25 layers over
    4 ranks the split is 7/6/6/6 rather than 6/6/6/7 — the first rank also does the embedding, but that is
    a gather of one row, far cheaper than a layer.
    """
    if size <= 1:
        return (0, n_layers)
    base, extra = divmod(n_layers, size)
    start = rank * base + min(rank, extra)
    return (start, start + base + (1 if rank < extra else 0))


def forward(model, cache, group, ids=None, embeds=None, pos=None, mask=None, last_only=True):
    """One forward pass spread across the group; every rank returns the logits.

    The order of sends and receives is the same on every rank, which is what keeps it from deadlocking: a
    rank receives from the one before it (rank 0 embeds instead), runs its own layers, and either passes the
    hidden state on or — being last — applies the head and sends the logits back to rank 0.

    `cache` holds only this rank's layers, in the same order as its slice.
    """
    from .model import Positions
    rank, size = group.rank(), group.size()
    lm = model.language_model
    layers = lm.model.layers
    lo, hi = shard(len(layers), rank, size)
    pos = pos or Positions()

    if rank == 0:
        h = model.embed(ids) if embeds is None else embeds
    else:
        shape = (embeds.shape if embeds is not None else (1, len(ids[0]), model.args["hidden_size"]))
        h = mx.distributed.recv(list(shape), mx.float16, rank - 1, group=group)

    for layer, c in zip(layers[lo:hi], cache):
        h = layer(h, model.ops, c, pos, mask)

    if rank == size - 1:
        if last_only:
            h = h[:, -1:, :]
        head = lm.lm_head if "lm_head" in lm else lm.model.embed_tokens
        from .model import _w
        logits = model.ops.head(h, lm.model.norm.weight, *_w(head))
        out = mx.distributed.send(logits.astype(mx.float16), 0, group=group) if size > 1 else logits
        mx.eval(out, logits)
        return logits
    mx.eval(mx.distributed.send(h.astype(mx.float16), rank + 1, group=group))
    rows = 1 if last_only else h.shape[1]
    logits = mx.distributed.recv([h.shape[0], rows, model.args["vocab_size"]], mx.float16,
                                 size - 1, group=group)
    mx.eval(logits)
    return logits
