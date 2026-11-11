"""Replaceable compute kernels: one operation, several implementations, chosen at run time.

This is the seam the engine's own Metal code plugs into. An *operation* (say, one decode step of
attention over the KV cache) has a reference implementation built from MLX primitives and, optionally,
others — a hand-written Metal kernel today, a native C/CPU one later. The model code asks for the
operation by name and never learns which implementation it got.

Rules an implementation has to meet (they are what makes a replacement safe to ship):

  * same signature and same result as the reference, within the tolerance its test states;
  * it declines what it does not handle by returning None, and the caller then runs the reference —
    so a kernel can cover only the hot case (one sequence, one new token) and stay small;
  * it has a row in `tests/bench_kernels.py`, measured against the reference on the same inputs.

Choosing:

    LLM_KERNELS=mlx                    every operation on its MLX reference (the comparison baseline)
    LLM_KERNELS=attn_decode=metal      one operation pinned to one implementation
    LLM_KERNELS=auto  (or unset)       each operation's default — the measured winner, else the reference

An implementation that is unavailable on this machine, or whose kernel fails to build, is skipped and the
reference runs: selection never turns into an error at request time.
"""
import os

REFERENCE = "mlx"
_ops = {}            # op -> {impl name: (fn, available)}
_default = {}        # op -> impl name used by "auto"
_cache = {}          # op -> resolved (name, fn)


def register(op, name, fn, available=None, default=False):
    """Add implementation `name` of operation `op`. `available()` says whether it can run here;
    `default=True` makes it what "auto" picks — set it only with a benchmark that says so."""
    _ops.setdefault(op, {})[name] = (fn, available or (lambda: True))
    if default:
        _default[op] = name
    _cache.pop(op, None)


def _requested(op):
    spec = os.environ.get("LLM_KERNELS", "auto").strip()
    if not spec or spec == "auto":
        return None
    if "=" not in spec:
        return spec                      # one name for every operation, e.g. "mlx"
    for part in spec.split(","):
        k, _, v = part.partition("=")
        if k.strip() == op:
            return v.strip()
    return None


def resolve(op):
    """(implementation name, callable) for `op`, honouring LLM_KERNELS, falling back to the reference."""
    hit = _cache.get(op)
    if hit is not None:
        return hit
    impls = _ops[op]
    want = _requested(op) or _default.get(op, REFERENCE)
    if want not in impls or not impls[want][1]():
        want = REFERENCE
    _cache[op] = (want, impls[want][0])
    return _cache[op]


def get(op):
    return resolve(op)[1]


def reference(op):
    return _ops[op][REFERENCE][0]


def implementations(op):
    """{name: available?} — what the benchmark and the health report iterate over."""
    return {name: bool(avail()) for name, (_, avail) in _ops[op].items()}


def active():
    """{op: implementation name in use} for every registered operation."""
    return {op: resolve(op)[0] for op in sorted(_ops)}


def reset():
    """Forget resolved choices (after changing LLM_KERNELS in a test or a benchmark)."""
    _cache.clear()


from . import attn_decode  # noqa: E402,F401  (registers itself)
