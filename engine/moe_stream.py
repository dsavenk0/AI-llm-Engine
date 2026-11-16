"""Mixture-of-Experts models bigger than memory: the experts stay on the SSD and are read as the router asks.

A MoE model has most of its weights in experts, and a token uses only a few of them: Qwen3-30B-A3B holds 30 B
parameters and touches 3 B per token. The attention, the router, the embeddings and the shared parts are small
and always needed; the experts are large and mostly idle. So the small part is loaded as usual and each layer
gets a *pool* — room for C of its E experts — filled on demand from the checkpoint file.

It is exact. Whatever the router picks is made resident before it is used, so the output is the same as with
every expert in memory; only the speed differs. (Some projects instead restrict routing to the experts that
happen to be resident, which is faster and changes the model's answers; this one never does.)

What makes the reads affordable, taken from what Colibri, mlx-moe and kimi-k3-in-c report:

* **One positional read per expert tensor.** An expert's gate, up and down matrices are each a contiguous slab
  of the stacked tensor in the safetensors file, so a miss is three `preadv`s at known offsets — queued
  together, including when only one expert misses — with no decoding: the bytes go straight into the pool in
  the quantized layout the matmul takes.
* **A pool per layer that keeps what is used often.** Eviction takes the expert with the lowest decayed use
  count, the least recently used among equals, so the hot experts of a conversation stay and a passing one does
  not push them out.
* **The hot set outlives the process.** Counts are saved next to the model's cache and used to fill the pools
  before the first token, so a model that has been used starts warm instead of paying its first minutes in
  misses.
* **A big prompt is read once per expert.** Prefill sorts its tokens by expert; when one call needs more
  distinct experts than a pool holds, it is split into the fewest groups that fit, each expert loaded once.

* **The next layer's experts can be read while this one computes** (`LLM_MOE_PREFETCH=auto`, decode only; off by
  default until it measurably helps — see `prefetch_mode`). The next
  layer's router, applied to this layer's input, names the experts it will most likely pick — the prediction
  InferSSD makes (its `build_moe_prefetch_hint`), and exact to set up here: an RMSNorm only scales by its weight,
  so the next layer's normed input is this one's times w_next / w_this. Reads go to CPU buffers on their own
  threads, beside this layer's demand reads; a buffer enters the pool only when the router really asks for that
  expert, on the MLX thread, so a wrong guess costs a read and never a resident expert. Which reads are worth
  issuing follows InferSSD's value of a speculative read (`place/runtime-policy.h`): the chance the rank's guess
  is used, against how long the drive has been idle before the next layer asked.

Not done: reading with O_DIRECT. Every step still waits for whatever was neither resident nor prefetched; the
pool hit rate and the prefetch counters in `stats()` say how often that was.

The synchronisation is the price: reading which experts the router chose forces everything before it to
finish, once per layer per token, where an all-resident model lets MLX run ahead. With hits that costs a
fraction of a millisecond a layer; with misses the disk dominates anyway.
"""
import glob
import json
import os
import re
import struct
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import mlx.core as mx
import mlx.nn as nn
import numpy as np

GiB = 2 ** 30
DTYPES = {"F32": (np.float32, mx.float32), "F16": (np.float16, mx.float16), "BF16": (np.uint16, mx.bfloat16),
          "U32": (np.uint32, mx.uint32), "U8": (np.uint8, mx.uint8), "I32": (np.int32, mx.int32),
          "I8": (np.int8, mx.int8)}
DECAY = 0.9995            # per use of a layer: old routing fades, a conversation's own experts win
READERS = 4
PREFETCH_READERS = 2      # speculative reads: bounded, and on their own threads so demand reads never queue behind them
# routers whose choice is the top-k of their gate's logits (softmax or not, the order is the same): the next layer's
# picks can be predicted from its gate alone. Others (DeepSeek's grouped sigmoid with a bias, …) get no prediction.
TOPK_ROUTERS = {"Qwen3MoeSparseMoeBlock", "Qwen2MoeSparseMoeBlock", "OlmoeSparseMoeBlock", "MixtralSparseMoeBlock",
                "Qwen3NextSparseMoeBlock"}
# the names mlx-lm gives the stacked expert projections of a layer
GLU = ("gate_proj", "up_proj", "down_proj")
MLP = ("fc1", "fc2")


class StreamError(RuntimeError):
    pass


# ───────────────────────────── the checkpoint's files ─────────────────────────────
def index(folder):
    """{tensor name: (file, absolute offset, byte count, dtype name, shape)} from the safetensors headers."""
    out = {}
    for path in sorted(glob.glob(os.path.join(folder, "*.safetensors"))):
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(n))
        base = 8 + n
        for name, t in header.items():
            if name != "__metadata__":
                a, b = t["data_offsets"]
                out[name] = (path, base + a, b - a, t["dtype"], tuple(t["shape"]))
    return out


def expert_bytes(folder, idx=None):
    """(bytes held in stacked expert tensors, bytes of everything else) of a checkpoint."""
    idx = idx or index(folder)
    experts = sum(v[2] for k, v in idx.items() if _is_expert(k, v[4]))
    return experts, sum(v[2] for v in idx.values()) - experts


_PER_EXPERT = re.compile(r"\.experts\.\d+\.")


def _is_expert(name, shape):
    """A tensor of the routed experts: stacked ([E, …] under switch_mlp) or one expert's own (experts.N.…)."""
    if "shared" in name:
        return False
    return (len(shape) >= 3 and ".switch_mlp." in name) or bool(_PER_EXPERT.search(name))


# what each checkpoint calls an expert's matrices, by the name mlx-lm gives the module
ALIASES = {"gate_proj": ("gate_proj", "w1"), "up_proj": ("up_proj", "w3"), "down_proj": ("down_proj", "w2"),
           "fc1": ("fc1", "w1"), "fc2": ("fc2", "w2")}


def find_tensors(idx, path, proj, n_experts):
    """Where the weights of projection `proj` of the experts layer `path` are in the files, whichever way the
    checkpoint stores them: one stacked tensor ([E, …] — an entry), or one tensor per expert (a list of E
    entries, in expert order; mlx-lm stacks those when it loads, which is why it needs this to look past it).
    {"weight": …, "scales": …, "biases": …} with the parts that exist."""
    base = path.rsplit(".", 1)[0] + ".experts."
    out = {}
    for part in ("weight", "scales", "biases"):
        stacked = idx.get(f"{path}.{proj}.{part}")
        if stacked is not None:
            out[part] = stacked
            continue
        for alias in ALIASES.get(proj, (proj,)):
            each = [idx.get(f"{base}{e}.{alias}.{part}") for e in range(n_experts)]
            if all(t is not None for t in each):
                if len({t[3:] for t in each}) != 1:
                    raise StreamError(f"{base}*.{alias}.{part}: the experts' tensors differ in type or shape")
                out[part] = each
                break
    return out


class Files:
    """Open files for positional reads; one descriptor per file, shared by the reader threads.

    `direct` (LLM_MOE_DIRECT=1) reads around the system's file cache (F_NOCACHE): what is read is not kept there.
    Off by default — when the experts fit in the page cache they are served from memory, which is the fast case —
    and on for measuring the disk itself, or for a model so large that its experts would only push everything
    else out of the cache."""

    def __init__(self, direct=None):
        self.fds = {}
        self.lock = threading.Lock()
        self.direct = os.environ.get("LLM_MOE_DIRECT") == "1" if direct is None else direct

    def _fd(self, path):
        fd = self.fds.get(path)
        if fd is not None:
            return fd
        # Several reader threads can see a new shard together. Open it once: leaking the losing descriptor is
        # especially costly for checkpoints split into many files, and applying F_NOCACHE to only one of two
        # racing descriptors would make the selected I/O policy nondeterministic.
        with self.lock:
            fd = self.fds.get(path)
            if fd is None:
                fd = os.open(path, os.O_RDONLY)
                try:
                    if self.direct:
                        import fcntl
                        fcntl.fcntl(fd, getattr(fcntl, "F_NOCACHE", 48), 1)
                except Exception:
                    os.close(fd)
                    raise
                self.fds[path] = fd
        return fd

    def read(self, path, offset, nbytes):
        buf = os.pread(self._fd(path), nbytes, offset)
        if len(buf) != nbytes:
            raise StreamError(f"{path}: wanted {nbytes} bytes at {offset}, the file gave {len(buf)} (truncated?)")
        return buf

    def read_into(self, path, offset, out):
        """Fill the writable buffer `out` from the file at `offset` — no intermediate copy."""
        view = memoryview(out).cast("B")
        n = os.preadv(self._fd(path), [view], offset)
        if n != len(view):
            raise StreamError(f"{path}: wanted {len(view)} bytes at {offset}, the file gave {n} (truncated?)")

    def close(self):
        for fd in self.fds.values():
            try:
                os.close(fd)
            except OSError:
                pass
        self.fds = {}


# ───────────────────────────── one layer's experts ─────────────────────────────
class LayerPool:
    """Room for `capacity` of a layer's `n_experts` experts, for every projection, filled on demand.

    The slot an expert lives in is the same in every projection, so one routing decision is one set of slots
    and a miss is one expert's three slabs read together."""

    def __init__(self, files, tensors, names, n_experts, capacity, readers, usage=None):
        """`tensors`: {projection name: {"weight": T, "scales": T, "biases": T}} where T is the entry
        (path, offset, nbytes, dtype, shape) of one stacked tensor, or a list of one such entry per expert."""
        self.files, self.names, self.E = files, names, n_experts
        self.capacity = max(1, min(capacity, n_experts))
        self.readers = readers
        self.slab, self.parts, self.arrays = {}, {}, {}
        for p in names:
            self.parts[p] = [k for k in ("weight", "scales", "biases") if k in tensors[p]]
            for k in self.parts[p]:
                t = tensors[p][k]
                path, off, nbytes, dt, shape = t[0] if isinstance(t, list) else t
                npdt, mxdt = DTYPES[dt]
                if isinstance(t, list):                               # one tensor per expert
                    if len(t) != n_experts:
                        raise StreamError(f"{p}.{k}: {len(t)} expert tensors, {n_experts} experts")
                    size, slab_shape = nbytes, shape
                    where = (lambda e, t=t: (t[e][0], t[e][1]))
                else:                                                  # one stacked tensor
                    if shape[0] != n_experts:
                        raise StreamError(f"{p}.{k}: first dimension {shape[0]} is not the {n_experts} experts")
                    size, slab_shape = nbytes // n_experts, shape[1:]
                    where = (lambda e, path=path, off=off, size=size: (path, off + e * size))
                self.slab[p, k] = (size, npdt, mxdt, slab_shape, where)
                self.arrays[p, k] = mx.zeros((self.capacity,) + tuple(slab_shape), dtype=mxdt)
        self.resident = {}                       # expert -> slot
        self.free = list(range(self.capacity - 1, -1, -1))
        self.score = np.zeros(n_experts, np.float32) if usage is None else np.asarray(usage, np.float32).copy()
        self.tick = np.zeros(n_experts, np.int64)
        self.now = 0
        self.hits = self.misses = self.bytes_read = 0
        self.read_ops = self.evictions = 0
        self.read_s = 0.0
        # speculative reads: expert -> (buffers {key: row}, future). CPU memory only; the MLX pool is not touched
        # until ensure() is asked for that expert. Owned by the thread that calls ensure()/prefetch().
        self.staged = {}
        self.prefetch_issued_at = None
        self.prefetch_used = self.prefetch_late = self.prefetch_unused = 0
        self.prefetch_bytes = 0

    @property
    def nbytes(self):
        return sum(a.nbytes for a in self.arrays.values())

    @property
    def expert_nbytes(self):
        return sum(s[0] for s in self.slab.values())

    def _read_slab(self, job):
        """Read one expert tensor straight into its row of the staging block."""
        key, i, e, blocks = job
        path, off = self.slab[key][4](e)
        self.files.read_into(path, off, blocks[key][i])

    def _read_expert(self, e, bufs):
        """A speculative read: every tensor of expert `e` into its own CPU buffers (one expert per job, like
        InferSSD's readers), timed for the policy."""
        t0 = time.perf_counter()
        for key, buf in bufs.items():
            path, off = self.slab[key][4](e)
            self.files.read_into(path, off, buf)
        return time.perf_counter() - t0

    def prefetch(self, experts, executor):
        """Start reading `experts` (best first) that are neither resident nor already on their way. Returns the
        experts issued. Called on the MLX thread right after a layer's own demand reads."""
        issued = []
        for e in experts:
            e = int(e)
            if e in self.resident or e in self.staged or not 0 <= e < self.E:
                continue
            bufs = {key: np.empty(tuple(s[3]), dtype=s[1]) for key, s in self.slab.items()}
            self.staged[e] = (bufs, executor.submit(self._read_expert, e, bufs))
            issued.append(e)
        self.prefetch_issued_at = time.perf_counter()
        return issued

    def _wait_staged(self, e, bufs, fut):
        """The buffers of a speculative read of `e` once it has landed; None when it failed (the caller then reads
        it on demand — a failed guess never fails the step)."""
        if not fut.done():
            self.prefetch_late += 1
        try:
            fut.result()
        except Exception:
            return None
        self.prefetch_used += 1
        self.prefetch_bytes += sum(b.nbytes for b in bufs.values())
        return bufs

    def drop_staged(self):
        """Forget the guesses this step did not use. A read still running finishes into a buffer nobody keeps."""
        self.prefetch_unused += len(self.staged)
        self.staged = {}

    def warm(self, order):
        """Fill the pool with these experts, best first (the saved hot set)."""
        self.ensure([e for e in order[:self.capacity]], count=False)

    def ensure(self, experts, count=True):
        """Make every expert of `experts` resident; returns {expert: slot}. They are all protected from
        eviction, so a call never loses one it asked for to make room for another it asked for."""
        want = [int(e) for e in experts]
        if len(want) > self.capacity:
            raise StreamError(f"{len(want)} distinct experts in one step, the pool holds {self.capacity}")
        self.now += 1
        miss = [e for e in want if e not in self.resident]
        if count:
            self.hits += len(want) - len(miss)
            self.misses += len(miss)
        if miss:
            protect = set(want)
            slots = {}
            for e in miss:
                if self.free:
                    slots[e] = self.free.pop()
                else:
                    victim = min((x for x in self.resident if x not in protect),
                                 key=lambda x: (self.score[x], self.tick[x]))
                    slots[e] = self.resident.pop(victim)
                    self.evictions += 1
            t0 = time.perf_counter()
            staged = {e: self.staged.pop(e) for e in miss if e in self.staged}
            fresh = [e for e in miss if e not in staged]
            blocks = {key: np.empty((len(fresh),) + tuple(s[3]), dtype=s[1]) for key, s in self.slab.items()}
            # Keep adjacent expert slabs of one projection next to each other in the queue. This preserves
            # locality in a stacked safetensors tensor while still allowing gate/up/down of a single miss to
            # occupy separate readers — the common decode path used to serialize those reads.
            jobs = [(key, i, e, blocks) for key in self.slab for i, e in enumerate(fresh)]
            pending = self.readers.map(self._read_slab, jobs) if len(jobs) > 1 else None
            # the demand reads are running; meanwhile take the guesses that landed (or wait for those still going)
            landed = []
            for e, (bufs, fut) in staged.items():
                got = self._wait_staged(e, bufs, fut)
                if got is None:                                    # a failed speculative read: read it now
                    one = {key: np.empty((1,) + tuple(s[3]), dtype=s[1]) for key, s in self.slab.items()}
                    for key in self.slab:
                        self._read_slab((key, 0, e, one))
                    got = {key: b[0] for key, b in one.items()}
                landed.append((e, got))
            if pending is not None:
                list(pending)
            elif jobs:
                self._read_slab(jobs[0])
            self.read_ops += len(jobs)
            self.read_s += time.perf_counter() - t0
            for group, rows in ((fresh, None), ([e for e, _ in landed], [b for _, b in landed])):
                if not group:
                    continue
                order = mx.array(np.asarray([slots[e] for e in group], np.int32))
                for key, s in self.slab.items():
                    # one copy into MLX either way: the demand block whole, a guess expert by expert
                    block = mx.array(blocks[key]) if rows is None else mx.stack([mx.array(r[key]) for r in rows])
                    if s[2] == mx.bfloat16:
                        block = block.view(mx.bfloat16)
                    self.arrays[key][order] = block                 # one update per source for all the misses
                    self.bytes_read += block.nbytes
            for e, s in slots.items():
                self.resident[e] = s
        for e in want:
            self.tick[e] = self.now
        return {e: self.resident[e] for e in want}

    def use(self, counts):
        """Note how often each expert was chosen: {expert: times}. The score decays, so it follows the
        conversation rather than the whole history."""
        self.score *= DECAY
        for e, c in counts.items():
            self.score[e] += c

    def stats(self):
        n = self.hits + self.misses
        return {"hits": self.hits, "misses": self.misses, "hit_rate": round(self.hits / n, 4) if n else None,
                "bytes_read": self.bytes_read, "read_s": round(self.read_s, 3), "resident": len(self.resident),
                "capacity": self.capacity, "read_ops": self.read_ops, "evictions": self.evictions,
                "prefetch_used": self.prefetch_used, "prefetch_late": self.prefetch_late,
                "prefetch_unused": self.prefetch_unused}


# ───────────────────────────── which speculative reads are worth it ─────────────────────────────
def prefetch_mode():
    """off: demand reads only · on: read every novel guess within the depth · auto: only the reads the drive's
    measured idle time makes worth it (InferSSD's value of a speculative read).

    Off by default: on OLMoE-1B-7B (M5, 16 GB, paired runs, experts both from the page cache and from the SSD) the
    guesses were 85 % right but the end-to-end speed did not improve beyond run-to-run noise. A model whose layers
    take longer to compute leaves the reads a longer window; measure with tests/bench_moe.py --prefetch before
    switching it on."""
    m = os.environ.get("LLM_MOE_PREFETCH", "off").lower()
    return m if m in ("off", "on", "auto") else "off"


def read_value(p, start_us, read_us, windows):
    """What a speculative read is worth, in reads of it — InferSSD's `prefetch_read_value` (place/runtime-policy.h,
    Apache-2.0, revision ca8db2e), re-derived for this pool, where a guess takes no resident slot: the read that
    lands before the next layer asks saves a read with probability p; one still running when it asks has to be
    waited for, saving the share it had done (p · done) and costing the rest of the drive's time if the guess was
    wrong; one not started yet is waited for whole. No window measured yet: p."""
    if not windows:
        return p
    value = 0.0
    for w in windows:
        if w >= start_us + read_us:
            value += p
        elif w > start_us:
            value += p - (start_us + read_us - w) / read_us
        else:
            value += p - 1
    return value / len(windows)


def best_run(worth):
    """How many reads, taken in order, to keep: the prefix with the largest summed worth (none if none is > 0)."""
    total = best_total = 0.0
    best = 0
    for i, w in enumerate(worth):
        total += w
        if total > best_total:
            best_total, best = total, i + 1
    return best


class _Link:
    """What a layer needs to guess the next layer's picks. A plain object, so MLX never walks it as a module's
    parameters or children."""
    __slots__ = ("pool", "gate", "ratio", "k", "policy", "executor")

    def __init__(self, pool, gate, ratio, k, policy, executor):
        self.pool, self.gate, self.ratio, self.k = pool, gate, ratio, k
        self.policy, self.executor = policy, executor


class PrefetchPolicy:
    """Learned state of the speculative reads of one loaded model: per rank of the guess, how often a novel guess
    was really picked; how long one expert's read takes; how long the drive stayed idle before the next layer
    asked. Small, bounded, and touched only by the MLX thread (read times come back through futures)."""

    def __init__(self, top_k, mode=None, depth=None):
        self.mode = mode or prefetch_mode()
        self.depth = max(0, min(int(os.environ.get("LLM_MOE_PREFETCH_DEPTH", depth or top_k)), 64))
        self.novel = np.zeros(64, np.int64)
        self.used = np.zeros(64, np.int64)
        self.read_us = 0.0
        self.windows = []
        self.issued = self.declined = self.predicted = 0
        self._pending = []            # (pool, {expert: rank}) of the last guess, checked against the real pick
        self._timed = []              # futures whose read time is still to be learned

    def use_probability(self, rank):
        return (self.used[rank] + 1.0) / (self.novel[rank] + 2.0)

    def observe_window(self, us):
        self.windows.append(us)
        if len(self.windows) > 256:
            del self.windows[:-256]

    def observe_routes(self, pool, picked):
        """The layer a guess was for has now picked: credit the novel guesses that were right."""
        keep = []
        for p, ranks in self._pending:
            if p is pool:
                for e, r in ranks.items():
                    self.used[r] += e in picked
            else:
                keep.append((p, ranks))
        self._pending = keep
        done = [f for f in self._timed if f.done()]
        self._timed = [f for f in self._timed if not f.done()]
        for f in done:
            try:
                us = f.result() * 1e6
            except Exception:
                continue
            self.read_us = us if not self.read_us else 0.9 * self.read_us + 0.1 * us

    def choose(self, pool, hint):
        """The experts of `hint` (the next layer's guess, best first) to read now."""
        novel = {}
        for r, e in enumerate(hint[:min(len(hint), 64)]):
            e = int(e)
            self.predicted += 1
            if e not in pool.resident and e not in pool.staged:
                self.novel[r] += 1
                novel[e] = r
        self._pending.append((pool, novel))
        if self.mode == "off" or not self.depth:
            return []
        cands = [(e, r) for e, r in novel.items() if r < self.depth]
        if self.mode == "on" or not self.read_us:
            return [e for e, _ in cands]           # no read measured yet: reading is how it gets measured
        cands.sort(key=lambda er: -self.use_probability(er[1]))
        worth = [read_value(self.use_probability(r), (j // PREFETCH_READERS) * self.read_us, self.read_us, self.windows)
                 for j, (_, r) in enumerate(cands)]
        n = best_run(worth)
        self.declined += len(cands) - n
        return [e for e, _ in cands[:n]]

    def track(self, futures):
        self._timed.extend(futures)
        self.issued += len(futures)


# ───────────────────────────── the block that replaces mlx-lm's SwitchGLU / SwitchMLP ─────────────────────────────
class StreamedSwitch(nn.Module):
    """What `SwitchGLU` (gate, up, down) or `SwitchMLP` (fc1, fc2) computes, with its experts in a LayerPool.

    The router has already chosen `indices`; this reads which, makes them resident and runs the same
    `gather_qmm` the original does — on the pool, with the expert numbers translated to slots."""

    def __init__(self, pool, names, activation, quant):
        super().__init__()
        self.pool, self.names, self.activation, self.quant = pool, names, activation, quant
        self.glu = names == GLU
        # set by stream_experts: the model's prefetch policy, and a _Link when the next layer's picks can be guessed
        self._policy = None
        self._next = None

    def _gather(self, x, proj, local, srt):
        p = self.pool.arrays
        if self.quant["bits"]:
            return mx.gather_qmm(x, p[proj, "weight"], p[proj, "scales"], p.get((proj, "biases")),
                                 rhs_indices=local, transpose=True, group_size=self.quant["group_size"],
                                 bits=self.quant["bits"], mode=self.quant["mode"], sorted_indices=srt)
        return mx.gather_mm(x, p[proj, "weight"].swapaxes(-1, -2), rhs_indices=local, sorted_indices=srt)

    def _run(self, x, local, srt=False):
        if self.glu:
            up = self._gather(x, "up_proj", local, srt)
            gate = self._gather(x, "gate_proj", local, srt)
            return self._gather(self.activation(up, gate), "down_proj", local, srt)
        return self._gather(self.activation(self._gather(x, "fc1", local, srt)), "fc2", local, srt)

    def _guess(self, x):
        """The next layer's likely experts for a one-token step, best first, as a lazy array (None: no guess)."""
        nxt = self._next
        if nxt is None or nxt.policy.mode == "off" or x.size != x.shape[-1]:
            return None
        logits = nxt.gate(x.reshape(1, -1) * nxt.ratio).reshape(-1)
        return mx.stop_gradient(mx.argsort(-logits.astype(mx.float32))[:nxt.k])

    def _prefetch(self, guess):
        n = self._next
        n.policy.track([n.pool.staged[e][1] for e in n.pool.prefetch(n.policy.choose(n.pool, guess), n.executor)])

    def __call__(self, x, indices):
        guess = self._guess(x)
        x = mx.expand_dims(x, (-2, -3))
        if guess is not None:
            mx.eval(indices, guess)                                     # one wait for both
            guess = np.asarray(guess).tolist()
        idx = np.asarray(mx.stop_gradient(indices))                 # waits for everything before the router
        flat = idx.reshape(-1)
        uniq, counts = np.unique(flat, return_counts=True)
        self.pool.use(dict(zip(uniq.tolist(), counts.tolist())))
        policy = self._policy
        if policy is not None:
            if self.pool.prefetch_issued_at is not None:              # how long the drive had before this layer asked
                policy.observe_window((time.perf_counter() - self.pool.prefetch_issued_at) * 1e6)
                self.pool.prefetch_issued_at = None
            policy.observe_routes(self.pool, set(uniq.tolist()))
        if guess is not None:
            # issued now, beside this layer's own demand reads (on other threads): measured on an M5, starting them
            # after those reads left the window too short for most of them to land
            self._prefetch(guess)
        if flat.size < 64 and len(uniq) <= self.pool.capacity:          # a decode step: the direct path
            slots = self.pool.ensure(uniq.tolist())
            self.pool.drop_staged()
            table = np.zeros(self.pool.E, np.int32)
            for e, s in slots.items():
                table[e] = s
            return self._run(x, mx.array(table[idx])).squeeze(-2)
        # a prompt (mlx-lm sorts the (token, expert) pairs by expert from 64 of them up, so each expert's rows
        # are one matrix product), or more distinct experts than the pool holds: the same sort, run in the
        # fewest groups that fit, each expert read once
        xb = mx.broadcast_to(x, tuple(indices.shape) + (1, x.shape[-1])).reshape(-1, 1, x.shape[-1])
        order = np.argsort(flat, kind="stable")
        start, parts, done = 0, [], []
        while start < len(order):
            end, seen = start, set()
            while end < len(order) and (flat[order[end]] in seen or len(seen) < self.pool.capacity):
                seen.add(int(flat[order[end]]))
                end += 1
            sel = order[start:end]
            slots = self.pool.ensure(sorted(seen))
            table = np.zeros(self.pool.E, np.int32)
            for e, s in slots.items():
                table[e] = s
            # sorted_indices=True means the *slots* ascend, and a slot is not its expert's number: sort by slot
            local = table[flat[sel]]
            by_slot = np.argsort(local, kind="stable")
            sel, local = sel[by_slot], local[by_slot]
            parts.append(self._run(xb[mx.array(sel)], mx.array(local), srt=True))
            done.append(sel)
            start = end
        self.pool.drop_staged()
        res = mx.concatenate(parts, axis=0)[mx.array(np.argsort(np.concatenate(done)))]
        return res.reshape(tuple(indices.shape) + (1, -1)).squeeze(-2)


# ───────────────────────────── deciding, loading ─────────────────────────────
def mode():
    return os.environ.get("LLM_MOE_STREAM", "auto").lower()


def _layer_of(name):
    return name.split(".switch_mlp.")[0] if ".switch_mlp." in name else _PER_EXPERT.split(name)[0]


def plan(folder, config, mem_limit, total_ram, headroom_gb=2.5, idx=None):
    """None when the model is not a MoE or fits whole; else {"capacity": experts per layer, …}.

    `auto` streams only when the weights plus headroom would not fit under the RAM limit; `on` always (to try it
    on a model that fits), `off` never. The capacity is what is left of the budget after the always-resident
    part, spread over the MoE layers — but never below what one token needs (top-k) times two."""
    if mode() == "off":
        return None
    tc = config.get("text_config") or config
    n_exp = tc.get("num_experts") or tc.get("n_routed_experts") or tc.get("num_local_experts")
    if not n_exp:
        return None
    idx = idx or index(folder)
    experts, rest = expert_bytes(folder, idx)
    if not experts:
        return None
    total = (experts + rest) / GiB
    budget = total_ram * mem_limit / GiB - headroom_gb
    if mode() != "on" and total <= budget:
        return None
    top_k = tc.get("num_experts_per_tok") or tc.get("num_experts_per_token") or 2
    layers = len({_layer_of(k) for k, v in idx.items() if _is_expert(k, v[4])})
    per_expert = experts / max(1, layers) / n_exp                      # bytes of one expert, all its matrices
    forced = os.environ.get("LLM_MOE_CACHE")
    room = budget - rest / GiB
    cap = int(forced) if forced else int(room * GiB // max(1, layers * per_expert))
    cap = min(cap, n_exp)
    need = max(2 * top_k, top_k + 2)
    # A pool of any size works (a long prompt is split into groups, a step that wants more experts than fit is
    # too), so a capacity the user forced is honoured. An automatic one must leave room for two tokens' experts,
    # or the model would read most of what it uses from disk on every token.
    if cap < need and not forced:
        raise StreamError(
            f"even streamed, this model needs about {(rest + layers * per_expert * need) / GiB + headroom_gb:.1f} GB "
            f"({need} experts resident per layer next to the {rest / GiB:.1f} GB that is always loaded); "
            f"{total_ram * mem_limit / GiB:.1f} GB is available")
    return {"capacity": cap, "thrashing": cap < top_k, "experts": n_exp, "top_k": top_k, "layers": layers, "total_gb": total,
            "experts_gb": experts / GiB, "resident_gb": (rest + layers * cap * per_expert) / GiB}


def usage_file(folder):
    root = os.path.expanduser(os.environ.get("LLM_MOE_USAGE_DIR", "~/.cache/ai-llm-engine/moe"))
    return os.path.join(root, os.path.basename(folder.rstrip("/")) + ".json")


class Streamer:
    """The pools of a loaded model, the readers they share, and the saved hot set."""

    def __init__(self, folder, plan_, log=print):
        self.folder, self.plan, self.log = folder, plan_, log
        self.files, self.pools = Files(), {}
        self.readers = ThreadPoolExecutor(READERS, thread_name_prefix="moe-read")
        self.prefetchers = ThreadPoolExecutor(PREFETCH_READERS, thread_name_prefix="moe-prefetch")
        self.policy = PrefetchPolicy(plan_.get("top_k") or 2)
        self.predicting = 0                      # layers whose picks are guessed one layer ahead
        self.lock = threading.Lock()

    def stats(self):
        pools = list(self.pools.values())
        hits, misses = sum(p.hits for p in pools), sum(p.misses for p in pools)
        bytes_read = sum(p.bytes_read for p in pools)
        read_s = sum(p.read_s for p in pools)
        return {"format": "mlx-safetensors", "source": "preadv", "layers": len(pools),
                "capacity": self.plan["capacity"], "experts": self.plan["experts"],
                "hit_rate": round(hits / (hits + misses), 4) if hits + misses else None,
                "hits": hits, "misses": misses,
                "evictions": sum(p.evictions for p in pools),
                "read_ops": sum(p.read_ops for p in pools),
                "gb_read": round(bytes_read / GiB, 3), "read_s": round(read_s, 2),
                "read_mbps": round(bytes_read / 2**20 / read_s, 1) if read_s else None,
                "pool_gb": round(sum(p.nbytes for p in pools) / GiB, 2),
                # F_NOCACHE is a cache-bypass hint on macOS, not O_DIRECT; name it precisely in the API.
                "bypass_file_cache": self.files.direct, "reader_threads": READERS,
                "prefetch": self.prefetch_stats()}

    def prefetch_stats(self):
        pools, pol = list(self.pools.values()), self.policy
        used, unused = sum(p.prefetch_used for p in pools), sum(p.prefetch_unused for p in pools)
        return {"mode": pol.mode, "depth": pol.depth, "layers_predicted": self.predicting,
                "issued": pol.issued, "declined": pol.declined, "used": used,
                "late": sum(p.prefetch_late for p in pools), "unused": unused,
                "useful_rate": round(used / (used + unused), 4) if used + unused else None,
                "gb_prefetched": round(sum(p.prefetch_bytes for p in pools) / GiB, 3),
                "read_us": round(pol.read_us, 1) if pol.read_us else None,
                "window_us_median": round(float(np.median(pol.windows)), 1) if pol.windows else None}

    def save_usage(self):
        path = usage_file(self.folder)
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            tmp = path + ".tmp"
            with open(tmp, "w") as f:
                json.dump({"experts": self.plan["experts"],
                           "layers": {name: [round(float(x), 3) for x in p.score] for name, p in self.pools.items()}}, f)
            os.replace(tmp, path)
        except OSError as e:
            self.log(f"moe: could not save the hot set: {e}")

    def load_usage(self):
        try:
            with open(usage_file(self.folder)) as f:
                saved = json.load(f)
            return saved["layers"] if saved.get("experts") == self.plan["experts"] else {}
        except (OSError, ValueError, KeyError):
            return {}

    def close(self):
        self.save_usage()
        # Quiesce every positional read before closing the shared descriptors. `wait=False` could close a shard
        # while a worker was still inside `preadv`, turning an otherwise clean model unload into an intermittent
        # EBADF/truncated-read error. InferSSD follows the same ownership rule: readers stop before their sources.
        self.prefetchers.shutdown(wait=True, cancel_futures=True)
        self.readers.shutdown(wait=True)
        self.files.close()
        for p in self.pools.values():
            p.staged = {}
        self.pools = {}


def stream_experts(model, folder, plan_, log=print):
    """Swap every SwitchGLU / SwitchMLP of `model` for a StreamedSwitch. `model` must have been loaded lazily,
    so the experts it holds are file-backed arrays nobody has read: dropping them costs nothing. Returns the
    Streamer, or raises StreamError if the checkpoint's layout is not the stacked one this reads."""
    idx = index(folder)
    st = Streamer(folder, plan_, log)
    saved = st.load_usage()
    swapped, streamed = 0, []
    for path, mod in list(model.named_modules()):
        cname = type(mod).__name__
        if cname not in ("SwitchGLU", "SwitchMLP"):
            continue
        names = GLU if cname == "SwitchGLU" else MLP
        first = getattr(mod, names[0])
        n_exp = first["weight"].shape[0]
        tensors = {}
        for p in names:
            sub = getattr(mod, p)
            parts = find_tensors(idx, path, p, n_exp)
            if "weight" not in parts:
                raise StreamError(f"{path}.{p}.weight is in neither layout this reads (stacked, or one tensor "
                                  f"per expert) in this checkpoint's files")
            if "bias" in sub:
                raise StreamError(f"{path}.{p} has an expert bias: not streamed")
            tensors[p] = parts
        quant = {"bits": getattr(first, "bits", 0), "group_size": getattr(first, "group_size", 0),
                 "mode": getattr(first, "mode", "affine")}
        pool = LayerPool(st.files, tensors, names, n_exp, plan_["capacity"], st.readers, saved.get(path))
        if path in saved:
            pool.warm(list(np.argsort(-pool.score)))
        st.pools[path] = pool
        parent_path, _, attr = path.rpartition(".")
        parent = model
        for part in parent_path.split(".") if parent_path else []:
            parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
        sw = StreamedSwitch(pool, names, mod.activation, quant)
        sw._policy = st.policy
        setattr(parent, attr, sw)
        streamed.append((path, sw))
        swapped += 1
    if not swapped:
        raise StreamError("no SwitchGLU / SwitchMLP layers in this model: nothing to stream")
    _link_predictions(model, streamed, st)
    return st


def _module(model, path):
    m = model
    for part in path.split(".") if path else []:
        m = m[int(part)] if part.isdigit() else getattr(m, part, None)
        if m is None:
            return None
    return m


def _router_of(model, switch_path):
    """(the decoder layer's FFN norm, the MoE block's gate) for a switch at "…layers.N.mlp.switch_mlp", when the
    block routes by the top-k of its gate and the norm is a plain RMSNorm; else None."""
    block_path = switch_path.rpartition(".")[0]
    block = _module(model, block_path)
    layer = _module(model, block_path.rpartition(".")[0])
    if block is None or layer is None or type(block).__name__ not in TOPK_ROUTERS:
        return None
    norm, gate = getattr(layer, "post_attention_layernorm", None), getattr(block, "gate", None)
    if type(norm) is not nn.RMSNorm or gate is None or "weight" not in norm:
        return None
    return norm, gate


def _link_predictions(model, streamed, st):
    """Give each streamed layer the means to guess the next streamed layer's picks (see the module docstring).
    A pair whose norm or router does not fit the prediction gets none."""
    # named_modules() lists "layers.10" before "layers.2": order by the numbers in the path, the order they run in
    streamed = sorted(streamed, key=lambda ps: [int(t) if t.isdigit() else -1 for t in ps[0].split(".")])
    for (path, sw), (npath, nsw) in zip(streamed, streamed[1:]):
        this, nxt = _router_of(model, path), _router_of(model, npath)
        if this is None or nxt is None:
            continue
        w_this = this[0].weight.astype(mx.float32)
        w_next = nxt[0].weight.astype(mx.float32)
        # norm_next(h) = norm_this(h) * w_next / w_this; a zero weight carries no information about h: drop it
        ratio = mx.where(mx.abs(w_this) > 1e-8, w_next / mx.where(mx.abs(w_this) > 1e-8, w_this, 1.0), 0.0)
        ratio = ratio.astype(this[0].weight.dtype)
        mx.eval(ratio)
        sw._next = _Link(nsw.pool, nxt[1], ratio, st.plan.get("top_k") or 2, st.policy, st.prefetchers)
        st.predicting += 1


def load(folder, plan_, log=print):
    """(model, tokenizer, Streamer) — the model without its experts, which come from `folder` as needed."""
    from mlx_lm import load as mlx_load
    model, tok = mlx_load(folder, lazy=True)
    st = stream_experts(model, folder, plan_, log)
    mx.eval(model.parameters())                  # the experts are gone from the tree: only the rest is read
    mx.eval([a for p in st.pools.values() for a in p.arrays.values()])
    return model, tok, st
