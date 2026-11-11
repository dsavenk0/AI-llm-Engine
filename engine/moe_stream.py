"""Mixture-of-Experts models bigger than memory: the experts stay on the SSD and are read as the router asks.

A MoE model has most of its weights in experts, and a token uses only a few of them: Qwen3-30B-A3B holds 30 B
parameters and touches 3 B per token. The attention, the router, the embeddings and the shared parts are small
and always needed; the experts are large and mostly idle. So the small part is loaded as usual and each layer
gets a *pool* — room for C of its E experts — filled on demand from the checkpoint file.

It is exact. Whatever the router picks is made resident before it is used, so the output is the same as with
every expert in memory; only the speed differs. (Some projects instead restrict routing to the experts that
happen to be resident, which is faster and changes the model's answers; this one never does.)

What makes the reads affordable, taken from what Colibri, mlx-moe and kimi-k3-in-c report:

* **One read per expert.** An expert's gate, up and down matrices are each a contiguous slab of the stacked
  tensor in the safetensors file, so a miss is three `pread`s at known offsets — issued in parallel — with no
  decoding: the bytes go straight into the pool in the quantized layout the matmul takes.
* **A pool per layer that keeps what is used often.** Eviction takes the expert with the lowest decayed use
  count, the least recently used among equals, so the hot experts of a conversation stay and a passing one does
  not push them out.
* **The hot set outlives the process.** Counts are saved next to the model's cache and used to fill the pools
  before the first token, so a model that has been used starts warm instead of paying its first minutes in
  misses.
* **A big prompt is read once per expert.** Prefill sorts its tokens by expert; when one call needs more
  distinct experts than a pool holds, it is split into the fewest groups that fit, each expert loaded once.

Not done, and said so: prefetching the next layer's experts while this one computes (Colibri finds the next
layer's routing about 70 % predictable from the current hidden state), and reading with O_DIRECT. Every step
here waits for its experts; the pool hit rate is what keeps that cheap, and `stats()` reports it.

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
        self.direct = os.environ.get("LLM_MOE_DIRECT") == "1" if direct is None else direct

    def _fd(self, path):
        fd = self.fds.get(path)
        if fd is None:
            fd = self.fds[path] = os.open(path, os.O_RDONLY)
            if self.direct:
                import fcntl
                fcntl.fcntl(fd, getattr(fcntl, "F_NOCACHE", 48), 1)
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
        self.read_s = 0.0

    @property
    def nbytes(self):
        return sum(a.nbytes for a in self.arrays.values())

    @property
    def expert_nbytes(self):
        return sum(s[0] for s in self.slab.values())

    def _read_into(self, job):
        """Read expert `e`'s matrices straight into row `i` of each staging block."""
        i, e, blocks = job
        for key, (size, npdt, mxdt, shape, where) in self.slab.items():
            path, off = where(e)
            self.files.read_into(path, off, blocks[key][i])

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
            t0 = time.time()
            blocks = {key: np.empty((len(miss),) + tuple(s[3]), dtype=s[1]) for key, s in self.slab.items()}
            jobs = [(i, e, blocks) for i, e in enumerate(miss)]
            if len(miss) > 1:
                list(self.readers.map(self._read_into, jobs))
            else:
                self._read_into(jobs[0])
            self.read_s += time.time() - t0
            order = mx.array(np.asarray([slots[e] for e in miss], np.int32))
            for key, buf in blocks.items():
                block = mx.array(buf)
                if self.slab[key][2] == mx.bfloat16:
                    block = block.view(mx.bfloat16)
                self.arrays[key][order] = block                     # one update for all the misses
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
                "capacity": self.capacity}


# ───────────────────────────── the block that replaces mlx-lm's SwitchGLU / SwitchMLP ─────────────────────────────
class StreamedSwitch(nn.Module):
    """What `SwitchGLU` (gate, up, down) or `SwitchMLP` (fc1, fc2) computes, with its experts in a LayerPool.

    The router has already chosen `indices`; this reads which, makes them resident and runs the same
    `gather_qmm` the original does — on the pool, with the expert numbers translated to slots."""

    def __init__(self, pool, names, activation, quant):
        super().__init__()
        self.pool, self.names, self.activation, self.quant = pool, names, activation, quant
        self.glu = names == GLU

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

    def __call__(self, x, indices):
        x = mx.expand_dims(x, (-2, -3))
        idx = np.asarray(mx.stop_gradient(indices))                 # waits for everything before the router
        flat = idx.reshape(-1)
        uniq, counts = np.unique(flat, return_counts=True)
        self.pool.use(dict(zip(uniq.tolist(), counts.tolist())))
        if flat.size < 64 and len(uniq) <= self.pool.capacity:          # a decode step: the direct path
            slots = self.pool.ensure(uniq.tolist())
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
        self.lock = threading.Lock()

    def stats(self):
        pools = [p.stats() for p in self.pools.values()]
        hits, misses = sum(p["hits"] for p in pools), sum(p["misses"] for p in pools)
        return {"layers": len(pools), "capacity": self.plan["capacity"], "experts": self.plan["experts"],
                "hit_rate": round(hits / (hits + misses), 4) if hits + misses else None, "misses": misses,
                "gb_read": round(sum(p["bytes_read"] for p in pools) / GiB, 3),
                "read_s": round(sum(p["read_s"] for p in pools), 2),
                "pool_gb": round(sum(p.nbytes for p in self.pools.values()) / GiB, 2)}

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
        self.readers.shutdown(wait=False)
        self.files.close()
        self.pools = {}


def stream_experts(model, folder, plan_, log=print):
    """Swap every SwitchGLU / SwitchMLP of `model` for a StreamedSwitch. `model` must have been loaded lazily,
    so the experts it holds are file-backed arrays nobody has read: dropping them costs nothing. Returns the
    Streamer, or raises StreamError if the checkpoint's layout is not the stacked one this reads."""
    idx = index(folder)
    st = Streamer(folder, plan_, log)
    saved = st.load_usage()
    swapped = 0
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
        setattr(parent, attr, StreamedSwitch(pool, names, mod.activation, quant))
        swapped += 1
    if not swapped:
        raise StreamError("no SwitchGLU / SwitchMLP layers in this model: nothing to stream")
    return st


def load(folder, plan_, log=print):
    """(model, tokenizer, Streamer) — the model without its experts, which come from `folder` as needed."""
    from mlx_lm import load as mlx_load
    model, tok = mlx_load(folder, lazy=True)
    st = stream_experts(model, folder, plan_, log)
    mx.eval(model.parameters())                  # the experts are gone from the tree: only the rest is read
    mx.eval([a for p in st.pools.values() for a in p.arrays.values()])
    return model, tok, st
