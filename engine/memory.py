# System memory watchdog (macOS). Keeps the whole machine under a RAM ceiling:
# above ~90 % used a 16 GB Mac starts swapping hard and the UI freezes.
import ctypes
import ctypes.util
import os

_libc = ctypes.CDLL(ctypes.util.find_library("c"))
_PAGE = os.sysconf("SC_PAGE_SIZE")
TOTAL = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")


def _sysctl_int(name):
    val = ctypes.c_uint64(0)
    size = ctypes.c_size_t(ctypes.sizeof(val))
    if _libc.sysctlbyname(name.encode(), ctypes.byref(val), ctypes.byref(size), None, 0) != 0:
        return None
    return val.value


def available_bytes():
    """Free + inactive + speculative + purgeable pages: what can be handed out without swapping."""
    free = _sysctl_int("vm.page_free_count") or 0
    spec = _sysctl_int("vm.page_speculative_count") or 0
    purg = _sysctl_int("vm.page_purgeable_count") or 0
    inactive = _sysctl_int("vm.page_pageable_internal_count")  # anonymous pageable, not directly freeable
    ext = _sysctl_int("vm.page_pageable_external_count") or 0     # file cache: freeable
    del inactive
    return (free + spec + purg + ext) * _PAGE


def used_fraction():
    return 1.0 - available_bytes() / TOTAL


def pressure_level():
    """1 = normal, 2 = warning, 4 = critical (kern.memorystatus_vm_pressure_level)."""
    return _sysctl_int("kern.memorystatus_vm_pressure_level") or 1


def headroom_bytes(limit):
    """Bytes that may still be allocated before the system crosses `limit` (fraction of RAM)."""
    return int(available_bytes() - (1.0 - limit) * TOTAL)


_TASK_VM_INFO = 22
_PHYS_FOOTPRINT_OFFSET = 144  # task_vm_info_data_t.phys_footprint


GiB = 2**30
GRACE = 0.02                 # how far past the limit a request already under way may run


class MemoryLimitError(RuntimeError):
    """The request would push the whole Mac past its RAM ceiling."""


# ───────────────────────── the RAM guard ─────────────────────────
# One implementation for every engine. An engine (`host`) supplies its numbers — mem_limit, kv_per_token,
# act_per_token, base_work, prefill_step, prefill_guard — and `release`, the call that hands reusable
# buffers back to the system (mx.clear_cache on MLX). Nothing here knows which backend that is.

def reserve(host, new_tokens, release, shrink=True):
    """Make sure a prefill of `new_tokens` fits under the system RAM limit, and set `host.step_now` to the
    chunk size to prefill with.

    With `shrink`, the largest chunk (prefill_step → 512 → 256 → 128) whose peak fits the current headroom
    is chosen, so tight memory means smaller chunks before it means a refusal. Reusable buffers are released
    and the engine's restore points given up before refusing; a request that only misses by the grace band
    — a big model resident, usage sitting exactly at the limit — still goes through."""
    if not host.prefill_guard:           # the guard is off: never refuse, never shrink the chunk
        host.step_now = host.prefill_step
        return

    def need(step):
        return new_tokens * host.kv_per_token + host.base_work + min(new_tokens, step) * host.act_per_token

    steps = [s for s in (host.prefill_step, 512, 256, 128) if s <= host.prefill_step] if shrink \
        else [host.prefill_step]
    mgr = getattr(host, "memory", None)
    for attempt in range(3):
        head = headroom_bytes(host.mem_limit)
        fit = next((s for s in steps if need(s) <= head), None)
        if fit:
            host.step_now = fit
            if fit < host.prefill_step:
                host.log(f"memory: {used_fraction():.0%} RAM in use, prefill chunk {fit}")
            return
        if mgr is not None:
            mgr.reclaim(need(steps[-1]), upto="buffers" if attempt == 0 else "restore_points")
        else:
            release()
            if attempt == 1 and getattr(host, "snapshots", None):
                host.snapshots = []      # restore points are the cheapest thing to give up
    need_last = need(steps[-1])
    if headroom_bytes(min(1.0, host.mem_limit + GRACE)) >= need_last:
        host.step_now = steps[-1]
        if shrink:
            host.log(f"memory: {used_fraction():.0%} RAM in use, prefill chunk {steps[-1]}")
        return
    raise MemoryLimitError(
        f"not enough free RAM: {used_fraction():.0%} of system memory in use, limit "
        f"{host.mem_limit:.0%}; the prompt needs ~{need_last / GiB:.1f} GB more. Close other apps or "
        f"shorten the conversation.")


def fit_output(host, max_tokens):
    """The output length that still fits under the RAM limit (never below 16 tokens)."""
    head = headroom_bytes(host.mem_limit) - host.base_work
    fit = max(16, head // max(1, host.kv_per_token))
    if fit < max_tokens:
        host.log(f"memory: output capped at {fit} tokens ({used_fraction():.0%} RAM in use)")
    return int(min(max_tokens, fit))


def check(host, stage, release):
    """Called during long work. True while the limit holds (after releasing buffers, within the grace band);
    during prefill a breach raises, during decode it returns False and the answer ends where it is."""
    if used_fraction() <= host.mem_limit:
        return True
    release()
    if used_fraction() <= host.mem_limit + GRACE:
        return True
    if stage == "prefill":
        raise MemoryLimitError(f"system memory reached {used_fraction():.0%} during prefill "
                               f"(limit {host.mem_limit:.0%}); request stopped to keep the Mac responsive")
    return False


def process_footprint():
    """This process's physical footprint in bytes (what Activity Monitor shows as Memory),
    including Metal/MLX allocations that RSS does not count."""
    try:
        task = ctypes.c_uint.in_dll(_libc, "mach_task_self_").value
        buf = (ctypes.c_uint8 * 1024)()
        count = ctypes.c_uint(len(buf) // 4)
        if _libc.task_info(task, _TASK_VM_INFO, buf, ctypes.byref(count)) != 0:
            return None
        return int.from_bytes(bytes(buf[_PHYS_FOOTPRINT_OFFSET:_PHYS_FOOTPRINT_OFFSET + 8]), "little")
    except (ValueError, OSError):
        return None


# ───────────────────────── the memory manager ─────────────────────────
class Manager:
    """Everything one engine holds in RAM (and on the SSD for it), and the one order in which it gives
    memory back. Before this, the RAM guard, the embeddings path and sleep each freed things on their own;
    now every one of them asks here, so the order below is the only policy:

        buffers          MLX's freed-buffer cache            free to give up: re-allocated on demand
        restore_points   prefix snapshots inside the cache   costs a re-prefill of a shared prefix
        conversation     the live KV / recurrent state       written to the SSD prefix cache first (when
                                                             there is one), then dropped — only between
                                                             requests, never under a running one

    What cannot be given back without unloading — the weights (with a streamed MoE model's resident
    expert pools inside them), the activations of a running request — is reported, not reclaimed; sleep
    (Engine.sleep) is the step past the last one here. Sizes come from the engine's `memory_breakdown()`,
    so there is one measurement as well as one policy.

    `state()` is what the scheduler (engine/scheduler.py) reads: "ok", "tight" within 5 % of the limit or
    under the kernel's warning level, "critical" past the limit's grace band or at the critical level.
    """
    ORDER = ("buffers", "restore_points", "conversation")

    def __init__(self, host, release):
        self.host, self.release = host, release
        self.given_up = {name: 0 for name in self.ORDER}

    def state(self):
        used, level, limit = used_fraction(), pressure_level(), self.host.mem_limit
        if level >= 4 or used > limit + GRACE:
            return "critical"
        if level >= 2 or used > limit - 0.05:
            return "tight"
        return "ok"

    def _give_up(self, name):
        h = self.host
        if name == "buffers":
            self.release()
            return True
        if name == "restore_points":
            if not getattr(h, "snapshots", None):
                return False
            h.snapshots = []
        elif name == "conversation":
            if getattr(h, "cache", None) is None:
                return False
            persist = getattr(h, "persist_prefix", None)
            if persist:
                persist()                            # resumable from the SSD without a re-prefill
            h.cache, h.cache_tokens, h.snapshots = None, [], []
        self.release()
        self.given_up[name] += 1
        return True

    def reclaim(self, need=0, upto="restore_points"):
        """Give memory back, cheapest first, until `need` more bytes fit under the limit or the step `upto`
        has been taken. `upto="conversation"` is for between requests only (the server's idle check).
        Returns whether `need` now fits."""
        last = self.ORDER.index(upto)
        for name in self.ORDER[:last + 1]:
            if headroom_bytes(self.host.mem_limit) >= need and name != "buffers":
                break
            if self._give_up(name) and name != "buffers":
                self.host.log(f"memory: gave up the {name.replace('_', ' ')} "
                              f"({used_fraction():.0%} RAM in use)")
        return headroom_bytes(self.host.mem_limit) >= need

    def relieve(self):
        """Between requests, while the Mac is past its limit: release everything that can be released
        without unloading the model. Returns what was given up."""
        if self.state() != "critical":
            return []
        before = dict(self.given_up)
        self.reclaim(need=1, upto="conversation")     # until usage is back under the limit
        return [n for n in self.ORDER if self.given_up[n] > before[n]]

    def report(self):
        h = self.host
        try:
            br = h.memory_breakdown()
        except Exception:
            br = {}
        store = getattr(h, "store", None)
        return {
            "state": self.state(), "used": round(used_fraction(), 3), "limit": h.mem_limit,
            "headroom_gb": round(headroom_bytes(h.mem_limit) / GiB, 2),
            "resident_gb": round((br.get("weights", 0) + br.get("vision", 0)) / GiB, 2),
            "reclaimable_gb": {"buffers": round(br.get("mlx_cache", 0) / GiB, 3),
                               "restore_points": round(br.get("snapshots", 0) / GiB, 3),
                               "conversation": round((br.get("kv", 0) + br.get("ssm", 0)) / GiB, 3)},
            "ssd_prefix_gb": round(store.total_bytes / GiB, 2) if store else None,     # a property of PrefixStore
            "given_up": dict(self.given_up),
        }

