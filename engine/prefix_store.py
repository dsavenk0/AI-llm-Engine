# SSD (L2) prefix cache: KV + recurrent (SSM) state of long prompt prefixes.
#
# One entry = the cache state right before a generation prompt (the engine's
# snapshot point) for a token prefix of >= min_tokens. Entries survive server
# restarts, auto-sleep and model switches, so a long agent system prompt is
# prefilled once and later only restored from disk.
import hashlib
import json
import os
import time

import mlx.core as mx
import numpy as np

from .model import KVCache, SSMCache


class PrefixStore:
    def __init__(self, root, max_bytes, min_tokens=1024, log=print):
        self.root, self.max_bytes, self.min_tokens, self.log = root, max_bytes, min_tokens, log
        os.makedirs(root, exist_ok=True)
        self.index_path = os.path.join(root, "index.json")
        try:
            with open(self.index_path) as f:
                self.entries = json.load(f)
        except (OSError, ValueError):
            self.entries = {}
        # drop index rows whose files vanished
        self.entries = {k: e for k, e in self.entries.items() if os.path.exists(self._file(k))}

    def _file(self, key, ext="safetensors"):
        return os.path.join(self.root, f"{key}.{ext}")

    @staticmethod
    def _key(tokens):
        return hashlib.sha1(np.asarray(tokens, np.int32).tobytes()).hexdigest()[:20]

    def _save_index(self):
        tmp = self.index_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.entries, f)
        os.replace(tmp, self.index_path)

    @property
    def total_bytes(self):
        return sum(e["bytes"] for e in self.entries.values())

    def has(self, tokens):
        return self._key(tokens) in self.entries

    def save(self, tokens, cache, snaps):
        """Persist the state of `cache` at len(tokens) (SSM layers from `snaps`)."""
        n = len(tokens)
        if n < self.min_tokens or self.max_bytes <= 0 or self.has(tokens):
            return
        arrays = {}
        for i, (c, s) in enumerate(zip(cache, snaps)):
            if s is None:  # attention layer: KV up to n
                if c.keys is None or c.offset < n:
                    return
                arrays[f"{i}.k"] = c.keys[..., :n, :]
                arrays[f"{i}.v"] = c.values[..., :n, :]
            else:
                conv, state, _ = s
                if conv is None or state is None:
                    return
                arrays[f"{i}.conv"] = conv
                arrays[f"{i}.state"] = state
        nbytes = sum(a.nbytes for a in arrays.values())
        if nbytes > self.max_bytes:
            return
        self._evict(nbytes)
        key = self._key(tokens)
        t0 = time.time()
        mx.save_safetensors(self._file(key), arrays)
        np.save(self._file(key, "tokens.npy"), np.asarray(tokens, np.int32))
        self.entries[key] = {"n": n, "bytes": nbytes, "used": time.time()}
        self._save_index()
        self.log(f"prefix cache: saved {n} tokens to SSD ({nbytes / 2**20:.0f} MB, {time.time() - t0:.1f}s)")

    def _evict(self, incoming):
        while self.entries and self.total_bytes + incoming > self.max_bytes:
            key = min(self.entries, key=lambda k: self.entries[k]["used"])
            for ext in ("safetensors", "tokens.npy"):
                try:
                    os.remove(self._file(key, ext))
                except OSError:
                    pass
            del self.entries[key]
        self._save_index()

    def lookup(self, ids):
        """Longest stored prefix of `ids` (shorter than ids), or None."""
        best = None
        for key, e in self.entries.items():
            n = e["n"]
            if n >= len(ids) or (best and n <= best[1]):
                continue
            try:
                toks = np.load(self._file(key, "tokens.npy"))
            except OSError:
                continue
            if len(toks) == n and np.array_equal(toks, np.asarray(ids[:n], np.int32)):
                best = (key, n, toks.tolist())
        return best

    def load(self, entry, model):
        """Rebuild caches (and SSM snapshots) for a lookup() result."""
        key, n, tokens = entry
        t0 = time.time()
        arrays = mx.load(self._file(key))
        cache, snaps = model.make_cache(), []
        for i, c in enumerate(cache):
            if isinstance(c, KVCache):
                c.keys, c.values, c.offset = arrays[f"{i}.k"], arrays[f"{i}.v"], n
                snaps.append(None)
            else:
                c.conv, c.state, c.offset = arrays[f"{i}.conv"], arrays[f"{i}.state"], n
                snaps.append(c.snapshot())
        mx.eval([a for a in arrays.values()])
        self.entries[key]["used"] = time.time()
        self._save_index()
        self.log(f"prefix cache: restored {n} tokens from SSD in {time.time() - t0:.2f}s")
        return cache, snaps, tokens
