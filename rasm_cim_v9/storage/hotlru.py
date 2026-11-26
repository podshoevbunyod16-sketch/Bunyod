"""M3-A: bounded hot-KEY LRU (resolved get() answers) on top of V3Store.

Caches (status, value) per key — NOT pages. Byte-budgeted, evict-oldest.
Worth keeping only if the workload is skewed (repeat keys); the bench
(bench_v9/m3_tuning.py) decides by measurement, per the LRU-multi-workload
rule — see results_v9/m3_tuning.json.
"""
from __future__ import annotations
import sys
from collections import OrderedDict


class HotKeyLRU:
    def __init__(self, store, max_bytes: int = 2 * 2**20):
        self.store = store
        self.max_bytes = max_bytes
        self._d: OrderedDict[int, tuple] = OrderedDict()
        self._bytes = 0
        self.hits = 0
        self.misses = 0

    @staticmethod
    def _cost(k, v):
        return sys.getsizeof(k) + 56 + (sys.getsizeof(v) if isinstance(v, list) else 0)

    def get(self, key: int):
        v = self._d.get(key, _MISS)
        if v is not _MISS:
            self._d.move_to_end(key)
            self.hits += 1
            return v
        self.misses += 1
        res = self.store.get(key)
        c = self._cost(key, res)
        self._d[key] = res
        self._bytes += c
        while self._bytes > self.max_bytes and self._d:
            k0, v0 = self._d.popitem(last=False)
            self._bytes -= self._cost(k0, v0)
        return res

    def invalidate(self):
        self._d.clear()
        self._bytes = 0

    def stats(self):
        return {"size": len(self._d), "bytes": self._bytes,
                "hits": self.hits, "misses": self.misses}


class _Miss:
    pass


_MISS = _Miss()
