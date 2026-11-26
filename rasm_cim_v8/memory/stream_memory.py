"""MobileStreamMemory — M2 memory layer over v2 files.

Tiers (memory/speed trade-off is MEASURED in bench8, not promised):
  "stream"  file-backed get/scan; resident = index arrays only (16 B/record)
  +residue  in-RAM inverted residue index (dict of lists) like v0.6's —
            quantified separately (M6 asks exactly this)
  +dict     full in-RAM dict fallback (baseline for the ≤2× p95 target)

Semantics parity with v0.6 MemorySystem where defined:
  exact query -> ("TRUE", v, 1.0) | ("UNKNOWN", None, 0.0) | ("CONFLICT", vs, 0.0)
Write-time dedup: NOT done in stream mode (would need a resident key set —
defeats the memory goal); duplicates collapse on READ (set semantics), so
results are equivalent for query purposes; `dedup_deferred=True` states this.
Noisy/HV retrieval lives in v8/hv (bitpack) and is an optional layer here.
"""
from __future__ import annotations
from typing import Iterable

from ..storage.streaming import LazyStore, open_lazy
from ..storage import format as F


class MobileStreamMemory:
    def __init__(self, path: str, tiers=("stream",), cache_pages: int = 8,
                 mode: str = "pread", on_bad: str = "raise"):
        self.path = path
        self.tiers = set(tiers)
        self.store: LazyStore | None = None
        self.mode = mode
        self.cache_pages = cache_pages
        self.on_bad = on_bad
        self._residue: dict[tuple[int, int], list[int]] | None = None
        self._dict: dict[int, int | list[int]] | None = None
        self.dedup_deferred = "dict" not in self.tiers
        self.stats = {}
        self._hv = None
        self._hv_d = 256
        self._hv_seed = 0
        self._hvstream = None          # M2-refactor: StreamHV backend
        self._hv_backend = None        # "dict" (legacy) | "stream"

    # -- construction -----------------------------------------------------
    @classmethod
    def build(cls, path: str, facts: Iterable[tuple[int, int]], page_recs: int = 2048,
              **kw) -> "MobileStreamMemory":
        meta = F.write_v2(facts, path, page_recs=page_recs)
        m = cls(path, **kw)
        m.stats["written"] = meta
        return m

    def open(self):
        if self.store is None:
            self.store = open_lazy(self.path, mode=self.mode,
                                   cache_pages=self.cache_pages, on_bad=self.on_bad)
            if "residue" in self.tiers:
                self._residue = {}
                for k in self.store._keys:
                    for pos, mod in enumerate((97, 89, 83)):
                        self._residue.setdefault((pos, k % mod), []).append(k)
            if "dict" in self.tiers:
                self._dict = self.store.to_full_memory()
        return self

    def close(self):
        if self.store is not None:
            self.store.close()
            self.store = None
        self._hv = None

    def reload(self):
        """Serialization consistency probe: drop everything, reopen from disk."""
        self.close()
        self._residue = None
        self._dict = None
        return self.open()

    # -- queries ----------------------------------------------------------
    def query_exact(self, key: int):
        self.open()
        if self._dict is not None:
            v = self._dict.get(key)
            if v is None:
                return ("UNKNOWN", None, 0.0)
            if isinstance(v, list):
                return ("CONFLICT", v, 0.0)
            return ("TRUE", v, 1.0)
        st, val = self.store.get(key)
        if st == "TRUE":
            return ("TRUE", val, 1.0)
        if st == "CONFLICT":
            return ("CONFLICT", val, 0.0)
        if st == "CORRUPT":
            return ("CORRUPT", None, 0.0)
        return ("UNKNOWN", None, 0.0)

    def query_partial(self, residue: int, modulus: int) -> list[int]:
        self.open()
        if self._residue is not None and modulus in (97, 89, 83):
            pos = (97, 89, 83).index(modulus)
            return sorted(self._residue.get((pos, residue % modulus), ()))
        return self.store.query_partial(residue, modulus)

    # -- noisy associative retrieval (M5 tier) ------------------------------
    def open_hv(self, d: int = 256, hv_seed: int = 0):
        """Build in-RAM packed-HV cache over the streamed records.
        HV(key) regenerated deterministically (no HV bytes on disk).
        Cost is O(N * d/64) ints at open; memory ~60 B/record for D=256."""
        from ..hv.bitpack import gen_packed_hv
        self.open()
        self._hv_backend = "dict"
        if getattr(self, "_hv", None) is None:
            self._hv = []                      # (packed_int, value) scan order
            self._hv_d = d
            self._hv_seed = hv_seed
            for k, v in self.store.read_records(0, self.store.n):
                self._hv.append((gen_packed_hv(k, d, hv_seed), v))
        return self

    def open_hv_stream(self, d: int = 256, hv_seed: int = 0, method: str = "auto",
                       block: int = 8192, cache_mb: float = 2.0):
        """M2-REFACTOR backend: noisy retrieval WITHOUT full HV cache.
        Same semantics/results as open_hv+query_noisy (proven by tests);
        resident memory = paged store + optional bounded LRU (cache_mb)."""
        from .hv_stream import StreamHV
        self.open()
        self._hvstream = StreamHV(self.store, d=d, seed=hv_seed, method=method,
                                  block=block, cache_mb=cache_mb)
        self._hv_backend = "stream"
        self._hv_d, self._hv_seed = d, hv_seed
        return self

    def hv_stream_summary(self) -> dict:
        if self._hvstream is None:
            return {"error": "no stream HV backend; call open_hv_stream()"}
        return self._hvstream.summary()

    def query_noisy(self, q_packed: int, tau: float = 0.5):
        """v0.6 decision rule: argmax sim; tau gate; conf=(sim+1)/2.
        Backend = whichever of open_hv (legacy dict) / open_hv_stream
        (M2-refactor) was enabled; results are identical by contract."""
        if self._hv_backend == "stream":
            return self._hvstream.query(q_packed, tau)
        if getattr(self, "_hv", None) is None:
            raise RuntimeError("call open_hv() or open_hv_stream() first")
        d = self._hv_d
        best_i, best = -1, -2.0
        for i, (p, _v) in enumerate(self._hv):
            s = (d - 2 * (q_packed ^ p).bit_count()) / d
            if s > best:
                best, best_i = s, i
        if best_i < 0 or best < tau:
            return ("UNKNOWN", None, best if best_i >= 0 else 0.0)
        return ("TRUE", self._hv[best_i][1], (best + 1.0) / 2.0)

    def hv_bytes_estimate(self) -> int:
        """Measured-ish: sum of int object sizes in the cache (sys.getsizeof).
        Stream backend: bounded by cache budget only (0 for 'gen')."""
        import sys
        if self._hv_backend == "stream":
            return self._hvstream.resident_bytes()
        if getattr(self, "_hv", None) is None:
            return 0
        return sum(sys.getsizeof(p) + 28 for p, _v in self._hv)

    def scan(self, pred=None):
        self.open()
        return self.store.scan(pred)

    def n_records(self) -> int:
        self.open()
        return self.store.n

    def sizes(self) -> dict:
        """Storage + index sizes: file bytes, resident index bytes,
        residue index entry count (list slots)."""
        import os
        self.open()
        out = {"file_bytes": os.path.getsize(self.path),
               "resident_index_bytes": self.store.stats["index_bytes"],
               "page_bytes_bound": self.cache_pages * self.store.page_recs * 16}
        if self._residue is not None:
            out["residue_entries"] = sum(len(v) for v in self._residue.values())
            out["residue_lists"] = len(self._residue)
        if self._dict is not None:
            import sys
            out["dict_entries"] = len(self._dict)
            out["dict_sysgetsizeof_bytes"] = sys.getsizeof(self._dict)
        return out

    def page_stats(self) -> dict:
        self.open()
        return dict(self.store.stats, corrupt_pages=sorted(self.store.corrupt_pages))
