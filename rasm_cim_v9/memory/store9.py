"""v0.9 memory layer: MobileMemory9 — compact base index + journal updates
+ streaming HV (reuses v0.8 StreamHV; old algorithms untouched)."""
from __future__ import annotations
import os
from typing import Iterable

from rasm_cim_v8.memory.hv_stream import StreamHV
from rasm_cim_v9.storage import index as IX
from rasm_cim_v9.storage.journal import V3Store, append_txn, compact


class MobileMemory9:
    """Tiers:
      "stream"   V3Store only (16 B/rec resident base index + journal arrays)
      "residue"  + compact ResidueIndex (array buckets, ~8 B/entry)
      "hv"       + STREAMING hypervector retrieval (no full cache — v0.8
                  M2-refactor path; legacy open_hv still available below)
    update(key, value) = journal txn (O(1) append), NOT a rebuild.
    """

    def __init__(self, path: str, tiers=("stream",), cache_pages: int = 8,
                 on_bad: str = "raise"):
        self.path = path
        self.tiers = set(tiers)
        self.cache_pages = cache_pages
        self.on_bad = on_bad
        self.store: V3Store | None = None
        self._res: IX.ResidueIndex | None = None
        self.hv: StreamHV | None = None
        self.stats: dict = {}

    # -- lifecycle --------------------------------------------------------
    @classmethod
    def build(cls, path: str, facts: Iterable[tuple[int, int]],
              page_recs: int = 2048, sort: str = "auto", **kw):
        meta = IX.build_v3_base(facts, path, page_recs=page_recs, sort=sort)
        m = cls(path, **kw)
        m.stats["written"] = meta
        return m

    def open(self):
        if self.store is None:
            self.store = V3Store(self.path, cache_pages=self.cache_pages,
                                 on_bad=self.on_bad)
            if "residue" in self.tiers:
                self._res = self.store.build_residue()
        return self

    def reload(self):
        self.close()
        self.store = None
        self._res = None
        self.hv = None
        return self.open()

    def close(self):
        if self.store is not None:
            self.store.close()
            self.store = None

    # -- writes (THE v0.9 difference: no full rebuild) --------------------
    def update(self, key: int, value: int, fsync: bool = True) -> int:
        seq = append_txn(self.path, [(key, value)], fsync=fsync)
        self.refresh()
        return seq

    def update_txn(self, records: list[tuple[int, int]], fsync: bool = True) -> int:
        seq = append_txn(self.path, records, fsync=fsync)
        self.refresh()
        return seq

    def refresh(self):
        """Re-open so new committed txns become visible (journal walk only)."""
        if self.store is not None:
            self.close()
            self.open()

    def commit_journal(self) -> dict:
        meta = compact(self.path)
        self.refresh()
        return meta

    # -- reads ------------------------------------------------------------
    def query_exact(self, key: int):
        self.open()
        st, v = self.store.get(key)
        if st == "TRUE":
            return ("TRUE", v, 1.0)
        if st == "CONFLICT":
            return ("CONFLICT", v, 0.0)
        if st == "CORRUPT":
            return ("CORRUPT", None, 0.0)
        return ("UNKNOWN", None, 0.0)

    def query_partial(self, residue: int, modulus: int) -> list[int]:
        self.open()
        if self._res is not None and modulus in IX.ResidueIndex.MODS:
            return self._res.query(residue, modulus)
        keys = sorted(k for k, _ in self.store.scan())          # fallback O(N)
        seen, out = set(), []
        for k in keys:
            if k % modulus == residue and k not in seen:
                seen.add(k)
                out.append(k)
        return out

    def scan(self):
        self.open()
        yield from self.store.scan()

    def n_records(self) -> int:
        self.open()
        return self.store.n_records()

    # -- HV (streaming by default; legacy kept per M2-refactor policy) ----
    def open_hv_stream(self, d: int = 256, hv_seed: int = 0, method: str = "auto",
                       block: int = 8192, cache_mb: float = 2.0):
        self.open()
        self.hv = StreamHV(self.store, d=d, seed=hv_seed, method=method,
                           block=block, cache_mb=cache_mb)
        return self

    def open_hv_legacy(self, d: int = 256, hv_seed: int = 0):
        """KEPT for the A/B (spec M4: do not remove before comparison done)."""
        from rasm_cim_v8.hv.bitpack import gen_packed_hv
        self.open()
        self._legacy_hv = [(gen_packed_hv(k, d, hv_seed), v)
                           for k, v in self.store.read_records(0, self.store.n_records())]
        self._legacy_d = d
        return self

    def query_noisy(self, q_packed: int, tau: float = 0.5):
        if getattr(self, "hv", None) is not None:
            return self.hv.query(q_packed, tau)
        if getattr(self, "_legacy_hv", None) is not None:
            d = self._legacy_d
            best, best_v = -2.0, None
            for p, v in self._legacy_hv:
                s = (d - 2 * (q_packed ^ p).bit_count()) / d
                if s > best:
                    best, best_v = s, v
            if best_v is None or best < tau:
                return ("UNKNOWN", None, best if self._legacy_hv else 0.0)
            return ("TRUE", best_v, (best + 1.0) / 2.0)
        raise RuntimeError("call open_hv_stream() or open_hv_legacy() first")

    # -- sizes ------------------------------------------------------------
    def sizes(self) -> dict:
        self.open()
        out = {"file_bytes": os.path.getsize(self.path),
               "resident_index_bytes": self.store.stats["index_bytes"],
               "journal_entries": len(self.store.journal_keys),
               "journal_last_seq": self.store.last_seq}
        if self._res is not None:
            out["residue_entries"] = self._res.entries
            out["residue_bytes"] = self._res.bytes_resident()
        if self.hv is not None:
            out["hv_cache_bytes"] = self.hv.resident_bytes()
        return out
