"""M2-refactor: streaming Hypervector noisy retrieval — no full in-RAM HV cache.

Problem (user-reported on real Android ARM64): MobileStreamMemory.open_hv()
materialized ALL HVs (self._hv list); peak RSS was 38.7 MiB @20k and
56.4 MiB @100k — over the <40 MiB mobile target at 100k.

Because gen_packed_hv is DETERMINISTIC in (key, d, seed) and no HV bytes
exist on disk, the HV of any record can be regenerated at query time. This
module streams records block-by-block from the paged LazyStore and compares
against the query word-wise — resident memory is ONE BLOCK, not N records.

Methods (all produce results IDENTICAL to the legacy in-RAM path by
construction: same file order, same similarity formula, same strict-> tie
rule, same tau gate; equivalence is asserted by tests, not assumed):
  "gen"         stdlib scalar; zero cache; O(N*d/64) per query work
  "lru"         gen + bounded Hypervector LRU cache (byte budget, default 2 MiB)
  "block"       gen collected into a block-local list, then a tight loop
  "block_numpy" vectorized splitmix64 generation + vectorized xor/popcount
                per block (uint64 wrap-around == &MASK64 by construction);
                requires numpy; d%64==0 fast path, else per-record scalar gen
                + vectorized compare (still block-bounded)
  "combo"       block_numpy when numpy present, lru otherwise (device-safe
                default: "auto" == "combo")

NO SIMD/NEON claim is made: numpy paths are "C-level vectorizable loops";
instruction-level proof would require on-device objdump/perf (out of scope,
PLAN_v0.9). The legacy open_hv path is KEPT (spec: not removed before the
comparison benchmark finishes; see bench/m2_hv_refactor.py).

Format files are untouched: this module only READS via LazyStore.
"""
from __future__ import annotations

from collections import OrderedDict

try:
    import numpy as _np
except ImportError:      # bare Termux python
    _np = None

MASK64 = (1 << 64) - 1
_GOLDEN = 0x9E3779B97F4A7C15
_M1 = 0xBF58476D1CE4E5B9
_M2 = 0x94D049BB133111EB


class HVCache:
    """Bounded (byte-budget) LRU of packed HVs — variant 2 of the study."""

    def __init__(self, max_bytes: int = 2 * 2**20):
        self.max_bytes = max_bytes
        self._d: OrderedDict[int, int] = OrderedDict()
        self._bytes = 0
        self.hits = 0
        self.misses = 0

    @staticmethod
    def _entry_bytes(k: int, v: int) -> int:
        import sys
        return sys.getsizeof(k) + sys.getsizeof(v) + 104   # dict slot + link

    def get(self, key: int):
        v = self._d.get(key)
        if v is None and key not in self._d:
            self.misses += 1
            return None
        self._d.move_to_end(key)
        self.hits += 1
        return v

    def put(self, key: int, val: int):
        if key in self._d:
            self._d.move_to_end(key)
            return
        b = self._entry_bytes(key, val)
        self._d[key] = val
        self._bytes += b
        while self._bytes > self.max_bytes and self._d:
            _k, v0 = self._d.popitem(last=False)
            self._bytes -= self._entry_bytes(_k, v0)

    @property
    def size(self) -> int:
        return len(self._d)


class StreamHV:
    """Query-time streaming noisy retrieval over a LazyStore."""

    def __init__(self, store, d: int = 256, seed: int = 0,
                 method: str = "auto", block: int = 8192,
                 cache_mb: float = 2.0):
        if method not in ("auto", "gen", "lru", "block", "block_numpy", "combo"):
            raise ValueError(f"unknown hv method {method!r}")
        if block < 1:
            raise ValueError("block must be >= 1")
        self.store = store
        self.d = d
        self.seed = seed
        numpy_ok = _np is not None
        if method == "auto":
            method = "combo"
        self.method = method
        if self.method == "combo":
            # measured policy: numpy -> vectorized blocks; bare python -> plain
            # "gen". LRU with cap < working set was the SLOWEST full-scan mode
            # (643 ms vs 38 ms @100k host — zero hits + bookkeeping), so auto
            # does NOT pick it. See results_v8/m2_hv_refactor.json sweep.
            self.method = "block_numpy" if numpy_ok else "gen"
        elif self.method == "block_numpy" and not numpy_ok:
            raise RuntimeError("block_numpy requested but numpy unavailable; "
                               "use method='gen'/'lru'/'block' or 'auto'")
        self.block = block
        self.cache = HVCache(int(cache_mb * 2**20)) if self.method == "lru" else None
        self.stats = {"queries": 0, "records_examined": 0, "gen_calls": 0,
                      "blocks": 0, "cache_hits": 0, "cache_misses": 0}
        self.last = {}
        # fast-path flag: vectorized generation needs d % 64 == 0
        self._vec_gen = (self.method == "block_numpy" and d % 64 == 0)
        self._nwords = -(-d // 64)          # words cover d bits incl. partial top

    # ---------------- numpy helpers (uint64 wrap == Python & MASK64) ----
    def _gen_block_np(self, keys):
        """(r, nwords) uint64 chunks; bit-identical to gen_packed_hv words."""
        st = _np.asarray(keys, dtype=_np.int64).view(_np.uint64)
        u = _np.uint64
        with _np.errstate(over="ignore"):
            state = (st * u(_GOLDEN) + u((self.seed + 1) & MASK64)).astype(_np.uint64)
            out = _np.empty((len(keys), self._nwords), dtype=_np.uint64)
            for w in range(self._nwords):
                state = (state + u(_GOLDEN)).astype(_np.uint64)
                z = state.copy()
                z = ((z ^ (z >> 30)) * u(_M1)).astype(_np.uint64)
                z = ((z ^ (z >> 27)) * u(_M2)).astype(_np.uint64)
                z = (z ^ (z >> 31)).astype(_np.uint64)
                out[:, w] = z
        return out

    def _qwords(self, q: int):
        return _np.array([(q >> (64 * w)) & MASK64
                          for w in range(self._nwords)], dtype=_np.uint64)

    # ---------------- query paths ---------------------------------------
    def query(self, q_packed: int, tau: float = 0.5):
        """Returns (status, value, conf) EXACTLY as legacy query_noisy:
        argmax sim (first max wins), gate best<tau -> UNKNOWN with raw best
        as conf, TRUE with (best+1)/2. Raises the store's page errors
        (same corruption policy propagation as open_hv)."""
        from ..hv.bitpack import gen_packed_hv
        m = self.store
        d = self.d
        n = m.n
        best, best_v = -2.0, None
        examined = gen_calls = blocks = 0
        ch0 = self.cache.hits if self.cache else 0
        cm0 = self.cache.misses if self.cache else 0
        if self.method in ("block_numpy",):
            qw = self._qwords(q_packed)
            for start in range(0, n, self.block):
                recs = m.read_records(start, min(self.block, n - start))
                examined += len(recs)
                blocks += 1
                if not recs:
                    continue
                if self._vec_gen:
                    mat = self._gen_block_np([k for k, _ in recs])
                    gen_calls += len(recs)
                else:
                    pack = [gen_packed_hv(k, d, self.seed) for k, _ in recs]
                    gen_calls += len(recs)
                    mat = _np.array(
                        [[(p >> (64 * w)) & MASK64 for w in range(self._nwords)]
                         for p in pack], dtype=_np.uint64)
                    # top word keeps only d%64 relevant bits (scalar mask)
                with _np.errstate(over="ignore", invalid="ignore"):
                    x = mat ^ qw
                    dist = _np.bitwise_count(x).sum(axis=1, dtype=_np.int64)
                sims = (d - 2 * dist) / d
                i = int(_np.argmax(sims))          # first max == scalar rule
                if sims[i] > best:
                    best, best_v = float(sims[i]), recs[i][1]
        elif self.method == "block":
            for start in range(0, n, self.block):
                recs = m.read_records(start, min(self.block, n - start))
                examined += len(recs)
                blocks += 1
                packs = [gen_packed_hv(k, d, self.seed) for k, _ in recs]
                gen_calls += len(recs)
                vals = [v for _, v in recs]
                for i, p in enumerate(packs):
                    s = (d - 2 * (q_packed ^ p).bit_count()) / d
                    if s > best:
                        best, best_v = s, vals[i]
        else:   # gen / lru
            use_cache = self.cache is not None
            for start in range(0, n, self.block):
                recs = m.read_records(start, min(self.block, n - start))
                examined += len(recs)
                blocks += 1
                for k, v in recs:
                    if use_cache:
                        p = self.cache.get(k)
                        if p is None:
                            p = gen_packed_hv(k, d, self.seed)
                            self.cache.put(k, p)
                            gen_calls += 1
                    else:
                        p = gen_packed_hv(k, d, self.seed)
                        gen_calls += 1
                    s = (d - 2 * (q_packed ^ p).bit_count()) / d
                    if s > best:
                        best, best_v = s, v
        self.stats["queries"] += 1
        self.stats["records_examined"] += examined
        self.stats["gen_calls"] += gen_calls
        self.stats["blocks"] += blocks
        self.last = {"records_examined": examined, "gen_calls": gen_calls,
                     "blocks": blocks,
                     "cache_hits": (self.cache.hits - ch0) if self.cache else 0,
                     "cache_misses": (self.cache.misses - cm0) if self.cache else 0,
                     "cache_size": self.cache.size if self.cache else 0,
                     "best_sim": best, "n_records": n}
        if best_v is None or best < tau:            # legacy: UNKNOWN keeps raw best
            return ("UNKNOWN", None, best if examined else 0.0)
        return ("TRUE", best_v, (best + 1.0) / 2.0)

    # ---------------- reporting ----------------------------------------
    def resident_bytes(self) -> int:
        """Actual resident HV bytes (cache only; block memory is transient)."""
        if self.cache is None:
            return 0
        return self.cache._bytes

    def summary(self) -> dict:
        return dict(self.stats, method=self.method, block=self.block,
                    d=self.d, seed=self.seed,
                    cache_size=self.cache.size if self.cache else 0,
                    cache_bytes=self.resident_bytes(),
                    vectorized_gen=self._vec_gen)
