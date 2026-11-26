"""LazyStore — paged, checksum-verified reader for v2 files (M2).

Modes (measured against each other in bench8, NOT assumed):
  "pread"    os.pread per page — true kernel pread, single 16 B/record index
             resident; data region never bulk-read.
  "mmap"     mmap over the file; pages materialize on touch. HONESTY NOTE:
             mmap is NOT "streaming" in the sense of an application-level
             page budget — resident bytes follow the OS page cache; measured
             separately so nobody has to trust labels.
  "buffered" plain file reads (seek+read) — reference.

Reader model: index-resident (array('q') keys + offsets, 16 B/record),
data-paged with an LRU cache bounded by `cache_pages`. A get() touches
O(1) data pages; scan() streams page-by-page with O(cache) memory.
"""
from __future__ import annotations
import bisect
import mmap
import os
import struct
from array import array
from collections import OrderedDict

from .format import MAGIC, REC, FOOTER, StorageError, fnv1a, page_sum, read_v2_meta

DATA_OFF0 = 24  # first data page start (see format: 24-byte header)


class CorruptPageError(StorageError):
    def __init__(self, page_id: int):
        super().__init__(f"page {page_id} checksum mismatch")
        self.page_id = page_id


class PageCache:
    """Bounded LRU of page payloads (bytes). count-based cap; caller sizes
    pages, so bytes cap = cache_pages * page_recs * 16 is predictable."""

    __slots__ = ("cap", "_d")

    def __init__(self, cap: int):
        self.cap = max(1, cap)
        self._d: OrderedDict[int, bytes] = OrderedDict()

    def get(self, key: int):
        v = self._d.get(key)
        if v is not None:
            self._d.move_to_end(key)
        return v

    def put(self, key: int, val: bytes):
        self._d[key] = val
        if len(self._d) > self.cap:
            self._d.popitem(last=False)


class LazyStore:
    """open_lazy(path) — search without loading the whole file."""

    def __init__(self, path: str, mode: str = "pread", cache_pages: int = 8,
                 on_bad: str = "raise", verify: bool = True):
        if mode not in ("pread", "mmap", "buffered"):
            raise ValueError(f"unknown mode {mode}")
        if on_bad not in ("raise", "skip"):
            raise ValueError(f"unknown on_bad {on_bad}")
        self.path = path
        self.mode = mode
        self.on_bad = on_bad
        self.verify = verify
        self.cache = PageCache(cache_pages)
        self.stats = {"pages_read": 0, "cache_hits": 0, "bytes_read": 0,
                      "index_bytes": 0}
        self.corrupt_pages: set[int] = set()
        self._fd = os.open(path, os.O_RDONLY)
        self._mm = None
        try:
            meta = read_v2_meta(path)
            self.page_recs = meta["page_recs"]
            self.n = meta["n_records"]
            self._page_span = self.page_recs * 16 + 8
            self._index_off = meta["index_off"]
            self._size = os.fstat(self._fd).st_size
            if mode == "mmap":
                self._mm = mmap.mmap(self._fd, 0, prot=mmap.PROT_READ)
            self._load_index()
        except BaseException:
            # M4 FUZZ FINDING (regression guard): a raising init used to leak
            # the os.open fd (EMFILE after ~1k corrupt files, atheris t2).
            if self._mm is not None:
                self._mm.close()
                self._mm = None
            os.close(self._fd)
            raise

    # -- construction helpers -------------------------------------------
    def _pread(self, off: int, ln: int) -> bytes:
        self.stats["bytes_read"] += ln
        if self._mm is not None:
            return self._mm[off:off + ln]
        if self.mode == "pread":
            return os.pread(self._fd, ln, off)
        os.lseek(self._fd, off, os.SEEK_SET)
        return os.read(self._fd, ln)

    def _load_index(self):
        """Read the index region into two resident arrays (16 B/record).
        This is the documented resident part; data stays on disk."""
        keys = array("q")
        offs = array("Q")
        ipr = self.page_recs
        n_pages = (self.n + ipr - 1) // ipr
        poff = self._index_off
        for _ in range(n_pages):
            span = min(ipr, self.n - len(keys))
            payload = self._pread(poff, span * 16)
            if self.verify:
                csum = struct.unpack_from("<Q", self._pread(poff + span * 16, 8))[0]
                if csum != page_sum(payload):
                    raise CorruptPageError(-1)  # corrupt index page: fatal
            for i in range(0, len(payload), 16):
                k, o = struct.unpack_from("<qQ", payload, i)
                keys.append(k)
                offs.append(o)
            poff += span * 16 + 8
        self._keys = keys
        self._offs = offs
        self.stats["index_bytes"] = self.n * 16

    # -- data pages ------------------------------------------------------
    def _page_id_for_off(self, rec_off: int) -> int:
        return (rec_off - DATA_OFF0) // self._page_span

    def _read_page(self, page_id: int, expected_recs: int | None = None) -> bytes:
        hit = self.cache.get(page_id)
        if hit is not None:
            self.stats["cache_hits"] += 1
            return hit
        nrec = self.n
        if expected_recs is None:
            start_rec = page_id * self.page_recs
            span = min(self.page_recs, nrec - start_rec)
        else:
            span = expected_recs
        poff = DATA_OFF0 + page_id * self._page_span
        payload = self._pread(poff, span * 16)
        if len(payload) != span * 16:
            raise StorageError("truncated file (data region)")
        if self.verify:
            try:
                csum = struct.unpack_from("<Q", self._pread(poff + span * 16, 8))[0]
            except struct.error:
                raise StorageError("missing page checksum (truncated)")
            if csum != page_sum(payload):
                if self.on_bad == "raise":
                    raise CorruptPageError(page_id)
                self.corrupt_pages.add(page_id)
                payload = b""
        self.cache.put(page_id, payload)
        self.stats["pages_read"] += 1
        return payload

    # -- public API ------------------------------------------------------
    def read_records(self, start: int, count: int) -> list[tuple[int, int]]:
        """Partial record reading by ordinal range [start, start+count).
        Only the covering pages are fetched (bounded memory)."""
        if start < 0 or count < 0 or start + count > self.n:
            raise IndexError(f"range [{start},{start + count}) outside 0..{self.n}")
        out: list[tuple[int, int]] = []
        p = start // self.page_recs
        end = start + count
        while p * self.page_recs < end:
            payload = self._read_page(p)
            lo = max(start, p * self.page_recs) - p * self.page_recs
            hi = min(end, (p + 1) * self.page_recs) - p * self.page_recs
            if payload:
                for i in range(lo, hi):
                    out.append(REC.unpack_from(payload, i * 16))
            p += 1
        return out

    def get(self, key: int):
        """Exact lookup via resident index + one touched data page.
        Returns ('TRUE', value) | ('CONFLICT', [values]) | ('UNKNOWN', None)."""
        i = bisect.bisect_left(self._keys, key)
        if i >= len(self._keys) or self._keys[i] != key:
            return ("UNKNOWN", None)
        # collect the run of equal keys (same key => same or conflicting vals)
        j = i
        vals: list[int] = []
        while j < len(self._keys) and self._keys[j] == key:
            off = self._offs[j]
            payload = self._read_page(self._page_id_for_off(off))
            if not payload:  # corrupt-skipped page: value unreadable
                return ("CORRUPT", None)
            slot = (off - DATA_OFF0) % self._page_span // 16
            vals.append(REC.unpack_from(payload, slot * 16)[1])
            j += 1
        distinct = sorted(set(vals))
        if len(distinct) > 1:
            return ("CONFLICT", distinct)
        return ("TRUE", distinct[0])

    def scan(self, pred=None):
        """Streaming full scan (memory = one page). pred(k, v) -> keep.
        Corrupt skipped pages yield nothing (recorded in .corrupt_pages)."""
        p = 0
        while p * self.page_recs < self.n:
            payload = self._read_page(p)
            if payload:
                for i in range(0, len(payload), 16):
                    k, v = REC.unpack_from(payload, i)
                    if pred is None or pred(k, v):
                        yield (k, v)
            p += 1

    def query_partial(self, residue: int, modulus: int) -> list[int]:
        """Partial search over the resident INDEX only (keys), no data
        pages touched — matches v0.6 residue semantics for modulus in
        (97, 89, 83); other moduli compute directly from keys."""
        if modulus <= 0:
            raise ValueError("modulus must be > 0")
        # Python floor-mod semantics intentionally match v0.6 index
        return sorted(k for k in self._keys if k % modulus == residue % modulus)

    def to_full_memory(self) -> dict[int, int | list[int]]:
        """Reference loader (the thing streaming must match); NOT used by
        the lazy path — provided for accuracy comparison in benches."""
        d: dict[int, list[int]] = {}
        for k, v in self.read_records(0, self.n):
            d.setdefault(k, []).append(v)
        return {k: (vs[0] if len(set(vs)) == 1 else sorted(set(vs))) for k, vs in d.items()}

    def close(self):
        if self._mm is not None:
            self._mm.close()
            self._mm = None
        os.close(self._fd)
        self.cache = PageCache(1)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()

    def __len__(self):
        return self.n


def open_lazy(path: str, **kw) -> LazyStore:
    """Spec-name alias for LazyStore(path, **kw)."""
    return LazyStore(path, **kw)
