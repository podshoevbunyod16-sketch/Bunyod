"""v0.9 M2 — v3 storage: committed v2-style base + append-only journal.

Format v3 (NEW, documented here and in V09 docs; v1/v2 UNTOUCHED and still
readable via open_any()). Rationale for a new version instead of mutating v2:
v2's layout ends with a footer at EOF-24 and the reader trusts it; appending a
journal would sit past the footer, i.e. silently changing v2 semantics —
forbidden by the spec ("не изменять v2 молча").

    v3 file := header(40B, crc32, COMMITTED LAST by the base writer)
             | data pages   : [records × 16 B][u32 crc32][u32 0]   (same bytes
             | index pages  : [(i64 key,u64 off) × 16 B][u32 crc][u32 0] as v2's
                               <Q crc slot — slot content is byte-identical
                               since crc32 < 2^32)
             | journal      : zero or more frames
    frame    := [u64 seq][u32 count][u32 crc32(body)] body( count × (i64,i64) )
                the crc at the frame HEAD (before body) is deliberate: a
                crash mid-body leaves an incomplete tail that the reader
                detects as short/invalid and truncates logically.
    commit   := a frame is committed iff len(frame)==complete AND crc matches
                AND seq == last_seq+1. Anything else ends replay.
    journal_off/base_end live in the header, so no footer rewrite is needed
    to grow the file — appends never invalidate base bytes.

Guarantees (each has a kill-injection / property test in tests_v9):
  R3.1 base commit = header last; interrupted base ⇒ header crc fail ⇒
       file rejected (never half-trusted).
  R3.2 interrupted txn (crash mid-frame) ⇒ trailing garbage ignored; last
       committed txn intact; reopen returns it.
  R3.3 replay idempotency: frames with seq <= last_committed are SKIPPED,
       so re-appending the same txn after a retry cannot duplicate data.
  R3.4 corrupt journal in the MIDDLE (bad crc where more bytes follow) ⇒
       controlled JournalCorruptError (not silent, not a crash class).
  R3.5 read semantics identical to v0.8 LazyStore (bisect runs,
       sorted(set(vals)) dedup/CONFLICT, TRUE/UNKNOWN, per-page crc policy
       raise|skip); journal values take precedence-free union like extra
       records with the same rules.
Legacy note: v2 has no free-slot management, so compaction = rewrite
(build_v3_base from merged scan, journal cleared) — atomic via rename.
"""
from __future__ import annotations
import mmap
import os
import struct
from array import array
from bisect import bisect_left

from rasm_cim_v8.storage.format import MAGIC, REC, page_sum, StorageError
from rasm_cim_v8.storage.streaming import PageCache, CorruptPageError
from .index import HDR_SIZE, ResidueIndex, build_v3_base, read_v3_header

FRAME_HDR = struct.Struct("<QII")     # seq, count, crc32(body)
FRAME_HDR_SIZE = 16


class JournalCorruptError(StorageError):
    pass


def _last_seq_from_headers(path: str) -> int:
    """Walk journal FRAME HEADERS ONLY from base_end; stop at first
    short/bad frame. Never touches data pages or the resident index."""
    meta = read_v3_header(path)
    p = meta["base_end"]
    last = 0
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        while p + FRAME_HDR_SIZE <= size:
            hdr = os.pread(f.fileno(), FRAME_HDR_SIZE, p)
            if len(hdr) < FRAME_HDR_SIZE:
                break
            seq, count, _crc = FRAME_HDR.unpack(hdr)
            if seq > last:
                last = seq
            p += FRAME_HDR_SIZE + count * 16
    return last


# ---------------------------------------------------------------- writer
def append_txn(path: str, records: list[tuple[int, int]], seq: int | None = None,
               fsync: bool = True) -> int:
    """Append one txn frame atomically-enough for crash rules R3.2/R3.3.
    seq auto = last+1. Returns committed seq. Body is written before crc is
    verifiable — a crash here leaves an incomplete tail (ignored on replay)."""
    last_seq = _last_seq_from_headers(path)   # O(#frames) header-only walk —
    # NOT a full index read: single-record update must stay O(frame)
    if seq is None:
        seq = last_seq + 1
    elif seq <= last_seq:
        return seq                              # R3.3: already-committed replay
    body = b"".join(REC.pack(k, v) for k, v in records)
    frame = FRAME_HDR.pack(seq, len(records), page_sum(body)) + body
    with open(path, "r+b") as f:
        f.seek(0, os.SEEK_END)
        f.write(frame)
        f.flush()
        if fsync:
            os.fsync(f.fileno())
    return seq


# ---------------------------------------------------------------- reader
class V3Store:
    """Compact-index store over v3 files with journal semantics.
    Resident: keys/offs arrays (16 B/rec base) + journal arrays (24 B/entry)
    + optional bounded residue index."""

    def __init__(self, path: str, cache_pages: int = 8, on_bad: str = "raise",
                 recover_quiet: bool = False, mode: str = "pread"):
        if on_bad not in ("raise", "skip"):
            raise ValueError(f"unknown on_bad {on_bad}")
        if mode not in ("pread", "mmap"):
            raise ValueError(f"unknown mode {mode}")
        self.path = path
        self.mode = mode
        self._mm = None
        self.on_bad = on_bad
        self.cache = PageCache(cache_pages)
        self.stats = {"pages_read": 0, "cache_hits": 0, "bytes_read": 0,
                      "journal_replayed": 0, "journal_skipped_frames": 0,
                      "journal_truncated_bytes": 0}
        self.corrupt_pages: set[int] = set()
        meta = read_v3_header(path)
        self.page_recs = meta["page_recs"]
        self.n_base = meta["n_base"]
        self._index_off = meta["index_off"]
        self._base_end = meta["base_end"]
        self._page_span = self.page_recs * 16 + 8
        self._fd = os.open(path, os.O_RDONLY)
        try:
            if mode == "mmap":
                self._mm = mmap.mmap(self._fd, 0, prot=mmap.PROT_READ)
            self._load_index()
            self._replay_journal(recover_quiet)
        except BaseException:
            if self._mm is not None:
                self._mm.close()
            os.close(self._fd)
            raise

    # -- base pages (same math as v0.8, v3 offsets) ------------------------
    def _pread(self, off: int, ln: int) -> bytes:
        self.stats["bytes_read"] += ln
        if self._mm is not None:
            return self._mm[off:off + ln]
        return os.pread(self._fd, ln, off)

    def _load_index(self):
        keys, offs = array("q"), array("Q")
        ipr = self.page_recs
        n_pages = (self.n_base + ipr - 1) // ipr if self.n_base else 0
        poff = self._index_off
        for _ in range(n_pages):
            span = min(ipr, self.n_base - len(keys))
            payload = self._pread(poff, span * 16)
            csum = struct.unpack_from("<II", self._pread(poff + span * 16, 8))[0]
            if csum != page_sum(payload):
                raise CorruptPageError(-1)      # index corruption: fatal
            for i in range(0, len(payload), 16):
                k, o = struct.unpack_from("<qQ", payload, i)
                keys.append(k)
                offs.append(o)
            poff += span * 16 + 8
        self._keys = keys                        # sorted (key,off) order
        self._offs = offs
        self.stats["index_bytes"] = keys.buffer_info()[1] * 8 + offs.buffer_info()[1] * 8

    def _page_id_for_off(self, rec_off: int) -> int:
        return (rec_off - HDR_SIZE) // self._page_span

    def _read_page(self, page_id: int) -> bytes:
        hit = self.cache.get(page_id)
        if hit is not None:
            self.stats["cache_hits"] += 1
            return hit
        start = HDR_SIZE + page_id * self._page_span
        span = min(self.page_recs, self.n_base - page_id * self.page_recs)
        if span <= 0:
            return b""
        payload = self._pread(start, span * 16)
        if len(payload) != span * 16:
            payload = payload.ljust(span * 16, b"\x00")
            ok = False
        else:
            cslot = self._pread(start + span * 16, 8)
            ok = struct.unpack_from("<II", cslot)[0] == page_sum(payload)
        if not ok:
            if self.on_bad == "raise":
                raise CorruptPageError(page_id)
            self.corrupt_pages.add(page_id)
            payload = b""
        self.cache.put(page_id, payload)
        self.stats["pages_read"] += 1
        return payload

    def _rec_at(self, off: int) -> tuple[int, int]:
        payload = self._read_page(self._page_id_for_off(off))
        if not payload:
            return None                          # None sentinel = CORRUPT
        slot = (off - HDR_SIZE) % self._page_span // 16
        return REC.unpack_from(payload, slot * 16)

    # -- journal replay -----------------------------------------------------
    def _replay_journal(self, recover_quiet: bool):
        self.journal_keys = array("q")
        self.journal_vals = array("q")
        self.journal_offs = array("Q")
        self.last_seq = 0
        p = self._base_end
        size = os.fstat(self._fd).st_size
        pending = []
        while True:
            hdr = os.pread(self._fd, FRAME_HDR_SIZE, p)
            if len(hdr) < FRAME_HDR_SIZE:
                self.stats["journal_truncated_bytes"] += len(hdr)
                break                            # R3.2 tail shorter than a header
            seq, count, crc = FRAME_HDR.unpack(hdr)
            # BOUNDS first (fuzz BUG-2): a corrupt count must never drive a
            # giant allocation — short-read as torn tail instead of MemoryError
            if count > (size - p - FRAME_HDR_SIZE) // 16:
                self.stats["journal_truncated_bytes"] += FRAME_HDR_SIZE \
                    + max(0, size - p - FRAME_HDR_SIZE)
                break                            # claimed body runs past EOF
            body = os.pread(self._fd, count * 16, p + FRAME_HDR_SIZE)
            if len(body) < count * 16:
                self.stats["journal_truncated_bytes"] += len(hdr) + len(body)
                break                            # R3.2 crash mid-body
            if page_sum(body) != crc:
                # corruption: if nothing follows at all, treat as torn tail
                # (crash exactly at crc byte) — data was never committed;
                # if MORE frames follow behind, it's mid-file corruption.
                after = os.pread(self._fd, FRAME_HDR_SIZE, p + FRAME_HDR_SIZE + len(body))
                if len(after) >= FRAME_HDR_SIZE and not recover_quiet:
                    raise JournalCorruptError(
                        f"journal frame at {p} has bad crc and successors "
                        f"(mid-file corruption)")
                self.stats["journal_truncated_bytes"] += FRAME_HDR_SIZE + len(body)
                break
            if seq <= self.last_seq:             # R3.3 duplicate replay
                self.stats["journal_skipped_frames"] += 1
                p += FRAME_HDR_SIZE + count * 16
                continue
            self.last_seq = seq
            for i in range(count):
                k, v = REC.unpack_from(body, i * 16)
                pending.append((k, v, p + FRAME_HDR_SIZE + i * 16))
            self.stats["journal_replayed"] += count
            p += FRAME_HDR_SIZE + count * 16
        # FILE order (scan/read_records) and SORTED order (bisect lookup) are
        # separate views — journal frames arrive unsorted.
        for k, v, off in pending:
            self.journal_keys.append(k)        # frame order
            self.journal_vals.append(v)
            self.journal_offs.append(off)
        m = len(pending)
        sidx = sorted(range(m), key=lambda i: (pending[i][0], pending[i][2]))
        self.jkeys_s = array("q", (pending[i][0] for i in sidx))
        self.jvals_s = array("q", (pending[i][1] for i in sidx))
        self.joffs_s = array("Q", (pending[i][2] for i in sidx))

    # -- queries (v0.8-compatible semantics over base+journal) ------------
    def n_records(self) -> int:
        return self.n_base + len(self.journal_keys)

    @property
    def n(self) -> int:              # duck-type parity with v0.8 LazyStore
        return self.n_base + len(self.journal_keys)   # so StreamHV(store) works

    def get(self, key: int):
        vals = []
        lo = bisect_left(self._keys, key)
        i = lo
        while i < len(self._keys) and self._keys[i] == key:
            r = self._rec_at(self._offs[i])
            if r is None:
                return ("CORRUPT", None)
            vals.append(r[1])
            i += 1
        j = bisect_left(self.jkeys_s, key)
        while j < len(self.jkeys_s) and self.jkeys_s[j] == key:
            vals.append(self.jvals_s[j])
            j += 1
        if not vals:
            return ("UNKNOWN", None)
        distinct = sorted(set(vals))
        if len(distinct) > 1:
            return ("CONFLICT", distinct)
        return ("TRUE", distinct[0])

    def read_records(self, start: int, count: int) -> list[tuple[int, int]]:
        """FILE ORDER = base records (file order), then journal records."""
        if start < 0 or count < 0 or start + count > self.n_records():
            raise IndexError("range outside 0..n")
        out = []
        n = min(count, max(0, self.n_base - start))
        if n:
            p = start // self.page_recs
            end = start + n
            while p * self.page_recs < end:
                payload = self._read_page(p)
                lo = max(start, p * self.page_recs) - p * self.page_recs
                hi = min(end, (p + 1) * self.page_recs) - p * self.page_recs
                if payload:
                    for i in range(lo, hi):
                        out.append(REC.unpack_from(payload, i * 16))
                p += 1
        if start + count > self.n_base:
            j0 = max(0, start - self.n_base)
            for i in range(j0, min(len(self.journal_keys), start + count - self.n_base)):
                out.append((self.journal_keys[i], self.journal_vals[i]))
        return out

    def scan(self):
        for i in range(0, self.n_base, self.page_recs):
            for rec in self.read_records(i, min(self.page_recs, self.n_base - i)):
                yield rec
        for i in range(len(self.journal_keys)):
            yield (self.journal_keys[i], self.journal_vals[i])

    def build_residue(self) -> ResidueIndex:
        import heapq
        if not len(self.journal_keys):
            return ResidueIndex.from_sorted_keys(self._keys)
        merged = array("q", heapq.merge(self._keys, self.jkeys_s))
        return ResidueIndex.from_sorted_keys(merged)

    def __enter__(self):                      # v0.10 production polish: no leaked fds
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def close(self):
        try:
            if self._mm is not None:
                self._mm.close()
                self._mm = None
            os.close(self._fd)
        except OSError:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


def open_any(path: str, **kw):
    """Dispatch by format version: v1 blob -> list, v2 -> v0.8 LazyStore
    (read-only reuse), v3 -> V3Store. Old files keep old readers verbatim."""
    with open(path, "rb") as f:
        head = f.read(8)
    ver = struct.unpack_from("<I", head, 4)[0] if head[:4] == MAGIC else 0
    # v3 stores ver as u16 followed by hdr_size u16 -> raw u32 reads 3|(40<<16)
    if head[:4] == MAGIC and ver not in (1, 2) and (ver & 0xFFFF) == 3 and (ver >> 16) >= 32:
        ver = 3
    if ver == 1:
        from rasm_cim_v8.storage.format import deserialize_v1
        return deserialize_v1(open(path, "rb").read())
    if ver == 2:
        from rasm_cim_v8.storage.streaming import open_lazy
        return open_lazy(path, **kw)
    if ver == 3:
        return V3Store(path, **kw)
    raise StorageError(f"unrecognized file (magic={head[:4]!r} ver={ver})")


def compact(path: str, page_recs: int | None = None) -> dict:
    """Fold committed journal into a fresh base (v2-style full rewrite for
    THIS op — documented; the journal exists so we don't need it per-update).
    Atomic: build .new, os.replace."""
    st = V3Store(path)
    pr = page_recs or st.page_recs
    try:
        meta = build_v3_base(list(st.scan()), path + ".new", page_recs=pr)
    finally:
        st.close()
    os.replace(path + ".new", path)
    return meta
