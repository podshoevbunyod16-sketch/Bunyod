"""v0.9 M1 — compact index structures.

Targets, all set by the V09_BASELINE measurements:
  * write_v2 build peak 141.2 MiB @1M (transient list[tuple[int,int]] sort);
  * residue index +103.8 MiB @1M (dict of Python-int lists);
  * resident key/offset arrays are ALREADY compact in v0.8 (array q/Q) —
    this module keeps that shape and adds: tuple-free sorting + array buckets.

sort_to_arrays(): three sort strategies, measured in bench_v9/m1_index.py:
  "tuples" — v0.8 reference (materializes (k,off) tuples)
  "order"  — list of small ints sorted with key=, then gather into arrays
  "numpy"  — np.argsort on a uint64 view (sign-flipped for int64 order)
             auto: numpy if importable AND n >= 200k (import costs ~15 MiB — a
             measured net loss below that; see results_v9/m1_m2.json), else order.

Sign-order trick: int64 k maps to order-unsigned  u = k ^ (1<<63); sorting u
as unsigned == sorting k as signed; numpy view does the flip in C.

ResidueIndex: dict[mod][(residue)] -> array('q') buckets, built from the
SORTED key array so every bucket is sorted with no per-bucket sort (the
query contract returns list[int], matching v0.8's sorted(...) output).

No file format is redefined here: build_v3_base() writes the NEW v3 base
region (documented in journal.py) with the same per-page layout math as v2
(payload + 8-byte crc32 slot), 40-byte header COMMITTED LAST (writer writes
a placeholder first, seeks back to finalize; an unfinished file fails the
header crc and is treated as no-file — same protection v2's footer gave).
"""
from __future__ import annotations
import os
import struct
from array import array
from bisect import bisect_left  # re-exported use in stores

from rasm_cim_v8.storage.format import MAGIC, REC, page_sum

MASK64 = (1 << 64) - 1
SIGN = 1 << 63
HDR_BODY = struct.Struct("<4sHHIQQQ")     # magic,ver,hdrsize,page_recs,n_base,index_off,base_end = 36 B
HDR_SIZE = 40                             # = 36 body + 4 crc (crc covers body only; NO pad — a 44-byte write would clobber data page 0 key bytes; found by hypothesis P5, see tests_v9/test_m5_regressions.py)


# ---------------------------------------------------------------- sorting
def sort_to_arrays(keys: array, offs: array, method: str = "auto"):
    """(keys, offs) same-length arrays -> sorted parallel arrays by key.
    Order-preserving on int64 keys; duplicates keep ORIGINAL relative order
    (stable) — matching v0.8 `index.sort()` on (key, off) tuples is NOT
    guaranteed stable on offs, so we sort by (key, off) like v0.8: tuples
    sorted by key then off. All three methods implement (key, off) order."""
    n = len(keys)
    if n == 0:
        return array("q"), array("Q")
    if method == "auto":
        # measured (m1_m2.json): numpy import = ~15 MiB peak — only attempt
        # it for big builds; small n takes the import-free two-pass sort
        if len(keys) >= 200_000:
            try:
                import numpy  # noqa: F401
                method = "numpy"
            except ImportError:
                method = "order"
        else:
            method = "order"
    if method == "tuples":
        pairs = sorted(zip(keys, offs))
        k2 = array("q", (k for k, _ in pairs))
        o2 = array("Q", (o for _, o in pairs))
        return k2, o2
    if method == "order":
        # two-pass STABLE integer-key sort: secondary (off) then primary (key).
        # Never materializes (key, off) tuples — sort keys are plain ints.
        order = list(range(n))
        order.sort(key=offs.__getitem__)
        order.sort(key=keys.__getitem__)          # stable -> (key, off) order
        k2 = array("q", (keys[i] for i in order))
        o2 = array("Q", (offs[i] for i in order))
        del order
        return k2, o2
    if method == "numpy":
        import numpy as np
        ka = np.frombuffer(keys, dtype=np.int64)             # zero-copy view
        oa = np.frombuffer(offs, dtype=np.uint64)
        ku = ka.view(np.uint64) ^ np.uint64(SIGN)            # order-preserving
        order = np.lexsort((oa, ku))                          # (key, off) lex
        k2 = array("q")
        k2.frombytes(ka[order].tobytes())                     # no Python ints
        o2 = array("Q")
        o2.frombytes(oa[order].tobytes())
        return k2, o2
    raise ValueError(f"unknown sort method {method!r}")


# ---------------------------------------------------------------- residues
class ResidueIndex:
    """array('q') buckets keyed by (pos, residue). Compact + sorted-free."""

    MODS = (97, 89, 83)

    def __init__(self):
        self.buckets: dict[tuple[int, int], array] = {}
        self.entries = 0

    @classmethod
    def from_sorted_keys(cls, keys_sorted: array) -> "ResidueIndex":
        ri = cls()
        blist = [dict() for _ in cls.MODS]
        for k in keys_sorted:                     # already ascending
            for pos, mod in enumerate(cls.MODS):
                r = k % mod
                d = blist[pos]
                b = d.get(r)
                if b is None:
                    d[r] = array("q", (k,))
                else:
                    b.append(k)
        for pos, d in enumerate(blist):
            for r, b in d.items():
                ri.buckets[(pos, r)] = b
                ri.entries += len(b)
        return ri

    def query(self, residue: int, modulus: int) -> list[int]:
        if modulus not in self.MODS:
            raise ValueError(f"modulus {modulus} not in {self.MODS}")
        pos = self.MODS.index(modulus)
        b = self.buckets.get((pos, residue % modulus))
        return list(b) if b is not None else []

    def bytes_resident(self) -> int:
        import sys
        return sum(sys.getsizeof(b) + b.buffer_info()[1] * 8
                   for b in self.buckets.values()) + sys.getsizeof(self.buckets)


# ---------------------------------------------------------------- v3 base
def build_v3_base(facts, path: str, page_recs: int = 2048,
                  sort: str = "auto", tmp: bool = True) -> dict:
    """Stream facts into a v3 base region. `facts` may be an iterator — only
    two flat arrays are held during build (16 B/record), no tuple list.
    Page layout math == v2 (payload + 8B crc32 slot); header (32B) is
    finalized LAST with a crc over itself = base commit."""
    keys = array("q")
    offs = array("Q")
    off = HDR_SIZE
    n = 0
    tmp_path = path + ".building" if tmp else path
    with open(tmp_path, "wb") as f:
        f.write(b"\x00" * HDR_SIZE)               # placeholder header
        buf = bytearray()
        base = off
        for k, v in facts:
            keys.append(k)
            offs.append(base + len(buf))
            buf += REC.pack(k, v)
            n += 1
            if len(buf) >= page_recs * 16:
                f.write(bytes(buf))
                f.write(struct.pack("<II", page_sum(bytes(buf)), 0))
                base += len(buf) + 8
                buf.clear()
        if buf:
            f.write(bytes(buf))
            f.write(struct.pack("<II", page_sum(bytes(buf)), 0))
            base += len(buf) + 8
        k2, o2 = sort_to_arrays(keys, offs, sort)
        del keys, offs
        index_off = base
        ipage = 0
        ibuf = bytearray()
        for i in range(n):
            ibuf += struct.pack("<qQ", k2[i], o2[i])
            ipage += 1
            if ipage == page_recs:
                f.write(bytes(ibuf))
                f.write(struct.pack("<II", page_sum(bytes(ibuf)), 0))
                ibuf.clear()
                ipage = 0
        if ibuf:
            f.write(bytes(ibuf))
            f.write(struct.pack("<II", page_sum(bytes(ibuf)), 0))
        base_end = f.tell()
        body = HDR_BODY.pack(MAGIC, 3, HDR_SIZE, page_recs, n, index_off, base_end)
        hdr = body + struct.pack("<I", page_sum(body))   # exactly 40 B:
        # NO trailing pad — 44 would clobber the first data-page key bytes
        assert len(hdr) == HDR_SIZE
        f.seek(0)
        f.write(hdr)                              # commit point (last base write)
        f.flush()
        if tmp:
            os.replace(tmp_path, path)
    return {"n_base": n, "page_recs": page_recs, "index_off": index_off,
            "base_end": base_end, "bytes": os.path.getsize(path)}


def read_v3_header(path: str) -> dict:
    """Returns meta or raises StorageError if base was never committed."""
    from rasm_cim_v8.storage.format import StorageError
    with open(path, "rb") as f:
        hdr = f.read(HDR_SIZE)
    if len(hdr) < HDR_SIZE or hdr[:4] != MAGIC:
        raise StorageError("v3: bad magic/short file")
    magic, ver, hsz, page_recs, n, index_off, base_end = HDR_BODY.unpack(hdr[:36])
    crc = struct.unpack_from("<I", hdr, 36)[0]
    if ver != 3:
        raise StorageError(f"v3: version {ver} != 3")
    if crc != page_sum(hdr[:36]):
        raise StorageError("v3: header checksum mismatch (base not committed)")
    return {"version": 3, "page_recs": page_recs, "n_base": n,
            "index_off": index_off, "base_end": base_end, "hdr_size": hsz}
