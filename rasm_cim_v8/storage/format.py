"""RASM-CIM v0.8 — binary storage formats (Python-only).

Format v1 (SPEC: v0.7 docs/ARCHITECTURE.md §5 — mirrored, Rust never called):
    [b"RASM"][u32 ver=1][u64 payload_len][payload: (i64 key, i64 value) LE * n]
    [u64 FNV-1a over payload]

Format v2 (NEW in v0.8; v1 stays readable) — paged, per-page checksummed:
    header  : [b"RASM"][u32 ver=2][u32 page_recs][u64 n_records]           (24 B)
    pages   : page k: [records payload (up to page_recs*16 B)][u64 FNV]
    index   : sorted (i64 key, u64 data_byte_off) entries grouped in pages
              of page_recs entries, each page [payload][u64 FNV]
    footer  : [u64 index_off][u64 index_n_entries][u64 FNV over footer[:16]] (24 B)

v2 design notes (honest scope):
- the reader keeps the INDEX resident (16 B/record) and pages the DATA
  (per-page LRU); i.e. "index-resident, data-paged". It is NOT full-file
  loading; the data region is only touched page-by-page. For 100k records
  the resident index is 1.6 MB. At 10M it would be 160 MB — documented limit.
- v1 partial reads cannot verify integrity per page (single checksum by
  design); that is exactly why v2 exists.
"""
from __future__ import annotations
import struct
import zlib
from typing import Iterable

MAGIC = b"RASM"
FNV_OFFSET = 0xCBF29CE484222325
FNV_PRIME = 0x100000001B3
MASK64 = (1 << 64) - 1

HDR_V1 = struct.Struct("<4sIQ")          # magic, ver, payload_len
REC = struct.Struct("<qq")               # 16 bytes
FOOTER = struct.Struct("<QQQ")           # index_off, index_n, checksum


def page_sum(data: bytes) -> int:
    """v2 page/footer checksum: zlib.crc32 (C, single pass).

    Design change during M2, measured: pure-python FNV-1a costs ~2-3 ms per
    32 KiB page and dominated streaming p95 (m2_streaming.json round 1);
    crc32 costs ~10 us. Trade-off recorded: crc32 (32-bit) is weaker against
    adversarial crafted corruption than FNV-64, adequate for accidental
    corruption detection, which is the format's stated purpose. v1 KEEPS
    FNV-64 (bit-compatibility with the frozen v0.7 spec)."""
    return zlib.crc32(data) & 0xFFFFFFFF


def fnv1a(data: bytes) -> int:
    h = FNV_OFFSET
    for b in data:
        h = ((h ^ b) * FNV_PRIME) & MASK64
    return h


class StorageError(ValueError):
    """Corruption/format failure. Subclasses ValueError (Rust side mapped
    these to ValueError through PyO3; Python tests keep the same contract)."""


# ---------------------------------------------------------------- v1

def serialize_v1(facts: list[tuple[int, int]]) -> bytes:
    payload = b"".join(REC.pack(k, v) for k, v in facts)
    return MAGIC + struct.pack("<IQ", 1, len(payload)) + payload + struct.pack("<Q", fnv1a(payload))


def deserialize_v1(data: bytes) -> list[tuple[int, int]]:
    if len(data) < 24:
        raise StorageError("too short")
    if data[:4] != MAGIC:
        raise StorageError("bad magic")
    ver = struct.unpack_from("<I", data, 4)[0]
    if ver != 1:
        raise StorageError(f"format version {ver} != 1 for deserialize_v1")
    plen = struct.unpack_from("<Q", data, 8)[0]
    if len(data) != 16 + plen + 8:
        raise StorageError("length mismatch")
    payload = data[16:16 + plen]
    if struct.unpack_from("<Q", data, 16 + plen)[0] != fnv1a(payload):
        raise StorageError("checksum mismatch")
    if plen % 16:
        raise StorageError("payload not multiple of 16")
    return [REC.unpack_from(payload, i) for i in range(0, plen, 16)]


# ---------------------------------------------------------------- v2

PAGE_RECS_DEFAULT = 2048


def write_v2(facts: Iterable[tuple[int, int]], path: str, page_recs: int = PAGE_RECS_DEFAULT) -> dict:
    """Stream records to a v2 file in FILE ORDER (write side needs no full
    materialization beyond building the index, which is accumulated as
    (key, off) tuples and sorted at finalize; facts may be an iterator)."""
    index: list[tuple[int, int]] = []
    n = 0
    off = 24  # header
    with open(path, "wb") as f:
        # 24-byte header: magic + ver + page_recs + n_records (n patched at end)
        f.write(MAGIC + struct.pack("<IIQ", 2, page_recs, 0))
        f.write(b"\x00" * 4)   # reserved pad so data starts at offset 24 (reader math)
        buf = bytearray()
        for k, v in facts:
            rec = REC.pack(k, v)
            index.append((k, off + len(buf)))
            buf += rec
            n += 1
            if len(buf) >= page_recs * 16:
                f.write(bytes(buf))
                f.write(struct.pack("<Q", page_sum(bytes(buf))))
                off += len(buf) + 8
                buf.clear()
        if buf:
            f.write(bytes(buf))
            f.write(struct.pack("<Q", page_sum(bytes(buf))))
            off += len(buf) + 8
        index.sort()
        index_off = off
        ibuf = bytearray()
        ipage = 0
        f.seek(8)
        f.write(struct.pack("<IQ", page_recs, n))  # patch page_recs + n_records
        f.seek(0, 2)
        for k, o in index:
            ibuf += struct.pack("<qQ", k, o)
            ipage += 1
            if ipage == page_recs:
                f.write(bytes(ibuf))
                f.write(struct.pack("<Q", page_sum(bytes(ibuf))))
                ibuf.clear()
                ipage = 0
        if ibuf:
            f.write(bytes(ibuf))
            f.write(struct.pack("<Q", page_sum(bytes(ibuf))))
        f.write(FOOTER.pack(index_off, n, page_sum(FOOTER.pack(index_off, n, 0)[:16])))
    import os
    return {"n_records": n, "page_recs": page_recs, "bytes": os.path.getsize(path),
            "index_off": index_off}


def read_v2_meta(path: str) -> dict:
    with open(path, "rb") as f:
        head = f.read(24)
        if head[:4] != MAGIC:
            raise StorageError("bad magic")
        ver, page_recs, n = struct.unpack("<IIQ", head[4:20])
        if ver != 2:
            raise StorageError(f"version {ver} != 2")
        if page_recs < 1:
            # FUZZ FINDING (atheris t2, regression): page_recs=0 in the header
            # caused ZeroDivisionError in LazyStore._load_index. Now rejected
            # at metadata parse time as a StorageError (validation class).
            raise StorageError(f"page_recs {page_recs} < 1")
        f.seek(-24, 2)
        index_off, index_n, csum = FOOTER.unpack(f.read(24))
        if csum != page_sum(FOOTER.pack(index_off, index_n, 0)[:16]):
            raise StorageError("footer checksum mismatch")
        if index_n != n:
            raise StorageError("footer/header record count mismatch")
        return {"version": 2, "page_recs": page_recs, "n_records": n, "index_off": index_off}
