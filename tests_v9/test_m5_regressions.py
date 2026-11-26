"""v0.9 M5 — regression tests for every bug fuzzing/property-testing found.

BUG-1 (found by hypothesis property P5, before any atheris run):
  build_v3_base wrote header = body(36)+crc(4)+pad(4) = 44 bytes, overflowing
  the 40-byte header by 4 zero bytes that clobbered the low half of the FIRST
  data-page record's key. Masked whenever the smallest key is 0 (0|0 = 0):
  all prior benches used facts starting at (0, 0) and never noticed.
  Fix: header is exactly HDR_SIZE bytes. Regression: first-key round-trip.
"""
from __future__ import annotations
import os
import sys

sys.path.insert(0, "/home/user")
import pytest

from rasm_cim_v9.storage.index import HDR_SIZE, build_v3_base
from rasm_cim_v9.storage.journal import V3Store

TMP = "/tmp/rasm_v9_regs"
os.makedirs(TMP, exist_ok=True)


@pytest.mark.parametrize("k,v", [(42, 1181), (1, 7), (-5, -9), (2**40, 3)])
def test_bug1_first_key_roundtrip(k, v):
    """The smallest key lives in data page 0, slot 0 — the clobber site."""
    p = f"{TMP}/bug1_{k}.rasm"
    build_v3_base([(k, v), (k + 10**6, v + 1)], p, page_recs=16)
    st = V3Store(p)
    assert st.get(k) == ("TRUE", v)
    assert st.get(k + 10**6) == ("TRUE", v + 1)
    st.close()


def test_bug1_header_size_exactly_40():
    p = f"{TMP}/bug1_hdr.rasm"
    info = build_v3_base([(7, 9), (8, 10)], p, page_recs=16)
    import struct
    from rasm_cim_v8.storage.format import REC, page_sum
    raw = open(p, "rb").read()
    payload = raw[HDR_SIZE:HDR_SIZE + 32]               # page holds BOTH recs
    assert payload == REC.pack(7, 9) + REC.pack(8, 10)  # untouched first rec
    crc = struct.unpack_from("<I", raw, HDR_SIZE + 32)[0]
    assert crc == page_sum(payload)                     # crc sees true bytes


def test_bug2_atheris_memoryerror_count_bounds():
    """BUG-2 (atheris `journal`/`records`, coverage-guided): frame headers are
    unprotected by design (commit-per-frame), so a corrupted count field used
    to drive `pread(fd, count*16)` -> MemoryError (68 GB claim). Fixed with an
    fstat bounds check BEFORE allocating: such a frame is classified as a
    torn tail; base answers stay intact; no MemoryError, no unbounded alloc."""
    for hx in ("2323232323232323232323232323230a", "000000cf0000000a"):
        data = bytes.fromhex(hx)
        p = f"{TMP}/bug2_{hx[:8]}.rasm"
        facts = [(i * 7 + 3, (i * 7919) % 15015) for i in range(200)]
        build_v3_base(facts, p, page_recs=16)
        with open(p, "ab") as f:
            f.write(data)
        st = V3Store(p)                       # must NOT raise MemoryError
        assert st.get(facts[0][0]) == ("TRUE", facts[0][1])
        assert st.n_records() == 200
        st.close()


def test_bug1_many_page_sizes_first_key():
    for pr in (1, 2, 3, 16, 1024):
        p = f"{TMP}/bug1_pr{pr}.rasm"
        facts = [(i * 100 + 3, i * 100 + 7) for i in range(30)]
        build_v3_base(facts, p, page_recs=pr)
        st = V3Store(p)
        assert all(st.get(k) == ("TRUE", v) for k, v in facts), pr
        st.close()
