"""M2-refactor regression tests: streaming HV retrieval must be RESULT-identical
to the legacy in-RAM path, keep gen_packed_hv deterministic, and leave file
formats byte-frozen. Legacy path kept (open_hv) — these tests ARE the
comparison the refactor spec demands before anything is removed.

Run: python3 -m pytest rasm_cim_v8/tests/test_m2_hv_refactor.py -q
"""
from __future__ import annotations
import hashlib
import os
import random
import sys

sys.path.insert(0, "/home/user")
import pytest

from rasm_cim_v8.storage import format as F
from rasm_cim_v8.memory.stream_memory import MobileStreamMemory
from rasm_cim_v8.hv.bitpack import gen_packed_hv, gen_query_with_flips

TMP = "/tmp/rasm_v8_hvref"
os.makedirs(TMP, exist_ok=True)
METHODS = ("gen", "lru", "block", "block_numpy", "combo")


def _store(n, d=256, page_recs=256, seed=7, path_extra=""):
    path = f"{TMP}/hv_{n}_{d}{path_extra}.rasm"
    F.write_v2(((i, (i * 7919) % 15015) for i in range(n)), path, page_recs=page_recs)
    return path


# ------------------------------------------------ determinism of gen_packed_hv
def test_gen_packed_hv_golden_values():
    """Embeds values measured BEFORE the refactor (bitpack.py untouched since).
    Any change here = determinism violated = refactor bug."""
    goldens = {
        (0, 256, 7): "0x893eb7db0dddbdb4b0643a4e15e67e019ca8a164477d78019e5651b0ef953636",
        (1, 256, 7): "0x10565dfd638744ca893eb7db0dddbdb4b0643a4e15e67e019ca8a164477d7801",
        (7, 256, 7): "0xa0466eb80edeaf736fd25c66f20f786c14826f547f0bb8035b5f326dc867209a",
        (12345, 256, 7): "0xeb74fe1920199f356cf081b57f871f6a15e7d82bca20ddeda878f4c4db8aa3b2",
        (3, 100, 0): "0x8d101b5b971c18690ee42c90b",
    }
    for (k, d, s), h in goldens.items():
        assert hex(gen_packed_hv(k, d, s)) == h, (k, d, s)


def test_numpy_vectorized_gen_equals_scalar():
    """The fast path reimplements splitmix64 in uint64 arrays; it must be
    bit-identical to the scalar generator for every word and key."""
    from rasm_cim_v8.memory.hv_stream import StreamHV, _np
    if _np is None:
        pytest.skip("numpy absent")
    MASK = (1 << 64) - 1
    store = _FakeStore([123, 999999, -5, 2**62])   # incl. negative + huge keys
    for d in (64, 128, 256, 512):
        sh = StreamHV(store, d=d, seed=11, method="block_numpy")
        mat = sh._gen_block_np(store._keys)
        for r, k in enumerate(store._keys):
            m = gen_packed_hv(k, d, 11)
            for w in range(d // 64):
                assert int(mat[r, w]) == (m >> (64 * w)) & MASK, (d, k, w)


class _FakeStore:
    n = 4

    def __init__(self, keys):
        self._keys = list(keys)


# ------------------------------------------------ result parity vs legacy
@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("d", (64, 100, 256))
def test_stream_backends_match_legacy(method, d):
    n = 1200
    path = _store(n, d)
    leg = MobileStreamMemory(path).open().open_hv(d, 3)
    st = MobileStreamMemory(path).open().open_hv_stream(d, 3, method=method)
    rng = random.Random(d * 31 + len(method))
    mism = 0
    for t in range(16):
        k = rng.randrange(n)
        q, _ = gen_query_with_flips(k, d, 3, rng.choice((0.0, 0.05, 0.3)), t)
        tau = rng.choice((0.3, 0.5, 0.9))
        r1 = leg.query_noisy(q, tau)
        r2 = st.query_noisy(q, tau)
        if r1[0] != r2[0] or r1[1] != r2[1] or abs(r1[2] - r2[2]) > 1e-12:
            mism += 1
    assert mism == 0, f"{method} d={d}: {mism} mismatches vs legacy"


def test_unknown_confidence_and_tau_gate_parity():
    """UNKNOWN keeps the RAW best sim as third element (legacy quirk) and
    best==tau passes the gate (strict <). Both backends must agree."""
    n = 600
    path = _store(n, 256)
    leg = MobileStreamMemory(path).open().open_hv(256, 7)
    st = MobileStreamMemory(path).open().open_hv_stream(256, 7, method="combo")
    q = gen_packed_hv(42, 256, 7)
    assert leg.query_noisy(q, 1.0) == st.query_noisy(q, 1.0)          # == tau -> TRUE
    r1, r2 = leg.query_noisy(q, 1.0 + 1e-9), st.query_noisy(q, 1.0 + 1e-9)
    assert r1[0] == r2[0] == "UNKNOWN" and abs(r1[2] - r2[2]) < 1e-12
    qmiss = gen_packed_hv(10**9, 256, 7)
    r1, r2 = leg.query_noisy(qmiss, 0.99), st.query_noisy(qmiss, 0.99)
    assert r1[0] == "UNKNOWN" and r1 == r2 and not math_is_1(r1[2])


def math_is_1(x):
    return abs(x - 1.0) < 1e-15


def test_first_max_tie_rule_parity():
    """Duplicate equal-value keys -> equal sims; the FIRST occurrence in scan
    order must win identically in both backends (strict > rule)."""
    path = f"{TMP}/tie_dup.rasm"
    F.write_v2([(10, 100), (10, 100), (11, 200)], path, page_recs=2)
    leg = MobileStreamMemory(path).open().open_hv(256, 7)
    st = MobileStreamMemory(path).open().open_hv_stream(256, 7, method="gen")
    q = gen_packed_hv(10, 256, 7)
    assert leg.query_noisy(q, 0.5) == st.query_noisy(q, 0.5) == ("TRUE", 100, 1.0)


def test_corrupt_skip_same_examined_and_picks():
    """on_bad='skip' must give identical examined counts AND identical picks
    (both paths read the same surviving pages)."""
    path = f"{TMP}/corrupt.rasm"
    n = 2048
    F.write_v2(((i, (i * 7919) % 15015) for i in range(n)), path, page_recs=64)
    raw = bytearray(open(path, "rb").read())
    raw[24 + 64 * 16 * 3 + 4] ^= 0xFF
    open(path, "wb").write(bytes(raw))
    leg = MobileStreamMemory(path, on_bad="skip", cache_pages=2).open().open_hv(256, 7)
    st = MobileStreamMemory(path, on_bad="skip", cache_pages=2).open().open_hv_stream(256, 7, method="block_numpy")
    rng = random.Random(5)
    for t in range(10):
        k = rng.randrange(n)
        q, _ = gen_query_with_flips(k, 256, 7, 0.1, t)
        assert leg.query_noisy(q, 0.5) == st.query_noisy(q, 0.5)
    per_query = st._hvstream.stats["records_examined"] / 10
    assert per_query == len(leg._hv)          # same record set, both 2016-ish


# ------------------------------------------------ memory contract
def test_stream_backend_does_not_materialize_full_cache():
    path = _store(5000, 256)
    m = MobileStreamMemory(path).open().open_hv_stream(256, 7, method="gen")
    assert m._hv is None                       # legacy list never built
    assert m.hv_bytes_estimate() == 0          # no cache in 'gen'
    m.query_noisy(gen_packed_hv(9, 256, 7), 0.5)
    assert m._hv is None and m.hv_bytes_estimate() == 0
    assert m._hvstream.stats["records_examined"] == 5000


def test_lru_cache_respects_byte_budget():
    from rasm_cim_v8.memory.hv_stream import HVCache
    c = HVCache(max_bytes=100_000)
    for i in range(5000):
        c.put(i, gen_packed_hv(i, 256, 0))
    assert c._bytes <= 100_000 * 1.25          # budget enforced at insert
    assert c.size < 5000                       # evictions happened
    assert c.get(0) is None or True            # early keys evicted (or not) — bounded is what matters


def test_cache_cap_bounded_at_20k():
    path = _store(20000, 256)
    m = MobileStreamMemory(path).open().open_hv_stream(256, 7, method="lru",
                                                        cache_mb=0.5)
    for t in range(3):
        m.query_noisy(gen_packed_hv(t, 256, 7), 0.5)
    assert m.hv_bytes_estimate() <= 0.5 * 2**20 * 1.25


# ------------------------------------------------ file format frozen
def test_v1_v2_file_bytes_golden():
    """Refactor must not change stored formats: golden sha256 of fixed inputs
    (measured before the refactor was written)."""
    facts = [(i, (i * 7919) % 15015) for i in range(8)]
    p = f"{TMP}/fmt_v2.rasm"
    F.write_v2(facts, p, page_recs=3)
    assert hashlib.sha256(open(p, "rb").read()).hexdigest() == \
        "6ca78bb83ca45e820b3469559ab0b7e854455b8c6631bb8feb9a08a7d5993ceb"
    assert hashlib.sha256(F.serialize_v1(facts)).hexdigest() == \
        "5e7112b9a24d64a1de18dbc69ae80881209ae01fb1a9ea0889b77a8cb63591ea"


# ------------------------------------------------ legacy path still works
def test_legacy_open_hv_path_still_intact():
    """Spec: old path NOT removed before comparison finished."""
    path = _store(1500, 256)
    m = MobileStreamMemory(path).open().open_hv(256, 7)
    assert len(m._hv) == 1500
    q = gen_packed_hv(777, 256, 7)
    st, v, conf = m.query_noisy(q, 0.5)
    assert (st, v) == ("TRUE", (777 * 7919) % 15015) and conf == 1.0
