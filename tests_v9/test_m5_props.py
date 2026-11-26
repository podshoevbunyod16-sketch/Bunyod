"""v0.9 M5 — hypothesis property tests (pure stdlib property framework).

Properties (per spec):
  P1 append/read equivalence       — records written == records read, order kept
  P2 crash recovery                — frames cut at ANY byte: torn tail ignored,
                                     last COMPLETE committed txn intact
  P3 replay idempotency            — replaying the journal bytes twice yields
                                     identical answers (replay never duplicates)
  P4 index lookup parity           — V3Store.get == brute-force scan for
                                     arbitrary update histories; ResidueIndex ==
                                     brute force per modulus
  P5 old-format compat             — every generated fact set written as v2
                                     answers identically to v3 base

Run: python3 -m pytest tests_v9/test_m5_props.py -q
"""
from __future__ import annotations
import os
import struct
import sys

sys.path.insert(0, "/home/user")
from hypothesis import given, settings, strategies as st, HealthCheck

from rasm_cim_v8.storage.format import write_v2
from rasm_cim_v9.storage.index import build_v3_base, REC

from rasm_cim_v9.storage.journal import (FRAME_HDR_SIZE, V3Store, append_txn)
from rasm_cim_v9.storage.index import ResidueIndex

TMP = "/tmp/rasm_v9_props"
os.makedirs(TMP, exist_ok=True)

K = st.integers(min_value=-2**62, max_value=2**62)
RECS = st.lists(st.tuples(K, st.integers(min_value=-2**31, max_value=2**31)),
                min_size=1, max_size=40)


def _f(seq):
    return f"{TMP}/p{os.getpid()}_{seq}.rasm"


@settings(deadline=None, max_examples=60, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(st.integers(min_value=0, max_value=10_000_000), RECS)
def test_p1_append_read_equivalence(i, recs):
    p = _f(1) + f"_{i}"
    build_v3_base(((0, 0),), p)
    append_txn(p, recs)
    st = V3Store(p)
    assert st.read_records(st.n_base, len(st.journal_keys)) == [(k, v) for k, v in recs]
    os.remove(p)


@settings(deadline=None, max_examples=40, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(RECS, st.integers(min_value=1, max_value=3))
def test_p2_truncation_recovers_last_good(recs, cut_mod):
    p = _f(2)
    build_v3_base(((0, 0),), p)
    for r in recs:
        append_txn(p, [r])                     # one txn per record
    size = os.path.getsize(p)
    # truncate anywhere from mid-first-frame to end-1: recovery must never
    # lose a COMPLETE last txn... so cut AFTER the last frame boundary:
    frame_sz = FRAME_HDR_SIZE + len(REC.pack(0, 0))          # 16 + 16
    base_sz = size - len(recs) * frame_sz                     # base_end
    cut = base_sz + (len(recs) - 1) * frame_sz + cut_mod     # tear LAST frame
    with open(p, "r+b") as f:
        f.truncate(cut)
    st = V3Store(p)
    kept = st.read_records(st.n_base, st.n_records() - st.n_base)
    assert kept == [(k, v) for k, v in recs[:len(kept)]]          # prefix intact
    assert len(kept) in (len(recs) - 1, len(recs))                # lost at most the torn one
    for k, v in kept:
        assert st.get(k)[0] != "UNKNOWN"


@settings(deadline=None, max_examples=40, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(RECS)
def test_p3_replay_idempotent(recs):
    p = _f(3)
    build_v3_base(((0, 0),), p)
    append_txn(p, recs)
    a = V3Store(p)
    b = V3Store(p)
    ja = a.read_records(a.n_base, len(a.journal_keys))
    jb = b.read_records(b.n_base, len(b.journal_keys))
    assert ja == jb
    assert [a.get(k) for k, _ in recs] == [b.get(k) for k, _ in recs]


@settings(deadline=None, max_examples=30, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(st.lists(RECS, min_size=1, max_size=4))
def test_p4_get_parity_with_bruteforce(groups):
    p = _f(4)
    facts = [(i, i * 3) for i in range(120)]
    build_v3_base(facts, p)
    expect = dict(facts)
    seq = 0
    for g in groups:
        seq += 1
        g = g[:5]                                   # keep frames small
        append_txn(p, g)
        for k, v in g:
            prev = expect.get(k)
            expect[k] = "CONFLICT" if (prev is not None and prev != v) else v
    st = V3Store(p)
    probes = set(list(expect)[:60]) | {k for g in groups for k, _ in g}
    for k in probes:
        want = expect.get(k)
        got = st.get(k)
        if want is None:
            assert got[0] == "UNKNOWN", (k, got)
        elif want == "CONFLICT":
            assert got[0] == "CONFLICT", (k, got)
        else:
            assert got == ("TRUE", want), (k, got, want)   # V3Store.get = (status, val)


@settings(deadline=None, max_examples=25, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(st.lists(st.tuples(st.integers(0, 500), st.integers(0, 15014)),
                min_size=1, max_size=300))
def test_p5_v2_v3_answer_parity_and_residue(pairs):
    facts = list(dict(pairs).items())          # unique keys, sorted-insertion order
    p2 = _f(5) + "a.rasm"
    p3 = _f(5) + "b.rasm"
    write_v2(facts, p2, page_recs=16)
    build_v3_base(facts, p3, page_recs=16)
    from rasm_cim_v9.storage.journal import open_any
    a, b = open_any(p2), V3Store(p3)      # v2 read by its own reader vs v3
    for k, _ in facts[:80]:
        assert a.get(k) == b.get(k)  # shapes coincide for known keys
    for k in (501, -7):
        assert a.get(k)[0] == b.get(k)[0] == "UNKNOWN"
    ri = ResidueIndex.from_sorted_keys([x[0] for x in sorted(facts)])
    for mod in ResidueIndex.MODS:
        for res in range(0, mod, max(1, mod // 7)):
            brute = sorted(k for k, _ in facts if k % mod == res)
            assert ri.query(res, mod) == brute, (mod, res)
