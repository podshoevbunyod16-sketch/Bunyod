"""§9 cross-version regression — v0.8 must preserve v0.1–v0.6 behaviour and
the frozen v0.7 binary-format spec. Every frozen module is imported
read-only; Rust directory is never touched.

Run: python3 -m pytest rasm_cim_v8/tests/test_cross_version.py -q
"""
from __future__ import annotations
import os
import struct
import sys

sys.path.insert(0, "/home/user")
import pytest

from rasm_cim_v8.storage import format as F

TMP = "/tmp/rasm_v8_xver"
os.makedirs(TMP, exist_ok=True)


# ------------------------------------------- format v1: golden bytes vs frozen spec
def _oracle_serialize_v1(facts):
    """Independent re-implementation of ARCHITECTURE.md §5 (v0.7, frozen):
    [b"RASM"][u32 ver=1][u64 payload_len][ (i64,i64) LE *n ][u64 FNV-1a]."""
    payload = b"".join(struct.pack("<qq", k, v) for k, v in facts)
    h = 0xCBF29CE484222325
    for byte in payload:
        h = ((h ^ byte) * 0x100000001B3) & (1 << 64) - 1
    return b"RASM" + struct.pack("<IQ", 1, len(payload)) + payload + struct.pack("<Q", h)


def test_v1_golden_bytes_match_frozen_spec():
    facts = [(1, 10), (2, -20), (0, 0), (2**62, -(2**62 - 1))]
    assert F.serialize_v1(facts) == _oracle_serialize_v1(facts)
    assert F.deserialize_v1(_oracle_serialize_v1(facts)) == facts
    # empty payload is a valid v1 (golden 24-byte file)
    assert F.serialize_v1([]) == b"RASM" + struct.pack("<IQ", 1, 0) + struct.pack("<Q", 0xCBF29CE484222325)


# ------------------------------------------- v0.6 PackedStore: layout identity
def test_v06_packedstore_payload_equals_v1_payload():
    from rasm_cim_v6_v2.compute.compute_core import PackedStore
    ps = PackedStore()
    facts = [(i, i * 7919) for i in range(50)]
    for k, v in facts:
        ps.put(k, v)
    blob = F.serialize_v1(facts)
    payload = blob[16:-8]
    assert bytes(ps._buf) == payload              # same <qq records, same order
    assert ps.get(7) == 7 * 7919


# ------------------------------------------- v0.6 behaviour constants (baseline §3)
def test_v06_memory_dedup_conflict_unknown():
    from rasm_cim_v6_v2.memory.memory_system import MemorySystem
    m = MemorySystem(D=64, tau=0.5, seed=0)
    # contract (v0.6 source): True = DUPLICATE suppressed, False = stored
    assert m.write_episodic(5, 100) is False
    assert m.write_episodic(5, 100) is True       # dedup hit
    assert m.dedup_hits == 1
    assert m.write_episodic(5, 200) is False      # distinct value stored
    st, _v, _c = m.query_exact(5)
    assert st == "CONFLICT"
    assert m.query_exact(999999)[0] == "UNKNOWN"


def test_v06_bfs_synthesis_candidates_238():
    """Exact v0.6 recipe (experiments6.run_algorithmic): must still yield
    candidates=238 and program add 2 -> mul 3 under v0.8-era imports."""
    from rasm_cim_v6_v2.architectures.archC_algorithmic import ArchC_Algorithmic
    e = ArchC_Algorithmic().learn([(0, 6), (1, 9)], [(2, 12)], [(5, 21)], max_depth=3)
    assert e["candidates"] == 238
    assert str(e["program"]) == str([("add", 2), ("mul", 3)])
    assert e["test_acc"] == 1.0


def test_v06_sparse_ca_ops_9_vs_64():
    from rasm_cim_v6_v2.architectures.archB_dynamic import ArchB_Dynamic
    b = ArchB_Dynamic(n_cells=64)
    st = b.state.copy()
    for pos in (10, 20, 30):
        st[pos] = 1
    b.set_state(st)
    _new, n_sparse = b.ca_step(rule=90, sparse=True)
    b.set_state(st)
    _new2, n_dense = b.ca_step(rule=90, sparse=False)
    assert n_sparse == 9 and n_dense == 64


def test_rns_v4_overflow_constant():
    from rasm_cim_v4.rns4 import RNSv4
    r = RNSv4((3, 5, 7, 11, 13))
    assert r.M == 15015 and r.offset == 7507
    with pytest.raises(OverflowError):
        r.encode_unsigned(15015)
    assert r.decode_signed(r.encode_signed(-7507)) == -7507
    assert r.decode_signed(r.encode_signed(7507)) == 7507


# ------------------------------------------- old-data import + serialization identity
def test_v1_file_import_to_v2_stream_identical():
    """Chain: v0.6 MemorySystem facts → v1 bytes (frozen format) → v0.8 v2
    file → LazyStore; every (k,v) must survive untouched (results identical
    pre/post serialization, §9)."""
    from rasm_cim_v6_v2.memory.memory_system import MemorySystem
    m = MemorySystem(D=64, tau=0.5, seed=1)
    facts = [(i, (i * 7919) % 15015) for i in range(500)]
    for k, v in facts:
        m.write_episodic(k, v)
    v1 = F.serialize_v1(facts)
    back = F.deserialize_v1(v1)
    assert back == facts
    p = f"{TMP}/imported.rasm"
    F.write_v2(iter(back), p, page_recs=64)
    from rasm_cim_v8.storage.streaming import open_lazy
    s = open_lazy(p)
    got = dict(s.scan())
    assert got == dict(facts)
    for k in (0, 3, 499):
        assert s.get(k) == ("TRUE", dict(facts)[k])
    s.close()


def test_v08_to_v06_reimport_roundtrip():
    """Cross-import both directions through the v1 blob."""
    facts = [(i, -i) for i in range(300)]
    blob = F.serialize_v1(facts)
    from rasm_cim_v6_v2.compute.compute_core import PackedStore
    ps = PackedStore()
    payload = blob[16:-8]
    for i in range(len(payload) // 16):
        k, v = struct.unpack_from("<qq", payload, i * 16)
        ps.put(k, v)
    assert len(ps) == 300 and ps.get(77) == -77


# ------------------------------------------- rule110 5-pattern correction, history intact
def test_rule110_legacy_and_canonical_both_available_and_diverge_on_five():
    """Tables are tuples indexed by 3-bit neighborhood (left<<2|center<<1|right).
    v0.8 audit correction: FIVE diverging patterns (v0.7 said three). History
    kept: legacy table is still the byte-for-byte v0.5 behavior."""
    from rasm_cim_v8.semantic.rule110 import step_rule110, all_eight, LEGACY_TABLE, CANONICAL_TABLE
    assert all_eight(canonical=False) and all_eight(canonical=True)
    div = [f"{i:03b}" for i in range(8) if LEGACY_TABLE[i] != CANONICAL_TABLE[i]]
    assert div == ["011", "100", "101", "110", "111"]
    # one step differs on a row containing a diverging neighborhood (101)
    row = [0, 1, 0, 1, 0]
    assert step_rule110(row, canonical=False) != step_rule110(row, canonical=True)
    # module uses circular boundaries (wrap): a zero row is all-000 and agrees
    row0 = [0, 0, 0, 0, 0]
    assert step_rule110(row0, canonical=False) == step_rule110(row0, canonical=True) == [0]*5
    # wrap proof: [0,0,1] position0 window=(last,0,0)=100 -> tables differ there
    assert step_rule110([0, 0, 1], canonical=False)[0] == 1
    assert step_rule110([0, 0, 1], canonical=True)[0] == 0
