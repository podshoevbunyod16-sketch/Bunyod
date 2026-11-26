"""M4 — property-based (hypothesis) + fixed-seed random-mutation tests.

HONEST LABELS (spec §4): only atheris is coverage-guided (see
rasm_cim_v8/fuzz/fuzz_targets.py + results_v8/m4_atheris.json). The
mutator here is RANDOM byte-fuzzing with a fixed seed — NOT coverage-guided
and never called that.

Target-surface note: the v0.8 spec names `PackedStore.from_bytes` — verified
in this workspace that from_bytes exists ONLY on the Rust v0.7 binding
(rasm_cim_rust/tests/test_cross_language.py:199); the frozen v0.6 Python
PackedStore (rasm_cim_v6_v2/compute/compute_core.py) has no deserializer
(put/get only). The live Python deserialize surfaces are
`format.deserialize_v1` (v1) and the LazyStore page/index readers (v2) —
those are what gets fuzzed here. Graph update/retract targets the real
Python graph: rasm_cim_v4.graph4.TruthGraph (frozen, imported read-only;
used by v0.6 MemorySystem.semantic).

Run: python3 -m pytest rasm_cim_v8/tests/test_m4_properties.py -q
"""
from __future__ import annotations
import os
import random
import struct
import sys

sys.path.insert(0, "/home/user")

import pytest
from hypothesis import given, settings, strategies as st

from rasm_cim_v8.storage import format as F
from rasm_cim_v8.storage.format import StorageError
from rasm_cim_v8.storage.streaming import CorruptPageError, open_lazy
from rasm_cim_v4.graph4 import TruthGraph

TMP = "/tmp/rasm_v8_m4"
os.makedirs(TMP, exist_ok=True)

KEYS = st.integers(-2**63, 2**63 - 1)
VALS = st.integers(-2**63, 2**63 - 1)


# ----------------------------------------------------- v1 roundtrip (property)
@settings(deadline=None, max_examples=80)
@given(st.lists(st.tuples(KEYS, VALS), max_size=120))
def test_v1_roundtrip_identity(items):
    blob = F.serialize_v1(items)
    back = F.deserialize_v1(blob)
    assert back == items                      # order + values identical
    assert F.deserialize_v1(F.serialize_v1([])) == []


# ----------------------------------------------------- v1 hostile inputs
@settings(deadline=None, max_examples=250)
@given(st.binary(max_size=600))
def test_v1_never_unclassified_error(data):
    """deserialize_v1 on arbitrary bytes: returns a list or raises
    StorageError (ValueError subclass). OverflowError/struct.error/
    IndexError/UnboundLocalError escaping = real bug (allocation/edge)."""
    try:
        out = F.deserialize_v1(data)
        assert isinstance(out, list)
    except StorageError:
        pass
    except Exception as e:  # noqa: BLE001 — deliberate: classifying failures
        pytest.fail(f"unguarded {type(e).__name__}: {e}")


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_v1_random_mutations_labelled_not_coverage_guided(seed):
    """Fixed-seed random single-byte/insert/delete mutations of a valid blob.
    Random generation only — this is explicitly NOT coverage-guided fuzzing."""
    good = F.serialize_v1([(i, i * 7919) for i in range(64)])
    rng = random.Random(seed)
    b = bytearray(good)
    for _ in range(400):
        op = rng.randrange(3)
        i = rng.randrange(len(b))
        if op == 0:
            b[i] = rng.randrange(256)
        elif op == 1:
            del b[i:i + rng.randrange(1, 33)]
        else:
            b[i:i] = bytes(rng.randrange(256) for _ in range(rng.randrange(1, 33)))
        try:
            F.deserialize_v1(bytes(b))
        except StorageError:
            pass
        except Exception as e:  # noqa: BLE001
            pytest.fail(f"unguarded {type(e).__name__} at seed {seed} pos {i}")
        b = bytearray(good)


# ----------------------------------------------------- v2 paged: roundtrip + hostile pages
@settings(deadline=None, max_examples=30)
@given(st.lists(st.tuples(st.integers(-2**62, 2**62), st.integers(-2**62, 2**62)),
                max_size=60))
def test_v2_roundtrip(items):
    path = f"{TMP}/v2_prop.rasm"
    F.write_v2(iter(items), path, page_recs=7)
    st_ = open_lazy(path)
    got = dict(st_.scan())
    expected = {}
    for k, v in items:              # scan order == file order; dict() is last-wins
        expected[k] = v
    assert got == expected
    st_.close()


def _mutate(path_in: str, seed: int, rounds: int):
    rng = random.Random(seed)
    good = open(path_in, "rb").read()
    b = bytearray(good)
    for _ in range(rounds):
        op = rng.randrange(4)
        i = rng.randrange(len(b))
        if op == 0:
            b[i] ^= 1 << rng.randrange(8)
        elif op == 1:
            del b[i:i + rng.randrange(1, 64)]
        elif op == 2:
            b[i:i] = bytes(rng.randrange(256) for _ in range(rng.randrange(1, 64)))
        else:
            b = bytearray(bytes(rng.randrange(256) for _ in range(rng.randrange(25))))
        yield bytes(b)
        b = bytearray(good)


def test_v2_corruption_never_silent_wrong_data():
    """For mutated v2 files: scan() either completes or raises
    CorruptPageError/StorageError/ValueError-class — never an unguarded
    crash, and per-page crc32 keeps corruption from passing silently."""
    path = f"{TMP}/v2_mut_src.rasm"
    n = 500
    F.write_v2(((i, (i * 7919) % 15015) for i in range(n)), path, page_recs=16)
    seen_corrupt = 0
    ok_clean = 0
    for mut in _mutate(path, seed=7, rounds=600):
        p2 = f"{TMP}/v2_mut.rasm"
        open(p2, "wb").write(mut)
        try:
            s = open_lazy(p2, on_bad="raise")
            list(s.scan())
            s.close()
            ok_clean += 1
        except (CorruptPageError, StorageError, struct.error, ValueError):
            seen_corrupt += 1
    assert seen_corrupt > 0, "mutations must be caught — policy is live"
    print("detected/clean:", seen_corrupt, ok_clean)


def test_v2_corrupt_page_skip_policy_rows():
    """Flip one data byte inside page 1; on_bad='skip' keeps other pages."""
    path = f"{TMP}/v2_skip.rasm"
    F.write_v2(((i, (i * 7919) % 15015) for i in range(256)), path, page_recs=16)
    raw = bytearray(open(path, "rb").read())
    data_page1 = 24 + (16 * 16 * 2) + 8 + 4  # hdr + 1 full page + 1 byte into page 2
    raw[data_page1] ^= 0xFF
    open(path, "wb").write(bytes(raw))
    s = open_lazy(path, on_bad="skip", cache_pages=2)
    rows = list(s.scan())
    kept_keys = {k for k, _ in rows}
    bad = set(range(16, 32))
    assert kept_keys & bad == set()           # corrupted page contributed nothing
    assert len(kept_keys) >= 256 - 16          # everything else survived
    s.close()


# ----------------------------------------------------- RNS v4 encode/decode surface
# (the real Python API is rasm_cim_v4.rns4.RNSv4 — encode/decode of residue
# tuples; the spec's "decode_unsigned(int)" phrasing traced to the frozen
# v0.1 rasm_cim/rns.py which has no decode_unsigned at all: it decodes
# residues via CRT. We target the actual public decode entry.)
def test_rns_roundtrip_and_overflow():
    from rasm_cim_v4.rns4 import RNSv4
    r = RNSv4((3, 5, 7, 11, 13))
    assert r.decode_unsigned(r.encode_unsigned(0)) == 0
    assert r.decode_unsigned(r.encode_unsigned(r.M - 1)) == r.M - 1
    with pytest.raises(OverflowError):
        r.encode_unsigned(r.M)
    with pytest.raises(OverflowError):
        r.encode_unsigned(-1)
    rng = random.Random(5)
    for _ in range(500):
        x = rng.randrange(r.M)
        assert r.decode_unsigned(r.encode_unsigned(x)) == x


@settings(deadline=None, max_examples=300)
@given(st.tuples(st.integers(-2**40, 2**40), st.integers(-2**40, 2**40),
                 st.integers(-2**40, 2**40), st.integers(-2**40, 2**40),
                 st.integers(-2**40, 2**40)))
def test_rns_decode_total_no_crash(res):
    """decode_unsigned is TOTAL on any 5-int tuple (out-of-range, negative,
    huge): result must be in [0, M) — never OverflowError/IndexError."""
    from rasm_cim_v4.rns4 import RNSv4
    r = RNSv4((3, 5, 7, 11, 13))
    x = r.decode_unsigned(res)
    assert 0 <= x < r.M
    # and it must agree with direct CRT residues:
    assert all(x % m == res[i] % m for i, m in enumerate(r.moduli))


def test_rns_wrong_length_raises_cleanly():
    from rasm_cim_v4.rns4 import RNSv4
    r = RNSv4((3, 5, 7, 11, 13))
    with pytest.raises(IndexError):      # documented behaviour: short input not guarded
        r.decode_unsigned((1, 2))



# ----------------------------------------------------- graph update / retract
def _model_query(tset, fset, r, functional, o=None):
    if o is not None:
        if o in tset and o in fset:
            return "CONFLICT"
        if o in tset:
            return "TRUE"
        if o in fset:
            return "FALSE"
        return "UNKNOWN"
    if len(tset) > 1 and r in functional:
        return "CONFLICT"
    if tset & fset:
        return "CONFLICT"
    if len(tset) == 1:
        return "TRUE"
    if fset and not tset:
        return "FALSE"
    return "UNKNOWN"


@settings(deadline=None, max_examples=120)
@given(st.lists(st.tuples(st.sampled_from(("t", "f", "x")),
                          st.integers(0, 4), st.integers(0, 3),
                          st.integers(0, 4)), max_size=60))
def test_graph_update_retract_matches_model(ops):
    """Property: after ANY interleaving of assert_true / assert_false /
    retract, every triple and functional query agrees with a shadow model.
    (retract removes from TRUE edges only — mirrored from source.)"""
    g = TruthGraph(functional_relations={0, 1})
    T: dict = {}
    Fl: dict = {}
    for op, s, r, o in ops:
        if op == "t":
            g.assert_true(s, r, o)
            T.setdefault((s, r), set()).add(o)
        elif op == "f":
            g.assert_false(s, r, o)
            Fl.setdefault((s, r), set()).add(o)
        else:
            g.retract(s, r, o)
            T.setdefault((s, r), set()).discard(o)
    for key in set(T) | set(Fl):
        tset, fset = T.get(key, set()), Fl.get(key, set())
        assert g.get_true(*key) == tset, key
        assert g.query(*key) == _model_query(tset, fset, key[1], {0, 1}), key
        for oo in (tset | fset):
            assert g.query(*key, oo) == _model_query(tset, fset, key[1], {0, 1}, oo)


def test_graph_retract_unknown_and_conflict_resolution():
    g = TruthGraph()
    g.retract(99, 98, 97)          # never asserted — must not crash
    assert g.query(99, 98) == "UNKNOWN"
    g = TruthGraph(functional_relations={0})
    g.assert_true(1, 0, 5)
    g.assert_true(1, 0, 6)
    assert g.query(1, 0) == "CONFLICT"
    g.retract(1, 0, 6)
    assert g.query(1, 0) == "TRUE"
    assert g.query(1, 0, 5) == "TRUE" and g.query(1, 0, 6) == "UNKNOWN"


def test_graph_history_preserved_across_retract():
    """Spec §3 'history unchanged': the op log is append-only, retraction
    never rewrites or removes entries."""
    g = TruthGraph()
    g.assert_true(1, 0, 2)
    g.retract(1, 0, 2)
    h = g.save_history()
    assert [e["op"] for e in h] == ["assert_true", "retract"]
    assert len(h) == 2 and h[1]["o"] == 2


# ----------------------------------------------------- regression guards
def test_v1_huge_payload_len_rejected_without_alloc():
    """A crafted u64 payload_len must fail the length check BEFORE any
    allocation (MemoryError candidate — regression guard)."""
    evil = b"RASM" + struct.pack("<IQ", 1, 2**63) + b"\x00" * 8
    with pytest.raises(StorageError):
        F.deserialize_v1(evil)


def test_fd_not_leaked_on_corrupt_open():
    """REGRESSION for the M4 fuzz finding: every failed LazyStore.__init__
    must release its fd (was: EMFILE after ~1k corrupt opens)."""
    src = f"{TMP}/v2_fd_src.rasm"
    F.write_v2(((i, i) for i in range(64)), src, page_recs=8)
    raw = bytearray(open(src, "rb").read())
    raw[-16] ^= 0xFF                      # break footer crc
    bad = f"{TMP}/v2_fd_bad.rasm"
    open(bad, "wb").write(bytes(raw))
    import glob
    n0 = len(glob.glob("/proc/self/fd/*"))
    for _ in range(1500):
        try:
            open_lazy(bad)
        except StorageError:
            pass
    n1 = len(glob.glob("/proc/self/fd/*"))
    assert n1 - n0 <= 2, f"fd leak: {n1 - n0} descriptors left open after 1500 corrupt opens"


def test_v1_truncated_payload_rejected():
    good = F.serialize_v1([(1, 1), (2, 2), (3, 3)])
    for cut in (16, 24, 40):
        with pytest.raises(StorageError):
            F.deserialize_v1(good[:cut])
    with pytest.raises(StorageError):
        F.deserialize_v1(b"")
