"""v0.9 M6 — TruthGraph semantics A/B (frozen v0.4 vs v0.9 modes).

The exhaustive check is over ALL op-sequences of length ≤ 5 on a tiny triple
space (3 ops × 2 triples ⇒ 6^5 ≈ 7.7k sequences) against shadow models for
each semantics; plus the stranded-state exhibits. Frozen v0.4 TruthGraph is
instantiated directly to prove `mode="legacy"` is behavior-identical.

Run: python3 -m pytest tests_v9/test_m6_graph.py -q
"""
from __future__ import annotations
import itertools
import sys

sys.path.insert(0, "/home/user")
import pytest

from rasm_cim_v4.graph4 import TruthGraph as FrozenGraph4      # read-only
from rasm_cim_v9.graph import TruthGraph9

TRIPLES = [(0, 0, 0), (0, 0, 1)]           # (s,r,o): s=0, functional r=0, o∈{0,1}
OPS = ("t", "f", "x")                       # true / false / retract


def _apply(g, op, s, r, o):
    if op == "t":
        g.assert_true(s, r, o)
    elif op == "f":
        g.assert_false(s, r, o)
    else:
        g.retract(s, r, o)


def _shadow(triples_state, functional, op, idx, mode):
    T, F = triples_state
    s, r, o = TRIPLES[idx]
    if op == "t":
        T.add((s, r, o))
    elif op == "f":
        F.add((s, r, o))
    else:
        T.discard((s, r, o))
        if mode == "symmetric":
            F.discard((s, r, o))
    return T, F


def _shadow_query(T, F, s, r, functional, o=None):
    trues = {oo for (ss, rr, oo) in T if (ss, rr) == (s, r)}
    falses = {oo for (ss, rr, oo) in F if (ss, rr) == (s, r)}
    if o is not None:
        if o in trues and o in falses:
            return "CONFLICT"
        if o in trues:
            return "TRUE"
        if o in falses:
            return "FALSE"
        return "UNKNOWN"
    if len(trues) > 1 and r in functional:
        return "CONFLICT"
    if trues & falses:
        return "CONFLICT"
    if len(trues) == 1:
        return "TRUE"
    if falses and not trues:
        return "FALSE"
    return "UNKNOWN"


@pytest.mark.parametrize("mode", ("symmetric", "legacy"))
@pytest.mark.parametrize("L", (3, 4, 5))
def test_exhaustive_sequences_match_shadow(mode, L):
    space = list(itertools.product(*([(op, idx) for op in OPS for idx in range(len(TRIPLES))]
                                     for _ in range(L)))) if L <= 4 else None
    seqs = space or list(itertools.islice(
        itertools.product([(op, idx) for op in OPS for idx in range(len(TRIPLES))], repeat=L), 0, None))
    for seq in seqs:
        g = TruthGraph9(functional_relations={0}, mode=mode)
        T, F = set(), set()
        for op, idx in seq:
            _apply(g, op, *TRIPLES[idx])
            T, F = _shadow((T, F), {0}, op, idx, mode)
        for s, r, o in TRIPLES:
            assert g.query(s, r, o) == _shadow_query(T, F, s, r, {0}, o), (seq, (s, r, o))
        assert g.query(s, r) == _shadow_query(T, F, s, r, {0}), (seq, (s, r))


def test_legacy_mode_is_identical_to_frozen_v04():
    """A/B requirement: the v0.9 legacy mode must not change v0.4 semantics
    by even one query outcome over all length-4 sequences."""
    for seq in itertools.product([(op, idx) for op in OPS for idx in range(len(TRIPLES))], repeat=4):
        a = FrozenGraph4(functional_relations={0})
        b = TruthGraph9(functional_relations={0}, mode="legacy")
        for op, idx in seq:
            _apply(a, op, *TRIPLES[idx])
            _apply(b, op, *TRIPLES[idx])
        for trip in TRIPLES + [(0, 0), (1, 1)]:
            if len(trip) == 3:
                assert a.query(*trip) == b.query(*trip), (seq, trip)
            else:
                assert a.query(*trip) == b.query(*trip), (seq, trip)
        assert a.get_true(0, 0) == b.get_true(0, 0)


def test_retract_clears_both_polarities_symmetric():
    g = TruthGraph9(mode="symmetric")
    g.assert_true(1, 0, 2)
    g.assert_false(1, 0, 2)
    assert g.query(1, 0, 2) == "CONFLICT"
    g.retract(1, 0, 2)
    assert g.query(1, 0, 2) == "UNKNOWN"          # provable invariant
    assert g.query(1, 0) == "UNKNOWN"


def test_legacy_stranded_state_exhibit():
    """The documented trap that motivated v0.9 default (kept as A/B evidence,
    not as a retro-edit of history): under legacy retract, false survives."""
    for mode, expected in (("legacy", "FALSE"), ("symmetric", "UNKNOWN")):
        g = TruthGraph9(mode=mode)
        g.assert_true(1, 0, 2)
        g.assert_false(1, 0, 2)
        g.retract(1, 0, 2)
        assert g.query(1, 0, 2) == expected


def test_functional_conflict_resolution_matrix():
    """Symmetric retract must always resolve functional CONFLICTs by
    retracting the involved triples; legacy can strand |T|==1 + F residue."""
    g = TruthGraph9(functional_relations={0}, mode="symmetric")
    g.assert_true(1, 0, 2)
    g.assert_true(1, 0, 3)
    assert g.query(1, 0) == "CONFLICT"
    g.retract(1, 0, 3)
    assert g.query(1, 0) == "TRUE" and g.query(1, 0, 2) == "TRUE"
    gl = TruthGraph9(functional_relations={0}, mode="legacy")
    gl.assert_true(1, 0, 2)
    gl.assert_false(1, 0, 2)
    gl.assert_true(1, 0, 3)
    gl.retract(1, 0, 3)
    gl.retract(1, 0, 2)
    assert gl.query(1, 0) == "FALSE"      # stranded FALSE residue under legacy
    gs = TruthGraph9(functional_relations={0}, mode="symmetric")
    gs.assert_true(1, 0, 2)
    gs.assert_false(1, 0, 2)
    gs.assert_true(1, 0, 3)
    gs.retract(1, 0, 3)
    gs.retract(1, 0, 2)
    assert gs.query(1, 0) == "UNKNOWN"    # clean slate under symmetric


def test_history_append_only_in_both_modes():
    for mode in ("symmetric", "legacy"):
        g = TruthGraph9(mode=mode)
        g.assert_true(1, 0, 2)
        g.retract(1, 0, 2)
        g.retract(1, 0, 2)               # no-op retraction still logged
        h = g.save_history()
        assert [e["op"] for e in h] == ["assert_true", "retract", "retract"]


def test_retract_unknown_is_noop_and_safe():
    for mode in ("symmetric", "legacy"):
        g = TruthGraph9(mode=mode)
        g.retract(9, 9, 9)
        assert g.query(9, 9) == "UNKNOWN" and len(g.save_history()) == 1
