"""v0.9 M6 — TruthGraph9: formally specified triple semantics.

STATE:      T, F  — sets of triples (s, r, o).
assert_true(s,r,o):   T := T ∪ {t}
assert_false(s,r,o):  F := F ∪ {t}
retract(s,r,o):       v0.9 (mode="symmetric"):  T := T \\ {t},  F := F \\ {t}
                      legacy (mode="legacy"):   T := T \\ {t}   (v0.4 behavior,
                      reproduced byte-for-byte via shadow tests)
query(s,r,o):   t ∈ T∩F → CONFLICT
                t ∈ T   → TRUE
                t ∈ F   → FALSE
                else    → UNKNOWN
functional query(s,r), r ∈ FUNCTIONAL:
                |T(s,r)| > 1            → CONFLICT
                T(s,r) ∩ F(s,r) ≠ ∅    → CONFLICT
                |T(s,r)| == 1          → TRUE
                F(s,r) ≠ ∅ = T(s,r)∅   → FALSE
                else UNKNOWN
history: append-only op log; retract NEVER rewrites/removes entries (all
modes).

WHY symmetric retract is the v0.9 default (documented, not decreed):
under the legacy rule, `assert_true(t); assert_false(t); retract(t)` leaves
t ∈ F only → the triple reads FALSE forever — a retract that silently
changes the surviving polarity is a semantic trap for update-without-
retraining flows. The symmetric rule makes retract = "forget this assertion,
any polarity": after retract(t) the state is provably t ∉ T ∪ F (test
`test_retract_clears_both_polarities`), and CONFLICT is always resolvable by
retracting the conflicting triples (legacy can strand a CONFLICT:
assert_true(t); assert_false(t); retract(t) → still CONFLICT? no — T loses t,
F keeps t → FALSE; but the pair-level conflict with functional r can strand:
see test_functional_conflict_resolution_matrix).

Frozen v0.4 TruthGraph is NOT modified; `mode="legacy"` here reproduces its
observable behavior and tests assert the equivalence — this is the A/B.
"""
from __future__ import annotations
from collections import defaultdict
from typing import Dict, Set, Tuple


class TruthGraph9:
    def __init__(self, functional_relations=None, mode: str = "symmetric"):
        if mode not in ("symmetric", "legacy"):
            raise ValueError(f"unknown retract mode {mode!r}")
        self.mode = mode
        self.true_edges: Dict[Tuple[int, int], Set[int]] = defaultdict(set)
        self.false_edges: Dict[Tuple[int, int], Set[int]] = defaultdict(set)
        self.functional = set(functional_relations or set())
        self.history: list[dict] = []
        self._step = 0

    def _log(self, op, s, r, o):
        self._step += 1
        self.history.append({"step": self._step, "op": op, "s": s, "r": r, "o": o})

    def assert_true(self, s, r, o):
        self._log("assert_true", s, r, o)
        self.true_edges[(s, r)].add(o)

    def assert_false(self, s, r, o):
        self._log("assert_false", s, r, o)
        self.false_edges[(s, r)].add(o)

    def retract(self, s, r, o):
        self._log("retract", s, r, o)
        self.true_edges.get((s, r), set()).discard(o)
        if self.mode == "symmetric":
            self.false_edges.get((s, r), set()).discard(o)

    # -- reads: identical formulas for both modes (only state differs) ---
    def query(self, s, r, o=None) -> str:
        trues = self.true_edges.get((s, r), set())
        falses = self.false_edges.get((s, r), set())
        if o is not None:
            if o in trues and o in falses:
                return "CONFLICT"
            if o in trues:
                return "TRUE"
            if o in falses:
                return "FALSE"
            return "UNKNOWN"
        if len(trues) > 1 and r in self.functional:
            return "CONFLICT"
        if trues & falses:
            return "CONFLICT"
        if len(trues) == 1:
            return "TRUE"
        if falses and not trues:
            return "FALSE"
        return "UNKNOWN"

    def get_true(self, s, r) -> Set[int]:
        return set(self.true_edges.get((s, r), set()))

    def save_history(self):
        return list(self.history)
