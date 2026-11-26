"""
TruthGraph for v0.4 — TRUE/FALSE/UNKNOWN/CONFLICT + history + retract.

Fix for v0.3: GraphMemory overwrote old value, no FALSE, no history.
v0.4 semantics:
  - assert_true(s,r,o): record true edge
  - assert_false(s,r,o): record false edge (explicit negation)
  - retract(s,r,o): remove true edge (history kept)
  - query(s,r): TRUE (one true, no false for it), FALSE (explicit false, no true),
    UNKNOWN (neither), CONFLICT (true for o1 and false for o1, or functional
    relation with two distinct true objects)
  - functional relations: declared set; second distinct true object -> CONFLICT
  - history: append-only log of all ops (for audit/replay)
"""
from __future__ import annotations
from typing import Dict, Set, Tuple, List
from collections import defaultdict, deque


class TruthGraph:
    def __init__(self, functional_relations: Set[int] | None = None):
        self.true_edges: Dict[Tuple[int,int], Set[int]] = defaultdict(set)
        self.false_edges: Dict[Tuple[int,int], Set[int]] = defaultdict(set)
        self.functional = set(functional_relations or set())
        self.history: List[dict] = []
        self._step = 0

    def _log(self, op: str, s: int, r: int, o: int):
        self._step += 1
        self.history.append({"step": self._step, "op": op, "s": s, "r": r, "o": o})

    def assert_true(self, s: int, r: int, o: int):
        self._log("assert_true", s, r, o)
        self.true_edges[(s,r)].add(o)

    def assert_false(self, s: int, r: int, o: int):
        self._log("assert_false", s, r, o)
        self.false_edges[(s,r)].add(o)

    def retract(self, s: int, r: int, o: int):
        self._log("retract", s, r, o)
        self.true_edges.get((s,r), set()).discard(o)

    def query(self, s: int, r: int, o: int | None = None) -> str:
        """
        If o given: status of specific triple.
        Else: status of (s,r) as functional query (returns TRUE if exactly one true and no conflict).
        Returns one of TRUE/FALSE/UNKNOWN/CONFLICT.
        """
        trues = self.true_edges.get((s,r), set())
        falses = self.false_edges.get((s,r), set())
        if o is not None:
            if o in trues and o in falses: return "CONFLICT"
            if o in trues: return "TRUE"
            if o in falses: return "FALSE"
            return "UNKNOWN"
        # functional-style query without o
        if len(trues) > 1 and r in self.functional:
            return "CONFLICT"
        if trues & falses:
            return "CONFLICT"
        if len(trues) == 1:
            return "TRUE"
        if falses and not trues:
            return "FALSE"
        return "UNKNOWN"

    def get_true(self, s: int, r: int) -> Set[int]:
        return set(self.true_edges.get((s,r), set()))

    def compose(self, s: int, r1: int, r2: int) -> str:
        """Two-step: s -r1-> mid -r2-> o. Returns TRUE if chain exists, else UNKNOWN/CONFLICT."""
        mids = self.get_true(s, r1)
        if not mids: return "UNKNOWN"
        results = set()
        for mid in mids:
            results.update(self.get_true(mid, r2))
        if not results: return "UNKNOWN"
        if len(results) > 1: return "CONFLICT"
        return "TRUE"

    def compose_value(self, s: int, r1: int, r2: int):
        mids = self.get_true(s, r1)
        if len(mids) != 1: return None
        mid = next(iter(mids))
        outs = self.get_true(mid, r2)
        return next(iter(outs)) if len(outs)==1 else None

    def reachable(self, s: int, max_depth: int = 3) -> Set[int]:
        adj: Dict[int, List[int]] = defaultdict(list)
        for (ss, _), objs in self.true_edges.items():
            for o in objs: adj[ss].append(o)
        seen = {s}; frontier = deque([(s,0)])
        while frontier:
            node, d = frontier.popleft()
            if d >= max_depth: continue
            for o in adj.get(node, []):
                if o not in seen:
                    seen.add(o); frontier.append((o, d+1))
        return seen

    def save_history(self) -> List[dict]:
        return list(self.history)
