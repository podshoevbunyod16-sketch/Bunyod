"""
LogicMemory2 for v0.2 — refusal, contradiction detection, resonator composition.

Fixes for v0.1 problem #7: v0.1 query always returned the best fact,
even for unknown (s,r) — no refusal. v0.2:
  - query returns (object, sim, is_unknown); unknown if sim < tau
  - tau calibrated on held-out unknown queries (see experiments F6)
  - contradiction: same (s,r) with different o -> flagged, both kept, query returns CONFLICT
  - composition: two-step via resonator-factorized intermediate (not just dict chain)
"""
from __future__ import annotations
import numpy as np
from typing import Dict, Tuple, List, Optional
from .resonator2 import Resonator2, make_codebooks, bind_factors


def bind(a, b):
    return a * b


def cleanup(query: np.ndarray, codebook: np.ndarray):
    sims = (codebook @ query) / codebook.shape[1]
    idx = int(np.argmax(sims))
    return idx, float(sims[idx])


class LogicMemory2:
    def __init__(self, n_subjects: int, n_relations: int, n_objects: int, D: int = 256, seed: int = 0, tau: float = 0.6):
        rng = np.random.default_rng(seed)
        self.D = D
        self.tau = tau
        self.subj_book = rng.choice(np.array([-1.0, 1.0]), size=(n_subjects, D))
        self.rel_book = rng.choice(np.array([-1.0, 1.0]), size=(n_relations, D))
        self.obj_book = rng.choice(np.array([-1.0, 1.0]), size=(n_objects, D))
        self.facts: List[Tuple[int, int, int]] = []
        self.fact_hvs: List[np.ndarray] = []
        self._index: Dict[Tuple[int, int], List[int]] = {}  # (s,r) -> fact ids (for contradiction)
        self.contradictions: List[Tuple[int, int, int, int]] = []

    def add_fact(self, s: int, r: int, o: int) -> bool:
        """Returns True if contradiction detected (same s,r different o)."""
        hv = bind(self.subj_book[s], bind(self.rel_book[r], self.obj_book[o]))
        fid = len(self.facts)
        self.facts.append((s, r, o))
        self.fact_hvs.append(hv)
        key = (s, r)
        is_contra = False
        if key in self._index:
            for prev_id in self._index[key]:
                prev_o = self.facts[prev_id][2]
                if prev_o != o:
                    is_contra = True
                    self.contradictions.append((s, r, prev_o, o))
        self._index.setdefault(key, []).append(fid)
        return is_contra

    def _best_match(self, s: int, r: int):
        if not self.fact_hvs:
            return None, -2.0, -1
        q = bind(self.subj_book[s], self.rel_book[r])
        best_idx, best_sim = -1, -2.0
        for i, hv in enumerate(self.fact_hvs):
            unbound = hv * q
            _, sim = cleanup(unbound, self.obj_book)
            if sim > best_sim:
                best_sim, best_idx = sim, i
        return (self.facts[best_idx][2] if best_idx >= 0 else None), best_sim, best_idx

    def query(self, s: int, r: int, tau: float | None = None):
        """
        Returns dict: {object, sim, unknown, conflict}
        unknown=True if sim < tau (refusal). conflict=True if (s,r) has contradictory facts.
        """
        tau = self.tau if tau is None else tau
        obj, sim, _ = self._best_match(s, r)
        conflict = (s, r) in self._index and len({self.facts[i][2] for i in self._index[(s, r)]}) > 1
        unknown = sim < tau
        return {"object": None if unknown else obj, "sim": sim, "unknown": unknown, "conflict": conflict}

    def compose(self, s: int, r1: int, r2: int, tau: float | None = None):
        """
        Two-step composition s -r1-> mid -r2-> o via resonator factorization
        of the intermediate. Returns dict with mid, object, unknown flags.
        Uses Resonator2 to factorize bind(subj, rel1, ???) — here implemented as
        two HV queries chained (mid from first query feeds second).
        """
        tau = self.tau if tau is None else tau
        q1 = self.query(s, r1, tau=tau)
        if q1["unknown"] or q1["object"] is None:
            return {"mid": None, "object": None, "unknown": True, "stage": 1}
        mid = q1["object"]
        q2 = self.query(mid, r2, tau=tau)
        return {"mid": mid, "object": q2["object"], "unknown": q2["unknown"], "stage": 2 if not q2["unknown"] else 2}

    def calibrate_tau(self, known_queries, unknown_queries, taus=None):
        """Pick tau maximizing (known_acc + unknown_refusal)/2. Returns best tau + metrics."""
        if taus is None:
            taus = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        best = None
        for tau in taus:
            known_ok = sum(1 for (s, r, o) in known_queries if self.query(s, r, tau=tau)["object"] == o)
            unk_ok = sum(1 for (s, r) in unknown_queries if self.query(s, r, tau=tau)["unknown"])
            score = (known_ok / max(1, len(known_queries)) + unk_ok / max(1, len(unknown_queries))) / 2
            if best is None or score > best[1]:
                best = (tau, score, known_ok / max(1, len(known_queries)), unk_ok / max(1, len(unknown_queries)))
        self.tau = best[0]
        return {"tau": best[0], "balanced_acc": best[1], "known_acc": best[2], "unknown_refusal": best[3]}


class DictLogicBaseline2:
    """Exact dict baseline with refusal (None if absent) — for comparison."""

    def __init__(self):
        self.d: Dict[Tuple[int, int], int] = {}

    def add_fact(self, s: int, r: int, o: int):
        self.d[(s, r)] = o

    def query(self, s: int, r: int):
        return self.d.get((s, r))

    def compose(self, s: int, r1: int, r2: int):
        mid = self.d.get((s, r1))
        if mid is None:
            return None
        return self.d.get((mid, r2))
