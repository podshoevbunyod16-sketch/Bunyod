"""
CAM2 for v0.2 — associative memory with inverted index + refusal threshold.

Fixes for v0.1 problem #2:
- v0.1 only had full-scan vectorized search (simulated CAM).
- v0.2 adds an inverted index per residue position for PARTIAL search
  (sublinear in N for selective queries), explicit operation counting,
  and a similarity threshold tau for UNKNOWN refusal.
- This is still a SOFTWARE SIMULATION. Hardware CAM speed/energy is
  modelled theoretically in docs, never claimed as measured.
"""
from __future__ import annotations
import numpy as np
from typing import Dict, Any, Tuple, List, Set
from collections import defaultdict


class CAM2:
    """
    Associative memory:
      - exact index: dict residue-tuple -> value (HashMap baseline inside)
      - inverted index: (pos, residue) -> set(ids) for partial search
      - HV store: bipolar keys for noisy search + Hopfield retrieval
    Inputs: residues (tuple/array), HV (D,), value
    Outputs: (value, similarity, ops_used) or UNKNOWN if sim < tau
    """

    def __init__(self, D: int = 256, tau: float = 0.5):
        self.D = D
        self.tau = tau
        self.values: List[Any] = []
        self.residues: List[Tuple[int, ...]] = []
        self.keys_hv: List[np.ndarray] = []
        self._exact: Dict[Tuple[int, ...], int] = {}  # residues -> id
        self._inv: Dict[Tuple[int, int], Set[int]] = defaultdict(set)
        self.ops = 0  # comparison counter

    def write(self, residues, hv: np.ndarray, value: Any) -> int:
        rid = len(self.values)
        res = tuple(int(r) for r in residues)
        self.values.append(value)
        self.residues.append(res)
        self.keys_hv.append(np.asarray(hv, dtype=float).copy())
        self._exact[res] = rid
        for pos, r in enumerate(res):
            self._inv[(pos, r)].add(rid)
        return rid

    def reset_ops(self):
        self.ops = 0

    # --- exact (HashMap) ---
    def search_exact(self, residues):
        self.ops += 1
        rid = self._exact.get(tuple(int(r) for r in residues))
        return (self.values[rid], 1.0, self.ops) if rid is not None else (None, -2.0, self.ops)

    # --- partial residue search via inverted index ---
    def search_partial(self, partial: Dict[int, int]):
        """
        partial: {pos: residue}. Returns list of values matching ALL given residues.
        Ops = size of smallest posting list (candidates examined), not N.
        """
        if not partial:
            return list(self.values), self.ops
        # pick smallest posting list as candidate set
        lists = [self._inv.get((pos, r), set()) for pos, r in partial.items()]
        if any(len(s) == 0 for s in lists):
            self.ops += 1
            return [], self.ops
        cand = min(lists, key=len)
        self.ops += len(cand)
        out = []
        for rid in cand:
            res = self.residues[rid]
            if all(res[pos] == r for pos, r in partial.items()):
                out.append(self.values[rid])
        return out, self.ops

    def search_partial_fullscan(self, partial: Dict[int, int]):
        """Baseline: scan all N (for comparison). Ops = N."""
        self.ops += len(self.values)
        out = []
        for res, val in zip(self.residues, self.values):
            if all(res[pos] == r for pos, r in partial.items()):
                out.append(val)
        return out, self.ops

    # --- noisy HV search with refusal ---
    def search_noisy(self, query_hv: np.ndarray, tau: float | None = None):
        """
        Returns (value, sim, is_unknown). If best sim < tau -> UNKNOWN (None).
        Ops = N comparisons (full associative scan, as in hardware CAM).
        """
        if not self.keys_hv:
            return None, -2.0, True
        tau = self.tau if tau is None else tau
        K = np.stack(self.keys_hv, axis=0)
        sims = (K @ np.asarray(query_hv, dtype=float)) / self.D
        self.ops += len(self.values)
        idx = int(np.argmax(sims))
        sim = float(sims[idx])
        if sim < tau:
            return None, sim, True
        return self.values[idx], sim, False

    def search_noisy_no_refusal(self, query_hv: np.ndarray):
        """v0.1 behaviour: always return best (for comparison)."""
        if not self.keys_hv:
            return None, -2.0
        K = np.stack(self.keys_hv, axis=0)
        sims = (K @ np.asarray(query_hv, dtype=float)) / self.D
        self.ops += len(self.values)
        idx = int(np.argmax(sims))
        return self.values[idx], float(sims[idx])

    # --- Hopfield retrieval (softmax attention view) ---
    def hopfield_retrieve(self, query_hv: np.ndarray, beta: float = 10.0):
        if not self.keys_hv:
            return None, None
        K = np.stack(self.keys_hv, axis=0)
        scores = beta * (K @ np.asarray(query_hv, dtype=float)) / self.D
        scores = scores - np.max(scores)
        w = np.exp(scores)
        w = w / np.sum(w)
        self.ops += len(self.values)
        # retrieved value = weighted vote (for categorical values, return top-1)
        idx = int(np.argmax(w))
        return self.values[idx], w

    def __len__(self):
        return len(self.values)


def random_bipolar(rng: np.random.Generator, D: int) -> np.ndarray:
    return rng.choice(np.array([-1.0, 1.0]), size=D)


def flip_bits(hv: np.ndarray, frac: float, rng: np.random.Generator) -> np.ndarray:
    out = np.array(hv, copy=True)
    n = int(frac * len(out))
    idx = rng.choice(len(out), size=n, replace=False)
    out[idx] = -out[idx]
    return out
