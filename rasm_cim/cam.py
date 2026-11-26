"""
Associative memory (CAM) simulation for RASM-CIM prototype.

Hardware CAM compares query against ALL cells in parallel in one cycle.
In Python simulation we emulate this with vectorized numpy (all cells at once)
and compare against:
  - dict hash lookup (exact match baseline, O(1) average in software)
  - Python linear scan (O(N) loop baseline)

Honest result expected: software CAM simulation does NOT beat dict for exact
lookup; hardware speedup only appears in CAM hardware. This is documented
as a negative result for software-only evaluation.
"""
from __future__ import annotations
import numpy as np
from typing import Dict, Any, Tuple, List


class AssociativeMemory:
    def __init__(self, D: int = 512):
        self.D = D
        self.keys_hv: List[np.ndarray] = []   # bipolar hypervectors (+-1)
        self.values: List[Any] = []
        self._dict: Dict[Tuple[int, ...], Any] = {}  # exact residue-tuple index

    def write_hv(self, hv: np.ndarray, value: Any):
        self.keys_hv.append(np.asarray(hv).copy())
        self.values.append(value)

    def write_residue(self, residues: Tuple[int, ...], value: Any):
        self._dict[tuple(residues)] = value

    # --- exact lookup baselines ---
    def search_dict(self, residues: Tuple[int, ...]):
        return self._dict.get(tuple(residues))

    def search_linear_hv(self, query: np.ndarray):
        """Python loop over all cells — O(N*D) with Python overhead."""
        best_idx, best_sim = -1, -2.0
        q = np.asarray(query)
        for i, k in enumerate(self.keys_hv):
            s = float(np.mean(q * k))  # bipolar similarity
            if s > best_sim:
                best_sim, best_idx = s, i
        return (self.values[best_idx], best_sim) if best_idx >= 0 else (None, best_sim)

    def search_cam_vectorized(self, query: np.ndarray):
        """
        CAM-like: all cells compared at once via matrix operation.
        K @ q computes all similarities in one BLAS call.
        """
        if not self.keys_hv:
            return (None, -2.0)
        K = np.stack(self.keys_hv, axis=0)  # (N, D)
        sims = (K @ np.asarray(query)) / self.D  # (N,)
        idx = int(np.argmax(sims))
        return (self.values[idx], float(sims[idx]))

    def search_residue_partial(self, partial: Dict[int, int]):
        """Return all values whose residues match partial {index: residue}."""
        out = []
        for res, val in self._dict.items():
            if all(res[i] == r for i, r in partial.items()):
                out.append(val)
        return out


def random_bipolar(rng: np.random.Generator, D: int) -> np.ndarray:
    return rng.choice(np.array([-1.0, 1.0]), size=D)


def flip_bits(hv: np.ndarray, frac: float, rng: np.random.Generator) -> np.ndarray:
    out = np.array(hv, copy=True)
    n_flip = int(frac * len(out))
    idx = rng.choice(len(out), size=n_flip, replace=False)
    out[idx] = -out[idx]
    return out


def hopfield_retrieve(query: np.ndarray, patterns: np.ndarray, beta: float = 10.0):
    """
    Modern Hopfield retrieval (Ramsauer et al. 2020): softmax(beta * QK^T) V.
    Here patterns are both keys and values. Returns retrieved pattern.
    This is the associative-memory view of transformer attention.
    """
    # patterns: (N, D), query: (D,)
    scores = beta * (patterns @ query)  # (N,)
    # softmax
    scores = scores - np.max(scores)
    w = np.exp(scores)
    w = w / np.sum(w)
    retrieved = w @ patterns  # (D,)
    return retrieved, w
