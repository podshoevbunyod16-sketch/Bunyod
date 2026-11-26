"""
Resonator network factorization (bipolar HDC version).

Composite = elementwise product (binding) of F factor hypervectors.
Resonator iteratively cleans up each factor:
  unbound_i = composite * prod_{j!=i} est_j
  est_i = argmax_{codebook_i} similarity(unbound_i, codebook_entry)

Reference: Frady et al., Resonator Networks, Neural Computation 2020.
Known limits: operational capacity ~ O(N^2) in D, no global convergence
guarantee, limit cycles possible. Tested explicitly vs brute force.
"""
from __future__ import annotations
import numpy as np
from typing import List, Tuple


class Resonator:
    def __init__(self, codebooks: List[np.ndarray], max_iter: int = 100, seed: int = 0, n_restarts: int = 20):
        """
        codebooks: list of (M_i, D) bipolar arrays.
        n_restarts: stochastic restarts (like stochastic factorizers) to escape
        spurious fixed points. Single-restart version is kept for comparison
        via n_restarts=1.
        """
        self.codebooks = [np.asarray(cb, dtype=float) for cb in codebooks]
        self.F = len(codebooks)
        self.D = codebooks[0].shape[1]
        self.max_iter = max_iter
        self.n_restarts = n_restarts
        self.rng = np.random.default_rng(seed)

    def _run_once(self, composite: np.ndarray):
        # init: random entry from each codebook
        est_idx = [int(self.rng.integers(0, cb.shape[0])) for cb in self.codebooks]
        est = [self.codebooks[i][est_idx[i]] for i in range(self.F)]
        iters = 0
        for it in range(self.max_iter):
            iters = it + 1
            changed = False
            for i in range(self.F):
                unbound = composite.copy()
                for j in range(self.F):
                    if j != i:
                        unbound = unbound * est[j]
                sims = (self.codebooks[i] @ unbound) / self.D
                best = int(np.argmax(sims))
                if best != est_idx[i]:
                    changed = True
                est_idx[i] = best
                est[i] = self.codebooks[i][best]
            if not changed:
                break
        # final similarity of reconstruction
        pred = np.ones(self.D)
        for i in range(self.F):
            pred = pred * est[i]
        sim = float(np.mean(pred * composite))
        return tuple(est_idx), iters, sim

    def factorize(self, composite: np.ndarray):
        composite = np.asarray(composite, dtype=float)
        best_idx, best_iters, best_sim = None, 0, -2.0
        for _ in range(self.n_restarts):
            idx, iters, sim = self._run_once(composite)
            if sim > best_sim:
                best_sim, best_idx, best_iters = sim, idx, iters
            if best_sim > 0.999:  # perfect reconstruction, early stop
                break
        return best_idx, best_iters

    def brute_force(self, composite: np.ndarray):
        """Exhaustive search over all M^F combinations — baseline."""
        composite = np.asarray(composite, dtype=float)
        best_combo, best_sim = None, -2.0
        # iterate via recursion / product
        import itertools
        ranges = [range(cb.shape[0]) for cb in self.codebooks]
        for combo in itertools.product(*ranges):
            pred = np.ones(self.D)
            for i, idx in enumerate(combo):
                pred = pred * self.codebooks[i][idx]
            sim = float(np.mean(pred * composite))
            if sim > best_sim:
                best_sim, best_combo = sim, combo
        return best_combo, best_sim


def make_codebooks(sizes: List[int], D: int, seed: int = 0) -> List[np.ndarray]:
    rng = np.random.default_rng(seed)
    books = []
    for M in sizes:
        books.append(rng.choice(np.array([-1.0, 1.0]), size=(M, D)))
    return books


def bind_factors(codebooks: List[np.ndarray], indices: Tuple[int, ...]) -> np.ndarray:
    out = np.ones(codebooks[0].shape[1])
    for cb, idx in zip(codebooks, indices):
        out = out * cb[idx]
    return out
