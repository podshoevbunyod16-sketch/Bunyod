"""
Resonator2 for v0.2 — confidence + brute-force fallback, used in pipeline.

Fix for v0.1 problem #4: v0.1 resonator was tested in isolation only.
v0.2 resonator returns reconstruction confidence and falls back to
brute force when confidence is low. Logic2 and Program2 call it
as part of their query path (not as a side demo).
"""
from __future__ import annotations
import numpy as np
import itertools
from typing import List, Tuple


class Resonator2:
    def __init__(self, codebooks: List[np.ndarray], max_iter: int = 50, seed: int = 0, n_restarts: int = 20):
        self.codebooks = [np.asarray(cb, dtype=float) for cb in codebooks]
        self.F = len(codebooks)
        self.D = codebooks[0].shape[1]
        self.max_iter = max_iter
        self.n_restarts = n_restarts
        self.rng = np.random.default_rng(seed)
        self.ops = 0

    def _run_once(self, composite: np.ndarray):
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
                self.ops += self.codebooks[i].shape[0]
                best = int(np.argmax(sims))
                if best != est_idx[i]:
                    changed = True
                est_idx[i] = best
                est[i] = self.codebooks[i][best]
            if not changed:
                break
        pred = np.ones(self.D)
        for i in range(self.F):
            pred = pred * est[i]
        sim = float(np.mean(pred * composite))
        return tuple(est_idx), iters, sim

    def factorize(self, composite: np.ndarray, fallback_threshold: float = 0.90):
        """
        Returns (indices, iters, confidence, used_fallback).
        If best resonator confidence < threshold, falls back to brute force.
        """
        composite = np.asarray(composite, dtype=float)
        best_idx, best_iters, best_sim = None, 0, -2.0
        for _ in range(self.n_restarts):
            idx, iters, sim = self._run_once(composite)
            if sim > best_sim:
                best_sim, best_idx, best_iters = sim, idx, iters
            if best_sim > 0.999:
                break
        if best_sim >= fallback_threshold:
            return best_idx, best_iters, best_sim, False
        # fallback: brute force (exact, expensive — counted)
        bf_idx, bf_sim = self.brute_force(composite)
        return bf_idx, best_iters, bf_sim, True

    def brute_force(self, composite: np.ndarray):
        composite = np.asarray(composite, dtype=float)
        best_combo, best_sim = None, -2.0
        ranges = [range(cb.shape[0]) for cb in self.codebooks]
        for combo in itertools.product(*ranges):
            pred = np.ones(self.D)
            for i, idx in enumerate(combo):
                pred = pred * self.codebooks[i][idx]
            sim = float(np.mean(pred * composite))
            self.ops += 1
            if sim > best_sim:
                best_sim, best_combo = sim, combo
        return best_combo, best_sim

    def reset_ops(self):
        self.ops = 0


def make_codebooks(sizes: List[int], D: int, seed: int = 0) -> List[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [rng.choice(np.array([-1.0, 1.0]), size=(M, D)) for M in sizes]


def bind_factors(codebooks: List[np.ndarray], indices: Tuple[int, ...]) -> np.ndarray:
    out = np.ones(codebooks[0].shape[1])
    for cb, idx in zip(codebooks, indices):
        out = out * cb[idx]
    return out
