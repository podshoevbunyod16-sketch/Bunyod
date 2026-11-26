"""Honest resonator for v0.3: reports fallback rate separately (no hidden failures)."""
from __future__ import annotations
import numpy as np
import itertools
from typing import List, Tuple


class Resonator3:
    def __init__(self, codebooks: List[np.ndarray], max_iter: int = 50, seed: int = 0, n_restarts: int = 20):
        self.codebooks = [np.asarray(cb, dtype=float) for cb in codebooks]
        self.F = len(codebooks); self.D = codebooks[0].shape[1]
        self.max_iter = max_iter; self.n_restarts = n_restarts
        self.rng = np.random.default_rng(seed)
        self.ops = 0; self.fallbacks = 0; self.calls = 0

    def _run_once(self, composite):
        est_idx = [int(self.rng.integers(0, cb.shape[0])) for cb in self.codebooks]
        est = [self.codebooks[i][est_idx[i]] for i in range(self.F)]
        for _ in range(self.max_iter):
            changed = False
            for i in range(self.F):
                unbound = composite.copy()
                for j in range(self.F):
                    if j != i: unbound = unbound * est[j]
                sims = (self.codebooks[i] @ unbound) / self.D
                self.ops += self.codebooks[i].shape[0]
                best = int(np.argmax(sims))
                if best != est_idx[i]: changed = True
                est_idx[i] = best; est[i] = self.codebooks[i][best]
            if not changed: break
        pred = np.ones(self.D)
        for i in range(self.F): pred = pred * est[i]
        return tuple(est_idx), float(np.mean(pred * composite))

    def factorize(self, composite, threshold: float = 0.90):
        self.calls += 1
        composite = np.asarray(composite, dtype=float)
        best_idx, best_sim = None, -2.0
        for _ in range(self.n_restarts):
            idx, sim = self._run_once(composite)
            if sim > best_sim: best_sim, best_idx = sim, idx
            if best_sim > 0.999: break
        if best_sim >= threshold:
            return best_idx, best_sim, False
        self.fallbacks += 1
        bf_idx, bf_sim = self.brute_force(composite)
        return bf_idx, bf_sim, True

    def brute_force(self, composite):
        composite = np.asarray(composite, dtype=float)
        best, best_sim = None, -2.0
        for combo in itertools.product(*[range(cb.shape[0]) for cb in self.codebooks]):
            pred = np.ones(self.D)
            for i, idx in enumerate(combo): pred = pred * self.codebooks[i][idx]
            sim = float(np.mean(pred * composite)); self.ops += 1
            if sim > best_sim: best_sim, best = sim, combo
        return best, best_sim

    def stats(self):
        return {"calls": self.calls, "fallbacks": self.fallbacks,
                "fallback_rate": self.fallbacks/max(1,self.calls), "ops": self.ops}


def make_codebooks(sizes, D, seed=0):
    rng = np.random.default_rng(seed)
    return [rng.choice(np.array([-1.0,1.0]), size=(M,D)) for M in sizes]

def bind_factors(codebooks, indices):
    out = np.ones(codebooks[0].shape[1])
    for cb, idx in zip(codebooks, indices): out = out * cb[idx]
    return out
