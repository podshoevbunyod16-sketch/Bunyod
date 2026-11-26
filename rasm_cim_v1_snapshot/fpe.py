"""
Fractional Power Encoding (FPE) for RHC prototype.
Phasor vectors: z(x) = exp(i * phi * x), phi random.
Binding (elementwise multiply) <=> addition of values.
Residue FPE: phi restricted to multiples of 2*pi/m, so z(x+m)=z(x).
Reference: Kymn & Olshausen, Computing with Residue Numbers in
High-Dimensional Representation (Neural Computation / arXiv 2311.04872).
"""
from __future__ import annotations
import numpy as np


class IntegerFPE:
    """Encode integers as phasors with continuous random phases."""

    def __init__(self, D: int = 512, seed: int = 0):
        self.D = D
        rng = np.random.default_rng(seed)
        self.phi = rng.uniform(0, 2 * np.pi, size=D)  # (D,)
        self.base = np.exp(1j * self.phi)

    def encode(self, x: int) -> np.ndarray:
        return np.exp(1j * self.phi * x)

    def bind(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return a * b

    def unbind(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return a * np.conj(b)

    @staticmethod
    def bundle(*vecs: np.ndarray) -> np.ndarray:
        return np.sum(np.stack(vecs, axis=0), axis=0)

    @staticmethod
    def similarity(a: np.ndarray, b: np.ndarray) -> float:
        # Re(mean(a * conj(b))) in [-1, 1]; 1 = identical phasors
        return float(np.real(np.mean(a * np.conj(b))))

    def decode(self, vec: np.ndarray, codebook: dict) -> int:
        """codebook: value -> vector. Return value with max similarity."""
        best_val, best_sim = None, -2.0
        for val, hv in codebook.items():
            s = self.similarity(vec, hv)
            if s > best_sim:
                best_sim, best_val = s, val
        return best_val


class ResidueFPE:
    """
    Residue FPE for a single modulus m.
    Phases are multiples of 2*pi/m, so encoding is periodic with period m.
    Distinct residues are quasi-orthogonal for large D.
    """

    def __init__(self, m: int, D: int = 512, seed: int = 0):
        self.m = m
        self.D = D
        rng = np.random.default_rng(seed)
        # random multiples: k in {0,...,m-1}
        k = rng.integers(0, m, size=D)
        self.phi = 2 * np.pi * k / m
        self.base = np.exp(1j * self.phi)

    def encode_residue(self, r: int) -> np.ndarray:
        r = r % self.m
        return np.exp(1j * self.phi * r)

    def encode(self, x: int) -> np.ndarray:
        return self.encode_residue(x % self.m)

    # reuse static ops
    bind = IntegerFPE.bind
    unbind = IntegerFPE.unbind
    bundle = IntegerFPE.bundle
    similarity = IntegerFPE.similarity
    decode = IntegerFPE.decode


def max_offdiag_similarity(fpe, values):
    """Max |similarity| between distinct values — lower is better (more orthogonal)."""
    vecs = {v: fpe.encode(v) for v in values}
    max_sim = 0.0
    vals = list(values)
    for i in range(len(vals)):
        for j in range(i + 1, len(vals)):
            s = abs(fpe.similarity(vecs[vals[i]], vecs[vals[j]]))
            max_sim = max(max_sim, s)
    return max_sim
