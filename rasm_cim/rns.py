"""
RNS core for RASM-CIM prototype.
Residue Number System: X <-> (X mod m1, ..., X mod mK), M = prod(mi).
Arithmetic is carry-free and component-wise, modulo M.
Limitation (documented): division and comparison are NOT native to RNS;
they require mixed-radix conversion / base extension (expensive).
"""
from __future__ import annotations
from math import gcd
from functools import reduce
from typing import Tuple, List


def _modinv(a: int, m: int) -> int:
    a %= m
    if gcd(a, m) != 1:
        raise ValueError(f"{a} has no inverse mod {m}")
    return pow(a, -1, m)


class RNS:
    def __init__(self, moduli: Tuple[int, ...]):
        if len(moduli) < 2:
            raise ValueError("Need at least 2 moduli")
        for i in range(len(moduli)):
            for j in range(i + 1, len(moduli)):
                if gcd(moduli[i], moduli[j]) != 1:
                    raise ValueError(f"Moduli must be pairwise coprime: {moduli}")
        self.moduli = tuple(moduli)
        self.K = len(moduli)
        self.M = reduce(lambda a, b: a * b, moduli)

    def encode(self, x: int) -> Tuple[int, ...]:
        return tuple(x % m for m in self.moduli)

    def decode(self, residues: Tuple[int, ...]) -> int:
        """CRT via iterative Garner-style reconstruction. Returns X in [0, M)."""
        if len(residues) != self.K:
            raise ValueError("Residue length mismatch")
        x = residues[0]
        M_partial = self.moduli[0]
        for i in range(1, self.K):
            mi = self.moduli[i]
            ri = residues[i]
            # solve x + M_partial*t = ri (mod mi)
            inv = _modinv(M_partial % mi, mi)
            t = ((ri - x) * inv) % mi
            x = x + M_partial * t
            M_partial *= mi
        return x % self.M

    # Component-wise arithmetic (mod M semantics)
    def add(self, a: Tuple[int, ...], b: Tuple[int, ...]) -> Tuple[int, ...]:
        return tuple((x + y) % m for x, y, m in zip(a, b, self.moduli))

    def sub(self, a: Tuple[int, ...], b: Tuple[int, ...]) -> Tuple[int, ...]:
        return tuple((x - y) % m for x, y, m in zip(a, b, self.moduli))

    def mul(self, a: Tuple[int, ...], b: Tuple[int, ...]) -> Tuple[int, ...]:
        return tuple((x * y) % m for x, y, m in zip(a, b, self.moduli))

    def neg(self, a: Tuple[int, ...]) -> Tuple[int, ...]:
        return tuple((-x) % m for x, m in zip(a, self.moduli))

    def mixed_radix(self, residues: Tuple[int, ...]) -> Tuple[int, ...]:
        """
        Convert residues to mixed-radix digits d where
        X = d0 + m0*d1 + m0*m1*d2 + ... , 0 <= di < mi.
        This is the standard way to compare RNS numbers without full CRT decode,
        but it still costs O(K^2) modular inverses/multiplications.
        """
        r = list(residues)
        d = [0] * self.K
        # Garner mixed-radix conversion
        # v_i computed iteratively
        v = [0] * self.K
        v[0] = r[0]
        d[0] = v[0]
        # precompute products
        prod = [1] * self.K
        for i in range(1, self.K):
            prod[i] = prod[i - 1] * self.moduli[i - 1]
        for i in range(1, self.K):
            # v_i = (r_i - sum_{j<i} prod[j]*d_j) * inv(prod[i] mod m_i) mod m_i
            s = sum(prod[j] * d[j] for j in range(i))
            inv = _modinv(prod[i] % self.moduli[i], self.moduli[i])
            v[i] = ((r[i] - s) * inv) % self.moduli[i]
            d[i] = v[i]
        return tuple(d)

    def compare(self, a: Tuple[int, ...], b: Tuple[int, ...]) -> int:
        """Return -1, 0, 1 by comparing mixed-radix digits from most significant."""
        da = self.mixed_radix(a)
        db = self.mixed_radix(b)
        for i in range(self.K - 1, -1, -1):
            if da[i] < db[i]:
                return -1
            if da[i] > db[i]:
                return 1
        return 0

    def base_extension(self, residues: Tuple[int, ...], new_modulus: int) -> int:
        """
        Compute X mod new_modulus. Naive implementation decodes via CRT first.
        This is O(K^2) and is a known RNS bottleneck — documented limitation.
        """
        x = self.decode(residues)
        return x % new_modulus

    def __repr__(self):
        return f"RNS(moduli={self.moduli}, M={self.M})"
