"""
Packed RNS for v0.2 — honest memory accounting.

Fix for v0.1 problem #1: v0.1 stored residues as Python tuple (huge overhead)
and never proved a memory advantage. v0.2 stores residues as packed numpy
arrays and compares bits/bytes against binary packed ints and hypervectors.

Honest theoretical result (proved in experiments F1):
  bits_binary(M) = ceil(log2(M))
  bits_RNS = sum(ceil(log2(mi)))
  bits_RNS >= bits_binary  (per-modulus padding)
=> RNS has NO memory advantage over binary for the same range.
Its potential advantage is carry-free parallelism / hardware energy,
which CANNOT be measured in Python and is not claimed here.
"""
from __future__ import annotations
import math
import numpy as np
from math import gcd
from functools import reduce
from typing import Tuple


def _modinv(a: int, m: int) -> int:
    return pow(a % m, -1, m)


class PackedRNS:
    """RNS with packed numpy storage and operation counting."""

    def __init__(self, moduli: Tuple[int, ...]):
        for i in range(len(moduli)):
            for j in range(i + 1, len(moduli)):
                if gcd(moduli[i], moduli[j]) != 1:
                    raise ValueError(f"Moduli must be pairwise coprime: {moduli}")
        self.moduli = tuple(moduli)
        self.K = len(moduli)
        self.M = reduce(lambda a, b: a * b, moduli)
        self.ops = 0  # modular add/mul counter

    # --- packed encode/decode ---
    def encode_packed(self, x: int) -> np.ndarray:
        """Return uint16 array of residues (packed storage)."""
        return np.array([x % m for m in self.moduli], dtype=np.uint16)

    def encode_batch(self, xs: np.ndarray) -> np.ndarray:
        """xs: (N,) ints -> (N, K) uint16 residues."""
        xs = np.asarray(xs, dtype=np.int64)
        return np.stack([xs % m for m in self.moduli], axis=1).astype(np.uint16)

    def decode(self, residues) -> int:
        residues = tuple(int(r) for r in residues)
        x = residues[0]
        Mp = self.moduli[0]
        for i in range(1, self.K):
            mi = self.moduli[i]
            inv = _modinv(Mp % mi, mi)
            t = ((residues[i] - x) * inv) % mi
            x = x + Mp * t
            Mp *= mi
        return x % self.M

    # --- arithmetic with op counting (packed arrays) ---
    def add_packed(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        self.ops += self.K
        mods = np.array(self.moduli, dtype=np.int64)
        return ((a.astype(np.int64) + b.astype(np.int64)) % mods).astype(np.uint16)

    def mul_packed(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        self.ops += self.K
        mods = np.array(self.moduli, dtype=np.int64)
        return ((a.astype(np.int64) * b.astype(np.int64)) % mods).astype(np.uint16)

    def reset_ops(self):
        self.ops = 0

    # --- memory accounting (theoretical bits + packed bytes) ---
    @property
    def bits_binary(self) -> int:
        return math.ceil(math.log2(self.M)) if self.M > 1 else 1

    @property
    def bits_rns(self) -> int:
        return sum(math.ceil(math.log2(m)) for m in self.moduli)

    def packed_bytes_per_number(self) -> int:
        """Bytes for one packed residue vector (uint16 per modulus)."""
        return self.K * 2

    def __repr__(self):
        return f"PackedRNS(moduli={self.moduli}, M={self.M}, bits_bin={self.bits_binary}, bits_rns={self.bits_rns})"


class BinaryBaseline:
    """Packed binary integer baseline for fair comparison."""

    def __init__(self, M: int):
        self.M = M
        self.bits = math.ceil(math.log2(M)) if M > 1 else 1
        # uint16 if M <= 65536 else uint32
        self.dtype = np.uint16 if M <= 65536 else np.uint32
        self.itemsize = np.dtype(self.dtype).itemsize
        self.ops = 0

    def encode_batch(self, xs: np.ndarray) -> np.ndarray:
        return np.asarray(xs, dtype=self.dtype)

    def add(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        self.ops += 1  # one machine add (carry chain inside, counted as 1 op here)
        return ((a.astype(np.int64) + b.astype(np.int64)) % self.M).astype(self.dtype)

    def mul(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        self.ops += 1
        return ((a.astype(np.int64) * b.astype(np.int64)) % self.M).astype(self.dtype)

    def packed_bytes_per_number(self) -> int:
        return self.itemsize

    def reset_ops(self):
        self.ops = 0
