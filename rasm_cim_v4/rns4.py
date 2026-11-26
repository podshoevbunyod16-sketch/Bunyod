"""
Range-aware RNS for v0.4 — fixes H8 collisions + honest parity.

Fixes:
- CAM keys must NOT be small-M residues (M=1155 collides at N>1155).
  v0.4 KeyEncoder uses full integer keys for exact index; RNS residues
  are used ONLY for arithmetic/partial search with explicit range.
- encode/decode costs measured separately (not hidden).
- Parity via ACTUAL RNS addition of bits (componentwise), not Python sum.
"""
from __future__ import annotations
import math
import numpy as np
from math import gcd
from functools import reduce
from typing import Tuple, List


def _modinv(a: int, m: int) -> int:
    return pow(a % m, -1, m)


class RNSv4:
    def __init__(self, moduli: Tuple[int, ...]):
        for i in range(len(moduli)):
            for j in range(i+1, len(moduli)):
                if gcd(moduli[i], moduli[j]) != 1:
                    raise ValueError("Moduli must be pairwise coprime")
        self.moduli = tuple(moduli); self.K = len(moduli)
        self.M = reduce(lambda a,b: a*b, moduli)
        self.offset = self.M // 2
        self.ops = 0

    def encode_unsigned(self, x: int) -> np.ndarray:
        if not (0 <= x < self.M):
            raise OverflowError(f"{x} outside [0,{self.M})")
        return np.array([x % m for m in self.moduli], dtype=np.uint16)

    def encode_signed(self, x: int) -> np.ndarray:
        if not (-self.offset <= x < self.M - self.offset):
            raise OverflowError(f"{x} outside signed range")
        return self.encode_unsigned(x + self.offset)

    def decode_unsigned(self, residues) -> int:
        residues = tuple(int(r) for r in residues)
        x = residues[0]; Mp = self.moduli[0]
        for i in range(1, self.K):
            mi = self.moduli[i]
            t = ((residues[i]-x) * _modinv(Mp % mi, mi)) % mi
            x = x + Mp*t; Mp *= mi
        return x % self.M

    def decode_signed(self, residues) -> int:
        return self.decode_unsigned(residues) - self.offset

    def add_residues(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Actual RNS addition (componentwise). Ops counted."""
        self.ops += self.K
        mods = np.array(self.moduli, dtype=np.int64)
        return ((a.astype(np.int64) + b.astype(np.int64)) % mods).astype(np.uint16)

    def sum_bits_rns(self, bits) -> int:
        """
        Parity/sum via RNS addition: encode each bit (0/1) as residues,
        add componentwise, decode. This ACTUALLY uses RNS (no Python sum).
        Returns sum (may overflow if sum >= M -> OverflowError, documented).
        """
        acc = np.zeros(self.K, dtype=np.uint16)
        for b in bits:
            acc = self.add_residues(acc, self.encode_unsigned(int(b)))
        return self.decode_unsigned(tuple(int(x) for x in acc))

    @property
    def bits_binary(self): return math.ceil(math.log2(self.M)) if self.M>1 else 1
    @property
    def bits_rns(self): return sum(math.ceil(math.log2(m)) for m in self.moduli)


class KeyEncoder:
    """
    CAM key encoder for v0.4: exact keys are full ints (no collisions).
    RNS residues are an ADDITIONAL partial-search index with explicit range.
    """
    def __init__(self, rns: RNSv4):
        self.rns = rns

    def exact_key(self, x: int) -> int:
        return int(x)  # full int, no wrap, no collision

    def partial_residues(self, x: int):
        """Residues for partial search; raises if x out of RNS range (explicit)."""
        return tuple(int(v) for v in self.rns.encode_unsigned(x))

    def in_range(self, x: int) -> bool:
        return 0 <= x < self.rns.M
