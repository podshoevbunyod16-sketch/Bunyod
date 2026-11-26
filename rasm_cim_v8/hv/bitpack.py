"""M5 — bit-packed bipolar hypervectors for noisy retrieval.

Representation: +1 -> bit 1, -1 -> bit 0. For D bits:
    dot(a, b) = D - 2 * popcount(xor(pack_a, pack_b))
    sim       = dot / D            (same normalization as v0.6 sims=(K@q)/D)

Variants benchmarked separately (bench8.m5; prep/query/RAM each timed):
  scalar   : python int xor + int.bit_count (C bigint ops; portable; NOT
             claimed to use any particular SIMD instruction set)
  numpy    : float64 K @ q (v0.6 baseline)
  numpy-i8 : uint64 chunk matrix + np.bitwise_count (numpy>=2.0), vectorized
  numba    : njit loop over uint64 chunks (JIT if available)

No NEON claim for any variant — instruction-level proof is not available
in this environment, and the TЗ forbids claiming SIMD without evidence.
"""
from __future__ import annotations
from typing import Sequence

try:
    import numpy as np            # optional: device-safe paths are stdlib-only
except ImportError:               # bare Termux python (M1) has none
    np = None


def pack_bits(hv: Sequence[float]) -> int:
    """bipolar (+/-1) vector -> python int bitset (bit i = value>0)."""
    m = 0
    for i, v in enumerate(hv):
        if v > 0:
            m |= 1 << i
    return m


def unpack_bits(m: int, d: int) -> np.ndarray:
    if np is None:
        raise RuntimeError("numpy required for this function; stdlib paths do not need it")
    return np.array([1.0 if (m >> i) & 1 else -1.0 for i in range(d)])


def sim_scalar(a: int, b: int, d: int) -> float:
    return (d - 2 * (a ^ b).bit_count()) / d


class BitpackIndex:
    """Store many packed HVs; best-match query with tau threshold —
    same decision rule as v0.6 query_noisy (argmax sim, tau gate,
    confidence=(sim+1)/2)."""

    def __init__(self, d: int, tau: float = 0.5):
        self.d = d
        self.tau = tau
        self.packed: list[int] = []
        self.vals: list[int] = []

    def add(self, hv: Sequence[float], value: int):
        self.packed.append(pack_bits(hv))
        self.vals.append(value)

    def add_packed(self, m: int, value: int):
        self.packed.append(m)
        self.vals.append(value)

    def query(self, q: Sequence[float] | int):
        if not self.packed:
            return ("UNKNOWN", None, 0.0)
        qi = q if isinstance(q, int) else pack_bits(q)
        d = self.d
        best_i, best_sim = 0, -2.0
        for i, p in enumerate(self.packed):
            s = (d - 2 * (qi ^ p).bit_count()) / d
            if s > best_sim:
                best_sim, best_i = s, i
        if best_sim < self.tau:
            return ("UNKNOWN", None, best_sim)
        return ("TRUE", self.vals[best_i], (best_sim + 1.0) / 2.0)

    def bytes_per_record(self) -> int:
        return (self.d + 7) // 8


def pack_to_u64_chunks(hvs: list[list[float]], d: int, nchunks: int | None = None):
    if np is None:
        raise RuntimeError("numpy required for this function; stdlib paths do not need it")
    """bipolar vectors -> uint64 matrix [N, ceil(d/64)] for numpy/numba."""
    if nchunks is None:
        nchunks = (d + 63) // 64
    out = np.zeros((len(hvs), nchunks), dtype=np.uint64)
    for r, hv in enumerate(hvs):
        bits = np.asarray(hv) > 0
        words = np.zeros(nchunks, dtype=np.uint64)
        for i in range(d):
            if bits[i]:
                words[i >> 6] |= np.uint64(1) << (i & 63)
        out[r] = words
    return out


def pack_to_u64_chunks_fast(hvs: np.ndarray) -> np.ndarray:
    if np is None:
        raise RuntimeError("numpy required for this function; stdlib paths do not need it")
    """Vectorized (float K x D) -> (uint64 K x ceil(D/64)); prep-side helper."""
    k, d = hvs.shape
    nch = (d + 63) // 64
    bits = (hvs > 0).astype(np.uint8)
    pad = nch * 64 - d
    if pad:
        bits = np.pad(bits, ((0, 0), (0, pad)))
    b = bits.reshape(k, nch, 64).astype(np.uint64)
    powers = (np.uint64(1) << np.arange(64, dtype=np.uint64))
    return (b * powers).sum(axis=2, dtype=np.uint64)


def sims_numpy_i8(mat: np.ndarray, q: np.ndarray, d: int) -> np.ndarray:
    if np is None:
        raise RuntimeError("numpy required for this function; stdlib paths do not need it")
    """vectorized popcount similarity, normalized by the TRUE d (padding
    bits are zero on both sides, so hamming is unaffected; scale is not)."""
    ham = np.bitwise_count(mat ^ q[None, :]).sum(axis=1, dtype=np.int64)
    return (d - 2 * ham) / d


def make_numba_kernel():
    if np is None:
        raise RuntimeError("numpy required for this function; stdlib paths do not need it")
    try:
        from numba import njit
    except Exception:
        return None

    @njit(cache=True)
    def _bc(x):
        # Kernighan popcount on uint64 — portable; explicitly NOT a SIMD
        # claim (see module docstring).
        c = np.uint64(0)
        while x != np.uint64(0):
            x &= x - np.uint64(1)
            c += np.uint64(1)
        return c

    @njit(cache=True)
    def best_match_tabled(mat, q, d, tau):
        n, c = mat.shape
        best_i = 0
        best_sim = -2.0
        for i in range(n):
            h = np.uint64(0)
            for j in range(c):
                h += _bc(mat[i, j] ^ q[j])
            sim = (d - 2.0 * h) / d
            if sim > best_sim:
                best_sim, best_i = sim, i
        return best_i, best_sim

    return best_match_tabled


# ---------------------------------------------------------------------------
# Deterministic HV generation from (key, seed): the HV of a record is
# reproducible from its key — no HV storage needed anywhere (M2 synergy:
# the file stays pure (key, value); the noisy layer regenerates packs on
# open and caches them in RAM, ~60 B/record for D=256 python ints).
# splitmix64 — same generator as v0.7 learning module (cross-checkable).

def _sm(s: int) -> int:
    s = (s + 0x9E3779B97F4A7C15) & MASK64
    z = s
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return s, z ^ (z >> 31)

MASK64 = (1 << 64) - 1


def gen_packed_hv(key: int, d: int = 256, seed: int = 0) -> int:
    state = (key * 0x9E3779B97F4A7C15 + seed + 1) & MASK64
    m = 0
    bits = 0
    while bits < d:
        state, z = _sm(state)
        take = min(64, d - bits)
        m |= (z & ((1 << take) - 1)) << bits
        bits += take
    return m


def gen_query_with_flips(key: int, d: int, seed: int, flip_frac: float,
                         rng_seed: int) -> tuple[int, int]:
    """(packed_hv_with_flips, number_of_flips) — for tests/benches."""
    m = gen_packed_hv(key, d, seed)
    import random
    rg = random.Random(rng_seed)
    nf = int(d * flip_frac)
    for _ in range(nf):
        m ^= 1 << rg.randrange(d)
    return m, nf
