"""
Arch-C: Hierarchical scalable long-term memory.
L1: exact dict (full-int keys, O(1))
L2: inverted index partial search (sublinear for selective queries)
L3: HV noisy retrieval with tau refusal (associative)
Consolidation: L1 -> L2 index built on write (no separate phase needed).
Compression: RNS-packed values for ints in range (documented, not claimed smaller than binary).
Contradiction: multi-map with CONFLICT (from v0.4 FixedCAM logic, extended).
Scaling tested to 100k (not extrapolated to billions).
"""
from __future__ import annotations
import numpy as np
from typing import Dict, Any, Tuple, List, Set
from collections import defaultdict


class HierarchicalMemory:
    def __init__(self, D: int = 256, tau: float = 0.5):
        self.D = D; self.tau = tau
        self.l1: Dict[int, List[Any]] = defaultdict(list)  # key -> values (multi-map)
        self.l2_inv: Dict[Tuple[int,int], Set[int]] = defaultdict(set)  # (pos,residue)->ids (for int keys split)
        self.l3_hvs: List[np.ndarray] = []
        self.l3_vals: List[Any] = []
        self.ops = 0

    def write(self, key: int, value: Any, hv: np.ndarray | None = None):
        self.l1[int(key)].append(value)
        # L2: index by key mod small primes (partial search on key residues)
        for pos, m in enumerate([97, 89, 83]):
            self.l2_inv[(pos, int(key) % m)].add(int(key))
        if hv is not None:
            self.l3_hvs.append(np.asarray(hv, dtype=float).copy()); self.l3_vals.append(value)

    def query_exact(self, key: int):
        self.ops += 1
        vals = self.l1.get(int(key), [])
        if not vals: return ("UNKNOWN", None)
        distinct = set(vals)
        if len(distinct) > 1: return ("CONFLICT", distinct)
        return ("TRUE", vals[0])

    def query_partial(self, residue: int, modulus: int):
        """Find all keys with key % modulus == residue via L2 (ops = candidates)."""
        # find pos for modulus
        mods = [97,89,83]
        if modulus not in mods: raise ValueError("modulus must be 97/89/83 for L2 index")
        pos = mods.index(modulus)
        cand = self.l2_inv.get((pos, residue), set())
        self.ops += len(cand)
        return sorted(cand)

    def query_noisy(self, qhv: np.ndarray):
        if not self.l3_hvs: return ("UNKNOWN", None, -2.0)
        K = np.stack(self.l3_hvs, axis=0)
        sims = (K @ np.asarray(qhv, dtype=float)) / self.D
        self.ops += len(self.l3_vals)
        idx = int(np.argmax(sims)); sim = float(sims[idx])
        if sim < self.tau: return ("UNKNOWN", None, sim)
        return ("TRUE", self.l3_vals[idx], sim)

    def __len__(self): return len(self.l1)
