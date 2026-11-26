"""
Fixed memory for v0.3: collision-safe CAM + overflow-safe RNS + graph memory.

Fixes:
- CAM collisions: exact index is multi-map (key -> list of ids). Query returns
  CONFLICT if multiple distinct values share a key (instead of silent overwrite).
- RNS overflow: encode raises OverflowError if x outside [0, M) (or signed range
  with offset). Multi-bank RNS extends range by product of banks (CRT across banks).
- Negative numbers: supported via signed offset (store x + OFFSET).
"""
from __future__ import annotations
import math
import numpy as np
from math import gcd
from functools import reduce
from typing import Dict, Any, Tuple, List, Set
from collections import defaultdict, deque
from .core import QueryResult


def _modinv(a: int, m: int) -> int:
    return pow(a % m, -1, m)


class RNSv3:
    """Overflow-safe packed RNS. Signed range: [-M//2, M//2)."""

    def __init__(self, moduli: Tuple[int, ...]):
        for i in range(len(moduli)):
            for j in range(i + 1, len(moduli)):
                if gcd(moduli[i], moduli[j]) != 1:
                    raise ValueError(f"Moduli must be pairwise coprime: {moduli}")
        self.moduli = tuple(moduli)
        self.K = len(moduli)
        self.M = reduce(lambda a, b: a * b, moduli)
        self.offset = self.M // 2  # signed offset

    def encode_unsigned(self, x: int) -> np.ndarray:
        if not (0 <= x < self.M):
            raise OverflowError(f"{x} outside [0,{self.M}) — would silently wrap in v0.2")
        return np.array([x % m for m in self.moduli], dtype=np.uint16)

    def encode_signed(self, x: int) -> np.ndarray:
        """Encode signed x in [-M//2, M//2) via offset."""
        if not (-self.offset <= x < self.M - self.offset):
            raise OverflowError(f"{x} outside signed range")
        return self.encode_unsigned(x + self.offset)

    def decode_unsigned(self, residues) -> int:
        residues = tuple(int(r) for r in residues)
        x = residues[0]; Mp = self.moduli[0]
        for i in range(1, self.K):
            mi = self.moduli[i]
            t = ((residues[i] - x) * _modinv(Mp % mi, mi)) % mi
            x = x + Mp * t; Mp *= mi
        return x % self.M

    def decode_signed(self, residues) -> int:
        return self.decode_unsigned(residues) - self.offset

    @property
    def bits_binary(self) -> int:
        return math.ceil(math.log2(self.M)) if self.M > 1 else 1

    @property
    def bits_rns(self) -> int:
        return sum(math.ceil(math.log2(m)) for m in self.moduli)


class MultiBankRNS:
    """Extend range: banks with coprime M's, combined via CRT (product)."""
    def __init__(self, bank_moduli_list):
        self.banks = [RNSv3(m) for m in bank_moduli_list]
        # check bank M's coprime
        Ms = [b.M for b in self.banks]
        for i in range(len(Ms)):
            for j in range(i+1, len(Ms)):
                if gcd(Ms[i], Ms[j]) != 1:
                    raise ValueError("Bank M's must be coprime")
        self.M = reduce(lambda a,b: a*b, Ms)

    def encode(self, x: int) -> List[np.ndarray]:
        if not (0 <= x < self.M):
            raise OverflowError(f"{x} outside [0,{self.M})")
        return [b.encode_unsigned(x % b.M) for b in self.banks]

    def decode(self, enc_list) -> int:
        # CRT across banks: each bank gives x mod M_i
        rems = [b.decode_unsigned(e) for b, e in zip(self.banks, enc_list)]
        x = rems[0]; Mp = self.banks[0].M
        for i in range(1, len(self.banks)):
            Mi = self.banks[i].M
            t = ((rems[i] - x) * _modinv(Mp % Mi, Mi)) % Mi
            x = x + Mp * t; Mp *= Mi
        return x % self.M


class FixedCAM:
    """
    Collision-safe associative memory (software simulation, labelled as such).
    Exact index: key -> list of ids. Query: 0 hits -> UNKNOWN, 1 distinct value -> value,
    >1 distinct values -> CONFLICT (instead of silent overwrite).
    Partial search via inverted index. Noisy HV search with tau refusal.
    """
    def __init__(self, D: int = 256, tau: float = 0.5):
        self.D = D; self.tau = tau
        self.values: List[Any] = []
        self.keys: List[Tuple[int, ...]] = []
        self.hvs: List[np.ndarray] = []
        self._exact: Dict[Tuple[int, ...], List[int]] = defaultdict(list)
        self._inv: Dict[Tuple[int,int], Set[int]] = defaultdict(set)
        self.ops = 0

    def write(self, key, hv: np.ndarray, value: Any) -> int:
        k = tuple(int(x) for x in key)
        rid = len(self.values)
        self.values.append(value); self.keys.append(k); self.hvs.append(np.asarray(hv, dtype=float).copy())
        self._exact[k].append(rid)
        for pos, r in enumerate(k):
            self._inv[(pos, r)].add(rid)
        return rid

    def query_exact(self, key) -> QueryResult:
        self.ops += 1
        k = tuple(int(x) for x in key)
        ids = self._exact.get(k, [])
        if not ids:
            return QueryResult(value=None, confidence=0.0, unknown=True, ops=self.ops, detail="no exact match")
        distinct = {self.values[i] for i in ids}
        if len(distinct) > 1:
            return QueryResult(value=None, confidence=0.0, unknown=False, conflict=True, ops=self.ops,
                               detail=f"collision: {len(distinct)} distinct values for same key")
        return QueryResult(value=self.values[ids[0]], confidence=1.0, unknown=False, ops=self.ops)

    def query_partial(self, partial: Dict[int,int]) -> List[Any]:
        if not partial:
            return list(self.values)
        lists = [self._inv.get((p,r), set()) for p,r in partial.items()]
        if any(len(s)==0 for s in lists):
            self.ops += 1; return []
        cand = min(lists, key=len); self.ops += len(cand)
        return [self.values[rid] for rid in cand if all(self.keys[rid][p]==r for p,r in partial.items())]

    def query_noisy(self, qhv: np.ndarray, tau: float | None = None) -> QueryResult:
        tau = self.tau if tau is None else tau
        if not self.hvs:
            return QueryResult(unknown=True, detail="empty")
        K = np.stack(self.hvs, axis=0)
        sims = (K @ np.asarray(qhv, dtype=float)) / self.D
        self.ops += len(self.values)
        idx = int(np.argmax(sims)); sim = float(sims[idx])
        if sim < tau:
            return QueryResult(value=None, confidence=sim, unknown=True, ops=self.ops, detail=f"sim {sim:.3f} < tau {tau}")
        return QueryResult(value=self.values[idx], confidence=sim, unknown=False, ops=self.ops)

    def __len__(self):
        return len(self.values)


class GraphMemory:
    """Knowledge graph: nodes + directed labelled edges, BFS composition (no matrices)."""
    def __init__(self):
        self.edges: Dict[Tuple[int,int], int] = {}  # (s,r) -> o (last wins, contradictions tracked separately)
        self.all_edges: List[Tuple[int,int,int]] = []
        self.adj: Dict[int, List[Tuple[int,int]]] = defaultdict(list)  # s -> [(r,o)]

    def add_edge(self, s: int, r: int, o: int) -> bool:
        """Returns True if contradiction (same s,r different o already exists)."""
        contra = (s,r) in self.edges and self.edges[(s,r)] != o
        self.edges[(s,r)] = o
        self.all_edges.append((s,r,o))
        self.adj[s].append((r,o))
        return contra

    def query(self, s: int, r: int) -> QueryResult:
        if (s,r) in self.edges:
            return QueryResult(value=self.edges[(s,r)], confidence=1.0, unknown=False)
        return QueryResult(value=None, confidence=0.0, unknown=True, detail="no edge")

    def compose_bfs(self, s: int, r1: int, r2: int) -> QueryResult:
        """BFS/2-step: s -r1-> mid -r2-> o. No matrices, pure graph traversal."""
        mid = self.edges.get((s, r1))
        if mid is None:
            return QueryResult(value=None, unknown=True, detail="stage1 missing")
        o = self.edges.get((mid, r2))
        if o is None:
            return QueryResult(value=None, unknown=True, detail="stage2 missing")
        return QueryResult(value=o, confidence=1.0, unknown=False, detail=f"via {mid}")

    def reachable(self, s: int, max_depth: int = 3) -> Set[int]:
        seen = {s}; frontier = deque([(s,0)])
        while frontier:
            node, d = frontier.popleft()
            if d >= max_depth: continue
            for _, o in self.adj.get(node, []):
                if o not in seen:
                    seen.add(o); frontier.append((o, d+1))
        return seen
