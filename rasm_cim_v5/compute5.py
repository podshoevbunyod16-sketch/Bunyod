"""
Arch-B: Computation without sequential matrix multiplication.
- LUT computation: multiplication via precomputed table lookup (no mul op at runtime)
- CA computation: Rule 90 XOR folding (local updates)
- BFS/rewriting: graph traversal + BFS program synthesis (from v0.4)
All ops counted. No matrix multiplication in these paths.
"""
from __future__ import annotations
import numpy as np
from typing import List, Tuple


class LUTComputer:
    """Multiplication/addition via lookup tables (precompute once, lookup O(1))."""
    def __init__(self, max_val: int = 100):
        self.max_val = max_val
        self.mul_table = np.zeros((max_val+1, max_val+1), dtype=np.int64)
        self.add_table = np.zeros((max_val+1, max_val+1), dtype=np.int64)
        for a in range(max_val+1):
            for b in range(max_val+1):
                self.mul_table[a,b] = a*b
                self.add_table[a,b] = a+b
        self.ops = 0

    def mul(self, a: int, b: int) -> int:
        self.ops += 1
        if not (0 <= a <= self.max_val and 0 <= b <= self.max_val):
            raise OverflowError("LUT out of range")
        return int(self.mul_table[a,b])

    def add(self, a: int, b: int) -> int:
        self.ops += 1
        if not (0 <= a <= self.max_val and 0 <= b <= self.max_val):
            raise OverflowError("LUT out of range")
        return int(self.add_table[a,b])


def ca_parity_ops(bits) -> Tuple[int, int]:
    """Parity via CA XOR folding. Returns (parity, ops)."""
    x = np.asarray(bits, dtype=int).copy(); ops = 0
    while len(x) > 1:
        new = np.zeros((len(x)+1)//2, dtype=int)
        for i in range(len(new)):
            a = x[2*i]; b = x[2*i+1] if 2*i+1 < len(x) else 0
            new[i] = a ^ b; ops += 1
        x = new
    return int(x[0]), ops


def bfs_shortest_path(adj: dict, src: int, dst: int):
    """BFS shortest path (graph computation, no matrices). Returns (path, ops)."""
    from collections import deque
    q = deque([(src, [src])]); seen = {src}; ops = 0
    while q:
        node, path = q.popleft(); ops += 1
        if node == dst: return path, ops
        for nb in adj.get(node, []):
            if nb not in seen:
                seen.add(nb); q.append((nb, path+[nb]))
    return None, ops
