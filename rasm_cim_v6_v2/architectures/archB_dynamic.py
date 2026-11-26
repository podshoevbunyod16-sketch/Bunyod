"""
Arch-B: Dynamic Computational Intelligence (independent prototype, §3).
Dynamics / local interactions / graphs instead of sequential token computation.
CA (dense/sparse), reservoir, graph propagation. Separate tests/metrics/report.
Does NOT use memory hierarchy or program synthesis.
"""
from __future__ import annotations
import numpy as np
from collections import deque
from typing import Dict, List, Set
from rasm_cim_v4.ca4 import ca_step_dense, ca_step_sparse, EchoStateReservoir


class ArchB_Dynamic:
    def __init__(self, n_cells: int = 64, seed: int = 0):
        self.n = n_cells
        self.state = np.zeros(n_cells, dtype=int)
        self.reservoir = EchoStateReservoir(input_dim=1, res_dim=64, seed=seed)
        self.adj: Dict[int, List[int]] = {}

    def set_state(self, bits): self.state = np.asarray(bits, dtype=int).copy()

    def ca_step(self, rule: int = 90, sparse: bool = False):
        if sparse:
            new, n = ca_step_sparse(self.state, rule=rule); self.state = new; return new, n
        new = ca_step_dense(self.state, rule=rule); self.state = new; return new, self.n

    def ca_parity(self, bits):
        from rasm_cim_v5.compute5 import ca_parity_ops
        return ca_parity_ops(bits)[0]

    def add_edge(self, a, b): self.adj.setdefault(a, []).append(b)

    def propagate(self, start, steps=3):
        seen = {start}; frontier = deque([(start, 0)])
        while frontier:
            node, d = frontier.popleft()
            if d >= steps: continue
            for nb in self.adj.get(node, []):
                if nb not in seen: seen.add(nb); frontier.append((nb, d + 1))
        return seen

    def reservoir_step(self, x): return self.reservoir.step(np.array([float(x)]))

    def metrics(self):
        return {"n_cells": self.n, "edges": sum(len(v) for v in self.adj.values())}
