"""
Arch-B Dynamic Computational Intelligence: CA + reservoir + graph propagation.
Local updates, event-driven sparse CA, reservoir dynamics.
No sequential token processing, no mandatory matrices in CA/graph paths
(reservoir uses matrices — documented; CA/graph do not).
"""
from __future__ import annotations
import numpy as np
from collections import deque
from typing import Dict, List, Set
from rasm_cim_v4.ca4 import ca_step_dense, ca_step_sparse, EchoStateReservoir


class DynamicSystem:
    def __init__(self, n_cells: int = 64, seed: int = 0):
        self.n = n_cells
        self.state = np.zeros(n_cells, dtype=int)
        self.reservoir = EchoStateReservoir(input_dim=1, res_dim=64, seed=seed)
        self.adj: Dict[int, List[int]] = {}

    def set_state(self, bits: np.ndarray):
        self.state = np.asarray(bits, dtype=int).copy()

    def ca_step(self, rule: int = 90, sparse: bool = False):
        """Returns (new_state, ops). Sparse ops = updated cells only."""
        if sparse:
            new, n = ca_step_sparse(self.state, rule=rule)
            self.state = new
            return new, n
        new = ca_step_dense(self.state, rule=rule)
        self.state = new
        return new, self.n

    def ca_parity(self, bits: np.ndarray) -> int:
        """Parity via CA XOR folding (local rules)."""
        from rasm_cim_v5.compute5 import ca_parity_ops
        return ca_parity_ops(bits)[0]

    def add_edge(self, a: int, b: int):
        self.adj.setdefault(a, []).append(b)

    def propagate(self, start: int, steps: int = 3) -> Set[int]:
        """Graph propagation (local message passing)."""
        seen = {start}; frontier = deque([(start,0)])
        while frontier:
            node, d = frontier.popleft()
            if d >= steps: continue
            for nb in self.adj.get(node, []):
                if nb not in seen:
                    seen.add(nb); frontier.append((nb, d+1))
        return seen

    def reservoir_step(self, x: float) -> np.ndarray:
        return self.reservoir.step(np.array([float(x)]))
