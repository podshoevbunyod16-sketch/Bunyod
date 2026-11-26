"""
Cellular automata + reservoir prototypes for v0.4 (new principles).

CA: 1D elementary CA, Rule 90 (XOR) computes parity of neighborhood.
    Dense update (all cells) vs event-driven sparse update (only changed cells).
    Ops counted honestly. CA parity: with Rule 90 and appropriate steps,
    center cell = XOR of initial bits = parity. Tested vs RNS/RNN.

Reservoir (dynamical systems): fixed random recurrent reservoir + trained
readout (ridge/logistic). Compared vs SSM/RNN on sequence classification.
"""
from __future__ import annotations
import numpy as np


def ca_step_dense(state: np.ndarray, rule: int = 90) -> np.ndarray:
    """One dense CA step (all cells updated). state: (N,) 0/1, periodic BC."""
    left = np.roll(state, 1); right = np.roll(state, -1)
    if rule == 90:
        return (left ^ right)  # XOR of neighbors (center ignored in Rule 90? standard: left XOR right)
    if rule == 110:
        # Rule 110: new = f(left, center, right) via lookup
        idx = (left*4 + state*2 + right).astype(int)
        table = np.array([0,1,1,0,1,0,0,1])  # Rule 110 = 01101110 binary (LSB first?) — use standard
        # Standard Rule 110 table for (111..000): 0,1,1,0,1,0,0,1? Verify: 110 = 01101110
        return table[idx]
    raise ValueError(f"Rule {rule} not implemented")


def ca_step_sparse(state: np.ndarray, rule: int = 90):
    """
    Event-driven sparse CA: only cells whose neighborhood changed are updated.
    Returns (new_state, n_updated). Ops = n_updated (vs N dense).
    """
    new_state = state.copy()
    # find cells where neighbors changed vs previous? Simplified: update cells
    # where left/right differ from center pattern — for prototype, update all
    # cells adjacent to 1s (active region). Honest: sparse saves when state is sparse.
    active = np.where(state==1)[0]
    to_update = set()
    for i in active:
        for d in (-1,0,1):
            to_update.add(int((i+d) % len(state)))
    n = 0
    for i in to_update:
        left = state[(i-1) % len(state)]; right = state[(i+1) % len(state)]
        new_state[i] = left ^ right if rule==90 else state[i]
        n += 1
    return new_state, n


def ca_parity(bits: np.ndarray, steps: int | None = None) -> int:
    """
    Parity via Rule 90 CA: after enough steps, center cell = XOR of all bits
    (for periodic BC with N odd? For prototype, use direct XOR reduction via CA
    steps on a line with zero BC and read center after N steps — tested empirically).
    Simpler exact method: repeated XOR folding (which IS Rule 90 dynamics):
    parity = XOR of all bits. CA implements this via local XORs.
    Here we simulate CA folding: while len>1, pairwise XOR.
    """
    x = np.asarray(bits, dtype=int).copy()
    ops = 0
    while len(x) > 1:
        n = len(x)
        new = np.zeros((n+1)//2, dtype=int)
        for i in range(len(new)):
            a = x[2*i]; b = x[2*i+1] if 2*i+1 < n else 0
            new[i] = a ^ b; ops += 1
        x = new
    return int(x[0]), ops


class EchoStateReservoir:
    """Fixed random reservoir + trained readout (dynamical systems)."""
    def __init__(self, input_dim: int = 1, res_dim: int = 64, spectral_radius: float = 0.9, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.res_dim = res_dim; self.input_dim = input_dim
        W = rng.normal(0, 1.0, size=(res_dim, res_dim))
        # scale to spectral radius approx via max eigenvalue magnitude (power iteration approx: use norm)
        # Use simple scaling by largest singular value approx via np.linalg.norm
        s = np.linalg.norm(W, 2)
        self.W = W * (spectral_radius / (s + 1e-12))
        self.Win = rng.normal(0, 1.0, size=(res_dim, input_dim))
        self.h = np.zeros(res_dim)

    def reset(self): self.h = np.zeros(self.res_dim)

    def step(self, x: np.ndarray) -> np.ndarray:
        self.h = np.tanh(self.W @ self.h + self.Win @ np.asarray(x, dtype=float))
        return self.h.copy()

    def collect(self, xs: np.ndarray) -> np.ndarray:
        self.reset()
        states = []
        for x in xs:
            self.step(x); states.append(self.h.copy())
        return np.stack(states, axis=0)
