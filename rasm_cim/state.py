"""
Diagonal SSM state machine (CIM-SSM inspired, software simulation).

h_{t+1} = lambda * h_t + B @ x_t   (event-driven: if no event, only decay)
y_t = C @ h_t

Reference hardware: Zhang et al., Nat Commun 17, 1513 (2026) — state evolution
natively in crossbar + memristor state nodes, asynchronous event processing.
Software simulation here uses numpy; no hardware energy/speed claim is made.

Known limitation: a purely LINEAR diagonal SSM cannot solve parity/XOR
(needs nonlinearity). This is tested explicitly as a negative result.
"""
from __future__ import annotations
import numpy as np


class DiagonalSSM:
    def __init__(self, input_dim: int, state_dim: int = 32, decay: float = 0.9, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.input_dim = input_dim
        self.state_dim = state_dim
        if np.isscalar(decay):
            self.lam = np.full(state_dim, float(decay))
        else:
            self.lam = np.asarray(decay, dtype=float)
        self.B = rng.normal(0, 1.0 / np.sqrt(state_dim), size=(state_dim, input_dim))
        self.C = rng.normal(0, 1.0 / np.sqrt(state_dim), size=(state_dim, state_dim))
        self.h = np.zeros(state_dim)

    def reset(self):
        self.h = np.zeros(self.state_dim)

    def step(self, x: np.ndarray, event: bool = True) -> np.ndarray:
        if event:
            self.h = self.lam * self.h + self.B @ np.asarray(x, dtype=float)
        else:
            self.h = self.lam * self.h  # decay only, no input — event-driven saving
        return self.C @ self.h

    def run(self, xs: np.ndarray) -> np.ndarray:
        self.reset()
        outs = []
        for x in xs:
            outs.append(self.step(x, event=True))
        return np.stack(outs, axis=0)


class NonlinearRNNBaseline:
    """Simple tanh RNN baseline (numpy, random weights, untrained) for comparison."""

    def __init__(self, input_dim: int, hidden: int = 32, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.Wxh = rng.normal(0, 1.0 / np.sqrt(hidden), size=(hidden, input_dim))
        self.Whh = rng.normal(0, 1.0 / np.sqrt(hidden), size=(hidden, hidden)) * 0.5
        self.Why = rng.normal(0, 1.0 / np.sqrt(hidden), size=(hidden, hidden))
        self.hidden = hidden
        self.h = np.zeros(hidden)

    def reset(self):
        self.h = np.zeros(self.hidden)

    def step(self, x: np.ndarray) -> np.ndarray:
        self.h = np.tanh(self.Wxh @ np.asarray(x, dtype=float) + self.Whh @ self.h)
        return self.Why @ self.h

    def run(self, xs: np.ndarray) -> np.ndarray:
        self.reset()
        return np.stack([self.step(x) for x in xs], axis=0)


def cumulative_sum_baseline(xs: np.ndarray) -> np.ndarray:
    """Classical baseline for cumulative-sum task: exact, O(N)."""
    return np.cumsum(xs, axis=0)
