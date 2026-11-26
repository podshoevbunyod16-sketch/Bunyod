"""
Selective nonlinear SSM for v0.2 — integrated with CAM via context.

Fix for v0.1 problem #3: v0.1 SSM was linear, fixed-decay, and never
influenced retrieval. v0.2:
  - selective decay: lam_t = sigmoid(W_sel @ x_t) * lam_max  (input-dependent, Mamba-style)
  - nonlinearity: h = tanh(lam_t * h + B @ x_t)
  - trainable readout: states collected, readout trained via sklearn
    (LogisticRegression / Ridge) — reproducible, closed-form-ish, no deep training
  - context output: context_id = argmax(proj @ h), used as part of CAM key
    in system2.py. Ablation (no SSM) must fail context-dependent recall.

Reference: Mamba selective SSM — parameters become functions of input [1](https://arxiv.org/pdf/2312.00752).
Parity needs state tracking (flip on 1) — selective + nonlinear can represent it
with trained readout; linear fixed SSM cannot (v0.1 E4: 0.34).
"""
from __future__ import annotations
import numpy as np


class SelectiveSSM:
    def __init__(self, input_dim: int, state_dim: int = 32, lam_max: float = 0.99, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.input_dim = input_dim
        self.state_dim = state_dim
        self.lam_max = lam_max
        self.B = rng.normal(0, 1.0 / np.sqrt(state_dim), size=(state_dim, input_dim))
        self.W_sel = rng.normal(0, 0.5, size=(state_dim, input_dim))  # selective gate
        self.b_sel = np.zeros(state_dim)
        self.C = rng.normal(0, 1.0 / np.sqrt(state_dim), size=(state_dim, state_dim))
        # context projection (kept for compatibility; context_id now uses sign pattern)
        self.ctx_proj = rng.normal(0, 1.0, size=(4, state_dim))
        self.h = np.zeros(state_dim)

    def reset(self):
        self.h = np.zeros(self.state_dim)

    @staticmethod
    def _sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

    def step(self, x: np.ndarray, event: bool = True) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if event:
            lam_t = self._sigmoid(self.W_sel @ x + self.b_sel) * self.lam_max
            self.h = np.tanh(lam_t * self.h + self.B @ x)
        else:
            # no event: decay only (event-driven saving), no input
            self.h = np.tanh(self.lam_max * self.h)
        return self.C @ self.h

    def run(self, xs: np.ndarray) -> np.ndarray:
        self.reset()
        return np.stack([self.step(x) for x in xs], axis=0)

    def collect_states(self, xs: np.ndarray) -> np.ndarray:
        """Return hidden states h_t (not outputs) for readout training."""
        self.reset()
        states = []
        for x in xs:
            self.step(x, event=True)
            states.append(self.h.copy())
        return np.stack(states, axis=0)

    def context_id(self) -> int:
        """
        Context as sign pattern of first 8 state dims -> 0..255.
        v0.1/v0.2-early used argmax of 4 random projections, which collided
        (ctxA zeros -> all-zero scores -> 0; ctxB also -> 0). Sign pattern
        guarantees distinct states (with distinct signs) get distinct contexts
        with high probability. Zero state maps to 128 (all signs +1 by convention).
        """
        signs = (self.h[:8] >= 0).astype(int)
        cid = 0
        for i, s in enumerate(signs):
            cid |= (int(s) << i)
        return int(cid)


class LinearSSMv1:
    """v0.1 linear fixed-decay SSM (baseline, kept for comparison)."""

    def __init__(self, input_dim: int, state_dim: int = 32, decay: float = 0.9, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.input_dim = input_dim
        self.state_dim = state_dim
        self.lam = np.full(state_dim, float(decay))
        self.B = rng.normal(0, 1.0 / np.sqrt(state_dim), size=(state_dim, input_dim))
        self.h = np.zeros(state_dim)

    def reset(self):
        self.h = np.zeros(self.state_dim)

    def step(self, x: np.ndarray, event: bool = True) -> np.ndarray:
        if event:
            self.h = self.lam * self.h + self.B @ np.asarray(x, dtype=float)
        else:
            self.h = self.lam * self.h
        return self.h.copy()

    def collect_states(self, xs: np.ndarray) -> np.ndarray:
        self.reset()
        states = []
        for x in xs:
            self.step(x, event=True)
            states.append(self.h.copy())
        return np.stack(states, axis=0)


class TanhRNNBaseline:
    """Tanh RNN baseline (random weights, trained readout only — same protocol as SSM)."""

    def __init__(self, input_dim: int, hidden: int = 32, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.Wxh = rng.normal(0, 1.0 / np.sqrt(hidden), size=(hidden, input_dim))
        self.Whh = rng.normal(0, 1.0 / np.sqrt(hidden), size=(hidden, hidden)) * 0.5
        self.hidden = hidden
        self.h = np.zeros(hidden)

    def reset(self):
        self.h = np.zeros(self.hidden)

    def step(self, x: np.ndarray) -> np.ndarray:
        self.h = np.tanh(self.Wxh @ np.asarray(x, dtype=float) + self.Whh @ self.h)
        return self.h.copy()

    def collect_states(self, xs: np.ndarray) -> np.ndarray:
        self.reset()
        states = []
        for x in xs:
            self.step(x)
            states.append(self.h.copy())
        return np.stack(states, axis=0)


def train_readout(states: np.ndarray, labels: np.ndarray, task: str = "clf"):
    """Train linear readout on final states. Returns (model, train_acc)."""
    from sklearn.linear_model import LogisticRegression, Ridge
    X = np.asarray(states)
    y = np.asarray(labels)
    if task == "clf":
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X, y)
        return clf, float(clf.score(X, y))
    else:
        reg = Ridge(alpha=1.0)
        reg.fit(X, y)
        pred = reg.predict(X)
        # R^2
        ss_res = np.sum((y - pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2) + 1e-12
        return reg, float(1 - ss_res / ss_tot)
