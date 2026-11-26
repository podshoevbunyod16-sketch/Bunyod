"""
Learning: BPTT selective SSM (numpy, no torch) — from v0.5 plan §6.
Selective SSM: input-dependent gate (Δ,B,C) like Mamba (Gu & Dao 2023).
Diagonal state, linear-time scan. Trained end-to-end via BPTT + Adam.
Compared vs TanhRNN-BPTT (v0.3) and readout-only baselines on parity/copy.
Honest: parity is hard; results reported even if negative.
"""
from __future__ import annotations
import numpy as np


class SelectiveSSM:
    """Diagonal selective SSM: h_t = a_t * h_{t-1} + b_t * x_t, y_t = c_t * h_t.
    a_t, b_t, c_t are input-dependent (selective). State dim D, input dim 1."""

    def __init__(self, D: int = 8, seed: int = 0):
        rng = np.random.default_rng(seed)
        s = 1.0 / np.sqrt(D)
        # gate projections: input (1) -> a_logit (D), b (D), c (D)
        self.Wa = rng.normal(0, s, size=(D, 1)); self.ba = np.zeros(D)
        self.Wb = rng.normal(0, s, size=(D, 1)); self.bb = np.zeros(D)
        self.Wc = rng.normal(0, s, size=(D, 1)); self.bc = np.zeros(D)
        self.Wy = rng.normal(0, s, size=(1, D)); self.by = np.zeros(1)
        self.D = D
        self._m = {}; self._v = {}; self._t = 0

    @staticmethod
    def _sigmoid(z): return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

    def _forward(self, xs: np.ndarray):
        """xs: (T,1). Returns cache for BPTT."""
        T = xs.shape[0]
        hs = np.zeros((T + 1, self.D))
        a_all = np.zeros((T, self.D)); b_all = np.zeros((T, self.D)); c_all = np.zeros((T, self.D))
        logits = np.zeros((T, 1))
        for t in range(T):
            a = self._sigmoid(self.Wa @ xs[t] + self.ba)          # decay gate in (0,1)
            b = self.Wb @ xs[t] + self.bb
            c = self.Wc @ xs[t] + self.bc
            hs[t + 1] = a * hs[t] + b * xs[t]
            a_all[t] = a; b_all[t] = b; c_all[t] = c
            logits[t] = self.Wy @ (c * hs[t + 1]) + self.by
        return hs, a_all, b_all, c_all, logits

    def predict_final(self, xs) -> int:
        _, _, _, _, logits = self._forward(np.asarray(xs, dtype=float))
        return int(self._sigmoid(logits[-1, 0]) > 0.5)

    def train_batch(self, seqs, labels, lr: float = 0.01):
        """One Adam step (BCE on final logit). Returns mean loss."""
        grads = {k: np.zeros_like(v) for k, v in
                 [("Wa", self.Wa), ("ba", self.ba), ("Wb", self.Wb), ("bb", self.bb),
                  ("Wc", self.Wc), ("bc", self.bc), ("Wy", self.Wy), ("by", self.by)]}
        total = 0.0; B = len(seqs)
        for xs, y in zip(seqs, labels):
            xs = np.asarray(xs, dtype=float); T = xs.shape[0]
            hs, a_all, b_all, c_all, logits = self._forward(xs)
            p = float(self._sigmoid(logits[-1, 0])); eps = 1e-12
            total += -(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
            dlogit = np.array([p - y])
            # grads through final readout
            hT = c_all[-1] * hs[-1]
            grads["Wy"] += np.outer(dlogit, hT); grads["by"] += dlogit
            dh = (self.Wy.T @ dlogit) * c_all[-1]  # dL/dh_T
            dc = (self.Wy.T @ dlogit) * hs[-1]     # dL/dc_T
            # BPTT backwards through selective scan
            for t in range(T - 1, -1, -1):
                a = a_all[t]
                # grads for b_t, c_t, a_t
                grads["Wb"] += np.outer(dh * xs[t, 0], xs[t]); grads["bb"] += dh * xs[t, 0]
                if t == T - 1:
                    grads["Wc"] += np.outer(dc, xs[t]); grads["bc"] += dc
                da = dh * hs[t] * a * (1 - a)  # through sigmoid gate
                grads["Wa"] += np.outer(da, xs[t]); grads["ba"] += da
                dh = dh * a  # propagate to h_t
        for k in grads: grads[k] /= B
        total /= B
        self._t += 1
        params = {"Wa": self.Wa, "ba": self.ba, "Wb": self.Wb, "bb": self.bb,
                  "Wc": self.Wc, "bc": self.bc, "Wy": self.Wy, "by": self.by}
        for k, g in grads.items():
            if k not in self._m: self._m[k] = np.zeros_like(g); self._v[k] = np.zeros_like(g)
            self._m[k] = 0.9 * self._m[k] + 0.1 * g
            self._v[k] = 0.999 * self._v[k] + 0.001 * (g ** 2)
            mhat = self._m[k] / (1 - 0.9 ** self._t); vhat = self._v[k] / (1 - 0.999 ** self._t)
            params[k] -= lr * mhat / (np.sqrt(vhat) + 1e-8)
        self.Wa, self.ba, self.Wb, self.bb = params["Wa"], params["ba"], params["Wb"], params["bb"]
        self.Wc, self.bc, self.Wy, self.by = params["Wc"], params["bc"], params["Wy"], params["by"]
        return float(total)

    def accuracy_final(self, seqs, labels) -> float:
        return float(np.mean([self.predict_final(s) == y for s, y in zip(seqs, labels)]))
