"""
Trainable recurrent models for v0.3 (numpy BPTT, no torch).

Fix for v0.2 F3: readout-only training cannot solve parity.
v0.3 trains recurrent weights end-to-end via BPTT + Adam on parity/copy.
Models: TanhRNN (trainable), SelectiveSSM-fixed (readout-only baseline, from v0.2 logic).
All training is reproducible (fixed seeds, logged hyperparams).
"""
from __future__ import annotations
import numpy as np


class TanhRNNBPTT:
    """Tanh RNN with BPTT. Input dim Din, hidden H, output 1 logit (binary)."""

    def __init__(self, input_dim: int = 1, hidden: int = 16, seed: int = 0):
        rng = np.random.default_rng(seed)
        s = 1.0 / np.sqrt(hidden)
        self.Wxh = rng.normal(0, s, size=(hidden, input_dim))
        self.Whh = rng.normal(0, s, size=(hidden, hidden)) * 0.5
        self.bh = np.zeros(hidden)
        self.Why = rng.normal(0, s, size=(1, hidden))
        self.by = np.zeros(1)
        self.hidden = hidden; self.input_dim = input_dim
        # Adam state
        self._m = {}; self._v = {}; self._t = 0

    def _forward_seq(self, xs: np.ndarray):
        """xs: (T, Din). Returns cache (hs, ys_logits)."""
        T = xs.shape[0]
        hs = np.zeros((T+1, self.hidden))
        logits = np.zeros((T, 1))
        for t in range(T):
            hs[t+1] = np.tanh(self.Wxh @ xs[t] + self.Whh @ hs[t] + self.bh)
            logits[t] = self.Why @ hs[t+1] + self.by
        return hs, logits

    @staticmethod
    def _sigmoid(z):
        return 1.0/(1.0+np.exp(-np.clip(z, -30, 30)))

    def predict_seq(self, xs: np.ndarray) -> np.ndarray:
        _, logits = self._forward_seq(np.asarray(xs, dtype=float))
        return (self._sigmoid(logits[:,0]) > 0.5).astype(int)

    def predict_final(self, xs: np.ndarray) -> int:
        _, logits = self._forward_seq(np.asarray(xs, dtype=float))
        return int(self._sigmoid(logits[-1,0]) > 0.5)

    def train_batch(self, seqs: list, labels: list, lr: float = 0.01):
        """One Adam step on batch (mean BCE over final-step prediction)."""
        # forward + backward per sequence, accumulate grads
        grads = {k: np.zeros_like(v) for k, v in [("Wxh",self.Wxh),("Whh",self.Whh),("bh",self.bh),("Why",self.Why),("by",self.by)]}
        total_loss = 0.0
        B = len(seqs)
        for xs, y in zip(seqs, labels):
            xs = np.asarray(xs, dtype=float); T = xs.shape[0]
            hs, logits = self._forward_seq(xs)
            p = float(self._sigmoid(logits[-1,0]))
            eps = 1e-12
            total_loss += -(y*np.log(p+eps) + (1-y)*np.log(1-p+eps))
            # grad at final logit
            dlogit = np.array([p - y])  # (1,)
            grads["Why"] += np.outer(dlogit, hs[-1])
            grads["by"] += dlogit
            dh = (self.Why.T @ dlogit) * (1 - hs[-1]**2)  # (H,)
            # BPTT backwards
            for t in range(T-1, -1, -1):
                grads["Wxh"] += np.outer(dh, xs[t])
                grads["Whh"] += np.outer(dh, hs[t])
                grads["bh"] += dh
                if t > 0:
                    dh = (self.Whh.T @ dh) * (1 - hs[t]**2)
        for k in grads:
            grads[k] /= B
        total_loss /= B
        # Adam update
        self._t += 1
        params = {"Wxh":self.Wxh,"Whh":self.Whh,"bh":self.bh,"Why":self.Why,"by":self.by}
        for k, g in grads.items():
            if k not in self._m:
                self._m[k] = np.zeros_like(g); self._v[k] = np.zeros_like(g)
            self._m[k] = 0.9*self._m[k] + 0.1*g
            self._v[k] = 0.999*self._v[k] + 0.001*(g**2)
            mhat = self._m[k]/(1-0.9**self._t); vhat = self._v[k]/(1-0.999**self._t)
            params[k] -= lr * mhat/(np.sqrt(vhat)+1e-8)
        self.Wxh, self.Whh, self.bh, self.Why, self.by = params["Wxh"], params["Whh"], params["bh"], params["Why"], params["by"]
        return float(total_loss)

    def accuracy_final(self, seqs, labels) -> float:
        preds = [self.predict_final(s) for s in seqs]
        return float(np.mean([p==y for p,y in zip(preds, labels)]))


def make_parity(n: int, length: int, seed: int):
    rr = np.random.default_rng(seed)
    seqs, labels = [], []
    for _ in range(n):
        bits = rr.integers(0,2,size=length)
        seqs.append(bits.astype(float).reshape(length,1))
        labels.append(int(np.sum(bits)%2))
    return seqs, labels
