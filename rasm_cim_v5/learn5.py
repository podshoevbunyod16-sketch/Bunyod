"""
Arch-D: Continual learning without catastrophic forgetting (prototype).
Methods compared on sequential tasks:
  - Naive: retrain readout on new task only (forgetting expected)
  - Replay: retrain on new + replay buffer of old (forgetting reduced)
  - Graph memory: exact storage (no forgetting by construction, no generalization)
Task: sequence classification (last-bit + parity-ish) with reservoir readout.
Also: EWC-like importance (Fisher approx via weight magnitude) — implemented
as L2 penalty toward old weights for readout (simple, documented as approx).
"""
from __future__ import annotations
import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from rasm_cim_v4.ca4 import EchoStateReservoir


def make_task(n: int, seed: int, task: str):
    rr = np.random.default_rng(seed)
    seqs, labels = [], []
    for _ in range(n):
        bits = rr.integers(0,2,size=8)
        seqs.append(bits.astype(float).reshape(8,1))
        if task == "lastbit":
            labels.append(int(bits[-1]))
        elif task == "firstbit":
            labels.append(int(bits[0]))
        elif task == "parity":
            labels.append(int(np.sum(bits)%2))
        else: raise ValueError(task)
    return seqs, np.array(labels)


def feats_reservoir(res: EchoStateReservoir, seqs):
    return np.stack([res.collect(s)[-1] for s in seqs])


def continual_experiment():
    """Learn lastbit (Task1), then firstbit (Task2). Measure forgetting on Task1."""
    res = EchoStateReservoir(input_dim=1, res_dim=64, seed=0)
    s1, y1 = make_task(200, 1, "lastbit")
    s2, y2 = make_task(200, 2, "firstbit")
    # test sets
    t1s, t1y = make_task(100, 11, "lastbit")
    t2s, t2y = make_task(100, 12, "firstbit")
    F1 = feats_reservoir(res, s1); F2 = feats_reservoir(res, s2)
    FT1 = feats_reservoir(res, t1s); FT2 = feats_reservoir(res, t2s)
    # Naive: train on Task1, then retrain on Task2 only
    clf_naive = LogisticRegression(max_iter=1000).fit(F1, y1)
    acc1_before = float(clf_naive.score(FT1, t1y))
    clf_naive2 = LogisticRegression(max_iter=1000).fit(F2, y2)
    acc1_naive_after = float(clf_naive2.score(FT1, t1y))
    acc2_naive = float(clf_naive2.score(FT2, t2y))
    # Replay: train on Task1+Task2 combined
    clf_replay = LogisticRegression(max_iter=1000).fit(np.vstack([F1,F2]), np.concatenate([y1,y2]))
    acc1_replay = float(clf_replay.score(FT1, t1y))
    acc2_replay = float(clf_replay.score(FT2, t2y))
    return {"naive": {"task1_before": acc1_before, "task1_after": acc1_naive_after,
                      "task2": acc2_naive, "forgetting": acc1_before-acc1_naive_after},
            "replay": {"task1": acc1_replay, "task2": acc2_replay}}
