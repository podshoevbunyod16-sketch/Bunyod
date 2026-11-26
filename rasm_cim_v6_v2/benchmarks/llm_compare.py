"""
Honest LLM comparison (§6): same data/conditions, multiple repeats.
LLM is an EXTERNAL tool (separate), never the core. No GPU/model in sandbox,
so LLM baseline = documented literature numbers + local baselines (HashMap,
sklearn) on the SAME tasks. No specialized-vs-universal unfair comparison.
Independent test sets with versions (not only own synthetic).
"""
from __future__ import annotations
import time, json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def baseline_hashmap(train_pairs, test_keys):
    """Exact memorization baseline: dict lookup, UNKNOWN on unseen."""
    d = dict(train_pairs)
    return [d.get(k, "UNKNOWN") for k in test_keys]


def baseline_sklearn(train_X, train_y, test_X):
    """Strong ML baseline: RandomForest on features (cannot memorize IDs)."""
    clf = RandomForestClassifier(n_estimators=50, random_state=0)
    clf.fit(train_X, train_y)
    return clf.predict(test_X)


def measure(fn, repeats: int = 5):
    """Measure accuracy/time over repeats. Returns mean/std."""
    times = []; outs = []
    for _ in range(repeats):
        t0 = time.perf_counter(); outs.append(fn()); times.append(time.perf_counter() - t0)
    return {"time_mean": float(np.mean(times)), "time_std": float(np.std(times)), "out": outs[0]}


# Literature LLM numbers (cited, NOT measured here — documented as such)
LITERATURE_LLM = {
    "note": "Literature values, not measured in sandbox (no GPU/model). Cited for context only.",
    "kv_cache_128k_70B_GB": 40,  # per blogcloud analysis
    "prefill": "quadratic O(T^2)",
    "decode": "linear memory-bound O(T) per token",
    "source": "https://www.blogcloud.io/b/Gal-Ratner/p/context-windows-explained-why-your-llms-memory-costs-more-than-you-think",
}
