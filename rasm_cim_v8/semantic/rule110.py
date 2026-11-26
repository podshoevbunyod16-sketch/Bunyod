"""M3a — Rule 110 semantics fix (v0.7 ERRORS #7; baseline audit correction).

Legacy table from rasm_cim_v4/ca4.py (kept for historical reproducibility):
    legacy = [0,1,1,0,1,0,0,1]      indexed by left*4 + center*2 + right
Canonical Rule 110 (01101110):
    out(pattern p) = (110 >> p) & 1
Diverging patterns (verified in V08_BASELINE §4.1 — FIVE, not three as
v0.7 wrote): 011, 100, 101, 110, 111.

Historical results stay on legacy (default canonical=False); canonical mode
is additive. Rule 90 path untouched (v0.4 experiments used it).
"""
from __future__ import annotations

LEGACY_TABLE = (0, 1, 1, 0, 1, 0, 0, 1)
CANONICAL_TABLE = tuple((110 >> p) & 1 for p in range(8))


def table(canonical: bool = False) -> tuple[int, ...]:
    return CANONICAL_TABLE if canonical else LEGACY_TABLE


def step_rule110(state, canonical: bool = False):
    """one periodic-BC step: new[i] = T[left, center, right]."""
    t = table(canonical)
    n = len(state)
    out = [0] * n
    for i in range(n):
        l = state[(i - 1) % n]
        c = state[i]
        r = state[(i + 1) % n]
        out[i] = t[l * 4 + c * 2 + r]
    return out


def all_eight(canonical: bool = False) -> dict[tuple[int, int, int], int]:
    t = table(canonical)
    return {(p >> 2 & 1, p >> 1 & 1, p & 1): t[p] for p in range(8)}
