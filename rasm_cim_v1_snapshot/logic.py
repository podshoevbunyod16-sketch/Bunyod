"""
Logic / compositional tasks + program generation for RASM-CIM prototype.

Logic encoding (bipolar HDC):
  fact HV = bind(subject, bind(relation, object))
  query(subject, relation): unbind -> cleanup in object codebook.

Program generation:
  Single-op programs encoded as bind(op, bind(arg, ...)).
  Decode via cleanup, generate Python code string, exec and verify.

Baselines: exact dict (perfect on clean data, fails under noise),
brute-force factorization.
"""
from __future__ import annotations
import numpy as np
from typing import Dict, Tuple, List


def bind(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a * b


def unbind(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a * b  # bipolar: self-inverse


def cleanup(query: np.ndarray, codebook: np.ndarray):
    """Return (index, similarity) of best codebook entry."""
    sims = (codebook @ query) / codebook.shape[1]
    idx = int(np.argmax(sims))
    return idx, float(sims[idx])


class LogicMemory:
    def __init__(self, n_subjects: int, n_relations: int, n_objects: int, D: int = 512, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.D = D
        self.subj_book = rng.choice(np.array([-1.0, 1.0]), size=(n_subjects, D))
        self.rel_book = rng.choice(np.array([-1.0, 1.0]), size=(n_relations, D))
        self.obj_book = rng.choice(np.array([-1.0, 1.0]), size=(n_objects, D))
        self.facts: List[Tuple[int, int, int]] = []
        self.fact_hvs: List[np.ndarray] = []

    def add_fact(self, s: int, r: int, o: int):
        hv = bind(self.subj_book[s], bind(self.rel_book[r], self.obj_book[o]))
        self.facts.append((s, r, o))
        self.fact_hvs.append(hv)

    def query(self, s: int, r: int):
        """Retrieve object for (s, r) by searching stored fact HVs."""
        if not self.fact_hvs:
            return None, -2.0
        # unbind query: fact * subj * rel -> object (approx, if fact exists)
        q = bind(self.subj_book[s], self.rel_book[r])
        best_idx, best_sim = -1, -2.0
        for i, hv in enumerate(self.fact_hvs):
            # similarity between hv and q*obj? Instead: unbind hv with q, cleanup
            unbound = unbind(hv, q)
            _, sim = cleanup(unbound, self.obj_book)
            # also require subj/rel match via direct similarity
            if sim > best_sim:
                best_sim, best_idx = sim, i
        if best_idx < 0:
            return None, best_sim
        return self.facts[best_idx][2], best_sim

    def query_noisy(self, s: int, r: int, noise_frac: float, rng: np.random.Generator):
        """Query with noisy subject/relation vectors (simulates sensor noise)."""
        # corrupt the query vectors by flipping bits
        def corrupt(v):
            out = v.copy()
            n = int(noise_frac * len(out))
            idx = rng.choice(len(out), size=n, replace=False)
            out[idx] = -out[idx]
            return out
        q = bind(corrupt(self.subj_book[s]), corrupt(self.rel_book[r]))
        best_idx, best_sim = -1, -2.0
        for i, hv in enumerate(self.fact_hvs):
            unbound = unbind(hv, q)
            _, sim = cleanup(unbound, self.obj_book)
            if sim > best_sim:
                best_sim, best_idx = sim, i
        return (self.facts[best_idx][2], best_sim) if best_idx >= 0 else (None, best_sim)


class DictLogicBaseline:
    """Exact symbolic baseline: dict lookup. Perfect clean, brittle to noise/unknown."""

    def __init__(self):
        self.d: Dict[Tuple[int, int], int] = {}

    def add_fact(self, s: int, r: int, o: int):
        self.d[(s, r)] = o

    def query(self, s: int, r: int):
        return self.d.get((s, r))

    def transitive(self, s: int, r1: int, r2: int):
        """Two-step composition s -r1-> mid -r2-> o, if both facts exist."""
        mid = self.d.get((s, r1))
        if mid is None:
            return None
        return self.d.get((mid, r2))


# --- Program generation ---
OPS = ["add", "sub", "mul", "max", "min", "neg"]

def generate_code(op: str, arg: int) -> str:
    if op == "add":
        return f"lambda x: x + ({arg})"
    if op == "sub":
        return f"lambda x: x - ({arg})"
    if op == "mul":
        return f"lambda x: x * ({arg})"
    if op == "max":
        return f"lambda x: max(x, ({arg}))"
    if op == "min":
        return f"lambda x: min(x, ({arg}))"
    if op == "neg":
        return f"lambda x: -x"
    raise ValueError(op)


def reference_apply(op: str, arg: int, x: int) -> int:
    fn = eval(generate_code(op, arg))
    return fn(x)


class ProgramMemory:
    """Encode (op, arg) as bipolar HV, decode via cleanup."""

    def __init__(self, arg_range=range(-5, 6), D: int = 512, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.D = D
        self.args = list(arg_range)
        self.op_book = rng.choice(np.array([-1.0, 1.0]), size=(len(OPS), D))
        self.arg_book = rng.choice(np.array([-1.0, 1.0]), size=(len(self.args), D))

    def encode(self, op: str, arg: int) -> np.ndarray:
        oi = OPS.index(op)
        ai = self.args.index(arg)
        return bind(self.op_book[oi], self.arg_book[ai])

    def decode(self, hv: np.ndarray):
        # brute-force over all op x arg combos (baseline decoder)
        best, best_sim = None, -2.0
        for oi, op in enumerate(OPS):
            for ai, arg in enumerate(self.args):
                cand = bind(self.op_book[oi], self.arg_book[ai])
                sim = float(np.mean(cand * hv))
                if sim > best_sim:
                    best_sim, best = sim, (op, arg)
        return best, best_sim
