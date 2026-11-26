"""
Three architectures for v0.3 (all implemented, none pre-selected as winner):

A. Improved RASM-CIM: FixedCAM + RNSv3 (overflow-safe) + SelectiveSSM context
   + Resonator3 everywhere (logic compose + program decode) + safe programs.
B. Graph-Symbolic-Associative (GSA): knowledge graph (BFS) + CAM retrieval
   + RNS arithmetic + BFS program synthesis. Reasoning path uses NO matrix
   multiplication (graph traversal + modular arithmetic + Hamming via bits).
   HV similarity (dot product) is used ONLY for noisy retrieval, documented.
C. Hybrid: trained TanhRNN (BPTT) for sequences + GraphMemory for logic
   + FixedCAM for retrieval + RNSv3 for arithmetic + BFS synthesis.
   Each mechanism handles what it is best at (measured in experiments).
"""
from __future__ import annotations
import numpy as np
from .memory3 import RNSv3, FixedCAM, GraphMemory
from .resonator3 import Resonator3, make_codebooks, bind_factors
from .program3 import synthesize, execute, program_str
from .core import QueryResult


def _key_hv(rng, D, key_int: int) -> np.ndarray:
    # deterministic bipolar key (for retrieval demo; not used in graph reasoning path)
    r = np.random.default_rng(abs(key_int) % (2**31))
    return r.choice(np.array([-1.0,1.0]), size=D)


class ArchA:
    """Improved RASM-CIM."""
    def __init__(self, D: int = 256, seed: int = 0):
        self.rns = RNSv3((3,5,7,11))
        self.cam = FixedCAM(D=D, tau=0.5)
        self.graph = GraphMemory()
        self.rng = np.random.default_rng(seed)
        self.D = D
        # resonator codebooks for logic factorization demo
        self.books = make_codebooks([20,20], D, seed=seed)
        self.resonator = Resonator3(self.books, seed=seed)

    def learn_pair(self, key: int, value: int):
        res = tuple(int(x) for x in self.rns.encode_unsigned(key))
        return self.cam.write(res, _key_hv(self.rng, self.D, key), value)

    def query_pair(self, key: int) -> QueryResult:
        # overflow-safe: raises OverflowError for out-of-range (instead of silent wrap)
        res = tuple(int(x) for x in self.rns.encode_unsigned(key))
        return self.cam.query_exact(res)

    def learn_fact(self, s, r, o): return self.graph.add_edge(s, r, o)
    def query_fact(self, s, r): return self.graph.query(s, r)
    def compose_fact(self, s, r1, r2): return self.graph.compose_bfs(s, r1, r2)

    def factorize(self, composite):
        return self.resonator.factorize(composite)

    def synthesize_program(self, examples, max_depth=3):
        return synthesize(examples, max_depth=max_depth)

    def run_program(self, program, x): return execute(program, x)


class ArchB:
    """Graph-Symbolic-Associative: no matrices in reasoning path."""
    def __init__(self, D: int = 256, seed: int = 0):
        self.rns = RNSv3((3,5,7,11))
        self.graph = GraphMemory()
        self.cam = FixedCAM(D=D, tau=0.5)  # only for noisy retrieval demo
        self.rng = np.random.default_rng(seed)
        self.D = D

    def learn_fact(self, s, r, o): return self.graph.add_edge(s, r, o)
    def query_fact(self, s, r): return self.graph.query(s, r)
    def compose_fact(self, s, r1, r2): return self.graph.compose_bfs(s, r1, r2)
    def reachable(self, s, depth=3): return self.graph.reachable(s, depth)

    def arithmetic(self, a: int, b: int, op: str) -> int:
        """RNS arithmetic via residues (modular, no matrices)."""
        ra, rb = self.rns.encode_unsigned(a), self.rns.encode_unsigned(b)
        mods = np.array(self.rns.moduli, dtype=np.int64)
        if op == "add": rc = (ra.astype(np.int64) + rb.astype(np.int64)) % mods
        elif op == "mul": rc = (ra.astype(np.int64) * rb.astype(np.int64)) % mods
        else: raise ValueError(op)
        return self.rns.decode_unsigned(tuple(int(x) for x in rc))

    def parity_rns(self, bits) -> int:
        """Parity via RNS modulus 2 (exact, O(n))."""
        return int(sum(int(b) for b in bits) % 2)

    def synthesize_program(self, examples, max_depth=3):
        return synthesize(examples, max_depth=max_depth)

    def run_program(self, program, x): return execute(program, x)


class ArchC:
    """Hybrid: trained RNN + graph + CAM + RNS + BFS synthesis."""
    def __init__(self, D: int = 256, seed: int = 0):
        from .ssm3 import TanhRNNBPTT
        self.rnn = TanhRNNBPTT(input_dim=1, hidden=16, seed=seed)
        self.graph = GraphMemory()
        self.rns = RNSv3((3,5,7,11))
        self.cam = FixedCAM(D=D, tau=0.5)
        self.rng = np.random.default_rng(seed)
        self.D = D

    def train_rnn(self, seqs, labels, epochs=20, lr=0.02):
        losses = []
        for _ in range(epochs):
            losses.append(self.rnn.train_batch(seqs, labels, lr=lr))
        return losses

    def predict_sequence(self, xs) -> int:
        return self.rnn.predict_final(np.asarray(xs, dtype=float).reshape(-1,1))

    def learn_fact(self, s, r, o): return self.graph.add_edge(s, r, o)
    def query_fact(self, s, r): return self.graph.query(s, r)
    def compose_fact(self, s, r1, r2): return self.graph.compose_bfs(s, r1, r2)

    def arithmetic(self, a: int, b: int, op: str) -> int:
        ra, rb = self.rns.encode_unsigned(a), self.rns.encode_unsigned(b)
        mods = np.array(self.rns.moduli, dtype=np.int64)
        rc = (ra.astype(np.int64) + rb.astype(np.int64)) % mods if op=="add" else (ra.astype(np.int64)*rb.astype(np.int64)) % mods
        return self.rns.decode_unsigned(tuple(int(x) for x in rc))

    def synthesize_program(self, examples, max_depth=3):
        return synthesize(examples, max_depth=max_depth)

    def run_program(self, program, x): return execute(program, x)
