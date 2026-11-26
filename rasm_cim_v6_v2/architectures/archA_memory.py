"""
Arch-A: Memory-Centric Intelligence (independent prototype, §3).
Associative retrieval, hierarchy, links, update, contradiction resolution.
Separate tests/metrics/report from Arch-B/C. Does NOT use CA/reservoir/synthesis.
"""
from __future__ import annotations
import numpy as np
from rasm_cim_v6_v2.memory.memory_system import MemorySystem


class ArchA_MemoryCentric:
    """Memory-centric: knowledge lives in memory; retrieval is associative."""

    def __init__(self, D: int = 256, seed: int = 0):
        self.mem = MemorySystem(D=D, seed=seed)

    def write(self, key, value): return self.mem.write_episodic(key, value)
    def learn_fact(self, s, r, o): return self.mem.learn_fact(s, r, o)
    def query_fact(self, s, r, o=None): return self.mem.query_fact(s, r, o)
    def query_exact(self, key): return self.mem.query_exact(key)
    def query_partial(self, residue, modulus): return self.mem.query_partial(residue, modulus)
    def query_noisy(self, qhv): return self.mem.query_noisy(qhv)
    def calibrate(self, known, unknown): return self.mem.calibrate_tau(known, unknown)
    def compress(self, x): return self.mem.compress_value(x)
    def decompress(self, r): return self.mem.decompress_value(r)
    def retention(self, kv): return self.mem.retention_check(kv)

    def metrics(self):
        return {"episodic_n": len(self.mem.episodic), "dedup_hits": self.mem.dedup_hits,
                "semantic_history": len(self.mem.semantic.history), "tau": self.mem.tau}
