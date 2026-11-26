"""
Arch-E: Unified NEXTGEN system (competing system, no LLM core).
Query interface: ask(query_dict) -> answer dict with confidence/UNKNOWN.
Handles: math (RNS/LUT), logic (hypergraph), programs (BFS synthesis),
sequences (reservoir), tools (validated), memory (hierarchical).
External LLM not used as core; if needed, must be separate tool (not here).
"""
from __future__ import annotations
import numpy as np
from .knowledge5 import HypergraphKB
from .compute5 import LUTComputer, ca_parity_ops
from .memory5 import HierarchicalMemory
from rasm_cim_v4.program4 import synthesize_split
from rasm_cim_v3.program3 import execute
from rasm_cim_v4.ca4 import EchoStateReservoir
from sklearn.linear_model import LogisticRegression


class NEXTGEN:
    def __init__(self, seed: int = 0):
        self.kb = HypergraphKB()
        self.lut = LUTComputer(max_val=100)
        self.mem = HierarchicalMemory(D=256, tau=0.5)
        self.res = EchoStateReservoir(input_dim=1, res_dim=64, seed=seed)
        self._clf = None
        self.rng = np.random.default_rng(seed)

    def learn_fact(self, rel, args, value):
        return self.kb.add_hyperedge(rel, args, value)

    def learn_pair(self, key: int, value: int):
        hv = self.rng.choice(np.array([-1.0,1.0]), size=256)
        self.mem.write(key, value, hv=hv)

    def train_sequence_classifier(self, seqs, labels):
        feats = np.stack([self.res.collect(s)[-1] for s in seqs])
        self._clf = LogisticRegression(max_iter=1000).fit(feats, np.array(labels))

    def ask(self, query: dict) -> dict:
        qtype = query.get("type")
        if qtype == "math":
            op, a, b = query["op"], query["a"], query["b"]
            if op == "add": return {"answer": self.lut.add(a,b), "confidence": 1.0}
            if op == "mul": return {"answer": self.lut.mul(a,b), "confidence": 1.0}
            return {"answer": None, "unknown": True}
        if qtype == "parity":
            p, _ = ca_parity_ops(query["bits"])
            return {"answer": p, "confidence": 1.0}
        if qtype == "fact":
            status, val = self.kb.query(query["rel"], tuple(query["args"]))
            return {"answer": val, "status": status, "confidence": 1.0 if status=="TRUE" else 0.0}
        if qtype == "memory":
            status, val = self.mem.query_exact(query["key"])
            return {"answer": val, "status": status}
        if qtype == "program":
            res = synthesize_split(query["train"], query["val"], query["test"], max_depth=query.get("max_depth",3))
            return {"answer": res["program"], "test_acc": res["test_acc"], "candidates": res["candidates"]}
        if qtype == "sequence":
            if self._clf is None: return {"answer": None, "unknown": True}
            feat = self.res.collect(np.asarray(query["seq"], dtype=float).reshape(-1,1))[-1].reshape(1,-1)
            return {"answer": int(self._clf.predict(feat)[0]), "confidence": float(np.max(self._clf.predict_proba(feat)))}
        return {"answer": None, "unknown": True, "detail": f"unknown query type {qtype}"}
