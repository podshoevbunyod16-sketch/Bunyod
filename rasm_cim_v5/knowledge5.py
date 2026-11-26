"""
Arch-A: Knowledge representation without tokens/vectors.
Typed hypergraph: nodes (id, type), hyperedges (relation, args, value).
Values that are ints use RNSv4 arithmetic (modular, no matrices).
No tokenization, no fixed vector space. Retrieval by graph traversal.
"""
from __future__ import annotations
from typing import Dict, List, Tuple, Set
from collections import defaultdict, deque
from rasm_cim_v4.rns4 import RNSv4


class HypergraphKB:
    def __init__(self):
        self.node_type: Dict[int, str] = {}
        self.hyperedges: List[dict] = []  # {rel, args: tuple, value, step}
        self._index: Dict[Tuple[int, tuple], List[int]] = defaultdict(list)  # (rel,args)->edge ids
        self.rns = RNSv4((3,5,7,11,13))
        self._step = 0

    def add_node(self, nid: int, ntype: str):
        self.node_type[nid] = ntype

    def add_hyperedge(self, rel: int, args: Tuple[int,...], value: int):
        self._step += 1
        eid = len(self.hyperedges)
        self.hyperedges.append({"rel": rel, "args": tuple(args), "value": int(value), "step": self._step})
        self._index[(rel, tuple(args))].append(eid)
        return eid

    def query(self, rel: int, args: Tuple[int,...]):
        ids = self._index.get((rel, tuple(args)), [])
        if not ids: return ("UNKNOWN", None)
        vals = {self.hyperedges[i]["value"] for i in ids}
        if len(vals) > 1: return ("CONFLICT", vals)
        return ("TRUE", next(iter(vals)))

    def arithmetic_hyperedge(self, rel_add: int, a: int, b: int) -> int:
        """Compute a+b via RNS residues (actual modular arithmetic)."""
        ra, rb = self.rns.encode_unsigned(a % self.rns.M), self.rns.encode_unsigned(b % self.rns.M)
        import numpy as np
        mods = np.array(self.rns.moduli, dtype=np.int64)
        rc = (ra.astype(np.int64) + rb.astype(np.int64)) % mods
        return self.rns.decode_unsigned(tuple(int(x) for x in rc))

    def compose(self, rel1: int, rel2: int, start: int):
        """BFS composition: start -rel1-> mid -rel2-> out (single-arg edges)."""
        # find mids
        mids = [e["value"] for e in self.hyperedges if e["rel"]==rel1 and e["args"]==(start,)]
        if not mids: return ("UNKNOWN", None)
        outs = set()
        for mid in mids:
            for e in self.hyperedges:
                if e["rel"]==rel2 and e["args"]==(mid,):
                    outs.add(e["value"])
        if not outs: return ("UNKNOWN", None)
        if len(outs)>1: return ("CONFLICT", outs)
        return ("TRUE", next(iter(outs)))
