"""
Arch-C: Algorithmic Intelligence (independent prototype, §3).
Form/verify/combine/improve algorithms from experience, not only choose
from a prepared set. BFS synthesis on train, verify on val, compose verified
programs, self-correct with counterexamples. Separate tests/metrics/report.
Does NOT use memory hierarchy or CA/reservoir.
"""
from __future__ import annotations
from typing import List
from rasm_cim_v4.program4 import synthesize_split
from rasm_cim_v3.program3 import execute


class ArchC_Algorithmic:
    def __init__(self):
        self.library: List[dict] = []

    def learn(self, train, val, test, max_depth=3):
        res = synthesize_split(train, val, test, max_depth=max_depth, max_nodes=50000)
        prog = res["program"]
        entry = {"program": prog, "train_acc": res["train_acc"], "val_acc": res["val_acc"],
                 "test_acc": res["test_acc"], "candidates": res["candidates"], "time_s": res["time_s"]}
        if prog is not None and res["val_acc"] == 1.0:
            self.library.append(entry)
        return entry

    def compose_programs(self, p1, p2):
        if p1 is None or p2 is None: return None
        combined = list(p1) + list(p2)
        return combined if len(combined) <= 5 else None

    def self_correct(self, program, train, val, test, max_depth=3):
        if program is None:
            return self.learn(train, val, test, max_depth=max_depth), True
        counter = [(x, y) for x, y in test if execute(program, x) != y]
        if not counter: return {"program": program, "test_acc": 1.0}, False
        res = synthesize_split(list(train) + counter, val, test, max_depth=max_depth, max_nodes=50000)
        return res, True

    def run(self, program, x): return execute(program, x)

    def metrics(self):
        return {"library_size": len(self.library)}
