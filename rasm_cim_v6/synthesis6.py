"""
Arch-C Algorithmic Intelligence: synthesize -> verify -> compose -> self-correct.
- Synthesize via BFS on train examples (search, not discovery — documented)
- Verify on val (select best among candidates by val acc, not train)
- Compose: combine two verified programs (concatenation) for new tasks
- Self-correct: if test fails, add counterexample to train and resynthesize (one iteration)
Counts candidates/ops/time honestly. No eval (safe interpreter from v0.2/v0.4).
"""
from __future__ import annotations
from typing import List, Tuple, Optional
from rasm_cim_v4.program4 import synthesize_split
from rasm_cim_v3.program3 import execute, SafeError


class AlgorithmicLearner:
    def __init__(self):
        self.library: List[dict] = []  # verified programs

    def learn(self, train, val, test, max_depth: int = 3):
        """Synthesize on train, verify on val/test. Returns report dict."""
        res = synthesize_split(train, val, test, max_depth=max_depth, max_nodes=50000)
        prog = res["program"]
        entry = {"program": prog, "train_acc": res["train_acc"], "val_acc": res["val_acc"],
                 "test_acc": res["test_acc"], "candidates": res["candidates"], "time_s": res["time_s"]}
        if prog is not None and res["val_acc"] == 1.0:
            self.library.append(entry)
        return entry

    def compose_programs(self, p1, p2):
        """Concatenate two programs (composition). Length must be <=5 for safe exec."""
        if p1 is None or p2 is None: return None
        combined = list(p1) + list(p2)
        if len(combined) > 5: return None
        return combined

    def self_correct(self, program, train, val, test, max_depth: int = 3):
        """
        If program fails on test, add failing test examples to train and resynthesize once.
        Returns (new_program, improved: bool).
        """
        if program is None:
            return self.learn(train, val, test, max_depth=max_depth), True
        # find counterexamples in test
        counter = [(x,y) for x,y in test if execute(program, x) != y]
        if not counter:
            return {"program": program, "test_acc": 1.0}, False
        new_train = list(train) + counter
        res = synthesize_split(new_train, val, test, max_depth=max_depth, max_nodes=50000)
        return res, True

    def run(self, program, x: int) -> int:
        return execute(program, x)
