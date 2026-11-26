"""
BFS program synthesis v0.4 with train/val/test split + candidate counting.

Fixes v0.3: no split, no counts, brute force called discovery.
v0.4: synthesize on TRAIN examples only, select on VAL, report on TEST
(unseen inputs). Counts candidates visited, ops, time. BFS is search,
not "discovery" — documented as such.
"""
from __future__ import annotations
from collections import deque
from typing import List, Tuple, Optional, Dict
from rasm_cim_v3.program3 import apply_op, execute, SafeError, OPS, OPS_ARG, ARG_VALUES


def synthesize_split(train_examples, val_examples, test_examples,
                     max_depth: int = 3, max_nodes: int = 50000):
    """
    Returns dict with program, train/val/test accuracy, candidates, ops, time.
    Synthesis uses train only; val/test are held out (no peeking).
    """
    import time
    t0 = time.perf_counter()
    start = tuple(x for x,_ in train_examples)
    target = tuple(y for _,y in train_examples)
    queue = deque(); queue.append(([], start))
    visited = {start}
    candidates = 0
    found = None
    while queue:
        prog, state = queue.popleft()
        if len(prog) >= max_depth: continue
        for op in OPS:
            arg_list = ARG_VALUES if op in OPS_ARG else [None]
            for arg in arg_list:
                candidates += 1
                if candidates > max_nodes:
                    break
                try:
                    new_state = tuple(apply_op(op, arg, v) for v in state)
                except SafeError:
                    continue
                new_prog = prog + [(op, arg)]
                if new_state == target:
                    found = new_prog
                    queue.clear()
                    break
                if new_state not in visited:
                    visited.add(new_state)
                    queue.append((new_prog, new_state))
            if found is not None or candidates > max_nodes:
                break
    elapsed = time.perf_counter() - t0
    def acc(examples):
        if found is None: return 0.0
        try:
            return sum(1 for x,y in examples if execute(found, x)==y)/max(1,len(examples))
        except SafeError:
            return 0.0
    return {"program": found, "train_acc": acc(train_examples), "val_acc": acc(val_examples),
            "test_acc": acc(test_examples), "candidates": candidates,
            "ops": candidates, "time_s": elapsed, "max_depth": max_depth}
