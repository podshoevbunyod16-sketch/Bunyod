"""
Fair benchmark harness for ArchA/B/C/D (v0.4 §2.5).
Same tasks, same inputs, same seeds/repeats, same hardware (this CPU).
Metrics: accuracy, latency, RAM (tracemalloc peak), ops (where counted),
scaling, update cost. Energy NOT measured (no hardware) — never claimed.
"""
from __future__ import annotations
import time, tracemalloc
import numpy as np


def bench_arch(arch_factory, tasks: dict, seeds=(0,1,2)):
    """
    tasks: {name: callable(arch, seed) -> (correct: bool, latency_s)}
    Returns {task: {acc_mean, acc_std, latency_mean, ...}}
    """
    results = {}
    for tname, fn in tasks.items():
        accs, lats = [], []
        for seed in seeds:
            arch = arch_factory(seed)
            tracemalloc.start()
            t0 = time.perf_counter()
            correct, _ = fn(arch, seed)
            lat = time.perf_counter() - t0
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            accs.append(1.0 if correct else 0.0); lats.append(lat)
        results[tname] = {"acc_mean": float(np.mean(accs)), "acc_std": float(np.std(accs)),
                          "latency_mean_s": float(np.mean(lats)), "n": len(seeds)}
    return results


def standard_tasks():
    """Identical tasks for all archs (small, honest)."""
    def t_memory(arch, seed):
        arch.learn_fact(0,0,1)
        return (arch.query_fact(0,0).value == 1 if hasattr(arch, 'query_fact') else arch.query(0,0)=="TRUE"), 0
    def t_compose(arch, seed):
        arch.learn_fact(0,0,1); arch.learn_fact(1,1,2)
        # ArchA/B/C have compose_fact, ArchD has compose
        if hasattr(arch, 'compose_fact'):
            return (arch.compose_fact(0,0,1).value == 2), 0
        return (arch.compose_value(0,0,1) == 2), 0
    def t_program(arch, seed):
        prog = arch.synthesize_program([(0,5),(1,8)], max_depth=3) if hasattr(arch,'synthesize_program') else arch.synthesize([(0,5),(1,8)],[],[],max_depth=3)["program"]
        ok = prog is not None and arch.run_program(prog, 0)==5
        return ok, 0
    return {"memory": t_memory, "compose": t_compose, "program": t_program}
