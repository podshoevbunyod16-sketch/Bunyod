"""
Device profiles for v0.6 (measured, not promised).
Mobile/Desktop/Server configs with real RAM/time measurements via tracemalloc.
No accuracy degradation for size (same algorithms, smaller D/graph only if stated).
"""
from __future__ import annotations
import time, tracemalloc
import numpy as np


PROFILES = {
    "mobile": {"D": 128, "max_facts": 1000, "note": "Android/Termux, CPU-first, numpy-only"},
    "desktop": {"D": 256, "max_facts": 10000, "note": "Laptop/PC, multithread numpy, optional GPU (unused)"},
    "server": {"D": 512, "max_facts": 100000, "note": "Large RAM, distributed (not implemented, single-node measured)"},
}


def measure_profile(profile_name: str, n_facts: int = 1000):
    """Write n_facts + query 100, measure peak RAM and time. Returns dict."""
    from rasm_cim_v6.memory6 import MemoryCentric
    cfg = PROFILES[profile_name]
    mem = MemoryCentric(D=cfg["D"], seed=0)
    tracemalloc.start()
    t0 = time.perf_counter()
    for i in range(n_facts):
        mem.write_episodic(i, i*2)
    t_write = time.perf_counter() - t0
    t0 = time.perf_counter()
    for i in range(100):
        mem.query_exact(i % n_facts)
    t_query = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {"profile": profile_name, "n_facts": n_facts, "D": cfg["D"],
            "write_s": t_write, "query100_s": t_query, "peak_ram_bytes": peak,
            "note": cfg["note"]}
