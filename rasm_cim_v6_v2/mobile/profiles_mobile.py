"""
Device profiles (§9): Mobile / Desktop / Server with REAL measurements.
Mobile: Android/Termux, limited RAM, CPU-first. Desktop: multithread, more RAM.
Server: large memory (single-node measured; distributed NOT implemented).
Real memory/speed/storage per mode. Not phone-ready from a single test.
"""
from __future__ import annotations
import time, tracemalloc, sys
import numpy as np


PROFILES = {
    "mobile": {"D": 128, "max_facts": 1000, "note": "Android/Termux, CPU-first, numpy-only, limited RAM"},
    "desktop": {"D": 256, "max_facts": 10000, "note": "Laptop/PC, multithread numpy, optional GPU (unused)"},
    "server": {"D": 512, "max_facts": 100000, "note": "Large RAM, single-node measured (distributed not implemented)"},
}


def measure_profile(profile_name: str, n_facts: int = 1000):
    from rasm_cim_v6_v2.memory.memory_system import MemorySystem
    cfg = PROFILES[profile_name]
    mem = MemorySystem(D=cfg["D"], seed=0)
    tracemalloc.start()
    t0 = time.perf_counter()
    for i in range(n_facts):
        mem.write_episodic(i, i * 2)
    t_write = time.perf_counter() - t0
    t0 = time.perf_counter()
    for i in range(100):
        mem.query_exact(i % n_facts)
    t_query = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    # storage: episodic list size estimate
    storage = sys.getsizeof(mem.episodic) + sum(sys.getsizeof(e) for e in mem.episodic)
    return {"profile": profile_name, "n_facts": n_facts, "D": cfg["D"],
            "write_s": t_write, "query100_s": t_query, "peak_ram_bytes": peak,
            "storage_bytes": storage, "note": cfg["note"]}
