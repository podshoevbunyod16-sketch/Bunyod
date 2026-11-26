"""M2-refactor comparison benchmark: legacy full-cache HV vs streaming modes.

Design for honest RAM attribution:
  * one subprocess per (n, mode);
  * legacy REFERENCE RESULTS (picks/conf) are produced in a separate legacy
    subprocess FIRST and loaded read-only by the stream subprocesses — so the
    stream runs never pay legacy's allocations (freed memory may not return to
    the OS: pymalloc arenas — the classic trap this layout avoids);
  * stream modes still open their OWN legacy-comparable check is BY REFERENCE
    FILE equality, not by running legacy in-process;
  * metrics: open_s, init_s, query p50/p95/max (ms), records_examined/query,
    gen_calls/query, cache size, RSS-at-query (VmRSS) and peak (VmHWM).

Modes: legacy | gen | lru(cap2MiB) | block | block_numpy | combo(auto)
Plus sweeps at 100k: block size {1024, 8192, 32768} and lru cap {0.5, 2, 8} MiB.
--big adds a REAL 1M run (stream modes + legacy reference).

Usage: python3 rasm_cim_v8/bench/m2_hv_refactor.py [--big]
Writes results_v8/m2_hv_refactor.json
"""
from __future__ import annotations
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

DATA_DIR = "/tmp/rasm_v8_hvref_bench"
OUT = "/home/user/results_v8/m2_hv_refactor.json"
M = 15015
NQ = {"1000": 40, "20000": 30, "100000": 20, "1000000": 6}


def _build_files(ns):
    os.makedirs(DATA_DIR, exist_ok=True)
    from rasm_cim_v8.storage import format as F
    paths = {}
    for n in ns:
        p = f"{DATA_DIR}/hv_{n}.rasm"
        if not os.path.exists(p):
            F.write_v2(((i, (i * 7919) % M) for i in range(n)), p, page_recs=2048)
        paths[n] = p
    return paths


CHILD = r'''
import json, os, sys, time, random, resource
sys.path.insert(0, "/home/user")
from rasm_cim_v8.memory.stream_memory import MobileStreamMemory
from rasm_cim_v8.hv.bitpack import gen_query_with_flips

def vm(field):
    try:
        for line in open("/proc/self/status"):
            if line.startswith(field):
                return int(line.split()[1]) / 1024.0
    except OSError:
        return None

n, mode, path, cap = {n!r}, {mode!r}, {path!r}, {cap!r}
block = {block!r}
d, seed = 256, 7
t0 = time.perf_counter()
m = MobileStreamMemory(path, cache_pages=8)
m.open()
open_s = time.perf_counter() - t0
t0 = time.perf_counter()
if mode == "legacy":
    m.open_hv(d, seed)
else:
    m.open_hv_stream(d, seed, method=mode, block=block, cache_mb=cap)
init_s = time.perf_counter() - t0
nq = {nq!r}
rng = random.Random(11)
keys = [rng.randrange(n) for _ in range(nq)]
res = []
lat = []
exam = 0
for t, k in enumerate(keys):
    q, _ = gen_query_with_flips(k, d, seed, 0.10, t)
    a = time.perf_counter()
    r = m.query_noisy(q, 0.5)
    lat.append((time.perf_counter() - a) * 1000.0)
    res.append([r[0], r[1] if r[0] == "TRUE" else None, round(r[2], 12)])
    if mode != "legacy":
        exam = m._hvstream.last["records_examined"]
lat.sort()
out = {{"n": n, "mode": mode, "block": block, "cache_mb": cap,
       "open_s": round(open_s, 4), "init_s": round(init_s, 4),
       "p50_ms": round(lat[len(lat)//2], 4), "p95_ms": round(lat[int(len(lat)*0.95)-1], 4),
       "max_ms": round(lat[-1], 4),
       "examined_per_query": exam if mode != "legacy" else n,
       "gen_calls_per_query": (m._hvstream.stats["gen_calls"]//nq if mode != "legacy" else 0),
       "cache_size": (m._hvstream.cache.size if mode == "lru" else 0),
       "cache_bytes": (m._hvstream.resident_bytes() if mode != "legacy" else m.hv_bytes_estimate()),
       "rss_at_query_mib": round(vm("VmRSS:"), 1),
       "peak_rss_mib": round(vm("VmHWM:") or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.0, 1),
       "picks": res}}
print(json.dumps(out))
'''


def run(n: int, mode: str, path: str, cap=2.0, block=8192, nq=None):
    code = CHILD.format(n=n, mode=mode, path=path, cap=cap, block=block,
                        nq=nq or NQ[str(n)])
    p = subprocess.run([sys.executable, "-c", code], capture_output=True,
                       text=True, timeout=1800)
    if p.returncode != 0:
        return {"n": n, "mode": mode, "failure": (p.stderr or p.stdout)[-300:]}
    return json.loads(p.stdout.strip().splitlines()[-1])


def main(big=False):
    ns = [1_000, 20_000, 100_000] + ([1_000_000] if big else [])
    paths = _build_files(ns)
    results = {"device_note": "HOST measurements (2-CPU linux). Android ARM64 "
               "numbers from the user: legacy 38.7 MiB @20k / 56.4 MiB @100k. "
               "Device re-run: mobile_bench.py (hv section).",
               "runs": [], "parity": {}}
    for n in ns:
        ref = run(n, "legacy", paths[n])
        results["runs"].append({k: v for k, v in ref.items() if k != "picks"})
        mism_tot = 0
        for mode in (("gen", "lru", "block", "block_numpy", "combo")
                     if n <= 20_000 else ("lru", "block_numpy", "combo")):
            r = run(n, mode, paths[n])
            if "failure" in r:
                results["runs"].append(r); continue
            if "picks" in ref:
                bad = sum(1 for a, b in zip(r.pop("picks"), ref["picks"]) if a != b)
                mism_tot += bad
            results["runs"].append(r)
        results["parity"][str(n)] = f"mismatches_vs_legacy={mism_tot} (n={n})"
        print(n, results["parity"][str(n)], flush=True)
        ref.pop("picks", None)
    # sweeps at 100k
    n = 100_000
    for b in (1_024, 8_192, 32_768):
        r = run(n, "block_numpy", paths[n], block=b)
        r.pop("picks", None); r["sweep"] = "block_bytes"
        results["runs"].append(r); print("block", b, r["p50_ms"], r["rss_at_query_mib"], flush=True)
    for c in (0.5, 2.0, 8.0):
        r = run(n, "lru", paths[n], cap=c)
        r.pop("picks", None); r["sweep"] = "lru_cap"
        results["runs"].append(r); print("lru cap", c, r["p50_ms"], r["cache_size"], flush=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(results, open(OUT, "w"), indent=1)
    print("saved", OUT)


if __name__ == "__main__":
    main(big="--big" in sys.argv)
