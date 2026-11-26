"""M2 benchmark — full-load vs streaming(pread/mmap/buffered) vs mmap.

Same data (100k records), same operations: build, 5000 random exact gets,
1 full scan, partial query; accuracy proven by equality of all three views
(streaming get == in-RAM dict == scan). Peak RSS attributed per mode by
subprocess isolation. Every number printed is MEASURED; targets
(RSS<40 MiB, p95<=2x) are reported as pass/fail, never asserted a priori.
"""
from __future__ import annotations
import json
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from rasm_cim_v8.bench.util import rss_mib, cur_rss_mib, stats_ms, in_subprocess, save  # noqa: E402
from rasm_cim_v8.storage import format as F  # noqa: E402
from rasm_cim_v8.storage.streaming import open_lazy  # noqa: E402
from rasm_cim_v8.memory.stream_memory import MobileStreamMemory  # noqa: E402

N = 100_000
PATH = "/tmp/rasm_v8_m2.rasm"
M = 15015


def facts_gen():
    for i in range(N):
        yield (i, (i * 7919) % M)


def measure_mode(mode: str, cache_pages: int) -> dict:
    import resource
    t0 = time.perf_counter()
    m = MobileStreamMemory(PATH, tiers=("stream",), cache_pages=cache_pages, mode=mode)
    st = m.open()
    t_open = time.perf_counter() - t0
    n = m.n_records()
    import random
    rng = random.Random(5)
    probes = [rng.randrange(N) for _ in range(5000)] + [-1, -2, 0, N - 1]
    lat = []
    acc = True
    for k in probes:
        t1 = time.perf_counter()
        got = m.query_exact(k)
        lat.append(time.perf_counter() - t1)
        if k < 0:
            acc &= got[0] == "UNKNOWN"
        else:
            acc &= got[0] == "TRUE" and got[1] == (k * 7919) % M
    t2 = time.perf_counter()
    scan_rows = [0]
    for k, v in m.scan():
        scan_rows[0] += 1
        if (k * 7919) % M != v:
            acc = False
    t_scan = time.perf_counter() - t2
    t3 = time.perf_counter()
    part = m.query_partial(7, 97)
    t_part = time.perf_counter() - t3
    pg = m.page_stats()
    pr = m.store.page_recs
    m.close()
    res = {"mode": mode, "cache_pages": cache_pages,
           "open_s": round(t_open, 4), "scan_s": round(t_scan, 3),
           "partial_s": round(t_part, 4), "scan_rows": scan_rows[0],
           "partial_len": len(part), "accuracy": acc,
           "get_lat": stats_ms(lat),
           "peak_rss_mib": round(rss_mib(), 1),
           "pages_read": pg["pages_read"], "cache_hits": pg["cache_hits"],
           "bytes_read": pg["bytes_read"]}
    return res


def baseline_dict() -> dict:
    m = MobileStreamMemory(PATH, tiers=("dict",))
    m.open()
    import random
    rng = random.Random(5)
    probes = [rng.randrange(N) for _ in range(5000)] + [-1, -2, 0, N - 1]
    lat = []
    acc = True
    for k in probes:
        t1 = time.perf_counter()
        got = m.query_exact(k)
        lat.append(time.perf_counter() - t1)
        if k < 0:
            acc &= got[0] == "UNKNOWN"
        else:
            acc &= got[0] == "TRUE" and got[1] == (k * 7919) % M
    out = {"mode": "dict-baseline", "accuracy": acc, "get_lat": stats_ms(lat),
           "peak_rss_mib": round(rss_mib(), 1)}
    return out


def v06_context() -> dict:
    """Historical in-RAM float-HV baseline at same N (from V08_BASELINE run)."""
    return {"mode": "v0.6 MemorySystem D=256 (from baseline §2)",
            "build_s": 1.813, "peak_rss_mib": 296.6,
            "note": "reference only; not rerun here"}


def main():
    # build once in parent
    t0 = time.perf_counter()
    meta = F.write_v2(facts_gen(), PATH, page_recs=2048)
    build_s = round(time.perf_counter() - t0, 3)
    build = {"build_s": build_s, "file_bytes": meta["bytes"], "n": meta["n_records"]}
    results = {"N": N, "build": build, "modes": []}
    # in parent we cannot isolate RSS per mode; use subprocess per mode
    mode_code = """
import sys, json
sys.argv = ["x"]
sys.path.insert(0, "/home/user")
from rasm_cim_v8.bench.m2_streaming import measure_mode
print(json.dumps(measure_mode(%r, %d)))
"""
    for mode, cp in [("pread", 2), ("pread", 8), ("pread", 32),
                     ("mmap", 8), ("buffered", 8)]:
        r = in_subprocess(mode_code % (mode, cp), timeout=600)
        results["modes"].append(r)
        print(mode, cp, "->", {k: v for k, v in r.items() if k != "get_lat"} if "error" not in r else r)
    base = in_subprocess("""
import sys, json
sys.path.insert(0, "/home/user")
from rasm_cim_v8.bench.m2_streaming import baseline_dict
print(json.dumps(baseline_dict()))
""", timeout=600)
    results["baseline_dict"] = base
    results["v06_context"] = v06_context()
    # criteria evaluation
    ok_rss = [r for r in results["modes"] if "error" not in r and r["peak_rss_mib"] < 40]
    if "get_lat" in base:
        for r in results["modes"]:
            if "error" in r:
                continue
            r["p95_ratio_vs_dict"] = round(r["get_lat"]["p95_ms"] / base["get_lat"]["p95_ms"], 3) if base["get_lat"]["p95_ms"] > 0 else None
    results["criteria"] = {
        "target_rss_40MiB": f"{len(ok_rss)}/{sum(1 for r in results['modes'] if 'error' not in r)} modes < 40 MiB (measured)",
        "target_p95_2x_dict": all(r.get("p95_ratio_vs_dict", 99) <= 2.0 for r in results["modes"] if "error" not in r),
        "accuracy_all_equal": all(r.get("accuracy", False) for r in results["modes"] if "error" not in r) and base.get("accuracy", False),
    }
    save("/home/user/results_v8/m2_streaming.json", results)
    print(json.dumps(results["criteria"], indent=1))


if __name__ == "__main__":
    main()
