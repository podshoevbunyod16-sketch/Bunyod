"""v0.9 M1+M2 benchmark: compact index & journal vs v0.8 reference.

Every row: TARGET / ACTUAL / PASS|FAIL where the spec states a target
(update <=5 ms @1M); everything else is pure measurement, no thresholds.
All RSS numbers come from ISOLATED subprocesses (VmHWM peak / VmRSS resident).

Usage: python3 bench_v9/m1_m2_bench.py [--skip-1m]
Writes results_v9/m1_m2.json
"""
from __future__ import annotations
import json
import os
import statistics
import subprocess
import sys

ROOT = "/home/user"
OUT = f"{ROOT}/results_v9/m1_m2.json"
DATA = "/tmp/rasm_v9_bench"
M = 15015
NS = [1_000, 10_000, 100_000, 1_000_000]

CHILD = r'''
import json, os, sys, time, random, bisect
sys.path.insert(0, "/home/user")
os.makedirs("/tmp/rasm_v9_bench", exist_ok=True)
def vm(f):
    for line in open("/proc/self/status"):
        if line.startswith(f): return int(line.split()[1]) / 1024.0
    return 0.0
task, n, extra = sys.argv[1], int(sys.argv[2]), sys.argv[3] if len(sys.argv) > 3 else ""
facts = ((i, (i * 7919) % 15015) for i in range(n))
out = {"task": task, "n": n}
t0 = time.perf_counter()
if task == "v2_build":
    from rasm_cim_v8.storage import format as F
    p = f"/tmp/rasm_v9_bench/v2_{n}.rasm"
    F.write_v2(facts, p, page_recs=2048)
    out.update(build_s=round(time.perf_counter()-t0, 3), peak_mib=round(vm("VmHWM:"), 1),
               rss_after_mib=round(vm("VmRSS:"), 1), bytes=os.path.getsize(p))
elif task == "v3_build":
    from rasm_cim_v9.storage import index as IX
    p = f"/tmp/rasm_v9_bench/v3_{n}.rasm"
    IX.build_v3_base(facts, p, page_recs=2048, sort=extra or "auto")
    out.update(build_s=round(time.perf_counter()-t0, 3), peak_mib=round(vm("VmHWM:"), 1),
               rss_after_mib=round(vm("VmRSS:"), 1), bytes=os.path.getsize(p), sort=extra or "auto")
elif task == "v2_query":
    from rasm_cim_v8.storage.streaming import open_lazy
    p = f"/tmp/rasm_v9_bench/v2_{n}.rasm"
    t0 = time.perf_counter(); st = open_lazy(p, cache_pages=8); open_s = time.perf_counter()-t0
    rng = random.Random(11); probes = [rng.randrange(n) for _ in range(4000)]
    lat = []
    for k in probes:
        a = time.perf_counter(); st.get(k); lat.append((time.perf_counter()-a)*1000)
    lat.sort()
    out.update(open_s=round(open_s, 4), p50_ms=round(lat[len(lat)//2], 5),
               p95_ms=round(lat[int(len(lat)*0.95)], 5), rss_at_query_mib=round(vm("VmRSS:"), 1),
               peak_mib=round(vm("VmHWM:"), 1))
    st.close()
elif task == "v3_query":
    from rasm_cim_v9.storage.journal import V3Store
    p = f"/tmp/rasm_v9_bench/v3_{n}.rasm"
    t0 = time.perf_counter(); st = V3Store(p, cache_pages=8); open_s = time.perf_counter()-t0
    rng = random.Random(11); probes = [rng.randrange(n) for _ in range(4000)]
    lat = []
    for k in probes:
        a = time.perf_counter(); st.get(k); lat.append((time.perf_counter()-a)*1000)
    lat.sort()
    rows = sum(1 for _ in st.scan())
    out.update(open_s=round(open_s, 4), p50_ms=round(lat[len(lat)//2], 5),
               p95_ms=round(lat[int(len(lat)*0.95)], 5), scan_rows=rows,
               index_bytes=st.stats["index_bytes"], rss_at_query_mib=round(vm("VmRSS:"), 1),
               peak_mib=round(vm("VmHWM:"), 1))
    st.close()
elif task == "residue_v2":
    from rasm_cim_v8.memory.stream_memory import MobileStreamMemory
    p = f"/tmp/rasm_v9_bench/v2_{n}.rasm"
    r0 = vm("VmRSS:")
    t0 = time.perf_counter()
    m = MobileStreamMemory(p, tiers=("stream", "residue")).open()
    build_s = time.perf_counter()-t0
    rng = random.Random(7)
    lat = []
    for _ in range(200):
        a = time.perf_counter(); m.query_partial(rng.randrange(97), 97); lat.append((time.perf_counter()-a)*1000)
    lat.sort()
    out.update(build_s=round(build_s, 3), rss_delta_mib=round(vm("VmRSS:")-r0, 1),
               p50_ms=round(lat[100], 4), p95_ms=round(lat[189], 4), peak_mib=round(vm("VmHWM:"), 1))
elif task == "residue_v3":
    from rasm_cim_v9.storage.journal import V3Store
    p = f"/tmp/rasm_v9_bench/v3_{n}.rasm"
    st = V3Store(p, cache_pages=8)
    r0 = vm("VmRSS:")
    t0 = time.perf_counter(); ri = st.build_residue(); build_s = time.perf_counter()-t0
    rng = random.Random(7)
    lat = []
    for _ in range(200):
        a = time.perf_counter(); ri.query(rng.randrange(97), 97); lat.append((time.perf_counter()-a)*1000)
    lat.sort()
    out.update(build_s=round(build_s, 3), rss_delta_mib=round(vm("VmRSS:")-r0, 1),
               bytes=ri.bytes_resident(), p50_ms=round(lat[100], 4), p95_ms=round(lat[189], 4),
               peak_mib=round(vm("VmHWM:"), 1))
    st.close()
elif task == "update_v3":
    from rasm_cim_v9.storage.journal import V3Store, append_txn, compact
    p = f"/tmp/rasm_v9_bench/v3_upd_{n}.rasm"
    if os.path.exists(p):
        os.remove(p)          # always fresh: journal count must be exact
    from rasm_cim_v9.storage.index import build_v3_base
    t0 = time.perf_counter(); build_v3_base(facts, p, page_recs=2048); out["base_build_s"] = round(time.perf_counter()-t0, 3)
    lat = []
    for t in range(12):
        a = time.perf_counter()
        s = append_txn(p, [(2_000_000 + t, t)], fsync=True)
        assert s == t + 1
        lat.append((time.perf_counter()-a)*1000)
    out["upd_fsync_median_ms"] = round(statistics_median := __import__("statistics").median(lat), 4)
    lat = []
    for t in range(12, 60):
        a = time.perf_counter()
        append_txn(p, [(2_000_000 + t, t)], fsync=False)
        lat.append((time.perf_counter()-a)*1000)
    out["upd_nofsync_median_ms"] = round(__import__("statistics").median(lat), 4)
    # open cost with a growing journal
    for frames, label in ((60, "open_after_60txns_s"),):
        a = time.perf_counter(); st = V3Store(p); out[label] = round(time.perf_counter()-a, 4)
        assert st.n == n + 60
        st.close()
    a = time.perf_counter(); compact(p); out["compact_s"] = round(time.perf_counter()-a, 3)
    out["peak_mib"] = round(vm("VmHWM:"), 1)
elif task == "dict_ref":
    t0 = time.perf_counter()
    d = {}
    for k, v in facts:
        d[k] = v
    build = time.perf_counter() - t0
    rng = random.Random(11); probes = [rng.randrange(n) for _ in range(4000)]
    lat = []
    for k in probes:
        a = time.perf_counter(); d.get(k); lat.append((time.perf_counter()-a)*1000)
    lat.sort()
    out.update(build_s=round(build, 3), p50_ms=round(lat[len(lat)//2], 5),
               p95_ms=round(lat[int(len(lat)*0.95)], 5),
               rss_at_query_mib=round(vm("VmRSS:"), 1), peak_mib=round(vm("VmHWM:"), 1))
print(json.dumps(out))
'''


def run(task, n, extra=""):
    p = subprocess.run([sys.executable, "-c", CHILD, task, str(n), extra],
                       capture_output=True, text=True, timeout=1800, cwd=ROOT)
    if p.returncode != 0:
        return {"task": task, "n": n, "failure": (p.stderr or "")[-250:]}
    return json.loads(p.stdout.strip().splitlines()[-1])


def main(skip_1m=False):
    ns = NS[:-1] if skip_1m else NS
    rows = []
    print("== warm reference files (v2 build also doubles as v0.8 baseline re-check) ==")
    for n in ns:
        r = run("v2_build", n); rows.append(r); print("v2_build", r, flush=True)
    for n in ns:
        for sort in ("tuples", "order", "numpy", "auto"):
            if n >= 100_000 and sort in ("tuples", "order"):
                pass  # keep them: build-peak is the point
            r = run("v3_build", n, sort); rows.append(r); print("v3_build", sort, "->", r, flush=True)
    for n in ns:
        for task in ("v2_query", "v3_query", "dict_ref", "residue_v2", "residue_v3"):
            r = run(task, n); rows.append(r); print(task, n, r, flush=True)
    for n in ns:
        r = run("update_v3", n); rows.append(r); print("update_v3", n, r, flush=True)
    # verdicts
    verdicts = {}
    upd1m = next((r for r in rows if r.get("task") == "update_v3" and r.get("n") == 1_000_000), None)
    if upd1m and "upd_fsync_median_ms" in upd1m:
        verdicts["M2_update_1M"] = {"TARGET": "<=5 ms (fsync incl.)",
                                    "ACTUAL_fsync_ms": upd1m["upd_fsync_median_ms"],
                                    "ACTUAL_nofsync_ms": upd1m["upd_nofsync_median_ms"],
                                    "verdict": "PASS" if upd1m["upd_fsync_median_ms"] <= 5 else "FAIL"}
    v2p = {r["n"]: r for r in rows if r.get("task") == "v2_build"}
    v3p = {(r["n"], r.get("sort")): r for r in rows if r.get("task") == "v3_build"}
    verdicts["M1_build_peak"] = {f"{n:,}": {"v08_tuples_peak": v2p.get(n, {}).get("peak_mib"),
                                           "v09_numpy_peak": v3p.get((n, "numpy"), {}).get("peak_mib"),
                                           "v09_order_peak": v3p.get((n, "order"), {}).get("peak_mib")}
                                 for n in ns}
    r2 = {r["n"]: r for r in rows if r.get("task") == "residue_v2"}
    r3 = {r["n"]: r for r in rows if r.get("task") == "residue_v3"}
    verdicts["M1_residue_rss"] = {f"{n:,}": {"v08_dict_delta_mib": r2.get(n, {}).get("rss_delta_mib"),
                                             "v09_array_delta_mib": r3.get(n, {}).get("rss_delta_mib")}
                                  for n in ns}
    results = {"rows": rows, "verdicts": verdicts,
               "note": "v0.8 baselines re-measured in the same host session (v2_* rows) "
                       "for like-for-like comparison; historical JSONs untouched."}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(results, open(OUT, "w"), indent=1)
    print("\nVERDICTS:", json.dumps(verdicts, indent=1))
    print("saved", OUT)


if __name__ == "__main__":
    main(skip_1m="--skip-1m" in sys.argv)
