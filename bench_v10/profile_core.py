"""v0.10 Ш1 — reproducible core profiling pipeline.

Rules from the spec: same data/conditions for both engines, multiple repeats,
process isolation (every cell = fresh subprocess), p50/p95/p99 + throughput +
build time + storage size + VmHWM + VmRSS delta + tracemalloc peak/total.
`--quick` runs a Termux-friendly subset (1k/20k, 2 repeats).

Cells (per volume): v2_build(v0.8 ref), v3_build, v3_open, v2_get, v3_get,
v3_partial(+residue build), v3_scan, v3_update_fsync, v3_frames_curve
(open cost vs #committed single-record frames), v3_compact.

Output: results_v10/profile_matrix.json (+ human table on stdout).
Run: python3 bench_v10/profile_core.py [--quick|--reps 3]
"""
from __future__ import annotations
import argparse
import json
import os
import statistics
import subprocess
import sys

ROOT = "/home/user"
M = 15015

CHILD = r'''
import json, os, statistics, sys, time, random
import tracemalloc
sys.path.insert(0, os.environ.get("RASM_ROOT", "/home/user"))
task, n, arg = sys.argv[1], int(sys.argv[2]), sys.argv[3]
M = 15015
D = "/tmp/rasm_v10_prof"
os.makedirs(D, exist_ok=True)
def vm(f):
    for line in open("/proc/self/status"):
        if line.startswith(f): return int(line.split()[1]) / 1024.0
    return 0.0
def tm():
    return None                  # global tracing OFF (v0.10 finding: it multiplies latency)
facts = ((i, (i * 7919) % M) for i in range(n))
out = {"task": task, "n": n}
t0 = time.perf_counter()
FPATH = f"{D}/{'v2' if task=='v2_build' else 'v3'}_{n}.rasm"
def do_build():
    if task == "v2_build":
        from rasm_cim_v8.storage.format import write_v2
        write_v2(((i,(i*7919)%M) for i in range(n)), FPATH, page_recs=2048)
    else:
        from rasm_cim_v9.storage.index import build_v3_base
        build_v3_base(((i,(i*7919)%M) for i in range(n)), FPATH, page_recs=2048)
if task in ("v2_build", "v3_build"):
    t1 = time.perf_counter(); do_build(); wall_clean = time.perf_counter() - t1
    tracemalloc.start(); t2 = time.perf_counter(); do_build()
    tw = time.perf_counter() - t2
    cur, peak = tracemalloc.get_traced_memory()
    c, pk = round(cur/2**20, 2), round(peak/2**20, 2)
    tracemalloc.stop()
    tm = lambda: (c, pk)
    out["wall_untraced_s"] = round(wall_clean, 4)
    out["traced_pass_s"] = round(tw, 4)
    t0 = t1
    out.update(wall_s=round(time.perf_counter()-t0,4), peak_mib=round(vm("VmHWM:"),1),
               rss_after_mib=round(vm("VmRSS:"),1), trace_cur_mib=c, trace_peak_mib=pk,
               file_mib=round(os.path.getsize(FPATH)/2**20, 3))
elif task in ("v2_get", "v3_get", "v3_open"):
    if task == "v2_get":
        from rasm_cim_v8.storage.streaming import open_lazy
        st = open_lazy(f"{D}/v2_{n}.rasm", cache_pages=8)
    elif task == "v3_get":
        from rasm_cim_v9.storage.journal import V3Store
        st = V3Store(f"{D}/v3_{n}.rasm", cache_pages=8)
    else:
        from rasm_cim_v9.storage.journal import V3Store
        st = V3Store(f"{D}/v3_{n}.rasm", cache_pages=8)
        out["open_s"] = round(time.perf_counter()-t0, 4)
    if task != "v3_open":
        rng = random.Random(11); probes = [rng.randrange(n) for _ in range(6000)]
        lat = []
        for k in probes:
            a = time.perf_counter(); st.get(k); lat.append((time.perf_counter()-a)*1e6)
        lat.sort()
        out.update(p50_us=round(lat[len(lat)//2],2), p95_us=round(lat[int(len(lat)*.95)],2),
                   p99_us=round(lat[int(len(lat)*.99)],2),
                   throughput_qps=int(len(probes)/ (sum(lat)/1e6))),
        out["rss_mib"] = round(vm("VmRSS:"),1); out["trace"] = tm()
    st.close()
elif task == "v3_partial":
    from rasm_cim_v9.storage.journal import V3Store
    st = V3Store(f"{D}/v3_{n}.rasm", cache_pages=8)
    r0 = vm("VmRSS:")
    t1 = time.perf_counter(); ri = st.build_residue(); t2 = time.perf_counter()
    rng = random.Random(7); lat = []
    for _ in range(400):
        a = time.perf_counter(); ri.query(rng.randrange(97), 97); lat.append(time.perf_counter()-a)
    lat.sort()
    out.update(residue_build_s=round(t2-t1,4), rss_delta_mib=round(vm("VmRSS:")-r0,1),
               residue_bytes=ri.bytes_resident(),
               p50_us=round(lat[200]*1e6,2), p95_us=round(lat[379]*1e6,2),
               p99_us=round(lat[395]*1e6,2), trace=tm())
    st.close()
elif task == "v3_scan":
    from rasm_cim_v9.storage.journal import V3Store
    st = V3Store(f"{D}/v3_{n}.rasm", cache_pages=8)
    t1 = time.perf_counter(); rows = sum(1 for _ in st.scan()); t2 = time.perf_counter()
    out.update(rows=rows, wall_s=round(t2-t1,4), recs_per_s=int(rows/(t2-t1)), trace=tm())
    st.close()
elif task == "v3_update_fsync":
    from rasm_cim_v9.storage.journal import append_txn
    p = f"{D}/v3_upd_{n}.rasm"
    if not os.path.exists(p):
        import shutil; shutil.copy(f"{D}/v3_{n}.rasm", p)
    lat = []
    for t in range(15):
        a = time.perf_counter(); append_txn(p, [(2_000_000+t, t)], fsync=True)
        lat.append((time.perf_counter()-a)*1e3)
    out.update(update_ms_median=round(statistics.median(lat),4),
               update_ms_p99=round(sorted(lat)[min(len(lat)-1,int(len(lat)*.99))],4),
               update_ms_max=round(max(lat),4), trace=tm())
elif task == "v3_frames_curve":
    from rasm_cim_v9.storage.index import build_v3_base
    from rasm_cim_v9.storage.journal import append_txn, V3Store
    p = f"{D}/v3_fr_{n}.rasm"
    build_v3_base(((i,(i*7919)%M) for i in range(n)), p, page_recs=2048)
    curve = {}
    F = [1, 10, 50, 150, 400, 1000] if n < 1_000_000 else [1, 10, 50, 150, 400]
    made = 0
    for f in F:
        for k in range(made + 1, f + 1):
            append_txn(p, [(5_000_000 + k, k)], fsync=False)
        made = f
        a = time.perf_counter(); st = V3Store(p); dt = time.perf_counter() - a
        curve[f] = round(dt, 4)
        assert st.n_records() == n + f
        st.close()
    out.update(open_wall_s_by_frames=curve, trace=tm())
elif task == "v3_compact":
    import shutil
    from rasm_cim_v9.storage.journal import append_txn, compact, V3Store
    p = f"{D}/v3_cp_{n}.rasm"
    shutil.copy(f"{D}/v3_{n}.rasm", p)
    for k in range(200):
        append_txn(p, [(6_000_000 + k, k)], fsync=False)
    sz0 = os.path.getsize(p)
    a = time.perf_counter(); compact(p); dt = time.perf_counter() - a
    sz1 = os.path.getsize(p)
    with V3Store(p) as st:
        assert st.n_records() == n + 200 and st.get(6_000_123) == ("TRUE", 123)
    out.update(compact_s=round(dt,4), size_ratio=round(sz1/sz0,3), trace=tm())
print(json.dumps(out))
'''


def one(task, n):
    env = {**os.environ}      # build cells self-instrument pass 2 only
    p = subprocess.run([sys.executable, "-c", CHILD, task, str(n), ""],
                       capture_output=True, text=True, timeout=1800, cwd=ROOT, env=env)
    if p.returncode != 0:
        return {"task": task, "n": n, "failure": (p.stderr or "")[-200:]}
    try:
        return json.loads(p.stdout.strip().splitlines()[-1])
    except Exception:
        return {"task": task, "n": n, "failure": "unparsable: " + p.stdout[-200:]}


def med(xs, key):
    v = [x[key] for x in xs if key in x]
    return round(statistics.median(v), 4) if v else None


def rng3(xs, key):
    v = [x[key] for x in xs if key in x]
    return [min(v), max(v)] if v else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--reps", type=int, default=0)
    a = ap.parse_args()
    reps = a.reps or (2 if a.quick else 3)
    ns = [1_000, 20_000] if a.quick else [1_000, 10_000, 100_000, 1_000_000]
    tasks = ["v2_build", "v3_build", "v3_open", "v2_get", "v3_get",
             "v3_partial", "v3_scan", "v3_update_fsync", "v3_frames_curve",
             "v3_compact"]
    matrix = []
    for n in ns:
        one("v2_build", n); one("v3_build", n)          # ensure files (1 quiet pass)
        for t in tasks:
            runs = [one(t, n) for _ in range(reps)]
            bad = [r for r in runs if "failure" in r]
            if bad:
                matrix.append({"task": t, "n": n, "failure": bad[0]["failure"]})
                continue
            row = {"task": t, "n": n, "reps": reps}
            for key in ("wall_untraced_s", "wall_s", "open_s", "residue_build_s", "compact_s", "size_ratio",
                        "update_ms_median", "update_ms_p99", "update_ms_max",
                        "p50_us", "p95_us", "p99_us", "throughput_qps", "recs_per_s",
                        "peak_mib", "rss_after_mib", "rss_mib", "rss_delta_mib",
                        "residue_bytes", "file_mib", "rows"):
                if any(key in r for r in runs):
                    row[key + "_med"] = med(runs, key)
                    if key in ("p95_us", "p99_us", "update_ms_median", "compact_s",
                               "wall_s", "peak_mib"):
                        row[key + "_range"] = rng3(runs, key)
            fr_runs = [r["open_wall_s_by_frames"] for r in runs if "open_wall_s_by_frames" in r]
            if fr_runs:
                keys = sorted({k for fr in fr_runs for k in fr}, key=int)
                row["open_wall_s_by_frames_med"] = {
                    k: med([{"v": fr[k]} for fr in fr_runs if k in fr], "v") for k in keys}
            txs = [r["trace"] for r in runs if r.get("trace")]
            if txs:
                row["trace_med"] = sorted(txs)[len(txs)//2]
            matrix.append(row)
            print({k: v for k, v in row.items() if k not in ("open_wall_s_by_frames_med",)}, flush=True)
    env = {"python": sys.version.split()[0], "cpu": os.cpu_count(),
           "mem_kib": next((l.split()[1] for l in open("/proc/meminfo") if l.startswith("MemTotal")), "?"),
           "tmpfs_tmp": os.popen("df -T /tmp | tail -1 | awk '{print $2}'").read().strip()}
    out = {"env": env, "reps": reps, "ns": ns, "matrix": matrix,
           "method": ("isolated subprocess per cell, median of reps; ranges exposed; "
                      "p99 computed on 6000 (get) / 400 (partial) samples; "
                      "v0.8(v2) and v0.10(v3) cells share identical files/seed/probes")}
    os.makedirs(f"{ROOT}/results_v10", exist_ok=True)
    json.dump(out, open(f"{ROOT}/results_v10/profile_matrix.json", "w"), indent=1)
    # human table
    print("\n" + "=" * 88)
    print(f"{'cell':26} {'n':>9} | {'p50':>8} {'p95':>8} {'p99':>8} | {'peak/RSS MiB':>13}")
    for r in matrix:
        if "failure" in r:
            print(f"{r['task']:26} {r['n']:>9} | FAIL {r['failure'][:50]}")
            continue
        lat = " ".join(f"{r.get(k,'-'):>8}" for k in ("p50_us_med", "p95_us_med", "p99_us_med"))
        mem = r.get("peak_mib_med") or r.get("rss_mib_med") or r.get("rss_delta_mib_med") or "-"
        print(f"{r['task']:26} {r['n']:>9} | {lat} | {str(mem):>13}")
    print("saved results_v10/profile_matrix.json")


if __name__ == "__main__":
    main()
