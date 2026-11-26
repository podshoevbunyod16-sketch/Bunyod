"""v0.8 plots — RAM + latency (+ p95-target gap, method comparison, concurrency).
All data read from results_v8/*.json (measured artifacts; no synthetic points)."""
from __future__ import annotations
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/home/user/plots_v8"
import os
os.makedirs(OUT, exist_ok=True)
m6 = json.load(open("/home/user/results_v8/m6_scaling.json"))
m2 = json.load(open("/home/user/results_v8/m2_streaming.json"))
m5 = json.load(open("/home/user/results_v8/m5_hv.json"))
mc = json.load(open("/home/user/results_v8/m3_m7_concurrency.json"))

# 1) peak RSS by scale × condition (failures plotted as cap-hit marker)
conds = ["stream", "stream+hv", "dict", "v06mem"]
labels, series = [1_000, 10_000, 100_000, 1_000_000], {c: [] for c in conds}
marks = {c: [] for c in conds}
for c in conds:
    for n in labels:
        r = m6["runs"].get(f"{n}:{c}", {})
        if "failure" in r:
            series[c].append(r.get("peak_rss_mib") or 0); marks[c].append("f")
        else:
            v = r.get("rss_at_query_mib") or r.get("peak_rss_mib", 0)
            series[c].append(v); marks[c].append("")
plt.figure(figsize=(7.2, 4.4))
w = 0.2
for i, c in enumerate(conds):
    xs = [j + (i - 1.5) * w for j in range(len(labels))]
    plt.bar(xs, series[c], w, label=c)
    for j, (x, m_) in enumerate(zip(xs, marks[c])):
        if m_ == "f":
            plt.text(x, series[c][j] + 8, "OOM", ha="center", fontsize=7, color="crimson")
plt.axhline(40, ls="--", c="k", lw=1); plt.text(2.9, 44, "mobile target 40 MiB", fontsize=8)
plt.xticks(range(len(labels)), [f"{n:,}" for n in labels])
plt.xlabel("records"); plt.ylabel("peak RSS (MiB)"); plt.yscale("log")
plt.title("v0.8 M6 — RSS at query time by scale × mode (VmRSS; OOM = MemoryError at 1.5 GiB cap)")
plt.legend(fontsize=8); plt.tight_layout(); plt.savefig(f"{OUT}/ram_by_scale.png"); plt.close()

# 2) exact-query latency p50/p95 vs scale (streaming modes)
plt.figure(figsize=(7.2, 4.4))
for c, st in (("stream", "-o"), ("stream+hv", "-s"), ("dict", "-^")):
    p50, p95 = [], []
    for n in labels:
        r = m6["runs"].get(f"{n}:{c}", {})
        e = r.get("exact")
        p50.append(e["p50_ms"] if e else float("nan"))
        p95.append(e["p95_ms"] if e else float("nan"))
    plt.plot(labels, p50, st, label=f"{c} p50")
    plt.plot(labels, p95, st.replace("-", "--"), label=f"{c} p95")
plt.xscale("log"); plt.yscale("log")
plt.xlabel("records"); plt.ylabel("exact query latency (ms)")
plt.title("M6 — exact-query latency at ACTUAL scales (no extrapolation)")
plt.grid(alpha=.3, which="both"); plt.legend(fontsize=8); plt.tight_layout()
plt.savefig(f"{OUT}/latency_by_scale.png"); plt.close()

# 3) M2 p95-vs-target gap (the honest FAIL)
modes = [f"{e['mode']}·cp{e['cache_pages']}" for e in m2["modes"]]
ratios = [e["p95_ratio_vs_dict"] for e in m2["modes"]]
plt.figure(figsize=(7.2, 4.0))
plt.bar(range(len(modes)), ratios, color="#c44")
plt.axhline(2, c="k", ls="--", lw=1); plt.text(0.1, 3.2, "target ≤2×", fontsize=9)
plt.xticks(range(len(modes)), modes, rotation=18, fontsize=7)
plt.yscale("log"); plt.ylabel("p95 ratio vs in-RAM dict baseline")
plt.title("M2 — p95 target ≤2× : FAILED at every mode (recorded, not hidden)")
plt.tight_layout(); plt.savefig(f"{OUT}/m2_p95_vs_target.png"); plt.close()

# 4) M5 method comparison (p50 per query, log)
plt.figure(figsize=(7.4, 4.4))
meths = list(m5["runs"]["1000"]["methods"].keys())
for n, mk in zip(labels, ("x", "o", "s")):
    if str(n) not in m5["runs"]:
        continue
    ys = [m5["runs"][str(n)]["methods"][m]["query_lat"]["p50_ms"] for m in meths]
    plt.plot(range(len(meths)), ys, mk + "-", label=f"n={n:,}")
plt.yscale("log"); plt.xticks(range(len(meths)), meths, rotation=22, fontsize=8)
plt.ylabel("p50 per query (ms)"); plt.grid(alpha=.3)
plt.axhline(0, c="none")
plt.title("M5 — bipolar-HV sim: scalar vs bitpack vs numpy vs numba (all correct)")
plt.legend(fontsize=8); plt.tight_layout(); plt.savefig(f"{OUT}/m5_methods.png"); plt.close()

# 5) concurrency speedups
scen = [k for k in ("S1_cpu_python", "S2_cpu_numpy", "S3_io_files", "S4_stream_fanout") if k in mc]
thr = [mc[k].get("threads_speedup", 0) for k in scen]
prc = [mc[k].get("proc_speedup", 0) for k in scen]
x = range(len(scen))
plt.figure(figsize=(7.2, 4.0))
plt.bar([i - .19 for i in x], thr, .36, label="threads×4 / GIL-released")
plt.bar([i + .19 for i in x], prc, .36, label="processes×2")
plt.axhline(1, c="k", lw=1)
plt.xticks(list(x), [s.split("_")[0] + " " + s.split("_")[1] for s in scen], fontsize=8)
plt.ylabel("speedup vs sequential (×)"); plt.legend(fontsize=8)
plt.title("M3/M7 — measured speedups (2-CPU host; >1 only where claimed)")
plt.tight_layout(); plt.savefig(f"{OUT}/m3_m7_speedups.png"); plt.close()
print("plots:", sorted(os.listdir(OUT)))
