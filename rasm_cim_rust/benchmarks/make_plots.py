#!/usr/bin/env python3
"""Plots for v0.7: latency ratios, RAM, scaling, debug-vs-release."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/home/user"
OUT = os.path.join(ROOT, "plots_v7")
os.makedirs(OUT, exist_ok=True)

bench = json.load(open(os.path.join(ROOT, "results_v7", "bench_v7.json")))
scal = json.load(open(os.path.join(ROOT, "results_v7", "scaling_v7.json")))

# ---- 1. latency ratios (median Python / median Rust) for primary metric
PRIMARY = {
    "rns_arithmetic": ("encode_decode_s", "encode+decode 2000"),
    "exact_memory": ("build_s", "build 5000 facts"),
    "partial_memory": ("query_s", "97 partial queries"),
    "noisy_retrieval": ("query_s", "200 noisy queries D=256"),
    "graph_traversal": ("query_s", "2000 graph queries"),
    "ca_dense": ("steps_s", "200x1024 dense CA"),
    "ca_sparse": ("steps_s", "200x1024 sparse CA"),
    "synthesis": ("time_s", "BFS synthesis len3"),
    "packed_storage": ("build_s", "PackedStore 50k put"),
    "batch_ops": ("get_s", "batch get 50k (seq)"),
}
labels, ratios, notes = [], [], []
for suite, (key, note) in PRIMARY.items():
    p = bench["suites"][suite]["python"]
    r = bench["suites"][suite]["rust"]
    if key in p and key in r and isinstance(p[key], dict):
        ratios.append(p[key]["median_ms"] / r[key]["median_ms"])
        labels.append(suite)
        notes.append(note)
p95s = []
for suite, (key, _n) in PRIMARY.items():
    p = bench["suites"][suite]["python"]; r = bench["suites"][suite]["rust"]
    p95s.append(p[key]["p95_ms"] / r[key]["p95_ms"] if isinstance(p.get(key), dict) else 0)

fig, ax = plt.subplots(figsize=(11, 6))
x = range(len(labels))
ax.bar([i - 0.2 for i in x], ratios, 0.4, label="median ratio", color="#2a6f97")
ax.bar([i + 0.2 for i in x], p95s, 0.4, label="p95 ratio", color="#a9cfe0")
ax.axhline(1.0, color="red", ls="--", lw=1, label="parity (no speedup)")
ax.set_xticks(list(x))
ax.set_xticklabels(labels, rotation=30, ha="right")
ax.set_ylabel("Python time / Rust time (x)")
ax.set_title("RASM-CIM v0.7: Rust vs Python per-suite speedups (release build, 12 reps median)\n"
             "Same data & algorithms; includes PyO3 call overhead on Rust side")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(OUT, "speedups.png"), dpi=120)

# ---- 2. RAM
fig, ax = plt.subplots(figsize=(11, 5.5))
pkib_p = [bench["suites"][s]["python"].get("peak_rss_kib", 0) for s in PRIMARY]
pkib_r = [bench["suites"][s]["rust"].get("peak_rss_kib", 0) for s in PRIMARY]
labels2 = list(PRIMARY.keys())
ax.bar([i - 0.2 for i in range(len(labels2))], pkib_p, 0.4, label="Python peak RSS", color="#8d6a9f")
ax.bar([i + 0.2 for i in range(len(labels2))], pkib_r, 0.4, label="Rust .so peak RSS", color="#4c3d5c")
ax.set_xticks(range(len(labels2)))
ax.set_xticklabels(labels2, rotation=30, ha="right")
ax.set_ylabel("peak RSS (KiB)")
ax.set_title("Peak process RSS per suite (includes ~30 MiB Python interpreter + numpy import)\n"
             "RAM reduction here is implementation-level, NOT intelligence increase")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(OUT, "ram.png"), dpi=120)

# ---- 3. scaling log-log
ns = [1000, 10000, 100000]
fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
metrics = [("rns_enc_ms", "RNS encode (ms)"), ("mem_build_ms", "Memory build (ms)"),
           ("mem_query_ms", "Memory queries x5000 (ms)")]
for ax, (key, title) in zip(axes, metrics):
    for side, color in (("python", "#c44e52"), ("rust", "#55a868")):
        vals = [scal["scaling"][f"{side}_n{n}"][key] for n in ns]
        ax.plot(ns, vals, "o-", color=color, label=side)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("N (log)"); ax.set_ylabel("ms (log)")
    ax.set_title(title); ax.legend()
axes[0].plot(ns, [ns[0] * n / ns[0] * 0.0032 for n in ns], "k:", lw=0.8, alpha=0.5)
fig.suptitle("Scaling vs N (both sides linear; slope is what matters, not the constant)", fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "scaling.png"), dpi=120)

# ---- 4. debug vs release + parallel vs sequential
dr = scal["debug_vs_release"]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))
keys = ["rns_enc_ms", "synth_ms", "ca200_ms"]
w = 0.35
xs = range(len(keys))
ax1.bar([i - w / 2 for i in xs], [dr["debug"][k] for k in keys], w, label="Rust debug")
ax1.bar([i + w / 2 for i in xs], [dr["release"][k] for k in keys], w, label="Rust release (LTO)")
ax1.set_xticks(list(xs)); ax1.set_xticklabels(keys)
ax1.set_ylabel("ms"); ax1.set_title("Rust Debug vs Release (20k RNS enc 1-rep median x5)")
ax1.legend()
b = bench["suites"]["batch_ops"]["rust"]
ax2.bar(["put seq", "put rayon", "get seq", "get rayon"],
        [b["put_s"]["median_ms"], b["put_par_s"]["median_ms"],
         b["get_s"]["median_ms"], b["get_par_s"]["median_ms"]],
        color=["#2a6f97", "#7fb069", "#2a6f97", "#7fb069"])
ax2.set_ylabel("median ms")
ax2.set_title("Parallelism (2-CPU host): rayon put SLOWER (negative result),\nget faster")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "debug_release_parallel.png"), dpi=120)

print("saved:", os.listdir(OUT))
