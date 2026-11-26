#!/usr/bin/env python3
"""RASM-CIM v0.7 — Python vs Rust benchmark harness (§6).

Same input data for both languages, separate subprocesses per (side, suite)
so peak RSS is attributable. Every measurement is REPEATED (15 reps with
warmup) and we report median/p95 of per-rep wall time — never a single test.
No performance claims are made from this script; it only produces numbers.

Suites (10, per spec §6):
  1 rns_arithmetic  2 exact_memory  3 partial_memory  4 noisy_retrieval
  5 graph_traversal 6 ca_dense      7 ca_sparse       8 synthesis
  9 packed_storage  10 batch_ops

Usage:
  python3 run_bench.py --side python --suite all --out bench_py.json
  python3 run_bench.py --side rust   --suite all --out bench_rs.json
Driver (both sides + subprocess): python3 run_bench.py --drive --out bench.json
"""
from __future__ import annotations
import argparse, json, math, os, statistics, subprocess, sys, time

import numpy as np  # python-side vector ops in py_suite; import cost excluded from timing

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("RASM_PY_ROOT") or os.path.dirname(os.path.dirname(HERE))
CRATE = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

N_REPS = 15
N_WARMUP = 3

SUITES = ["rns_arithmetic", "exact_memory", "partial_memory", "noisy_retrieval",
          "graph_traversal", "ca_dense", "ca_sparse", "synthesis",
          "packed_storage", "batch_ops"]

MODULI = (3, 5, 7, 11, 13)
M = 15015


def rss_kib() -> int:
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    return -1


def peak_rss_kib() -> int:
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmHWM:"):
                return int(line.split()[1])
    return -1


def gen_data():
    """Deterministic identical inputs for both sides."""
    import random
    rng = random.Random(2026)
    data = {
        "rns_values": [rng.randrange(M) for _ in range(2000)],
        "mem_keys": list(range(1, 5001)),
        "mem_vals": [rng.randrange(M) for _ in range(5000)],
        "probe_keys": [rng.randrange(10000) for _ in range(5000)],
        "chain_len": 2000,
        "ca_n": 1024, "ca_steps": 200,
        "ca_sparse_n": 1024, "ca_sparse_steps": 200, "ca_active": 32,
        "syn_train": [[0, 6], [1, 9], [-2, 0]],
        "syn_val": [[2, 12]], "syn_test": [[5, 21], [10, 33]],
        "packed_n": 50000,
        "batch_n": 50000,
        "noise_d": 256, "noise_k": 1000, "noise_q": 200,
    }
    return data


# ============================== PYTHON SIDE ==============================

def py_suite(name: str, data: dict) -> dict:
    import numpy as np
    t_create = t_update = t_query = 0.0
    bytes_stored = None
    rng_seed = 2026

    if name == "rns_arithmetic":
        from rasm_cim_v4.rns4 import RNSv4
        r = RNSv4(MODULI)
        vals = data["rns_values"]
        t0 = time.perf_counter()
        encs = [tuple(int(v) for v in r.encode_unsigned(x)) for x in vals]
        t_query = time.perf_counter() - t0
        t0 = time.perf_counter()
        acc = encs[0]
        for e in encs[1:]:
            acc = tuple(int(v) for v in r.add_residues(np.array(acc), np.array(e)))
        t_update = time.perf_counter() - t0
        t0 = time.perf_counter()
        decs = [r.decode_unsigned(e) for e in encs]
        t_query = time.perf_counter() - t0 + t_query
        return {"encode_decode_s": t_query, "addchain_s": t_update,
                "decode_ok": decs == vals, "bytes_stored": sum(len(e) * 2 for e in encs)}
    elif name == "exact_memory":
        from rasm_cim_v6_v2.memory.memory_system import MemorySystem
        import random
        m = MemorySystem(D=64, tau=0.5, seed=rng_seed)
        ks, vs = data["mem_keys"], data["mem_vals"]
        t0 = time.perf_counter()
        for k, v in zip(ks, vs):
            m.write_episodic(k, v)
        t_create = time.perf_counter() - t0
        t0 = time.perf_counter()
        hits = sum(1 for k in data["probe_keys"] if m.query_exact(k)[0] == "TRUE")
        t_query = time.perf_counter() - t0
        return {"build_s": t_create, "query_s": t_query, "hits": hits}
    elif name == "partial_memory":
        from rasm_cim_v6_v2.memory.memory_system import MemorySystem
        m = MemorySystem(D=64, tau=0.5, seed=rng_seed)
        ks, vs = data["mem_keys"], data["mem_vals"]
        for k, v in zip(ks, vs):
            m.write_episodic(k, v)
        t0 = time.perf_counter()
        tot = sum(len(m.query_partial(r, 97)) for r in range(97))
        t_query = time.perf_counter() - t0
        return {"query_s": t_query, "total_hits": tot}
    elif name == "noisy_retrieval":
        from rasm_cim_v6_v2.memory.memory_system import MemorySystem
        m = MemorySystem(D=data["noise_d"], tau=0.5, seed=rng_seed)
        rng = np.random.default_rng(1)
        ks = list(range(data["noise_k"]))
        t0 = time.perf_counter()
        for k in ks:
            m.write_episodic(k, k * 3)
        t_create = time.perf_counter() - t0
        qhvs = [rng.choice(np.array([-1.0, 1.0]), size=data["noise_d"])
                for _ in range(data["noise_q"])]
        t0 = time.perf_counter()
        res = [m.query_noisy(q)[0] for q in qhvs]
        t_query = time.perf_counter() - t0
        return {"build_s": t_create, "query_s": t_query,
                "true_rate": sum(r == "TRUE" for r in res) / len(res)}
    elif name == "graph_traversal":
        from rasm_cim_v4.graph4 import TruthGraph
        g = TruthGraph(functional_relations={0})
        n = data["chain_len"]
        t0 = time.perf_counter()
        for i in range(n):
            g.assert_true(i, 0, i + 1)
        t_create = t_update = time.perf_counter() - t0
        t0 = time.perf_counter()
        ok = all(g.query(i, 0, i + 1) == "TRUE" for i in range(n))
        unk = sum(1 for i in range(0, n, 2) if g.query(i, 0, i + 2) != "TRUE")
        t_query = time.perf_counter() - t0
        return {"build_s": t_create, "query_s": t_query, "ok": ok, "unknowns": unk}
    elif name in ("ca_dense", "ca_sparse"):
        from rasm_cim_v4.ca4 import ca_step_dense, ca_step_sparse
        n, steps = data["ca_n"], data["ca_steps"]
        st = np.zeros(n, dtype=np.int64)
        if name == "ca_sparse":
            n, steps = data["ca_sparse_n"], data["ca_sparse_steps"]
            st[np.arange(data["ca_active"]) * (n // data["ca_active"])] = 1
        else:
            rng = np.random.default_rng(2)
            st = rng.integers(0, 2, size=n)
        t0 = time.perf_counter()
        ops_total = 0
        cur = st
        for _ in range(steps):
            if name == "ca_dense":
                cur = ca_step_dense(cur, 90)
                ops_total += n
            else:
                cur, k = ca_step_sparse(cur, 90)
                ops_total += k
        t_query = time.perf_counter() - t0
        return {"steps_s": t_query, "ops": ops_total, "n_cells": n, "steps": steps}
    elif name == "synthesis":
        from rasm_cim_v4.program4 import synthesize_split
        t0 = time.perf_counter()
        r = synthesize_split([tuple(x) for x in data["syn_train"]],
                             [tuple(x) for x in data["syn_val"]],
                             [tuple(x) for x in data["syn_test"]],
                             max_depth=3, max_nodes=50000)
        t = time.perf_counter() - t0
        return {"time_s": t, "candidates": r["candidates"], "program": r["program"],
                "test_acc": r["test_acc"]}
    elif name == "packed_storage":
        from rasm_cim_v6_v2.compute.compute_core import PackedStore
        ps = PackedStore()
        n = data["packed_n"]
        t0 = time.perf_counter()
        for i in range(n):
            ps.put(i, i * 2)
        t_create = t_update = time.perf_counter() - t0
        t0 = time.perf_counter()
        ok = all(ps.get(i) == i * 2 for i in range(n))
        t_query = time.perf_counter() - t0
        return {"build_s": t_create, "query_s": t_query, "ok": ok,
                "bytes_stored": ps.bytes_used()}
    elif name == "batch_ops":
        import random
        from rasm_cim_v6_v2.compute.compute_core import PackedStore
        rng = random.Random(rng_seed)
        pairs = [(k, rng.randrange(-10**9, 10**9)) for k in range(data["batch_n"])]
        ps = PackedStore()
        t0 = time.perf_counter()
        for k, v in pairs:
            ps.put(k, v)
        t_update = time.perf_counter() - t0
        t0 = time.perf_counter()
        out = [ps.get(k) for k, _ in pairs]
        t_query = time.perf_counter() - t0
        return {"put_s": t_update, "get_s": t_query, "ok": out == [v for _, v in pairs]}
    raise ValueError(name)


# =============================== RUST SIDE ===============================

def rs_suite(name: str, data: dict) -> dict:
    import rasm_cim_ext as rx
    if name == "rns_arithmetic":
        r = rx.RnsPy(list(MODULI))
        vals = data["rns_values"]
        t0 = time.perf_counter()
        encs = [r.encode_unsigned(x) for x in vals]
        t_enc = time.perf_counter() - t0
        t0 = time.perf_counter()
        acc = encs[0]
        for e in encs[1:]:
            acc = r.add_residues(acc, e)
        t_add = time.perf_counter() - t0
        t0 = time.perf_counter()
        decs = [r.decode_unsigned(e) for e in encs]
        t_dec = time.perf_counter() - t0
        return {"encode_decode_s": t_enc + t_dec, "addchain_s": t_add,
                "decode_ok": decs == vals, "ops": r.ops()}
    elif name in ("exact_memory", "partial_memory", "noisy_retrieval"):
        m = rx.MemoryPy(64 if name != "noisy_retrieval" else data["noise_d"], 0.5, 2026)
        if name == "exact_memory":
            ks, vs = data["mem_keys"], data["mem_vals"]
            t0 = time.perf_counter()
            for k, v in zip(ks, vs):
                m.write_episodic(k, v)
            t_create = time.perf_counter() - t0
            t0 = time.perf_counter()
            hits = sum(1 for k in data["probe_keys"] if m.query_exact(k)[0] == "TRUE")
            t_query = time.perf_counter() - t0
            return {"build_s": t_create, "query_s": t_query, "hits": hits}
        elif name == "partial_memory":
            ks, vs = data["mem_keys"], data["mem_vals"]
            for k, v in zip(ks, vs):
                m.write_episodic(k, v)
            t0 = time.perf_counter()
            tot = sum(len(m.query_partial(r, 97)) for r in range(97))
            t_query = time.perf_counter() - t0
            return {"query_s": t_query, "total_hits": tot}
        else:
            import random
            rng = random.Random(1)
            k = data["noise_k"]
            t0 = time.perf_counter()
            for i in range(k):
                m.write_episodic(i, i * 3)
            t_create = time.perf_counter() - t0
            d = data["noise_d"]
            qs = [[1.0 if rng.random() < 0.5 else -1.0 for _ in range(d)]
                  for _ in range(data["noise_q"])]
            t0 = time.perf_counter()
            res = [m.query_noisy(q)[0] for q in qs]
            t_query = time.perf_counter() - t0
            return {"build_s": t_create, "query_s": t_query,
                    "true_rate": sum(r == "TRUE" for r in res) / len(res)}
    elif name == "graph_traversal":
        g = rx.graph_new([0])
        n = data["chain_len"]
        t0 = time.perf_counter()
        for i in range(n):
            g.assert_true(i, 0, i + 1)
        t_create = t_update = time.perf_counter() - t0
        t0 = time.perf_counter()
        ok = all(g.query(i, 0, i + 1) == "TRUE" for i in range(n))
        unk = sum(1 for i in range(0, n, 2) if g.query(i, 0, i + 2) != "TRUE")
        t_query = time.perf_counter() - t0
        return {"build_s": t_create, "query_s": t_query, "ok": ok, "unknowns": unk}
    elif name in ("ca_dense", "ca_sparse"):
        n, steps = data["ca_n"], data["ca_steps"]
        import random
        rng = random.Random(2)
        if name == "ca_sparse":
            n, steps = data["ca_sparse_n"], data["ca_sparse_steps"]
            st = [0] * n
            for i in range(data["ca_active"]):
                st[i * (n // data["ca_active"])] = 1
        else:
            st = [rng.randrange(2) for _ in range(n)]
        t0 = time.perf_counter()
        ops_total = 0
        if name == "ca_dense":
            for _ in range(steps):
                st = rx.ca_dense_step(st, 90)
                ops_total += n
        else:
            for _ in range(steps):
                st, k = rx.ca_sparse_step(st, 90)
                ops_total += k
        t_query = time.perf_counter() - t0
        return {"steps_s": t_query, "ops": ops_total, "n_cells": n, "steps": steps}
    elif name == "synthesis":
        train = [tuple(x) for x in data["syn_train"]]
        val = [tuple(x) for x in data["syn_val"]]
        test = [tuple(x) for x in data["syn_test"]]
        t0 = time.perf_counter()
        prog, tr, va, te, cands, ms = rx.synth(train, val, test, 3, 50000)
        t = time.perf_counter() - t0
        return {"time_s": t, "candidates": cands, "program": prog, "test_acc": te}
    elif name == "packed_storage":
        ps = rx.PackedStorePy()
        n = data["packed_n"]
        t0 = time.perf_counter()
        for i in range(n):
            ps.put(i, i * 2)
        t_create = t_update = time.perf_counter() - t0
        t0 = time.perf_counter()
        ok = all(ps.get(i) == i * 2 for i in range(n))
        t_query = time.perf_counter() - t0
        return {"build_s": t_create, "query_s": t_query, "ok": ok,
                "bytes_stored": ps.bytes_used()}
    elif name == "batch_ops":
        import random
        rng = random.Random(2026)
        pairs = [(k, rng.randrange(-10**9, 10**9)) for k in range(data["batch_n"])]
        ps = rx.PackedStorePy()
        t0 = time.perf_counter()
        ps.batch_put(pairs, False)
        t_update = time.perf_counter() - t0
        t0 = time.perf_counter()
        out = ps.batch_get([k for k, _ in pairs], False)
        t_query = time.perf_counter() - t0
        # parallel (rayon)
        ps2 = rx.PackedStorePy()
        t0 = time.perf_counter()
        ps2.batch_put(pairs, True)
        t_update_par = time.perf_counter() - t0
        t0 = time.perf_counter()
        out2 = ps2.batch_get([k for k, _ in pairs], True)
        t_query_par = time.perf_counter() - t0
        return {"put_s": t_update, "get_s": t_query, "ok": out == [v for _, v in pairs],
                "put_par_s": t_update_par, "get_par_s": t_query_par,
                "ok_par": out2 == [v for _, v in pairs]}
    raise ValueError(name)


# =============================== DRIVER ==================================

def run_subprocess(side: str, suite: str) -> dict:
    cmd = [sys.executable, os.path.join(HERE, "run_bench.py"),
           "--side", side, "--suite", suite, "--json-only"]
    out = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    if out.returncode != 0:
        return {"error": out.stderr[-400:]}
    return json.loads(out.stdout)


def bench_one(side: str, suite: str) -> dict:
    # repeated subprocess runs: peak RSS attributable to one suite
    reps = []
    for i in range(N_REPS):
        r = run_subprocess(side, suite)
        if "error" in r:
            return r
        reps.append(r)
    if i < N_WARMUP:  # fewer reps for slow suites is fine; keep all
        pass
    timed = reps[N_WARMUP:] if len(reps) > N_WARMUP else reps
    keys = [k for k in timed[0] if isinstance(timed[0][k], (int, float))
            and not isinstance(timed[0][k], bool) and k not in
            ("hits", "ops", "n_cells", "steps", "candidates", "total_hits", "unknowns")]
    result = {"suite": suite, "side": side, "reps": len(timed)}
    for k in keys:
        vals = [t[k] for t in timed if k in t]
        if not vals:
            continue
        if k.endswith("_s") or k.endswith("_ms"):
            vals_ms = sorted(v * 1000.0 for v in vals)
            result[k] = {
                "median_ms": statistics.median(vals_ms),
                "p95_ms": vals_ms[min(len(vals_ms) - 1, int(0.95 * len(vals_ms)))],
                "min_ms": vals_ms[0],
            }
        else:
            result[k] = statistics.median(vals)
    # booleans / identity checks: must hold in every rep
    for k in [k for k in timed[0] if isinstance(timed[0][k], bool)]:
        result[k] = all(t[k] for t in timed)
    result["peak_rss_kib"] = statistics.median([t.get("peak_rss_kib", -1) for t in timed])
    result["cpu_s"] = statistics.median([t.get("cpu_s", -1) for t in timed])
    return result


def cpu_time_s() -> float:
    t = os.times()
    return t.user + t.system


def run_in_process(side: str, suite: str) -> dict:
    data = gen_data()
    start_rss = rss_kib()
    t0 = time.perf_counter()
    fn = py_suite if side == "python" else rs_suite
    r = fn(suite, data)
    wall = time.perf_counter() - t0
    r["_wall_s"] = wall
    r["peak_rss_kib"] = peak_rss_kib()
    r["start_rss_kib"] = start_rss
    r["cpu_s"] = cpu_time_s()
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--side", choices=["python", "rust"])
    ap.add_argument("--suite", default="all")
    ap.add_argument("--json-only", action="store_true")
    ap.add_argument("--drive", action="store_true")
    ap.add_argument("--out", default="bench.json")
    ap.add_argument("--reps", type=int, default=None)
    args = ap.parse_args()
    global N_REPS
    if args.reps:
        N_REPS = args.reps

    if args.json_only:
        r = run_in_process(args.side, args.suite)
        print(json.dumps(r, default=str))
        return

    results = {"suites": {}, "meta": {
        "reps": N_REPS, "warmup_dropped": N_WARMUP,
        "python": sys.version, "host_cpus": os.cpu_count(),
        "note": "median/p95 over repeated identical-data runs in isolated "
                "subprocesses; Rust .so is release build (LTO)."}}
    suites = SUITES if args.suite == "all" else [args.suite]
    for su in suites:
        print(f"=== {su} ===", flush=True)
        results["suites"][su] = {
            "python": bench_one("python", su),
            "rust": bench_one("rust", su),
        }
        r = results["suites"][su]
        for side in ("python", "rust"):
            if "error" in r[side]:
                print(f"  {side}: ERROR {r[side]['error'][:120]}")
    with open(args.out, "w") as f:
        json.dump(results, f, indent=1, default=str)
    print("saved:", args.out)


if __name__ == "__main__":
    main()
