#!/usr/bin/env python3
"""RASM-CIM v0.7 — scaling sweep + Debug-vs-Release comparison.

Scaling: N ∈ {1k, 10k, 100k} for RNS encode/decode chain and exact-memory
build (both languages, identical data). One subprocess per (side, N).

Debug-vs-release: 3 key suites run against rasm_cim_ext_debug.so and
rasm_cim_ext.so. Reported, not cherry-picked.
"""
import json, os, statistics, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("RASM_PY_ROOT") or os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

SCALING_N = [1_000, 10_000, 100_000]
M = 15015


def run_scale(side: str, n: int) -> dict:
    code = f'''
import sys, time, json
sys.path.insert(0, {ROOT!r})
M = {M}
N = {n}
vals = [(i * 7919) % M for i in range(N)]
side = "{side}"
out = {{"n": N}}
if side == "python":
    from rasm_cim_v4.rns4 import RNSv4
    r = RNSv4({(3, 5, 7, 11, 13)})
    t0 = time.perf_counter()
    encs = [tuple(int(v) for v in r.encode_unsigned(x)) for x in vals]
    t1 = time.perf_counter()
    decs = [r.decode_unsigned(e) for e in encs]
    t2 = time.perf_counter()
    out["rns_enc_ms"] = (t1 - t0) * 1000
    out["rns_dec_ms"] = (t2 - t1) * 1000
    out["rns_ok"] = decs == vals
    from rasm_cim_v6_v2.memory.memory_system import MemorySystem
    t0 = time.perf_counter()
    m = MemorySystem(D=64, tau=0.5, seed=0)
    for i, v in enumerate(vals):
        m.write_episodic(i + 1, v)
    t1 = time.perf_counter()
    out["mem_build_ms"] = (t1 - t0) * 1000
    t0 = time.perf_counter()
    hits = sum(1 for i in range(0, N, max(1, N // 5000)) if m.query_exact(i + 1)[0] == "TRUE")
    t1 = time.perf_counter()
    out["mem_query_ms"] = (t1 - t0) * 1000
    out["mem_hits"] = hits
    out["mem_len"] = len(m._exact)
else:
    import rasm_cim_ext as rx
    r = rx.RnsPy([3, 5, 7, 11, 13])
    t0 = time.perf_counter()
    encs = [r.encode_unsigned(x) for x in vals]
    t1 = time.perf_counter()
    decs = [r.decode_unsigned(e) for e in encs]
    t2 = time.perf_counter()
    out["rns_enc_ms"] = (t1 - t0) * 1000
    out["rns_dec_ms"] = (t2 - t1) * 1000
    out["rns_ok"] = decs == vals
    t0 = time.perf_counter()
    m = rx.MemoryPy(64, 0.5, 0)
    for i, v in enumerate(vals):
        m.write_episodic(i + 1, v)
    t1 = time.perf_counter()
    out["mem_build_ms"] = (t1 - t0) * 1000
    t0 = time.perf_counter()
    hits = sum(1 for i in range(0, N, max(1, N // 5000)) if m.query_exact(i + 1)[0] == "TRUE")
    t1 = time.perf_counter()
    out["mem_query_ms"] = (t1 - t0) * 1000
    out["mem_hits"] = hits
    out["mem_len"] = m.episodic_len()
print(json.dumps(out))
'''
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT)
    if p.returncode != 0:
        return {"error": p.stderr[-300:]}
    return json.loads(p.stdout)


def run_debug_release() -> dict:
    """Same suites via debug and release ext .so (env RASM_EXT selects)."""
    out = {}
    for mode, pathpre in (("debug", f'sys.path.insert(0, {os.path.join(HERE, "dbg")!r})'),
                          ("release", "pass")):
        code = f'''
import sys, time, json, shutil
{pathpre}
sys.path.insert(0, {ROOT!r})
import rasm_cim_ext as rx
M = {M}
N = 20000
vals = [(i * 7919) % M for i in range(N)]
r = rx.RnsPy([3, 5, 7, 11, 13])
t0 = time.perf_counter()
encs = [r.encode_unsigned(x) for x in vals]
out = {{"rns_enc_ms": (time.perf_counter() - t0) * 1000}}
t0 = time.perf_counter()
s = rx.synth([(0, 6), (1, 9), (-2, 0)], [(2, 12)], [(5, 21)], 3, 50000)
out["synth_ms"] = (time.perf_counter() - t0) * 1000
out["cands"] = s[4]
st = [0] * 1024
for i in range(0, 1024, 2):
    st[i] = 1
t0 = time.perf_counter()
for _ in range(200):
    st = rx.ca_dense_step(st, 90)
out["ca200_ms"] = (time.perf_counter() - t0) * 1000
print(json.dumps(out))
'''
        res = []
        for _ in range(5):
            p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT)
            if p.returncode != 0:
                out[mode] = {"error": p.stderr[-200:]}; break
            res.append(json.loads(p.stdout))
        else:
            out[mode] = {k: round(statistics.median(r[k] for r in res), 4) for k in res[0]}
    return out


def main():
    data = {"scaling": {}, "debug_vs_release": run_debug_release()}
    for n in SCALING_N:
        for side in ("python", "rust"):
            vals_rep = []
            for _ in range(3):
                r = run_scale(side, n)
                if "error" in r:
                    vals_rep = [r]; break
                vals_rep.append(r)
            if "error" in vals_rep[0]:
                data["scaling"][f"{side}_n{n}"] = vals_rep[0]
            else:
                keys = [k for k in vals_rep[0] if isinstance(vals_rep[0][k], float)]
                agg = {k: round(statistics.median(v[k] for v in vals_rep), 4) for k in keys}
                bools = [k for k in vals_rep[0] if isinstance(vals_rep[0][k], bool)]
                for k in bools:
                    agg[k] = all(v[k] for v in vals_rep)
                ints = [k for k in vals_rep[0] if isinstance(vals_rep[0][k], int) and k != "n"]
                for k in ints:
                    agg[k] = vals_rep[0][k]
                agg["n"] = n
                data["scaling"][f"{side}_n{n}"] = agg
        print(f"n={n} done", flush=True)
    with open(os.path.join(ROOT, "results_v7", "scaling_v7.json"), "w") as f:
        json.dump(data, f, indent=1)
    print("saved results_v7/scaling_v7.json")
    for k, v in data["scaling"].items():
        print(k, v)
    print("dbg/rel:", json.dumps(data["debug_vs_release"], indent=1))


if __name__ == "__main__":
    main()
