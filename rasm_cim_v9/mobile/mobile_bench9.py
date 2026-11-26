"""v0.9 M4 — Termux device harness: v0.8 engine vs v0.9 engine A/B on ARM64.

RUN ON DEVICE (Termux). Never extrapolate from host. Results go to
$HOME/rasm_v9_device_results/ on the phone.

r3 fix: EVERY ARM RUNS IN ITS OWN SUBPROCESS. The r2 harness ran arms
sequentially in one process, so VmRSS/VmHWM were cross-contaminated
(Python/glibc arenas are never returned to the OS; VmHWM never decreases) —
device runs showed all arms at the same ~54.8/68.8 MiB and the streaming RSS
win could not be judged. r3 measures each arm from a fresh process.

Arms (identical facts/keys/queries/seed/dim/bits for every arm — spec):
  v08_legacy_hv          MobileStreamMemory + open_hv()            (reference)
  v08_stream_hv          MobileStreamMemory + open_hv_stream(auto)
  v09_v3_store           V3Store over v3 base                       queries
  v09_v3_journal_updates fresh v3 build + N update txns, fsync on FLASH
  v09_mem9               MobileMemory9 (build + hv stream)          full stack
Missing temperature sysfs is NOT a failure (fields stay null).
"""
from __future__ import annotations
import json
import os
import sys
import time

sys.path.insert(0, os.environ.get("RASM_PKG", os.path.expanduser("~/rasm_cim_v9_pkg")))
M = 15015
SEED = 20261125
ARMS = ["v08_legacy_hv", "v08_stream_hv", "v09_v3_store",
        "v09_v3_journal_updates", "v09_mem9"]


def vm(field):
    try:
        for line in open("/proc/self/status"):
            if line.startswith(field):
                return int(line.split()[1]) / 1024.0
    except OSError:
        pass
    return None


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(len(xs) * q))]


def readings():
    out = {}
    for z in ("/sys/class/thermal/thermal_zone0/temp",
              "/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq"):
        key = "temp" if "thermal" in z else "scaling_cur_freq"
        try:
            out[key] = int(open(z).read().strip())
        except (OSError, ValueError):
            out[key] = None      # absent sensor is NOT a failure
    return out


# ------------------------------------------------------------------ one arm
def arm(name, n, dim, bits, updates):
    import random
    rng = random.Random(SEED)
    facts = [(i, (i * 7919) % M) for i in range(n)]
    qkeys = [rng.randrange(n) for _ in range(600)]
    from rasm_cim_v8.hv.bitpack import gen_query_with_flips
    qvecs = [gen_query_with_flips(rng.randrange(n), dim, bits, 0.10, SEED + i)[0]
             for i in range(50)]
    d = os.environ["RASM_DEVDIR"]
    p2, p3 = f"{d}/v2.rasm", f"{d}/v3.rasm"
    out = {}
    if name == "v08_legacy_hv":
        from rasm_cim_v8.memory.stream_memory import MobileStreamMemory
        m = MobileStreamMemory(p2, tiers=("stream",)).open()
        m.open_hv(dim, bits)
        qs = []
        for k in qkeys:
            t0 = time.perf_counter(); m.query_exact(k); qs.append(time.perf_counter() - t0)
        out = {"q_p50_ms": round(pct(qs, .5) * 1e3, 3),
               "q_p95_ms": round(pct(qs, .95) * 1e3, 3),
               "picks": list(m.query_noisy(fv, 0.5) for fv in qvecs)[:20]}
        m.close()
    elif name == "v08_stream_hv":
        from rasm_cim_v8.memory.stream_memory import MobileStreamMemory
        m = MobileStreamMemory(p2, tiers=("stream",)).open()
        m.open_hv_stream(dim, bits, method="auto")
        qs = []
        for k in qkeys:
            t0 = time.perf_counter(); m.query_exact(k); qs.append(time.perf_counter() - t0)
        out = {"q_p50_ms": round(pct(qs, .5) * 1e3, 3),
               "q_p95_ms": round(pct(qs, .95) * 1e3, 3),
               "picks": list(m.query_noisy(fv, 0.5) for fv in qvecs)[:20]}
        m.close()
    elif name == "v09_v3_store":
        from rasm_cim_v9.storage.journal import V3Store
        st = V3Store(p3, cache_pages=16)
        qs = []
        for k in qkeys:
            t0 = time.perf_counter(); st.get(k); qs.append(time.perf_counter() - t0)
        out = {"q_p50_ms": round(pct(qs, .5) * 1e3, 3),
               "q_p95_ms": round(pct(qs, .95) * 1e3, 3)}
        st.close()
    elif name == "v09_v3_journal_updates":
        from rasm_cim_v9.storage.index import build_v3_base
        from rasm_cim_v9.storage.journal import V3Store, append_txn
        p3u = f"{d}/v3_upd.rasm"
        if os.path.exists(p3u):
            os.remove(p3u)
        tb = time.perf_counter()
        build_v3_base(facts, p3u, page_recs=2048)          # fresh each run
        out["base_build_s"] = round(time.perf_counter() - tb, 4)
        lat = []
        for t in range(updates):
            a = time.perf_counter()
            append_txn(p3u, [(n + t, t)], fsync=True)       # FLASH fsync — the point
            lat.append((time.perf_counter() - a) * 1e3)
        st = V3Store(p3u, cache_pages=16)
        assert st.n_records() == n + updates                # all txns committed
        st.close()
        out.update({"upd_p50_ms": round(pct(lat, .5), 3),
                    "upd_p95_ms": round(pct(lat, .95), 3),
                    "TARGET": "<=5 ms",
                    "verdict": "PASS" if pct(lat, .5) <= 5 else "FAIL"})
    elif name == "v09_mem9":
        from rasm_cim_v9.memory.store9 import MobileMemory9
        p9 = f"{d}/mem9.rasm"
        tb = time.perf_counter()
        m = MobileMemory9.build(p9, facts, page_recs=2048)
        out["build_s"] = round(time.perf_counter() - tb, 4)
        m.open_hv_stream(dim, bits)
        qs = []
        for k in qkeys:
            t0 = time.perf_counter(); m.query_exact(k); qs.append(time.perf_counter() - t0)
        out.update({"q_p50_ms": round(pct(qs, .5) * 1e3, 3),
                    "q_p95_ms": round(pct(qs, .95) * 1e3, 3),
                    "picks": list(m.query_noisy(fv, 0.5) for fv in qvecs)[:20],
                    "sizes": m.sizes()})
    out["rss_mib"] = round(vm("VmRSS:") or 0, 1)
    out["peak_mib"] = round(vm("VmHWM:") or 0, 1)
    return out


# ------------------------------------------------------------------ parent
def main(n=20000, dim=256, bits=7, updates=200):
    import subprocess
    base = os.path.expanduser("~/rasm_v9_device_results")
    os.makedirs(base, exist_ok=True)
    d = os.path.expanduser(f"~/rasm_v9_dev_{n}")
    os.makedirs(d, exist_ok=True)
    # shared inputs built ONCE here so every arm process reads identical files
    t0 = time.perf_counter()
    import random
    rng = random.Random(SEED)
    facts = [(i, (i * 7919) % M) for i in range(n)]
    from rasm_cim_v8.storage.format import write_v2
    from rasm_cim_v9.storage.index import build_v3_base
    p2, p3 = f"{d}/v2.rasm", f"{d}/v3.rasm"
    if not os.path.exists(p2):
        write_v2(facts, p2, page_recs=2048)
    if not os.path.exists(p3):
        build_v3_base(facts, p3, page_recs=2048)
    shared_s = round(time.perf_counter() - t0, 3)
    out = {"n": n, "dim": dim, "bits": bits, "seed": SEED,
           "harness": "r3-isolated-arms", "shared_inputs_s": shared_s,
           "updates": updates, "host": os.uname().release,
           "device_readings": readings()}
    env = {**os.environ, "RASM_DEVDIR": d,
           "PYTHONPATH": os.environ.get("PYTHONPATH", "")}
    for a in ARMS:
        t0 = time.perf_counter()
        r = subprocess.run([sys.executable, os.path.abspath(__file__), "--arm", a,
                            str(n), str(dim), str(bits), str(updates)],
                           capture_output=True, text=True, env=env)
        wall = round(time.perf_counter() - t0, 3)
        if r.returncode != 0:
            out[a] = {"error": (r.stderr or r.stdout or "")[-300:], "wall_s": wall}
        else:
            out[a] = json.loads(r.stdout.strip().splitlines()[-1])
            out[a]["wall_s"] = wall
        print(a, "->", {k: v for k, v in out[a].items() if k != "picks"}, flush=True)
    a_, b_, c_ = (out.get("v08_legacy_hv", {}), out.get("v08_stream_hv", {}),
                  out.get("v09_mem9", {}))
    out["mismatch"] = {
        "stream_vs_legacy": (sum(1 for x, y in zip(a_["picks"], b_["picks"]) if x != y)
                             if "picks" in a_ and "picks" in b_ else None),
        "mem9_vs_legacy": (sum(1 for x, y in zip(a_["picks"], c_["picks"]) if x != y)
                           if "picks" in a_ and "picks" in c_ else None)}
    f = f"{base}/v09_ab_n{n}_{int(time.time())}.json"
    json.dump(out, open(f, "w"), indent=1)
    print("saved", f)


if __name__ == "__main__":
    if "--arm" in sys.argv:
        i = sys.argv.index("--arm")
        print(json.dumps(arm(sys.argv[i + 1], int(sys.argv[i + 2]),
                             int(sys.argv[i + 3]), int(sys.argv[i + 4]),
                             int(sys.argv[i + 5]))))
        sys.exit(0)
    ns = [int(x) for x in sys.argv[1:] if x.isdigit()] or [20000]
    for n in ns:
        main(n=n)
