"""M6 — actual scaling at 1k / 10k / 100k / 1M. NO extrapolation:
every N is really run (or really fails, and the failure is recorded).

Conditions per N (subprocess-isolated for honest RSS):
  stream   : MobileStreamMemory tiers=("stream",)   (pread LazyStore)
  stream+hv: + in-RAM packed-HV cache (open_hv)
  dict     : tiers=("dict",) full in-RAM (the ≤2× p95 reference)
  v06mem   : v0.6 MemorySystem in-RAM float HV D=256 — EXPECTED to OOM at
             1M on a 1.9 GiB host; run under resource.setrlimit and record
             the actual failure (this is a result, not an excuse).

Metrics per condition: build/write, reopen(open_s), update(rebuild with one
extra record), exact p50/p95 (5000 random probes), partial (per-modulus
latency + candidate count = ops proxy), residue index size (when built),
file bytes, resident index bytes, serialize(same as build)/deserialize.
"""
from __future__ import annotations
import json, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

N_LIST = [1_000, 10_000, 100_000, 1_000_000]
M = 15015
OUT_DIR = "/tmp/rasm_v8_m6"
RESULTS = "/home/user/results_v8/m6_scaling.json"


def worker(n: int, cond: str, memcap_mib: int) -> dict:
    import resource
    if memcap_mib > 0:
        lim = memcap_mib * 2**20
        resource.setrlimit(resource.RLIMIT_AS, (lim, lim))
    out: dict = {"n": n, "cond": cond, "memcap_mib": memcap_mib}
    try:
        os.makedirs(OUT_DIR, exist_ok=True)
        path = f"{OUT_DIR}/m6_{n}_{cond}.rasm"
        facts = ((i, (i * 7919) % M) for i in range(n))
        if cond in ("stream", "stream+hv", "dict"):
            from rasm_cim_v8.storage import format as F
            t0 = time.perf_counter()
            meta = F.write_v2(facts, path, page_recs=2048)
            out["build_s"] = round(time.perf_counter() - t0, 3)
            out["file_bytes"] = meta["bytes"]
            tiers = {"stream": ("stream",), "stream+hv": ("stream",), "dict": ("dict",)}[cond]
            from rasm_cim_v8.memory.stream_memory import MobileStreamMemory
            t0 = time.perf_counter()
            m = MobileStreamMemory(path, tiers=tiers).open()
            out["open_s"] = round(time.perf_counter() - t0, 4)
            if cond == "stream+hv":
                t0 = time.perf_counter()
                m.open_hv(256, 7)
                out["hv_open_s"] = round(time.perf_counter() - t0, 3)
                out["hv_ram_mib"] = round(m.hv_bytes_estimate() / 2**20, 1)
                import random as _r
                rg = _r.Random(3)
                lat = []
                hits = 0
                from rasm_cim_v8.hv.bitpack import gen_query_with_flips
                qn = min(200, n)
                for qi in rg.sample(range(n), qn):
                    q, _ = gen_query_with_flips(qi, 256, 7, 0.10, qi)
                    a = time.perf_counter()
                    st, v, _c = m.query_noisy(q, 0.5)
                    lat.append(time.perf_counter() - a)
                    hits += (st == "TRUE" and v == (qi * 7919) % M)
                from rasm_cim_v8.bench.util import stats_ms
                out["noisy"] = stats_ms(lat)
                out["noisy_hits"] = f"{hits}/{qn}"
            import random
            rng = random.Random(11)
            probes = [rng.randrange(n) for _ in range(5000)] + list(range(min(10, n)))
            lat = []
            acc = True
            for k in probes[:5000]:
                a = time.perf_counter()
                st, v, _c = m.query_exact(k)
                lat.append(time.perf_counter() - a)
                acc &= (st == "TRUE" and v == (k * 7919) % M)
            unk = all(m.query_exact(k + n)[0] == "UNKNOWN" for k in probes[:50])
            out["exact_acc"] = acc and unk
            from rasm_cim_v8.bench.util import stats_ms
            out["exact"] = stats_ms(lat)
            def _vmrss():
                try:
                    for line in open("/proc/self/status"):
                        if line.startswith("VmRSS:"):
                            return round(int(line.split()[1]) / 1024.0, 1)
                except OSError:
                    return None
            out["rss_at_query_mib"] = _vmrss()   # excludes update materialization
            # update: append one record -> rebuild whole file (no in-place
            # append in v2; documented limit; this IS the update cost)
            t0 = time.perf_counter()
            with open(path, "rb") as _f:
                pass
            extra = list(m.scan())
            extra.append((n, 12345))
            F.write_v2(extra, path + ".upd", page_recs=2048)
            out["update_rebuild_s"] = round(time.perf_counter() - t0, 3)
            os.remove(path + ".upd")
            # partial
            lat = []
            cand = 0
            for r in (7, 42, 90):
                a = time.perf_counter()
                got = m.query_partial(r, 97)
                lat.append(time.perf_counter() - a)
                cand = max(cand, len(got))
            out["partial"] = stats_ms(lat)
            out["partial_candidates_max"] = cand
            out["resident_index_bytes"] = m.sizes().get("resident_index_bytes", 0)
            m.close()
            # deserialize timing = re-open from disk (already open_s) + scan
            t0 = time.perf_counter()
            m2 = MobileStreamMemory(path, tiers=("dict",)).open()
            out["deserialize_full_s"] = round(time.perf_counter() - t0, 3)
            m2.close()
        elif cond == "v06mem":
            from rasm_cim_v6_v2.memory.memory_system import MemorySystem
            t0 = time.perf_counter()
            mm = MemorySystem(D=256, tau=0.5, seed=0)
            for i in range(n):
                mm.write_episodic(i + 1, (i * 7919) % M)
            out["build_s"] = round(time.perf_counter() - t0, 3)
            import random
            rng = random.Random(11)
            lat = []
            for _ in range(2000):
                k = rng.randrange(n) + 1
                a = time.perf_counter()
                st, v, _ = mm.query_exact(k)
                lat.append(time.perf_counter() - a)
            from rasm_cim_v8.bench.util import stats_ms
            out["exact"] = stats_ms(lat)
            out["exact_acc_note"] = "hits by construction (same dict path)"
        out["peak_rss_mib"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1)
    except MemoryError:
        out["failure"] = "MemoryError (cap hit) — recorded, not extrapolated"
        out["peak_rss_mib"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1)
    except Exception as e:  # any other failure is a result too
        out["failure"] = f"{type(e).__name__}: {e}"[:220]
    return out


CODE = '''
import sys, json
sys.path.insert(0, "/home/user")
from rasm_cim_v8.bench.m6_scaling import worker
print(json.dumps(worker({n}, "{cond}", {cap})))
'''


def main():
    import subprocess
    results = {"N_list": N_LIST, "note": "all numbers measured at stated N; "
               "MemoryError rows are real failures, not omissions",
               "runs": {}}
    for n in N_LIST:
        for cond in ("stream", "stream+hv", "dict", "v06mem"):
            cap = 1500  # headroom below 1982 MiB host total
            code = CODE.format(n=n, cond=cond, cap=cap)
            p = subprocess.run([sys.executable, "-c", code], capture_output=True,
                               text=True, timeout=1800)
            if p.returncode != 0:
                r = {"n": n, "cond": cond, "failure": "subprocess crash",
                     "stderr_tail": p.stderr[-300:]}
            else:
                r = json.loads(p.stdout.strip().splitlines()[-1])
            results["runs"][f"{n}:{cond}"] = r
            done = {k: v for k, v in r.items() if k in ("build_s", "open_s", "peak_rss_mib", "failure", "exact")}
            print(f"n={n:>8} {cond:<10} {json.dumps(done, default=str)[:180]}", flush=True)
    # residue index size study at the largest n that stream+hv survived
    results["residue_index_study"] = {}
    for n in N_LIST:
        code = f'''
import sys, json, time, resource
sys.path.insert(0, "/home/user")
sys.setrecursionlimit(10000)
resource.setrlimit(resource.RLIMIT_AS, (1500*2**20, 1500*2**20))
try:
    from rasm_cim_v8.storage import format as F
    from rasm_cim_v8.memory.stream_memory import MobileStreamMemory
    n = {n}
    path = "/tmp/rasm_v8_m6/m6_{n}_stream.rasm"
    t0 = time.perf_counter()
    m = MobileStreamMemory(path, tiers=("residue",)).open()
    t1 = time.perf_counter()
    sizes = m.sizes()
    import random
    rng = random.Random(5)
    lat = []
    for r in rng.sample(range(97), 20):
        a = time.perf_counter(); m.query_partial(r, 97); lat.append(time.perf_counter()-a)
    print(json.dumps({{"n": n, "residue_build_s": round(t1-t0, 3),
                       "residue_entries": sizes.get("residue_entries"),
                       "partial20_s": round(sum(lat), 4),
                       "peak_rss_mib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.0, 1)}}))
except Exception as e:
    print(json.dumps({{"n": {n}, "failure": type(e).__name__ + ": " + str(e)[:120]}}))
'''
        p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=900)
        results["residue_index_study"][str(n)] = json.loads(p.stdout.strip().splitlines()[-1]) if p.returncode == 0 else {"failure": p.stderr[-200:]}
        print("residue@%s" % n, results["residue_index_study"][str(n)], flush=True)
    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    json.dump(results, open(RESULTS, "w"), indent=1)
    print("saved", RESULTS)


if __name__ == "__main__":
    main()
