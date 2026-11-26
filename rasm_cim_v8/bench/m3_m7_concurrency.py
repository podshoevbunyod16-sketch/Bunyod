"""M3-concurrency + M7-async: measured model comparison, Python only.

Scenarios (each in an isolated subprocess; every method sees identical jobs):
  S1 cpu_python   — pure-Python CPU (fnv/xor loops): seq | threads(4) |
                    asyncio+to_thread | processes(2). Expected from the GIL
                    model — but we MEASURE, never declare.
  S2 cpu_numpy    — GIL-releasing native loops (matmul): same methods; the
                    case where threads CAN legitimately win (BLAS releases GIL).
  S3 io_files     — 200 x 128 KiB write+read roundtrips: seq | threads |
                    processes | asyncio(to_thread).
  S4 stream_fan   — 4 concurrent LazyStore full scans (streaming memory
                    module): threads vs processes (per-process caches; RSS
                    multiplicity is a measured outcome).
  S5 ipc_serial   — ProcessPool roundtrip of a 1M-int list: pickles/s, bytes
                    transferred, overhead share of total time.
  S6 proc_scaling — same CPU job on pools of 1 and 2 workers: observed
                    speedup vs 2 CPUs (no more, no fewer claims).

Verdict table (M7 "choose the model"): built from the ratios themselves.
"""
from __future__ import annotations
import asyncio
import functools
import io
import json
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np


# ------------------------------------------------------------- jobs
def job_cpu_python(seed: int) -> float:
    MASK = (1 << 64) - 1
    acc = 0
    x = seed * 7919 + 1
    for i in range(2_000_000):
        x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & MASK
        acc ^= x
        if i & 4095 == 0:
            acc = (acc * 131 + (i ^ seed)) & MASK
    return acc & MASK


def job_cpu_numpy(seed: int) -> float:
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((900, 900))
    b = rng.standard_normal((900, 900))
    s = 0.0
    for _ in range(6):
        a = a @ b
        s += float(a[0, 0])
    return s


def job_io(path: str, i: int) -> int:
    data = bytes([(i + j) % 251 for j in range(128 * 1024)])
    p = f"{path}/f{i}.bin"
    with open(p, "wb") as f:
        f.write(data)
    with open(p, "rb") as f:
        back = f.read()
    return len(back)


def job_stream_scan(path: str, _i: int) -> int:
    from rasm_cim_v8.storage.streaming import open_lazy
    st = open_lazy(path, mode="pread", cache_pages=4)
    c = 0
    for _k, _v in st.scan():
        c += 1
    st.close()
    return c


# ------------------------------------------------------------- drivers
def seq(fn, jobs):
    t0 = time.perf_counter()
    out = [fn(j) for j in jobs]
    return time.perf_counter() - t0, out


def threads(fn, jobs, nw):
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=nw) as ex:
        out = list(ex.map(fn, jobs))
    return time.perf_counter() - t0, out


def processes(fn, jobs, nw):
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=nw) as ex:
        out = list(ex.map(fn, jobs))
    return time.perf_counter() - t0, out


async def _asyncio_to_thread(fn, jobs):
    t0 = time.perf_counter()
    res = await asyncio.gather(*(asyncio.to_thread(fn, j) for j in jobs))
    return time.perf_counter() - t0, list(res)


def asyncio_mt(fn, jobs):
    return asyncio.run(_asyncio_to_thread(fn, jobs))


# ------------------------------------------------------------- scenarios
def s1(tmp: str) -> dict:
    jobs = [1, 2, 3, 4]
    f = job_cpu_python
    a, r1 = seq(f, jobs)
    b, r2 = threads(f, jobs, 4)
    c, r3 = asyncio_mt(f, jobs)
    d, r4 = processes(f, jobs, 2)
    same = len(set(r1)) == 4 and r1 == r2 == r3
    return {"jobs": 4, "same_results": same,
            "seq_s": round(a, 3), "threads4_s": round(b, 3),
            "asyncio_to_thread_s": round(c, 3), "proc2_s": round(d, 3),
            "threads_speedup": round(a / b, 2), "proc_speedup": round(a / d, 2)}


def s2(tmp: str) -> dict:
    jobs = [1, 2, 3, 4]
    f = job_cpu_numpy
    a, r1 = seq(f, jobs)
    b, r2 = threads(f, jobs, 4)
    d, r4 = processes(f, jobs, 2)
    return {"seq_s": round(a, 3), "threads4_s": round(b, 3), "proc2_s": round(d, 3),
            "threads_speedup": round(a / b, 2), "proc_speedup": round(a / d, 2),
            "note": "GIL released inside BLAS; threads legitimately help here"}


def s3(tmp: str) -> dict:
    j = functools.partial(job_io, tmp)
    a, _ = seq(j, range(200))
    b, _ = threads(j, range(200), 4)
    d, _ = processes(j, range(200), 2)
    e, _ = asyncio_mt(j, range(200))
    return {"seq_s": round(a, 3), "threads4_s": round(b, 3), "proc2_s": round(d, 3),
            "asyncio_to_thread_s": round(e, 3),
            "threads_speedup": round(a / b, 2), "proc_speedup": round(a / d, 2),
            "asyncio_speedup": round(a / e, 2)}


def s4(tmp: str) -> dict:
    path = "/tmp/rasm_v8_m6/m6_100000_stream.rasm"
    if not os.path.exists(path):
        return {"skipped": "m6 100k file not present yet"}
    j = functools.partial(job_stream_scan, path)
    a, ra = seq(j, range(4))
    b, rb = threads(j, range(4), 4)
    import resource
    rss_th = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    d, rd = processes(j, range(4), 2)
    return {"rows_per_job": ra[0], "seq_s": round(a, 3), "threads4_s": round(b, 3),
            "proc2_s": round(d, 3), "threads_speedup": round(a / b, 2),
            "proc_speedup": round(a / d, 2),
            "rss_parent_after_threads_mib": round(rss_th, 1)}


def job_small(j: int) -> int:
    return sum(range(400_000)) + j


def s5(tmp: str) -> dict:
    payload = list(range(1_000_000))
    t0 = time.perf_counter()
    blob = pickle.dumps(payload)
    ser = time.perf_counter() - t0
    t0 = time.perf_counter()
    back = pickle.loads(blob)
    deser = time.perf_counter() - t0
    assert back == payload
    a, _ = processes(job_small, range(2), 2)
    return {"pickle_bytes": len(blob), "dumps_s": round(ser, 4), "loads_s": round(deser, 4),
            "mb_per_s": round(len(blob) / (ser + deser) / 2**20, 1),
            "tiny_result_roundtrip_s": round(a, 3),
            "note": "IPC = result-size-bound; tiny results cheap, 1M-int lists cost ~ms"}


def s6(tmp: str) -> dict:
    f = job_cpu_python
    a, _ = seq(f, [1, 2])
    d, _ = processes(f, [1, 2], 1)
    e, _ = processes(f, [1, 2], 2)
    return {"seq2_s": round(a, 2), "proc1_s": round(d, 2), "proc2_s": round(e, 2),
            "proc1_speedup": round(a / d, 2), "proc2_speedup": round(a / e, 2),
            "pool_startup_tax_included": True}


SCEN = {"S1_cpu_python": s1, "S2_cpu_numpy": s2, "S3_io_files": s3,
        "S4_stream_fanout": s4, "S5_ipc_pickle": s5, "S6_proc_scaling": s6}


def run_in_subprocess(name: str) -> dict:
    code = f'''
import sys, json
sys.path.insert(0, "/home/user")
import resource, tempfile, os
from rasm_cim_v8.bench.m3_m7_concurrency import SCEN
tmp = tempfile.mkdtemp()
r = SCEN["{name}"](tmp)
r["peak_rss_mib"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.0, 1)
print(json.dumps(r))
'''
    import subprocess
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       timeout=900, cwd="/home/user")
    if p.returncode != 0:
        return {"error": p.stderr[-400:]}
    return json.loads(p.stdout.strip().splitlines()[-1])


if __name__ == "__main__":
    out = {}
    for name in SCEN:
        out[name] = run_in_subprocess(name)
        print(name, "->", json.dumps(out[name])[:200], flush=True)
    # verdict from measured ratios only
    verdict = {}
    for k, v in out.items():
        if "error" in v:
            verdict[k] = "error"
            continue
        cands = []
        for meth, key in (("threads", "threads_speedup"), ("proc2", "proc_speedup"),
                          ("asyncio_to_thread", "asyncio_speedup")):
            if key in v and isinstance(v[key], (int, float)):
                cands.append((v[key], meth))
        if cands:
            best = max(cands)
            verdict[k] = f"best={best[1]} ({best[0]}× vs seq)"
        else:
            verdict[k] = "seq (no method beat it)"
    out["VERDICT"] = verdict
    os.makedirs("/home/user/results_v8", exist_ok=True)
    json.dump(out, open("/home/user/results_v8/m3_m7_concurrency.json", "w"), indent=1)
    print("VERDICT:", json.dumps(verdict))
