"""RASM-CIM v0.8 mobile self-bench — STDLIB ONLY (no numpy/numba imports).

Runs on bare Termux Python. Measures, per spec M1: start time, runtime,
peak RSS (VmHWM from /proc/self/status), CPU time (getrusage), serial vs
parallel (threads/processes), and temperature/throttling IF the device
exposes thermal sysfs (best-effort; absence is reported, never faked).

Usage (on device):  python mobile_bench.py [--n 20000] [--outdir .]
Writes mobile_bench_result.json next to the script.
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import resource
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
sys.path.insert(0, os.path.dirname(PKG))


def vm_hwm_mib() -> float:
    try:
        for line in open("/proc/self/status"):
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) / 1024.0
    except OSError:
        pass
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1)


def temps() -> dict:
    """Read every thermal zone / hwmon; if none exist (most cloud sandboxes,
    this dev container included) say so explicitly — no invented numbers."""
    out = {}
    for p in sorted(glob.glob("/sys/class/thermal/thermal_zone*/temp")):
        try:
            out[os.path.basename(os.path.dirname(p))] = int(open(p).read().strip()) / 1000.0
        except (OSError, ValueError):
            pass
    for p in sorted(glob.glob("/sys/class/hwmon/hwmon*/temp*_input")):
        try:
            out[os.path.basename(p)] = int(open(p).read().strip()) / 1000.0
        except (OSError, ValueError):
            pass
    if not out:
        out = {"available": False,
               "note": "no /sys/class/thermal or hwmon nodes visible in this environment"}
    return out


def cpu_freq_now() -> dict:
    out = {}
    for p in sorted(glob.glob("/sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq")):
        try:
            out[p.split("/cpu")[1].split("/")[0]] = int(open(p).read().strip()) // 1000
        except (OSError, ValueError):
            pass
    return out or {"available": False}


def _scan_child(args):        # module-level: picklable for ProcessPoolExecutor
    path, _i = args
    sys.path.insert(0, PKG)
    from rasm_cim_v8.storage.streaming import open_lazy as _o
    s = _o(path, cache_pages=4)
    c = sum(1 for _ in s.scan())
    s.close()
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20_000)
    ap.add_argument("--outdir", default=HERE)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    R = {"device_probe": {"platform": sys.platform, "machine": os.uname().machine,
                          "python": sys.version.split()[0],
                          "cpu_count": os.cpu_count()}}
    t_import0 = time.perf_counter()
    from rasm_cim_v8.storage import format as F
    from rasm_cim_v8.storage.streaming import open_lazy
    from rasm_cim_v8.memory.stream_memory import MobileStreamMemory
    from rasm_cim_v8.hv.bitpack import gen_packed_hv
    R["import_s"] = round(time.perf_counter() - t_import0, 4)

    data = os.path.join(a.outdir, f"mobile_{a.n}.rasm")
    M = 15015
    t0 = time.perf_counter()
    F.write_v2(((i, (i * 7919) % M) for i in range(a.n)), data, page_recs=1024)
    R["build_s"] = round(time.perf_counter() - t0, 3)
    R["file_bytes"] = os.path.getsize(data)

    t0 = time.perf_counter()
    st = MobileStreamMemory(data, tiers=("stream",)).open()
    R["open_s"] = round(time.perf_counter() - t0, 4)
    t0 = time.perf_counter()
    lat = []
    for k in range(0, min(a.n, 2000)):
        t1 = time.perf_counter()
        st.query_exact(k)
        lat.append(time.perf_counter() - t1)
    R["get2000_s"] = round(time.perf_counter() - t0, 3)
    R["get_p50_ms"] = round(sorted(lat)[len(lat) // 2] * 1000, 4)
    st.close()

    # noisy-HV path (stdlib bit_count) — the memory-relevant one
    t0 = time.perf_counter()
    m = MobileStreamMemory(data, tiers=("stream",)).open()
    m.open_hv(256, 7)
    R["hv_open_s"] = round(time.perf_counter() - t0, 3)
    q = gen_packed_hv(9 % a.n, 256, 7)
    t0 = time.perf_counter()
    ok = m.query_noisy(q, 0.5)
    R["noisy_first_s"] = round(time.perf_counter() - t0, 4)
    R["noisy_first_ok"] = ok[0] == "TRUE"
    m.close()

    # serial vs parallel: 4 independent full scans
    def scan_once(i):
        s = open_lazy(data, cache_pages=4)
        c = sum(1 for _ in s.scan())
        s.close()
        return c

    t0 = time.perf_counter()
    [scan_once(i) for i in range(4)]
    R["scans_serial_s"] = round(time.perf_counter() - t0, 3)
    from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=4) as ex:
        assert list(ex.map(scan_once, range(4))) == [a.n] * 4
    R["scans_threads4_s"] = round(time.perf_counter() - t0, 3)
    try:
        t0 = time.perf_counter()
        with ProcessPoolExecutor(max_workers=2) as ex:
            assert list(ex.map(_scan_child, [(data, i) for i in range(4)])) == [a.n] * 4
        R["scans_proc2_s"] = round(time.perf_counter() - t0, 3)
    except Exception as e:
        R["scans_proc2_s"] = None
        R["scans_proc2_error"] = f"{type(e).__name__}: {e}"[:160]

    ru = resource.getrusage(resource.RUSAGE_SELF)
    # --- M2-refactor A/B: legacy full-cache vs streaming HV (subprocesses,
    # isolated so freed cache pages never contaminate the stream measurement)
    try:
        from rasm_cim_v8.bench.m2_hv_refactor import run as _hvr, _build_files as _bf
        _paths = _bf([a.n])
        _leg = _hvr(a.n, "legacy", _paths[a.n], nq=10); _leg.pop("picks", None)
        _str = _hvr(a.n, "combo", _paths[a.n], nq=10); _str.pop("picks", None)
        R["hv_ab_legacy"] = _leg
        R["hv_ab_stream"] = _str
        if "rss_at_query_mib" in _leg and "rss_at_query_mib" in _str:
            R["hv_ab_rss_delta_mib"] = round(_leg["rss_at_query_mib"] - _str["rss_at_query_mib"], 1)
    except Exception as e:
        R["hv_ab_error"] = f"{type(e).__name__}: {e}"[:160]
    R["cpu_time_user_kernel_s"] = round(ru.ru_utime + ru.ru_stime, 3)
    R["peak_rss_mib_proc"] = vm_hwm_mib()
    R["peak_rss_mib_rusage"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1)
    R["temperature_C"] = temps()
    R["cpu_freq_mhz_after"] = cpu_freq_now()
    out = os.path.join(a.outdir, "mobile_bench_result.json")
    json.dump(R, open(out, "w"), indent=1)
    print(json.dumps(R, indent=1))
    print("RESULT FILE:", out)


if __name__ == "__main__":
    main()
