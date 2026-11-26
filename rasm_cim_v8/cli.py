"""RASM-CIM v0.8 CLI — Python-only mobile-capable entry point.

No numpy/numba import at module level: the file/stream paths must run on
bare Termux Python (M1). Subcommands print JSON (grep-friendly on device).

usage: python -m rasm_cim_v8.cli <cmd>
  selftest            — import + micro smoke (Termux health check)
  build <n> <path>    — write v2 file with n synthetic records, print timing
  get <path> <key>    — streaming exact get (no full load), JSON result
  scan <path>         — full streaming scan count (memory-bounded)
  noisy <path> <key>  — build hv cache from file, query noisy(hv(key)), JSON
  m5-demo             — tiny scalar-vs-bitpack loop timing demo (device-safe)
"""
from __future__ import annotations
import json
import os
import sys
import time


def _cmd_selftest() -> dict:
    t0 = time.perf_counter()
    from rasm_cim_v8.storage import format as F
    from rasm_cim_v8.storage.streaming import open_lazy
    facts = [(i, (i * 7919) % 15015) for i in range(1000)]
    path = "/tmp/_v8_selftest.rasm"
    F.write_v2(facts, path, page_recs=128)
    st = open_lazy(path, cache_pages=2)
    ok_get = st.get(42) == ("TRUE", (42 * 7919) % 15015)
    ok_scan = sum(1 for _ in st.scan()) == 1000
    st.close()
    os.remove(path)
    return {"ok": ok_get and ok_scan, "elapsed_s": round(time.perf_counter() - t0, 3)}


def _cmd_build(n: int, path: str) -> dict:
    from rasm_cim_v8.storage import format as F
    t0 = time.perf_counter()
    meta = F.write_v2(((i, (i * 7919) % 15015) for i in range(n)), path, page_recs=2048)
    return {"n": n, "path": path, "file_bytes": meta["bytes"],
            "build_s": round(time.perf_counter() - t0, 3)}


def _cmd_get(path: str, key: int) -> dict:
    from rasm_cim_v8.storage.streaming import open_lazy
    st = open_lazy(path, mode="pread", cache_pages=4)
    t0 = time.perf_counter()
    r = st.get(key)
    out = {"status": r[0], "value": r[1] if r[0] != "CONFLICT" else r[1],
           "latency_ms": round((time.perf_counter() - t0) * 1000, 4),
           "pages_read": st.stats["pages_read"], "bytes_read": st.stats["bytes_read"],
           "n_records": st.n}
    st.close()
    return out


def _cmd_scan(path: str) -> dict:
    from rasm_cim_v8.storage.streaming import open_lazy
    st = open_lazy(path, mode="pread", cache_pages=4)
    t0 = time.perf_counter()
    c = sum(1 for _ in st.scan())
    import resource
    return {"rows": c, "scan_s": round(time.perf_counter() - t0, 3),
            "peak_rss_mib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1),
            "bytes_read": st.stats["bytes_read"]}


def _cmd_noisy(path: str, key: int) -> dict:
    from rasm_cim_v8.memory.stream_memory import MobileStreamMemory
    from rasm_cim_v8.hv.bitpack import gen_packed_hv
    m = MobileStreamMemory(path).open()
    t0 = time.perf_counter()
    m.open_hv_stream(256, 7, method="auto")   # M2-refactor default: streaming
    t_hv = time.perf_counter() - t0
    q = gen_packed_hv(key % m.n_records(), 256, 7)
    st, v, conf = m.query_noisy(q, 0.5)
    out = {"status": st, "value": v, "confidence": round(conf, 4),
           "hv_init_s": round(t_hv, 4), "hv_backend": "stream",
           "hv_cache_bytes": m.hv_bytes_estimate()}
    m.close()
    return out


def _cmd_m5demo() -> dict:
    from rasm_cim_v8.hv.bitpack import gen_packed_hv, sim_scalar
    D, N, Q = 256, 2000, 20
    packs = [gen_packed_hv(k, D, 7) for k in range(N)]
    qs = [gen_packed_hv(k, D, 7) for k in range(0, N, N // Q)]
    t0 = time.perf_counter()
    for q in qs:
        best, bi = -2.0, 0
        for i, p in enumerate(packs):
            s = (D - 2 * (q ^ p).bit_count()) / D
            if s > best:
                best, bi = s, i
    t_bp = time.perf_counter() - t0
    return {"n": N, "queries": Q, "bitpack_total_s": round(t_bp, 4),
            "note": "device-safe demo: pure stdlib, no numpy"}


def main(argv):
    if len(argv) < 2 or argv[1] not in ("selftest", "build", "get", "scan", "noisy", "m5-demo"):
        print(__doc__)
        return 2
    c = argv[1]
    try:
        if c == "selftest":
            r = _cmd_selftest()
        elif c == "build":
            r = _cmd_build(int(argv[2]), argv[3])
        elif c == "get":
            r = _cmd_get(argv[2], int(argv[3]))
        elif c == "scan":
            r = _cmd_scan(argv[2])
        elif c == "noisy":
            r = _cmd_noisy(argv[2], int(argv[3]))
        else:
            r = _cmd_m5demo()
        print(json.dumps(r))
        return 0 if r.get("ok", True) and "error" not in r else 1
    except Exception as e:
        print(json.dumps({"error": f"{type(e).__name__}: {e}"}))
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
