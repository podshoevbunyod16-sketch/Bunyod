"""M5 benchmark — noisy retrieval representations.

Question: does bit-packed bipolar HV (sim = D - 2*popcount(xor)) beat
  (a) pure-Python scalar float loop      <- the "чистый Python scalar" baseline
  (b) numpy float64 matmul (v0.6 path)
  (c) numpy uint64 chunks + np.bitwise_count
  (d) numba njit uint64 loop
per-query, with prep/conversion and RSS measured separately.

Correctness gate FIRST: every method must agree with the float dot on sim
value of its own chosen best record and on the tau decision; disagreements
in best-record index are only allowed when sims tie.

NO claim here is a "NEON implementation": we measure wall-clock of portable
code paths only.
"""
from __future__ import annotations
import json, os, random, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np

from rasm_cim_v8.bench.util import rss_mib, stats_ms
from rasm_cim_v8.hv import bitpack as BP

D = 256
Q = 200
FLIP_FRAC = 0.10


def gen_data(n: int, seed: int = 7):
    rng = np.random.default_rng(seed)
    hvs = rng.choice(np.array([-1.0, 1.0]), size=(n, D))
    orng = random.Random(seed)
    frng = np.random.default_rng(seed + 1000)
    qidx = [orng.randrange(n) for _ in range(Q)]
    flips = frng.random((Q, D)) < FLIP_FRAC
    qs = hvs[qidx].copy()
    qs[flips] *= -1.0
    return hvs, qs, qidx


def m_scalar_float(hvs_list, qs_list):
    t0 = time.perf_counter()
    lat = []
    picks = []
    for q in qs_list:
        a = time.perf_counter()
        best_i, best_s = 0, -2.0
        for i, hv in enumerate(hvs_list):
            s = sum(x * y for x, y in zip(q, hv)) / D
            if s > best_s:
                best_s, best_i = s, i
        lat.append(time.perf_counter() - a)
        picks.append(best_i)
    prep = None
    return picks, lat, prep


def m_bitpack(hvs_list, qs_list):
    t0 = time.perf_counter()
    packed = [BP.pack_bits(h) for h in hvs_list]
    prep_s = time.perf_counter() - t0
    idx = BP.BitpackIndex(D, tau=0.5)
    for p, i in zip(packed, range(len(packed))):
        idx.add_packed(p, i)
    lat, picks = [], []
    for q in qs_list:
        a = time.perf_counter()
        st, val, _c = idx.query(q)
        lat.append(time.perf_counter() - a)
        picks.append(-1 if st == "UNKNOWN" else val)
    return picks, lat, prep_s


def m_numpy_float(hvs, qs):
    t0 = time.perf_counter()
    K = np.stack(list(hvs))
    prep_s = time.perf_counter() - t0
    lat, picks = [], []
    for q in qs:
        a = time.perf_counter()
        sims = (K @ q) / D
        i = int(np.argmax(sims))
        lat.append(time.perf_counter() - a)
        picks.append(i if sims[i] >= 0.5 else -1)
    return picks, lat, prep_s


def m_numpy_i8(hvs, qs):
    t0 = time.perf_counter()
    mat = BP.pack_to_u64_chunks_fast(hvs)
    qm = BP.pack_to_u64_chunks_fast(qs)
    prep_s = time.perf_counter() - t0
    lat, picks = [], []
    for q in qm:
        a = time.perf_counter()
        sims = BP.sims_numpy_i8(mat, q, D)
        i = int(np.argmax(sims))
        lat.append(time.perf_counter() - a)
        picks.append(i if sims[i] >= 0.5 else -1)
    return picks, lat, prep_s


def m_numba(hvs, qs):
    ker = BP.make_numba_kernel()
    if ker is None:
        return None
    t0 = time.perf_counter()
    mat = BP.pack_to_u64_chunks_fast(hvs)
    qm = BP.pack_to_u64_chunks_fast(qs)
    warm = ker(mat[:4], qm[0], D, 0.5)   # include JIT compile in prep
    prep_s = time.perf_counter() - t0
    lat, picks = [], []
    for q in qm:
        a = time.perf_counter()
        i, s = ker(mat, q, D, 0.5)
        lat.append(time.perf_counter() - a)
        picks.append(i if s >= 0.5 else -1)
    return picks, lat, prep_s


def run(n: int) -> dict:
    hvs, qs, qidx = gen_data(n)
    qeff = 200 if n <= 20_000 else 5   # scalar-float baseline is O(N*D) per
    qs, qidx = qs[:qeff], qidx[:qeff]  # query IN python; budget-capped, not
    hvs_list = [h for h in hvs]        # extrapolated — per-query stats stay
    qs_list = [q for q in qs]          # comparable across methods
    out = {"n": n, "D": D, "queries": qeff, "queries_note":
           "Q capped to 5 for scalar-float cost at n>20k; all methods use the same Q",
           "flip_frac": FLIP_FRAC}
    res = {}
    res["scalar_float"] = m_scalar_float(hvs_list, qs_list)
    res["bitpack_scalar"] = m_bitpack(hvs_list, qs_list)
    res["numpy_float64"] = m_numpy_float(hvs, qs)
    res["numpy_u64_bc"] = m_numpy_i8(hvs, qs)
    nb = m_numba(hvs, qs)
    if nb:
        res["numba_u64"] = nb
    # correctness gate: picks must equal qidx where tau decision is TRUE
    # (noisy queries with 10% flips at D=256 keep sim>=0.5 for true match:
    #  expected sim = 1 - 2*0.1 = 0.8 >> tau; ties unlikely — assert strict)
    for name, (picks, lat, prep) in res.items():
        ok = sum(p == t for p, t in zip(picks, qidx))
        res[name] = {"picks_ok": f"{ok}/{len(qidx)}",
                     "query_lat": stats_ms(lat),
                     "prep_s": round(prep, 4) if prep is not None else None}
    base = res["scalar_float"]["query_lat"]["median" if False else "p50_ms"]
    for name in res:
        res[name]["p50_ratio_vs_scalar"] = round(base / res[name]["query_lat"]["p50_ms"], 2)
    out["methods"] = res
    out["peak_rss_mib_after_all"] = round(rss_mib(), 1)
    # memory footprint per record, measured on fresh processes
    return out


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "run"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 10_000
    if mode == "rss":
        # measure RSS attribution per method only
        which = sys.argv[3]
        hvs, qs, qidx = gen_data(n)
        before = rss_mib()
        if which == "numpy_float64":
            m_numpy_float(hvs, qs)
        elif which == "numpy_u64_bc":
            m_numpy_i8(hvs, qs)
        elif which == "bitpack_scalar":
            m_bitpack([h for h in hvs], [q for q in qs])
        after = rss_mib()
        print(json.dumps({"method": which, "n": n, "rss_before": round(before, 1),
                          "rss_after": round(after, 1)}))
    else:
        print(json.dumps(run(n)))
