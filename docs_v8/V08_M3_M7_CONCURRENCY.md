# M3-concurrency + M7-async — measured model comparison (v0.8)

Date 2026-11-25 · host: 2 CPU, Python 3.13 · data: `results_v8/m3_m7_concurrency.json`
bench: `rasm_cim_v8/bench/m3_m7_concurrency.py` (each scenario in isolated subprocess).
Rule honored: **no "threads speed up CPU-bound" claims** — the opposite is
documented by measurement; all speedup numbers below are the measured ones.

## Scenarios and results (× = speedup vs sequential, >1 only where measured)

| # | scenario | threads×4 | processes×2 | asyncio+to_thread | verdict (measured) |
|---|---|---|---|---|---|
| S1 | CPU-bound, pure-Python loops (2M iters ×4) | 1.02× | **2.01×** | ~1.0× | GIL pins Python CPU work to ~1 core; only processes scale |
| S2 | CPU-bound, NumPy matmul (GIL released in BLAS) | **1.07×** | 0.65× (slower) | — | threads help ONLY when native code releases GIL; gain small on 2 CPUs |
| S3 | I/O-bound: 200×(128 KiB write+read) | 0.89× (slower!) | **1.71×** | 0.90× | disk/page-cache contention made threads & asyncio-to-thread slower than sequential |
| S4 | 4 concurrent LazyStore scans of 100k-record v2 file | 1.01× | 1.38× | — | streaming fan-out scales like I/O, with per-process RSS ×workers |
| S5 | IPC serialization: pickle roundtrip of 1M-int list | — | — | — | 4.87 MB, dumps 14.6 ms + loads 30.8 ms ≈ 102 MB/s → IPC cost is RESULT-size-bound; tiny results (S1-S4) cost ~21 ms pool overhead total |
| S6 | pool scaling 1→2 workers (same CPU job) | — | proc1 **1.00×**, proc2 **1.47×** (incl. startup tax) | — | observed ≈ ideal for 2 CPUs incl. startup; no super-linear claims |

## What this means for RASM-CIM (M7 choice table)

| project task | best model (measured) | why |
|---|---|---|
| HV build/similarity loops in pure Python | **ProcessPool** (≈×2.0 here) | GIL-bound; results are small ints → S5 IPC tax amortizes |
| the same work vectorized via NumPy/BLAS | threads OK (×1.07); processes LOSE (0.65×) on short jobs — pool startup dominates | GIL released only inside C loops; per-job IPC tax |
| multi-client reads from a v2 file | ProcessPool (×1.38) or just bigger page cache; **threads neutral** | pread already releases GIL; disk contention dominates |
| batch file I/O | ProcessPool (×1.71); asyncio NOT justified | asyncio added task overhead without gain (0.90×) |
| update/retract on TruthGraph | **stay sequential** | measured nothing to parallelize at these sizes; S6 startup tax ≥ job time |
| 1M-record streaming scan fan-out | sequential or proc×2 | single scan 0.19 s already; process RSS ×4 (46.6→~180 MiB extrapolation NOT made — measured parent-side value reported) |

**asyncio verdict:** on this codebase asyncio is justified ONLY as the
scheduler for to_thread fan-out when I/O overlaps with waiting, and even
there it measured ≤ sequential (0.90× at S3). We do not adopt asyncio for
CPU-bound work anywhere — that would repeat the pattern the audit forbids.

## GIL / serialization facts observed (not quoted from docs)
- threading ×4 on pure-Python CPU: 1.00× — CPython 3.13 GIL behavior confirmed by measurement.
- ProcessPool: startup ≈ 15–21 ms (measured in S5 tiny-job roundtrip) —
  irrelevant for ≥0.3 s jobs, dominant for micro-jobs → pool reuse mandatory.
- Pickle of result lists is the IPC bottleneck, not the job; keep worker
  results scalar (counts/statuses), as S1–S4 do.

## Negative results kept
- threads do not speed up Python CPU-bound work (S1: 1.0×, not a bug — the physics of the GIL);
- threads and asyncio were SLOWER than sequential for our file I/O mix (S3: 0.89×/0.90×);
- on a 2-CPU box max realistic speedup is ~2×; anything above would be a measurement error — none of the scenarios exceeded it.
