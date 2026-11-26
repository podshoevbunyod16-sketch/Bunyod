# PLAN v0.9 — from v0.8's measured gaps (Python-only continues)

Every item traces to a v0.8 measurement; no item is aspirational filler.
Rust v0.7 stays frozen unless the user explicitly reopens it.

1. **Compact resident index (fixes M6's O(N) object index)** — replace the
   Python list of (key,off) tuples with `array('q')` key/off arrays +
   bisect (or numpy if present, stdlib fallback). Target: 1M-record stream
   tier ≤ ~10 MiB RSS at query (measured baseline: 43.7 MiB), p95 ≤ 0.015 ms
   (measured: 0.0235). Acceptance: rerun of m6 suite at 1k/10k/100k/1M with
   before/after table; all existing 84 tests still green.

2. **In-place append + journal for v2 (fixes update = full rebuild 1.003 s @1M)** —
   append page + index-delta pages; crash-consistent (footer rewrite last,
   crc32 per page already). Target: single-record update ≤ 5 ms @1M
   (measured: 1003 ms). Acceptance: kill-injection test mid-update leaves file
   readable in pre-state (fuzz target for the journal writer).

3. **p95 gap engineering (M2 FAIL ≤2×)** — options measured in order until
   best: (a) read-only hot-key LRU of resolved (k,v) above page cache
   (p50 0.0029→? ; p95 floor is pread+crc+bisect ≈ 2–25 µs), (b) optional
   index-only "warm" mode that mmaps the DATA region too (RAM cost recorded),
   (c) accept-and-document if the floor is physical. Honest exit rule: if
   (a)+(b) cannot reach ≤2× at 100k, the report keeps the FAIL with the
   measured floor decomposition (already 25–35× now).

4. **Real Android execution (M1 completion)** — user runs `mobile/termux_run.sh`
   on a device; artifacts: mobile_bench_result.json (+ thermal/frequency if
   sysfs readable). Optional follow-up: `pkg install python-numpy` variant run
   for the M5 comparison ON DEVICE (still no SIMD claims without instruction
   evidence — aarch64 `cnt`/`ldp` loop proof would need perf/objdump on the
   phone; treat as out of scope unless tooling exists).

5. **Fuzzing continuity** — add the v0.9 journal writer + compact index parser
   as atheris t5/t6; keep corpora from results_v8 era; weekly 30-min default.

6. **Graph/semantic layer hardening** — TruthGraph fuzz found nothing, but the
   model test exposed that `retract` ignores false_edges (asymmetric);
   decide (document-only vs behavior change in v8 wrapper — frozen v0.4 never
   edited).

## v0.8 M2-refactor addendum (2026-11-25, streaming HV)

CLOSED (host): noisy HV retrieval no longer needs the full in-RAM cache —
`memory/hv_stream.py` + `open_hv_stream()`, default `auto`; 0 result-mismatches
vs legacy at 1k/20k/100k/1M; host RSS @100k 54.2→32.1 MiB, @1M 294.8→46.2.
Legacy `open_hv` KEPT (removal = v0.9 decision only after the device A/B
from `mobile_bench.py --n 100000` arrives). v0.9 items affected:
* item 1 (compact index) — the streaming HV made the resident Python index
  the DOMINANT cost at 1M (46 MiB is mostly index); priority raised;
* item 3 (p95 gap) — the same floor analysis now applies to query_noisy
  streaming latency (page-touch + per-query regen); a sidecar pre-generated
  HV file would erase the regen cost but changes storage (new format v3,
  existing files still readable) — proposed, NOT decided;
* new item: decide gen-vs-lru for skewed device workloads — LRU cap<N was
  measured worst-case for full scans (0 hits), untested for repeat-key
  queries: measure before keeping or removing the mode;
* new item: `block` size auto-tune from a memory budget (sweep measured the
  trade: 1024→32768 recs = 29.2→42.7 MiB for 42.3→37.3 ms @100k).

Out of scope for v0.9: asyncio (measured ≤ sequential), threads-for-CPU
(measured 1.02×), rewriting Python in other languages (policy), 10M-record
claims without an actual run.
