# RASM-CIM v0.8 — Python-only stage: full report

Date: 2026-11-25 · Language policy: **Python only** (Rust v0.7 frozen and verified untouched).
Everything below is a measured artifact of this workspace; artifact paths are given for every number.

---

## 0. Freeze of Rust v0.7 (prerequisite, verified)

* Zero `.rs` files have an mtime newer than the v0.8 start; `diff -r` of
  `rasm_cim_rust/{src,bindings}` against the frozen archive
  `RASM-CIM_v0.1-v0.7_RUST_FULL_2026-11-25.zip`: **ALL .rs IDENTICAL**.
* Reproducible hash (recorded in `docs_v8/V08_BASELINE.md` + `V08_AUDIT.md`):
  `find rasm_cim_rust -name '*.rs' -exec md5sum {} + | sort | md5sum` =
  `451bbfa52e67a72b04c0046d9df4e9d5` (identical at stage start and stage close).
  (The audit-day note carried a hash string taken with an unrecorded command
  variant; the ZIP diff above is the authoritative proof and was run at close.)
* No Rust was recompiled, no new Rust component, no Python result was ever
  presented as a Rust one, and no Rust number was reused as evidence for Python.

## 1. Audit (§1)

`docs_v8/V08_AUDIT.md`, `V08_COMPONENT_MAP.md`, `V08_BASELINE.md`.
Key baseline facts fixed **before** any v0.8 code: 57/57 historical pytest
green; Python harness medians (RNS 9.98 ms, mem-build 81.6 ms @5k, noisy 181 ms);
100k in-RAM reference `build 1.813 s / RSS 296.6 MiB`; **rule-110 correction**:
v0.7 ERRORS #7 said three legacy-vs-canonical diverging neighborhoods — actual
count is **five** (011, 100, 101, 110, 111), now pinned by test
`test_cross_version.py` (historical reports left untouched; the correction
lives in v0.8 docs only).

## 2. M2 — Streaming memory (Python)

Modules: `rasm_cim_v8/storage/format.py` (v1 byte-compatible with the frozen
v0.7 spec + new paged **v2** with crc32), `storage/streaming.py`
(`open_lazy/LazyStore`: os.pread / mmap / buffered modes, LRU PageCache,
read_records/get/scan/query_partial, corrupt-page raise|skip policy),
`memory/stream_memory.py` (MobileStreamMemory tiers).

Measured at 100k records, one dataset, identical ops across variants
(`results_v8/m2_streaming.json`, bench `m2_streaming.py`):

| criterion | target | actual | verdict |
|---|---|---|---|
| exact accuracy (get/slice/scan vs full-load ground truth) | 100% | **5/5 all modes, 100.000%** | PASS |
| peak RSS @100k streaming | <40 MiB | **26.2 MiB** (pread/mmap/buffered all) | PASS |
| p95 vs in-RAM dict baseline | ≤2× | **25.4–35.6×** | **FAIL — kept as a result** |
| open (index-only) | — | 0.027 s (vs 1.407 s full dict load at 1M, M6) | streaming real |
| full scan | — | 0.035–0.038 s | |
| LRU effect | — | p50 0.0154 ms → **0.0029 ms** at cache_pages=32 | cache works |

Why the p95 target missed and why we didn't "fix" the baseline: the in-RAM
dict with float HVs answers a hot get in ~0.53 µs; any page-touch path has a
~2–25 µs floor (pread + per-page crc32 + bisect) — that's the ratio. The
streaming path is ~6× faster **than its own cold start** and uses 1/11 of the
RAM; on mobile that trade is the point, and p50 (0.003–0.016 ms) is within an
order of magnitude of the dict. Design change forced by measurement: v2 page
checksums moved FNV-1a→crc32 after FNV in pure Python cost 2–3 ms/page and
dominated p95 (round-1 files were overwritten by the rerun; the per-mode round-1 p95 ratios are recorded in the JSON `history` note and in this paragraph; both rounds failed the ≤2× target for different reasons — round-1 additionally by checksum cost); v1 keeps
FNV-64 for spec compatibility. Full-file loads are never called "streaming":
the `to_full_memory()` path exists but is reported separately from `open_lazy`.

## 3. M3 — rule110 semantics + concurrency

* `semantic/rule110.py`: `step_rule110(canonical=False|True)`, both tables,
  `all_eight` covers all 8 neighborhoods; legacy output unchanged (tests).
* Concurrency models measured, not asserted (`docs_v8/V08_M3_M7_CONCURRENCY.md`,
  `results_v8/m3_m7_concurrency.json`): pure-Python CPU threads = **1.02×**
  (GIL — measured, as predicted); processes ≈ **2.01×** on 2 CPUs; numpy
  threads 1.07× while processes 0.65× on short jobs; file I/O threads 0.89×,
  processes 1.71×; streaming fan-out (4× LazyStore scans of 100k v2 file):
  threads 1.01×, processes 1.38×.

## 4. M4 — Fuzzing (two classes, honestly labeled)

`results_v8/m4_atheris.json`; property/random tests in
`rasm_cim_v8/tests/test_m4_properties.py` (18 tests, all green).

* **Coverage-guided (atheris/libFuzzer, instrumented)**: v1 deserializer
  19.1M execs, v2 paged reader 1.96M execs, RNS decode 12.4M execs, graph
  op-sequences 1.26M execs in 90 s each → **2 real bugs found & fixed**:
  **V8-F1** fd-leak in `LazyStore.__init__` on corrupt files (fuzzer died with
  EMFILE; fixed with try/except cleanup) and **V8-F2** `page_recs=0` header →
  unguarded `ZeroDivisionError` (now `StorageError` at parse time). Both have
  regression tests and re-runs of the fuzzers are clean.
* **Property-based (hypothesis)**: v1 roundtrip on full int64 domain, hostile
  bytes classification, v2 roundtrip vs last-wins model, RNS decode totality +
  range invariant, TruthGraph vs shadow model over assert_true/assert_false/
  retract interleavings.
* **Fixed-seed random mutation** (explicitly *not* coverage-guided): 600+1600
  mutated artifacts through both formats; every result classified clean or
  StorageError/CorruptPageError — zero silent-wrong-data events (the property
  that matters).
* Spec target `PackedStore.from_bytes` **does not exist on any Python side**
  (verified: only in the Rust binding's test); live equivalents fuzzed
  instead — no fake target was invented.

## 5. M5 — SIMD probe (bit packing) — no instruction claims

`results_v8/m5_hv.json` (regenerated with per-size Q and honest picks
denominator), bench `m5_hv.py`, module `hv/bitpack.py`.

p50 per query, D=256 (same queries/data; correctness = picks vs truth):

| n | scalar-float | bitpack stdlib | numpy f64 dot | numpy u64+bitwise_count | numba u64 |
|---|---|---|---|---|---|
| 1k  | 45.6 ms | 0.199 ms (236×) | 0.068 (650×) | 0.038 (1225×) | 0.0034 (13737×) |
| 10k | 468 ms | 1.78 (260×) | 0.44 (1052×) | 0.31 (1503×) | 0.029 (15855×) |
| 100k (Q=5) | 4718 ms | 17.5 (269×) | 8.37 (564×) | 4.30 (1098×) | 0.333 (14158×) |

* 1.5× target vs scalar-float: exceeded by every variant — but the **main API
  keeps `bitpack_scalar` stdlib path** because numpy adds +30 MiB (f64 matrix)
  to +224 MiB (u64 matrix) at 100k (RSS attribution in JSON) while packed
  int-lists add **0.0 MiB**; numba adds a 0.49–0.61 s JIT inside prep and is
  not Termux-installable — excluded from the default path by measurement.
* **NO NEON/SIMD claim**: no instruction-level proof exists on this host; the
  gains are (a) int.bit_count on machine words, (b) C-level loops. Stated in
  the module docstring, the report and the plan.

## 6. M6 — Scale 1k/10k/100k/1M, ALL ACTUAL runs

`results_v8/m6_scaling.json` (16 runs + 4 residue-index runs, every row from a
real subprocess; nothing between 100k and 1M extrapolated).

| records | dict peak | stream peak | **stream RSS at query** | stream p95 exact | stream+hv p95 | v0.6 MemorySystem (float HV) |
|---|---|---|---|---|---|---|
| 1,000 | 13.8 | 14.2 | **13.8** | 0.0022 ms | 0.0023 | 37.1 |
| 10,000 | 19.0 | 18.0 | **14.1** | 0.0025 ms | 0.0030 | 61.8 |
| 100,000 | 76.4 | 64.7 | **17.6** | 0.0285 ms | 0.0284 | 296.3 |
| 1,000,000 | 597.3 | 497.5 | **43.7** | 0.0235 ms | 0.0233 | MemoryError |


* 1M build (write_v2 from iterator) **0.871 s**, file 32.0 MB, index-resident
  at open 0.296 s; update = full rebuild 1.003 s (documented v2 limitation: no
  in-place append yet → PLAN_v0.9); full deserialize-to-dict 1.2455 s.
* Residue/inverted index: entries = 3n (3 moduli), build 0.912 s @1M,
  partial-modulus query 0.365 ms avg (20 queries in 7.3 ms); resident size dominated by int-tuple
  overhead — for 1M it costs ~90 MB of the stream RSS; a compact array index
  is the v0.9 target.
* The v0.6 float-HV memory system is **impossible at 1M by construction**
  (2 KB HV/record ≈ 2 GB) — the MemoryError row is the honest answer to
  "can it scale", not the streaming tier's failure.
* Mobile relevance: stream tier at 100k = 17.6 MiB at query (<40 MiB target ✓
  again, now with the fuller MobileStreamMemory), at 1M = 43.7 MiB.

## 7. M7 — async model per scenario

Verdict table in `docs_v8/V08_M3_M7_CONCURRENCY.md`. Short form: ProcessPool
for Python-CPU batches (~2× on 2 CPUs, results kept scalar because pickle is
~102 MB/s measured); threads only where native code releases the GIL
(numpy/BLAS — 1.07×); **asyncio measured ≤ sequential in every RASM-CIM
scenario (0.90–0.91×) and is therefore not adopted**; small jobs pay the
~15–21 ms pool startup and should reuse a pool.

## 8. M1 — Android/Termux package (prepared; **NOT complete — no device**)

`rasm_cim_v8/mobile/` (package + `termux_install.sh`, `termux_run.sh`,
stdlib-only `mobile_bench.py` measuring start time, runtime, VmHWM+ru_maxrss
peak RSS, CPU time, serial vs threads×4 vs processes×2, temperature/frequency
from sysfs **if present**), plus `rasm_cim_v8/cli.py` JSON CLI.

* Validated on host: full pipeline under an emulated bare interpreter (numpy /
  numba / psutil imports **blocked**) — selftest 7 ms, 20k records end-to-end,
  33.8 MiB RSS, noisy hit OK, processes path works, thermal nodes reported as
  `"available": false`. Artifact: `results_v8/mobile_bench_host_validation.json`
  carries an explicit `"_label"` field stating it is NOT an Android measurement.
* **M1 acceptance (real phone: start time, runtime, peak RSS, CPU, serial vs
  parallel, temperature) remains OPEN.** The user runs `termux_run.sh` on a
  device and the harness records everything; until that file exists from
  Android, v0.8 says "Android capability = package-ready, unproven".
* Memory budget vs the 3 GB target: 100k records ≈ 3.2 MB file + 17–55 MiB
  streaming RSS on host — the budget is not approached at ≤1M on desktop
  numbers; device numbers TBD.

## 9. Regression (§9) after every stage

`python3 -m pytest test_rasm*.py rasm_cim_v8/tests/ -q` → **84 passed** at
close (57 historical + 27 v0.8, including golden-v1-bytes vs frozen spec,
v0.6 PackedStore layout identity, 238-candidate BFS synthesis, sparse-CA 9 vs
64 ops, RNS M=15015 overflow, dedup/CONFLICT/UNKNOWN, rule110 5-divergence,
old-data v1→v2 import chain both directions, results identical pre/post
serialization). Historical `results_v*` JSONs untouched; every v0.8 result in
`results_v8/`.

## 10. Answers to the five v0.8 questions (§12)

1. **Can Python RASM-CIM run on real Android?** Package-ready and
   host-validated on stdlib-only; **unproven on device** — that's exactly what
   the Termux harness exists to settle. No hard third-party dependency in the
   core path (numpy/numba optional).
2. **Can memory handle big volumes without full RAM load?** YES at 100k
   (17.6 MiB at query, 100% accuracy, scan 0.035 s) and YES at 1M
   (43.7 MiB at query vs 597 MiB dict). Not "free": index residency is O(N)
   Python objects; a compact index is v0.9 work. p95 is 25–35× the hot dict —
   recorded FAIL of the ≤2× target.
3. **Performance vs scale?** build 0.001→0.871 s and exact p95 0.0022→0.0235 ms
   from 1k→1M stream (≈linear, measured at all four points, no gap-filling);
   dict path degrades super-linearly in RSS (13.8→597 MiB) and v0.6 float-HV
   OOMs at 1M.
4. **Which optimizations measurably help?** Packed-HV `int.bit_count` similarity
   (236–15855× in microbenches, zero RAM cost) → integrated into the stream
   tier (query_noisy over paged data, 2858/2858 hits @10% flips); crc32 for
   page integrity (FNV cost was a measured p95 killer); LRU page cache
   (p50 5× better at 32 pages); streaming paged v2 format; ProcessPool for
   Python-CPU batch work where GIL pins threads. Which **don't**: asyncio
   here (0.9×), threads for CPU-bound (1.0×), numpy/numba in the mobile path
   (RAM/JIT/install-cost with no device benefit demonstrated).
5. **Which Python limits block mobile?** GIL (CPU parallelism capped at ~1×,
   measured), interpreter page-touch floor (25–35× dict p95), per-object
   index overhead (O(N) resident), no NEON exposure without ctypes/intrinsic
   proof (none claimed), cold-process startup on Android TBD.

## 11. Deliverables index

source `rasm_cim_v8/` (storage, memory, hv, semantic, bench, fuzz, tests, cli, mobile) ·
Streaming Memory (M2) · Termux package + install instructions (M1, device-run OPEN) ·
fuzzing suite (hypothesis + atheris + labeled random) · SIMD/NumPy/Numba benchmarks ·
1M benchmark (actual) · cross-version regression tests ·
plots `plots_v8/{ram_by_scale,latency_by_scale,m2_p95_vs_target,m5_methods,m3_m7_speedups}.png` ·
this report · `docs_v8/PLAN_v0.9.md` · archive `RASM-CIM_v0.1-v0.8_…zip`.

Negative results kept where they were found: p95 FAIL (M2), threads-are-not-
magic (M3/M7), float-HV 1M infeasibility (M6), numpy/numba mobile exclusion
(M5), EMFILE/ZeroDivision fuzz bugs (fixed, guarded).
