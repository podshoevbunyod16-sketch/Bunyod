# RASM-CIM v0.9 — Report: architecture & storage evolution (Python-only)

Date: 2026-11-25. Host: Linux 6.1.189+, 2 CPU, 1982 MiB RAM, Python 3.13.18,
/tmp on **tmpfs** (mount-verified). Baseline archive: `RASM-CIM_v0.1-v0.8_PYTHON_FULL_2026-11-25_r2.zip`
(freeze manifest `results_v9/v08_freeze_manifest.md5`, 51 files; nothing in v0.1–v0.8 or
the Rust v0.7 tree was edited — Rust aggregate md5 `451bbfa52e67a72b04c0046d9df4e9d5`
re-verified unchanged this session).

All numbers below are **host** measurements from this session (JSON artifacts in
`results_v9/`). Device numbers: none exist yet — M4 is OPEN (no Android hardware
in the sandbox); the Termux harness is shipped and host-smoke-tested only.

## Scope verdict table (TARGET / ACTUAL / PASS|FAIL)

| # | Target (spec) | ACTUAL (this session, host) | Verdict |
|---|---|---|---|
| M0 | baseline + freeze before changes | `docs_v9/V09_BASELINE.md`, 51-file md5 manifest, probes re-run | PASS |
| M1 | compact index: no full tuple-list sort, array residues | build VmHWM @1M **88.3** vs v0.8 **141.8 MiB** (−38%); @100k 21.0 vs 24.8; residue @1M **+27.3** vs **+72.2 MiB** (−62%); partial p50/p95 @1M 165.7/188.2 µs vs 274.5/378.4; small-n partial p50 regression 2.5 vs 1.4 µs @10k (documented, kept) | PASS |
| M1 | resident key/offs arrays | v0.8 already `array q/Q` (16 B/rec) — no-op; NOT re-done | N/A by M0 audit |
| M2 | update ≤5 ms @1M (target, not promise) | median **0.032 ms** fsync incl. (median of 12) — BUT on tmpfs; device flash cost unmeasured; v0.8 full rebuild was 1.003 s | PASS (host) / OPEN (device) |
| M2 | v3 journal: commit-last, torn-tail ignore, no replay dup, corrupt→controlled error | 16 tests incl. real SIGKILL at 3 write stages; dup-seq skip; mid-file bad crc → `JournalCorruptError` | PASS |
| M2 | old v1/v2 readable, APIs compatible, migration tested | `open_any` v1/v2/v3; v2→v3 migration test answers identical; v3 data pages = v2 pages under a fixed 40 B header | PASS |
| M3 | p95 ≤ 2× dict | warm64 p95 2.65 µs vs dict 0.366 µs @100k = **7.2×** | **FAIL (kept)** — floor decomposed below |
| M3 | LRU judged over ≥4 workloads | 5 workloads: repeat 6.57×↑, zipf1.0 5.7×↑ (76% hits), zipf1.2 5.69×↑, uniform 0.95×↓, full-scan-once 0.75×↓ | PASS (as method) |
| M4 | real-device A/B | **DEVICE DATA RECEIVED** (2026-11-26, user-run Termux, kernel 4.9.77+ @2.001 GHz; temp sysfs absent=null, not a failure): 20k×2 + 100k×2 runs; mismatches **0 across all arms (20/20 identical picks)**; device q p50/p95 @100k — legacy 0.101–0.154/0.121–0.179 ms, stream 0.100/0.117–0.118, **v09_mem9 0.093–0.100/0.107–0.112 (best p95, both runs)**; v3 file on device = 3 200 824 B, resident index 1 600 000 B (16 B/rec as designed). TWO gaps kept OPEN: (1) update arm numbers (flash fsync p50/p95) exist in the raw JSONs but were filtered out of the paste — awaiting extraction; (2) r2 harness ran arms in one process ⇒ RSS/peak columns cross-contaminated (all arms ~54.8/68.8 MiB @100k — allocator arenas not returned, VmHWM global); harness **r3 (isolated subprocess per arm) written + host-verified**, device re-run needed for a valid RSS verdict. The v0.8 projection “device streaming ≈34.3 MiB @100k” is hereby superseded/flagged unreliable. | PARTIAL PASS (latency/mismatch/compat) — update + RSS: OPEN |
| M5 | coverage-guided fuzzing + hypothesis props + regression corpus | atheris 5 targets (≈9.3 M runs total incl. reruns); 5 hypothesis props; 2 real library bugs found, both fixed, 6 regression tests w/ exact fuzz bytes | PASS |
| M6 | retract formal semantics first, then code; v0.4 untouched | spec in `rasm_cim_v9/graph.py`; exhaustive shadow-model eq. over all op sequences L≤5 (7776); legacy-mode == frozen v0.4 over 1296; 12 tests | PASS |
| M7 | historical suites green after every stage; v0.9 tests separate | `test_rasm*.py rasm_cim_v8/tests` 109 passed; `tests_v9` 40 passed (final run below) | PASS |

## M1 — compact index (`rasm_cim_v9/storage/index.py`)

- `sort_to_arrays(keys, offs, method)`: `tuples` = v0.8-equivalent reference;
  `order` = two-pass STABLE index sort on int keys (no tuple materialization);
  `numpy` = `np.lexsort((offs, keys.view(uint64)^SIGN))` with zero-copy
  `frombytes` results — handles negative int64 via the order-preserving sign flip.
- `auto` (measured tuning, two iterations): numpy only when n ≥ 200k AND importable —
  first patch still imported numpy at n=1k (peak 26.7 MiB) and re-measurement caught it.
  Final auto peaks **11.8/12.8/21.0/88.3 MiB** vs v0.8 **11.7/13.1/24.8/141.8** at
  1k/10k/100k/1M (results_v9/m1_m2.json, `M1_auto_tuning`).
- `ResidueIndex`: (pos,residue) → `array('q')` buckets built from the ALREADY-SORTED
  key array ⇒ every bucket sorted with zero per-bucket sorts. Same semantics as v0.8
  `query_partial` (asserted equal for mods 97/89/83 in tests).
  Trade-offs kept (not hidden): at 10k/100k partial p50 is 2.5/17.3 µs vs 1.4/8.7 µs
  for the dict version — array→list copy per query costs on small buckets; at 1M it
  flips to 165.7 vs 274.5 µs AND saves 44.9 MiB. Small-n path stays default-off in the
  memory tier (residue built on demand only).
- Build input is a GENERATOR: `build_v3_base` keeps exactly two flat arrays + one page
  buffer resident; no fact-materialized list.
- Resident exact-lookup index: 16 B/rec both engines (v0.8 already compact — the M0
  audit's "don't re-optimize" finding honored); @1M both open in ~0.27–0.29 s,
  exact p50/p95 16.5/22.3 µs (v3) vs 16.3/22.1 (v2) @1M — parity, as predicted.

## M2 — append-only journal + v3 format (documented in module docstrings)

- v3 = v2 base (pages byte-identical modulo 40 B header) + header CRC commit point
  (base uncommitted ⇒ `read_v3_header` refuses it) + journal frames
  `<QII(seq,count,crc32(body))>` + body; commit-per-frame; **footer = last complete
  frame**, no separate footer to corrupt.
- `update` = append one frame: header-only seq walk (O(#frames), never touches the
  index/pages); fsync per txn by default. @1M median 0.032/0.055 ms (fsync/nofsync) —
  **tmpfs caveat** recorded verbatim in results JSON; real-flash number is an M4 job.
- Recovery rules tested: torn tail (any partial frame) ⇒ ignored, last complete txn
  intact (`journal_truncated_bytes`); bad-crc frame WITH successors ⇒
  `JournalCorruptError` (controlled), with `on_bad="skip"` alternative; duplicate
  `seq` frames skipped (R3.3 replay-safety); SIGKILL at stage 0 (partial body),
  1 (header-only frame) and 2 (during 60 appends) ⇒ reopen is clean, base answers
  unpolluted, `get(5000)==("TRUE",1)` post-recovery.
- `compact()` = stream-rebuild + atomic `os.replace`; answers preserved; for
  1-record frames it does NOT shrink the file (measured ratio ~1.0 — kept as a
  documented design outcome, not papered over); it reclaims journal-lookup overhead.
- Old APIs: `MobileMemory9` mirrors `MobileStreamMemory` call shapes; `open_any`
  keeps v1/v2 readable; nothing renamed away.

## M3 — query study (results_v9/m3_tuning.json)

- A (hot-key LRU over resolved answers, 2 MiB cap): speedup is workload-determined:
  repeat ×6.57, zipf(α=1.0/1.2) ×5.7, uniform ×0.95 (slightly WORSE), full-scan-once
  ×0.75 (worse, eviction churn). Conclusion kept opt-in per spec: LRU is enabled by
  call site on skew, never silently in `auto`.
- B (warm): cache_pages 8→64 ⇒ p50 15.4→2.31 µs, p95 19.3→2.65 µs; the gain is page
  cache hits (5951/3000… cache_hits field), not index.
- C (mmap vs pread): at cp2/cp8 mmap p50/p95 13.6/17.1 vs 15.5/19.9 µs (~12–15%
  better); at cp64 mmap is WORSE on p95 (4.13 vs 2.64 µs — page-fault jitter). mmap
  kept available (`mode="mmap"`, parity-tested 0 mismatches + identical scan) but not
  made default.
- D (floor decomposition, 100k, µs): bisect on array 0.398; rec unpack 0.267;
  crc32(32 KiB) **9.02**; pread(32 KiB, hot) 2.05; full get w/ cached page 1.77;
  dict p95 0.366 (m1_m2). Decomposition: with a cached page the Python path floor is
  ~1.8–2.7 µs ⇒ **~5–7× dict even at zero I/O** — the 2× target is unreachable for
  an interpreted per-query path; the remaining cold-page p95 tail is dominated by
  crc32+pread, not the index. FAIL kept per spec: `M3_p95_vs_dict_warm64 = 7.2×,
  FAIL`; path to ≤2× would require answer-level caching (that's the LRU, A) or a
  compiled hot loop (banned this version) — recorded, not silently dropped.
  Note: the 82.7 µs `dict_get_reference` line in D is a lambda-rebuild artifact,
  annotated in the JSON and used nowhere.
- F (scans): 100k 26–32 ms, 1M 295–312 ms; mmap ≈4% faster than pread;
  page_recs 2048 ≈ 8192 (no real difference) — no tuning claims.

## M4 — device results (ARM64 Termux, received 2026-11-26; HOST tables above are separate)

Transcribed in `results_v9/device/DEVICE_RESULTS_PASTED.json` (raw JSONs remain on the
phone — user-reported data, never mixed into host rows). Device: kernel 4.9.77+,
scaling_cur_freq 2 001 000 kHz, no thermal sysfs (null ≠ failure).

| n | arm | q p50 (ms) | q p95 (ms) | init (s) | RSS/peak (MiB)* |
|---|---|---|---|---|---|
|20k| v08 legacy HV | 0.026 | 0.097–0.103 | 2.76–2.96 | 32.5–33.9 / 35.0–35.4 |
|20k| v08 stream  | 0.027 | 0.117–0.122 | 4.88–5.55 | 33.1–34.4 / 35.0–36.1 |
|20k| **v09 mem9**| 0.026 | 0.097–0.103 | 5.09–5.38 | 33.3–34.5 / 35.0–36.2 |
|100k| v08 legacy | 0.101–0.154 | 0.121–0.179 | 13.91–14.10 | 54.5–54.9 / 68.5–68.8 |
|100k| v08 stream | 0.100 | 0.117–0.118 | 25.30–25.48 | 54.5–54.8 / 68.5–68.8 |
|100k| **v09 mem9**| 0.093–0.100 | **0.107–0.112** | 27.85–28.43 | 54.5–54.8 / 68.5–68.8 |

\* CONTAMINATED columns: r2 harness ran arms in one process; VmRSS/VmHWM therefore
bound each other (identical 68.8 across arms @100k is the fingerprint). RSS verdict
WITHHELD pending r3 (per-arm subprocess isolation; host-verified 2026-11-26:
five arms, independent 27.8–29.0 MiB baselines).

Findings that ARE valid from this data: pick parity 20/20 vs legacy in every arm and
run (semantic equivalence on ARM64); v09_mem9 has the best p95 at 100k (0.107–0.112 vs
0.121–0.179 legacy) and matches legacy p50; init cost of mem9 = full v3 BUILD + HV
(not like-for-like vs legacy open — recorded so nobody misreads it: on device mem9
init ≈ 2× stream, because build writes the base); resident index 1.6 MB @100k =
16 B/rec as designed; v3 file size on device equals host (3 200 824 B).

Still OPEN: (1) `v09_v3_journal_updates` upd p50/p95 + `v09_v3_store` latencies —
present in raw JSONs, filtered out of the printed summary (printer kept only dicts
with "picks"); extraction command shipped in mobile README. (2) r3 re-run for clean
RSS/peak. The v0.8 host-delta projection (~34.3 MiB @100k device) is RETIRED.

## M5 — fuzzing (results_v9/fuzz_atheris.json, rasm_cim_v9/fuzz/)

- atheris+libFuzzer (coverage-guided, bytecode instrumentation, persistent corpora):
  targets header/journal/records/corruptmid/compact — header 6.32 M runs clean;
  corruptmid 1.23 M clean; compact 827 k clean; journal/records: crashes found →
  fixed → rerun 732 k/461 k clean.
- BUG-1 (also caught by hypothesis P5 first): v3 commit header wrote 44 B (4 B pad)
  and clobbered data-page-0's first key; masked when min key = 0 ⇒ every earlier
  bench (facts from (0,·)) never saw it; fix = header exactly HDR_SIZE.
  Regression: 3 tests incl. page_recs ∈ {1,2,3,16,1024} first-key round-trip.
  Impact on prior v0.9 numbers: none (layout size unchanged; only key-0's shadowed
  value was affected and it stored 0) — stated, not hidden; all benches re-run after
  the fix anyway.
- BUG-2 (atheris-found): journal frame `count` (u32) unchecked before
  `pread(fd, count*16)` ⇒ MemoryError/DoS from a corrupt tail; fix = fstat bounds
  check first ⇒ oversized claim = torn tail per R3.2. Regression = the two exact
  crash byte-strings.
- HARNESS-1: fuzz-driver fd leak (EMFILE) — not a library bug, recorded anyway.
- Hypothesis props (5): append/read equivalence (60 ex), truncation-at-any-byte
  recovery (40 ex), replay idempotency (40 ex), get-vs-bruteforce parity incl.
  conflicts (30 ex), v2↔v3 answer parity + residue-vs-brute force (25 ex).

## M6 — TruthGraph9 (rasm_cim_v9/graph.py, tests_v9/test_m6_graph.py)

- Semantics written BEFORE code (module docstring): T/F sets, retract rules per mode,
  CONFLICT definitions for triple & functional queries, append-only history in every
  mode.
- Exhaustive A/B: all op-sequences length ≤5 (6^5 = 7776) vs independent shadow model
  per mode; `mode="legacy"` == frozen v0.4 `TruthGraph` over all length-4 sequences —
  v0.4 file untouched (read-only import).
- v0.9 default `retract` = symmetric (T∧F cleared). Justification + both stranded
  exhibits are tests: legacy strands `FALSE` after true+false+retract and can strand
  a functional FALSE-residue; symmetric guarantees retract ⇒ t∉T∪F and always lets
  retracts resolve CONFLICTs. Old behavior remains available and tested.

## M7 — compatibility

- `python -m pytest test_rasm*.py rasm_cim_v8/tests -q` → **109 passed** (after final
  code state — see run log `results_v9/final_tests.log`).
- `python -m pytest tests_v9 -q` → **40 passed**.
- v1 read, v2 read, v1→v2 historical migrations untouched; v2→v3 migration test added;
  dup keys, conflicts, UNKNOWN, corrupt pages, interrupted journal, replay, kill
  injection all covered by the 16+5+6+12 tests above.

## Kept failures / open items (per spec, not swept)

1. M3 p95 ≤2× dict: **FAIL**, floor decomposition recorded; no code change pretends otherwise.
2. M4 device A/B: **first real device data received and archived**
   (`results_v9/device/DEVICE_RESULTS_PASTED.json`, transcribed verbatim from the
   user's Termux run, 2026-11-26): query-latency, mismatch=0 and v3-on-device
   correctness GATES PASS; two things still OPEN — (a) the flash-fsync update
   p50/p95 and the v09_v3_store latency arm live in the raw device JSONs but were
   excluded from the printed summary (printer filtered on "picks") and must be
   re-extracted, (b) RSS/peak need an r3-harness re-run (isolated per-arm
   processes — r2 numbers are cross-contaminated within a process, so the
   streaming-RSS-advantage claim is UNjudged on device, and the v0.8 projection
   34.3 MiB is formally retired). Gates and commands: `rasm_cim_v9/mobile/README_v09.md`.
3. M2 ≤5 ms update: PASS on host tmpfs; the tmpfs caveat is embedded in the JSON verdict;
   flash-fsync latency stays an M4 measurement.
4. Residue arrays are slower at small n (p50 2.5 vs 1.4 µs @10k): kept as measured;
   tier default unchanged.
5. compact() ≈ no size win for many-tiny-frames journals: measured and documented.

## Files

code: `rasm_cim_v9/{storage/{index,journal,hotlru}.py, memory/store9.py, graph.py, mobile/*}`
tests: `tests_v9/{test_m1_m2,test_m5_props,test_m5_regressions,test_m6_graph}.py` (40)
benches: `bench_v9/{m1_m2_bench.py, m3_tuning.py, fuzz_m5_atheris.py}`
results: `results_v9/{m1_m2.json, m3_tuning.json, fuzz_atheris.json, *_run.log, v08_freeze_manifest.md5, final_tests.log}`; device: `results_v9/device/DEVICE_RESULTS_PASTED.json` (kept separate from host by spec)
docs: this file, `docs_v9/V09_BASELINE.md`, `docs_v9/PLAN_v1.0.md`, `docs_v9/INSTALL_v0.9.md`
