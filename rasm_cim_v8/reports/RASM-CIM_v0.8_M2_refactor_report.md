# RASM-CIM v0.8 — M2 Streaming Hypervector Refactor report

Date 2026-11-25 · Python-only (Rust v0.7 untouched: freeze hash
`451bbfa52e67a72b04c0046d9df4e9d5` identical before/after this refactor;
`diff -r` against the v0.7 archive stayed clean at v0.8 close).

## 1. Confirmed problem (user-reported on real Android ARM64 — quoted, not re-measured here)

| data | legacy `open_hv()` peak RSS (device) |
|---|---|
| 20 000 records | **38.7 MiB** |
| 100 000 records | **56.4 MiB** — over the <40 MiB target |

Root cause in code (`memory/stream_memory.py`, pre-refactor): `open_hv()`
materialized **every** packed HV into `self._hv` (list of Python ints +
values) — O(N·(d/32·4+64) B) resident, on top of the paged store that was
already memory-lean. The full-cache build also cost O(N) time at open
(host: 0.48 s @100k, 4.3 s @1M).

## 2. What was built

New module `rasm_cim_v8/memory/hv_stream.py` (`StreamHV`) + wiring in
`MobileStreamMemory`:

- `open_hv_stream(d, seed, method="auto", block=8192, cache_mb=2.0)` —
  registers a streaming backend; `query_noisy()` dispatches to it with the
  **same signature and same return contract**;
- **legacy `open_hv()` kept unchanged** (spec: no removal before comparison
  finished — the comparison is §4 below, and the tests keep both live);
- CLI `noisy` and the Termux bench now use the streaming backend by default
  (`method="auto"`), with an A/B block in `mobile_bench.py` that measures
  legacy vs stream on the phone itself;
- HV are regenerated from keys (`gen_packed_hv` determinism ⇒ results are a
  function of the same inputs), file format v2/v1 **not touched** (golden
  sha256 of both formats pinned by test).

## 3. The four required approaches, compared (all numbers host-measured:
`results_v8/m2_hv_refactor.json`, per-mode isolated subprocesses, RSS read
as VmRSS at query time so freed memory of other modes never contaminates)

### 1) HV generated during the streaming pass ("gen")
Zero resident HV. Correctness identical (see §4). Full-scan latency is the
raw Python regeneration cost — measured at every scale, no extrapolation:
p50 5.0 ms @1k, 84.9 ms @20k, **437.4 ms @100k** (p95 453.7, RSS 30.8 MiB;
post-hoc run, JSON note `note_100k_scalar`). Device-safe: stdlib only.

### 2) Bounded Hypervector LRU ("lru", byte-budget eviction, `HVCache`)
Cap < working set on a full scan = **every query misses** (2016…  measured
0 % hit benefit; bookkeeping makes it the SLOWEST mode: 643 ms p50 @100k vs
38 ms for the winner, and cap 0.5→8 MiB changes latency by ±3 % only —
sweep in JSON). It is kept as an explicit method (useful for skewed/repeat
workloads, bounded memory guaranteed: `_bytes ≤ budget` asserted by test)
but **auto never selects it** — recorded as a negative result, not hidden.

### 3) Block processing ("block")
Chunked `read_records(start, count)` (1024–32768 recs), HV list built per
block and discarded: resident = one block. Scalar speed ≈ gen (block-local
list reuse saves little; measured 86.8 vs 84.9 ms @20k). Its real value is
as the substrate of (4): memory bound = block·(32 B + slack).
Block sweep @100k with numpy: 1024 → 29.2 MiB/42.3 ms; 32768 → 42.7 MiB/37.3 ms
— a clean RAM↔latency dial, no cliff.

### 4) Combination — the new default ("block_numpy"/"combo" via "auto")
Block streaming **plus vectorized splitmix64 generation** in uint64 arrays
(wrap-around arithmetic reproduces `& MASK64` bit-for-bit; per-block
`np.bitwise_count(xor)` for the similarity; first-max via `argmax` + strict
`>` reproduces the scalar tie rule across block boundaries). Where numpy is
absent (bare Termux), `auto` falls back to `gen` — the measured best
scalar mode.

| records | mode | init (s) | p50 query (ms) | p95 (ms) | RSS at query (MiB) | resident HV bytes |
|---|---|---|---|---|---|---|
| 20k | legacy | 0.092 | 3.7 | 3.9 | 32.6 | 1.76 MB |
| 20k | **combo (host=numpy)** | 0.0003 | 8.3 | 8.9 | **31.1** | 0 |
| 100k | legacy | 0.485 | 18.3 | 20.2 | 54.2 | 8.80 MB |
| 100k | **combo** | 0.0004 | 38.1 | 41.0 | **32.1** | 0 |
| 1M | legacy | 4.30 | 192 | 192 | 294.8 | 88.0 MB |
| 1M | **combo** | 0.0004 | 410 | 429 | **46.2** | 0 |

(host RSS floor ~27 MiB = interpreter + numpy import; the legacy-vs-stream
delta is the meaningful number: **−22.1 MiB @100k, −248.6 MiB @1M**.)

## 4. Requirements compliance (all measured/proven, none assumed)

| requirement | evidence |
|---|---|
| no full-record load | `test_stream_backend_does_not_materialize_full_cache`: `self._hv is None`, resident HV bytes = 0 for `gen`; stream subprocess RSS at 1M = 46.2 MiB vs legacy 294.8 |
| similarity/threshold/confidence semantics preserved | same formula `(d−2·popcount)/d`, strict-`>` first-max, `best<tau→UNKNOWN` with raw best as conf, TRUE conf=(best+1)/2 — pinned by `test_unknown_confidence_and_tau_gate_parity`, `test_first_max_tie_rule_parity` (dup-key ties), tau==best boundary |
| `gen_packed_hv` determinism preserved | bitpack.py untouched; golden hex values embedded in test; numpy vectorized generator proven word-identical to scalar incl. negative/2^62 keys and d∈{64,128,256,512} |
| search results identical to old implementation | **0 mismatches** vs legacy in `m2_hv_refactor` at n=1k/20k/100k/1M (40/30/20/6 noisy queries @10 % flips, tau=0.5) + parametrized parity tests (5 methods × d∈{64,100,256}) incl. corrupt-skip datasets (same examined counts AND same picks) |
| existing file formats unchanged | no edits to `format.py`/`streaming.py` read paths in this refactor; golden sha256 of v2(8 recs, pr=3) and v1 blob pinned in `test_v1_v2_file_bytes_golden`; full v1/v2 suites green |
| approximate results not presented as exact | the streaming path is **exact** (deterministic regeneration), labeled as such; no sampling/approximation is used anywhere |
| old path kept until comparison done | legacy intact + its own test (`test_legacy_open_hv_path_still_intact`); §3/§4 tables ARE the comparison; removal deferred to v0.9 decision |
| RSS < 40 MiB @100k | **target, not guaranteed outcome:** host streaming = 32.1 MiB (legacy 54.2). Applying the measured host delta (−22.1 MiB) to the user's device numbers projects device streaming ≈ **34.3 MiB @100k** — a projection, explicitly NOT a measurement; the device number must come from `termux_run.sh` |

## 5. Test/coverage totals & artifacts

- `rasm_cim_v8/tests/test_m2_hv_refactor.py` — **25 new tests** (parity,
  determinism, budgets, corruption parity, format goldens, legacy intact);
- full suite: **109 passed** (57 historical v0.1–v0.6 + 52 v0.8 incl. M4
  fuzz/property and cross-version suites); CLI selftest ok;
- `results_v8/m2_hv_refactor.json`, `plots_v8/m2_hv_refactor.png`,
  `plots_v8/m2_hv_sweeps.png`;
- tests at 1k/20k/100k **and 1M actually run on host** (`--big`); the 1M row
  is above — device runs 1M only "if the device allows", the harness accepts
  any `--n`.

## 6. Android/Termux execution — procedure (M1 remains OPEN, unchanged policy)

`mobile_bench.py --n 100000` now includes an **hv A/B section**: two isolated
subprocesses (legacy vs auto-stream, 10 queries each) reporting
`hv_ab_legacy` / `hv_ab_stream` / `hv_ab_rss_delta_mib`, plus VmHWM peaks.
On a real device: `bash termux_run.sh 100000`. With numpy installed
(`pkg install python-numpy`) `auto` = vectorized blocks; without it = plain
gen (slower, still memory-lean; both exact). No NEON/SIMD claim is made for
either — instruction-level proof on aarch64 is not available in this
workspace. **Host numbers in this file are not Android results.**

## 7. Negative results and limits kept in the record

- LRU-cap mode is the slowest full-scan mode (§3.2) — retained with the
  measured explanation and the skew-case caveat;
- streaming trades latency: 2–3× slower p50 than the legacy cache @100k
  (38 vs 18 ms) and ~2× @1M — exact-but-regenerating; the legacy mode stays
  the speed option when RAM allows;
- scalar `gen`/`block` at 100k: 437.4 / 428.6 ms p50 (measured; ~11× the numpy
  block path at the same n — the vectorized C loops, not the algorithm, are
  what buys the latency; on a bare-phone interpreter this is the price);
- open-time savings are a side effect, not free: init 0.48→0.0004 s @100k
  because generation moved from open to query.
