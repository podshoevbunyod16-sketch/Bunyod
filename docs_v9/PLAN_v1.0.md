# RASM-CIM PLAN v1.0 — from v0.9 evidence (no retro-changes, targets only)

Frozen: v0.1–v0.9 trees and all results; Rust v0.7 hash `451bbfa5…` immutable.
v1.0 may not declare any item below DONE without a measurement in the same
session, TARGET/ACTUAL/PASS|FAIL rows, and a JSON artifact under results_v10/.

## P0 — close v0.9 open items FIRST (gates before anything new)
1. **M4 device run** (user-side): Termux harness `rasm_cim_v9/mobile/`;
   arms × sizes {20k,100k}; store under `results_v9/device/`; gates: mismatches=0,
   flash fsync update p50/p95 recorded (PASS/FAIL vs 5 ms from DEVICE data),
   init & RSS table for v08-legacy vs v08-stream vs v09-mem9.
   v1.0 may not switch defaults until this exists.
2. **Device flash fsync policy**: if device upd p95 > 5 ms with fsync=True,
   evaluate batched-commit (fsync per K txns) + power-loss semantics doc;
   keep per-txn mode as default-safe option. Measure, then decide.

## P1 — storage endgame
3. **Journal frame-batch compaction tuning**: compact trigger by (frame_count,
   bytes) matrix on 10k/100k/1M incl. open-time recovery (host: open after
   60 txns @1M = 0.29 s — bound it: decide max journal before forced compact).
4. **Sidecar index file (v4?)**: persist sorted key/off arrays so open() skips the
   0.26–0.29 s index walk at 1M; invalidation via base header crc+size check;
   A/B open-time vs cold rebuild; ONLY as a cache (file always authoritative).
5. **Residue small-n hybrid**: buckets as array above threshold, list below
   (fixes the measured 2.5 vs 1.4 µs @10k regression) — must keep the parity
   tests and re-measure both ends.

## P2 — query path
6. Keep LRU opt-in; add per-call `use_hot_lru=` on MobileMemory9 once a device
   skew workload (repeat/zipf) proves the ×5.7 benefit on ARM.
7. mmap: document as cp≤8 win / cp64 p95 loss; consider madvise(WILLNEED) probe.
8. p95 ≤2× dict stays FAIL-by-design in pure Python; revisit ONLY with an
   explicitly authorized change (it is not a v1.0 Python goal).

## P3 — semantics
9. TruthGraph9: promote `mode="symmetric"` to the only default in v1.0 AFTER
   replaying the v0.8 graph regression corpus; keep legacy mode as
   `mode="legacy"` for stored-graph compatibility; add retract-transaction
   boundary (retract set as journal txn, crash-safe) — spec it before coding,
   same rule as v0.9 M6.
10. Shadow-model property tests extended to L≤6 (budget-checked) each release.

## P4 — quality rails
11. Fuzz corpora (`rasm_cim_v9/fuzz/corpus_*`) are regression inputs: replay all
    corpora every release (deterministic, `-runs`-capped), plus 2×60 s new fuzz
    per changed parser module; every finding = its own regression test (v0.9 rule).
12. Keep hypothesis props in CI-equivalent manual runs; seed fixed artifacts.

## Explicitly rejected / deferred
- Rust/C++/Go hot loops, SIMD claims (version rules).
- Replacing v2 readers or rewriting any historical result file.
- Any single-workload LRU default; any pre-device tuning claim about Android.
