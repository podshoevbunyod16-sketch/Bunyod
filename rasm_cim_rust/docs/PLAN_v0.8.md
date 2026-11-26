# RASM-CIM v0.8 — plan (from v0.7 findings, ordered by evidence value)

Motivation each item carries = a number or a documented gap from v0.7, not
taste. Items M1/M2 are hard gates: "Android support" stays incomplete until
they are done.

## M1. Real Android measurement (closes v0.7 §8 honestly)
- Run `mobile/TERMUX.md` on an ARM64 device; capture: CLI, `cargo test
  --release`, cross-language pytest, full bench JSON (`bench_termux.md` →
  `reports/`), peak RSS, thermal throttling sample (grep cpu-freq before/after).
- Gate: no "Android supported" wording until numbers land; if Termux rust
  links/behaves differently, record failure modes verbatim.

## M2. Mobile streaming storage (the one §8 requirement not yet met)
- Paged reader over the v1 format: `open_lazy(path) -> Header`,
  `read_records(range)` via `pread`/`mmap` (mmap = 1 documented `unsafe` block,
  separate safety tests, align/fail checks), LRU page cache with profile
  budget; `retention_check` must work without loading all pages.
- Acceptance: build 100k-record file; peak RSS < 40 MiB on mobile profile
  while query p95 stays within 2× of in-RAM Rust (measured, not claimed).

## M3. Fix the two semantics debts found in v0.7 (versioned, both languages)
- Rule 110 table (ERRORS #7): add `rule110_canonical` flag in Python
  `ca4.py` and Rust `ca` module; default OFF (legacy numbers stay
  reproducible); benchmark both tables; report delta.
- GIL release for long batch calls (`Python::allow_threads`): measure
  whether Python main-thread overlap changes batch par/seq ratios (today
  rayon put 0.87× may improve; may not — keep negative result if not).

## M4. Coverage-guided fuzzing
- nightly toolchain + cargo-fuzz targets: `storage::deserialize_facts`,
  `PackedStore::from_bytes`, `Rns::decode_unsigned` (residues are
  untrusted-shaped input), graph update/retract sequences.
- 1h corpus per target; minimization of any crash into a unit test.

## M5. SIMD probe (only where a win is plausible)
- Noisy retrieval dot-product over ±1 HV: u8 popcount trick
  (bit-packed {+1→1,−1→0}, sim = N − 2·popcount(xor)) + runtime-neon detection
  on aarch64 in M1 data; accept into API only if ≥1.5× vs scalar release
  on both x86_64 and the Android device, else keep scalar and record why.

## M6. Scale-up honesty at 1M records
- Re-run exact/partial/memory benchmarks at N=10^6 in Rust only (Python side
  is known O(hours) at that N); report RSS, per-op ns, index bloat (inv
  index is ~3N keys → quantify its bytes), and whether partial ops still
  track N/97. No extrapolation from 100k beyond a stated error band.

## M7. Async boundary (evaluate, likely reject)
- Only if M2 exists: tokio `spawn_blocking` wrapper or plain thread pool for
  file IO in a server profile. Expected outcome: reject tokio (no async
  workload), documented either way (v0.7 kept this as a measured negative).

## Explicitly NOT in v0.8
- No architecture A/B/C merge before separate per-platform data (M1).
- No re-tuning of the learning experiment to flip L1 REFUTED.
- No hardware CAM/compute-in-memory claims; simulations stay labeled.
- No LLM-in-the-loop; comparative experiments stay separate modules.
- No deletion of Python v0.1–v0.6; Rust stays an implementation layer.
