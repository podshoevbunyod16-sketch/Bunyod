# RASM-CIM v0.7 — Errors found (porting + pre-existing, all kept visible)

## A. Bugs introduced during the Rust port, found by tests, then fixed

1. **CRT decode off-by-modulus** (`src/rns/mod.rs`). First implementation
   computed `t = (r_i − x) · inv mod m_i` using the *unreduced* accumulator
   `x`; from the second modulus onward the subtraction needed `x mod m_i`
   (Python `%` semantics are floor-based, `u64::wrapping_sub` is not).
   Symptom: `decode_unsigned(enc(42)) = 5847` instead of 42.
   Found by: `cargo test` roundtrip + step-by-step comparison against
   `rasm_cim_v4.rns4.RNSv4`. Fixed; regression guarded by
   `tests/property.rs::rns_crt_matches_naive_sum` (naive CRT formula vs
   iterative, 9 000 random values over 3 modulus sets).

2. **BPTT propagation off-by-one** (`src/learning/mod.rs`). The carry
   gradient multiplied by `1 − h_{t+1}²` where the Python reference
   (`ssm3.py`) uses `1 − h_t²` at loop index `t`. Caught by the
   finite-difference gradient check (initially max rel err ≫ 1e-4, after
   fix 8.8e-8 across ALL 29 params for both models). This is why gradcheck
   over *every* parameter exists as an example + unit test, not a spot check.

3. **Adam state sized wrong** (learning, `Adam::new(h*h*2)` literal in an
   early draft) → reallocation guard added in `Adam::step` and exact sizing
   in `set_weights`. Silent misbehavior impossible now (assert on length).

4. **Unused heavy-ish deps declared but not used**: `serde`, `serde_json`
   were in Cargo.toml from the skeleton; compiler emitted
   `unused_dependencies` warnings. Removed rather than kept "for later".

5. **Benchmark harness `UnboundLocalError`**: `np` referenced in a branch of
   `py_suite` while another branch imported numpy locally → Python treated
   `np` as function-local. Fixed with a function-top import. (Would have
   crashed exactly one suite; the harness surfaced it via `error` fields in
   the JSON instead of silently dropping rows — by design.)

6. **Debug/release harness bug**: `run_scaling.py` debug-vs-release
   comparison initially never swapped the .so (compared release twice). Fixed
   by loading debug via a `dbg/` path prefix; the first numbers (debug
   ≈ release for RNS) were therefore wrong and were re-measured
   (debug: rns 8.64 ms / ca 3.75 ms vs release 7.41 / 1.90 — real gap now
   only in CA-heavy code).

## B. Pre-existing issues found in the Python reference while porting
(kept as-is in the port to preserve comparability — NOT silently "fixed")

7. **Rule-110 table in `rasm_cim_v4/ca4.py` is not Rule 110.**
   Table `[0,1,1,0,1,0,0,1]` indexed by `left*4+center*2+right` gives
   outputs 0,1,1,0,1,0,0,1 for patterns 000..111, whereas canonical Rule
   110 is 0,1,1,0,1,1,1,0 (01101110). The three top patterns (101,110,111)
   deviate. The Rust port mirrors the same table for now (cross-language
   equivalence tests pass), and `rule_110_canonical` is scheduled for v0.8
   as a *flagged, versioned* fix in both languages. Impact: CA experiments
   in v0.4 used rule 90, so no published result is affected.

8. **Sparse-CA op counts depend on activity clustering.** `ca_step_sparse`
   updates the *neighborhood union* of active cells: 9 scattered actives on
   N=64 give 25 updates, not 9. The v0.6 D1 figure "9 vs 64" is exact only
   for the documented scenario (3 isolated actives at 10/20/30). My first
   cross-language test wrongly hard-coded 9; both scenarios are now tested
   (union semantics verified identical in both languages).

9. **`TruthGraph.query(s, r, o)` with explicit `o` performs no functional
   inference** (`graph4.py`): with only `FALSE(5,2,6)` asserted,
   `query(5,2,7)` is UNKNOWN, not TRUE; the functional rule applies only to
   `query(s,r,None)`. Preserved in the port; a test asserts the asymmetry in
   both languages so the semantics can't drift apart silently.

10. **`graph4.retract` on a defaultdict** uses
    `self.true_edges.get((s,r), set()).discard(o)` — safe (discards into a
    throwaway empty set when key absent) but subtle; Rust version has the
    same behavior with explicit `Option` handling.

## C. Limitations found (not fixable in-scope)

11. **No cargo-fuzz on stable**: fuzzing of the deserializer is done with a
    200 000-sample random-byte property test (no panics, always `Err`).
    Real coverage-guided fuzzing requires nightly + cargo-fuzz → v0.8.
12. **Allocation counts**: no allocator profiler in the sandbox; allocation
    behavior is documented structurally (fixed-size records, amortized Vec)
    and RSS deltas are measured, but per-op allocation counts are reported
    as NOT MEASURED rather than estimated.
13. **CPU utilization**: `os.times()` user+sys captured per run (cpu_s in
    bench JSON), but no per-core sampling; rayon put/get parallelism
    efficiency numbers come from wall-vs-serial ratios on a 2-CPU host only.
14. **GIL and rayon**: the binding methods do not call `allow_threads`, so
    the GIL stays held during Rust work. rayon worker threads themselves do
    not need the GIL, so intra-call parallelism is real (measured: batch get
    1.29×, batch put 0.87×); what is blocked is other *Python* threads
    running concurrently with a Rust call. Releasing the GIL for long batch
    calls → v0.8.
15. **Android**: codegen for aarch64-linux-android verified
    (`file … → ELF 64-bit ARM aarch64`), final **linking and on-device run
    were NOT performed** here (no NDK sysroot, no device). Android support
    is therefore explicitly *not* declared complete. Termux instructions
    provided in `mobile/TERMUX.md`.
