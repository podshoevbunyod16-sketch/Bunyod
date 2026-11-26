# RASM-CIM v0.7 — Rust Compute Core

Full Rust port of the RASM-CIM v0.1–v0.6 compute core: RNS arithmetic,
hierarchical memory (exact/partial/noisy/semantic/episodic/procedural,
dedup, conflicts, RNS compression), TruthGraph, cellular automata
(dense/sparse), reservoir dynamics, BFS program synthesis with
train/val/test split, packed storage and a versioned binary format —
with PyO3 bindings so **Python stays the research interface** while the
ported computation runs in Rust.

- Rust **stable**, `1.100.0`. `#![forbid(unsafe_code)]` (compiler-enforced). Deps: `thiserror`, `rayon`,
  `rand` (dev: `criterion`). No tokio/serde — removed as unused (see
  `docs/ERRORS_FOUND_v0.7.md` #4).
- This core changes **no experimental result**: synthesis still finds
  `add 2 → mul 3` with exactly 238 candidates, sparse CA still 9 vs 64 ops
  on the v0.6 D1 pattern, memory still dedup/CONFLICT/calibrated-tau, and
  learning still reproduces the **REFUTED** selective-SSM parity hypothesis
  at chance. Rust performance ≠ architecture intelligence.

## Layout

```
src/                 core crates (rns, memory, graph, ca, synthesis, learning,
                     reasoning, compute, storage, architectures, profiles, core)
bindings/python/     PyO3 cdylib `rasm_cim_ext` (abi3-py38)
tests/               cross-language pytest (28 tests vs Python v0.1–v0.6 refs)
benches/core.rs      criterion micro-benchmarks
examples/            gradcheck.rs (finite-diff, ALL params), train_probe.rs
benchmarks/          run_bench.py (10 suites, subprocess-isolated), run_scaling.py,
                     make_plots.py
mobile/              TERMUX.md (build+run instructions, honest build matrix)
docs/                AUDIT_v0.1-v0.6.md (pre-port), ARCHITECTURE.md,
                     ERRORS_FOUND_v0.7.md, NON_PORTED_v0.7.md, PLAN_v0.8.md
reports/             v0.7 report + JSON results (bench, scaling, criterion)
```

## Quick start

```bash
cargo test --release                              # 32 unit + 6 property tests
cargo build --release && ./target/release/rasm_cim memory-demo
cd bindings/python && cargo build --release        # librasm_cim_ext.so
# put .so next to the Python project (or site-packages), then:
python - <<'EOF'
import rasm_cim_ext as rx
r = rx.RnsPy([3,5,7,11,13])
enc = r.encode_unsigned(42); print(r.decode_unsigned(list(enc)))
m = rx.MemoryPy(128, 0.5, 0); m.write_episodic(1, 100); print(m.query_exact(1))
print(rx.synth([(0,6),(1,9)],[(2,12)],[(5,21)],3,50000)[0])  # [('add',2),('mul',3)]
EOF
python benchmarks/run_bench.py --drive --out results_v7/bench_v7.json   # needs Python refs
```

## Measured (release, 2-CPU Linux x86_64; medians over 12–15 reps; Rust side includes PyO3 overhead)

| Suite | Python | Rust | Ratio |
|---|---|---|---|
| RNS enc+dec ×2000 | 9.98 ms | 1.04 ms | 9.6× |
| Memory build 5000 | 81.6 ms | 5.78 ms | 14.1× |
| Partial query 97 | 0.235 ms | 0.195 ms | 1.2× (p95 0.8× — Rust tail slower, kept) |
| Noisy 200×D256 | 181 ms | 56.5 ms | 3.2× |
| Graph queries 2000 | 1.55 ms | 0.92 ms | 1.7× |
| CA 200×1024 dense | 3.83 ms | 1.90 ms | 2.0× |
| CA sparse steps | 5.49 ms | 2.03 ms | 2.7× |
| BFS synthesis | 0.53 ms | 0.097 ms | 5.4× |
| PackedStore 50k | 22.4 ms | 10.7 ms | 2.1× |
| Batch get 50k | 13.5 ms | 5.78 ms | 2.3× |

Negative/limiting results preserved: rayon batch-put 0.87× (slower than
sequential on 2 CPUs); debug-vs-release ext differs ~1.2× only; RAM win is
bounded by the Python interpreter (~30 MiB baseline) — the *store* is
16 B/record, not the process. No on-device Android run: see `mobile/`.

Docs for every claim: `reports/RASM-CIM_v0.7_report_2026-11-25.md`.
