# RASM-CIM v0.7 — Rust Compute Core: Architecture Documentation

Crate: `rasm-cim-rust` (lib) + `rasm-cim` (CLI bin) + `rasm_cim_ext` (PyO3 module).
Rust **stable** (1.100.0), edition 2021. `#![forbid(unsafe_code)]` at the crate root —
zero unsafe is enforced by the compiler (a probe `unsafe` block fails the build;
this was tested during v0.7). No `tokio` (no async I/O
workload exists in the core — adding it would be ceremony without benefit;
documented decision). `rayon` used only where measured: see negative result
on parallel batch-put below.

## 1. Module map

```
src/
├── lib.rs            crate root, RasmError (thiserror), NO unsafe anywhere
├── main.rs           CLI: rns-roundtrip | memory-demo | ca-demo | synth-demo | packed-demo | storage-demo
├── core/mod.rs       shared primitives (HV similarity); intentionally thin
├── rns/              Residue Number System: encode/decode (CRT), add/sub/mul,
│                     range checks (OverflowError equivalent), signed via offset,
│                     op counting. Port of rns.py/rns2.py/rns4.py.
├── memory/           Hierarchical memory: episodic (append log), semantic
│                     (TruthGraph facade), procedural (program store), exact
│                     HashMap index, partial residue index (97/89/83), noisy
│                     bipolar-HV retrieval with calibrated confidence, dedup,
│                     CONFLICT, RNS-based exact compression, tau calibration.
│                     Port of memory_system.py (v0.6).
├── graph/            TruthGraph: TRUE/FALSE/UNKNOWN/CONFLICT + history + retract.
│                     Port of graph4.py (v0.4).
├── ca/               Rule-90/110 CA dense + event-driven sparse (ops counted),
│                     XOR-fold parity, EchoStateReservoir (dynamic step only).
│                     Port of ca4.py (v0.4).
├── synthesis/        BFS program synthesis with train/val/test split, safe
│                     interpreter (no eval). Port of program4.py (v0.4).
├── reasoning/        Reasoner facade over TruthGraph (composition r1*100+r2).
├── learning/         TanhRNNBPTT + selective SSM with Adam — line-by-line port
│                     of ssm3.py / selective_ssm.py. Grads verified by
│                     finite-difference gradcheck (all params, rel err ~1e-7).
├── compute/          PackedStore: 16 B/record LE buffer + HashMap index,
│                     rayon batch ops, bit-identical bytes with Python
│                     struct.pack("<qq").
├── storage/          Versioned binary format (magic+version+len+payload+FNV-1a),
│                     safe deserialize (no panics on byte soup — fuzz-tested),
│                     partial header read, Python packed-bytes import.
├── architectures/    arch_a (memory-centric), arch_b (dynamic: CA/graph/reservoir),
│                     arch_c (algorithmic: synthesis/compose/self-correct).
│                     NOT merged into one system (v0.6 rule kept).
├── profiles/         Mobile(1 thread, no parallelism, 256MB budget)/Desktop(4)/Server.
└── benchmarks/../    criterion benches/core.rs; Python driver in benchmarks/.
bindings/python/      PyO3 0.23 (abi3-py38) cdylib `rasm_cim_ext`.
```

## 2. Key design decisions (measured or justified, not aesthetic)

| Decision | Rationale |
|---|---|
| `Vec<u8>` LE buffer + `HashMap<i64,usize>` index for PackedStore | Same wire format as Python v0.6 (`struct.pack("<qq")`), so cross-language byte-identity test is possible; 16 B/record verified. |
| Residue partial index `(pos, key % 97|89|83)` | Port of v0.6; keeps partial-query op counts comparable (N/97 scaling preserved). |
| Storage format: fixed 24 B header + 16 B records | Zero-copy-ish read (single pass), partial read without payload parse, FNV-1a integrity. Version field for future migration. |
| `rem_euclid` everywhere for residue indexing | Matches Python `%` semantics for negative keys (Python modulo is floor-based); a naive `%` in Rust would silently break the partial index for negative keys. |
| `Rng` = splitmix64 (no `rand` in learning paths) | Determinism of training across runs/platforms; fuzzing uses `rand` explicitly. |
| rayon only in `batch_*` + profiles pool | Measured: batch *get* 1.29× faster; batch *put* 0.87× **slower** on 2-CPU host (negative result kept, §6 of report). |
| `thiserror` enum for all failures | Python exceptions map 1:1 (ValueError ← RasmError) in bindings; no panics on user data paths (fuzz-verified). |
| No SIMD | Core loops are latency-bound through PyO3 boundary; SIMD candidate (CA fold) already 0.5–2 µs/step; no measurable win justified. Rule documented: "SIMD only with measurable advantage". |
| No tokio | No I/O-bound paths in core; storage is synchronous file-sized buffers. Adding an async runtime would violate "no heavy deps without justification". |

## 3. Threading & memory model

- All public types are plain structs; `&mut self` for mutation — Rust
  guarantees no data races. For Python: one `MemoryPy`/`PackedStorePy` per
  thread is safe (GIL serializes access to one object; independent objects
  exercised in `TestThreadSafety`).
- rayon calls (`batch_get_parallel`, `batch_put_parallel`) run inside
  `par_iter`; called from Python they execute with the GIL released at the
  PyO3 boundary in the benchmark path… (note: the binding currently holds the
  GIL for these; measured wall time therefore includes no GIL contention —
  documented limitation for v0.8).
- Buffer reuse: `ca_step_sparse` reuses `to_update` HashSet allocation per
  call; PackedStore never reallocs per record (amortized Vec growth).
- Allocations counted only via RSS deltas (no allocator profiler in sandbox)
  — documented as unmeasured rather than faked.

## 4. Cross-language equivalence contract

Defined per component:
- **Exact** (bit-identical results): RNS enc/dec/ops, exact/partial memory
  queries, dedup/conflict counters, CA dense/sparse/parity (incl. ops),
  synthesis program + candidate count + accuracies, PackedStore bytes,
  storage serialized bytes, graph truth states.
- **Kernel-exact** with injected data: noisy retrieval (HV stream differs by
  design; pure kernel `noisy_query` compared with identical hvs).
- **Formula-exact with injected weights**: learning forward logits and grads
  (1e-9 vs numpy / 1e-3 vs finite-diff).
- **Qualitative only**: trained accuracies (RNG init streams differ),
  reservoir states (init differs), timing.

## 5. Binary format (v1)

```
offset  size  field
0       4     magic "RASM"
4       4     u32 version = 1
8       8     u64 payload_len
16      n*16  records: i64 key LE, i64 value LE
16+n*16 8     u64 FNV-1a(payload)
```
Python export/import: `storage::import_python_packed` reads raw "<qq" streams;
Python writes via `struct.pack` — byte identity tested.

## 6. Known limits (do not over-read)

- Software CAM/CA models remain *models*: no hardware CAM timing claimed.
- The 5.8×/16 B figures measure implementation storage, not "intelligence".
- partial_memory p95 ratio 0.8 (Rust slower at tail on 2-CPU noisy-neighbor
  host): kept in plots, not averaged away.
- Learning module reproduces the REFUTED L1 hypothesis near chance in Rust
  (0.505/0.500) — no re-tuning until it "passes".
