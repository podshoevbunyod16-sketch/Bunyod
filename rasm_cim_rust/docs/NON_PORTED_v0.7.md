# RASM-CIM v0.7 — Components NOT ported to Rust (with reasons)

Rule applied: a component is ported only if it is (a) algorithmic (not a
Python-library wrapper), (b) affects the core's numbers, (c) not a
measurement artifact we must keep in Python for cross-checking.

| # | Component (Python origin) | Status | Reason |
|---|---|---|---|
| 1 | sklearn experiments: digits/iris classifier probes (v0.6 §R1) | Not ported | Purpose is *dataset sanity of retrieval*, sklearn IS the experiment. sklearn has no Rust equivalent; porting would change the experiment, not just the implementation. |
| 2 | LLM literature comparison table (v0.6 `llm_compare.py`) | Not ported | Pure documentation/numbers-from-papers; no computation to accelerate. |
| 3 | Demo HTML generation (demo*.py) | Not ported | Reporting/UI tooling; belongs to Python research interface (§5 of v0.7 spec: Python stays the research interface). |
| 4 | FPE/hyperdimensional float bundles (v0.1/v0.2 `logic`, `hv`) | Not ported as float HV | The v0.6 memory path that WAS ported uses bipolar ±1 HV + RNS; float HV experiments showed no memory win (v0.2 F9) — negative result, no perf motivation. The *similarity kernel* used by memory is ported (core::hv_similarity). |
| 5 | LUT compute-in-memory model (v0.5 N2, REFUTED: slower than binary in Python) | Not ported | Negative result preserved in v0.5 report; porting would only re-measure a refuted path. |
| 6 | Reservoir *training* (ridge readout, v0.5) | Not ported | Ported: reservoir dynamics (step/collect) in `ca::EchoStateReservoir` for Arch-B probes. Readout ridge is numpy `lstsq`; linear-solver quality belongs to BLAS, not this core. Documented for v0.8 if needed. |
| 7 | `profiles_mobile.py` tracemalloc-based per-op RAM attribution | Partially | Rust reports process RSS + struct sizes; per-line allocator tracing has no stable equivalent here (see ERRORS #12). Memory *budgets and thread caps* are ported (profiles). |
| 8 | Episodic *context embeddings* (v0.5 `memory5` full variant) | Not ported | v0.6 `memory_system.py` is the reference ported; v0.5 superset kept in Python for history (v0.7 rule: don't rewrite architectures). |
| 9 | Resonator (v0.1/v0.3 `resonator3.py`) | Not ported | Multi-restart brute search; measured earlier as rarely beating brute force (v0.2 F4 fallbacks 4/20). No measured advantage to reproduce in Rust yet. |
| 10 | CAM *hardware* timing models | Not ported (deliberately) | Never claimed as hardware in Python either (software CAM simulation only). Rust core makes no CAM-hardware claims. |
| 11 | `benchmarks/llm_compare`, plots generation | Not ported | Python plotting (matplotlib) is the measurement presentation layer. |
| 12 | Streaming/paged store for mobile | Planned v0.8 | Current storage loads whole file buffers; §8 mobile "не загружать всю базу в RAM" needs mmap/chunked reader — designed in PLAN_v0.8, not faked now. |

## Non-portable by construction
- `numpy` RNG streams (`default_rng(seed)`) → Rust uses splitmix64 for own
  init and **injected weights** for exact cross-language checks (see
  ARCHITECTURE §4). Trained-weight bit-equality is out of scope by design.
- `sklearn`, `matplotlib`, `criterion html` (rust side has criterion ✓),
  `pytest` fixtures — remain in their native languages.
