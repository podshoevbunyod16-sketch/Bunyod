# RASM-CIM — Residue-Addressable Streaming Memory (v0.10)

Experimental associative memory over a compact streaming store: exact/partial/noisy
lookups, crash-safe append-only journal, hypervector recall, formal knowledge graph
— **stdlib-only Python, offline, ARM64/Termux-friendly**. This is a research build;
it does not claim language understanding (roadmap: docs_v9/PLAN_v0.10.md, external
spec: audit + v0.11+ plan in docs_v9/V09_BASELINE_AUDIT.md).

Quick start:
```bash
python -m rasm_cim_v9.cli bench --n 50000     # 15-сек витрина: p50/p95/p99, v0.8 vs v0.10
python -m rasm_cim_v9.cli crash               # живой SIGKILL-демо восстановления
python -m pytest test_rasm*.py rasm_cim_v8/tests -q   # 109 passed
python -m pytest tests_v9 tests_v10 -q                 # 54 passed
```
Docs: docs_v10/{ARCHITECTURE,CHANGELOG,TEST_REPORT,BENCHMARK_REPORT,LIMITATIONS,REPRODUCE}.md
Install: `pip install -e .` (dev extras: -r requirements-dev.txt; numpy optional).
History v0.1–v0.9 immutable: reports + results in docs_v*/results_v*/.
