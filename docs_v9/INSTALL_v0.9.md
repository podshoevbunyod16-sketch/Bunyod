# Install — RASM-CIM v0.9 (desktop / host)

1. Unpack `RASM-CIM_v0.1-v0.9_PYTHON_FULL_2026-11-25.zip` anywhere (keeps
   v0.1–v0.8 for the A/B; nothing outside the tree is needed).
2. Requirements: **Python ≥ 3.10, stdlib only** for all v0.9 functionality
   (`array`, `mmap`, `struct`, `heapq`, `bisect`). numpy is OPTIONAL and only
   accelerates v3 base-build sorting (auto uses it at n≥200k when importable).
   Test extras: `pip install pytest hypothesis`; fuzzing extras: `pip install atheris`
   (requires a C toolchain for libFuzzer runtime; atheris ships wheels for common platforms).
3. Verify:
   `python -m pytest test_rasm*.py rasm_cim_v8/tests -q`  → expect 109 passed
   `python -m pytest tests_v9 -q`                          → expect 40 passed
4. API: `from rasm_cim_v9.memory.store9 import MobileMemory9`
   - `MobileMemory9.build(path, facts_iterable, page_recs=2048)` / `.open(path)`
   - `.update(k,v)` / `.update_txn([...])` (journal txns, fsync per txn)
   - `.query_exact / .query_partial / .scan / .query_noisy / .open_hv_stream / .open_hv_legacy`
   - `from rasm_cim_v9.storage.journal import open_any` reads v1, v2 AND v3 files.
   Migration: `python -c "...build_v3_base..."` from a v2 path's records — see
   tests_v9/test_m1_m2.py::test_v2_to_v3_migration for the canonical snippet.
5. Android/Termux: `rasm_cim_v9/mobile/README_v09.md` (device A/B is OPEN;
   no device numbers are shipped in this archive — by design).
