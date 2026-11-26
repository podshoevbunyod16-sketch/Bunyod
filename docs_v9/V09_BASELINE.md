# V09_BASELINE — v0.9 audit baseline (M0), fixed BEFORE any v0.9 code

Date 2026-11-25. Baselines for the "measured efficiency over pretty claims"
rule. Everything here was executed in this workspace (commands embedded so
any number can be re-derived).

## 0. Freeze state (history preserved)

| item | value |
|---|---|
| Rust v0.7 freeze hash | `451bbfa52e67a72b04c0046d9df4e9d5` (`find rasm_cim_rust -name '*.rs' -exec md5sum {} + \| sort \| md5sum`) — re-checked at v0.9 baseline time |
| v0.8 tree manifest | `results_v9/v08_freeze_manifest.md5` (51 files, md5 per file) — v0.8 becomes read-only history at v0.9 start |
| v0.8 suite | 109 passed (`test_rasm*.py` + `rasm_cim_v8/tests`, incl. M2-refactor parity) |

## 1. Where RAM is ACTUALLY consumed (measured this turn, host, n=1M unless noted)

1. **write-time index (transient, the big one at build)**: `format.write_v2`
   materializes `list[tuple[key, off]]` before sorting. Probe:
   build_s 0.956, **VmHWM 141.2 MiB**, VmRSS after 13.6 MiB → ~127 MiB
   transient Python tuples+ints at 1M. → M1 target.
2. **resident key/offset index: ALREADY compact** — `LazyStore._load_index`
   builds `array("q")` + `array("Q")` (16 B/record, bisect-compatible).
   **Do NOT "optimize" this again** (audit explicitly checked; a naive
   "compact the resident index" task would be a no-op).
3. **residue/inverted index (resident, the big one at query)**:
   `stream_memory._residue` = dict of Python-int lists: measured
   **+103.8 MiB @1M** (rss 42.5 → 146.2 with 3×N entries; ~35 B/record-entry).
   → M1 compact-residue target (arrays: theoretical 24 MiB @1M).
4. **HV layer**: solved in v0.8-M2-refactor (streaming regen, 0 resident
   bytes; lru mode = 0-hit trap at cap<N). v0.9 keeps streaming as default.
5. Baseline floor: bare python interpreter+module import here ≈ 13.6 MiB.

## 2. What already uses array / what is still dict-list-tuple

| structure | file | representation | v0.9 action |
|---|---|---|---|
| LazyStore `_keys/_offs` | storage/streaming.py | **array q/Q** | keep (reference impl for parity) |
| write-time sort list | storage/format.py `write_v2` | list[tuple] + ints | replace: sort-to-arrays (M1) |
| residue index | memory/stream_memory.py | dict → list[int] | replace: dict → array('q') buckets (M1) |
| `to_full_memory`, `_dict` tier | streaming/stream_memory | full dicts | keep as explicit dict-tier only |
| HVCache LRU | memory/hv_stream.py | OrderedDict | keep (bounded, measured; v0.9 re-measures on journal store) |
| BitpackIndex lists | hv/bitpack.py | list[int] | out of scope (v0.8 measured: adds 0 MiB) |

## 3. Operations that require FULL file rebuild (→ M2)

- append/update of any record on v2: v0.8 documented `update = write_v2 over
  all records` — measured **1.003 s @1M**, 0.010 s @10k (m6_scaling.json).
- no in-place page rewrite exists (would need free-slot management).
→ v0.9 v3 base + append-only journal targets single-record update **≤5 ms @1M**
(TARGET, not promised).

## 4. Where p95 comes from (→ M3 decomposition)

v0.8 measured (m2_streaming.json): p95 ratio vs in-RAM dict = 25.4–35.6×
FAIL of ≤2× target. Component floor re-measured this turn @100k hot-page:
bisect 0.608 µs, full get on cached page 2.08 µs (unpack+run-scan+cache
lookup), miss 0.59 µs; the cold path adds per-page 32 KiB pread (~5–20 µs
device-dependent) + crc32 (~10 µs per page, the value that motivated FNV→crc32
in v0.8). So p95 >> p50 is page-READ dominated (cache miss), not index
dominated. M3 measures LRU/warm/mmap against exactly this split and keeps the
FAIL if the floor is physical.

## 5. What can change without breaking compatibility

- internal build/sort strategy of the writer (bytes-out identical if we
  reproduce the same page layout — v0.9 pins v2 golden sha256:
  `6ca78bb83ca45e820b3469559ab0b7e854455b8c6631bb8feb9a08a7d5993ceb` for the
  fixed 8-record fixture);
- residue index representation (pure RAM, no file format);
- NEW file format allowed **only as an explicit new version (v3)**, documented,
  v1/v2 readers kept (spec M2: "не изменять v2 молча").
- immutable: v1 layout, v2 layout, read-side dedup/conflict semantics
  (TRUE/CONFLICT/UNKNOWN + sorted(set(vals))), first-max tie rule, crc32
  page slots, file magic b"RASM".

## 6. Historical v0.8 numbers v0.9 MUST NOT regress without noting

| metric (v0.8, host) | value |
|---|---|
| 100k stream RSS @query | 17.6 MiB (m6) |
| 100k build | 0.098 s; 1M build 0.871–0.956 s |
| exact p50/p95 @100k cached | 0.003–0.016 ms / p95 ratio 25–35× dict |
| HV streaming | 0 resident bytes; combo p50 38.1 ms @100k, 410 ms @1M; 0 mismatches @1k/20k/100k/1M |
| formats | v1/v2 goldens above; v0.6 PackedStore layout identity test green |
| tests | 109 passed at v0.9 baseline |

## 7. Environment

Host: 2 CPU, 1982 MiB RAM, Python 3.13.18, numpy available, no thermal
sysfs (verified — "available": false, not a core failure), atheris+hypothesis
installed. Android ARM64: **no device in this workspace** — user-reported v0.8
numbers (38.7 MiB @20k, 56.4 MiB @100k legacy HV) are inputs; M4 stays OPEN.
