# V08_COMPONENT_MAP.md — карта компонентов: источник → v0.8 Python

Правило: v0.8 = новый пакет `rasm_cim_v8/`; старые пакеты переиспользуются
через импорт (без правок). Rust-колонка — только как спецификация
форматов/семантики (заморожена).

| Компонент | Python-источник | Rust v0.7 (спецификация) | v0.8 действие |
|---|---|---|---|
| Бинарный формат v1 | — (был только в Rust `src/storage`) | ARCHITECTURE §5 | `v8/storage/format.py`: writer+reader v1, побайтно-совместим (golden-тест против bytes замороженного CLI) |
| Формат v2 (постраничный) | — | — (НОВОЕ в v0.8) | `v8/storage/format.py`: per-page FNV, footer index; v1↔v2 конвертер |
| PackedStore | v0.6 `compute_core.py` | `src/compute` | переиспользуется импортом; `from_bytes` добавлен в v8-обёртку (в v0.6 его не было) |
| LazyStore / paging / LRU | — | планировалось в Rust v0.8-M2 ⇒ перенесено в Python v0.8 | `v8/storage/streaming.py`: open_lazy, read_records, pread+mmap, cache budget, corrupt-page policy |
| MobileStreamMemory (in-RAM+файл гибрид) | `memory_system.py` (тяжёлый HV float) | `src/memory` | `v8/memory/stream_memory.py`: dict exact-index + residue index + bitpack HV; build/query/serialise поверх format |
| RNS | v0.4 `rns4.py` | `src/rns` | импорт as-is (fuzz-цель decode_unsigned) |
| КА rule110 semantics | v0.4 `ca4.py` (legacy таблица) | `src/ca` (зеркало legacy; ERRORS #7) | `v8/semantic/rule110.py`: legacy + `rule110_canonical` флаг, 8-комбо тест; ca4 не правится |
| TruthGraph | v0.4 `graph4.py` | `src/graph` | импорт as-is (fuzz-цель update/retract) |
| HV noisy retrieval | v0.6 `memory_system.py` | `src/memory` | `v8/hv/bitpack.py`: ±1→{1,0}, int.xor+bit_count; numpy и numba-ветки; НЕ заявляется как NEON |
| Benchmarks | `rasm_cim_v6_v2/experiments`, v0.7 harness | `benches/`, `benchmarks/` | `v8/bench/bench8.py` (M2/M3/M5/M6/M7), свой CSV/JSON формат, Python-side замеры только |
| Fuzzing | — | property.rs (стабильный substitution) | `v8/fuzz/`: Atheris (coverage-guided) + Hypothesis (property) + fixed-seed random; разделение зафиксировано |
| Мобильный пакет | `mobile/profiles_mobile.py` (профили) | `src/profiles`, `mobile/TERMUX.md` | `rasm_cim_v8/mobile/`: install/run скрипты Termux, harness замеров (start time, RSS, thermal best-effort), статус-чеклист M1 |
| Cross-version regression | test_rasm*.py (57) | test_cross_language.py | `v8/tests/test_cross_version.py`: импорт v1-файлов Rust CLI, сериализация/десериализация roundtrip, mobile profile |

## Границы ответственности v0.8 (что вне)
- Никаких Rust-сборок/правок; «Rust быстрее»-утверждения по Python-тестам
  запрещены и не делаются.
- Исторические эксперименты не пересчитываются «под новые оптимизации»:
  legacy-путь обязан воспроизвести старые числа (regression test сверяет
  candidates=238, sparse ops, dedup/CONFLICT и т.д. через v0.6/venv-коды).
