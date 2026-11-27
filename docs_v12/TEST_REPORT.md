# TEST_REPORT.md — v0.12 (прогон 2026-11-26)

## Итог
| Набор | Команда | Результат |
|---|---|---|
| Исторические gate (v0.1–v0.10) | `python -m pytest test_rasm*.py rasm_cim_v8/tests tests_v9 tests_v10 -q` | **163 passed** |
| v0.11 | `python -m pytest tests_v11 -q` | **56 passed** |
| v0.12 новые | `python -m pytest tests_v12 -q` | **27 passed** (22 unit/integration + 5 property) |
| Итого | | **246 passed / 0 failed** |
| v0.8 freeze | md5sum -c | 51/51 OK |
| Baseline freeze (272) | md5sum -c | 0 mismatches |
| Rust v0.7 | hash | 451bbfa5… без изменений |

SKIP: нет. INCONCLUSIVE: только унаследованные из v0.10 (device fsync), см. docs_v10.

## Что доказывают тесты
- факт прослеживается до источника: fragment + position + source/doc_id/method/
  parser_version — ассерты в test_knowledge.py::test_teach_creates_entities_and_statement;
- повторный импорт идемпотентен (test_jsonl, test_props): imported=0, счётчики стабильны;
- конфликтующие утверждения НЕ затирают друг друга; status=CONFLICT; после retract одной
  стороны остаётся вторая (test_deny_assert…, test_polarity_conflicts…);
- функциональный конфликт (две столицы) → CONFLICT;
- retract → UNKNOWN (не FALSE); symmetric semantics;
- гипотезы не подтверждают факты до confirm_hypothesis;
- prefix-consistency replay property-тестом (interruption);
- torn last line не фатален; mid-file битая строка → replay_errors, остальные применяются;
- strict-импорт ничего не применяет; битый schema → отказ; provenance сохраняется.

## Найденное и исправленное
- CLI _open не делал replay (0 событий после seed) — исправлено и зафиксировано
  (cmd kb-info после seed → 60 entities / 34 statements / 94 events).
- update_statement объект-эвристика — упрощена до документированного конвента.
- Первый property-прогон test_reimport_twice_is_idempotent падал на MY assert
  (imported==statements) — тест выровнен (imported ≥ statements; r2.imported == 0).
- Seed-демо выявил ложную неоднозначность «Берлин» (строка данных v0.11) — исправлено
  данными (является_столицей), не семантикой.

## Bench (host tmpfs; результаты_v12/bench_kb.json)
500 фактов: build 0.41 s; экспорт 457 104 B; import 0.44 s (1017 событий);
query predicate is_a: ~238 µs (материализация 500 строк — не индекс).
Устройство НЕ измерялось.
