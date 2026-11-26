# TEST_REPORT.md — v0.10 (прогон 2026-11-26 11:58 UTC, свежая среда)

| Набор | Команда | Результат |
|---|---|---|
| Корневые v0.1–v0.7 | `python -m pytest test_rasm*.py -q` | **57 passed** (3 warnings = matplotlib deprecencies в plots-коде) |
| v0.8 | `python -m pytest rasm_cim_v8/tests -q` | **52 passed** |
| v0.9 | `python -m pytest tests_v9 -q` | **40 passed** (16 storage/journal + kill-injection, 5 hypothesis-свойств, 6 fuzz-регрессий, 12 граф A/B) |
| v0.10 новые | `python -m pytest tests_v10 -q` | **14 passed** (kill-в-compact ×7 задержек, 11 точек kill в журнальной фазе, fd-утечки, двойной close) |
| Итого | | **163 passed / 0 failed** |
| Заморозка v0.8 | `md5sum -c results_v9/v08_freeze_manifest.md5` | **51/51 OK** |
| Rust v0.7 | `find rasm_cim_rust -name '*.rs' -exec md5sum {} + \| sort \| md5sum` | **451bbfa52e67a72b04c0046d9df4e9d5** (без изменений) |
| Упаковка | чистая venv + `pip install -e .` | импорт MobileMemory9 + build/open/query **OK**, `rasm-cim-demo ask` **OK** |
| atheris (все 5 целей, 60 с) | `bench_v9/fuzz_m5_atheris.py …` | rc=0; новых крэшей нет; `results_v9/fuzz_atheris_full.json` |
| CLI-демо | `python -m rasm_cim_v9.cli crash` | 4×SIGKILL в журнал → reopen чистый, records 5000→5145, get(7)=TRUE, **VERDICT PASS** |

## Покрытие требований промпта (§4 v0.10 «корректность, повреждённые и частичные
записи, повторное применение, восстановление после прерывания»)
- повреждённые/partial: corrupt-page тесты (v0.9 m1_m2 #10–11), torn-tail произвольной
  длины (hypothesis P2, 40 примеров), bad-crc с преемниками → JournalCorruptError.
- повторное применение: dup-seq frames пропускаются (тест + свойство идемпотентности P3).
- прерывания: SIGKILL на 3 стадиях append (реальный signal, не симуляция) + **7 задержек
  kill вокруг compact** (закрытая дыра аудита №4).
- кросс-версионность: open_any(v1/v2/v3), миграция v2→v3 с идентичными ответами.

## Найденное и исправленное В ХОДЕ этого этапа
- Профилировочный конвейер сам был искажён: tracemalloc во всех ячейках завышал
  латентность в разы (partial p50 6145 vs 166 µs). Исправлено (TRACE только для
  memory-ячеек), матрица перезапущена начисто. Цифры BENCHMARK_REPORT — из чистого
  прогона. Это иллюстрация принципа «не оптимизируй, не измерив стоимость инструмента».

## Чего тесты НЕ доказывают
- Понимания языка (слоя нет — см. LIMITATIONS); только детерминированный поиск.
- Устройственной fsync-латентности (M4-хвосты открыты: 2 arm-числа в сырых JSON,
  r3-RSS-прогон не выполнен) — статус INCONCLUSIVE, гейт ≤5 ms на flash НЕ закрыт.
- Чисел на Python 3.14 (среда 3.13.18) и wheel-дистрибуции (только editable).
