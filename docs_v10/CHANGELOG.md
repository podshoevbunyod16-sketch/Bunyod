# CHANGELOG — v0.10 (стабильность и надёжность ядра; 2026-11-26)

## Added
- `pyproject.toml` (name rasm-cim, v0.10.0, requires-python ≥3.10, extras dev/numpy,
  console-script `rasm-cim-demo`) — проверено `pip install -e` в чистой venv
  (импорт, CLI-вызов, рантائم-смок). Wheel-сборка — вне объёма (см. LIMITATIONS).
- `requirements-dev.txt` — декларация pytest/hypothesis/atheris (раньше их отсутствие
  молча ломало collection: воспроизведённый факт).
- `rasm_cim_v9/cli.py` + `python -m rasm_cim_v9.cli`: build | ask | partial | update |
  bench | crash. `crash` = живой демо SIGKILL писателя в 4 точки журнала → повторное
  открытие intact (первый прогон: records 5000→5145, get(7)=TRUE, VERDICT PASS).
- `bench_v10/profile_core.py` — воспроизводимый профилировочный конвейер: 4 объёма
  (1k/10k/100k/1M) × 10 ячеек × 3 повтора, полная изоляция процессов, **p99 добавлен**,
  tracemalloc peak/cur, throughput, файл-сайз, v0.8(v2) против v0.10(v3) на одних данных.
- `tests_v10/test_reliability.py` (14): SIGKILL в/перед compact() 7 задержками — файл
  всегда валиден (или старая, или новая база, никогда гибрид); 11 точек kill внутри
  журнальной фазы — префиксное свойство записей; context-manager V3Store без утечки fd;
  двойной close() тихий.
- `V3Store.__enter__/__exit__` (journal.py) — production-polish против утечек fd.
- `results_v9/fuzz_atheris_full.json` — свежий полноцелевой ре-прогон 5 atheris-мишеней
  (закрытие учётного пробела аудита; fuzz_atheris.json не переписывался).
- `docs_v9/V09_BASELINE_AUDIT.md`, `docs_v9/PLAN_v0.10.md` — вход в этап (разрешён §2/§10).
- Документы §9 этого каталога (ARCHITECTURE/LIMITATIONS/TEST_REPORT/BENCHMARK_REPORT/
  REPRODUCE/CHANGELOG).

## Changed (поведение совместимо; всё под тестами)
- `V3Store` — только аддитивный ctx-manager; формат v3, журнал, семантика get — без изменений.

## Not changed (намеренно)
- v0.1–v0.8, Rust v0.7 (hash-подтверждение в TEST_REPORT), форматы v1/v2, золотые хэши.
- p95≤2×dict остаётся FAIL (структурный; в v0.10 не трогался ради статуса).
- Авто-политика compact НЕ введена: кривая open-vs-#frames (BENCHMARK_REPORT) задаёт
  рекомендацию порога, автоматика — после устройствых данных (M4 открыт).

## Device (M4-хвосты v0.10)
- Извлечение `v09_v3_store`/`v09_v3_journal_updates` из сырых телефонных JSON и r3-прогон
  RSS остаются за пользователем ( Termux ); команды — `docs_v10/REPRODUCE.md` §4.
  До их появления любые flash-fsync вердикты = INCONCLUSIVE (не повышаем статус).
