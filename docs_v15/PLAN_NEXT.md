# PLAN_NEXT.md — фаза C и фаза D (финализация)

## Фаза C — единая точка входа `python -m rasm_cim`
- Режимы: Chat / Ask / Teach / Memory (список/экспорт/reset) / Evaluate / System Info.
- Команды: help, exit, reset, teach, inspect, evidence; stdout=результат, stderr=диагностика;
  exit codes 0/2/3/4; всё offline; demo-корпус по opt-in (flag --demo или env RASM_CIM_DEMO=1);
  `python -m rasm_cim_v9.cli` продолжает работать (legacy path preserved).
- System Info: версия пакета, python, платформа, путь KB, счётчик событий, freeze-статус.
- Ask выбирает reasoning pipeline (v0.14) + v0.15 artifacts при наличии; Chat = диалог (v0.13)
  с теми же инвариантами; опция `evidence` печатает st_id→provenance.

## Фаза D — регрессионная проверка и упаковка
- Полный прогон всех 321+ тестов + исторических 163; freeze manifest re-verify; Rust-хэш.
- Host bench: p50/p95/p99 + peak RSS (уже есть в results_v11..v15); Android/Termux —
  НЕ измерялся: нужен пользовательский прогон на устройстве; цифры host не подставляются
  в устройство-гейты; flash-fsync остаётся INCONCLUSIVE.
- README.md обновить до фактических возможностей v0.15; pyproject packages v8/v9/v11…v15 +
  rasm_cim; архив финального состояния (без кэшей/venv/.git); PR-описание
  с разницей «что сделано / что нет» к merge в main.
- Если что-то Incomplete — честно: пометить в FINAL_REPORT, не заявлять end-to-end готовым.
