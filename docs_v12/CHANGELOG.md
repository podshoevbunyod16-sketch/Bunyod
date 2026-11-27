# CHANGELOG — v0.12 (концепты/сущности/provenance; 2026-11-26)

## Added
- Пакет `rasm_cim_v12.knowledge`: модели с стабильными id, schema rasm-cim-kb/1,
  Provenance (8 полей, включая fragment/position/parser_version/reliability),
  event-sourced KnowledgeStore (append-only + replay с torn-tail/mid-file терпимостью),
  Teach/Remember API (teach/remember/retract/update_statement/teach_hypothesis/
  confirm_hypothesis, рус.→канон предикатов), JSONL import/export (header, strict,
  corrupted-line isolation, idempotent re-import), conflicts()/reverse_links()/
  provenance_source фильтр, find_by_name с контролем неоднозначности алиасов,
  to_truth_graph() проекция.
- `rasm_cim_v12/cli.py`: kb-init/teach/kb-query/conflicts/aliases/import-jsonl/
  export-jsonl/kb-info/retract/seed-demo. Exit codes документированы.
- tests_v12/: 17 юнит/интеграционных + 5 property-тестов (hypothesis, derandomize),
  включая prefix-consistency replay и polarity-conflict resolution cycles.
- docs_v12/…, results_v12/{test_run.log,bench_kb.json}.

## Fixed (найдено и исправлено в ходе этапа)
- `_open()` в CLI возвращал нереплеяящийся store → kb-info показывал 0 событий после
  успешного seed. Исправлено: всегда init+replay (idempotent).
- update_statement: некорректное определение new_obj_is_entity — упрощено до
  документированного конвента (явный object_ = сущность).

## Not changed
- Всё историческое (v0.1–v0.11), включая TruthGraph9-семантику.
- demo_facts.jsonl: строка "Берлин столица Германия" → "Берлин является_столицей Германия"
  (устранение ложной неоднозначности типов; v0.11 поведение не затронуто).
