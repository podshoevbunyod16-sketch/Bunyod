# ARCHITECTURE.md — RASM-CIM v0.12 (концепты, сущности, отношения, provenance)

## Единый модуль с чёткими границами (по ТЗ допустимо): пакет `rasm_cim_v12.knowledge`
```
models.py      Entity/Statement, SCHEMA=rasm-cim-kb/1, стабильные id (sha256),
               ENTITY_TYPES, STATEMENT_KINDS, FUNCTIONAL_PREDICATES
provenance.py  Provenance: source/doc_id/fragment/position/added_at/method/
               parser_version/reliability (категориальная low|medium|high)
store.py       KnowledgeStore — event-sourced, append-only events.jsonl + meta.json;
               replay с терпимостью к torn-tail и mid-file ошибкам (счётчики, не тишина)
teach.py       Teach/Remember API: teach/remember/retract/update_statement/
               teach_hypothesis/confirm_hypothesis, canon_predicate (рус.→канон)
jsonl.py       экспорт/импорт rasm-cim-kb/1: header, валидация, strict-режим,
               idempotent re-import, corrupted-line isolation
```

## Поток и границы слоёв
TruthGraph9 НЕ изменён. KB — отдельный слой: утверждения → события → сводка (replay).
Запросы statuses TRUE/FALSE/UNKNOWN/CONFLICT — семантика, совместимая с TruthGraph9
(для подтверждённых kind=user_assertion|extracted_fact; гипотезы в подтверждении не участвуют).
`kb.to_truth_graph()` проецирует KB → GraphAdapter (обмен с существующим графом v0.9/v0.11).

## Модель данных
- Entity id: `ent_<sha256(type|fold(name))[:16]>`; Statement id:
  `st_<sha256(subject|predicate|object|polarity|kind|doc_id)[:16]>` — повторный импорт
  идемпотентен по построению.
- Statement: polarity (assert|deny), kind (user_assertion|extracted_fact|hypothesis),
  object_entity|object_literal ровно один, status active|retracted (наследуется от событий),
  replaces (для update→append-only), retract_reason.
- Функциональные предикаты (capital_of …): >1 различных active-объекта на (s,p) →
  CONFLICT (never auto-resolved): оба сохраняются с provenance.
- Гипотезы (kind=hypothesis): видны, запрашиваемы, НО не подтверждают факты;
  `confirm_hypothesis` → kind=extracted_fact (явное действие).
- retract = новое событие statement_retracted; история append-only; update =
  retract+assert с replaces-ссылкой.

## Восстановление после прерывания
Каждое событие — одна строка JSONL + fsync. torn-tail при replay пропускается
(счётчик replay_torn_tail), mid-file битая строка → replay_errors c номером строки,
остальные применяются. Префиксная согласованность replay property-tested (hypothesis,
tests_v12/test_props.py).

## Запросы
subject/predicate/object фильтры, reverse_links(entity) (где сущность — объект),
provenance_source фильтр, include_retracted, conflicts() → polarity_conflict и
functional_conflict группы. find_by_name → found|ambiguous|not_found; alias на ≥2
сущностях НИКОГДА не сливается автоматически (type_hint — явное уточнение).

## CLI (`python -m rasm_cim_v12.cli`)
kb-init · teach · kb-query · conflicts · aliases · import-jsonl [--strict] ·
export-jsonl · kb-info · retract · seed-demo (offline демо-факты из v0.11).
Exit codes: 0 ok · 2 input · 3 not found · 4 ambiguous.

## Известные решения
- `open()` всегда = init+replay (idempotent); «чистый конструктор без replay» убран после
  найденного в этом этапе дефекта (append проходил, replay нет — см. CHANGELOG).
- Экспорт выкидывает schema/seq/ts и пере-создаёт их при импорте; provenance НЕ трогается.
