# CHANGELOG — v0.13 (диалог и контекст; 2026-11-26)

## Added
- `rasm_cim_v13/dialogue/`: context.py (ContextStore rasm-cim-context/1, max_turns=20,
  TTL=1800s, атомарные записи, suspend/flush батчинг, corrupt-tolerance),
  splitter.py (нумерация/'?'/';'/союзы; ≤8 подвопросов; порядок сохранён; лидирующие
  союзы и дискурсивные «а/и» отбрасываются),
  pronouns.py (род/число, личные + демонстративы без «это/то», эвристика рода),
  manager.py (DialogueManager: pollution-guard FIRST — hypothetical/quote/teach;
  per-подвопрос KB-lookup с TRUE/FALSE/UNKNOWN/CONFLICT; мульти-сущности веером;
  CLARIFY при неоднозначности; запись turns с statement_ids),
  session.py (export/show/reset).
- `rasm_cim_v13/cli.py`: ask / show-context / reset-context / export-session
  (exit 4 при CLARIFY).
- tests_v13/: 40 тестов — unit (context/TTL/corrupt/splitter/pronouns),
  сценарии 2–5 реплик (смена темы, возврат, множественные сущности, неоднозначные
  местоимения, CONFLICT показан, pollution-guard, teach-put provenance),
  restart-восстановление, CLI smoke (вкл. exit 4).
- В репликах turns сохраняются entity_ids и statement_ids (provenance-ссылки).

## Behaviour changes (additive)
- Teach-парсер: типы сущностей наследуются из KB (нет дублей Francia-thing vs Francia-place).
- leading «А/И/Но/Да» отдельного подвопроса считается дискурсивной частицей и отбрасывается.
- Follow-up без вопросительных слов + разрешённое местоимение → трактуется как вопрос (ask_fact).
- Plural «их/о них» резолвится по сущностям ПОСЛЕДНЕГО user-turn (веерный разбор).
- Context flush батчится (suspend/flush, 1 fsync на группу хода) — см. BENCHMARK.

## Fixed (найдено в ходе этапа)
- CLI v0.13/v0.12 _open без replay → пустые ответы (исправлено в v0.12, контекст v0.13 без изменений).
- seed-типинг создавал дубликаты сущностей с одним именем разных типов («Пингвин» thing/animal)
  → единая типизация seed (вопрос-совпадения теперь дают одну сущность).
- record_entities: kb_id=None ломал dedupe (notoriety MRU) — исправлено (or name).
- TTL-тест: save() по контракту бампает updated_at — тест переписан на прямую подмену.
- «это» убрано из демонстративов (давала ложный CLARIFY «Киты — это кто?»).
- Missing keyword «население» → population_of не матчился (только «населения»); добавлено «населени».
