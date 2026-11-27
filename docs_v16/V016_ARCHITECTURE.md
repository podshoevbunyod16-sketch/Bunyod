# V016_ARCHITECTURE — устройство слоя v0.16

Пакет `rasm_cim_v16` (16 модулей, **6377 строк**), только стандартная
библиотека. Поверх `rasm_cim_v9` как субстрата; исторические деревья не
импортируются (проверяется AST-тестом `test_v16_imports_only_v9_and_itself`).

Все числа в этом документе получены исполнением кода, а не переписаны из
комментариев.

---

## 1. Разделение ответственности

| Модуль | Обязанность | Ключевой интерфейс |
|---|---|---|
| `normalization.py` | сравнимые формы, стемирование, транслитерация | `normalize`, `squeeze`, `stem_candidates`, `transliterate`, `is_proper`, `word_tokens` |
| `relations.py` | словарь отношений как **данные** | `Relation`, `RelationSchema`, `DEFAULT_SCHEMA`, `register_dict` |
| `entity_resolver.py` | разрешение упоминаний в сущности | `EntityResolver.resolve/intern/link_alias`, `Candidate`, `Resolution` |
| `language_parser.py` | предложение → промежуточное представление | `LanguageParser.parse` → `list[QueryIR]` |
| `knowledge_graph.py` | типизированные утверждения + индексы | `KnowledgeGraph.assert_triple/find/objects_of/subjects_of/neighbours/walk/retract` |
| `relation_extractor.py` | `QueryIR` → факт (с порогом уверенности) | `RelationExtractor.teach/bind_question` |
| `inference_engine.py` | ограниченный вывод с доказательствами | `InferenceEngine.query/yes_no/explain` |
| `context_engine.py` | тема, активные сущности, местоимения, история | `ContextEngine.touch/resolve_pronoun/switch_topic/integrity_check` |
| `evidence_manager.py` | доказательства и их происхождение | `EvidenceManager.add/evidence_for` |
| `contradiction_detector.py` | конфликты, циклы, функциональные нарушения | `ContradictionDetector.scan/conflicts/resolve` |
| `memory_manager.py` | 4 слоя памяти, дубли, размеры | `MemoryManager.record/tiers/size_bytes` |
| `answer_engine.py` | 3 режима × 6 статусов ответа | `AnswerEngine.answer_definition/answer_wh/answer_yesno` |
| `dialogue_manager.py` | конвейер реплики и 14 команд `:cmd` | `DialogueManager.say/command` |
| `persistence.py` | атомарная запись, резервные копии, откат | `Store.attach/save/load/backup/rollback/needs_migration` |
| `migrate.py` | v1 → v2 без потерь | `migrate_file`, `migrate_context_file`, `legacy_to_v2` |
| `cli.py` / `__main__.py` | интерфейс | `chat ask say teach migrate rollback inspect stats entities graph memory rules context conflicts evidence explain help demo` |

Конвейер одной реплики (`DialogueManager.say`):

```
текст → LanguageParser.parse → [QueryIR]
      → (intent) teach:  RelationExtractor.teach  → KnowledgeGraph.assert_triple
                                                    → EvidenceManager.add
                                                    → MemoryManager.record
                                                    → ContradictionDetector.scan
      → (intent) ask:    RelationExtractor.bind_question (сущности → id)
                        → ContextEngine (местоимения, фокус, тема)
                        → AnswerEngine (прямые факты → InferenceEngine)
                        → Answer{concise, normal, detailed, status, facts, evidence}
```

Плохой разбор **не** становится утверждением: `RelationExtractor` пишет факт
только при `confidence >= ASSERT_THRESHOLD = 0.50` и при «чистом» разборе
(`clean_parse`), иначе возвращает уточняющий вопрос.

---

## 2. Семантическая модель (30 отношений)

Отношение — это запись `Relation` с направлением, допустимыми типами
аргументов, функциональностью, симметричностью, транзитивностью, поверхностными
формами, вопросительными словами и шаблоном рендеринга. Пустой кортеж типов =
«любой».

```
alias_of        inv=has_alias                             subj=()          obj=()
same_as         inv=-              symmetric                              subj=()  obj=()
related_to      inv=-              symmetric                              subj=()  obj=()
instance_of     inv=has_instance                          subj=()          obj=(CLASS,)
has_instance    inv=instance_of                           subj=(CLASS,)    obj=()
subclass_of     inv=has_subclass   transitive             subj=(CLASS,)    obj=(CLASS,)
has_subclass    inv=subclass_of    transitive             subj=(CLASS,)    obj=(CLASS,)
part_of         inv=has_part       transitive             subj=()          obj=()
has_part        inv=part_of        transitive             subj=()          obj=()
has_property    inv=-                                     subj=()          obj=()
has_gender      inv=-              functional             subj=()          obj=()
discovered      inv=discovered_by                         subj=(PERSON,)   obj=(THING,WORK,EVENT)
discovered_by   inv=discovered                            subj=()          obj=()
formulated      inv=formulated_by                         subj=(PERSON,)   obj=(THING,WORK)
formulated_by   inv=formulated                            subj=()          obj=()
created         inv=created_by                            subj=(PERSON,)   obj=(THING,WORK)
created_by      inv=created                               subj=()          obj=()
wrote           inv=written_by                            subj=(PERSON,)   obj=(WORK,)
written_by      inv=wrote                                 subj=()          obj=()
located_in      inv=contains       transitive             subj=()          obj=(PLACE,)
contains        inv=located_in     transitive             subj=(PLACE,)    obj=()
causes          inv=caused_by      transitive             subj=()          obj=()
caused_by       inv=causes         transitive             subj=()          obj=()
happened_at     inv=event_of                              subj=()          obj=(EVENT,)
event_of        inv=happened_at                           subj=()          obj=()
born_at         inv=birthplace_of                         subj=()          obj=(PLACE,)
birthplace_of   inv=born_at                               subj=()          obj=()
studied         inv=-                                     subj=(PERSON,)   obj=()
known_for       inv=-                                     subj=(PERSON,)   obj=()
```

Все 13 отношений, перечисленных в задании, присутствуют.

**Расширяемость без правки ядра** — `schema.register_dict({...})` добавляет
отношение; парсер подхватывает его `surface`-строки, механизм вывода — флаги
транзитивности/симметричности, рендеринг — шаблон. Проверено тестом
`test_a_semantics.py::test_schema_is_extensible_without_core_changes`.

**`is_a` не существует.** В словаре такого отношения нет; попытка его
использовать даёт `KeyError`. `«Ньютон — это Исаак Ньютон»` даёт `alias_of`
(тест `test_b_alias_not_is_a`), направление — от поверхности к каноническому
имени (тест `test_b_alias_direction_is_surface_to_canonical`).

**Нарушение типов не отбрасывается молча**: факт сохраняется, но
`confidence` снижается до ≤ 0.6 и добавляется запись в `type_warnings`.

---

## 3. Разрешение сущностей

Порядок шагов в `EntityResolver.resolve` (первое непустое попадание выигрывает,
но кандидаты сравниваются по уверенности):

| Шаг | Статус | Уверенность |
|---|---|---|
| дословное каноническое имя | `EXACT` | 1.00 |
| зарегистрированный псевдоним | `ALIAS` | 1.00 |
| регистр/пунктуация/титул («сэр Исаак Ньютон») | `NORMALIZED` | 0.98 |
| транслитерация («Moskva» → «Москва») | `TRANSLIT` | 0.88 |
| падежная форма того же имени (совпадает число слов) | `MORPHOLOGICAL` | 0.86 |
| односложное совпадение с более длинным именем («Ньютона» → «Исаак Ньютон») | `PARTIAL` | 0.78 |
| частично объяснённое упоминание (число слов не совпало) | `MORPHOLOGICAL` | 0.72 |
| опечатка, расстояние Левенштейна = 1 | `FUZZY` | 0.72 |
| односимвольное окончание у нарицательного («физика» vs «физик») | `MORPHOLOGICAL` | 0.60 |
| несколько равносильных кандидатов | `AMBIGUOUS` | 0.00 |

Правила, зафиксированные тестами:

* Разрыв между лучшим и вторым кандидатом ≤ `AMBIGUITY_MARGIN = 0.05` ⇒
  `AMBIGUOUS` и уточняющий вопрос (два разных «Ньютона» не сливаются).
* Морфология **только порождает кандидатов**: хранимый текст никогда не
  переписывается, сущности не объединяются, кроме явного `link_alias`/`same_as`.
* `intern` переиспользует сущность только для `MORPHOLOGICAL`/`NORMALIZED`
  с уверенностью ≥ 0.75 при единственном кандидате: «физиком» → «физик»,
  но «Ньютона» → «Исаак Ньютон» **не** сливается (это может быть другой человек).
* Односимвольное окончание у нарицательного существительного оценивается в
  0.60: «физика» (наука) и «физик» (человек) не склеиваются.

---

## 4. Граф знаний

`Statement` = `(statement_id, subject, predicate, object, polarity, status,
confidence, source, created_at, modified_at, evidence[], provenance{}, history[],
reinforce)`.

Индексы (точечный поиск без обхода всей базы):

* `_by_key[(s,p,o,polarity)]` — дедупликация и усиление;
* `_spo[(s,p,polarity)] → {o: id}` и `_pos[(p,o,polarity)] → {s: id}` —
  **полярность входит в ключ намеренно**: одно и то же утверждение как истина и
  как ложь — это конфликт, и индекс без полярности позволял второму утверждению
  «спрятать» первое;
* `_by_subject`, `_by_object`, `_by_pred` — срезы для `:graph` и `:memory`.

Повторное обучение **усиливает** запись (`reinforce += 1`, запись в `history`),
а не дублирует её. `retract()` симметричен и никогда не удаляет молча —
утверждение получает `status="retracted"`.

Самопетля тождества (`X alias_of X`, возникающая при миграции старого
псевдонима) сохраняется с `status="redundant"`: она остаётся в файле и в
`stats()["redundant"]`, но не индексируется и не участвует ни в одном запросе.

---

## 5. Механизм вывода (отдельный модуль)

5 встроенных правил, каждое с условиями применимости:

| Правило | Уверенность | Макс. цепочка | Условие |
|---|---|---|---|
| `alias_bridge` | ×0.98 | 2 | подстановка сущности по `alias_of`/`has_alias`/`same_as` |
| `symmetric` | ×0.99 | 1 | только для симметричных отношений |
| `inverse` | ×0.95 | 1 | только если `inverse(Q) == P` |
| `instance_subclass` | ×0.90 | 3 | `instance_of(x,C) + subclass_of(C,D) ⇒ instance_of(x,D)` |
| `transitive` | ×0.85 | 3 | только для отношений с флагом `transitive` |

Ограничения: `max_depth = 3`, `max_nodes = 400` (настраиваются из CLI
`--max-depth/--max-nodes`). Посещённые цели запоминаются — циклы не зацикливают
обход. Транзитивность **не** применяется огульно: `instance_of` в словаре не
транзитивно, `subclass_of` — транзитивно.

Каждый выведенный факт несёт: `source_statement_ids`, применённые правила,
путь по графу (`proof`), `depth`, `confidence`, `status ∈ {direct, inferred}`.
**Выведенные факты никогда не записываются в граф как подтверждённые** —
проверяется тестом `test_d_inference_is_never_persisted`.

`transitive` работает и при двух связанных концах («палец — часть руки?»):
в этом случае ищется середина, а не перечисляется всё достижимое.

---

## 6. Контекст диалога

`ContextEngine`: сессия, стек тем, активные сущности со значимостью
(`salience`), фокус `(subject_id, predicate, object_id)`, последнее событие,
история реплик, нерешённые уточнения (`pending_clarification`).

* Местоимение разрешается по **роду, роли и значимости**, а не «последнему
  существительному»; при неоднозначности — `PR_AMBIGUOUS` и вопрос.
* Смена темы определяется **до** обновления значимости
  (`classify_topic` → `switch_topic`), иначе новая сущность выглядела бы
  частью старой темы и переключение никогда бы не срабатывало.
* `return_to_previous` восстанавливает предыдущую тему из стека.
* `integrity_check()` и миграция контекста v1 → v2; состояние переживает
  перезапуск процесса (тест `test_g_context_survives_a_restart`).

---

## 7. Память: 4 слоя

| Слой | Содержимое | Ограничение |
|---|---|---|
| A. Рабочая | последние записи/события | `WORKING_CAP = 256`, вытеснение со счётчиком |
| B. Эпизодическая | реплики, активные сущности, темы | история ограничена, старое вытесняется |
| C. Семантическая | факты графа и сущности | без ограничения, индексируется |
| D. Процедурная | словарь отношений и правила вывода | расширяется пользователем |

Каждая запись: уникальный id, источник, время создания/изменения, надёжность,
тип знания, связи, история обновлений. Дубликаты сливаются, противоречия
сохраняются **обеими сторонами** с происхождением.

Измерено на 1000 фактах: A = 8/256, B = 8 реплик, C = 4 факта в демо-сценарии;
при 800 репликах история стабилизируется на 200 записях, RSS не растёт
(см. V016_TEST_REPORT.md).

---

## 8. Ответы

Три режима: `concise` / `normal` / `detailed` (старых имён в проекте не было —
в v0.9 режимов ответа нет). Один объект `Answer` всегда содержит **все три**
текста, поэтому режим вывода не может «деградировать» до нормального.

Шесть статусов: `DIRECT`, `INFERRED`, `CONTEXTUAL`, `AMBIGUOUS`, `UNKNOWN`,
`CONTRADICTED`. `detailed` добавляет блок «Доказательства» (с id доказательств
и реплик) и «Как получен ответ» (цепочку правил с id утверждений).

`UNKNOWN` говорит, чего именно не хватает (отношение / субъект / объект), и не
выдумывает факт. `CONTRADICTED` показывает обе записи и их источники.

---

## 9. Хранение и миграция

`Store` пишет атомарно (временный файл + `os.replace`), всегда делает резервную
копию `.bak-<timestamp>`, после записи перечитывает файл и сверяет число
записей. `migrate_file` **отказывается** писать, если `records_in != records_out`.
Повторная миграция возвращает `already-current` (идемпотентность).
Откат — `Store.rollback(backup)`.

Схема: `SCHEMA_VERSION = 2`, `LEGACY_SCHEMA_VERSION = 1`.
