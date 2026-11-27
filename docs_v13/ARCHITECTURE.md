# ARCHITECTURE.md — RASM-CIM v0.13 (диалог, контекст, мульти-вопросы)

## Слои
```
сообщение пользователя
  → mode guard (pollution): normal | hypothetical | quote | teach   (ПЕРВЫМ)
  → split_questions():  нумерация "1) 2)", '?', ';', союз ", и/а + вопрос" → упорядоченные подвопросы (≤8)
  → per sub-question: v0.11 analyze() → intent;
        сущности = стем-матч имён/алиасов KB (rasm-cim_v12);
        местоимения/эллипсис = find_pronouns + resolve против context.recent_entities
        (plural → сущности ПОСЛЕДНЕЙ ходовой реплики; singular → гендерный фильтр)
  → routing:
        greeting → canned;  ask_*/yes_no → KB statement lookup (TRUE/FALSE/UNKNOWN/CONFLICT);
        compare → is_a-объекты обеих сущностей (partial);  list → KB scan по категории;
        unknown → CLARIFY «О ком или о чём идёт речь?»
  → агрегат (номера сохранены) → запись хода в context (1 fsync на группу через suspend/flush)
```

## ContextStore (`rasm_cim_v13/dialogue/context.py`)
Отдельно от KB. JSON-файл `context.json`, схема rasm-cim-context/1; атомарные записи
(tmp+replace+fsync), `suspend()/flush()` для батчинга. Ограничения: max_turns (по умолчанию
20, FIFO), ttl_seconds (по умолчанию 1800), recent_entities ≤12 (MRU), open_questions ≤10.
Битый/чужой context.json НИКОГДА не роняет диалог: файл уезжает в *.corrupt, стейт пустой.
Хранит: turns (role, text, intent, entities[], entity_ids[], statement_ids[] — ссылки на
provenance-id KB), active_topic + topic_stack (≤5), recent_entities (name, gender, kb_id),
open_questions (id, question, missing, candidates).

## Местоимения (`pronouns.py`) — ограниченно, по ТЗ
- Род/число: он/его/ему/нём/него→m, она/её/ей/ней/неё→f, оно→n, они/их/им/них→p.
- Демонстративы: этот/эта/эти/тот/та/те. «это/то» ИСКЛЮЧЕНЫ намеренно (связка/дискурс-
  частица, не референт — давали ложные CLARIFY).
- Род сущности по окончанию (эвристика + закрытый список несклоняемых: кофе→m, пальто→n…).
- singular: кандидаты = recent_entities того же рода → 1 = resolved, ≥2 = AMBIGUOUS →
  уточняющий вопрос со списком кандидатов (никогда не случайный выбор), 0 = UNKNOWN.
- plural «их/их»: кандидаты = сущности ПОСЛЕДНЕГО user-turn (иначе последние ≤4 MRU),
  веерный разбор (France & Germany).
- Эллипсис: «А её население?» → местоимение + ключевое слово предиката →
  follow-up без вопросительных слов ЯВЛЯЕТСЯ вопросом (context_followup).

## Режимы загрязнения (`manager._mode`)
- hypothetical (допустим|например|представь|предположим|пусть|гипотетическ…|если бы) →
  QA разрешён, НИЧЕГО не сохраняется; ответ помечается «(гипотетика — не сохраняю как факт)».
- quote («…» или "…") → не сохраняется.
- teach (запомни|запиши|учти|сохрани факт) → явный Teach-путь с full provenance
  (fragment = исходная реплика, source=user_teach); типы сущностей наследуются из KB
  (нет дублей «Франция-place vs Франция-thing»); паттерны: «А — это Б» (is_a),
  «столица X — Y» (capital_of), «X написал Y» (написал); иначе TEACH_HELP с подсказкой формата.
- Смешанный «допустим, запомни…» → hypothetical выигрывает (ничего не пишем).

## Инварианты
1. Гипотетика/цитата никогда не попадает в KB (проверено тестами).
2. При неоднозначности — вопрос, не угадывание; CLI возвращает exit code 4.
3. Каждое содержательное утверждение несёт evidence: statement_ids + source/doc_id/method.
4. CONFLICT в KB не скрывается: показываются оба противоречащих утверждения с id.
5. Контекст восстанавливается после рестарта (turns, entities, topic) без потери ссылок.

## CLI (`python -m rasm_cim_v13.cli`)
ask CTX --kb KB "вопрос" [exit 4 при CLARIFY] · show-context · reset-context · export-session.
reset-context/show-context/export-session — команды ТЗ; чувствительные реплики наружу
только по явному export-session.
