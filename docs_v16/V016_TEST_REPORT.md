# V016_TEST_REPORT — результаты тестирования v0.16

Окружение: Python 3.11.2 (GCC 12.2.0), Linux x86_64, 2 CPU, 3939 MiB RAM,
pytest 9.2.0 в локальном `.venv`. Дата прогона: 2026-11-27.
Команда полного прогона:

```
.venv/bin/python -m pytest tests_v16 test_rasm*.py rasm_cim_v8/tests tests_v9 tests_v10 -q
```

---

## 1. Итоговые числа

| Набор | Тестов | Результат | Время |
|---|---|---|---|
| `tests_v16` (новый слой) | 348 | **348 passed** | 5.67 s |
| `test_rasm*.py` + `rasm_cim_v8/tests` (база до v0.16) | 109 | **109 passed** | 3.06 s |
| `tests_v9` + `tests_v10` | 54 | **54 passed** | 2.83 s |
| **Всего** | **511** | **511 passed, 0 failed** | 11.57 s |

**RESULT** — провалов нет. **RESULT** — базовые наборы дают ровно те же числа,
что и до начала этапа (109 и 54), то есть регрессий в v0.8/v0.9 не появилось,
включая изменение токенизаторов (цифры больше не отбрасываются).

### Разбивка `tests_v16` по модулям

| Файл | Тестов | Что покрывает |
|---|---|---|
| `test_a_semantics.py` | 26 | обучение, `alias_of` вместо `is_a`, словарь отношений, типы, расширяемость |
| `test_b_nlu.py` | 38 | разбор 20+ русских конструкций, намерения, отрицание, время, модальность, эллипсис |
| `test_c_entities.py` | 38 | нормализация, стемирование, транслитерация, разрешение, неоднозначность, род |
| `test_d_inference.py` | 24 | TEST D, правила вывода, глубина, бюджет узлов, циклы, полярность |
| `test_e_context.py` | 19 | TEST E/F/L, местоимения, смена и возврат темы, длинный диалог |
| `test_f_persistence.py` | 22 | TEST G, перезапуск, миграция на реальном файле, откат, идемпотентность |
| `test_g_conflicts.py` | 25 | TEST H/I/J/K, неоднозначность, противоречия, отрицание, дедупликация, отзыв |
| `test_h_answers.py` | 46 | TEST M/N, три режима, шесть статусов, `explain`, все команды `:cmd`, CLI |
| `test_i_memory.py` | 17 | четыре слоя памяти, поля записи, дубли, размеры, целостность |
| `test_j_perf.py` | 11 | пропускная способность, p95, бюджет, RSS, время save/load, линейность роста |
| `test_k_portability.py` | 82 | stdlib-only, отсутствие сети и абсолютных путей, синтаксис 3.10, запуск CLI |

---

## 2. Соответствие требуемым сценариям TEST A–N

| Требуемый сценарий | Тест(ы) | Статус |
|---|---|---|
| A. Обучение факту | `test_a_teach_stores_fact`, `test_a_teach_without_the_verb_also_works`, `test_a_teach_records_evidence_and_provenance` | **RESULT: pass** |
| B. Псевдонимы, не `is_a` | `test_b_alias_not_is_a`, `test_b_alias_direction_is_surface_to_canonical`, `test_b_reverse_order_is_still_alias`, `test_b_two_proper_names_are_not_merged_silently`, `test_h_two_unrelated_names_do_not_become_one_entity` | **RESULT: pass** |
| C. Связанные факты | `test_c_entities_are_linked_by_real_statements`, `test_c_instance_of_sets_class_kind`, `test_graph_neighbours_and_indexes` | **RESULT: pass** |
| D. Многошаговый вывод | `test_d_multistep_inference_over_alias` (alias_bridge, глубина 1), `test_d_inference_through_two_taxonomy_hops` (глубина 3), `test_d_confidence_decays_with_depth`, `test_inference_never_writes_to_the_graph` | **RESULT: pass** |
| E. Местоимения | `test_e_pronoun_resolves_to_the_subject_just_discussed`, `test_e_pronoun_gender_prevents_a_wrong_referent`, `test_e_pronoun_with_no_referent_asks`, `test_e_ambiguous_pronoun_asks_instead_of_guessing`, `test_e_pronoun_resolution_is_logged` | **RESULT: pass** |
| F. Смена темы | `test_f_topic_switch_and_return`, `test_f_topic_stack_is_maintained`, `test_f_classify_topic_before_touching`, `test_f_return_to_previous` | **RESULT: pass** |
| G. Переживание перезапуска | `test_g_knowledge_survives_a_full_restart`, `test_g_context_survives_a_restart`, `test_g_restart_is_repeatable`, `test_g_polarity_oracle_is_rebuilt_after_load` | **RESULT: pass** |
| H. Неоднозначность | `test_h_ambiguous_name_triggers_a_question`, `test_h_ambiguity_is_remembered_as_pending`, `test_h_a_question_the_system_can_answer_is_not_deflected` | **RESULT: pass** |
| I. Противоречия | `test_i_contradiction_is_detected_and_both_sides_kept`, `test_i_contradiction_changes_the_answer_status`, `test_i_no_silent_deletion_on_contradiction`, `test_i_functional_conflict`, `test_i_disjoint_class_conflict`, `test_i_alias_divergence_conflict`, `test_i_taxonomy_cycle_is_reported`, `test_i_scan_is_idempotent` | **RESULT: pass** |
| J. Отрицание | `test_j_negation_does_not_become_a_positive_fact`, `test_j_negation_is_rendered_as_a_denial`, `test_j_negated_and_positive_together_is_a_conflict_not_a_choice`, `test_j_negation_of_an_event` | **RESULT: pass** |
| K. Повторное обучение без дублей | `test_k_repeated_teaching_does_not_duplicate`, `test_k_repeated_teaching_is_reported_as_known`, `test_k_repeated_teaching_keeps_the_first_timestamp`, `test_k_evidence_is_not_duplicated` | **RESULT: pass** |
| L. Диалог ≥ 20 реплик | `test_l_long_dialogue_keeps_context_clean` (скрипт `LONG_SCRIPT`, 22 реплики), `test_l_no_runaway_growth` | **RESULT: pass** |
| M. Неподтверждённый вывод → UNKNOWN | `test_m_unknown_instead_of_an_invented_answer`, `test_m_unknown_names_what_is_missing`, `test_m_unknown_still_says_what_is_known`, `test_m_no_numbers_appear_out_of_nowhere` | **RESULT: pass** |
| N. Подробный ответ с доказательствами | `test_n_detailed_answer_is_built_from_real_facts`, `test_n_answer_is_not_a_single_line_when_data_exists`, `test_n_answer_lists_only_true_facts`, `test_n_unknown_relations_are_reported`, `test_all_three_renderings_exist_on_every_answer` | **RESULT: pass** |

---

## 3. Производительность (измерено, не заявлено)

Масштаб: **1000 утверждений, 852 сущности** (фикстура `test_j_perf.py`,
масштаб проверяется отдельным тестом `test_kb_size`, чтобы остальные
утверждения не измеряли не тот объём).

Смешанная база (таксономия + отношения действия + места):

| Показатель | Значение |
|---|---|
| Обучение (разбор + запись), 1000 фактов | 0.41 s → **2432 фактов/с** |
| Ответ «Кто такой X?»: p50 / p95 / max | **0.52 мс / 6.76 мс / 7.22 мс** (300 замеров) |
| Ответ «X — Y?»: p50 / p95 / max | **0.64 мс / 0.72 мс / 0.93 мс** (200 замеров) |
| 200 обходов графа на глубину 2 | **0.2 мс суммарно** |
| `save()` при 1000 фактов | **0.061 s** |
| `load()` при 1000 фактов | **0.076 s** |
| `kb.json` / `context.json` | **1359 КБ / 65 КБ** |
| RSS пик (весь процесс, включая интерпретатор) | **31.4 MiB** (база интерпретатора 17.5 MiB) |
| Диалог 800 реплик | 0.36 s; история стабилизируется на 200 записях; RSS 31.9 MiB |

Рост базы (однородные факты, второй замер — на уже заполненной базе):

| Фактов | Обучение | Запрос p95 | `kb.json` | RSS |
|---|---|---|---|---|
| 1000 | **1762 фактов/с** | 0.28 мс | 1.6 МБ | 37 MiB |
| 5000 | **1832 фактов/с** | 0.26 мс | 7.7 МБ | 109 MiB |

**RESULT** — стоимость обучения на факт **не растёт** с размером базы (1762 →
1832 фактов/с при пятикратном росте). Это исправление реального дефекта:
раньше проверка противоречий обходила все утверждения при каждом обучении, и
замеры давали 1096 фактов/с на 1000 фактах против 328 фактов/с на 5000.
Замер с отключённым сканером: 669/с против 1968/с на 2000 фактах — то есть
сканер и был узким местом. Сейчас сканирование при обучении ограничено новой
тройкой; полный обход остался для `:conflicts`, загрузки и миграции.
Защищено тестом `test_teaching_does_not_get_slower_as_the_kb_grows`.

**RESULT** — загрузка всей базы в память линейна по размеру: ~92 MiB на 5000
утверждений. Точечные запросы при этом не зависят от размера (индексы).

**RESULT** — точечный запрос не сканирует базу: проверяется тестом
`test_no_full_kb_scan_for_a_point_lookup` (индексы `_spo`/`_pos`/`_by_key`).
**RESULT** — рост диалога не деградирует: `test_long_dialogue_does_not_leak_memory`.
**RESULT** — вывод ограничен: `test_inference_node_budget_on_a_dense_graph`,
`test_adversarial_cycle_does_not_blow_up`.

---

## 4. Совместимость и переносимость

| Проверка | Тест | Статус |
|---|---|---|
| Только стандартная библиотека и локальные импорты | `test_only_stdlib_and_local_imports` | **RESULT: pass** |
| Нет абсолютных путей в коде | `test_no_hardcoded_absolute_paths` | **RESULT: pass** |
| Нет сетевых вызовов | `test_no_network_usage` | **RESULT: pass** |
| Синтаксис совместим с Python 3.10 | `test_python_310_syntax` | **RESULT: pass** |
| Нет удалённых/устаревших API | `test_no_removed_or_deprecated_apis` | **RESULT: pass** |
| Импорт в чистом подпроцессе | `test_runtime_imports_cleanly_in_a_subprocess` | **RESULT: pass** |
| CLI: help / demo / одиночный вопрос / неверный режим / команды | 5 тестов, запуск реальным подпроцессом | **RESULT: pass** |
| Замороженные версии не импортируются | `test_frozen_versions_are_untouched`, `test_v16_imports_only_v9_and_itself` (AST) | **RESULT: pass** |
| **Запуск на Python 3.14** | — | **INCONCLUSIVE**: в окружении есть только 3.11.2, собрать 3.14 нельзя (нет доступа к python.org) |
| **Запуск на Android/Termux ARM64** | — | **INCONCLUSIVE** на устройстве; проверяемые предпосылки (stdlib, нет сети, нет абсолютных путей, 3.10-синтаксис) — **pass** |

---

## 5. Сквозной сценарий, выполненный реальным CLI

Команды выполнялись подпроцессом на временной базе; вывод приведён дословно.

```
ты> teach Исаак Ньютон — это физик
rasm[DIRECT]> Записал: Исаак Ньютон — физик [уверенность 0.90, st-000001]
ты> teach Ньютон — это Исаак Ньютон
rasm[DIRECT]> Записал: Ньютон — также известен как Исаак Ньютон [уверенность 0.95, st-000002]
ты> teach Физик — это учёный
rasm[DIRECT]> Записал: физик — подкласс ученый [уверенность 0.92, st-000005]
ты> Ньютон — физик?
rasm[INFERRED]> Да. Ньютон — физик.
Основание: Исаак Ньютон — физик [вывод, глубина 1, правила alias_bridge+lookup,
уверенность 0.88, st-000001]
ты> Ньютон — учёный?
rasm[INFERRED]> Да. Ньютон — ученый.  [вывод, глубина 2,
правила alias_bridge+instance_subclass+lookup, уверенность 0.79]
ты> Что он открыл?
rasm[CONTEXTUAL]> Исаак Ньютон открыл закон всемирного тяготения.
ты> teach Исаак Ньютон не был физиком
rasm[DIRECT]> Записал: Неверно, что Исаак Ньютон — физик [уверенность 0.95, st-000007]
ты> Ньютон — физик?
rasm[CONTRADICTED]> Противоречие в базе: по этому вопросу есть несовместимые
записи, обе сохранены.
  • POLARITY: «Исаак Ньютон — instance_of → физик» утверждается и как истина,
    и как ложь; оба источника сохранены
    источники: turn:u-00001, turn:u-00007
ты> Кто написал Войну и мир?
rasm[UNKNOWN]> Не знаю. В вопросе нет ни одной известной сущности.
```

Три режима на одном вопросе (реальный CLI, `--answer` до и после подкоманды):

```
concise : Да. Ньютон — ученый.
normal  : + «Основание: … [вывод, глубина 2, правила alias_bridge+instance_subclass+lookup, 0.79, st-000001]»
detailed: + «Доказательства:» ev-000001…ev-000003 с текстом реплик
          + «Как получен ответ:» 1. [alias_bridge] … 2. [instance_subclass] … 3. [lookup] …
```

---

## 6. Что тестирование НЕ подтвердило

**INCONCLUSIVE** — работа на Python 3.14 (нет интерпретатора).
**INCONCLUSIVE** — работа на реальном Android/Termux (нет устройства).
**INCONCLUSIVE** — совместимость с реальной пользовательской базой v1: в
репозитории и в `~/.rasm-cim/` такой базы нет, миграция проверена на
синтетическом файле (см. V016_MIGRATION_REPORT.md).
**INCONCLUSIVE** — многопоточная нагрузка: слой однопоточный, тестов на
параллельный доступ нет.
