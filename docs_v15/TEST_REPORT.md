# TEST_REPORT.md — v0.15 (прогон 2026-11-26)

| Набор | Команда | Итог |
|---|---|---|
| Исторические | pytest test_rasm*.py rasm_cim_v8/tests tests_v9 tests_v10 | 163 passed |
| v0.11 | pytest tests_v11 | 56 passed |
| v0.12 | pytest tests_v12 | 27 passed |
| v0.13 | pytest tests_v13 | 40 passed |
| v0.14 | pytest tests_v14 | 21 passed |
| v0.15 | pytest tests_v15 | 14 passed |
| Итого | | **321 passed / 0 failed** |
| v0.8 freeze | md5sum -c results_v9/v08_freeze_manifest.md5 | 0 mismatch |
| baseline-272 | md5sum -c results_v11/freeze_manifest_baseline_v0.10.md5 | 0 mismatch |
| Rust v0.7 | sha256 set | 451bbfa5… без изменений |

SKIP: нет. Унаследованные INCONCLUSIVE (device fsync) без изменений.

## Что покрывают 14 тестов v0.15
- Corpus: размеры 265/84/88 в пределах требований ТЗ; split by template family; ни одно семейство
  не пересекает splits; идентичный текст не встречается в разных splits; manifest sha256 совпадает
  с файлами; полная пересборка детерминирована и даёт тот же corpus_sha256.
- Artifacts: train→save→load roundtrip (fallback counts равны); двух проходов обучения с одним seed
  дают идентичные counts; rollback возвращает current.json и содержимое родителя; новые версии
  не стирают старые; ссылка parent корректна.
- Corrections: хэш-цепочка (правка файла ловится verify_chain); relation-correction end-to-end:
  baseline → неверный маршрут ответа → коррекция → artefact lv+1 → ответ TRUE с evidence ids →
  rollback → точно baseline-поведение (verdict + evidence ids равны); конфликтующие коррекции
  (capital_of vs located_in на один и тот же target) → обе в истории, конфликт в отчёте, latest-wins;
  alias-коррекция, перевязывающая «Берлин» на «Париж» → отказ (ambiguity control); ошибка
  инициализации fallback-коррекции → новая версия содержит correction id и metrics с before/after.

## Поймано held-out test set-ом (важно)
Два подлинных дефекта найдены ТОЛЬКО на test (после того, как он перестал быть "своим"):
(1) в knowledge keyword-таблице не было стемов написан/реальн/существов → конвейер отвечал
definition-слоем на «Кем написана…» и дисквалифицированную truth-фразу «Дракон реален?» маскировал
как supported вместо CONFLICT; (2) _definition не проверял assert+deny. Оба исправлены как
общая коррекция словаря/гейта (НЕ подгонка под test), регрессия 6 проб и 321 тест зелёные.

## Честные замечания
- verdict_acc — только KB-answerable подмножество (capital_of субъект, can-fly deny, exists,
  wrote reverse). is_a object-side и population метрикой не покрыты (правильное UNKNOWN).
- 321 passed ≠ модель понимает русский; это «не выдаёт неподтверждённого» + детерминизм + rollback.
- FAILED-случаев в финальном прогоне нет.
