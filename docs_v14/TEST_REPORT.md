# TEST_REPORT.md — v0.14 (прогон на 2026-11-26)

| Набор | Команда | Итог |
|---|---|---|
| Исторические | pytest test_rasm*.py rasm_cim_v8/tests tests_v9 tests_v10 | 163 passed |
| v0.11 | pytest tests_v11 | 56 passed |
| v0.12 | pytest tests_v12 | 27 passed |
| v0.13 | pytest tests_v13 | 40 passed |
| v0.14 | pytest tests_v14 | 21 passed |
| Итого | | **307 passed / 0 failed** |
| v0.8 freeze | md5sum -c results_v9/v08_freeze_manifest.md5 | 0 mismatch |
| baseline-272 | md5sum -c results_v11/freeze_manifest_baseline_v0.10.md5 | 0 mismatch |
| Rust v0.7 | hash set | 451bbfa5… без изменений |

SKIP: нет. INCONCLUSIVE унаследованные (device fsync) без изменений.

## Что покрывают 21 тест v0.14
Прямой факт + evidence id в скобках · definition с depth-2/3 R01-цепью (все три
предка цитатуются) · yes_no FALSE (deny) · assert+deny → CONFLICT с двумя ids ·
гипотеза «Венера is_a планета» НЕ становится фактом (verdict UNKNOWN, планета не в
ответе) · расположен Париж (R02) · depth-cap при max_depth=1 · цикл А↔Б —
завершается · ширина 20 веток — завершается · adversarial/abstain (4 формы) ·
пустой KB · determinism (2 прогона идентичны) · compare (с обоими числами →
TRUE+evidence; без одного → UNKNOWN+missing_prereqs) · evidence-gate unit:
предложение со ссылкой на несуществующий id отбрасывается; missing-item → partial;
conflict-агрегация; каждое предложение ответа со ссыкой st_ либо nothing-found.

## Честные замечания
- Бенчмарк/разметка — 11 ручных вопросов на маленькой фикстуре; это НЕ общий
  NLU-бенчмарк и не измеряет разговорную гибкость.
- Абстенция отменно-правее уверенности ВЕЗДЕ: нехватило доказательств → UNKNOWN.
- 307 passed ≠ «система понимает язык»; это «не делает неподтверждённых утверждений».
