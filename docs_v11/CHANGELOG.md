# CHANGELOG — v0.11 (русский языковой слой; 2026-11-26)

## Added
- Пакет `rasm_cim_v11` (+ подпакет `language/`): normalize.py (NFKC+casefold, оригинал
  сохранён, flags), tokenize.py (offsets в исходной строке, ручной Unicode-скан),
  morphology.py (псевдо-стемминг, слова-числа 0..1000, стоп-слова), intent.py
  (7 типов: greeting/ask_fact/ask_definition/yes_no/compare/list/unknown; веса+порог
  0.5; explainable matched/scores), models.py (dataclass-модели), public API
  `rasm_cim_v11.language.__init__` (`analyze` + …).
- `graph_adapter.py` — мост к TruthGraph9 (int64 id = sha256(fold(text))[:8]);
  семантика TRUE/FALSE/UNKNOWN/CONFLICT и retract-symmetric НЕ изменены.
- `cli.py`: `language-check TEXT`, `ask-text TEXT` (старые команды v0.10 не тронуты).
- `demo.py` + `data/demo_facts.jsonl` (34 факта, включая намеренную конфликтную пару
  Дракон→TRUE/FALSE) + `data/intent_corpus.jsonl` (161 пример, split по template-
  семействам, schema rasm-cim-corpus/0).
- `tests_v11/` (56 тестов) — см. docs_v11/TEST_REPORT.md;
  `bench_v11/gen_corpus_v11.py` (детерминированная генерация корпуса),
  `bench_v11/eval_intent.py` (→ results_v11/intent_eval.json).

## Changed
- Ничего исторического. v0.1–v0.10 код/тесты/результаты не изменялись
  (freeze: results_v11/freeze_manifest_baseline_v0.10.md5 — 272/272 OK).

## Not changed (намеренно)
- TruthGraph9, MobileMemory9, форматы v1/v2/v3, v0.9/v0.10 поведение.
- p95-vs-dict FAIL из v0.10 остаётся как был (не трогалось).
- Morphology — псевдо-стемминг, НЕ словарная лемматизация (задокументировано).

## Measured (host, Python 3.11.2)
- 163 исторических теста зелёные; tests_v11 = 56 passed; итого 219 passed.
- intent accuracy: train 86/86, dev 26/27, test 48/48 (161 примеров, тем-генерированы —
  НЕ мера понимания языка, см. LIMITATIONS/TEST_REPORT).
- Латентность analyze(): µs-порядки, см. results_v11/bench_latency.json.
