# ARCHITECTURE.md — RASM-CIM v0.11 (русский языковой слой)

## Позиция в стеке
```
пользовательский текст
      ↓
normalize  (rasm_cim_v11/language/normalize.py)   NFKC+casefold, оригинал сохранён
      ↓
tokenize   (tokenize.py)                          offsets в ИСХОДНОЙ строке
      ↓
morphology (morphology.py)                        псевдо-стемминг, числа словами, стоп-слова
      ↓
intent     (intent.py)                            взвешенные правила, 7 типов
      ↓
(AnalysisResult, models.py)                       единый результат анализа сообщения
      ↓
adapter    (graph_adapter.py)                     Russian names → int64 для TruthGraph9
      ↓
TruthGraph9 (rasm_cim_v9/graph.py, БЕЗ ИЗМЕНЕНИЙ) TRUE/FALSE/UNKNOWN/CONFLICT
```

## Принятые решения (и почему)
- **NFKC, а не NFC**: NFKC фолдит compatibility-дубли (ﬁ→fi, fullwidth-цифры и буквы),
  которые приходят из реального копипаста; NFC оставляет их как два байтовых ряда одного
  и того же смысла и разрезал бы совпадения фактов пополам. Оригинал текста НИКОГДА не
  переписывается — хранится `original` + `flags` применённых преобразований. Решение
  версионировано (часть формата lang-0.11); смена требует новой версии.
- **casefold(), не lower()** — Unicode-корректное сравнение без регистра.
- **Offset-токенизация по оригиналу**: ручной скан (str.isalpha/isdigit), слова с
  внутритокенными дефисами = один токен; offsets всегда индексируют ИСХОДНУЮ строку,
  `original[start:end] == token.text` — это проверено тестами. Так поздние слои смогут
  связывать факты/провенанс с фрагментами исходного текста.
- **Морфология = псевдо-стемминг**, НЕ лемматизация по словарю: упорядоченное отсечение
  суффиксов (длинные→короткие, мин. основа 3) + малая таблица иррегулярностей
  (люди→человек, лучше→хороший…) + слова-числа 0..1000. Запрос и словарь стеммятся
  ОДНИМИ ПРАВИЛАМИ — совпадение работает, хотя псевдо-стем не является корректным словом.
  Это честный потолок rule-based слоя; словоформы не покрытые суффиксами получают
  собственный стем и честно не матчатся (UNKNOWN, не вымысел).
- **Intent = взвешенные правила** (сумма весов совпавших правил, cap 0.99), без ML.
  Каждый ответ объясняем: `matched` (rule ids) + `scores`. Порог 0.5; ниже → `unknown`.
  Кириллица отсутствует → `unknown` (никакой имитации английского/таджикского).
  Специализированные правила (определение/сравнение/список) побеждают общий вопрос-маркер.
- **TruthGraph9 не менялся**: адаптер отображает строки в int64 (sha256(fold(text))[:8],
  big-endian). Семантика TRUE/FALSE/UNKNOWN/CONFLICT и symmetric retract — как в v0.9.
- **Пакет `rasm_cim_v11` + подпакет `language/`** — репо-конвенция versioned-пакетов;
  расплывчатое top-level имя `language` намеренно не занято.

## Публичный API (`rasm_cim_v11.language`)
`analyze(text) -> AnalysisResult` · `normalize` · `tokenize` · `lemma/lemmas` ·
`is_stopword` · `num_to_int` · `classify_intent` · `GraphAdapter` · `stable_id` ·
`INTENTS` (greeting, ask_fact, ask_definition, yes_no, compare, list, unknown).

## Файлы v0.11 (новые; исторические не тронуты)
```
rasm_cim_v11/{__init__,cli,demo}.py
rasm_cim_v11/language/{__init__,models,normalize,tokenize,morphology,intent,graph_adapter}.py
rasm_cim_v11/data/{demo_facts.jsonl,intent_corpus.jsonl}
tests_v11/{test_language.py,test_corpus.py}
bench_v11/{gen_corpus_v11.py,eval_intent.py}
docs_v11/…, results_v11/…
```
CLI: `python -m rasm_cim_v11.cli language-check "..."` · `ask-text "..."`.
Старые команды v0.10 (`rasm_cim_v9.cli …`) не тронуты.

## Демо-ответчик (`demo.py`) — НЕ pipeline v0.14
Показывает связывание языкового слоя с TruthGraph9 на bundled-фактах (34 строки,
вкл. намеренную конфликтную пару Дракон). Отвечает только на столицы / определения /
yes-no из демо-памяти; остальное — UNSUPPORTED/UNKNOWN. Псевдо-стем-матчинг субъекта,
никакого fuzzy-объединения сущностей.
