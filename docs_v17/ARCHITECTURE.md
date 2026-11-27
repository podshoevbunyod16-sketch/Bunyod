# ARCHITECTURE v0.17 — интеграционный слой и точка роста

v0.16 — семантическое ядро (типизированный граф, provenance, evidence,
ограниченный вывод, ответы). v0.17 его **не переписывает**: ядро используется
через публичный API, новые возможности живут в отдельных модулях.

```
rasm_cim_v16/                  заморожено: граф, вывод, ответы, контекст
        ▲                      (348 тестов остаются зелёными)
        │ публичный API
rasm_cim_v17/
├── semantics/                 План B: QueryIR → MeaningIR + метрики
│   ├── meaning.py             MeaningIR, focus, пресуппозиции, unknowns
│   ├── refine.py              исправления речевого акта (с основаниями)
│   ├── corpus.py              формат корпуса rasm-cim-eval-cases/1
│   ├── metrics.py             акт / сущности / ответы / unsupported rate
│   └── harness.py             демо-база: только teach + импорт A
├── pipeline.py                SemanticPipeline: делегирование или исполнение
├── interop/                   План A: данные v0.11–v0.15 → граф v0.16
│   ├── legacy.py              читатель rasm-cim-kb/1 и demo_facts.jsonl
│   ├── learned.py             читатель rasm-cim-learned/1 (+sha256)
│   ├── mapping.py             таблица отношений и типов (данные, не код)
│   └── importer.py            plan → apply → ledger → restore
├── routing.py                 routing.json: подсказки маршрута (не знания)
├── verify.py                  проверка целостности базы
├── cli.py                     единый CLI  (python -m rasm_cim[_v17], rasm-cim)
└── __main__.py
rasm_cim/__main__.py           добавленная точка входа python -m rasm_cim
```

## Границы, которые нельзя размывать

1. **Знание vs маршрутизация.** Факты — только в `kb.json` (граф v0.16).
   Подсказки «этот вопрос про `capital_of`» — в `routing.json`. Смешать их
   значит потерять возможность откатить обучение, не тронув знания.
2. **Чтение vs запись.** `interop/legacy.py` и `interop/learned.py` не пишут
   ничего и никогда не мутируют источник. Запись — единственная точка
   `importer.apply_plan`.
3. **Guessing.** Неизвестный предикат, гипотеза без подтверждения, запись без
   объекта — не «примерно то же самое», а пропуск с причиной в отчёте.
4. **Обратимость.** Каждый импорт: резервная копия `kb.json`/`context.json`
   до записи, запись в `import_ledger.json`, откат одной командой.

## План A — как устроен импорт

```
build_plan(source)                 чистая функция: ничего не пишет
   legacy.LegacyKB.load (replay)  torn-tail / битые строки / неизвестные события
   mapping.map_predicate          отношение v0.16 или отказ
   mapping.map_entity_kind        тип сущности v0.16 или ANY + предупреждение
   → ImportPlan{entities, statements(skipped_reason), relations, warnings}
        │  dry-run: печать отчёта, файлы не создаются
        ▼
apply_plan(plan, dm, store)
   1. зарегистрировать кастомные отношения (+ их inverse)
   2. сущности: resolver.intern + алиасы
   3. утверждения: assert_triple(provenance={"legacy": …}) + evidence(EV_IMPORT)
   4. артефакт: routing.json (+ алиасы из alias_delta, с контролем двусмысленности)
   5. store.save + запись в ledger (sha256 источника, digest плана, бэкапы)
```

Идемпотентность держится на двух независимых механизмах: журнал
(`(realpath, sha256)` уже применён → no-op) и сам граф (`assert_triple`
на идентичном триплете не создаёт дубль, а поднимает `reinforce`).

## План B — как устроена семантика

```
text
 └─ LanguageParser (v0.16)            syntax: intent, wh, mentions, predicate
     └─ build_meaning (v0.17)         meaning: act, slot, focus, presuppositions
         ├─ repaired == False  ->  DialogueManager.say()      (делегирование)
         └─ repaired == True   ->  SemanticPipeline._respond_* (исполнение)
```

`MeaningIR.repaired` истинно ровно тогда, когда слой смысла расходится с
синтаксисом: исправлен речевой акт, восстановлен фокус из хвоста фразы,
вычищено упоминание.  Для неисправленных реплик поведение остаётся поведением
v0.16 — именно поэтому 348 тестов ядра не теряют смысла.

Три правила, которые дороже полноты:

1. **Не выдавать чужое за нужное.** «отец Исаака Ньютона» не превращается в
   «Исаак Ньютон»: родственные слова накладывают вето на восстановление фокуса.
2. **Не исполнять не тот акт.** Если парсер назвал вопрос обучением, вопрос не
   записывается в базу.
3. **Не отвечать при неоднозначности.** Местоимение без контекста → уточняющий
   вопрос; два кандидата → `AMBIGUOUS`, выбор запрещён.

## Планы C–G (ещё не реализованы)

`discourse/` (План C — контекст и coreference, План E — `DiscoursePlanner`/
`ResponseIR`/NLG), `learning/` (План F — версии артефакта, совместимые по схеме
с `rasm-cim-learned/1`), `robust/` (План G — fuzz, бюджеты, Termux).  Ни одного
из этих модулей пока нет — в `PROGRESS.md` они честно помечены «не начато».

## Reasoning (План D)

```
ответ ядра UNKNOWN?
        │ да
        ▼
answer_with_inference ──► Fact (только из InferenceEngine)
        │                        │
        │                        ▼
        │                 fact_within_policy ──► глубина / доказательство /
        │                                       порог уверенности
        ▼
Answer(status=INFERRED, facts=[Fact], explanation=proof)
        │
        ▼
audit_answer ──► evidence_precision, premise_coverage
```

Правила этого слоя:

1. **Ни одного собственного вывода.** Ответ собирается из `Fact.proof`; если
   доказательства нет, ответа нет.
2. **Политика проверяема.** `ReasoningPolicy` — единственное место, где
   решается, что факт допустим (глубина, наличие вывода, уверенность); CLI
   `explain` печатает вердикт для каждого факта.
3. **Переопределение только «не знаю».** Резервный путь включается, если
   ядро вернуло UNKNOWN; уверенный ответ ядра не заменяется никогда
   (`test_fallback_never_overrides_a_confident_core_answer`).
4. **Вывод не пишется в базу.** `InferenceEngine` возвращает выведенные факты
   вызывающему; в граф они не попадают, и `test_audit_does_not_write_anything`
   это фиксирует.
