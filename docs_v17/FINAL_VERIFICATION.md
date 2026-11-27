# FINAL_VERIFICATION — v0.17.0

Дата: 2026-11-27. Ветка: `arena/72c9200d-bunyod`. База: `main` @ `be5551ea`.
Исходное ТЗ: `RASM-CIM_v0.16_INTEGRATION_AND_COGNITIVE_ROADMAP_PROMPT.pdf` (10 стр.).

Документ отвечает на один вопрос: **по каждому требованию ТЗ — где оно
реализовано, чем измерено и что осталось непроверенным.** Там, где проверки нет,
стоит INCONCLUSIVE, а не «сделано».

---

## 1. Главное ограничение соблюдено: v0.16 не переписана

```bash
git diff --stat be5551ea -- rasm_cim_v16 tests_v16 docs_v16
# (пусто) — 0 строк
```

`git diff be5551ea -- rasm_cim_v16 tests_v16 docs_v16 | wc -l` → **0**. Вся
работа — новый пакет `rasm_cim_v17` (38 файлов, 5781 строка) и новые наборы
тестов. Ядро v0.16 импортируется как библиотека.

Итог по репозиторию: **72 файла, +10558 / −17**.

## 2. Тесты

| Набор | Команда | Результат |
|---|---|---|
| ядро | `pytest tests_v16 -q` | **348 passed** |
| слой v0.17 | `pytest tests_v17 -q` | **305 passed** |
| v9/v10 | `pytest tests_v9 tests_v10 -q` | **54 passed** |
| исторические (корневые + v8) | `pytest test_rasm*.py rasm_cim_v8/tests -q` | **109 passed** (57 + 52) |
| **Всего** | | **816 passed, 0 failed, 0 skipped** |

`test_rasm_v5.py` требует scikit-learn (не входит в dev-зависимости) —
ограничение существовало до v0.17, зафиксировано в `LIMITATIONS.md` и в CI.
С установленным scikit-learn он даёт 7 passed и входит в 163.
> **Исправлено аудитом 2026-11-27** (коммит `c143166`, ветка `arena/ec75e510-bunyod`): опубликованное число «102 passed» для исторических наборов считало только шесть корневых файлов и `rasm_cim_v8/tests`, без `test_rasm_v5.py` и без `tests_v9`/`tests_v10`. Фактический итог исторических наборов — **163 passed**: 57 корневых (`test_rasm*.py`, включая 7 в `test_rasm_v5.py`, для которых нужен scikit-learn), 54 в `tests_v9`+`tests_v10`, 52 в `rasm_cim_v8/tests`. Перепроверено запуском, вывод: 348 + 305 + 163 = 816 passed.


## 3. Матрица требований ТЗ

| Фаза / требование | Где | Чем измерено | Статус |
|---|---|---|---|
| **A. Интеграция v0.11–v0.16** | | | |
| импорт данных без общего предка с PR №1 | `rasm_cim_v17/interop/*` (форматы `rasm-cim-kb/1`, `rasm-cim-learned/1`, `demo_facts.jsonl`) | `pytest tests_v17/test_plan_a_import.py`: 34/34 факта, `unmapped_predicates() == {}` | DONE |
| единый CLI и `python -m rasm_cim` | `rasm_cim_v17/cli.py`, `rasm_cim/__main__.py` | 24 подкоманды, смоук каждой в тестах | DONE |
| проверка целостности базы | `rasm_cim_v17/verify.py` | 7 групп инвариантов, `rasm-cim verify` | DONE |
| сохранение старых API и форматов | `rasm_cim_v9…v16` не тронуты | `git diff` = 0 строк | DONE |
| **B. Семантический слой (QueryIR → MeaningIR)** | | | |
| смысл отдельно от слов | `semantics/meaning.py` | 38 кейсов, sha256 `743c8c18…` | DONE |
| измеренное качество акта | `semantics/metrics.py` | **97.4 %** (37/38) | DONE |
| качество сущностей | там же | P=R=F1=**0.975** (tp 39, fp 1, fn 1) | DONE |
| ответы по смыслу | `pipeline.py` | **96.4 %** (27/28) | DONE |
| ответ без источника запрещён | `metrics.evaluate_answers` | **0.0 %** (0/38) | DONE |
| три промаха названы, не скрыты | «Который час?», сочинение, «Париж — столица Франции?» | `BENCHMARK_REPORT.md` | DONE (как дефект) |
| **C. Устойчивый контекст и кореференция** | | | |
| связка «он/она» через `ContextEngine` v0.16 | `discourse/binding.py` | 5 отсылок: **80 %** верно, **0** ошибочных | DONE |
| честный отказ вместо догадки | там же | 1 отказ (`d04`, tie) | DONE |
| контекст не «остывает» после исправленной реплики | `record_turn` | `test_repaired_turn_keeps_the_context_warm` | DONE |
| ответы внутри диалога | `discourse/dialogue_metrics.py` | **4/4 = 100 %** | DONE |
| эллипсис сверх v0.16, «первый/второй» | — | — | **NOT DONE** (`PLAN_NEXT.md`) |
| **D. Reasoning с доказательствами** | | | |
| политика глубины/ветвления | `reasoning/policy.py` | `max_depth=3`, `max_nodes=400`, порог уверенности | DONE |
| факт без доказательства не попадает в ответ | `fact_within_policy` | тесты на глубину/доказательство/уверенность | DONE |
| вывод используется там, где ядро «не знает» | `reasoning/use.py` | 2 измеренных разрыва закрыты (обратное прочтение, ориентация wh) | DONE |
| уверенный ответ ядра не переопределяется | `test_fallback_never_overrides_a_confident_core_answer` | — | DONE |
| evidence precision | `reasoning/audit.py` | **1.0** (42 факта, 13 выведенных) | DONE |
| покрытие посылок доказательств | там же | **13/13 = 1.0** | DONE |
| цепочки 4+ | — | политика их не пускает | **PARTIAL** (осознанно) |
| **E. Связная генерация** | | | |
| план ответа как данные | `generation/response_ir.py` (`rasm-cim-response-ir/1`) | roundtrip, `validate` | DONE |
| утверждение без `statement_id` невозможно | там же | 7 тестов схемы | DONE |
| доказательство только для вывода | `generation/planner.py` | 42.9 % планов | DONE |
| регистры short/normal/detailed | `generation/nlg.py` | `short ≤ normal ≤ detailed` | DONE |
| детерминизм | там же | два рендера совпадают | DONE |
| утверждения с источником | `generation/metrics.py` | **13/13 = 100 %** | DONE |
| анафоры в выводе («он/это/первый») | — | — | **NOT DONE** (`LIMITATIONS.md` 23) |
| **F. Обучение без разрушения знаний** | | | |
| обучение тем же путём, что разговор | `learning/session.py` | уроки идут через `SemanticPipeline` | DONE |
| учебный план с ожиданиями | `learning/curriculum.py` (`rasm-cim-curriculum/1`) | 12 уроков, sha256 `931bcc55…` | DONE |
| удержание старых знаний | `learning/forgetting.py` | **100 %**, потеряно 0 | DONE |
| новые противоречия | там же | **0** (было 1, стало 1) | DONE |
| отсутствие деградации ответов | там же | **0 %** (один ответ расширился) | DONE |
| обратимость | `rollback_check` | `rollback_exact = true` | DONE |
| откат на «живой» базе | — | `load_dict` аддитивен | **PARTIAL** (`LIMITATIONS.md` 26) |
| **G. Надёжность, Termux, производительность** | | | |
| нет сети и телеметрии | `robustness/offline.py` | 38 файлов, **0** нарушений (AST) | DONE |
| нет обязательных зависимостей | `pyproject.toml` | `dependencies = []` | DONE |
| устойчивость к мусорному вводу | `robustness/fuzz.py` | 100 входов, **0** падений | DONE |
| время ответа | `robustness/perf.py` | p50 **0,54 мс**, p95 0,65 мс | DONE (десктоп) |
| память | там же | пик **26 КБ** на серию вопросов | DONE (Python, не RSS) |
| рост стоимости | `scale_test` | ×0,98–1,16 при 88→263 утверждениях | DONE |
| ARM64 / Termux / ~3 ГБ RAM | `docs_v17/TERMUX.md` | замеров нет | **INCONCLUSIVE** |
| **8. Финальная верификация** | этот документ | `metrics`, `coref`, `generation`, `learn`, `stress` | DONE |

## 3a. Проверка в чистом окружении (без numpy/hypothesis)

Повторная проверка на свежем клоне и голом Python 3.11 без `numpy` нашла дефект:
`python -m rasm_cim` падал, потому что исторический `rasm_cim/__init__.py`
импортировал numpy-модули прототипа до `__main__`. Исправлено ленивой загрузкой
(PEP 562); CLI теперь запускается на одной стандартной библиотеке, что и было
обещано для Termux. Регрессии: `test_entry_point_does_not_import_numpy`,
`test_cli_works_without_numpy`, `test_historical_names_are_lazy_and_still_present`.

`numpy` и `hypothesis` нужны только историческим слоям v8/v9 и property-тестам
(`requirements-dev.txt`), для v0.17 — нет.

## 4. Воспроизводимость

Все измеряющие команды детерминированы — два прогона подряд дают одинаковый
JSON:

```bash
for f in metrics coref generation learn; do
  python -m rasm_cim $f --json > /tmp/a.json; python -m rasm_cim $f --json > /tmp/b.json
  diff -q /tmp/a.json /tmp/b.json && echo "$f identical"
done
# metrics identical / coref identical / generation identical / learn identical
```

`stress` не сравнивался побайтово: в нём есть тайминги (они и должны
отличаться), но `crash_count = 0` и `offline.clean = true` стабильны.

## 5. Что честно не сделано (и записано, а не спрятано)

1. **Эллипсис сверх v0.16** и «первый/второй» — `PLAN_NEXT.md`, п. 1.
2. **Анафоры в сгенерированном тексте** — `LIMITATIONS.md` 23.
3. **Цепочки вывода длиной 4+** — политика `max_answer_depth = 3`;
   `LIMITATIONS.md` 18.
4. **Откат обучения на живой базе без перезагрузки** — `LIMITATIONS.md` 26.
5. **ARM64/Termux и реальный RSS** — `TERMUX.md`, `LIMITATIONS.md` 30–31.
6. **Python 3.14** — не установлен, `LIMITATIONS.md`.
7. **Реальная пользовательская `kb.json`** — отсутствует в окружении.
8. **Качество за пределами собственных корпусов** (38 кейсов, 5 отсылок,
   12 уроков, 14 реплик, 100 фазз-входов) — размеры и sha256 указаны рядом с
   каждым числом именно поэтому.

## 6. Утверждения, которых этот отчёт НЕ делает

* что система «понимает» язык или приближается к AGI — измерены конкретные
  метрики на конкретных наборах;
* что 97,4 % акта и 100 % удержания сохранятся на других данных;
* что «нет сети» доказано для всей системы: проверен только пакет v0.17
  (v0.16 проверяется тем же сканером отдельно, если нужно).
