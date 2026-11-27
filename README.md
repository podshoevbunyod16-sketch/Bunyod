# RASM-CIM — офлайн-система работы со смыслом (v0.17)

Локальная, полностью офлайновая система: русскоязычный диалог, типизированная
база знаний с происхождением каждого факта, ограниченный вывод с доказательствами
и ответы, каждое предложение которых опирается на утверждение в базе.
**Только стандартная библиотека Python, без сети, без LLM, без телеметрии.**

Это исследовательская сборка. Она **не** заявляет «понимание языка» или AGI:
качество измеряется конкретными метриками, а всё непроверенное честно помечено
как INCONCLUSIVE в `docs_v17/LIMITATIONS.md`.

## Что где

| Пакет | Роль |
|---|---|
| `rasm_cim_v16/` | семантическое ядро (заморожено): типизированный граф, provenance, evidence, контекст, вывод, ответы |
| `rasm_cim_v17/` | интеграция с данными v0.11–v0.15, единый CLI, проверка целостности |
| `rasm_cim_v9/`, `rasm_cim_v8/`, `rasm_cim/`, `rasm_cim_v1_snapshot`…`v6` | исторические слои, сохранены как есть |

## Быстрый старт

```bash
python -m rasm_cim version         # rasm-cim v0.17.0 (ядро v0.16.0, схема kb 2)
python -m rasm_cim chat            # диалог; :help — команды
python -m rasm_cim teach "Исаак Ньютон — это физик"
python -m rasm_cim ask "Кто такой Исаак Ньютон?" --answer detailed
python -m rasm_cim verify          # проверка целостности базы
```

`python -m rasm_cim`, `python -m rasm_cim_v17` и консольный скрипт `rasm-cim` —
один и тот же CLI. База: `~/.rasm-cim/` (`$RASM_CIM_HOME` или `--home`).

## Интеграция старых данных (v0.11–v0.15)

Данные предыдущих версий читаются как **данные**, а не как код:

```bash
python -m rasm_cim import path/to/kb/          --dry-run   # rasm-cim-kb/1
python -m rasm_cim import demo_facts.jsonl     --dry-run   # плоские факты v0.11
python -m rasm_cim import path/to/learned/     --dry-run   # rasm-cim-learned/1
```

* dry-run ничего не пишет и показывает, что будет импортировано, что пропущено
  и почему;
* импорт идемпотентен (журнал `import_ledger.json` по sha256 источника плюс
  дедупликация в самом графе);
* каждый факт сохраняет происхождение (legacy statement id, doc_id, фрагмент,
  надёжность) в `Statement.provenance["legacy"]` и запись `EV_IMPORT` в журнале
  источников;
* откат — `python -m rasm_cim restore <ledger_id>`;
* неизвестный предикат **не** подменяется «похожим»: запись пропускается с
  причиной, таблицу нужно расширить в `rasm_cim_v17/interop/mapping.py`.

## Тесты

```bash
python -m pytest tests_v16 -q      # 348 passed
python -m pytest tests_v9 tests_v10 -q                       # 54 passed
python -m pytest tests_v17 -q                                # 91 passed
python -m pytest test_rasm*.py rasm_cim_v8/tests -q          # 102 passed (без test_rasm_v5: нужен sklearn)
```

## Документация

* `AUDIT_REPORT.md` — фактическое состояние репозитория и найденные дефекты;
* `PROGRESS.md` — фазы, точки продолжения;
* `docs_v17/` — ARCHITECTURE, CHANGELOG, TEST_REPORT, LIMITATIONS, REPRODUCE, PLAN_NEXT;
* `docs_v16/` — аудит, архитектура и миграция ядра;
* `docs_v10/`, `docs_v9/`, … — история версий.

Установка: `pip install -e .` (для тестов — `-r requirements-dev.txt`;
numpy нужен только историческим слоям v8/v9).
