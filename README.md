# RASM-CIM — v0.15 (офлайн, русский язык)

Экспериментальный комплекс **Compound Information Machine** поверх v0.10-ядра:
русский языковой слой → provenance-KB → диалоговый контекст → reasoning с EvidenceSet →
управляемое улучшение (только по дельте измерений). **Stdlib-first, offline, без внешних LLM
для ответов.** Это исследовательская сборка: она честно отвечает «нет данных», когда
доказательств нет, и показывает, из чего составлен ответ (st_-идентификаторы).

> Не заявляем: общую языковую компетентность, понимание языка, поддержку таджикского.
> Заявляем: вербализацию **только** из подтверждённых данных, детерминизм, откат обучения.

## Возможности по версиям

| Версия | Что умеет (по факту, с тестами) |
|---|---|
| v0.10 | TruthGraph9 (TRUE/FALSE/UNKNOWN/CONFLICT), crash-safe journal, v0.9/v0.10 bench, freeze v0.8 |
| v0.11 | Русский слой: нормализация, морфология, интент (7 классов), CLI `language-check`/`ask-text` — 56 тестов |
| v0.12 | KB rasm-cim-kb/1: сущности/отношения/алиасы (контроль неоднозначности, без автомержа), provenance-цепочки, append-only журнал, идемпотентный JSONL import/export — 27 тестов |
| v0.13 | Диалог: контекст (20 реплик, TTL 30 мин, атомарен, corrupt-safe), местоимения (gender/number), мульти-вопросы (≤8, порядок), CLARIFY при неоднозначности, hypothetical/quote не сохраняются — 40 тестов |
| v0.14 | Reasoning EvidenceSet: direct/derived/missing, категориальная confidence, правила R01–R03 с KB-BFS, составление ответа ТОЛЬКО из доказательств, финальный evidence-check — 21 тест |
| v0.15 | Корпус (265/84/88, split по семействам шаблонов), обучаемые артефакты rasm-cim-learned/1 (откат, sha256), коррекции пользователя (hash-chain, verify/undo, конфликты показываются), offline eval — 14 тестов |
| `python -m rasm_cim` | Единая точка входа: режимы chat / ask / teach / memory / evaluate / system-info. `python -m rasm_cim_v9.cli` сохранён как v0.10-совместимый путь |

Итого: **362 теста** одной командой (163 исторических + 56 + 27 + 40 + 23 + 15 + 10 unified-CLI + 28 cross-language). freeze v0.8 = 0 mismatch; baseline-272 — ровно 3 задокументированных исключения (README/pyproject/requirements-dev, см. results_v15/freeze_delta_report_v0.10_to_v0.15.md), Rust v0.7 не изменялся. CI: .github/workflows/ci.yml — матрица Python 3.10–3.13 (тесты + freeze-гейты), Rust-задача (cargo test/release/clippy + cdylib + cross-language против CI-собранного .so), package-задача (build + установка колеса + smoke из чужой cwd). Android/Termux НЕ измерялся (см. results_verify/ANDROID_TERMUX_PROFILE.md).

## Быстрый старт

```bash
# единый entry point (демо-корпус — строго opt-in)
python -m rasm_cim --demo ask "Какая столица у Франции?"
# → Париж [TRUE] [st_c2c72f875a98e0b3]
python -m rasm_cim --demo chat        # интерактив (help, exit, reset, teach, inspect)
python -m rasm_cim --demo evaluate --split dev
python -m rasm_cim memory stats
python -m rasm_cim system-info

# версии по отдельности
python -m rasm_cim_v12.cli seed-demo /tmp/kb && python -m rasm_cim_v13.cli ask /tmp/ctx --kb /tmp/kb "А её население?"
python -m rasm_cim_v15.cli train
python -m rasm_cim_v15.cli correct --kb /tmp/kb --kind relation --wrong "" --correct capital_of \
    --target-text "Столицу Франции зовут?" --apply     # коррекция с проверяемым эффектом + rollback
```

## Оценка (host, Linux, Python 3.11; Termux НЕ измерялся)

- intent_acc на dev/test: 0.833→0.857 / 0.761→0.784 (fallback, 2–4 срабатывания; маргинально, не значимо)
- verdict_acc на KB-answerable подмножестве: 1.0 (n=29 dev / 32 test); unsupported-rate 0.0; regression-пробы 6/6
- диалоговый round-trip p50 ≈ 1.4 ms (fsync-батчинг); reasoning p50 ≈ 236–343 µs; RSS ≤ 23 MB
- **Android/Termux: не измерялся** — host-числа не переносимы в device-гейты; flash-fsync INCONCLUSIVE в v0.10

## Документация

docs_v11…docs_v15/{ARCHITECTURE,CHANGELOG,TEST_REPORT,BENCHMARK_REPORT,LIMITATIONS,REPRODUCE,PLAN_NEXT}.md ·
результаты: results_v11…results_v15/ (логи, bench JSON, прогоны) · PROGRESS.md — статус фаз ·
исторические отчёты v0.1–v0.10 в docs_v*/ — неизменяемые.

## Честные ограничения (кратко)

Поймите перед использованием: маленькая проверенная поверхность; русский язык только;
гипотетика и цитаты не сохраняются; обучение на синтетике demo-домена (не доказательство
языковой компетентности); артефакты дают +2pp маршрутизации без деградаций, и это предел
того, на что собранные 265 train-примеров способны. Полные списки — в docs_vNN/LIMITATIONS.md.

License: MIT. Установка: `pip install -e .`
