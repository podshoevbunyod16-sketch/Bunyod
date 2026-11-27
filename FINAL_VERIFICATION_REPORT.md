# FINAL_VERIFICATION_REPORT.md — RASM-CIM v0.15.0

Дата: 2026-11-27 · Аудитор: независимый инженер по качеству (Arena Agent Mode)
Репозиторий: podshoevbunyod16-sketch/Bunyod · Ветка: `arena/aa17ff3d-bunyod` · PR #1
Исходный commit (задание): **56d275ce690e78562c14b7c40b9afbd3e4b9e424** 
Отчёт фиксирует состояние ветки на 2026-11-27; каждый коммит верификации продвигает HEAD, поэтому авторитетный итоговый HEAD — tip ветки в PR #1 (это же относится к финальному архиву, чей SHA256 публикуется в теле PR).
Полная версия: RASM-CIM 0.15.0.

## Вердикт (коротко)

**PR #1 технически готов к review ПО ЛИНЕ ПРОВЕРОК, ПЕРЕЧИСЛЕННЫХ В ЭТОМ ОТЧЁТЕ.**
Все доступные проверки выполнены и зелёные; подтверждённые дефекты исправлены с
регрессионными тестами; Android/Termux и локальная сборка Rust физически недоступны
в этой среде и честно помечены NOT TESTED / verified-by-CI. Решение о merge — за владельцем.

## Обязательная таблица

| Проверка | Результат |
|---|---|
| Исходный commit | `56d275ce690e78562c14b7c40b9afbd3e4b9e424` (verified; ветка была содержательно откачена в песочнице, восстановлена из origin, потерь данных нет — см. «Инциденты») |
| Python tests | **362 passed / 0 failed / 0 skipped** — одна команда `pytest test_rasm*.py rasm_cim_v8/tests tests_v9 tests_v10 tests_v11 tests_v12 tests_v13 tests_v14 tests_v15 tests_phase_c rasm_cim_rust/tests/test_cross_language.py -q` (breakdown ниже; clean venv, Python 3.11.2) |
| Rust tests | **cargo test, cargo test --release — PASS** (GitHub Actions, job `rust`, run 45919888140; канал `stable` из rust-toolchain.toml). Локально — NOT EXECUTED: в песочнице нет Rust, apt заблокирован, static.rust-lang.org/crates.io/sh.rustup.rs недоступны (000). Количество тестов офлайн не видно (логи CDN вне allowlist сети песочницы) |
| Cross-language | **28 passed** локально (prebuilt release .so, ELF64 x86-64, abi3-py38 → Python 3.11); **CI `Cross-language equivalence against CI-built .so` — success** (cargo build + RASM_CIM_EXT). Тест-файл v0.7 заморожен (md5-locked) и не менялся; добавлен аддитивный `rasm_cim_rust/tests/conftest.py` (discovery + честный SKIP с причиной) |
| Package build | **PASS** — `python -m build` → `dist/rasm_cim-0.15.0-py3-none-any.whl` + `.tar.gz`; данные включены (v11 data, v15 corpus+learned); локальных путей нет; предупреждения сборки устранены (license → SPDX, data-dirs → explicit namespace packages); воспроизводимость: sdist → wheel |
| Package install | **PASS** — чистое venv, `pip check` чисто; `import rasm_cim*` из чужой cwd OK; repo не на sys.path |
| CLI smoke tests | **PASS** — `rasm-cim --version/--help`, `python -m rasm_cim`, `--demo ask` → «Франция: Париж [TRUE] [st_c2c72f875a98e0b3]», `ask --json`, `evaluate --split dev` (exit 0, regression 6/6), `memory stats/export/import` (report JSON — см. Исправления), `system-info`, `teach`, legacy `rasm-cim-demo` / `python -m rasm_cim_v9.cli`, CONFLICT → **exit 4**, из чужой cwd — всё PASS |
| Functional tests | **PASS с ограничениями** — подтверждено: TRUE/UNKNOWN/CONFLICT ответы с evidence-id; pronoun continuity; multi-question; KB export→wipe→import (идемпотентность: 2-й import imported=0/skipped_duplicates=89); torn journal tail → recovery (replay_torn_tail=1, 55 сущностей восстановлено); corrupt context-file → не падает; empty/huge (10k)/emoji/пустая строка → UNKNOWN без traceback; English/Tajik → честный UNKNOWN; rollback train→lv_000 восстановил pointer (hash b491d515…) в изолированной копии пакета. Ограничения — раздел «Пределы» |
| Security checks | **PASS** — pip-audit: уязвимостей нет (и в runtime, и в dev-окружении); `pip check` чист; секреты в треке и истории не найдены; `.env`/ключи не трекаются; shell=True отсутствует; bare except отсутствует; `eval` только в замороженном v0.1-исследовательском коде на внутренних лямбдах (фиксированная таблица операций) и в AST-safe v0.2 (самопроверка); corrections.jsonl в .gitignore; запись/удаление user-путей не обнаружено в коде пакета |
| Performance | **Реальные измерения (Linux x86_64 host, Python 3.11.2)** — startup (rasm-cim --version) p50 ≈ 76–80 ms; ask p50 ≈ 0.6–0.8 ms (in-process, 300 reps ×3 trials); ask p50 (subprocess) ≈ 82–90 ms; ru_maxrss(100 asks) ≈ 18.9 MB; evaluate peak_rss ≈ 30.9–79.8 MB; determinism evaluate ×3 — identical (модульно latency/RSS-полей); KB replay: open 0.1 ms @0 фактов / 21 ms @1k / 103 ms @5k; scaling fix: ask p50 @ +1000 фактов **19.3 ms → 0.60 ms**, @ +5000 **81 ms → 7.9 ms** (lru_cache); остаточная сложность O(#active entities) ≈ 8 ms @10k — задокументировано |
| Android/Termux | **NOT TESTED** — нет устройства/эмулятора; все числа выше — host (x86_64) и не переносятся; flash fsync INCONCLUSIVE с v0.10; профиль: `results_verify/ANDROID_TERMUX_PROFILE.md` (команды + требования к прогону). Действие: владелец запускает на Termux и прикладывает логи |
| GitHub Actions | **Run 45919888140 (commit 105fe83): ВСЕ 6 jobs SUCCESS** (python 3.10/3.11/3.12/3.13, rust, package); freeze-гейты зелёные во всех; run 45919884224 (push 105fe83) — success; первый прогон (7162869) — FAILURE только job rust (не хватало python-deps), исправлено в 105fe83. Логи raw-скачиванием из песочницы недоступны (CDN вне allowlist); свидетельство — jobs API (`api.github.com`), URL приведён в `results_verify/stage_h_ci.log` |

## Breakdown Python suite (одна команда, clean venv)

| Набор | Тестов |
|---|---|
| Исторические (root test_rasm*.py + rasm_cim_v8/tests + tests_v9 + tests_v10) | 163 |
| v0.11 (tests_v11) | 56 |
| v0.12 (tests_v12) | 27 |
| v0.13 (tests_v13) | 40 |
| v0.14 (tests_v14, incl. 2 новых регресс-теста _stems) | 23 |
| v0.15 (tests_v15, incl. регресс-тест pointer leak) | 15 |
| Phase C unified CLI (tests_phase_c, incl. регресс import-report) | 10 |
| Cross-language (rasm_cim_rust/tests/test_cross_language.py) | 28 |
| **Итого** | **362** |

SKIP: нет. Hypothesis property tests (test_m4_properties.py) — в составе набора, зелёные.

## Пределы / что НЕ подтверждено

1. **Android/Termux** — NOT TESTED (см. таблицу).
2. **Локальная сборка Rust** — NOT EXECUTED (нет toolchain в песочнице). Rust-поведение подтверждено решением CI (cargo test/release/clippy + cdylib + cross-language против CI-собранного .so). Количество Rust-тестов офлайн не видно.
3. intent_acc-улучшение (v0.15): +2.3–2.4pp, 2–4 срабатывания fallback — **не статистически значимо**; НЕ заявлено как «модель стала лучше».
4. Corpus — синтетика (54 семейства); morphology — псевдо-стемминг; интент — 7 классов; таджикского нет; ответов «понимаю русский» НЕ заявляется: система «не фабрикует неподтверждённое + детерминирована + откатываема».
5. Correction hash-chain — аудит, не защита от пересоздания файла с пересчётом хэшей (нет внешнего witness).
6. verdict_acc=1.0 — только на KB-answerable подмножестве (n=29 dev/32 test).
7. `memory import` JSON report — исправлен при этой верификации (см. ниже); внутреннее несоответствие ключей `skipped` (fatal path) vs `skipped_duplicates` (success path) в замороженном v0.12 — **заморожено, не изменено**, задокументировано.
8. `teach` не дедуплицирует идентичные факты (append-only bill); идемпотентен только JSONL import. Наблюденное поведение, не дефект.
9. `_find_entities` — O(активные сущности) на запрос (~8 ms @10k); фундаментальная стоимость текущего дизайна без изменения замороженного v0.12.
10. `system-info` из установленного колеса показывает `freeze_baseline_v0.10: UNKNOWN` — манифесты лежат в репозитории, не в wheel; значение честное (UNKNOWN, не «OK»).
11. Демо-корпус opt-in; после smoke в `~/.rasm-cim/` могла остаться demo-KB (удалить: `rm -rf ~/.rasm-cim`).
12. Мульти-референтное «её» → CLARIFY (по дизайну, проверено); «А её население?» при единственном референте разрешается (проверено).

## Проблемы, найденные этой верификацией, и их устранение (все — по коммитам)

1. **Дефект (корректность, unified CLI):** `memory import` печатал `{}` вместо отчёта о импорте (ImportReport — подкласс dict; `res.__dict__` всегда пуст). Коммит `323c9ef`. Регресс: `tests_phase_c::TestMemoryMode::test_export_import_roundtrip` расширен (непустой отчёт, imported≥1, идемпотентный повторный импорт через `skipped_duplicates`).
2. **Дефект (производительность):** `_find_entities` (v0.14) пересчитывал морфологию каждого entity на каждый запрос: ask p50 ≈19.3 ms @+1000 фактов, ≈81 ms @+5000. Коммит `cd8cd6f`: `_stems` → `lru_cache(maxsize=32768)`; первый вариант maxsize=8192 **не помог** при +5000 фактах (трэшинг кэша: 10k сущностей > 8192 ключей) — измерено и отклонено. Итог: 0.60 ms (+1000) / 7.9 ms (+5000). Регресс: `tests_v14/test_stems_cache.py` (cache identity, без тайминг-ассертов). Замороженный v0.1–v0.10 не затронут; v0.14 получил документированный bugfix (прецедент — два bugfix-а v0.14 в v0.15).
3. **Дефект (эргономика/корректность тестов):** cross-language тесты требовали ручной PYTHONPATH (иначе collection ImportError). Коммит `acc817e`: аддитивный `conftest.py` — discovery (existing import → `$RASM_CIM_EXT` → shipped prebuilt .so, только linux-x86_64, копирование во временный каталог); недоступность → честный module-SKIP с причиной (проверено принудительно через `RASM_CIM_EXT_DISABLE=1`). Замороженный v0.7 тест не изменён. debug-.so (benchmarks/dbg) намеренно НЕ используется (Rust debug паникает на overflow — семантика иная).
4. **Packaging-предупреждения:** `license = {text = ...}` (deprecated) → `license = "MIT"`; data-dirs (`v11.data`, `v15.corpus`, `v15.learned`) объявлены explicit namespace-пакетами — предупреждения сборки устранены, wheel-набор неизменен (79 файлов, данные на месте). Коммит `323c9ef`.
5. **CI отсутствовал.** Коммиты `7162869` + `105fe83`: `.github/workflows/ci.yml` (jobs: python 3.10–3.13, rust, package). Первый прогон реально упал на job rust (нет python-deps для pyo3/pytest) → исправлено setup-python + pip-install шагом; повторный прогон — всё зелёное (evidence: jobs API).
6. **Инцидент (восстановлен):** при первой проверке `train`/rollback PYTHONPATH был задан, но cwd оставался репо-корнем → `python -m rasm_cim_v15.cli train` создал `rasm_cim_v15/learned/lv_001.json` В РЕПОЗИТОРИИ. `lv_001.json` вынесен из репозитория в `/tmp/verify_smoke/lv_001_accidental_repo_train.json` (сохранён, не коммитился, не удалён); `current.json` подтверждён byte-identical коммиту (`git diff` пуст; md5 `5a2c518a…`); rollback перепроверен в изолированной копии пакета (cwd=/tmp, PYTHONPATH=копия): train → lv_001 создан в копии, rollback → pointer восстановлен b491d515…, репо не тронуто. Логи: `stage_e_rollback.log` (пометка SUPERSEDED) + `stage_e_rollback_v2.log`.
7. **Песочница откачена к base-commit f784e76** в начале сессии (local .git lost work while fs persisted): восстановлено `git reset --hard origin/arena/aa17ff3d-bunyod` после полного бэкапа дерева; статус чист; HEAD = целевой коммит. Потерь нет (все work-файлы совпали с коммитами; дифа после reset не было).
8. **Свежие окружения теряют pip-пакеты** — discoveries записаны в `requirements-dev.txt`/`pyproject[dev]` (scikit-learn для frozen v0.5, numpy для frozen v0.1). CI ставит зависимости заново на каждый прогон — это гарантирует воспроизводимость.
9. **Freeze-дельта уточнена** (`results_v15/freeze_delta_report_v0.10_to_v0.15.md`): pyproject md5 эволюция 3f017453… → a66a96c9… → 18048e5f…; requirements-dev.txt — единственный dev-инфраструктурный exception; историческая md5-проверка — ровно 3 documented exceptions.

## Команды воспроизведения (host)

```bash
git checkout arena/aa17ff3d-bunyod && git rev-parse HEAD    # …should be >= 3e40d0c
python -m venv /tmp/venv && /tmp/venv/bin/pip install -r requirements-dev.txt
/tmp/venv/bin/python -m pytest test_rasm*.py rasm_cim_v8/tests tests_v9 tests_v10 tests_v11 \
    tests_v12 tests_v13 tests_v14 tests_v15 tests_phase_c rasm_cim_rust/tests/test_cross_language.py -q
/tmp/venv/bin/python -m rasm_cim --demo ask "Какая столица у Франции?"
python -m build && python -m venv /tmp/inst && /tmp/inst/bin/pip install dist/rasm_cim-*.whl
cd /tmp && /tmp/inst/bin/rasm-cim system-info
md5sum -c results_v9/v08_freeze_manifest.md5                 # 0 mismatch
md5sum -c results_v11/freeze_manifest_baseline_v0.10.md5     # 3 documented FAILED lines
find rasm_cim_rust -name '*.rs' -exec md5sum {} + | sort | md5sum   # 451bbfa52e67a72b04c0046d9df4e9d5
```

CI: `gh run watch` для последнего run; jobs API: `gh api repos/podshoevbunyod16-sketch/Bunyod/actions/runs/45919888140/jobs`.

## Артефакты верификации

- `results_verify/` — stage_a_baseline, stage_b_* (suite/xlang/full), stage_c_rust, stage_f_performance (+addendum),
  stage_g_security, stage_gate_final, stage_gate_after_perf, stage_h_ci, ANDROID_TERMUX_PROFILE.md.
- Финальный архив `rasm-cim-v0.15.0.tar.gz` (git archive HEAD) — SHA256 публикуется в теле PR #1
  и дублируется в `results_verify/final_verification.log`. Само-ссылка на себя в этом файле
  невозможна (артефакт содержит результат сборки, не наоборот), поэтому хэш хранится снаружи.
- FINAL_REPORT (продуктовый итог фазы): `docs_v15/FINAL_REPORT.md`.

## Итоговая целостность

- freeze v0.8: **0 mismatch** (md5sum -c).
- baseline-272: **ровно 3 documented exceptions** (README.md, pyproject.toml, requirements-dev.txt).
- Rust v0.7 source-set md5 = **451bbfa52e67a72b04c0046d9df4e9d5** — неизменен.
- Frozen v0.1 `rasm_cim/__init__.py`: byte-identical (md5 9c19f403e3e5f9d2f1d18739373b6233).
- Исторические тесты (163) зелёные до и после всех изменений цикла.

## Решение по PR #1

Готов к review с оговорками: (1) принять 3 spec-mandated freeze-exception как таковые;
(2) device-гейты открыты до прогона на Termux ARM64 (см. ANDROID_TERMUX_PROFILE.md);
(3) Rust verification = CI-свидетельство, не локальный прогон; (4) 362 теста — не «понимание русского»,
а отсутствие фабрикации неподтверждённого + детерминизм + откат.
