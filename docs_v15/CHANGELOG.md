# CHANGELOG — v0.15 (managed improvement; 2026-11-26)

## Added
- rasm_cim_v15/: corpus_build.py (54 семейства, split by family, seed 0,
  manifest.json + sha256, 265/84/88), artifacts.py (rasm-cim-learned/1,
  Naive Bayes intent-fallback + predicate_hints + alias_delta; lv_000… с parent-
  ссылками и sha256; rollback перевыставляет current.json, история не стирается),
  pipeline15.py (ask_v15: analyze → fallback-on-unknown или predicate-hint path,
  ответы всё ещё только из EvidenceSet), corrections.py (hash-chained correction
  events, validate_event, apply_corrections с before/after dev-метриками и
  regression-пробами, find_conflicts с latest-wins + surfacing, undo), evaluate.py
  (intent_acc на dev/test, verdict_acc KB-answerable-подмножество, unsupported,
  latency, RSS, regression probes), cli.py (build-corpus, corpus-info, train,
  eval, correct, apply-corrections, corrections, rollback, ask).
- tests_v15/test_v15.py: 14 тестов — размеры/расщепления/leakage/детерминизм,
  roundtrip/откат/версия parent, hash-chain tamper, relation-correction e2e + undo,
  conflicting corrections surfaced, alias ambiguity refusal, fallback correction.
- Корпус committed (train/dev/test.jsonl + manifest.json) — артефакт репозитория.
- rasm_cim_v15/learned/lv_000.json + current.json — committed baseline-artifact.
- .gitignore: corrections.jsonl (runtime, приватные данные).

## Measured (честно)
- intent_acc dev: 0.8333 → 0.8571 (fallback: 2 срабатывания, +2.4pp).
- intent_acc test: 0.7614 → 0.7841 (fallback: 4 срабатывания, +2.3pp).
  Дельта маргинальна (n≈85–88, не significance-тестировалась); НЕ претендуем на «модель стала лучше».
- verdict_acc: 1.0 dev (n=29) / 1.0 test (n=32) на KB-answerable подмножестве.
- unsupported 0.0; latency p50 ≈0.6–0.7 ms; RSS 18–23 MB; regression probes 6/6 ok.
- Abl. attribution.: 100% дельты intent_acc — вклад fallback; predicate_hints/alias_delta в lv_000 = 0.

## Fixed in other packages (found by held-out test, not by unit fuzz)
- pipeline keyword stems: added написан→written_by, существов/реальн→существует;
  _definition now checks assert+deny → CONFLICT (было: оба факта печатались как supported).
- evaluate.py verdict_acc metric redefined на KB-answerable подмножество (определение было неверным:
  ожидала TRUE от вопроса «что такое млекопитающее?», на который подтверждать факт не может).

## verification cycle (2026-11-27, commits acc817e..HEAD)
### Fixed
- rasm_cim/cli.py `memory import`: ImportReport — подкласс dict, `res.__dict__` всегда {} —
  stdout был пустым JSON. Печатает содержимое отчёта; регресс-тест в tests_phase_c.
- rasm_cim_v14/reasoning/pipeline.py `_find_entities`: O(entities) пересчёт морфологии на каждый
  ask (~19 ms @ +1000 фактов, ~81 ms @ +5000) → `_stems` закэширован (lru_cache 32768);
  ask p50: 19.3 ms → 0.60 ms (+1000), 81 ms → 7.9 ms (+5000). Регресс-тесты tests_v14/test_stems_cache.py.
### Added
- rasm_cim_rust/tests/conftest.py — обнаружение rasm_cim_ext (existing import → $RASM_CIM_EXT →
  shipped prebuilt .so, linux-x86_64); недоступность => честный SKIP с причиной; frozen v0.7 тест не изменён.
- .github/workflows/ci.yml — python 3.10–3.13 (тесты + freeze-гейты), rust (cargo test/release/clippy,
  cdylib, cross-language против CI-собранного .so), package (build + установка колеса + foreign-cwd smoke).
- FINAL_VERIFICATION_REPORT.md (корень), results_verify/ (все логи верификации).
### Measured (host, см. результаты verification)
- 362 passed одной командой; pip-audit чист; CI jobs API: всё зелёное (см. results_verify/stage_h_ci.log).

## v0.15 packaging (Phase C/D, 2026-11-26, commit ee81645)
### Added
- `python -m rasm_cim` — unified CLI (chat/ask/teach/memory/evaluate/system-info);
  stdout=результаты, stderr=диагностика; exit codes 0/2/3/4/5; `--demo` strictly opt-in;
  help/exit/reset/teach/inspect in chat; evidence печатает st_id→provenance;
  legacy `python -m rasm_cim_v9.cli` untouched (regression-tested).
- pyproject → 0.15.0 (packages v8…v15 + rasm_cim; entry script `rasm-cim`);
  requirements-dev.txt + pyproject[dev] дополнены scikit-learn (frozen v0.5 tests) и numpy
  (frozen v0.1 import). 332 passed.
- docs_v15/FINAL_REPORT.md — честный итог v0.11…v0.15 (host vs device, что не сделано).

### Fixed
- `rasm_cim_v15/artifacts.py`: save/current/rollback игнорировали `out_dir` при записи/чтении
  `current.json` → тесты утёкали dangling-указатель в tracked-файл `rasm_cim_v15/learned/current.json`.
  Исправлено; регресс-тест test_save_does_not_touch_repo_pointer.

### Freeze
- v0.8 manifest: 0 mismatch. baseline-272: ровно 3 задокументированных исключения
  (README.md, pyproject.toml, requirements-dev.txt) — results_v15/freeze_delta_report_v0.10_to_v0.15.md.
  Frozen v0.1 rasm_cim/__init__.py восстановлен byte-identical после случайного касания.
- Rust v0.7: sha256-set 451bbfa5… без изменений.

## Compatibility
- v0.11…v0.14 frozen/untouched (кроме 2 багфиксов выше); 163 исторических теста зелёные;
  freeze v0.8: 0 mismatch; baseline-272: 3 задокументированных исключения; Rust v0.7 хэш неизменён.
- Corrections НЕ пишут в KB напрямую; only fold into artifacts (кроме kind=answer — только лог+promote API, не автоприменяется).
