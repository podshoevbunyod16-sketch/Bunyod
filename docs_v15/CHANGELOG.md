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

## Compatibility
- v0.11…v0.14 frozen/untouched (кроме 2 багфиксов выше); 163 исторических теста зелёные;
  freeze v0.8: 0 mismatch; baseline-272: 0 mismatch; Rust v0.7 хэш неизменён.
- Corrections НЕ пишут в KB напрямую; only fold into artifacts (кроме kind=answer — только лог+promote API, не автоприменяется).
