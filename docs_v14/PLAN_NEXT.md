# PLAN_NEXT.md — v0.15 (managed improvement / training)

§8 из ТЗ: controlled improvement WITHOUT overfitting/test leakage.

- Corpus: 200–500 train / 50–100 dev / 50–100 test; template-source splits
  (разные шаблоны фраз в разных сплитах; полe source_id в каждом примере).
- Обучаемые артефакты (версионированные с откатом): словарь алиасов KB-объектов,
  intent-веса (НЕ семантика TruthGraph9), rule-precedence (НЕ semantics правил),
  concept-mapping (is_a-подсказки для категоризации), alias-таблица entity-link
  с ambiguity-метрикой (доля вредных слияний).
- НЕ учим: frozen semantics (TRUE/FALSE/UNKNOWN/CONFLICT), замороженные v0.1–v0.8,
  форматы v1–v3, Rust v0.7. Downgrade/rollback командa: learned-rollback на любой
  прошлый artifact-version.
- Correction interface: пользователь помечает wrong entity / relation / intent / answer;
  correction events сохраняются с provenance (epoch, session id, artifact version);
  verifiable before/after diff по тем же тестовым корпусам.
- Offline eval: baseline vs post-training на test-сплите, per-component ablation,
  regression suite (исторические + v11+v12+v13+v14 — все зелёные ПОСЛЕ обучения),
  anti-overfit (dev id holdout), interference (старые задачи до/после),
  memory/latency budgets (RSS, p50/p95) задокументированы.
- Честность: если измеримого улучшения нет → вердикт NO IMPROVEMENT фиксируется
  в results_v15; не выдавать за успех.
- Таджикcкий язык — только отдельным экспериментальным пакетом в будущем, вне скоупа v0.15.
