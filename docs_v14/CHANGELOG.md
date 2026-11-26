# CHANGELOG — v0.14 (EvidenceSet-связанный reasoning; 2026-11-26)

## Added
- rasm_cim_v14/reasoning/{evidence,rules,pipeline}.py:
  EvidenceSet/EvidenceItem (direct|derived|missing), категориальная confidence;
  правила R01 (is_a транз.), R02 (capital_of⇒located_in), R03 (located_in транз.)
  с KB-BFS fetch (max_depth ≤3, expansion budget 64, fetch cap 128);
  детерминированный pipeline parse→retrieve→resolve→reason→verify→compose→evidence-check;
  счётчик hypothesis изоляции (гипотезы — никогда не посылка);
  verdict (TruthGraph9-совместимо) отделён от confidence (категориальна);
  составление ответа ТОЛЬКО шаблонами над EvidenceSet; каждое содержательное предложение
  содержит evidence ids; финальный evidence-check отбрасывает неподтверждённые ссылки.
- tests_v14/test_reasoning.py: 21 тест — direct/multi-hop (глубина 2–3, cap),
  цикл-и-ширина termination, premise-hypothesis isolation, functional-conflict,
  abstain/UNKNOWN, determinism, evidence-gate (_finalize unit), adversarial.
- bench_v14/bench_reasoning.py → results_v14/bench_reasoning.json:
  p50 ≈236–343 µs, p95 ≈271–434, p99 ≈306–671; RSS 22.7 MB;
  answer_accuracy 1.0 (11 ручных вопросов), unsupported 0.0, abstention 1.0.

## Fixed (найдено в ходе этапа)
- Runner: max_depth=2 по умолчанию — chain depth-2 definition теперь не считается
  partial-only; raw-подкреплённые ответы → supported (честнее).
- _aggregate_verdict: пустой набор статусов → UNKNOWN (было TRUE — опасный дефект).
- yes_no: вердикт FALSE при чистых deny; CONFLICT при assert+deny.
- Evidence joint ordering: предложения с несколькими derived суммируют supports.
- EvidenceSet.status: missing-only → partial (было unknown).
- seed-demo (v0.12 CLI): единый тип сущности по имени (Пушкин/Толстой/Достоевский:
  было thing+person → дубликаты); personas-type-override; entities 58→55 на демо KB.
- Definition-ответ перечисляет цепочку по глубине, ссылается на все supports.
- "Голубица"-класс вопросов (нет сущностей) → UNKNOWN с missing-прerequisite,
  без rulе-конфигурации, без записи в KB.

## Compatibility
- v0.11/v0.12/v0.13 пакеты не изменены (кроме seed-demo datarepair в v0.12 CLI);
  163 исторических + 56 + 27 + 40 + 21 = 307 тестов зелёные. Freeze v0.8: 0 mismatch;
  baseline-272: 0 mismatch; Rust v0.7 хэш без изменений.
