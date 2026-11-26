# PLAN_NEXT.md — v0.14 (reasoning + генерация ответов из доказательств)

Цель v0.14: детерминированный pipeline parse→retrieve→resolve→reason→verify→compose→
evidence-check; ответы только из подтверждённых данных; каждое предложение ответа
ссылается на evidence-id.
- EvidenceSet: facts (statement_ids + provenance), graph statuses, применённые правила,
  недостающие предпосылки; статусы supported|partial|unknown|conflict (категориальные).
- Правила: is_a transitive, capital_of⇒located_in(столица, страна), located_in transitive
  (глубина ≤2–3 по умолчанию, конфигурируется), сравнение числовых литералов, противоречие
  assert/deny → conflict. У каждого правила — предусловия, область, тесты.
- Multi-hop с ограничением глубины/ширины и защитой от циклов (visited set).
- Compose по шаблонам русского из EvidenceSet; нет EvidenceSet → нет фактового предложения.
- UNKNOWN: «чего не хватает»; CONFLICT: «какие утверждения противоречат».
- Adversarial/negative тесты: ложная предпосылка, неверная сущность, пропущенный факт,
  противоречие, цикл, несколько ответов, вне домена.
- Метрики: exact-answer accuracy, evidence precision/recall, unsupported-answer rate,
  abstention accuracy, latency p50/p95, peak RSS (bench_v14).
- End-to-end от строки до ответа (тестировать через DialogueManager/KB).
