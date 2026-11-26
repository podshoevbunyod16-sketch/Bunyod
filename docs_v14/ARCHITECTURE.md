# ARCHITECTURE.md — RASM-CIM v0.14 (EvidenceSet reasoning)

## Pipeline (детерминированный, без внешних LLM)
parse (v0.11 analyze) → retrieve (KB v0.12: подтверждённые утверждения, +reverse-link)
→ resolve (сущности/предикаты) → reason (правила RULES + BFS KB-fetch, depth ≤2 default,
≤3 cap, expansion budget 64, fetch cap 128) → verify (EvidenceSet) → compose (шаблоны)
→ evidence-check (финальный гейт).

## Evidence (`reasoning/evidence.py`)
EvidenceItem: role direct|derived|missing · statement_id / rule_id+support_ids / note ·
provenance passthrough (source, doc_id, reliability) · depth.
EvidenceSet.confidence — КАТЕГОРИАЛЬНАЯ: supported | partial | unknown | conflict.
Оценка выводится из СТРУКТУРЫ доказательств, не из весов:
- supported: есть подтверждённый прямой факт (raw) без пропусков/конфликта
- partial:  только derived-цепочки, или упомянуты недостающие предпосылки
- unknown: нет используемого доказательства
- conflict: противоречащие активные утверждения (assert+deny / многозначный
  functional predicate) — показываются вместе, ничего не скрывается.

## Правила (`reasoning/rules.py`)
R01 is_a транзитивно: A→B, B→C ⇒ A→C.
R02 capital_of(Страна, Город) ⇒ located_in(Город, Страна).
R03 located_in транзитивно: A→B, B→C ⇒ A→C.
Доказательства правил — всегда пара statement-id из KB (полная цепочка в
EvidenceSet.applied_rules и в суффиксе предложения `[derived via R01; supports: st_…, st_…]`).
Гипотезы никогда не посылка. Кольца: depth-счётчик + budget исчерпания (нет циклов/зависаний).

## Verdict vs confidence
verdict (TRUE|FALSE|UNKNOWN|CONFLICT, TruthGraph9-совместимо) — про УТВЕРЖДЕНИЕ.
confidence (supported|partial|unknown|conflict) — про ДОКАЗАТЕЛЬСТВА.
Формула: conflict+conflict→CONFLICT · unknown→UNKNOWN · partial-only→UNKNOWN ·
supported→TRUE. FALSE только при явном deny-факте (пингвин: «нет [st_…]», verdict FALSE).

## Compose & evidence-check (`pipeline._finalize`)
Каждое содержательное предложение помечается `[st_…]` или `[derived via R…; supports: …]`.
Финальный гейт отбрасывает предложение, ссылающееся на отсутствующий в EvidenceSet id,
либо заменяет тело текста астеническим «Не хватает подтверждённых данных…».
Отсутствующий факт = missing-item + verdict UNKNOWN. Галлюцинации запрещены архитектурно:
формулировать нечем кроме шаблонов над EvidenceSet.

## Предикаты/карты
capital_of, population_of, height_of, wrote/written_by (канонизация v0.12),
is_a, located_in, может_летать, существует, имеет_спутник, кипит_при, написал→written_by.

## Интеграция
Standalone: `python -m rasm_cim_v14.reasoning` → ask(). Единая точка входа
(`python -m rasm_cim`, фазa C) подключает reasoning к режимам Ask/Chat.
В v0.13 dialogue слой не трогали (стабилен, 40 тестов зелёные).
