"""v0.11 intent classifier — deterministic weighted rules, Russian only.

Design rules (honest constraints):
  * NO machine learning here. Scoring = sum of matched rule weights per
    intent, capped at 0.99. Deterministic; every decision is explainable via
    `matched` (rule ids) and `scores`.
  * If the winning score < THRESHOLD → "unknown". Low confidence NEVER
    fabricates an intent (spec: unsupported question → unknown).
  * Non-Cyrillic input → "unknown" (no imitation of English/Tajik support).
  * Single-label per message. Multi-question messages are split later by the
    v0.13 dialogue layer; at this layer the whole message is one intent.
  * Rules were tuned on the TRAIN split of the committed corpus only;
    dev/test splits stayed untouched during rule development.
"""
from __future__ import annotations

import re
from typing import List, Pattern, Tuple

from .models import IntentResult

INTENTS = ("greeting", "ask_fact", "ask_definition", "yes_no", "compare",
           "list", "unknown")
THRESHOLD = 0.5

# (rule_id, intent, weight, regex) — regex applied to normalized text
_RULES: List[Tuple[str, str, float, str]] = [
    # greeting
    ("g_word",        "greeting", 0.85, r"\b(привет(ик)?|здравствуй(те)?|салют|хай|хэллоу)\b"),
    ("g_time_of_day", "greeting", 0.90, r"\b(доброе утро|добрый день|добрый вечер|доброй ночи|доброго дня)\b"),
    ("g_phrase",      "greeting", 0.60, r"\b(доброго времени суток|как дела|приветствую)\b"),
    # ask_definition
    ("d_what_is",     "ask_definition", 0.90, r"\bчто (такое|значит|означает)\b"),
    ("d_who_is",      "ask_definition", 0.90, r"\bкто (такой|такая|такие)\b"),
    ("d_define",      "ask_definition", 0.85, r"\b(определени\w* (слова|термина?)|что называют|что имеется в виду)\b"),
    # compare
    ("c_compare",     "compare", 0.90, r"\bсравни(те|ть|мте)?\b"),
    ("c_difference",  "compare", 0.90, r"\b(чем отличается|в ч[её]м разница|в чем разница|разница между)\b"),
    ("c_comparative", "compare", 0.80, r"\b(лучше|хуже|быстрее|медленнее|старше|моложе|больше|меньше|выше|ниже)\b"),
    ("c_or_question", "compare", 0.45, r"\bили\b[^?]*\?\s*$"),
    # list
    ("l_imperative",  "list", 0.90, r"\b(перечисли\w*|назови все|назови несколько|покажи все)\b"),
    ("l_what_all",    "list", 0.85, r"\b(какие (есть|бывают|существуют)|какой (список|перечень)|список|перечень)\b"),
    ("l_name",        "list", 0.40, r"\bназови\b"),
    # yes_no
    ("y_true_or_not", "yes_no", 0.90, r"\b(правда ли|верно ли|так ли|точно ли)\b"),
    ("y_aux_ли",      "yes_no", 0.85, r"\b(есть|может|умеет|бывает|будет|можно|получается|умеют|живёт|живет)\s+ли\b"),
    ("y_any_ли",     "yes_no", 0.55, r"\b[а-яё][\w-]*\s+ли\b"),
    # ask_fact
    ("f_question_word", "ask_fact", 0.65,
        r"\b(кто|что|где|когда|откуда|куда|почему|зачем|сколько|какой|какая|какое|какие|чей|чья|чьё|который|которая|которое)\b"),
    ("f_bare_question", "ask_fact", 0.15, r"\?\s*$"),
    ("f_name_answer",   "ask_fact", 0.55, r"\bназови\b"),
    ("f_elliptical_capital", "ask_fact", 0.60, r"\bстолица\b[^?]*\?\s*$"),
]

_COMPILED: List[Tuple[str, str, float, Pattern[str]]] = [
    (rid, intent, w, re.compile(rx)) for rid, intent, w, rx in _RULES
]


def classify_intent(normalized_text: str, has_cyrillic: bool = True) -> IntentResult:
    text = (normalized_text or "").strip()
    if not text:
        return IntentResult("unknown", 0.0, (), (), "empty_input")
    if not has_cyrillic:
        return IntentResult("unknown", 0.05, (), (), "no_cyrillic_no_support")

    scores: dict[str, float] = {}
    matched: dict[str, list[str]] = {}
    for rid, intent, weight, rx in _COMPILED:
        if rx.search(text):
            scores[intent] = scores.get(intent, 0.0) + weight
            matched.setdefault(intent, []).append(rid)

    if not scores:
        return IntentResult("unknown", 0.1, (), (), "no_rule_matched")

    capped = {k: min(0.99, v) for k, v in scores.items()}
    best_intent = max(capped, key=capped.get)
    best = capped[best_intent]
    ordered = tuple(sorted(capped.items(), key=lambda kv: -kv[1]))

    if best < THRESHOLD:
        return IntentResult("unknown", round(best, 3), tuple(matched.get(best_intent, ())),
                            ordered, f"below_threshold({best:.2f}<{THRESHOLD})")
    return IntentResult(best_intent, round(best, 3), tuple(matched[best_intent]),
                        ordered, "rules")
