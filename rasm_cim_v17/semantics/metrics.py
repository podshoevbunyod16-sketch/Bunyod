"""Measured quality of the semantic layer — no adjectives, only ratios.

Four numbers, each defined so that a reader can recompute it by hand:

    intent accuracy     доля реплик, у которых MeaningIR.speech_act совпал
                        с ожидаемым в корпусе
    unknown rate        доля реплик, распознанных как UNKNOWN (часть ошибок,
                        но её надо видеть отдельно: «не понял» лучше, чем
                        «понял неправильно»)
    entity P/R/F1       микро-усреднение по упоминаниям: сколько ожидаемых
                        сущностей попало в focus и сколько лишних там оказалось
    answer accuracy     доля вопросов, где совпал статус ответа И все
                        обязательные подстроки присутствуют
    unsupported rate    доля фактов в ответах, у которых НЕТ записи об
                        источнике — структурный прокси, а не оценка истинности.
                        Доказательство вывода НЕ засчитывается вместо источника:
                        посылки доказательства сами должны быть прослеживаемы.

Что эти метрики НЕ значат: они не измеряют «понимание».  Корпус маленький и
написан нами; в отчёте всегда указываются размер и sha256 файла, чтобы число
нельзя было прочитать без его происхождения.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .corpus import Case, Corpus
from .meaning import AS_UNKNOWN, MeaningIR, meanings_for


def _canonical_names(dm, ids) -> list[str]:
    names = []
    for eid in ids:
        ent = dm.kg.resolver.get(eid)
        if ent is not None:
            names.append(ent.canonical)
    return names


def _first_meaning(dm, text: str) -> MeaningIR:
    meanings = meanings_for(text, resolver=dm.kg.resolver,
                            schema=dm.kg.schema, parser=dm.parser)
    if not meanings:
        return MeaningIR(text=text, norm=text, speech_act=AS_UNKNOWN)
    return meanings[0]


# --------------------------------------------------------------------------
# intent
# --------------------------------------------------------------------------

def evaluate_intents(cases: list[Case], dm) -> dict:
    """Speech-act accuracy with an explicit confusion matrix and the misses."""
    scored = [c for c in cases if c.intent]
    if not scored:
        return {"n": 0, "note": "в корпусе нет ожидаемых речевых актов"}
    correct = 0
    unknown = 0
    confusion: dict[str, dict[str, int]] = {}
    misses: list[dict] = []
    for c in scored:
        m = _first_meaning(dm, c.text)
        got = m.speech_act
        confusion.setdefault(c.intent, {})
        confusion[c.intent][got] = confusion[c.intent].get(got, 0) + 1
        if got == AS_UNKNOWN:
            unknown += 1
        if got == c.intent:
            correct += 1
        else:
            misses.append({"id": c.id, "text": c.text,
                           "expected": c.intent, "got": got,
                           "notes": list(m.notes),
                           "unknowns": list(m.unknowns)})
    return {"n": len(scored), "correct": correct,
            "accuracy": round(correct / len(scored), 4),
            "unknown": unknown,
            "unknown_rate": round(unknown / len(scored), 4),
            "confusion": confusion, "misses": misses}


# --------------------------------------------------------------------------
# entities
# --------------------------------------------------------------------------

def evaluate_entities(cases: list[Case], dm) -> dict:
    """Micro-averaged precision/recall over expected entity mentions."""
    scored = [c for c in cases if c.entities is not None]
    tp = fp = fn = 0
    misses: list[dict] = []
    for c in scored:
        m = _first_meaning(dm, c.text)
        got = _canonical_names(dm, [f.entity_id for f in m.resolved_focus
                                    if f.entity_id is not None])
        expected = list(c.entities or [])
        got_set, exp_set = set(got), set(expected)
        tp += len(got_set & exp_set)
        fp += len(got_set - exp_set)
        fn += len(exp_set - got_set)
        if got_set != exp_set:
            misses.append({"id": c.id, "text": c.text,
                           "expected": expected, "got": got,
                           "focus": [f.as_dict() for f in m.focus]})
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) else 0.0)
    return {"n_cases": len(scored), "tp": tp, "fp": fp, "fn": fn,
            "precision": round(precision, 4), "recall": round(recall, 4),
            "f1": round(f1, 4), "misses": misses}


# --------------------------------------------------------------------------
# answers
# --------------------------------------------------------------------------

def evaluate_answers(cases: list[Case], dm, respond=None) -> dict:
    """Answer accuracy + the share of answer facts without any evidence row.

    ``respond`` defaults to ``dm.say``; the CLI passes the *shipped*
    ``SemanticPipeline`` so the measured behaviour is the behaviour a user gets.
    """
    respond = respond or dm.say
    scored = [c for c in cases if c.answer_status or c.answer_contains]
    if not scored:
        return {"n": 0, "note": "в корпусе нет ожидаемых ответов"}
    correct = 0
    misses: list[dict] = []
    facts_total = 0
    facts_with_evidence = 0
    evidence_ids = {e.statement_id for e in dm.evidence.all()}
    for c in scored:
        ans = respond(c.text)
        text_low = (ans.text or "").casefold()
        status_ok = (not c.answer_status) or ans.status == c.answer_status
        contains_ok = all(s.casefold() in text_low for s in c.answer_contains)
        for f in ans.facts:
            if not f.statement_id:
                continue
            facts_total += 1
            if f.statement_id in evidence_ids:
                facts_with_evidence += 1
        if status_ok and contains_ok:
            correct += 1
        else:
            misses.append({"id": c.id, "text": c.text,
                           "expected_status": c.answer_status,
                           "got_status": ans.status,
                           "expected_contains": list(c.answer_contains),
                           "got_text": ans.text,
                           "status_ok": status_ok,
                           "contains_ok": contains_ok})
    unsupported = facts_total - facts_with_evidence
    return {
        "n": len(scored), "correct": correct,
        "accuracy": round(correct / len(scored), 4),
        "facts_in_answers": facts_total,
        "facts_with_evidence": facts_with_evidence,
        "evidence_coverage": round(facts_with_evidence / facts_total, 4)
        if facts_total else None,
        "unsupported_facts": unsupported,
        "unsupported_rate": round(unsupported / facts_total, 4)
        if facts_total else None,
        "misses": misses,
    }


# --------------------------------------------------------------------------
# combined
# --------------------------------------------------------------------------

def run_all(corpus: Corpus, dm, name: str = "", respond=None) -> dict:
    """All four families of numbers, plus the corpus identity they refer to."""
    return {
        "corpus": {"name": name or corpus.path, "size": len(corpus.cases),
                   "sha256": corpus.sha256, "schema": corpus.schema,
                   "summary": corpus.summary()},
        "intents": evaluate_intents(corpus.cases, dm),
        "entities": evaluate_entities(corpus.cases, dm),
        "answers": evaluate_answers(corpus.cases, dm, respond=respond),
    }


def render_report(report: dict) -> str:
    """Plain-text rendering: the same numbers, no decoration."""
    c = report["corpus"]
    lines = [f"МЕТРИКИ — {c['name']}",
             f"  корпус: {c['size']} реплик, sha256 {c['sha256'][:16]}…, "
             f"схема {c['schema']}"]
    i = report["intents"]
    if i.get("n"):
        lines.append(f"  речевой акт: {i['correct']}/{i['n']} = "
                     f"{i['accuracy']:.1%}, UNKNOWN {i['unknown_rate']:.1%}")
        if i["misses"]:
            lines.append(f"    промахи ({len(i['misses'])}):")
            for m in i["misses"][:10]:
                lines.append(f"      - {m['text']!r}: ждали {m['expected']}, "
                             f"получили {m['got']}")
    e = report["entities"]
    if e.get("n_cases"):
        lines.append(f"  сущности: P={e['precision']:.3f} R={e['recall']:.3f} "
                     f"F1={e['f1']:.3f} (tp={e['tp']} fp={e['fp']} fn={e['fn']})")
    a = report["answers"]
    if a.get("n"):
        lines.append(f"  ответы: {a['correct']}/{a['n']} = {a['accuracy']:.1%}")
        if a["unsupported_rate"] is not None:
            lines.append(f"  ответы без источника: {a['unsupported_rate']:.1%} "
                         f"({a['unsupported_facts']}/{a['facts_in_answers']})")
    return "\n".join(lines)
