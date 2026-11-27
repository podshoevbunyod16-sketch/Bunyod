"""Forgetting: what survived learning, what appeared, what broke, and can it be undone.

Definitions used everywhere in the docs:

    retention      доля утверждений, известных до обучения, которые известны после
    forgetting     1 - retention
    new facts      утверждений, которых до обучения не было
    contradictions conflicts, которые детектор v0.16 находит после обучения
    answer drift   доля контрольных вопросов, чей статус или ответ изменился
    rollback       состояние до обучения восстанавливается побайтово из
                   `kg.to_dict()` и сравнивается со снимком

These are the numbers the roadmap asks for, and each of them can be *bad* — the
report shows the bad ones instead of averaging them away.
"""
from __future__ import annotations

import hashlib
import json

from .curriculum import Curriculum
from .session import apply_lessons

#: fixed guard questions: their answers must not change because of new lessons
GUARD_QUESTIONS = (
    "Кто написал Евгений Онегин?",
    "Где находится Париж?",
    "Кто президент Марса?",
)


def _norm(text: str) -> str:
    return (text or "").strip().rstrip(".!?").casefold()


def _statement_key(st) -> str:
    return f"{st.subject}|{st.predicate}|{st.object}|{st.polarity}"


def snapshot(dm) -> dict:
    statements = sorted(_statement_key(st) for st in dm.kg.all_statements())
    evidence = sorted((e.statement_id, e.kind, e.utterance)
                      for e in dm.evidence.all())
    payload = json.dumps({"statements": statements, "evidence": evidence},
                         ensure_ascii=False, sort_keys=True)
    return {"statements": statements, "evidence": evidence,
            "checksum": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
            "statement_count": len(statements),
            "entity_count": dm.kg.stats().get("entities", 0),
            "evidence_count": len(evidence)}


def compare_states(before: dict, after: dict) -> dict:
    old = set(before["statements"])
    new = set(after["statements"])
    retained = old & new
    rate = round(len(retained) / len(old), 4) if old else None
    return {
        "before": before["statement_count"],
        "after": after["statement_count"],
        "retained": len(retained),
        "missing": sorted(old - new),
        "new": sorted(new - old),
        "retention": rate,
        "forgetting": None if rate is None else round(1 - rate, 4),
        "identical": before["statements"] == after["statements"],
    }


def _scan_conflicts(dm) -> tuple[list[dict], str]:
    """Full conflict inventory.

    ``ContradictionDetector.scan`` is idempotent — it returns only conflicts it
    has not reported yet — so an inventory needs a fresh detector on the same
    graph.  Using ``dm.detector`` here would silently report "no conflicts"
    after the first scan (measured: it reported 1, then 0).
    """
    try:
        from rasm_cim_v16.contradiction_detector import ContradictionDetector
        detector = ContradictionDetector(dm.kg, getattr(dm, "clock", None))
        return [c.as_dict() for c in detector.scan()], ""
    except Exception as exc:            # детектор не должен ломать отчёт
        return [], str(exc)


def _guard_answers(dm, questions) -> dict[str, tuple[str, str]]:
    from ..pipeline import SemanticPipeline
    pipeline = SemanticPipeline(dm)
    out = {}
    for question in questions:
        answer = pipeline.respond(question)
        out[question] = (answer.status, (answer.short or answer.text or "")[:160])
    return out


def rollback_check(make_dm, prepare) -> dict:
    """Restore the pre-learning state into a **fresh** manager and compare.

    ``KnowledgeGraph.load_dict`` is additive in v0.16: loading a snapshot into a
    graph that already has statements merges the two.  Rolling back therefore
    means "reload the snapshot into a clean instance", and the comparison below
    checks exactly that (a limitation, documented rather than papered over).
    """
    original = make_dm()
    before = snapshot(original)
    raw = json.dumps(original.kg.to_dict(), ensure_ascii=False, sort_keys=True)

    learned = make_dm()
    prepare(learned)
    during = snapshot(learned)

    restored_dm = make_dm()
    restored_dm.kg.load_dict(json.loads(raw))
    restored = snapshot(restored_dm)
    return {"before": before["checksum"], "during": during["checksum"],
            "restored": restored["checksum"],
            "learning_changed": before["checksum"] != during["checksum"],
            "rollback_exact": restored["statements"] == before["statements"],
            "restored_evidence_equal": restored["evidence"] == before["evidence"]}


def apply_curriculum(dm, curriculum: Curriculum, guards=GUARD_QUESTIONS) -> dict:
    """Apply every lesson, then measure retention, drift and contradictions."""
    guards = tuple(guards)
    before = snapshot(dm)
    guard_before = _guard_answers(dm, guards)
    conflicts_before, conflicts_before_error = _scan_conflicts(dm)
    results = apply_lessons(dm, curriculum)
    after = snapshot(dm)
    guard_after = _guard_answers(dm, guards)
    improved, degraded, changed = [], [], []
    for q in guards:
        (status_b, text_b), (status_a, text_a) = guard_before[q], guard_after[q]
        if (status_b, text_b) == (status_a, text_a):
            continue
        was_unknown = status_b in ("UNKNOWN", "AMBIGUOUS")
        now_unknown = status_a in ("UNKNOWN", "AMBIGUOUS")
        if was_unknown and not now_unknown:
            improved.append(q)
        elif now_unknown or (_norm(text_b) and _norm(text_b) not in _norm(text_a)):
            degraded.append(q)               # ответ потерял то, что знал
        else:
            changed.append(q)                # ответ расширился или уточнился

    conflicts_after, conflicts_error = _scan_conflicts(dm)
    def _ck(c):
        return (c.get("kind"), c.get("subject"), c.get("predicate"),
                tuple(c.get("objects") or ()))

    known = {_ck(c) for c in conflicts_before}
    new_conflicts = [c for c in conflicts_after if _ck(c) not in known]

    state = compare_states(before, after)
    failed = [r.as_dict() for r in results if not r.ok]
    return {
        "curriculum": curriculum.summary(),
        "lessons": [r.as_dict() for r in results],
        "lessons_total": len(results),
        "lessons_ok": len(results) - len(failed),
        "lesson_failures": failed,
        "retention": state["retention"],
        "forgetting": state["forgetting"],
        "new_facts": len(state["new"]),
        "missing_facts": state["missing"],
        "identical_statements": state["identical"],
        "conflicts_before": len(conflicts_before),
        "conflicts_after": len(conflicts_after),
        "new_conflicts": new_conflicts,
        "conflicts_error": conflicts_error,
        "guards": [{"text": q, "before": guard_before[q], "after": guard_after[q],
                    "verdict": ("improved" if q in improved else
                                "degraded" if q in degraded else "stable")}
                   for q in guards],
        "improved_questions": improved,
        "degraded_questions": degraded,
        "changed_questions": changed,
        #: дрейф — только ухудшение; расширение ответа считается изменением
        "answer_drift": round(len(degraded) / len(guards), 4) if guards else None,
        "answer_change": round(len(changed) / len(guards), 4) if guards else None,
        "checksums": {"before": before["checksum"], "after": after["checksum"]},
    }


def evaluate_learning(curriculum: Curriculum, make_dm, guards=GUARD_QUESTIONS
                      ) -> dict:
    """Fresh manager in, report out — plus a rollback check on the same base."""
    dm = make_dm()
    report = apply_curriculum(dm, curriculum, guards)

    def prepare(manager):
        apply_lessons(manager, curriculum)

    report["rollback"] = rollback_check(make_dm, prepare)
    return report


def render_learning(report: dict) -> str:
    lines = ["Обучение без забывания:"]
    cur = report["curriculum"]
    lines.append(f"  учебный план: {cur['path']} ({cur['size']} уроков, "
                 f"sha256 {cur['sha256'][:16]}…)")
    lines.append(f"  уроки выполнены: {report['lessons_ok']}/"
                 f"{report['lessons_total']}")
    for failure in report["lesson_failures"][:10]:
        lines.append(f"    - {failure['id']}: {failure['reason']}")
    retention = report["retention"]
    lines.append(f"  удержание старых фактов: "
                 f"{'—' if retention is None else f'{retention:.1%}'} "
                 f"(потеряно {len(report['missing_facts'])})")
    lines.append(f"  новых фактов: {report['new_facts']}; "
                 f"противоречий: было {report['conflicts_before']}, "
                 f"стало {report['conflicts_after']}, "
                 f"добавлено обучением {len(report['new_conflicts'])}")
    drift = report["answer_drift"]
    drift_text = "—" if drift is None else f"{drift:.1%}"
    lines.append(f"  контрольные ответы: улучшилось "
                 f"{len(report['improved_questions'])}, изменилось "
                 f"{len(report['changed_questions'])}, ухудшилось "
                 f"{len(report['degraded_questions'])}; дрейф {drift_text}")
    rb = report.get("rollback")
    if rb:
        lines.append("  откат: " + ("точный" if rb["rollback_exact"] else "НЕ ТОЧНЫЙ")
                     + (" (обучение изменило базу)" if rb["learning_changed"]
                        else " (обучение не изменило базу)"))
    return "\n".join(lines)
