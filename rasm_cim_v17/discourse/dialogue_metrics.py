"""Coreference accuracy — measured on whole dialogues, not on single turns.

Definition (stated so it can be recomputed):

    correct ref   the pronoun's focus resolved to the expected entity
                  (compared by canonical name, not by internal id)
    refusal       the system said "I do not know / which one?" instead of
                  binding — counted separately, because a refusal is honest and
                  a wrong binding is not
    answer hit    the answer at that turn has the expected status and contains
                  every expected substring

Each dialogue runs on a fresh database built by ``build_demo`` in the caller,
because coreference depends on what was said earlier in *that* dialogue.
"""
from __future__ import annotations

from ..semantics.meaning import FOCUS_AMBIGUOUS, FOCUS_PRONOUN
from .dialogue_corpus import DialogueCorpus, DialogueCase


def _run_dialogue(case: DialogueCase, dm, respond) -> dict:
    """Execute a dialogue; return per-turn observations."""
    observations = []
    for i, text in enumerate(case.turns):
        answer = respond(text)
        pipe = getattr(respond, "__self__", None)
        meaning = getattr(pipe, "last_meaning", None) if pipe is not None else None
        resolved = []
        unresolved_pronouns = []
        if meaning is not None:
            for f in meaning.focus:
                if f.entity_id is not None:
                    resolved.append(dm.kg.name_of(f.entity_id))
                elif f.status in (FOCUS_PRONOUN, FOCUS_AMBIGUOUS):
                    unresolved_pronouns.append(f.surface)
        observations.append({"turn": i, "text": text, "answer": answer,
                             "resolved": resolved,
                             "unresolved_pronouns": unresolved_pronouns,
                             "status": answer.status,
                             "answer_text": answer.text})
    return {"case": case, "observations": observations}


def evaluate_coreference(corpus: DialogueCorpus, make_dm, respond_factory) -> dict:
    """``make_dm()`` -> fresh manager, ``respond_factory(dm)`` -> responder."""
    total = correct = refusals = wrong = 0
    answer_checks = answer_hits = 0
    misses: list[dict] = []
    for case in corpus.cases:
        dm = make_dm()
        respond = respond_factory(dm)
        run = _run_dialogue(case, dm, respond)
        by_turn = {o["turn"]: o for o in run["observations"]}
        for ref in case.refers_to:
            total += 1
            obs = by_turn[ref.turn]
            if ref.entity in obs["resolved"]:
                correct += 1
            elif obs["unresolved_pronouns"]:
                refusals += 1
                misses.append({"id": case.id, "turn": ref.turn, "kind": "refusal",
                               "pronoun": ref.pronoun, "expected": ref.entity,
                               "text": obs["text"],
                               "status": obs["status"]})
            else:
                wrong += 1
                misses.append({"id": case.id, "turn": ref.turn,
                               "kind": "wrong_binding",
                               "pronoun": ref.pronoun, "expected": ref.entity,
                               "got": obs["resolved"], "text": obs["text"]})
        for turn, expected_status in case.expect_status.items():
            obs = by_turn.get(turn)
            if obs is None:
                continue
            answer_checks += 1
            wanted = case.expect_contains.get(turn, [])
            text_low = (obs["answer_text"] or "").casefold()
            if obs["status"] == expected_status and all(
                    s.casefold() in text_low for s in wanted):
                answer_hits += 1
            else:
                misses.append({"id": case.id, "turn": turn, "kind": "answer",
                               "expected_status": expected_status,
                               "got_status": obs["status"],
                               "expected_contains": wanted,
                               "got_text": obs["answer_text"]})
    return {
        "dialogues": len(corpus.cases),
        "turns": sum(len(c.turns) for c in corpus.cases),
        "references": total,
        "correct": correct,
        "refused": refusals,
        "wrong": wrong,
        "accuracy": round(correct / total, 4) if total else None,
        "refusal_rate": round(refusals / total, 4) if total else None,
        "wrong_rate": round(wrong / total, 4) if total else None,
        "answer_checks": answer_checks,
        "answer_hits": answer_hits,
        "answer_accuracy": round(answer_hits / answer_checks, 4)
        if answer_checks else None,
        "sha256": corpus.sha256,
        "misses": misses,
    }


def render_coreference(report: dict) -> str:
    if not report.get("references"):
        return "КОРЕФЕРЕНЦИЯ: в корпусе нет местоименных отсылок"
    lines = [f"КОРЕФЕРЕНЦИЯ — {report['dialogues']} диалогов, "
             f"{report['turns']} реплик, sha256 {report['sha256'][:16]}…",
             f"  отсылок: {report['references']}, верно: {report['correct']} "
             f"= {report['accuracy']:.1%}, отказов: {report['refused']} "
             f"({report['refusal_rate']:.1%}), ошибочных связок: "
             f"{report['wrong']}"]
    if report.get("answer_checks"):
        lines.append(f"  ответы в диалогах: {report['answer_hits']}"
                     f"/{report['answer_checks']} = "
                     f"{report['answer_accuracy']:.1%}")
    for m in report["misses"][:10]:
        lines.append(f"    - {m}")
    return "\n".join(lines)
