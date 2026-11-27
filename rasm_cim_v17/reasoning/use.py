"""Putting v0.16's inference to work where its answer path does not.

Two measured gaps in the frozen layer motivated this module, and only these:

1. **Inverse reading.**  «Евгений Онегин написан Пушкиным?» resolves to
   ``(Онегин, wrote, Пушкин)`` — the stored statement is
   ``(Пушкин, wrote, Онегин)``.  v0.16 answers "не знаю", although the engine's
   own ``inverse`` rule can prove it.  v0.17 tries the reversed direction and
   quotes the proof.
2. **Orientation of a wh-question.**  «Что находится в Европе?» asks for the
   *subject*; if the parser hands the known entity over in the other role, the
   query has to be issued both ways.

Everything returned here is built from ``Fact`` objects that the engine produced
under the policy's bounds; a fact without a proof is not used.
"""
from __future__ import annotations

from rasm_cim_v16.answer_engine import A_DIRECT, A_INFERRED, A_UNKNOWN, Answer
from rasm_cim_v16.inference_engine import ST_DIRECT

from ..semantics.meaning import AS_ASK_WH, AS_ASK_YESNO, MeaningIR
from .policy import (DEFAULT_POLICY, ReasoningPolicy, explanation_lines,
                     fact_within_policy)

#: orientation attempts per predicate — both directions, nothing else
MAX_ORIENTATIONS = 2


def _resolved_pair(meaning: MeaningIR):
    resolved = [f for f in meaning.focus if f.entity_id is not None]
    if len(resolved) != 2:
        return None
    return resolved[0], resolved[1]


def _answer_from_facts(dm, meaning: MeaningIR, facts, mode: str,
                       why: str) -> Answer | None:
    if not facts:
        return None
    fact = facts[0]
    allowed, reason = fact_within_policy(fact, DEFAULT_POLICY)
    if not allowed:
        return None
    text = dm.answers.phrase(fact)
    lines = explanation_lines(dm, fact)
    body = f"{text}."
    if fact.status != ST_DIRECT:
        body += "\n" + why + "\n" + "\n".join(f"  {l}" for l in lines)
    return Answer(text=body,
                  status=A_DIRECT if fact.status == ST_DIRECT else A_INFERRED,
                  confidence=fact.confidence, mode=mode, facts=list(facts),
                  evidence=dm.answers._ev_for(list(facts)),  # noqa: SLF001
                  explanation=lines,
                  short=text, normal=body, detailed=body,
                  notes=list(meaning.notes) + ["reasoning:inference"])


def answer_with_inference(dm, meaning: MeaningIR, mode: str = "normal",
                          policy: ReasoningPolicy = DEFAULT_POLICY
                          ) -> Answer | None:
    """Try to answer from the inference engine; ``None`` when it cannot."""
    if meaning.speech_act == AS_ASK_YESNO:
        pair = _resolved_pair(meaning)
        if pair is None:
            return None
        a, b = pair
        predicates = [meaning.relation] if meaning.relation else []
        if not predicates:
            return None
        for subject, obj in ((a, b), (b, a)):
            verdict, fact = dm.inference.yes_no(subject.entity_id,
                                                predicates[0],
                                                obj.entity_id)
            if verdict == "YES" and fact is not None:
                reversed_reading = (subject is a) != (obj is b) or \
                    (a.role == "object")
                why = ("Основание (обратное прочтение отношения):"
                       if reversed_reading else "Основание (вывод):")
                return _answer_from_facts(dm, meaning, [fact], mode, why)
        return None

    if meaning.speech_act == AS_ASK_WH:
        resolved = [f for f in meaning.focus if f.entity_id is not None]
        if len(resolved) != 1 or not meaning.relation:
            return None
        ent = resolved[0]
        predicate = meaning.relation
        if meaning.asks == "subject":
            facts = dm.inference.query(predicate=predicate,
                                       obj=ent.entity_id)
        else:
            facts = dm.inference.query(subject=ent.entity_id,
                                       predicate=predicate)
        if not facts:
            # the parser may have handed the entity over in the other role
            other = (dm.inference.query(subject=ent.entity_id,
                                        predicate=predicate)
                     if meaning.asks == "subject"
                     else dm.inference.query(predicate=predicate,
                                             obj=ent.entity_id))
            facts = other
        if not facts:
            return None
        admissible = [f for f in facts
                      if fact_within_policy(f, policy)[0]]
        if not admissible:
            return None
        return _answer_from_facts(dm, meaning, admissible, mode,
                                  "Основание (вывод):")
    return None
