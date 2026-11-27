"""Binding MeaningIR to the v0.16 context: pronouns, and keeping it warm.

``apply_context`` mutates a ``MeaningIR`` in place (it is a per-turn object) and
returns it, because the decision "this is what «он» refers to" belongs to the
meaning of the turn, not to the answer.  Every change appends a note, so an
answer can always be traced back to the resolution that produced it.
"""
from __future__ import annotations

from rasm_cim_v16.context_engine import (PR_AMBIGUOUS, PR_RESOLVED,
                                        PR_UNRESOLVED, PronounResolution)

from ..semantics.meaning import (FOCUS_AMBIGUOUS, FOCUS_PRONOUN, FOCUS_RESOLVED,
                                 MeaningIR)


def bind_pronouns(meaning: MeaningIR, dm) -> list[PronounResolution]:
    """Resolve every pronoun in ``meaning.focus`` using the v0.16 context.

    Returns the raw resolutions so callers (and tests) can inspect refusals.
    """
    resolutions: list[PronounResolution] = []
    for focus in meaning.focus:
        if focus.status != FOCUS_PRONOUN:
            continue
        res = dm.context.resolve_pronoun(focus.surface, dm.kg.resolver)
        resolutions.append(res)
        if res.status == PR_RESOLVED and res.entity_id is not None:
            focus.entity_id = res.entity_id
            focus.status = FOCUS_RESOLVED
            focus.confidence = res.confidence
            focus.candidates = list(res.candidates)
            name = dm.kg.name_of(res.entity_id)
            focus.note = (f"pronoun {focus.surface!r} -> {name!r} "
                          f"({res.reason})")
            meaning.notes.append(focus.note)
            meaning.unknowns = [u for u in meaning.unknowns
                                if focus.surface not in u]
        elif res.status == PR_AMBIGUOUS:
            focus.status = FOCUS_AMBIGUOUS
            focus.candidates = list(res.candidates)
            meaning.unknowns.append(
                f"местоимение {focus.surface!r}: {res.question} "
                f"({res.reason})")
            meaning.notes.append(f"pronoun {focus.surface!r} ambiguous")
        else:  # PR_UNRESOLVED / anything else: stay explicit
            meaning.unknowns.append(
                f"местоимение {focus.surface!r} не относится ни к одной "
                f"сущности контекста ({res.reason})")
            if res.question:
                meaning.clarification = res.question
            meaning.notes.append(f"pronoun {focus.surface!r} unresolved")
    return resolutions


def apply_context(meaning: MeaningIR, dm) -> MeaningIR:
    """``bind_pronouns`` + a confidence cap when the binding was refused."""
    resolutions = bind_pronouns(meaning, dm)
    if any(r.status != PR_RESOLVED for r in resolutions):
        meaning.confidence = min(meaning.confidence, 0.5)
    return meaning


def resolved_referents(meaning: MeaningIR) -> list[int]:
    return [f.entity_id for f in meaning.focus
            if f.status == FOCUS_RESOLVED and f.entity_id is not None]


def record_turn(dm, text: str, meaning: MeaningIR, answer) -> None:
    """Do the context bookkeeping ``DialogueManager.say`` would have done.

    Only for turns the v0.17 pipeline executed itself.  Without this the
    salience state goes stale after the first repaired utterance, and the next
    «он» cannot be resolved — a bug that would look like a coreference failure.
    """
    clock = getattr(dm, "_clock", lambda: "")()
    entity_ids = resolved_referents(meaning)
    statement_ids = [f.statement_id for f in getattr(answer, "facts", [])
                     if f.statement_id][:8]
    dm.context.record_user_turn(text, intent=meaning.speech_act,
                                entity_ids=entity_ids[:8], when=clock)
    dm.context.record_system_turn(getattr(answer, "text", ""),
                                  intent=getattr(answer, "status", ""),
                                  entity_ids=entity_ids[:8],
                                  statement_ids=statement_ids,
                                  answer_status=getattr(answer, "status", ""),
                                  when=clock)
    try:
        from rasm_cim_v16.memory_manager import TIER_WORKING
        dm.memory.record(kind="answer", tier=TIER_WORKING,
                         payload={"status": getattr(answer, "status", ""),
                                  "text": getattr(answer, "text", "")[:400],
                                  "semantic": True},
                         source="semantic-pipeline",
                         confidence=float(getattr(answer, "confidence", 0.0)),
                         when=clock)
    except (AttributeError, TypeError, ValueError):
        # memory bookkeeping is a convenience, not a correctness requirement:
        # an internal API change must not turn a good answer into a crash.
        pass
