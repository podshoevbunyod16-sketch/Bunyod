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
    if explicit_focus(meaning, dm):
        # an explicit correction can bind every pronoun of the turn by itself
        already = [f for f in meaning.focus if f.status == FOCUS_RESOLVED]
        if already:
            return resolutions
    for focus in meaning.focus:
        if focus.status != FOCUS_PRONOUN:
            continue
        if _pronoun_gender(focus.surface) == "p":
            # «они» — это группа.  Связка с одним из названных людей была бы
            # подменой: движок ответов не умеет говорить о группе, поэтому
            # честный ответ — уточнить, о ком речь.
            pool = _candidate_pool(dm, focus.surface)
            if len(pool) != 1:
                focus.status = FOCUS_AMBIGUOUS
                focus.candidates = [a.entity_id for a, _ in pool][:5]
                meaning.unknowns.append(
                    f"местоимение {focus.surface!r}: группа из "
                    f"{len(pool)} сущностей — нужен уточняющий вопрос")
                meaning.notes.append("plural pronoun refused: group reference")
                if not meaning.clarification:
                    meaning.clarification = ("О ком именно речь? Назови, "
                                             "пожалуйста, по имени.")
                continue
        res = dm.context.resolve_pronoun(focus.surface, dm.kg.resolver)
        resolutions.append(res)
        chosen = _preferred_candidate(dm, res) if res.status == PR_RESOLVED \
            else None
        fallback = None
        if chosen is None:
            fallback = _fallback_binding(dm, focus.surface)
            if fallback is not None:
                chosen = fallback
                meaning.notes.append(
                    f"pronoun {focus.surface!r}: frozen resolver gave "
                    f"{res.status!r} ({res.reason}); bound by salience pool")
        if chosen is not None:
            focus.entity_id = chosen
            focus.status = FOCUS_RESOLVED
            focus.confidence = (res.confidence if fallback is None
                                else round(min(0.9, 0.5 + 0.4 *
                                               _salience_of(dm, chosen)), 3))
            focus.candidates = list(res.candidates)
            name = dm.kg.name_of(chosen)
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
        else:
            # резолвер что-то предложил, но кандидат не прозвучал в этом
            # диалоге (или это класс): связка была бы догадкой
            focus.status = FOCUS_AMBIGUOUS
            meaning.unknowns.append(
                f"местоимение {focus.surface!r}: в диалоге нет подходящего "
                f"антецедента (резолвер: {res.status}, {res.reason})")
            if res.question and not meaning.clarification:
                meaning.clarification = res.question
            meaning.notes.append(f"pronoun {focus.surface!r} ungrounded -> refused")
    return resolutions
    for focus in meaning.focus:
        if focus.status != FOCUS_PRONOUN:
            continue
        if _pronoun_gender(focus.surface) == "p":
            # «они» — это группа.  Связка с одним из названных людей была бы
            # подменой: движок ответов не умеет говорить о группе, поэтому
            # честный ответ — уточнить, о ком речь.
            pool = _candidate_pool(dm, focus.surface)
            if len(pool) != 1:
                focus.status = FOCUS_AMBIGUOUS
                focus.candidates = [a.entity_id for a, _ in pool][:5]
                meaning.unknowns.append(
                    f"местоимение {focus.surface!r}: группа из "
                    f"{len(pool)} сущностей — нужен уточняющий вопрос")
                meaning.notes.append("plural pronoun refused: group reference")
                if not meaning.clarification:
                    meaning.clarification = ("О ком именно речь? Назови, "
                                             "пожалуйста, по имени.")
                continue
        res = dm.context.resolve_pronoun(focus.surface, dm.kg.resolver)
        resolutions.append(res)
        chosen = _preferred_candidate(dm, res) if res.status == PR_RESOLVED \
            else None
        fallback = None
        if chosen is None:
            fallback = _fallback_binding(dm, focus.surface)
            if fallback is not None:
                chosen = fallback
                meaning.notes.append(
                    f"pronoun {focus.surface!r}: frozen resolver gave "
                    f"{res.status!r} ({res.reason}); bound by salience pool")
        if chosen is not None:
            focus.entity_id = chosen
            focus.status = FOCUS_RESOLVED
            focus.confidence = (res.confidence if fallback is None
                                else round(min(0.9, 0.5 + 0.4 *
                                               _salience_of(dm, chosen)), 3))
            focus.candidates = list(res.candidates)
            name = dm.kg.name_of(chosen)
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
        else:
            # резолвер что-то предложил, но ни один кандидат не прозвучал в этом
            # диалоге (или все они — классы): это не связка, а догадка
            focus.status = FOCUS_AMBIGUOUS
            meaning.unknowns.append(
                f"местоимение {focus.surface!r}: в диалоге нет подходящего "
                f"антецедента (резолвер: {res.status}, {res.reason})")
            meaning.notes.append(f"pronoun {focus.surface!r} ungrounded -> refused")
    return resolutions


def apply_context(meaning: MeaningIR, dm) -> MeaningIR:
    """``bind_pronouns`` + a confidence cap when the binding was refused."""
    resolutions = bind_pronouns(meaning, dm)
    if any(r.status != PR_RESOLVED for r in resolutions):
        meaning.confidence = min(meaning.confidence, 0.5)
    return meaning


#: how many turns an explicit correction keeps priority over salience
EXPLICIT_FOCUS_TURNS = 3

#: kinds a pronoun must NOT refer to: a class («поэт») and a literal value
#: are not things «он/она» can denote.  Everything else (PERSON, PLACE, ANY,
#: country, organisation) is allowed — an allow-list was wrong because the
#: demo base stores untyped entities as ANY and they were all rejected.
PRONOUN_DENIED_KINDS = ("CLASS", "LITERAL")

#: pronouns -> required gender, for the gender check of an explicit focus
_PRONOUN_GENDER = {"он": "m", "его": "m", "ему": "m", "им": "m", "него": "m",
                   "она": "f", "её": "f", "ее": "f", "ей": "f", "ею": "f",
                   "неё": "f", "нее": "f", "они": "p", "оно": "n"}


def explicit_focus(meaning: MeaningIR, dm) -> bool:
    """Bind a pronoun to a focus the user set explicitly («нет, я про Анну»).

    A correction outranks salience for a few turns: the user has just said whom
    they mean, and a heuristic that prefers "the most mentioned entity" would
    ignore the only deliberate signal in the dialogue.  The binding is refused,
    not guessed, when the pronoun's gender contradicts the explicit focus.
    """
    focus_state = getattr(dm, "v17_explicit_focus", None) or {}
    entity_id = focus_state.get("entity_id")
    if not entity_id:
        return False
    turn_now = getattr(dm.context, "turn_no", 0)
    if turn_now - focus_state.get("turn", -99) > EXPLICIT_FOCUS_TURNS:
        return False
    if _superseded_by_newer_mention(dm, focus_state):
        return False
    entity = dm.kg.resolver.get(entity_id)
    name = dm.kg.name_of(entity_id)
    gender = (getattr(entity, "gender", "") or "")[:1] if entity else ""
    for focus in meaning.focus:
        if focus.status != FOCUS_PRONOUN:
            continue
        wanted = _PRONOUN_GENDER.get((focus.surface or "").strip().casefold(), "")
        if wanted and gender and wanted != gender and wanted != "p" and gender != "p":
            meaning.unknowns.append(
                f"местоимение {focus.surface!r} не согласуется по роду с "
                f"исправлением {name!r}")
            meaning.notes.append(f"explicit focus rejected by gender "
                                 f"({focus.surface!r} vs {name!r})")
            continue
        focus.entity_id = entity_id
        focus.status = FOCUS_RESOLVED
        focus.confidence = 0.95
        focus.note = f"explicit correction focus -> {name!r}"
        meaning.notes.append(focus.note)
        meaning.unknowns = [u for u in meaning.unknowns
                            if focus.surface not in u]
    return True


def set_explicit_focus(dm, entity_id: int) -> None:
    """Stored on the manager, not in ``context.focus``.

    ``ContextEngine.focus`` is rewritten by every turn, so an explicit correction
    kept there disappeared on the next question.  The marker lives in v0.17 and
    expires when somebody *newer* is mentioned — a mention beats a correction.
    """
    note_mention(dm, entity_id)
    dm.v17_explicit_focus = {"entity_id": entity_id, "role": "correction",
                             "turn": getattr(dm.context, "turn_no", 0)}


def clear_explicit_focus(dm) -> None:
    dm.v17_explicit_focus = {}


#: minimum salience for an entity to be considered as a pronoun antecedent
MIN_SALIENCE = 0.15
#: the top candidate must be at least this much more salient than the second
DOMINANCE_RATIO = 1.3


def _pronoun_gender(surface: str) -> str:
    return _PRONOUN_GENDER.get((surface or "").strip().casefold(), "")


def _agrees(gender: str, wanted: str) -> bool:
    if not wanted or wanted in ("p", "n"):
        return True
    if not gender:
        return True                     # род неизвестен — не повод отказать
    return gender[:1] == wanted[:1]


def _candidate_pool(dm, surface: str) -> list:
    """Active entities a pronoun of this form could denote, most salient first.

    Used when the frozen resolver's top candidate is a class (or when it
    refuses): salience and gender agreement are the same signals it uses, minus
    the classes, which «он» cannot denote.
    """
    wanted = _pronoun_gender(surface)
    said = mentioned_in_dialogue(dm)
    pool = []
    for active in dm.context.salience_ranked():
        if active.salience < MIN_SALIENCE:
            continue
        if active.entity_id not in said:
            continue
        if not pronoun_kind_allowed(dm, active.entity_id):
            continue
        entity = dm.kg.resolver.get(active.entity_id)
        gender = (getattr(entity, "gender", "") or "")[:1] if entity else ""
        if not _agrees(gender, wanted):
            continue
        pool.append((active, gender))
    return pool


def mentioned_in_dialogue(dm) -> set[int]:
    """Entities the current dialogue has actually brought up.

    A pronoun must refer to something *said in this conversation*, not to
    whatever the knowledge base happens to have left activated: the demo base
    pre-activates background entities («Англия», «Ньютон»), and «она» after a
    single mention of «Тютчев» must not quietly become «Англия».
    """
    said = set(getattr(dm, "v17_mentioned", ()) or ())
    memory = getattr(dm, "v17_memory", None)
    if memory is not None:
        said |= memory.ids()
    for turn in getattr(dm.context, "history", ()) or ():
        for entity_id in getattr(turn, "entity_ids", ()) or ():
            if entity_id:
                said.add(int(entity_id))
    return said


def note_mention(dm, entity_id: int, surface: str = "") -> None:
    if not entity_id:
        return
    mentioned = set(getattr(dm, "v17_mentioned", ()) or ())
    mentioned.add(int(entity_id))
    dm.v17_mentioned = mentioned
    memory = getattr(dm, "v17_memory", None)
    if memory is not None:
        memory.note(entity_id, surface,
                    turn=int(getattr(dm.context, "turn_no", 0)))


def _salience_of(dm, entity_id: int) -> float:
    active = dm.context.active.get(entity_id)
    return float(getattr(active, "salience", 0.0) or 0.0)


def _fallback_binding(dm, surface: str) -> int | None:
    """Bind from the salience pool, or refuse when the pool does not decide.

    Rules, stated so they can be argued with:

    * plural pronouns denote a group: two eligible entities refuse;
    * one eligible entity is enough;
    * several are accepted only if the most salient dominates the second by
      ``DOMINANCE_RATIO`` — otherwise the dialogue does not determine a referent
      and a guess would be an unjustified binding.
    """
    pool = _candidate_pool(dm, surface)
    if not pool:
        return None
    if _pronoun_gender(surface) in ("p", "n"):
        return None if len(pool) > 1 else pool[0][0].entity_id
    if len(pool) == 1:
        return pool[0][0].entity_id
    top, second = pool[0][0].salience, pool[1][0].salience
    if second <= 0 or top / second >= DOMINANCE_RATIO:
        return pool[0][0].entity_id
    return None


def _preferred_candidate(dm, resolution) -> int | None:
    """Most salient candidate that is an entity rather than a class/literal."""
    candidates = list(getattr(resolution, "candidates", []) or [])
    if resolution.entity_id is not None:
        candidates.insert(0, resolution.entity_id)
    said = mentioned_in_dialogue(dm)
    seen: set[int] = set()
    for entity_id in candidates:
        if entity_id not in said:
            continue
        if entity_id in seen:
            continue
        seen.add(entity_id)
        if pronoun_kind_allowed(dm, entity_id):
            return entity_id
    return None


def _superseded_by_newer_mention(dm, state: dict) -> bool:
    since = state.get("turn", -1)
    target = state.get("entity_id")
    for turn in getattr(dm.context, "history", []):
        if getattr(turn, "turn_no", -1) <= since:
            continue
        if any(eid != target for eid in (getattr(turn, "entity_ids", ()) or ())):
            return True
    return False


def pronoun_kind_allowed(dm, entity_id: int) -> bool:
    """Whether an entity may be the referent of «он/она» (a class may not)."""
    entity = dm.kg.resolver.get(entity_id) if entity_id else None
    if entity is None:
        return False
    kind = (getattr(entity, "kind", "") or "").upper()
    return kind not in PRONOUN_DENIED_KINDS


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
    for entity_id in entity_ids:
        note_mention(dm, entity_id)
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
