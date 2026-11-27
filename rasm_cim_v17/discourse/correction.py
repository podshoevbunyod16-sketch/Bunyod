"""User corrections: «Нет, я про Анну», «не Пётр, а Иван», «точнее, Мария».

A correction is a *speech act about the conversation*, not a question about the
world: it says "the topic you should be holding is this entity".  The frozen
core has no such act, so v0.17 detects it, resolves the named entity (with
accusative recovery from ``semantics/morphology``) and moves the conversational
focus there — then answers with a one-line acknowledgement.

Patterns are Russian constructions, not test phrases: the marker set is
documented below and the entity is whatever the user named.  If the named
entity is not in the base, the correction is refused instead of guessed.
"""
from __future__ import annotations

import re

from ..semantics.morphology import resolve_form

#: markers that start a correction, and how the corrected entity is introduced
_MARKERS = (
    "^\\s*нет\\s*,?\\s*",
    "^\\s*не\\s+так\\s*,?\\s*",
    "^\\s*точнее\\s*,?\\s*",
    "^\\s*вернее\\s*,?\\s*",
    "\\bя\\s+говор(?:ил|ила)\\s+про\\b",
    "\\bя\\s+спрашивал[а]?\\s+про\\b",
    "\\bя\\s+имел[аи]?\\s+в\\s+виду\\b",
    "\\bречь\\s+(?:идёт\\s+)?о\\b",
)
#: "я про X", "я имел в виду X", "речь о X", "не A, а B", "точнее, X"
_TARGETS = (
    re.compile("\\bя\\s+про\\s+(?P<target>[^\\s,.!?]+)", re.IGNORECASE),
    re.compile("\\bя\\s+имел[аи]?\\s+в\\s+виду\\s+(?P<target>[^\\s,.!?]+)", re.IGNORECASE),
    re.compile("\\bя\\s+говор(?:ил|ила)\\s+про\\s+(?P<target>[^\\s,.!?]+)", re.IGNORECASE),
    re.compile("\\bя\\s+спрашивал[а]?\\s+про\\s+(?P<target>[^\\s,.!?]+)", re.IGNORECASE),
    re.compile("\\bречь\\s+(?:идёт\\s+)?\\s*о\\s+(?P<target>[^\\s,.!?]+)", re.IGNORECASE),
    re.compile("\\bне\\s+[^\\s,.!?]+,?\\s+а\\s+(?P<target>[^\\s,.!?]+)", re.IGNORECASE),
    re.compile("(?:точнее|вернее)\\s*,?\\s+(?P<target>[А-ЯЁA-Z][^,.!?]*)", re.IGNORECASE),
)
_HAS_MARKER = re.compile("|".join(_MARKERS), re.IGNORECASE)

#: answers are acknowledgements of the conversation, not claims about the world
#: кавычки — чтобы не согласовывать падеж с именем вnominative
ACK_TEMPLATE = "Понял: речь о «{name}»."


def find_target(text: str) -> str:
    """The corrected mention, or ``""`` when the utterance is not a correction."""
    text = text or ""
    for pattern in _TARGETS:
        match = pattern.search(text)
        if match:
            return match.group("target").strip()
    return ""


def detect(text: str) -> str:
    if not _HAS_MARKER.search(text or ""):
        return ""
    return find_target(text)


def apply_correction(dm, text: str) -> dict | None:
    """Resolve the corrected entity and move the conversational focus to it."""
    target = detect(text)
    if not target:
        return None
    entity_id, matched = resolve_form(target, dm.kg.resolver)
    if entity_id is None:
        return None
    from .binding import set_explicit_focus
    dm.context.touch(entity_id, role="correction")
    set_explicit_focus(dm, entity_id)
    return {"entity_id": entity_id, "surface": matched,
            "name": dm.kg.name_of(entity_id)}

#: introducers of an explicit topic switch ("а теперь про Куприна")
_SWITCH = re.compile(
    r"^\s*(?:а\s+)?(?:теперь|давай|давайте|поговорим|переключимся|"
    r"верн[её]мся)\b[^,.!?]*?\b(?:про|об|о)\s+(?P<target>[^\s,.!?]+)",
    re.IGNORECASE)


def detect_topic_switch(text: str) -> str:
    """The entity the user explicitly switched to, or ``""``.

    A deliberate topic change is the same signal as a correction: the user names
    the thing the next pronoun will be about.  Without it «А теперь про Куприна.
    Он писатель?» leaves the old topic in focus.
    """
    match = _SWITCH.search(text or "")
    return match.group("target").strip() if match else ""


def apply_topic_switch(dm, text: str) -> dict | None:
    target = detect_topic_switch(text)
    if not target:
        return None
    from .binding import set_explicit_focus
    from ..semantics.morphology import resolve_form

    entity_id, matched = resolve_form(target, dm.kg.resolver)
    if entity_id is None:
        return None
    dm.context.touch(entity_id, role="topic")
    set_explicit_focus(dm, entity_id)
    return {"entity_id": entity_id, "surface": matched,
            "name": dm.kg.name_of(entity_id)}
