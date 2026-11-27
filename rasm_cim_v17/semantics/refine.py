"""Speech-act refinement: the small set of cases where the frozen parser's
intent is provably wrong, decided by explicit rules with recorded evidence.

Why this exists at all.  ``rasm_cim_v16.language_parser`` classifies
"Перечисли, что открыл Ньютон" as ``TEACH`` because it sees an imperative verb
and a subject/object pair.  That is a real defect of the frozen layer, and
v0.17 is not allowed to fix it by editing v0.16.  It is fixed here instead —
in front of the parser's answer, with a rule that names its evidence.

Rules are deliberately conservative:

* an explicit teach marker ("запомни", "научи", "это факт") always wins — the
  refiner never overrides an explicit act of teaching;
* a rule fires only on positive evidence (a question word or an interrogative
  verb) AND the absence of a teaching marker;
* every refinement appends a note to ``MeaningIR.notes`` saying which rule
  fired, so an answer can always be traced back to the decision.
"""
from __future__ import annotations

import re

from rasm_cim_v16.language_parser import (INT_ASK_DEFINITION, INT_ASK_WH,
                                          INT_ASK_YESNO, INT_RETRACT, INT_TEACH,
                                          INT_UNKNOWN)

#: an imperative that asks for a listing, not for teaching
LIST_REQUEST_VERBS = ("перечисли", "перечислите", "список", "назови",
                      "назовите", "перечисли-ка")
#: an imperative that asks for an explanation ("расскажи про X")
TELL_ME_VERBS = ("расскажи", "расскажите", "объясни", "объясните",
                 "поясни", "опиши", "опишите")
#: explicit teaching markers — these veto any refinement
TEACH_MARKERS = ("запомни", "запомните", "научи", "научите", "это факт",
                 "сохрани", "запиши факт", "выучи")

_QUESTION = re.compile(r"\?\s*$")
_WORD = re.compile(r"[А-Яа-яЁёA-Za-z\-]+")


def has_teach_marker(text: str) -> bool:
    low = (text or "").casefold()
    return any(marker in low for marker in TEACH_MARKERS)


def _starts_with(text: str, verbs) -> bool:
    words = _WORD.findall((text or "").casefold())
    return bool(words) and any(words[0] == v or words[0].startswith(v)
                               for v in verbs)


def refine_speech_act(intent: str, text: str,
                      explicit_teach: bool = False) -> tuple[str, list[str]]:
    """Return ``(speech_act, notes)`` for one utterance.

    ``intent`` is the parser's verdict; ``text`` is the utterance the parser
    kept (it strips the teaching verb, so ``explicit_teach`` — the parser's own
    flag that a teach verb was present — must also veto refinement).
    """
    notes: list[str] = []
    act = intent or INT_UNKNOWN
    if explicit_teach or has_teach_marker(text):
        return act, notes                      # explicit teaching is sacred

    if act in (INT_TEACH, "TELL", INT_UNKNOWN):
        if _starts_with(text, LIST_REQUEST_VERBS):
            notes.append("refine: imperative listing -> ASK_LIST")
            return "ASK_LIST", notes
        if _starts_with(text, TELL_ME_VERBS):
            notes.append("refine: imperative 'tell me about' -> ASK_DEFINITION")
            return INT_ASK_DEFINITION, notes
        if _QUESTION.search(text or "") and act == INT_UNKNOWN:
            notes.append("refine: question mark with unknown intent -> ASK_WH "
                         "(unspecified slot)")
            return INT_ASK_WH, notes
    if act == INT_ASK_WH and _starts_with(text, LIST_REQUEST_VERBS):
        notes.append("refine: imperative listing with a question word -> "
                     "ASK_LIST")
        return "ASK_LIST", notes
    if act == INT_RETRACT:
        return act, notes
    if act == INT_ASK_YESNO and _QUESTION.search(text or "") is None \
            and (text or "").strip().endswith("."):
        # "Ньютон — физик." without a question mark is still a yes/no probe
        # only if the parser said so; nothing to change, kept for clarity.
        return act, notes
    return act, notes
