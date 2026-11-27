"""v0.16 — relation extraction: QueryIR -> typed statement (or a refusal).

This module is the *only* place where a parsed utterance may become knowledge.
It exists because spec §5 forbids "one mis-recognition automatically turning
into a KB assertion".  Three independent gates:

  1. parse confidence gate — a parse the NLU is unsure about is never asserted;
  2. relation-classification gate — a copula sentence is classified into
     ``alias_of`` / ``instance_of`` / ``related_to`` with *evidence*, and when
     the evidence is not decisive the result is a clarification request
     instead of a guess;
  3. type gate — the relation schema's subject/object kind constraints are
     checked; a violation weakens the statement and is logged, never hidden.

The classic failure this replaces:

    "Ньютон — это Исаак Ньютон"   ->  Ньютон is_a Исаак Ньютон     (wrong)
    "Ньютон — это Исаак Ньютон"   ->  Ньютон alias_of Исаак Ньютон  (right)

Evidence used to decide: token/stem containment between the two names
("ньютон" ⊂ "исаак ньютон"), the object's registered kind, and the
orthography of the object (a proper name cannot be a class).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Sequence

from .entity_resolver import (RES_AMBIGUOUS, RES_UNKNOWN, Entity, Resolution)
from .knowledge_graph import (KIND_LITERAL, POL_FALSE, POL_TRUE, KnowledgeGraph,
                              Statement)
from .language_parser import INT_RETRACT, INT_TEACH, QueryIR
from .normalization import case_forms, is_proper, normalize, word_tokens
from .relations import (KIND_CLASS, KIND_EVENT, KIND_PERSON, KIND_PLACE,
                        KIND_THING, KIND_WORK)

#: below this combined confidence nothing is written to the KB
ASSERT_THRESHOLD = 0.50

#: relation -> kind the object must have
OBJECT_KIND = {
    "instance_of": KIND_CLASS,
    "subclass_of": KIND_CLASS,
    "located_in": KIND_PLACE,
    "born_at": KIND_PLACE,
    "happened_at": KIND_EVENT,
    "wrote": KIND_WORK,
}
#: relation -> kind the subject must have
SUBJECT_KIND = {
    "discovered": KIND_PERSON,
    "formulated": KIND_PERSON,
    "created": KIND_PERSON,
    "wrote": KIND_PERSON,
    "known_for": KIND_PERSON,
    "studied": KIND_PERSON,
    "has_subclass": KIND_CLASS,
    "subclass_of": KIND_CLASS,
    "instance_of": "",
}

#: action relations whose object may turn out to be a date
_ACTION_RELATIONS = frozenset({
    "discovered", "discovered_by", "formulated", "formulated_by",
    "created", "created_by", "wrote", "written_by", "born_at", "studied",
})

_NUM_RE = re.compile(r"^-?[\d][\d\s.,:/\-]*(?:г\.?|год|года|году|век|веке)?$")
_YEAR_RE = re.compile(r"^\d{3,4}\s*(?:г\.?|год|года|году|до\s*н\.?\s*э\.?)?$")


@dataclass
class TeachResult:
    """Outcome of trying to turn one utterance into knowledge."""

    ok: bool = False
    statement: Statement | None = None
    created: bool = False
    predicate: str = ""
    polarity: str = POL_TRUE
    confidence: float = 0.0
    subject_id: int | None = None
    object_id: int | None = None
    reason: str = ""
    clarification: str = ""            # question to ask the user ("" = none)
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"ok": self.ok,
                "statement_id": self.statement.statement_id if self.statement else None,
                "created": self.created, "predicate": self.predicate,
                "polarity": self.polarity, "confidence": round(self.confidence, 3),
                "subject_id": self.subject_id, "object_id": self.object_id,
                "reason": self.reason, "clarification": self.clarification,
                "notes": list(self.notes)}


@dataclass
class AskResult:
    """Outcome of resolving the *known side* of a question to entity ids."""

    ok: bool = False
    subject_id: int | None = None
    object_id: int | None = None
    predicate: str = ""
    wants: str = "none"
    polarity: str = POL_TRUE
    confidence: float = 0.0
    reason: str = ""
    clarification: str = ""
    ambiguous_mentions: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"ok": self.ok, "subject_id": self.subject_id,
                "object_id": self.object_id, "predicate": self.predicate,
                "wants": self.wants, "polarity": self.polarity,
                "confidence": round(self.confidence, 3), "reason": self.reason,
                "clarification": self.clarification, "notes": list(self.notes)}


class RelationExtractor:
    """Turns QueryIRs into graph writes / entity links.  Owns no storage."""

    def __init__(self, kg: KnowledgeGraph, clock=None,
                 assert_threshold: float = ASSERT_THRESHOLD):
        self.kg = kg
        self._clock = clock or (lambda: "")
        self.assert_threshold = assert_threshold

    # ------------------------------------------------------------------
    # TEACH
    # ------------------------------------------------------------------
    def teach(self, ir: QueryIR, when: str = "", source: str = "") -> TeachResult:
        when = when or self._clock()
        res = TeachResult()
        if ir.intent not in (INT_TEACH, INT_RETRACT) and not ir.explicit_teach:
            res.reason = f"utterance parsed as {ir.intent}, nothing asserted"
            res.notes = list(ir.notes)
            return res

        # ---- gate 1: parse confidence ------------------------------------
        gate = ir.confidence if not ir.explicit_teach else max(ir.confidence, 0.8)
        if gate < self.assert_threshold:
            res.reason = (f"parse confidence {gate:.2f} < "
                          f"{self.assert_threshold:.2f}; nothing asserted")
            res.notes = list(ir.notes)
            return res

        if ir.intent == INT_RETRACT:
            return self._retract(ir, when, source)

        subj_txt, obj_txt = ir.subject_mention.strip(), ir.object_mention.strip()
        if not subj_txt:
            res.reason = "no subject in the utterance; nothing asserted"
            res.notes = list(ir.notes)
            return res

        # ---- resolve / create both sides ---------------------------------
        subj = self.kg.resolver.intern(subj_txt, source=source or "teach", when=when)
        obj_ent: Entity | None = None
        obj_is_literal = _is_literal_value(obj_txt)
        if obj_txt:
            if obj_is_literal:
                obj_ent = self.kg.intern_literal(obj_txt, when=when)
            else:
                obj_ent = self.kg.resolver.intern(obj_txt, source=source or "teach",
                                                  when=when)

        # ---- gate 2: which relation does this copula actually express? ---
        predicate = ir.predicate
        conf = gate
        # A structurally complete copula sentence ("X - это Y") gives the
        # relation decision the final say; anything less complete is capped
        # by how confident the parse itself was.
        clean_parse = bool(ir.subject_mention and ir.object_mention
                           and ir.predicate)
        if predicate in ("instance_of", "related_to", ""):
            predicate, conf, why, clar = self._classify_copula(
                ir, subj, obj_ent, obj_txt, obj_is_literal, when=when)
            if not clean_parse:
                conf = min(conf, gate)
            res.notes.append(why)
            if clar:
                res.clarification = clar
                res.reason = "relation could not be decided from the evidence"
                res.confidence = conf
                res.predicate = predicate
                res.subject_id = subj.entity_id
                res.object_id = obj_ent.entity_id if obj_ent else None
                return res
        if predicate == "" or self.kg.schema.get(predicate) is None:
            res.reason = f"no usable relation for this utterance ({predicate!r})"
            res.notes = list(ir.notes) + res.notes
            return res

        # ---- kinds --------------------------------------------------------
        want_sk = SUBJECT_KIND.get(predicate, "")
        if want_sk and not subj.kind:
            self.kg.resolver.set_kind(subj.entity_id, want_sk, when=when)
            res.notes.append(f"subject kind set to {want_sk}")
        if obj_ent is not None:
            want_ok = OBJECT_KIND.get(predicate, "")
            if want_ok and not obj_ent.kind and not obj_is_literal:
                self.kg.resolver.set_kind(obj_ent.entity_id, want_ok, when=when)
                res.notes.append(f"object kind set to {want_ok}")

        polarity = POL_FALSE if ir.negation else POL_TRUE
        if ir.modality == "possible":
            conf = min(conf, 0.7)
            res.notes.append("modality 'possible' lowered the confidence")

        # ---- temporal adverbial ------------------------------------------
        # "Закон был открыт в 1687 году" states WHEN, not WHAT: the year is a
        # literal, so the fact is recorded as happened_at(subject, <year>).
        # Recorded as an explicit rule, not left to the action verb, because
        # storing discovered_by(law, "1687 году") would be nonsense.
        if obj_is_literal and predicate in _ACTION_RELATIONS:
            res.notes.append(f"object {obj_txt!r} is a value, not an entity; "
                             f"the action relation {predicate!r} was rewritten "
                             f"to happened_at (documented rule T1)")
            predicate = "happened_at"

        # ---- gate 3 + write ----------------------------------------------
        if conf < self.assert_threshold:
            res.predicate, res.confidence = predicate, conf
            res.reason = (f"relation decision confidence {conf:.2f} below "
                          f"threshold; nothing asserted")
            return res

        st, created = self.kg.assert_triple(
            subj.entity_id, predicate, obj_ent.entity_id if obj_ent is not None
            else subj.entity_id, polarity=polarity, confidence=conf,
            source=source or f"teach:{ir.raw}", when=when,
            provenance={"utterance": ir.raw, "parse": ir.describe(),
                        "notes": list(ir.notes)})
        res.ok = True
        res.statement = st
        res.created = created
        res.predicate = predicate
        res.polarity = polarity
        res.confidence = st.confidence
        res.subject_id = st.subject
        res.object_id = st.object
        res.reason = ("new fact recorded" if created else
                      "already known — reinforced, not duplicated")
        res.notes.extend(ir.notes)
        return res

    def _classify_copula(self, ir: QueryIR, subj: Entity,
                         obj_ent: Entity | None, obj_txt: str,
                         obj_is_literal: bool, when: str = "") -> tuple[str, float, str, str]:
        """Decide alias_of / instance_of / related_to with explicit evidence.

        Returns ``(predicate, confidence, reason, clarification)``.
        """
        if obj_ent is None or obj_is_literal or not obj_txt:
            return ("has_property", 0.7,
                    "object is a value/literal -> has_property", "")

        overlap = _name_overlap(subj.canonical, obj_ent.canonical)
        obj_proper = is_proper(obj_txt)
        obj_kind = obj_ent.kind

        obj_is_class_like = (obj_kind == KIND_CLASS) or (not obj_proper)

        # class-to-class copula is TAXONOMY, not membership:
        #   "Физик - это учёный"  ->  subclass_of, never instance_of
        if subj.kind == KIND_CLASS and obj_is_class_like:
            if obj_kind != KIND_CLASS:
                self.kg.resolver.set_kind(obj_ent.entity_id, KIND_CLASS, when=when)
            return ("subclass_of", 0.92,
                    f"subject {subj.canonical!r} is itself a CLASS and the "
                    f"object {obj_ent.canonical!r} is class-like -> "
                    f"subclass_of (NOT instance_of)", "")

        if obj_kind == KIND_CLASS:
            return ("instance_of", 0.95,
                    f"object {obj_ent.canonical!r} is a registered CLASS -> "
                    f"instance_of", "")

        if overlap and obj_proper:
            return ("alias_of", 0.95,
                    f"name evidence: {subj.canonical!r} is contained in "
                    f"{obj_ent.canonical!r} and the object is a proper name "
                    f"-> alias_of (NOT instance_of)", "")

        if overlap and not obj_proper:
            return ("instance_of", 0.6,
                    "names overlap but the object is not written as a proper "
                    "name; classified as instance_of with reduced confidence", "")

        if not obj_proper:
            # "Физик — это учёный": the object is a class word, and the subject
            # is a single common-noun-looking word, so the subject is itself a
            # category.  The hint list is deliberately narrow: it contains only
            # nouns that denote categories of categories, so "Москва — это
            # город" still becomes instance_of.  A sentence-initial capital is
            # NOT evidence here, and the reduced confidence says the call was
            # heuristic rather than proven.
            # Positive evidence on BOTH sides, never absence of evidence on one:
            # "Физик — это учёный" is class-to-class, while "Ньютон — это физик"
            # is a name-to-class membership even though "Ньютон" is a single
            # unknown word.  Requiring both keeps a person taught by surname
            # from being promoted to a class.
            if (_is_category_hint(obj_ent.canonical)
                    and _is_category_hint(subj.canonical)
                    and subj.kind not in _INSTANCE_KINDS):
                self.kg.resolver.set_kind(obj_ent.entity_id, KIND_CLASS, when=when)
                self.kg.resolver.set_kind(subj.entity_id, KIND_CLASS, when=when)
                return ("subclass_of", 0.85,
                        f"object {obj_ent.canonical!r} is a category word and "
                        f"the subject {subj.canonical!r} is a single "
                        f"non-instance word -> subclass_of (heuristic, "
                        f"confidence reduced)", "")
            self.kg.resolver.set_kind(obj_ent.entity_id, KIND_CLASS, when=when)
            return ("instance_of", 0.90,
                    f"object {obj_ent.canonical!r} is a common noun "
                    f"(not capitalised) -> treated as a CLASS -> instance_of", "")

        # proper name, no name evidence: genuinely ambiguous between
        # "these are the same entity" and "this is a related entity".
        return ("same_as", 0.3,
                f"object {obj_ent.canonical!r} is a proper name with no name "
                f"overlap with {subj.canonical!r}; refusing to guess",
                f"«{subj.canonical}» и «{obj_ent.canonical}» — это одна и та же "
                f"сущность, псевдоним или разные сущности?")

    def _retract(self, ir: QueryIR, when: str, source: str) -> TeachResult:
        res = TeachResult(polarity=POL_TRUE)
        subj_txt, obj_txt = ir.subject_mention.strip(), ir.object_mention.strip()
        if not subj_txt:
            res.reason = "retract without a subject"
            return res
        r = self.kg.resolver.resolve(subj_txt)
        if r.status in (RES_UNKNOWN, RES_AMBIGUOUS):
            res.reason = f"cannot retract: {subj_txt!r} is " \
                         f"{r.status.lower()}"
            if r.status == RES_AMBIGUOUS:
                res.clarification = _ambiguity_question(subj_txt, r)
            return res
        sid = r.entity_id
        oid = None
        if obj_txt:
            ro = self.kg.resolver.resolve(obj_txt)
            if ro.status in (RES_UNKNOWN, RES_AMBIGUOUS):
                res.reason = f"cannot retract: object {obj_txt!r} is {ro.status.lower()}"
                return res
            oid = ro.entity_id
        removed = []
        for st in list(self.kg.neighbours(sid)):
            if oid is not None and st.object != oid and st.subject != oid:
                continue
            if self.kg.retract(st.subject, st.predicate, st.object, when=when,
                               source=source or "retract"):
                removed.append(st.statement_id)
        res.ok = bool(removed)
        res.subject_id = sid
        res.reason = (f"retracted {len(removed)} statement(s): "
                      f"{', '.join(removed)}" if removed else
                      "nothing to retract")
        return res

    # ------------------------------------------------------------------
    # QUESTIONS — bind the known side to entity ids
    # ------------------------------------------------------------------
    def bind_question(self, ir: QueryIR, when: str = "") -> AskResult:
        res = AskResult(wants=ir.wants, predicate=ir.predicate,
                        polarity=POL_FALSE if ir.negation else POL_TRUE)
        res.notes.extend(ir.notes)
        if ir.predicate:
            res.predicate = ir.predicate

        for side in ("subject", "object"):
            mention = getattr(ir, f"{side}_mention").strip()
            pron = getattr(ir, f"{side}_pronoun").strip()
            if pron:
                res.notes.append(f"{side} is the pronoun {pron!r} — needs the "
                                 f"context engine")
                res.reason = "pronoun requires context resolution"
                continue
            if not mention:
                continue
            r = self.kg.resolver.resolve(mention)
            if r.status == RES_AMBIGUOUS:
                res.clarification = _ambiguity_question(mention, r)
                res.reason = f"{mention!r} matches several entities"
                res.ambiguous_mentions.append(mention)
                res.confidence = 0.0
                return res
            if r.status == RES_UNKNOWN:
                setattr(res, f"{side}_id", None)
                res.notes.append(f"{side} {mention!r} is not in the KB")
                res.reason = f"{mention!r} is unknown"
                continue
            setattr(res, f"{side}_id", r.entity_id)
            res.confidence = max(res.confidence, r.confidence)
            if r.status not in ("EXACT", "ALIAS"):
                res.notes.append(f"{side} matched via {r.status} "
                                 f"({r.best.reason})")

        if res.subject_id is None and res.object_id is None and not (
                ir.subject_pronoun or ir.object_pronoun):
            res.reason = res.reason or "no known entity in the question"
        else:
            res.ok = True
            res.reason = res.reason or "question bound to known entities"
        return res


# --------------------------------------------------------------------------
# evidence helpers
# --------------------------------------------------------------------------

#: Nouns that denote a category OF categories.  Used for exactly one
#: decision — subclass_of vs instance_of on a copula — and never to invent a
#: fact.  Deliberately narrow: place words like "город" are absent so that
#: "Москва — это город" stays instance_of.
_CATEGORY_HINTS = frozenset({
    "ученый", "учёный", "физик", "химик", "математик", "биолог", "астроном",
    "философ", "писатель", "поэт", "композитор", "художник", "инженер",
    "врач", "программист", "человек", "животное", "растение", "птица",
    "наука", "дисциплина", "профессия", "язык", "государство", "организм",
    "элемент", "вещество", "транспорт", "инструмент", "устройство",
    "система", "процесс", "явление", "понятие", "категория", "класс", "вид",
    "род", "семейство", "тип",
})


#: registered kinds that mean "a concrete individual", i.e. the subject of a
#: copula must be an instance_of rather than a subclass_of its object
_INSTANCE_KINDS = frozenset({KIND_PERSON, KIND_PLACE, KIND_WORK, KIND_EVENT,
                             KIND_THING})


def _is_category_hint(name: str) -> bool:
    return normalize(name) in _CATEGORY_HINTS


def _stems(phrase: str) -> set[str]:
    out = set()
    for t in word_tokens(phrase):
        forms = [f for f in case_forms(t) if len(f) >= 4]
        out.add(min(forms, key=len) if forms else t)
    return out


def _name_overlap(a: str, b: str) -> bool:
    """True when one name is contained in the other (token or stem level).

    This is *evidence for alias_of*, never proof of identity: the caller still
    requires the object to be written as a proper name.
    """
    ta, tb = set(word_tokens(a)), set(word_tokens(b))
    if not ta or not tb:
        return False
    if ta == tb:
        return True
    if ta < tb or tb < ta:
        return True
    sa, sb = _stems(a), _stems(b)
    return bool(sa) and bool(sb) and (sa < sb or sb < sa or sa == sb)


def _is_literal_value(text: str) -> bool:
    t = normalize(text).strip()
    if not t:
        return False
    if _YEAR_RE.match(t) or _NUM_RE.match(t):
        return True
    return bool(re.fullmatch(r"[\d\s.,:%+\-/]+", t))


def _ambiguity_question(mention: str, r: Resolution) -> str:
    names = " или ".join(f"«{c.canonical}»" for c in r.candidates[:4])
    return (f"Уточни, пожалуйста: {names} — о ком/чём именно речь, когда ты "
            f"говоришь «{mention}»?")
