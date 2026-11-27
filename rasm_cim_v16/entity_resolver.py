"""v0.16 — entity resolution.

Contract (this is the part that must not lie):

  * A *mention* is any surface string ("Ньютона", "сэр Исаак Ньютон",
    "Isaac Newton").  An *entity* is a KB record with a canonical name and a
    set of confirmed aliases.
  * ``resolve(mention)`` never mutates the KB.  It returns candidates with a
    confidence and a *reason* string.  It never auto-merges two entities
    because their names look alike.
  * Merging happens only through an explicit act:
      - ``link_alias(a, b)`` (from ``teach "Ньютон — это Исаак Ньютон"``)
      - an explicit ``same_as`` assertion.
  * When two *distinct* entities both match a mention with comparable
    confidence the result is AMBIGUOUS and the dialogue layer must ask the
    user — it must not pick one at random.

Gender is resolved for pronoun agreement (он/она/оно/они):
  1. explicit ``has_gender`` statement in the KB, else
  2. a first-name lookup (persons) / last-word ending heuristic.
Heuristic gender is reported with ``gender_source="heuristic"`` so the
context engine can refuse to rely on it alone when several candidates agree.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, Iterable, Iterator, Sequence

from .normalization import (MIN_FUZZY_LEN, case_forms, cross_script_form,
                            edit_distance, is_proper, normalize,
                            stem_candidates, strip_honorifics, word_tokens)

# resolution statuses
RES_EXACT = "EXACT"                # verbatim canonical name
RES_ALIAS = "ALIAS"                # verbatim registered alias
RES_NORMALIZED = "NORMALIZED"      # matched after case/punct/honorific normalisation
RES_MORPHOLOGICAL = "MORPHOLOGICAL"  # matched via case-ending stripping
RES_PARTIAL = "PARTIAL"            # matched via one token of a longer name
RES_TRANSLIT = "TRANSLIT"          # matched across script (Isaac Newton)
RES_FUZZY = "FUZZY"                # edit distance 1 (typo tolerance)
RES_AMBIGUOUS = "AMBIGUOUS"
RES_UNKNOWN = "UNKNOWN"

#: a mention only partly explained by a registered stem (different token count)
RES_MORPH_PARTIAL_CONF = 0.72

#: a one-character strip of a COMMON noun («физика» vs «физик»): real, but too
#: weak to merge two entities or to be stated as a confident identification
RES_MORPH_WEAK_CONF = 0.60

#: confidence by status — deliberately below 1.0 whenever inference was used.
#: Ordering is intentional: a registered name worn in another case outranks a
#: single-token match against a longer name, which outranks a typo correction.
STATUS_CONFIDENCE = {
    RES_EXACT: 1.00,
    RES_ALIAS: 1.00,
    RES_NORMALIZED: 0.98,
    RES_TRANSLIT: 0.88,
    RES_MORPHOLOGICAL: 0.86,
    RES_PARTIAL: 0.78,
    RES_FUZZY: 0.72,
    RES_AMBIGUOUS: 0.0,
    RES_UNKNOWN: 0.0,
}

#: gap required between best and second-best score for a *confident* pick.
#: Below this the resolver reports AMBIGUOUS instead of guessing.
AMBIGUITY_MARGIN = 0.05

MASCULINE, FEMININE, NEUTER, PLURAL = "masculine", "feminine", "neuter", "plural"

#: Common Russian given names -> gender.  Small on purpose: it exists to make
#: pronoun agreement work for the corpus, not to be a name database.
GIVEN_NAMES = {
    "исаак": MASCULINE, "иван": MASCULINE, "пётр": MASCULINE, "петр": MASCULINE,
    "александр": MASCULINE, "михаил": MASCULINE, "николай": MASCULINE,
    "дмитрий": MASCULINE, "сергей": MASCULINE, "андрей": MASCULINE,
    "альберт": MASCULINE, "никита": MASCULINE, "илья": MASCULINE,
    "лев": MASCULINE, "юрий": MASCULINE, "борис": MASCULINE,
    "мария": FEMININE, "анна": FEMININE, "елена": FEMININE, "ольга": FEMININE,
    "екатерина": FEMININE, "софья": FEMININE, "дарья": FEMININE,
    "наталия": FEMININE, "наталья": FEMININE, "алиса": FEMININE,
    "isaac": MASCULINE, "john": MASCULINE, "albert": MASCULINE,
    "charles": MASCULINE, "alan": MASCULINE, "marie": FEMININE,
    "mary": FEMININE, "ada": FEMININE,
}

#: nouns whose gender contradicts the naive "-а => feminine" rule
MASCULINE_A = frozenset({"мужчина", "папа", "дядя", "юноша", "парень", "дедушка",
                         "юнга", "староста", "коллега", "глава"})
FEMININE_CONSONANT = frozenset({"ночь", "дверь", "кровь", "любовь", "смерть",
                                "мысль", "соль", "роль", "осень", "тетрадь",
                                "площадь", "молодёжь", "жизнь", "сеть", "степень"})


@dataclass
class Candidate:
    entity_id: int
    mention: str
    canonical: str
    status: str
    confidence: float
    reason: str

    def as_dict(self) -> dict:
        return {"entity_id": self.entity_id, "mention": self.mention,
                "canonical": self.canonical, "status": self.status,
                "confidence": round(self.confidence, 4), "reason": self.reason}


@dataclass
class Resolution:
    mention: str
    status: str
    candidates: list[Candidate] = field(default_factory=list)

    @property
    def best(self) -> Candidate | None:
        return self.candidates[0] if self.candidates else None

    @property
    def entity_id(self) -> int | None:
        return self.candidates[0].entity_id if self.candidates else None

    @property
    def confidence(self) -> float:
        return self.candidates[0].confidence if self.candidates else 0.0

    def as_dict(self) -> dict:
        return {"mention": self.mention, "status": self.status,
                "confidence": round(self.confidence, 4),
                "candidates": [c.as_dict() for c in self.candidates]}


@dataclass
class Entity:
    entity_id: int
    canonical: str
    kind: str = ""
    gender: str = ""
    gender_source: str = ""            # "" | "explicit" | "heuristic"
    aliases: list[str] = field(default_factory=list)
    source: str = ""                   # "teach" | "question" | "migration" ...
    created_at: str = ""
    modified_at: str = ""
    history: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"id": self.entity_id, "canonical": self.canonical,
                "kind": self.kind, "gender": self.gender,
                "gender_source": self.gender_source,
                "aliases": list(self.aliases), "source": self.source,
                "created_at": self.created_at, "modified_at": self.modified_at,
                "history": list(self.history)}

    @classmethod
    def from_dict(cls, d: dict) -> "Entity":
        return cls(entity_id=int(d["id"]), canonical=d["canonical"],
                   kind=d.get("kind", ""), gender=d.get("gender", ""),
                   gender_source=d.get("gender_source", ""),
                   aliases=list(d.get("aliases", [])), source=d.get("source", ""),
                   created_at=d.get("created_at", ""),
                   modified_at=d.get("modified_at", ""),
                   history=list(d.get("history", [])))


class EntityResolver:
    """Mention -> entity.  Holds the alias index; owns no facts."""

    def __init__(self, ambiguity_margin: float = AMBIGUITY_MARGIN,
                 fuzzy_enabled: bool = True):
        self._entities: Dict[int, Entity] = {}
        self._next_id = 1
        #: comparable form -> set of entity ids that claim it
        self._index: Dict[str, set[int]] = {}
        #: single name token -> entity ids (so "Ньютона" can reach the
        #: multi-word entity "Исаак Ньютон").  A weaker signal than a full
        #: form match, therefore a lower confidence and its own status.
        self._token_index: Dict[str, set[int]] = {}
        #: entity id -> set of stemmed name tokens (canonical + aliases)
        self._name_stems: Dict[int, set[str]] = {}
        #: canonical-normalised form -> entity id (for EXACT vs NORMALIZED)
        self._canonical_norm: Dict[str, int] = {}
        self.ambiguity_margin = ambiguity_margin
        self.fuzzy_enabled = fuzzy_enabled

    # -- construction ------------------------------------------------------
    def intern(self, name: str, kind: str = "", source: str = "",
               when: str = "") -> Entity:
        """Return the entity for ``name``, creating it if needed.

        ``name`` is a *surface* mention.  If it resolves EXACT/ALIAS to an
        existing entity the existing entity is returned unchanged (no new
        record, no new alias).  Otherwise a NEW entity is created with
        ``name`` as its canonical name — the resolver never silently merges.
        """
        if not name or not name.strip():
            raise ValueError("cannot intern an empty name")
        canonical = _pretty(name)
        res = self.resolve(canonical, allow_fuzzy=False)
        if res.status in (RES_EXACT, RES_ALIAS, RES_NORMALIZED):
            return self._entities[res.entity_id]
        # A STRONG morphological match is the same entity wearing a different
        # case ("физиком" -> "физик"): reuse it instead of creating a near
        # duplicate.  Deliberately narrow: single candidate, no fuzzy match,
        # and >=2 characters of ending stripped (confidence >= 0.80).  The new
        # surface form is NOT recorded as an alias — the stem is already in
        # the index, so the alias list stays a list of *names*, not of
        # inflections.
        # Reuse only for inflected forms of the SAME name (МОРPHOLOGICAL /
        # NORMALIZED).  A PARTIAL hit ("Ньютон" against "Исаак Ньютон") is
        # deliberately NOT a merge: a shorter name may well be a different
        # entity, and merging it here would erase the surface entity that the
        # alias_of statement refers to.
        if (res.status in (RES_MORPHOLOGICAL, RES_NORMALIZED)
                and res.confidence >= 0.75
                and len({c.entity_id for c in res.candidates}) == 1):
            return self._entities[res.entity_id]
        ent = Entity(entity_id=self._next_id, canonical=canonical, kind=kind,
                     source=source, created_at=when, modified_at=when)
        self._next_id += 1
        self._entities[ent.entity_id] = ent
        self._canonical_norm[normalize(canonical)] = ent.entity_id
        self._add_form(ent.entity_id, canonical)
        self._recompute_gender(ent)
        return ent

    def get(self, entity_id: int) -> Entity | None:
        return self._entities.get(entity_id)

    def all_entities(self) -> list[Entity]:
        return [self._entities[i] for i in sorted(self._entities)]

    def __len__(self) -> int:
        return len(self._entities)

    def set_kind(self, entity_id: int, kind: str, when: str = "") -> None:
        ent = self._entities.get(entity_id)
        if ent and kind and ent.kind != kind:
            ent.history.append({"at": when, "change": f"kind {ent.kind!r}->{kind!r}"})
            ent.kind = kind
            ent.modified_at = when

    # -- explicit linking (the ONLY way two entities become one) -----------
    def link_alias(self, surface_id: int, canonical_id: int, when: str = "",
                   source: str = "teach") -> Entity:
        """Make ``surface_id`` an alias of ``canonical_id``.

        The alias entity record is KEPT (its id may be referenced by existing
        statements and evidence) but it is folded into the canonical entity's
        alias list, and every comparable form it owned now also points at the
        canonical entity.  Returns the canonical entity.
        """
        surf, canon = self._entities.get(surface_id), self._entities.get(canonical_id)
        if surf is None or canon is None:
            raise KeyError("unknown entity id in link_alias")
        if surf.entity_id == canon.entity_id:
            return canon
        if surf.canonical not in canon.aliases:
            canon.aliases.append(surf.canonical)
        for a in surf.aliases:
            if a not in canon.aliases:
                canon.aliases.append(a)
        for form in list(self._index):
            if surf.entity_id in self._index[form]:
                self._index[form].add(canon.entity_id)
        for tok in list(self._token_index):
            if surf.entity_id in self._token_index[tok]:
                self._token_index[tok].add(canon.entity_id)
        self._canonical_norm[normalize(surf.canonical)] = canon.entity_id
        canon.history.append({"at": when, "change": f"absorbed alias entity "
                                                    f"#{surf.entity_id} {surf.canonical!r}",
                              "source": source})
        canon.modified_at = when
        if not canon.kind and surf.kind:
            canon.kind = surf.kind
        self._recompute_gender(canon)
        return canon

    def add_alias(self, entity_id: int, alias: str, when: str = "",
                  source: str = "teach") -> None:
        ent = self._entities.get(entity_id)
        if ent is None:
            raise KeyError(entity_id)
        alias = _pretty(alias)
        if alias and alias not in ent.aliases and alias != ent.canonical:
            ent.aliases.append(alias)
            ent.history.append({"at": when, "change": f"+alias {alias!r}",
                                "source": source})
            ent.modified_at = when
        self._add_form(entity_id, alias)

    # -- lookup ------------------------------------------------------------
    def resolve(self, mention: str, allow_fuzzy: bool = True,
                prefer_kind: str = "") -> Resolution:
        """Resolve a mention to candidates.  Never mutates the KB."""
        raw = _pretty(mention)
        if not raw:
            return Resolution(raw, RES_UNKNOWN)
        norm = normalize(raw)
        if not norm:
            return Resolution(raw, RES_UNKNOWN)

        hits: dict[int, Candidate] = {}

        def consider(eid: int, status: str, conf: float, reason: str):
            ent = self._entities[eid]
            if prefer_kind and ent.kind and prefer_kind not in ("", ent.kind):
                conf *= 0.85
            prev = hits.get(eid)
            if prev is None or conf > prev.confidence:
                hits[eid] = Candidate(eid, raw, ent.canonical, status, conf, reason)

        # 1. verbatim canonical / alias
        if norm in self._canonical_norm:
            consider(self._canonical_norm[norm], RES_EXACT,
                     STATUS_CONFIDENCE[RES_EXACT], "verbatim canonical name")
        ids = self._index.get(norm)
        if ids:
            for eid in ids:
                consider(eid, RES_ALIAS, STATUS_CONFIDENCE[RES_ALIAS],
                         "registered alias / canonical form")

        # 2. honorific-stripped
        if not hits:
            nohon = strip_honorifics(raw)
            if normalize(nohon) != norm:
                for eid in (self._index.get(normalize(nohon)) or ()):
                    consider(eid, RES_NORMALIZED, STATUS_CONFIDENCE[RES_NORMALIZED],
                             "matched after dropping an honorific/title")

        # 3. morphological (case-ending) candidates.  The evidence is that the
        #    stripped form IS a registered name, so the score depends on
        #    whether the mention is a full inflected form of it (same number
        #    of tokens -> 0.86) or only partly explained by it (-> 0.72).
        #    Strip LENGTH is deliberately not used: a one-character Russian
        #    ending ("Ньютона" -> "Ньютон") is no weaker than a two-character
        #    one ("Ньютоном"), and scoring it lower let a token match on a
        #    longer name beat an exact stem match on the registered short name.
        if not hits:
            mtok_n = len(word_tokens(raw))
            for form, conf in stem_candidates(raw):
                if conf >= 1.0:
                    continue                  # verbatim form: handled above
                for eid in (self._index.get(form) or ()):
                    ent = self._entities[eid]
                    full_form = mtok_n == len(word_tokens(ent.canonical))
                    stripped = len(norm) - len(form)
                    if not full_form:
                        conf = RES_MORPH_PARTIAL_CONF
                    elif stripped <= 1 and not is_proper(ent.canonical):
                        # «физика» (physics) vs «физик» (physicist): one stripped
                        # character on a COMMON noun is genuinely ambiguous, so
                        # the match is kept but scored too low to merge entities
                        # or to be presented as a confident identification.
                        # For a proper name the same strip is a plain case
                        # ending («Ньютона» -> «Ньютон») and scores normally.
                        conf = RES_MORPH_WEAK_CONF
                    else:
                        conf = STATUS_CONFIDENCE[RES_MORPHOLOGICAL]
                    consider(eid, RES_MORPHOLOGICAL, conf,
                             f"case-normalised form {form!r} "
                             f"({stripped} char(s) stripped, "
                             f"confidence {conf:.2f})")
                if hits:
                    break

        # 4. partial: the mention is one token of a longer registered name
        #    ("Ньютона" -> "Исаак Ньютон").  Deliberately NOT gated on "no
        #    hits yet": when both "Ньютон" and "Исаак Ньютон" exist, a
        #    morphological hit on the first and a token hit on the second are
        #    equally weak evidence, and the resolver must report AMBIGUOUS
        #    rather than quietly preferring whichever it found first.
        mtoks = word_tokens(raw)
        if mtoks:
            mstems = set()
            for t in mtoks:
                forms = [f for f, _n in stem_candidates(t) if len(f) >= 4]
                mstems.add(min(forms, key=len) if forms else t)
            cand_ids: set[int] = set()
            for st_tok in mstems:
                cand_ids |= self._token_index.get(st_tok, set())
            for eid in sorted(cand_ids):
                estems = self._name_stems.get(eid) or set()
                if not mstems <= estems:
                    # the user said something this name does not contain —
                    # a surname hit alone must not explain the whole mention
                    continue
                if mstems == estems:
                    # same number of tokens: this is a full-name match and is
                    # already owned by the exact/alias/morphological steps
                    continue
                consider(eid, RES_PARTIAL, STATUS_CONFIDENCE[RES_PARTIAL],
                         f"all of {sorted(mstems)} are name tokens of the "
                         f"longer name {self._entities[eid].canonical!r}")

        # 5. cross-script (Isaac Newton <-> Исаак Ньютон)
        if not hits:
            xs = cross_script_form(raw)
            if xs:
                for form, conf in stem_candidates(xs):
                    for eid in (self._index.get(form) or ()):
                        consider(eid, RES_TRANSLIT, STATUS_CONFIDENCE[RES_TRANSLIT],
                                 f"transliterated match {form!r}")
                    if hits:
                        break

        # 6. typo tolerance — only for long mentions, only distance 1
        if not hits and allow_fuzzy and self.fuzzy_enabled and len(norm) >= MIN_FUZZY_LEN:
            for form in self._index:
                if abs(len(form) - len(norm)) > 1:
                    continue
                if edit_distance(form, norm, cap=1) == 1:
                    for eid in self._index[form]:
                        consider(eid, RES_FUZZY, STATUS_CONFIDENCE[RES_FUZZY],
                                 f"edit-distance-1 match on {form!r}")

        if not hits:
            return Resolution(raw, RES_UNKNOWN)

        ordered = sorted(hits.values(), key=lambda c: (-c.confidence, c.entity_id))
        top = ordered[0]
        if len(ordered) > 1 and ordered[0].entity_id != ordered[1].entity_id:
            gap = ordered[0].confidence - ordered[1].confidence
            tied = [c for c in ordered
                    if ordered[0].confidence - c.confidence <= self.ambiguity_margin
                    and c.entity_id != ordered[0].entity_id]
            if tied and gap <= self.ambiguity_margin:
                return Resolution(raw, RES_AMBIGUOUS, ordered)
        return Resolution(raw, top.status, ordered)

    def is_ambiguous(self, mention: str) -> bool:
        return self.resolve(mention).status == RES_AMBIGUOUS

    def forms_of(self, entity_id: int) -> set[str]:
        return {f for f, ids in self._index.items() if entity_id in ids}

    # -- gender ------------------------------------------------------------
    def set_gender(self, entity_id: int, gender: str, when: str = "",
                   source: str = "explicit") -> None:
        ent = self._entities.get(entity_id)
        if ent is None or not gender:
            return
        ent.gender = gender
        ent.gender_source = source
        ent.modified_at = when

    def _recompute_gender(self, ent: Entity) -> None:
        if ent.gender and ent.gender_source == "explicit":
            return
        g = infer_gender(ent.canonical, ent.aliases)
        if g and g != ent.gender:
            ent.gender = g
            ent.gender_source = "heuristic"

    # -- (de)serialisation -------------------------------------------------
    def to_dict(self) -> dict:
        return {"next_id": self._next_id,
                "entities": [e.as_dict() for e in self.all_entities()]}

    @classmethod
    def from_dict(cls, d: dict) -> "EntityResolver":
        r = cls()
        r._next_id = int(d.get("next_id", 1))
        for ed in d.get("entities", []):
            ent = Entity.from_dict(ed)
            r._entities[ent.entity_id] = ent
            r._next_id = max(r._next_id, ent.entity_id + 1)
            r._canonical_norm[normalize(ent.canonical)] = ent.entity_id
            r._add_form(ent.entity_id, ent.canonical)
            for a in ent.aliases:
                r._add_form(ent.entity_id, a)
        return r

    # -- internals ---------------------------------------------------------
    def _add_form(self, entity_id: int, form: str) -> None:
        if not form:
            return
        for f in _all_forms(form):
            self._index.setdefault(f, set()).add(entity_id)
        # individual (stemmed) tokens, so a surname alone can be recognised
        stems = self._name_stems.setdefault(entity_id, set())
        for t in word_tokens(form):
            forms = [f for f, _n in stem_candidates(t) if len(f) >= 4]
            for f in {t} | set(forms):
                self._token_index.setdefault(f, set()).add(entity_id)
            stems.add(min(forms, key=len) if forms else t)

    def rebuild_index(self) -> int:
        self._index.clear()
        self._token_index.clear()
        self._name_stems.clear()
        self._canonical_norm.clear()
        for ent in self._entities.values():
            self._canonical_norm[normalize(ent.canonical)] = ent.entity_id
            self._add_form(ent.entity_id, ent.canonical)
            for a in ent.aliases:
                self._add_form(ent.entity_id, a)
        return len(self._index)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _pretty(name: str) -> str:
    """Display form: NFC, squeeze spaces, strip surrounding punctuation,
    collapse an internal em-dash separator."""
    s = unicodedata.normalize("NFC", str(name or ""))
    s = re.sub(r"\s+", " ", s).strip(" \t\n\r\"'`.,;:!?—–-")
    return s


def _all_forms(name: str) -> set[str]:
    """Every comparable form a name answers to: normalised, honorific-free,
    morphological variants and transliterations."""
    forms = {normalize(name)}
    forms.add(normalize(strip_honorifics(name)))
    for f, _c in stem_candidates(name):
        forms.add(f)
    xs = cross_script_form(name)
    if xs:
        forms.add(normalize(xs))
        for f, _c in stem_candidates(xs):
            forms.add(f)
    return {f for f in forms if f}


def infer_gender(name: str, aliases: Sequence[str] = ()) -> str:
    """Heuristic grammatical/natural gender for pronoun agreement."""
    toks = word_tokens(name)
    if not toks:
        return ""
    # 1. known given name anywhere in the first two tokens (persons are
    #    disambiguated by the given name, not the surname)
    for t in toks[:2]:
        if t in GIVEN_NAMES:
            return GIVEN_NAMES[t]
    head = toks[-1]
    if head in MASCULINE_A:
        return MASCULINE
    if head in FEMININE_CONSONANT or head.endswith("ь"):
        if head.endswith("ь"):
            return FEMININE if head in FEMININE_CONSONANT else MASCULINE
    if head.endswith(("а", "я")):
        return FEMININE
    if head.endswith(("о", "е")):
        return NEUTER
    if head.endswith(("ий", "ый", "ой")):        # adjective-like class name
        return MASCULINE
    return MASCULINE if head[-1].isalpha() else ""


def plural_of(name: str) -> str:
    """Very small plural helper used by answer rendering ("законы")."""
    n = name.rstrip()
    if n.endswith(("а", "я")):
        return n[:-1] + ("ы" if n.endswith("а") else "и")
    if n.endswith(("о", "е")):
        return n[:-1] + "а"
    if n.endswith(("ь", "й", "ж", "ш", "щ", "ч")):
        return n + "и"
    return n + "ы"
