"""The v0.12/v0.11 relation vocabulary -> v0.16 relation schema.

This table is the whole integration contract, and it is deliberately explicit:
every legacy predicate is either

  * a **pass-through** to a built-in v0.16 relation with the same meaning,
  * a **custom relation** (registered into the graph's schema, persisted in
    ``kb.json`` under ``custom_relations``) when v0.16 has no equivalent,
  * a **boolean normalisation** for legacy predicates whose object is the word
    "да"/"нет" — those become ``has_property`` with the predicate's own gloss
    as the literal, because "Пингвин — может_летать → да" is not a graph edge,
  * or **unmapped**, in which case the importer refuses to guess: the
    statement is skipped and listed with a reason, so a human can extend this
    table.  Silently coercing an unknown predicate to ``related_to`` is
    exactly the kind of lie the roadmap forbids.

Nothing here rewrites v0.16: custom relations go through the public
``RelationSchema.register_dict`` API.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from rasm_cim_v16.relations import (KIND_ANY, KIND_CLASS, KIND_EVENT,
                                    KIND_PERSON, KIND_PLACE, KIND_THING,
                                    KIND_WORK, DEFAULT_SCHEMA)

#: legacy ``Entity.type`` (v0.12 vocabulary) -> v0.16 entity kind
ENTITY_KIND_MAP = {
    "person": KIND_PERSON,
    "animal": KIND_THING,        # no zoological kind exists; THING is honest
    "place": KIND_PLACE,
    "organization": KIND_THING,  # an organisation is not a PLACE and not a PERSON
    "concept": KIND_CLASS,       # "поэт", "млекопитающее" are classes
    "thing": KIND_THING,
    "work": KIND_WORK,
    "event": KIND_EVENT,
    "other": KIND_ANY,
}

BOOL_YES = frozenset({"да", "yes", "true", "1", "истина"})
BOOL_NO = frozenset({"нет", "no", "false", "0", "ложь"})


@dataclass(frozen=True)
class PredicateMapping:
    legacy: str
    target: str
    definition: dict | None = None       # custom relation registration
    boolean_object: bool = False         # object is да/нет -> has_property
    object_is_literal: bool | None = None  # None: decided by the statement
    note: str = ""

    @property
    def is_custom(self) -> bool:
        return self.definition is not None


def _rel(name: str, human: str, template: str, **kw) -> dict:
    d = {"name": name, "human": human, "template": template}
    d.update(kw)
    return d


PREDICATE_MAP: dict[str, PredicateMapping] = {
    # ---- pass-throughs (v0.16 already has the relation) -------------------
    "located_in": PredicateMapping("located_in", "located_in"),
    "wrote": PredicateMapping("wrote", "wrote"),
    "написал": PredicateMapping("написал", "wrote",
                                note="Russian surface verb from the v0.11 corpus"),
    "part_of": PredicateMapping("part_of", "part_of"),
    "has_part": PredicateMapping("has_part", "has_part"),
    "instance_of": PredicateMapping("instance_of", "instance_of"),
    "is_a": PredicateMapping(
        "is_a", "instance_of",
        note="v0.16 renamed the class-membership relation; the legacy code also "
             "used is_a for aliases, so each statement is re-checked against the "
             "subject/object surface forms before import"),
    "alias_of": PredicateMapping("alias_of", "alias_of"),
    "causes": PredicateMapping("causes", "causes"),

    # ---- custom relations: the legacy semantics have no v0.16 twin -------
    "capital_of": PredicateMapping(
        "capital_of", "capital_of",
        definition=_rel("capital_of", "столица", "столица {s} — {o}",
                        functional=True, inverse="is_capital_of",
                        surface=("столица", "столицей"),
                        question_object=("какая", "какой", "где")),
        note="subject = country, object = its capital (v0.11 direction)"),
    "является_столицей": PredicateMapping(
        "является_столицей", "is_capital_of",
        definition=_rel("is_capital_of", "является столицей",
                        "{s} — столица {o}", functional=False,
                        inverse="capital_of",
                        question_subject=("что", "какой город")),
        note="inverse direction of capital_of"),
    "имеет_спутник": PredicateMapping(
        "имеет_спутник", "has_satellite",
        definition=_rel("has_satellite", "имеет спутник",
                        "{s} имеет спутник {o}",
                        inverse="satellite_of",
                        surface=("спутник", "спутники"))),
    "кипит_при": PredicateMapping(
        "кипит_при", "boils_at",
        definition=_rel("boils_at", "кипит при", "{s} кипит при {o}",
                        functional=True, object_is_literal=None),
        object_is_literal=True,
        note="the object is a measurement ('100 цельсия'), never an entity"),
    "может_летать": PredicateMapping(
        "может_летать", "has_property", boolean_object=True,
        note="boolean property: 'может_летать -> да' becomes "
             "has_property(<сущность>, 'может летать') with the polarity of the "
             "legacy row"),
    "существует": PredicateMapping(
        "существует", "has_property", boolean_object=True,
        note="boolean property; assert+deny of the same property is a genuine "
             "contradiction and is imported as one"),
    # v0.12 functional predicates used by the teach pipeline
    "population_of": PredicateMapping(
        "population_of", "population_of",
        definition=_rel("population_of", "население", "население {s} — {o}",
                        functional=True, object_is_literal=True),
        object_is_literal=True),
    "height_of": PredicateMapping(
        "height_of", "height_of",
        definition=_rel("height_of", "высота", "высота {s} — {o}",
                        functional=True, object_is_literal=True),
        object_is_literal=True),
    "director_of": PredicateMapping(
        "director_of", "director_of",
        definition=_rel("director_of", "директор", "директор {s} — {o}",
                        functional=True)),
    "ceo_of": PredicateMapping(
        "ceo_of", "ceo_of",
        definition=_rel("ceo_of", "генеральный директор",
                        "генеральный директор {s} — {o}", functional=True)),
    "governor_of": PredicateMapping(
        "governor_of", "governor_of",
        definition=_rel("governor_of", "губернатор", "губернатор {s} — {o}",
                        functional=True)),
}

#: gloss used when a boolean legacy predicate is normalised to has_property
BOOLEAN_GLOSS = {
    "может_летать": "может летать",
    "существует": "существует",
}


def map_entity_kind(legacy_type: str) -> tuple[str, bool]:
    """``legacy type -> (v0.16 kind, was_mapped)``.  Unknown never guesses."""
    key = (legacy_type or "other").strip().lower()
    if key in ENTITY_KIND_MAP:
        return ENTITY_KIND_MAP[key], True
    return KIND_ANY, False


def map_predicate(legacy: str) -> PredicateMapping | None:
    """Return the mapping for a legacy predicate, or ``None`` if unmapped.

    A predicate that already *is* a built-in v0.16 relation name is its own
    mapping (identity), so exports from a future v0.16-aware writer import
    without a table entry.
    """
    key = (legacy or "").strip()
    if key in PREDICATE_MAP:
        return PREDICATE_MAP[key]
    folded = key.casefold()
    for k, v in PREDICATE_MAP.items():
        if k.casefold() == folded:
            return v
    if key in DEFAULT_SCHEMA:
        return PredicateMapping(key, key,
                                note="pass-through: already a v0.16 relation")
    return None


def unmapped_predicates(legacy_predicates) -> dict[str, int]:
    out: dict[str, int] = {}
    for p in legacy_predicates:
        if map_predicate(p) is None:
            out[p] = out.get(p, 0) + 1
    return out


def boolean_value(text: str) -> bool | None:
    """``"да" -> True``, ``"нет" -> False``, anything else -> ``None``."""
    t = (text or "").strip().casefold()
    if t in BOOL_YES:
        return True
    if t in BOOL_NO:
        return False
    return None


#: relations that are needed only as an *inverse* of a mapped relation.  They
#: are registered too, otherwise ``RelationSchema.resolve_inverse`` would point
#: at a name the graph cannot resolve.
EXTRA_RELATIONS: list[dict] = [
    _rel("satellite_of", "спутник", "{s} — спутник {o}",
         inverse="has_satellite", question_subject=("что", "чей")),
]


def custom_relations() -> list[dict]:
    """Every custom relation definition the table can need, de-duplicated."""
    seen: dict[str, dict] = {}
    for definition in EXTRA_RELATIONS:
        seen[definition["name"]] = definition
    for m in PREDICATE_MAP.values():
        if m.definition:
            seen[m.definition["name"]] = m.definition
    return list(seen.values())


#: legacy statement kinds and what the importer does with them
KIND_POLICY = {
    "user_assertion": "import",
    "extracted_fact": "import",
    "hypothesis": "skip",     # v0.16 has no hypothesis status: importing it as
                              # a direct fact would overstate confidence
}
