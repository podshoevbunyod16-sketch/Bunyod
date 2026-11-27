"""v0.16 — typed relation schema (the semantic model).

Why this exists: v0.1–v0.10 had exactly one generic triple shape
(``TruthGraph9(s, r, o)`` with *integer* ids and a caller-supplied
``functional`` set).  There was no relation vocabulary at all, so a system
built on it could only record ``Ньютон is_a Исаак Ньютон`` — which conflates
"these two labels denote one thing" (alias) with "this thing belongs to a
class" (instance).  v0.16 separates those.

The schema is *data*, not code: relations are registered records.  Adding a
relation never touches the graph, the parser core or the inference core —
you register it (optionally from a plain dict / JSON) and, if it can be
inferred, attach rules to it.

Semantics per relation:
    direction        (subject, predicate, object) reading order
    subj_types/obj_types  allowed entity kinds (empty = unconstrained)
    functional       at most one object per subject
    symmetric        s P o  =>  o P s
    transitive       s P m, m P o  =>  s P o
    inverse          name of the mirror relation ("" = none)
    surface          Russian verbs/prepositions that express it
    question         Russian wh-words that ask for the object / subject
    disjoint_with    classes that cannot both hold for one subject
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, Iterable, Sequence

#: Entity kinds used by the type constraints.  Open set — unknown kinds are
#: accepted and simply never constrained.
KIND_PERSON = "PERSON"
KIND_CLASS = "CLASS"
KIND_PLACE = "PLACE"
KIND_WORK = "WORK"
KIND_EVENT = "EVENT"
KIND_THING = "THING"
KIND_ANY = "ANY"


@dataclass(frozen=True)
class Relation:
    """One relation type.  Frozen so a registered relation cannot be mutated
    behind the graph's back; use ``RelationSchema.register`` to replace one."""

    name: str
    human: str = ""                      # Russian gloss, for answer rendering
    subj_types: tuple[str, ...] = ()
    obj_types: tuple[str, ...] = ()
    functional: bool = False
    symmetric: bool = False
    transitive: bool = False
    inverse: str = ""
    surface: tuple[str, ...] = ()        # verbs / prepositions expressing it
    question_object: tuple[str, ...] = ()   # wh-words asking for the object
    question_subject: tuple[str, ...] = ()  # wh-words asking for the subject
    disjoint_with: tuple[str, ...] = ()
    note: str = ""
    #: Russian rendering template with {s} and {o}; empty -> fall back to the
    #: gloss.  Answer text is built from this, never from a hardcoded phrase.
    template: str = ""

    def allows(self, subj_kind: str, obj_kind: str) -> bool:
        """Type check.  ``KIND_ANY`` / unknown / empty constraint == allowed."""
        def ok(kind: str, allowed: Sequence[str]) -> bool:
            if not allowed or kind in ("", KIND_ANY) or KIND_ANY in allowed:
                return True
            return kind in allowed
        return ok(subj_kind, self.subj_types) and ok(obj_kind, self.obj_types)

    def to_dict(self) -> dict:
        d = asdict(self)
        for k in ("subj_types", "obj_types", "surface", "question_object",
                  "question_subject", "disjoint_with"):
            d[k] = list(d[k])
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Relation":
        kw = dict(d)
        for k in ("subj_types", "obj_types", "surface", "question_object",
                  "question_subject", "disjoint_with"):
            if k in kw:
                kw[k] = tuple(kw[k] or ())
        known = {f for f in cls.__dataclass_fields__}          # noqa: SLF001
        return cls(**{k: v for k, v in kw.items() if k in known})


# --------------------------------------------------------------------------
# built-in vocabulary
# --------------------------------------------------------------------------

def _builtin() -> list[Relation]:
    """The built-in relation vocabulary.

    Declared as a table rather than 30 constructor calls so that the
    semantics of each relation (direction, argument kinds, functionality,
    symmetry, transitivity, surface forms, question words, rendering
    template) can be read in one place and audited line by line.
    """
    T: list[dict] = [
        # ---- identity -----------------------------------------------------
        dict(name="same_as", human="это та же сущность", symmetric=True,
             template="{s} и {o} — одна и та же сущность",
             note="two records denote one and the same entity; identity is "
                  "never inferred from similarity, only asserted explicitly"),
        dict(name="alias_of", human="псевдоним / альтернативное имя",
             inverse="has_alias", template="{s} — также известен как {o}",
             note="surface name -> canonical entity.  This is exactly what "
                  "the legacy `is_a` was (mis)used for."),
        dict(name="has_alias", human="имеет псевдоним", inverse="alias_of",
             template="{s} также известен как {o}",
             note="canonical entity -> surface name (mirror of alias_of)."),

        # ---- taxonomy -----------------------------------------------------
        dict(name="instance_of", human="является экземпляром класса",
             obj_types=(KIND_CLASS,), inverse="has_instance",
             template="{s} — {o}",
             note="object belongs to a class.  NOT transitive over itself; "
                  "it composes over subclass_of instead."),
        dict(name="has_instance", human="имеет экземпляр",
             subj_types=(KIND_CLASS,), inverse="instance_of",
             template="{o} — это {s}"),
        dict(name="subclass_of", human="подкласс", subj_types=(KIND_CLASS,),
             obj_types=(KIND_CLASS,), transitive=True, inverse="has_subclass",
             template="{s} — подкласс {o}",
             note="class taxonomy; instance_of composes over this chain."),
        dict(name="has_subclass", human="надкласс", subj_types=(KIND_CLASS,),
             obj_types=(KIND_CLASS,), transitive=True, inverse="subclass_of",
             template="{o} — подкласс {s}"),

        # ---- mereology ----------------------------------------------------
        dict(name="part_of", human="часть целого", transitive=True,
             inverse="has_part", template="{s} — часть {o}"),
        dict(name="has_part", human="имеет часть", transitive=True,
             inverse="part_of", template="{s} содержит {o}"),

        # ---- attributes ---------------------------------------------------
        dict(name="has_property", human="свойство / признак",
             template="{s}: {o}",
             note="free-form attribute; the object may be a literal phrase."),
        dict(name="has_gender", human="грамматический/биологический род",
             functional=True, template="{s}: род {o}",
             note="used by pronoun resolution; masculine|feminine|neuter|"
                  "plural or free text."),

        # ---- agency / authorship -----------------------------------------
        dict(name="discovered", human="открыл", subj_types=(KIND_PERSON,),
             obj_types=(KIND_THING, KIND_WORK, KIND_EVENT),
             inverse="discovered_by", template="{s} открыл {o}",
             surface=("открыл", "открыла", "открыли", "открывает", "открывал",
                      "открытие"),
             question_object=("что",), question_subject=("кто",)),
        dict(name="discovered_by", human="открыто кем", inverse="discovered",
             template="{s} открыт {o}",
             surface=("открыт", "открыта", "открыто", "открыты"),
             question_subject=("кем", "кто")),
        dict(name="formulated", human="сформулировал",
             subj_types=(KIND_PERSON,), obj_types=(KIND_THING, KIND_WORK),
             inverse="formulated_by", template="{s} сформулировал {o}",
             surface=("сформулировал", "сформулировала", "сформулировали",
                      "формулирует", "сформулирован", "сформулирована",
                      "сформулировано"),
             question_object=("что", "какой", "какую", "какое"),
             question_subject=("кто",)),
        dict(name="formulated_by", human="сформулировано кем",
             inverse="formulated", template="{s} сформулирован {o}",
             surface=("сформулированы",), question_subject=("кем", "кто")),
        dict(name="created", human="создал", subj_types=(KIND_PERSON,),
             obj_types=(KIND_THING, KIND_WORK), inverse="created_by",
             template="{s} создал {o}",
             surface=("создал", "создала", "создали", "создает", "создаёт",
                      "разработал", "разработала", "изобрел", "изобрёл",
                      "изобрела"),
             question_object=("что",), question_subject=("кто",)),
        dict(name="created_by", human="создано кем", inverse="created",
             template="{s} создан {o}", surface=("создан", "создана"),
             question_subject=("кем", "кто")),
        dict(name="wrote", human="автор", subj_types=(KIND_PERSON,),
             obj_types=(KIND_WORK,), inverse="written_by",
             template="{s} написал {o}",
             surface=("написал", "написала", "написали", "автор", "написан",
                      "написана"),
             question_object=("что", "какое", "какую"),
             question_subject=("кто",)),
        dict(name="written_by", human="написано кем", inverse="wrote",
             template="{s} написан {o}", surface=("написаны",),
             question_subject=("кем", "кто")),

        # ---- space & time -------------------------------------------------
        dict(name="located_in", human="расположен в", obj_types=(KIND_PLACE,),
             transitive=True, inverse="contains",
             template="{s} находится в {o}",
             surface=("находится", "расположен", "расположена", "расположено",
                      "расположены"),
             question_object=("где",), question_subject=("что",)),
        dict(name="contains", human="содержит", subj_types=(KIND_PLACE,),
             transitive=True, inverse="located_in",
             template="{s} содержит {o}"),
        dict(name="born_at", human="место рождения", obj_types=(KIND_PLACE,),
             inverse="birthplace_of", template="{s} родился в {o}",
             surface=("родился", "родилась", "родились"),
             question_object=("где",)),
        dict(name="birthplace_of", human="место рождения (обр.)",
             inverse="born_at", template="{o} — место рождения {s}"),
        dict(name="happened_at", human="произошло (время)",
             obj_types=(KIND_EVENT,), inverse="event_of",
             template="{s}: {o}", surface=("году",),
             question_object=("когда",)),
        dict(name="event_of", human="время события", inverse="happened_at",
             template="{o}: {s}"),

        # ---- causality & loose links -------------------------------------
        dict(name="causes", human="причина", transitive=True,
             inverse="caused_by", template="{s} вызывает {o}",
             surface=("вызывает", "приводит", "причина"),
             question_object=("что",), question_subject=("что",),
             note="transitive but depth-capped by the inference engine — long "
                  "causal chains are exactly where runaway inference hurts."),
        dict(name="caused_by", human="следствие", transitive=True,
             inverse="causes", template="{s} вызвано {o}"),
        dict(name="known_for", human="известен благодаря",
             subj_types=(KIND_PERSON,), template="{s} известен благодаря {o}",
             surface=("известен", "известна", "известно", "известны",
                      "благодаря", "славится", "знаменит", "знаменита"),
             question_object=("чем",), question_subject=("кто",)),
        dict(name="studied", human="изучал", subj_types=(KIND_PERSON,),
             template="{s} изучал {o}",
             surface=("изучал", "изучала", "занимался", "занималась", "работал",
                      "работала"),
             question_object=("что", "чем"), question_subject=("кто",)),
        dict(name="related_to", human="связан (общая связь)", symmetric=True,
             template="{s} связан с {o}",
             note="FALLBACK ONLY: used when a more precise relation could not "
                  "be established.  Carries a confidence penalty and is always "
                  "reported as such in :explain."),
    ]
    return [Relation(**d) for d in T]


# --------------------------------------------------------------------------
# schema container
# --------------------------------------------------------------------------

class RelationSchema:
    """Registry of relation types.  Extensible at runtime.

    ``schema.register_dict({"name": "teaches", ...})`` adds a relation without
    touching any core module — the parser picks up its ``surface`` strings and
    the inference engine picks up its flags automatically.
    """

    def __init__(self, builtin: bool = True):
        self._rels: Dict[str, Relation] = {}
        self._surface: Dict[str, str] = {}          # surface word -> relation
        self._custom: list[str] = []                # names added after init
        if builtin:
            for r in _builtin():
                self.register(r, _custom=False)

    # -- registration ------------------------------------------------------
    def register(self, rel: Relation, _custom: bool = True) -> None:
        # anything not loaded as part of the built-in vocabulary is a
        # user/KB extension — recorded so :memory and kb.json can show it
        if _custom:
            self._custom.append(rel.name)
        self._rels[rel.name] = rel
        for s in rel.surface:
            self._surface.setdefault(s.lower(), rel.name)
        # inverse wiring is resolved on read (resolve_inverse) so registration
        # order does not matter.

    def register_dict(self, d: dict) -> Relation:
        rel = Relation.from_dict(d)
        if not rel.name:
            raise ValueError("relation needs a name")
        self.register(rel)
        return rel

    def register_many(self, items: Iterable[dict]) -> list[Relation]:
        return [self.register_dict(d) for d in items]

    # -- lookup ------------------------------------------------------------
    def get(self, name: str) -> Relation | None:
        return self._rels.get(name)

    def __contains__(self, name: str) -> bool:
        return name in self._rels

    def __iter__(self):
        return iter(self._rels.values())

    def names(self) -> list[str]:
        return sorted(self._rels)

    def custom_names(self) -> list[str]:
        return sorted(set(self._custom))

    def relation_for_surface(self, word: str) -> str | None:
        """Map an inflected/surface word to a relation name.  Exact surface
        match first, then the verb stem (so "открыли" -> discovered even when
        only "открыл" is registered)."""
        w = word.lower()
        if w in self._surface:
            return self._surface[w]
        from .normalization import verb_stem
        st = verb_stem(w)
        if st:
            for surf, rel in self._surface.items():
                if verb_stem(surf) == st:
                    return rel
        return None

    def relations_for_surfaces(self, words: Iterable[str]) -> list[tuple[str, str]]:
        """[(surface_word, relation_name)] for every recognised word."""
        out = []
        for w in words:
            r = self.relation_for_surface(w)
            if r:
                out.append((w, r))
        return out

    def resolve_inverse(self, name: str) -> str | None:
        rel = self._rels.get(name)
        if rel is None or not rel.inverse:
            return None
        return rel.inverse if rel.inverse in self._rels else None

    def functional_relations(self) -> set[str]:
        return {r.name for r in self._rels.values() if r.functional}

    def transitive_relations(self) -> set[str]:
        return {r.name for r in self._rels.values() if r.transitive}

    def symmetric_relations(self) -> set[str]:
        return {r.name for r in self._rels.values() if r.symmetric}

    # -- (de)serialisation -------------------------------------------------
    def to_dict(self) -> dict:
        return {"relations": {n: r.to_dict() for n, r in sorted(self._rels.items())}}

    @classmethod
    def from_dict(cls, d: dict) -> "RelationSchema":
        s = cls(builtin=False)
        for name in sorted(d.get("relations", {})):
            s.register(Relation.from_dict(d["relations"][name]), _custom=False)
        return s

    def merge_custom(self, other: "RelationSchema") -> list[str]:
        """Adopt relations that exist in ``other`` but not here (schema
        migration path for KBs that define their own vocabulary)."""
        added = []
        for r in other:
            if r.name not in self._rels:
                self.register(r)
                added.append(r.name)
        return added


DEFAULT_SCHEMA = RelationSchema()
