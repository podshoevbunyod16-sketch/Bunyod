"""TEST A / B / C — teaching, alias semantics and the typed relation schema.

Spec §3: ``Ньютон — это Исаак Ньютон`` must become ``alias_of``, never
``is_a`` / ``instance_of``.  ``Исаак Ньютон — это физик`` must become
``instance_of`` with ``физик`` registered as a CLASS.
"""
from __future__ import annotations

import pytest

from rasm_cim_v16.knowledge_graph import POL_FALSE, POL_TRUE
from rasm_cim_v16.relations import (KIND_CLASS, KIND_PERSON, Relation,
                                    RelationSchema)


# ---------------------------------------------------------------- TEST A
def test_a_teach_stores_fact(dm):
    a = dm.say("teach Исаак Ньютон — это физик")
    assert a.status == "DIRECT"
    st = dm.kg.all_statements()
    assert len(st) == 1
    assert st[0].predicate == "instance_of"
    assert dm.kg.name_of(st[0].subject) == "Исаак Ньютон"
    assert dm.kg.name_of(st[0].object) == "физик"
    assert st[0].polarity == POL_TRUE
    assert st[0].confidence >= 0.8


def test_a_teach_without_the_verb_also_works(dm):
    a = dm.say("Исаак Ньютон — это физик")
    assert a.status == "DIRECT"
    assert dm.kg.all_statements()[0].predicate == "instance_of"


def test_a_teach_records_evidence_and_provenance(dm):
    dm.say("teach Исаак Ньютон открыл закон всемирного тяготения")
    st = dm.kg.all_statements()[0]
    ev = dm.evidence.for_statement(st.statement_id)
    assert len(ev) == 1
    assert ev[0].utterance == "Исаак Ньютон открыл закон всемирного тяготения"
    assert st.provenance["utterance"] == "Исаак Ньютон открыл закон всемирного тяготения"
    assert st.source.startswith("turn:")


# ---------------------------------------------------------------- TEST B
def test_b_alias_not_is_a(dm):
    dm.say("teach Исаак Ньютон — это физик")          # makes физик a CLASS
    a = dm.say("teach Ньютон — это Исаак Ньютон")
    assert a.status == "DIRECT"
    sts = dm.kg.all_statements()
    alias = [s for s in sts if s.predicate == "alias_of"]
    assert len(alias) == 1, "an alias must be created"
    assert dm.kg.name_of(alias[0].subject) == "Ньютон"
    assert dm.kg.name_of(alias[0].object) == "Исаак Ньютон"
    assert not [s for s in sts if s.predicate == "is_a"], "is_a must not exist"
    assert not [s for s in sts
                if s.predicate == "instance_of"
                and dm.kg.name_of(s.subject) == "Ньютон"], \
        "Ньютон must NOT be an instance_of Исаак Ньютон"
    assert "alias_of" in a.text or "известен" in a.text


def test_b_alias_direction_is_surface_to_canonical(dm):
    dm.say("teach Ньютон — это Исаак Ньютон")
    st = dm.kg.all_statements()[0]
    assert st.predicate == "alias_of"
    assert len(dm.kg.name_of(st.subject)) < len(dm.kg.name_of(st.object))


def test_b_reverse_order_is_still_alias(dm):
    dm.say("teach Исаак Ньютон — это Ньютон")
    st = dm.kg.all_statements()[0]
    assert st.predicate == "alias_of"


def test_b_two_proper_names_are_not_merged_silently(dm):
    """Two proper names with no name evidence: refuse and ask (spec §4:
    "Не объединяй сущности только потому, что их названия похожи")."""
    a = dm.say("teach Альберт Эйнштейн — это Исаак Ньютон")
    assert a.status == "AMBIGUOUS"
    assert a.clarification, "the system must ask instead of guessing"
    assert dm.kg.all_statements() == [], "nothing may be asserted on a guess"
    assert dm.context.pending_clarification, "the question must be remembered"


def test_b_same_entity_can_be_declared_explicitly(dm):
    dm.say("teach Альберт Эйнштейн — это Исаак Ньютон")   # refused, asked
    assert dm.command("merge", "Альберт Эйнштейн | Исаак Ньютон").status == "DIRECT"
    sts = dm.kg.all_statements()
    assert [s.predicate for s in sts] == ["same_as"]


# ---------------------------------------------------------------- TEST C
def test_c_instance_of_sets_class_kind(dm):
    dm.say("teach Исаак Ньютон — это физик")
    st = dm.kg.all_statements()[0]
    assert dm.kg.resolver.get(st.object).kind == KIND_CLASS


def test_c_agent_relation_sets_person_kind(dm):
    dm.say("teach Исаак Ньютон открыл закон всемирного тяготения")
    st = dm.kg.all_statements()[0]
    assert dm.kg.resolver.get(st.subject).kind == KIND_PERSON


def test_c_class_to_class_becomes_subclass_of(dm):
    dm.say("teach Исаак Ньютон — это физик")     # физик becomes a CLASS
    dm.say("teach Физик — это учёный")
    sts = dm.kg.all_statements()
    sub = [s for s in sts if s.predicate == "subclass_of"]
    assert len(sub) == 1
    assert dm.kg.name_of(sub[0].subject) == "физик"
    assert dm.kg.name_of(sub[0].object) == "ученый"
    assert not [s for s in sts if s.predicate == "instance_of"
                and dm.kg.name_of(s.subject) == "физик"]


def test_c_entities_are_linked_by_real_statements(dm, taught):
    newton = dm.kg.resolver.resolve("Исаак Ньютон").entity_id
    phys = dm.kg.resolver.resolve("физик").entity_id
    assert dm.kg.polarity_of(newton, "instance_of", phys) == "TRUE"
    assert dm.kg.polarity_of(newton, "instance_of", phys) != "UNKNOWN"


# ------------------------------------------------------------------ schema
def test_schema_has_all_required_relations():
    s = RelationSchema()
    required = {"same_as", "alias_of", "instance_of", "subclass_of", "part_of",
                "has_property", "discovered", "formulated", "created", "wrote",
                "located_in", "causes", "related_to"}
    assert required <= set(s.names())


def test_schema_relations_are_strictly_typed():
    s = RelationSchema()
    assert s.get("subclass_of").subj_types == (KIND_CLASS,)
    assert s.get("subclass_of").obj_types == (KIND_CLASS,)
    assert s.get("subclass_of").transitive is True
    assert s.get("instance_of").transitive is False
    assert s.get("instance_of").obj_types == (KIND_CLASS,)
    assert s.get("alias_of").inverse == "has_alias"
    assert s.get("same_as").symmetric is True
    assert s.get("has_gender").functional is True
    assert s.get("related_to").symmetric is True


def test_schema_type_check():
    r = Relation(name="r", subj_types=(KIND_PERSON,), obj_types=(KIND_CLASS,))
    assert r.allows(KIND_PERSON, KIND_CLASS)
    assert not r.allows(KIND_CLASS, KIND_CLASS)
    assert r.allows("", ""), "unknown kinds are unconstrained, never rejected"


def test_schema_is_extensible_without_touching_the_core():
    s = RelationSchema()
    rel = s.register_dict({
        "name": "teaches_at", "human": "преподаёт в",
        "subj_types": [KIND_PERSON], "obj_types": ["ORG"],
        "surface": ["преподаёт", "преподавал", "преподавала"],
        "question_object": ["где"], "question_subject": ["кто"],
        "template": "{s} преподаёт в {o}",
    })
    assert rel.name in s
    assert s.relation_for_surface("преподавал") == "teaches_at"
    assert s.get("teaches_at").template == "{s} преподаёт в {o}"
    assert "teaches_at" in s.custom_names()


def test_schema_custom_relation_survives_serialisation():
    s = RelationSchema()
    s.register_dict({"name": "x_rel", "human": "икс", "template": "{s} x {o}"})
    s2 = RelationSchema.from_dict(s.to_dict())
    assert s2.get("x_rel").human == "икс"
    assert s2.get("x_rel").template == "{s} x {o}"
    assert set(s2.names()) == set(s.names())


def test_unknown_relation_is_rejected_by_the_graph(dm):
    e = dm.kg.resolver.intern("А")
    o = dm.kg.resolver.intern("Б")
    with pytest.raises(KeyError):
        dm.kg.assert_triple(e.entity_id, "no_such_relation", o.entity_id)


def test_type_violation_is_weakened_not_hidden(dm):
    a = dm.kg.resolver.intern("физик", kind=KIND_CLASS)
    b = dm.kg.resolver.intern("учёный", kind=KIND_CLASS)
    st, _ = dm.kg.assert_triple(a.entity_id, "discovered", b.entity_id,
                                confidence=1.0, when="t")
    assert st.confidence < 1.0
    assert dm.kg.type_warnings, "the violation must be visible"


# ------------------------------------------------------------ other verbs
@pytest.mark.parametrize("text,pred,subj,obj", [
    ("Исаак Ньютон открыл закон всемирного тяготения", "discovered",
     "Исаак Ньютон", "закон всемирного тяготения"),
    ("Исаак Ньютон сформулировал законы движения", "formulated",
     "Исаак Ньютон", "законы движения"),
    ("Толстой написал Войну и мир", "wrote", "Толстой", "Войну и мир"),
    ("Башня находится в Париже", "located_in", "Башня", "Париже"),
    ("Исаак Ньютон известен благодаря работам по оптике", "known_for",
     "Исаак Ньютон", "работам по оптике"),
])
def test_verb_relations_are_typed(dm, text, pred, subj, obj):
    a = dm.say("teach " + text)
    assert a.status == "DIRECT", a.text
    st = dm.kg.all_statements()[-1]
    assert st.predicate == pred
    assert dm.kg.name_of(st.subject) == subj
    assert dm.kg.name_of(st.object) == obj


def test_passive_surface_maps_to_the_passive_relation(dm):
    dm.say("teach Закон всемирного тяготения открыт Ньютоном")
    st = dm.kg.all_statements()[0]
    assert st.predicate == "discovered_by"
    assert dm.kg.name_of(st.subject) == "Закон всемирного тяготения"
    assert dm.kg.name_of(st.object) == "Ньютоном"


def test_literal_values_are_stored_as_literals(dm):
    dm.say("teach Исаак Ньютон родился в 1643 году")
    sts = dm.kg.all_statements()
    lit = [s for s in sts if dm.kg.is_literal(s.object)]
    assert lit, "a year must be stored as a literal, not an entity"
