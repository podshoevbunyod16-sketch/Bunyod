"""TEST D + spec §6 — knowledge graph traversal and bounded inference.

Every derived fact must carry its proof chain, its depth and a status that
distinguishes it from a stored statement.  Nothing derived may be written
back into the graph as a confirmed fact.
"""
from __future__ import annotations

import pytest

from rasm_cim_v16.inference_engine import (ST_DIRECT, ST_INFERRED,
                                           InferenceEngine, builtin_rules)
from rasm_cim_v16.knowledge_graph import POL_FALSE, POL_TRUE, KnowledgeGraph


@pytest.fixture
def kg():
    return KnowledgeGraph(clock=lambda: "t")


def ids(kg, name):
    return kg.resolver.intern(name).entity_id


# ------------------------------------------------------------------ TEST D
def test_d_multistep_inference_over_alias(dm):
    for t in ["teach Исаак Ньютон — это физик", "teach Ньютон — это Исаак Ньютон"]:
        dm.say(t)
    a = dm.say("Ньютон — физик?")
    assert a.status == "INFERRED"
    assert a.text.startswith("Да")
    fact = a.facts[0]
    assert fact.status == ST_INFERRED
    assert fact.depth == 1
    assert "alias_bridge" in fact.rules_used()
    assert len(fact.proof) == 2, "two pieces of evidence are required"
    assert all(step.get("statement_id") for step in fact.proof)


def test_d_inference_through_two_taxonomy_hops(dm):
    for t in ["teach Исаак Ньютон — это физик",
              "teach Ньютон — это Исаак Ньютон",
              "teach Физик — это учёный",
              "teach Учёный — это человек"]:
        dm.say(t)
    a = dm.say("Ньютон — человек?")
    assert a.status == "INFERRED"
    fact = a.facts[0]
    assert fact.depth == 3
    assert fact.rules_used().count("instance_subclass") == 2
    assert "alias_bridge" in fact.rules_used()


def test_d_confidence_decays_with_depth(dm):
    for t in ["teach Исаак Ньютон — это физик",
              "teach Ньютон — это Исаак Ньютон",
              "teach Физик — это учёный",
              "teach Учёный — это человек"]:
        dm.say(t)
    phys = dm.say("Ньютон — физик?").facts[0]
    man = dm.say("Ньютон — человек?").facts[0]
    assert phys.confidence > man.confidence
    assert man.confidence < 0.9


# --------------------------------------------------------------- inverse
def test_inverse_relation_answers_the_passive_question(dm):
    dm.say("teach Закон всемирного тяготения открыт Ньютоном")
    a = dm.say("Кто открыл закон всемирного тяготения?")
    assert a.status == "INFERRED"
    assert "Ньютон" in a.text


def test_inverse_both_directions(dm):
    dm.say("teach Исаак Ньютон открыл закон всемирного тяготения")
    a = dm.say("Кем открыт закон всемирного тяготения?")
    assert "Ньютон" in a.text


# ------------------------------------------------------------- symmetric
def test_symmetric_relation(kg):
    inf = InferenceEngine(kg)
    a, b = ids(kg, "A"), ids(kg, "B")
    kg.assert_triple(a, "same_as", b, when="t")
    facts = inf.query(subject=b, predicate="same_as", obj=a)
    assert facts and facts[0].status == ST_INFERRED


def test_asymmetric_relation_is_not_reversed(kg):
    inf = InferenceEngine(kg)
    a, b = ids(kg, "Исаак Ньютон"), ids(kg, "закон")
    kg.assert_triple(a, "discovered", b, when="t")
    assert inf.query(subject=b, predicate="discovered", obj=a) == []


# ------------------------------------------------------------- transitive
def test_transitive_part_of(kg):
    inf = InferenceEngine(kg)
    a, b, c = ids(kg, "палец"), ids(kg, "кисть"), ids(kg, "рука")
    kg.assert_triple(a, "part_of", b, when="t")
    kg.assert_triple(b, "part_of", c, when="t")
    facts = inf.query(subject=a, predicate="part_of", obj=c)
    assert facts
    assert facts[0].status == ST_INFERRED
    assert facts[0].depth >= 1


def test_transitivity_respects_the_depth_cap(kg):
    inf = InferenceEngine(kg, max_depth=1)
    nodes = [ids(kg, f"N{i}") for i in range(5)]
    for x, y in zip(nodes, nodes[1:]):
        kg.assert_triple(x, "part_of", y, when="t")
    # 4 hops away — unreachable with max_depth=1
    assert inf.query(subject=nodes[0], predicate="part_of", obj=nodes[4]) == []
    inf2 = InferenceEngine(kg, max_depth=4)
    assert inf2.query(subject=nodes[0], predicate="part_of", obj=nodes[4])


def test_non_transitive_relation_is_not_chained(kg):
    inf = InferenceEngine(kg)
    a, b, c = ids(kg, "Исаак Ньютон"), ids(kg, "физик"), ids(kg, "учёный")
    kg.assert_triple(a, "instance_of", b, when="t")
    kg.assert_triple(b, "instance_of", c, when="t")
    # instance_of is NOT transitive over itself: no rule may chain it
    facts = inf.query(subject=a, predicate="instance_of", obj=c)
    assert facts == [], "instance_of must not chain over instance_of"


# ----------------------------------------------------------------- cycles
def test_cycle_does_not_hang(kg):
    inf = InferenceEngine(kg, max_depth=5, max_nodes=200)
    a, b, c = ids(kg, "A"), ids(kg, "B"), ids(kg, "C")
    kg.assert_triple(a, "part_of", b, when="t")
    kg.assert_triple(b, "part_of", c, when="t")
    kg.assert_triple(c, "part_of", a, when="t")       # cycle
    facts = inf.query(subject=a, predicate="part_of", obj=None)
    assert facts is not None
    assert inf.stats["last_goals"] <= 200


def test_self_loop_is_safe(kg):
    inf = InferenceEngine(kg)
    a = ids(kg, "A")
    kg.assert_triple(a, "part_of", a, when="t")
    assert inf.query(subject=a, predicate="part_of", obj=a)


def test_node_budget_is_respected(kg):
    inf = InferenceEngine(kg, max_depth=4, max_nodes=10)
    nodes = [ids(kg, f"N{i}") for i in range(50)]
    for x, y in zip(nodes, nodes[1:]):
        kg.assert_triple(x, "part_of", y, when="t")
    inf.query(subject=nodes[0], predicate="part_of", obj=None)
    assert inf.stats["last_goals"] <= 11


# ---------------------------------------------------------- direct/inferred
def test_direct_fact_is_marked_direct(kg):
    inf = InferenceEngine(kg)
    a, b = ids(kg, "A"), ids(kg, "B")
    st, _ = kg.assert_triple(a, "related_to", b, when="t")
    facts = inf.query(subject=a, predicate="related_to", obj=b)
    assert facts[0].status == ST_DIRECT
    assert facts[0].depth == 0
    assert facts[0].statement_id == st.statement_id


def test_inference_never_writes_to_the_graph(kg):
    """Spec §6: derived conclusions must not be persisted as facts."""
    inf = InferenceEngine(kg)
    a, b, c = ids(kg, "Исаак Ньютон"), ids(kg, "физик"), ids(kg, "учёный")
    kg.assert_triple(a, "instance_of", b, when="t")
    kg.assert_triple(b, "subclass_of", c, when="t")
    before = len(kg.all_statements())
    inf.query(subject=a, predicate="instance_of", obj=None)
    assert len(kg.all_statements()) == before
    assert all(s.status == "direct" for s in kg.all_statements())


# ----------------------------------------------------------------- rules
def test_all_builtin_rules_are_present():
    names = {r.name for r in builtin_rules()}
    assert names == {"alias_bridge", "symmetric", "inverse",
                     "instance_subclass", "transitive"}


def test_rules_can_be_disabled(kg):
    a = ids(kg, "Ньютон")
    b = ids(kg, "Исаак Ньютон")
    c = ids(kg, "физик")
    kg.assert_triple(a, "alias_of", b, when="t")
    kg.assert_triple(b, "instance_of", c, when="t")
    inf = InferenceEngine(kg, enabled_rules=[])          # everything off
    assert inf.query(subject=a, predicate="instance_of", obj=c) == []
    inf.enable("alias_bridge")
    assert inf.query(subject=a, predicate="instance_of", obj=c)


def test_rule_report_is_inspectable(kg):
    inf = InferenceEngine(kg)
    rep = inf.rule_report()
    assert len(rep) == 5
    assert all({"name", "description", "confidence", "max_chain", "enabled"}
               <= set(r) for r in rep)


def test_explain_produces_a_readable_chain(dm):
    dm.say("teach Исаак Ньютон — это физик")
    dm.say("teach Ньютон — это Исаак Ньютон")
    a = dm.say("Ньютон — физик?")
    lines = a.explanation
    assert len(lines) >= 2
    assert any("alias_bridge" in l for l in lines)
    assert lines[-1].startswith("=")


# ------------------------------------------------------------- polarity
def test_negated_fact_is_not_returned_as_positive(kg):
    inf = InferenceEngine(kg)
    a, b = ids(kg, "Ньютон"), ids(kg, "химик")
    kg.assert_triple(a, "instance_of", b, polarity=POL_FALSE, when="t")
    assert inf.query(subject=a, predicate="instance_of", obj=b,
                     polarity=POL_TRUE) == []
    assert inf.query(subject=a, predicate="instance_of", obj=b,
                     polarity=POL_FALSE)


def test_yes_no_verdicts(kg):
    inf = InferenceEngine(kg)
    a, b = ids(kg, "A"), ids(kg, "B")
    assert inf.yes_no(a, "related_to", b)[0] == "UNKNOWN"
    kg.assert_triple(a, "related_to", b, when="t")
    assert inf.yes_no(a, "related_to", b)[0] == "YES"
    kg.assert_triple(a, "related_to", b, polarity=POL_FALSE, when="t")
    assert inf.yes_no(a, "related_to", b)[0] == "CONTRADICTED"


# --------------------------------------------------------------- graph
def test_graph_neighbours_and_indexes(kg):
    a, b, c = ids(kg, "A"), ids(kg, "B"), ids(kg, "C")
    kg.assert_triple(a, "discovered", b, when="t")
    kg.assert_triple(a, "wrote", c, when="t")
    assert len(kg.neighbours(a)) == 2
    assert len(kg.objects_of(a, "discovered")) == 1
    assert len(kg.subjects_of("discovered", b)) == 1
    assert len(kg.find(subject=a)) == 2
    assert len(kg.find(predicate="wrote")) == 1


def test_graph_walk_is_bounded(kg):
    nodes = [ids(kg, f"N{i}") for i in range(6)]
    for x, y in zip(nodes, nodes[1:]):
        kg.assert_triple(x, "part_of", y, when="t")
    seen = list(kg.walk(nodes[0], max_depth=2))
    assert seen and len(seen) == 2
    assert all(d <= 2 for _, _, d in seen)


def test_truthgraph9_polarity_oracle_is_reused(kg):
    """The v0.9 exhaustively-tested polarity semantics must be the oracle."""
    a, b = ids(kg, "A"), ids(kg, "B")
    kg.assert_triple(a, "related_to", b, when="t")
    assert kg.polarity_of(a, "related_to", b) == "TRUE"
    kg.assert_triple(a, "related_to", b, polarity=POL_FALSE, when="t")
    assert kg.polarity_of(a, "related_to", b) == "CONFLICT"
    kg.retract(a, "related_to", b, when="t")
    assert kg.polarity_of(a, "related_to", b) == "UNKNOWN"
    assert kg.truth.mode == "symmetric"
