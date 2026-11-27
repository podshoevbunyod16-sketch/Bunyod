"""TEST H / I / J / K + spec §8 — ambiguity, contradictions, negation,
duplicate teaching.
"""
from __future__ import annotations

import pytest

from rasm_cim_v16.contradiction_detector import (K_ALIAS, K_FUNCTIONAL,
                                                 K_POLARITY,
                                                 ContradictionDetector)
from rasm_cim_v16.knowledge_graph import POL_FALSE, POL_TRUE


# ------------------------------------------------------------------ TEST H
def test_h_ambiguous_name_triggers_a_question(dm):
    # no bare «Ньютон» entity is taught on purpose: a name the user registered
    # verbatim wins as an exact match, and only a name that fits two
    # *different* registered entities may be reported as ambiguous
    dm.say("teach Исаак Ньютон открыл закон всемирного тяготения")
    dm.say("teach Исаак Ньютон — это физик")
    dm.say("teach Джон Ньютон — это богослов")
    a = dm.say("Ньютон — физик?")
    assert a.status == "AMBIGUOUS", a.text
    assert a.clarification
    assert "Ньютон" in a.clarification


def test_h_ambiguity_is_remembered_as_pending(dm):
    dm.say("teach Исаак Ньютон открыл закон всемирного тяготения")
    dm.say("teach Исаак Ньютон — это физик")
    dm.say("teach Джон Ньютон — это богослов")
    dm.say("Ньютон — физик?")
    assert dm.context.pending_clarification
    assert dm.context.pending_clarification["kind"] == "entity"


def test_h_two_unrelated_names_do_not_become_one_entity(dm):
    dm.say("teach Исаак Ньютон — это физик")
    dm.say("teach Джон Ньютон — это богослов")
    ents = [e.canonical for e in dm.kg.resolver.all_entities()]
    assert "Исаак Ньютон" in ents and "Джон Ньютон" in ents
    assert len({e.entity_id for e in dm.kg.resolver.all_entities()
                if e.canonical.endswith("Ньютон")}) >= 2


def test_h_a_question_the_system_can_answer_is_not_deflected(dm):
    dm.say("teach Исаак Ньютон — это физик")
    a = dm.say("Исаак Ньютон — физик?")
    assert a.status == "DIRECT"


# ------------------------------------------------------------------ TEST I
def test_i_contradiction_is_detected_and_both_sides_kept(dm):
    dm.say("teach Исаак Ньютон — это физик")
    dm.say("teach Исаак Ньютон не был физиком")
    assert len(dm.kg.all_statements()) == 2
    cs = dm.detector.conflicts()
    assert len(cs) == 1
    assert cs[0].kind == K_POLARITY
    assert len(cs[0].statement_ids) == 2
    assert len(cs[0].sources) == 2
    assert cs[0].resolution == ""


def test_i_contradiction_changes_the_answer_status(dm):
    dm.say("teach Исаак Ньютон — это физик")
    dm.say("teach Исаак Ньютон не был физиком")
    a = dm.say("Ньютон — физик?")
    assert a.status == "CONTRADICTED"
    assert "Противоречие" in a.text


def test_i_no_silent_deletion_on_contradiction(dm):
    dm.say("teach Исаак Ньютон — это физик")
    first_id = dm.kg.all_statements()[0].statement_id
    dm.say("teach Исаак Ньютон не был физиком")
    ids = {s.statement_id for s in dm.kg.all_statements()}
    assert first_id in ids
    assert len(ids) == 2


def test_i_functional_conflict(dm):
    a = dm.kg.resolver.intern("Исаак Ньютон")
    m, f = dm.kg.resolver.intern("masculine"), dm.kg.resolver.intern("feminine")
    dm.kg.assert_triple(a.entity_id, "has_gender", m.entity_id, when="t")
    dm.kg.assert_triple(a.entity_id, "has_gender", f.entity_id, when="t")
    cs = dm.detector.scan()
    assert any(c.kind == K_FUNCTIONAL for c in cs)


def test_i_disjoint_class_conflict(dm):
    dm.detector.declare_disjoint("физик", "химик")
    dm.say("teach Исаак Ньютон — это физик")
    dm.say("teach Исаак Ньютон — это химик")
    assert any(c.kind == "DISJOINT_CLASS" for c in dm.detector.conflicts())


def test_i_alias_divergence_conflict(dm):
    x = dm.kg.resolver.intern("Ньютон")
    a = dm.kg.resolver.intern("Исаак Ньютон")
    b = dm.kg.resolver.intern("Джон Ньютон")
    dm.kg.assert_triple(x.entity_id, "alias_of", a.entity_id, when="t")
    dm.kg.assert_triple(x.entity_id, "alias_of", b.entity_id, when="t")
    assert any(c.kind == K_ALIAS for c in dm.detector.scan())


def test_i_alias_divergence_resolved_by_same_as(dm):
    x = dm.kg.resolver.intern("Ньютон")
    a = dm.kg.resolver.intern("Исаак Ньютон")
    b = dm.kg.resolver.intern("сэр Исаак Ньютон")
    dm.kg.assert_triple(x.entity_id, "alias_of", a.entity_id, when="t")
    dm.kg.assert_triple(x.entity_id, "alias_of", b.entity_id, when="t")
    dm.kg.assert_triple(a.entity_id, "same_as", b.entity_id, when="t")
    assert not [c for c in dm.detector.scan() if c.kind == K_ALIAS]


def test_i_taxonomy_cycle_is_reported(dm):
    dm.say("teach Физик — это учёный")
    dm.say("teach Учёный — это человек")
    dm.say("teach Человек — это физик")
    assert any(c.kind == "TAXONOMY_CYCLE" for c in dm.detector.conflicts())


def test_i_conflicts_can_be_resolved_explicitly(dm):
    dm.say("teach Исаак Ньютон — это физик")
    dm.say("teach Исаак Ньютон не был физиком")
    c = dm.detector.conflicts()[0]
    assert dm.detector.resolve(c.conflict_id, "user confirmed the positive fact")
    assert dm.detector.unresolved() == []


def test_i_scan_is_idempotent(dm):
    dm.say("teach Исаак Ньютон — это физик")
    dm.say("teach Исаак Ньютон не был физиком")
    # the conflict is detected while teaching, so a later scan finds nothing
    # NEW — that is the idempotency property under test, and the conflict
    # itself must survive the re-scan rather than being dropped
    before = len(dm.detector.conflicts())
    assert before == 1, [c.kind for c in dm.detector.conflicts()]
    assert dm.detector.scan() == [], "the same conflict must not duplicate"
    assert len(dm.detector.conflicts()) == before


# ------------------------------------------------------------------ TEST J
def test_j_negation_does_not_become_a_positive_fact(dm):
    dm.say("teach Ньютон не был физиком")
    sts = dm.kg.all_statements()
    assert len(sts) == 1
    assert sts[0].polarity == POL_FALSE
    a = dm.say("Ньютон — физик?")
    assert a.text.startswith("Нет"), a.text
    assert a.status != "DIRECT" or "Неверно" in a.text or a.text.startswith("Нет")


def test_j_negation_is_rendered_as_a_denial(dm):
    dm.say("teach Ньютон не был физиком")
    a = dm.say("teach Ньютон не был физиком")     # reinforce, read the text
    dm2_text = dm.say("Ньютон — физик?").text
    assert "Неверно" in dm2_text or dm2_text.startswith("Нет")


def test_j_negated_and_positive_together_is_a_conflict_not_a_choice(dm):
    dm.say("teach Ньютон — это физик")
    dm.say("teach Ньютон не был физиком")
    a = dm.say("Ньютон — физик?")
    assert a.status == "CONTRADICTED"


def test_j_negation_of_an_event(dm):
    dm.say("teach Исаак Ньютон не открыл теорию относительности")
    sts = dm.kg.all_statements()
    assert sts[0].predicate == "discovered"
    assert sts[0].polarity == POL_FALSE
    a = dm.say("Что открыл Исаак Ньютон?")
    assert "теорию относительности" not in a.text


# ------------------------------------------------------------------ TEST K
def test_k_repeated_teaching_does_not_duplicate(dm):
    for _ in range(5):
        dm.say("teach Исаак Ньютон — это физик")
    sts = dm.kg.all_statements()
    assert len(sts) == 1
    assert sts[0].reinforce == 5


def test_k_repeated_teaching_is_reported_as_known(dm):
    dm.say("teach Исаак Ньютон — это физик")
    a = dm.say("teach Исаак Ньютон — это физик")
    assert "уже известно" in a.text
    assert "st-000001" in a.text


def test_k_repeated_teaching_keeps_the_first_timestamp(dm):
    dm.say("teach Исаак Ньютон — это физик")
    st = dm.kg.all_statements()[0]
    created = st.created_at
    dm.say("teach Исаак Ньютон — это физик")
    st = dm.kg.all_statements()[0]
    assert st.created_at == created
    assert st.modified_at != created
    assert any("duplicate" in h["change"] for h in st.history)


def test_k_evidence_is_not_duplicated(dm):
    dm.say("teach Исаак Ньютон — это физик")
    st = dm.kg.all_statements()[0]
    dm.evidence.add(st.statement_id, utterance="Исаак Ньютон — это физик",
                    turn_id="u-00001", when="t")
    assert len(dm.evidence.for_statement(st.statement_id)) == 1
    dm.evidence.add(st.statement_id, utterance="Исаак Ньютон — это физик",
                    turn_id="u-00001", when="t")
    assert len(dm.evidence.for_statement(st.statement_id)) == 1


def test_k_memory_level_duplicate_detection(dm):
    for _ in range(3):
        dm.memory.record(kind="fact", tier="working",
                         payload={"same": "payload"}, source="test")
    assert len(dm.memory.working()) == 1
    assert dm.memory.duplicate_hits == 2


# -------------------------------------------------------------- retraction
def test_explicit_retract_keeps_history(dm):
    dm.say("teach Исаак Ньютон — это физик")
    st = dm.kg.all_statements()[0]
    a = dm.say("забудь Исаак Ньютон")
    assert a.status == "DIRECT"
    assert dm.kg.all_statements() == []
    kept = dm.kg.statement(st.statement_id)
    assert kept.status == "retracted"
    assert any(h["change"] == "retracted" for h in kept.history)
    b = dm.say("Исаак Ньютон — физик?")
    assert b.status == "UNKNOWN"


def test_retract_of_an_unknown_entity_is_honest(dm):
    a = dm.say("забудь Никого Такого")
    assert a.status == "UNKNOWN"
    assert "Не записал" in a.text or "не" in a.text.lower()
