"""v0.12 KB core tests (fresh tmp store per test)."""
import os

import pytest

from rasm_cim_v12.knowledge import (KnowledgeError, KnowledgeStore, Provenance,
                                     confirm_hypothesis, export_jsonl, import_jsonl,
                                     retract, teach, teach_hypothesis, update_statement)


@pytest.fixture()
def kb(tmp_path):
    return KnowledgeStore.init(str(tmp_path / "kb"))


def test_teach_creates_entities_and_statement(kb):
    r = teach(kb, "Франция", "столица", "Париж",
              subject_type="place", object_type="place",
              provenance=Provenance(source="user_teach", doc_id="doc-1",
                                    fragment="Париж — столица Франции",
                                    position=(0, 26), added_at="2026-11-26T00:00:00Z"))
    assert r["statement_id"].startswith("st_")
    assert kb.status_for(r["subject_id"], "capital_of", object_entity=r["object_id"]) == "TRUE"
    stmt = kb.statements[r["statement_id"]]
    assert stmt.provenance["fragment"] == "Париж — столица Франции"
    assert stmt.provenance["position"] == [0, 26]
    assert stmt.kind == "user_assertion"


def test_teach_is_idempotent_with_same_doc_id(kb):
    prov = Provenance(source="user_teach", doc_id="doc-fixed", method="manual",
                      added_at="2026-11-26T00:00:00Z")
    r1 = teach(kb, "Франция", "столица", "Париж", provenance=prov)
    r2 = teach(kb, "Франция", "столица", "Париж", provenance=prov)
    assert r1["statement_id"] == r2["statement_id"]
    assert kb.stats()["statements_total"] == 1


def test_unknown_triple_is_unknown(kb):
    teach(kb, "Франция", "столица", "Париж")
    s = teach(kb, "Германия", "столица", "Берлин")
    assert kb.status_for(s["subject_id"], "capital_of", object_entity="ent_nonexistent") == "UNKNOWN"


def test_deny_assert_create_conflict_and_retract_resolves(kb):
    a = teach(kb, "Земля", "плоская", "да", object_is_entity=False, polarity="assert")
    teach(kb, "Земля", "плоская", "да", object_is_entity=False, polarity="deny")
    subj = a["subject_id"]
    assert kb.status_for(subj, "плоская", object_literal="да") == "CONFLICT"
    conflicts = kb.conflicts()
    assert any(c["type"] == "polarity_conflict" for c in conflicts)
    # retracting ONE side leaves the other — never auto-resolved
    deny = [s for s in kb.query_statements(subject=subj, predicate="плоская")
            if s.polarity == "deny"][0]
    retract(kb, deny.id, reason="correct astronomy")
    assert kb.status_for(subj, "плоская", object_literal="да") == "TRUE"


def test_retract_makes_unknown_not_false(kb):
    r = teach(kb, "Киты", "is_a", "рыбы")
    retract(kb, r["statement_id"], reason="correction")
    assert kb.status_for(r["subject_id"], "is_a", object_entity=r["object_id"]) == "UNKNOWN"


def test_functional_predicate_conflict(kb):
    """Two different capitals for one country → CONFLICT (not overwritten)."""
    a = teach(kb, "Австралия", "capital_of", "Канберра", subject_type="place", object_type="place")
    teach(kb, "Австралия", "capital_of", "Сидней", subject_type="place", object_type="place")
    assert kb.status_for(a["subject_id"], "capital_of") == "CONFLICT"
    fc = [c for c in kb.conflicts() if c["type"] == "functional_conflict"]
    assert fc and len(fc[0]["objects"]) == 2


def test_hypothesis_is_not_fact_until_promoted(kb):
    r = teach_hypothesis(kb, "Плутон", "is_a", "планета")
    subj, obj = r["subject_id"], r["object_id"]
    assert kb.status_for(subj, "is_a", object_entity=obj) == "UNKNOWN"  # hypothesis ≠ proof
    assert kb.stats()["hypotheses_active"] == 1
    confirm_hypothesis(kb, r["statement_id"])
    assert kb.status_for(subj, "is_a", object_entity=obj) == "TRUE"


def test_update_is_append_only_with_replace_link(kb):
    r = teach(kb, "Казахстан", "capital_of", "Алма-Ата", subject_type="place", object_type="place")
    out = update_statement(kb, r["statement_id"], object_="Астана")
    assert out["old"] == r["statement_id"] and out["new"] != out["old"]
    assert kb.statements[out["old"]].status == "retracted"
    assert kb.statements[out["old"]].retract_reason.startswith("replaced_by:")
    new_stmt = kb.statements[out["new"]]
    assert new_stmt.replaces == out["old"]
    s = kb.status_for(new_stmt.subject, "capital_of", object_entity=new_stmt.object_entity)
    assert s == "TRUE"


def test_alias_ambiguity_never_auto_merges(kb):
    """One alias, two entities → AMBIGUOUS with candidates; never merged."""
    kb.define_entity("concept", "Мир", aliases=["мир"])          # peace/world-concept
    kb.define_entity("place", "Мир", aliases=["мир"],            # village name
                     provenance=Provenance(source="user_teach", doc_id="doc-2"))
    status, e, cand = kb.find_by_name("Мир")
    assert status == "ambiguous" and len(cand) == 2 and e is None
    status, e, cand = kb.find_by_name("Мир", type_hint="place")
    assert status == "found" and e.type == "place"


def test_provenance_filter_and_reverse_links(kb):
    p1 = Provenance(source="doc:x", doc_id="doc-x", added_at="2026-11-26T00:00:00Z")
    p2 = Provenance(source="user_teach", doc_id="adhoc:1", added_at="2026-11-26T00:00:01Z")
    r1 = teach(kb, "Франция", "столица", "Париж", provenance=p1)
    teach(kb, "Германия", "столица", "Берлин", provenance=p2)
    only_x = kb.query_statements(provenance_source="doc:x")
    assert [s.id for s in only_x] == [r1["statement_id"]]
    paris = [e for e in kb.entities.values() if e.name == "Париж"][0]
    rev = kb.reverse_links(paris.id)
    assert rev and rev[0].predicate == "capital_of"


def test_restart_replays_state(tmp_path):
    path = str(tmp_path / "kb")
    kb = KnowledgeStore.init(path)
    r = teach(kb, "Франция", "столица", "Париж")
    kb2 = KnowledgeStore.init(path)          # fresh open → replay
    assert r["statement_id"] in kb2.statements
    assert kb2.stats()["statements_active"] == 1


def test_torn_last_event_is_ignored_not_fatal(tmp_path):
    path = str(tmp_path / "kb")
    kb = KnowledgeStore.init(path)
    teach(kb, "Франция", "столица", "Париж")
    with open(os.path.join(path, "events.jsonl"), "a", encoding="utf-8") as fh:
        fh.write('{"event": "statement_asserted", "statement": {"id": "st_trunc"')  # torn
    kb2 = KnowledgeStore.init(path)
    assert kb2.stats()["statements_active"] == 1
    assert kb2.replay_torn_tail == 1


def test_conflicting_statements_keep_both_provenance(kb):
    r1 = teach(kb, "Столица Австралии", "equals", "Канберра",
               provenance=Provenance(source="doc:a", doc_id="doc-a"))
    r2 = teach(kb, "Столица Австралии", "equals", "Сидней",
               provenance=Provenance(source="doc:b", doc_id="doc-b"))
    # same subject/predicate, different objects, BOTH kept
    stmts = kb.query_statements(subject=r1["subject_id"], predicate="equals")
    assert len(stmts) == 2
    assert {s.provenance["source"] for s in stmts} == {"doc:a", "doc:b"}


def test_unknown_entity_type_rejected(kb):
    with pytest.raises(KnowledgeError):
        kb.define_entity("alien", "X")
