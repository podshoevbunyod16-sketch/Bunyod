"""Plan A — the importer: dry run, provenance, idempotency, rollback.

These tests are the acceptance criteria of Plan A written as assertions:
nothing is written during a dry run, a record is never silently dropped, a
second run changes nothing, and one command restores the pre-import state.
"""
from __future__ import annotations

import json
import os

import pytest

from conftest import (V11_FACTS, legacy_entity, legacy_statement,
                      write_learned, write_legacy_kb)

from rasm_cim_v16.dialogue_manager import DialogueManager
from rasm_cim_v16.persistence import Store

from rasm_cim_v17.interop import apply_plan, build_plan, plan_from_artifact
from rasm_cim_v17.interop.importer import (LEDGER_FILENAME, ImportError_,
                                           ImportLedger, restore)
from rasm_cim_v17.interop.learned import LearnedSnapshot
from rasm_cim_v17.routing import RoutingStore


# --------------------------------------------------------------------------
# dry run
# --------------------------------------------------------------------------

def test_dry_run_writes_nothing(legacy_kb, home, dm, store):
    plan = build_plan(legacy_kb)
    assert plan.summary()["statements_importable"] == 4
    assert not os.path.exists(store.kb_path)
    assert not os.path.exists(os.path.join(home, LEDGER_FILENAME))
    assert len(dm.kg.all_statements()) == 0


def test_plan_reports_skips_with_reasons(tmp_path):
    path = write_legacy_kb(
        str(tmp_path / "kb"),
        [legacy_entity("e1", "person", "Иван"),
         legacy_entity("e2", "concept", "человек")],
        [legacy_statement("s1", "e1", "is_a", object_entity="e2"),
         legacy_statement("s2", "e1", "is_a", object_entity="e2",
                          kind="hypothesis"),
         legacy_statement("s3", "e1", "неизвестное_отношение",
                          object_literal="что-то")])
    plan = build_plan(path)
    assert len(plan.importable) == 1
    reasons = plan.skip_reasons()
    assert reasons["statement kind 'hypothesis' is not imported"] == 1
    assert reasons["unmapped predicate 'неизвестное_отношение'"] == 1
    assert plan.unmapped_predicates() == {"неизвестное_отношение": 1}
    text = plan.render()
    assert "пропуск 2" in text and "НЕИЗВЕСТНЫЕ ПРЕДИКАТЫ" in text


def test_plan_is_deterministic(legacy_kb):
    a, b = build_plan(legacy_kb), build_plan(legacy_kb)
    assert a.digest() == b.digest()
    assert a.summary() == b.summary()


def test_plan_detects_conflicts_before_writing(tmp_path):
    path = write_legacy_kb(
        str(tmp_path / "kb"),
        [legacy_entity("e1", "place", "Франция"),
         legacy_entity("e2", "place", "Париж"),
         legacy_entity("e3", "place", "Лион")],
        [legacy_statement("s1", "e1", "capital_of", object_entity="e2"),
         legacy_statement("s2", "e1", "capital_of", object_entity="e3")])
    plan = build_plan(path)
    conflicts = [w for w in plan.warnings if w["kind"] == "planned_conflict"]
    assert conflicts, "a functional relation with two objects must be flagged"
    assert "Париж" in conflicts[0]["detail"]


def test_boolean_property_is_planned_as_has_property(legacy_kb):
    plan = build_plan(legacy_kb)
    st = [s for s in plan.statements if s.legacy_predicate == "может_летать"][0]
    assert st.predicate == "has_property"
    assert st.object_name == "может летать"
    assert st.object_kind == "literal"
    assert st.polarity == "false"          # да/нет flipped by the legacy row


def test_is_a_between_identical_surfaces_becomes_alias(tmp_path):
    path = write_legacy_kb(
        str(tmp_path / "kb"),
        [legacy_entity("e1", "person", "Ньютон")],
        [legacy_statement("s1", "e1", "is_a", object_literal=" ньютон ")])
    plan = build_plan(path)
    assert plan.statements[0].predicate == "alias_of"
    assert "alias_of" in plan.statements[0].note


# --------------------------------------------------------------------------
# apply
# --------------------------------------------------------------------------

def test_apply_imports_statements_and_evidence(legacy_kb, dm, store):
    result = apply_plan(build_plan(legacy_kb), dm, store=store)
    assert result.applied and result.duplicate_of == ""
    assert result.statements_added == 4
    assert result.statements_after == 4
    # 4 named entities + 2 literals interned as entities ('может летать',
    # '100 цельсия') — v0.16 represents literals as entities of kind LITERAL
    assert result.entities_after == 6
    assert len(dm.evidence) == 4
    assert os.path.exists(store.kb_path)
    assert os.path.exists(os.path.join(store.home, LEDGER_FILENAME))


def test_imported_statement_keeps_legacy_provenance(legacy_kb, dm, store):
    apply_plan(build_plan(legacy_kb), dm, store=store)
    st = dm.kg.find(predicate="capital_of")[0]
    assert st.provenance["legacy_statement_id"] == "st_1"
    assert st.provenance["legacy"]["provenance"]["doc_id"] == "doc:st_1"
    assert st.provenance["legacy_schema"] == "rasm-cim-kb/1"
    assert st.source.startswith("import:rasm-cim-kb/1:")
    assert st.confidence == pytest.approx(0.95)     # reliability high


def test_custom_relation_is_registered_and_persisted(legacy_kb, dm, store):
    result = apply_plan(build_plan(legacy_kb), dm, store=store)
    assert "capital_of" in result.relations_registered
    assert "capital_of" in dm.kg.schema
    assert "boils_at" in dm.kg.schema


def test_imported_facts_are_queryable(legacy_kb, dm, store):
    apply_plan(build_plan(legacy_kb), dm, store=store)
    ans = dm.say("Какая столица у Франция?")
    answered = [f for f in ans.facts
                if dm.kg.name_of(f.object) == "Париж"]
    assert answered, f"ответ не опирается на импорт: {ans.text!r}"
    for f in answered:
        assert any(e.statement_id == f.statement_id
                   for e in dm.evidence.all()), \
            f"ответ без записи об источнике: {f.statement_id}"


def test_second_apply_is_a_no_op(legacy_kb, dm, store):
    plan = build_plan(legacy_kb)
    first = apply_plan(plan, dm, store=store)
    second = apply_plan(build_plan(legacy_kb), dm, store=store)
    assert first.applied and not second.applied
    assert second.duplicate_of == first.ledger_id
    assert len(dm.kg.all_statements()) == first.statements_after
    assert len(ImportLedger(store.home).entries()) == 1


def test_reimport_after_source_touch_adds_no_statements(legacy_kb, dm, store):
    """Different sha256, same content: the graph dedupe still holds."""
    apply_plan(build_plan(legacy_kb), dm, store=store)
    before = len(dm.kg.all_statements())
    with open(os.path.join(legacy_kb, "events.jsonl"), "a", encoding="utf-8") as f:
        f.write("\n")                  # changes the hash, not the content
    plan2 = build_plan(legacy_kb)
    result = apply_plan(plan2, dm, store=store)
    assert result.applied
    assert result.statements_added == 0
    assert result.statements_reinforced == 4
    assert len(dm.kg.all_statements()) == before


def test_missing_predicate_never_reaches_the_graph(tmp_path, dm, store):
    path = write_legacy_kb(
        str(tmp_path / "kb"),
        [legacy_entity("e1", "person", "Иван")],
        [legacy_statement("s1", "e1", "загадочное_отношение",
                          object_literal="нечто")])
    result = apply_plan(build_plan(path), dm, store=store)
    assert result.statements_added == 0
    assert result.statements_skipped == 1
    assert dm.kg.all_statements() == []


def test_import_is_saved_and_reloadable(legacy_kb, dm, store):
    apply_plan(build_plan(legacy_kb), dm, store=store)
    fresh = DialogueManager()
    fresh_store = Store(home=store.home)
    fresh_store.attach(fresh)
    fresh_store.load(fresh)
    assert len(fresh.kg.all_statements()) == 4
    assert "capital_of" in fresh.kg.schema        # custom relations survive
    assert len(fresh.evidence) == 4


def test_ledger_records_plan_and_result(legacy_kb, dm, store):
    result = apply_plan(build_plan(legacy_kb), dm, store=store)
    entry = ImportLedger(store.home).get(result.ledger_id)
    assert entry["status"] == "applied"
    assert entry["source_sha256"] == build_plan(legacy_kb).source_sha256
    assert entry["plan"]["statements_importable"] == 4
    assert entry["result"]["statements_added"] == 4
    assert entry["backup"] == {} or "kb" in entry["backup"]


# --------------------------------------------------------------------------
# rollback
# --------------------------------------------------------------------------

def test_restore_returns_to_the_pre_import_state(legacy_kb, dm, store):
    dm.say("teach Иван — это человек")
    store.save(dm)
    before = len(dm.kg.all_statements())
    result = apply_plan(build_plan(legacy_kb), dm, store=store)
    assert len(dm.kg.all_statements()) > before
    assert result.backup.get("kb")

    restored = restore(result.ledger_id, store)
    assert "kb" in restored
    fresh = DialogueManager()
    s2 = Store(home=store.home)
    s2.attach(fresh)
    s2.load(fresh)
    assert len(fresh.kg.all_statements()) == before
    assert fresh.kg.find(predicate="capital_of") == []
    entry = ImportLedger(store.home).get(result.ledger_id)
    assert entry["status"] == "restored"


def test_restore_unknown_ledger_id_raises(store):
    with pytest.raises(ImportError_, match="unknown ledger id"):
        restore("imp_9999_deadbeef", store)


def test_restore_without_backup_raises(tmp_path, dm):
    """An import into a fresh store has no previous document to back up."""
    path = write_legacy_kb(
        str(tmp_path / "kb"), [legacy_entity("e1", "person", "Иван")],
        [legacy_statement("s1", "e1", "is_a", object_literal="человек")])
    store = Store(home=str(tmp_path / "home2"))
    plan = build_plan(path)
    result = apply_plan(plan, dm, store=store)
    assert result.backup == {}
    with pytest.raises(ImportError_, match="no pre-import backup"):
        restore(result.ledger_id, store)


# --------------------------------------------------------------------------
# real v0.11 corpus end to end
# --------------------------------------------------------------------------

def test_v11_corpus_imports_end_to_end(dm, store):
    plan = build_plan(V11_FACTS, kind="facts")
    assert plan.summary()["statements_importable"] == 34
    assert plan.unmapped_predicates() == {}
    result = apply_plan(plan, dm, store=store)
    assert result.statements_added == 34
    assert "capital_of" in dm.kg.schema
    ans = dm.say("Где столица Франция?")
    paris = [f for f in ans.facts if dm.kg.name_of(f.object) == "Париж"]
    assert paris, ans.text


# --------------------------------------------------------------------------
# learned artifact
# --------------------------------------------------------------------------

def test_artifact_plan_carries_hints_and_aliases(tmp_path):
    d = write_learned(str(tmp_path / "learned"),
                      predicate_hints={"где столица франция?": "capital_of"},
                      alias_delta={"французская республика": "Франция"})
    plan = plan_from_artifact(LearnedSnapshot.load(d))
    assert plan.summary()["kind"] == "learned"
    assert plan.summary()["predicate_hints"] == 1
    assert plan.summary()["alias_delta"] == 1
    assert plan.statements == []            # an artifact never carries facts


def test_unverified_artifact_contributes_nothing(tmp_path):
    d = write_learned(str(tmp_path / "learned"),
                      predicate_hints={"q?": "capital_of"}, tamper_sha=True)
    plan = plan_from_artifact(LearnedSnapshot.load(d))
    assert plan.predicate_hints == {} and plan.alias_delta == {}
    assert any(w["kind"] == "artifact_unverified" for w in plan.warnings)


def test_artifact_import_writes_routing_and_links_aliases(tmp_path, dm, store):
    dm.say("teach Франция — это государство")
    apply_plan(build_plan(legacy_kb_path(tmp_path)), dm, store=store)
    d = write_learned(str(tmp_path / "learned"),
                      predicate_hints={"где столица франция?": "capital_of"},
                      alias_delta={"французская республика": "Франция"})
    result = apply_plan(build_plan(d), dm, store=store)
    assert result.predicate_hints_written == 1
    assert RoutingStore(store.home).hint_for("  Где столица Франция? ") == \
        "capital_of"
    france = dm.kg.resolver.resolve("Франция", allow_fuzzy=False)
    alias = dm.kg.resolver.resolve("Французская республика", allow_fuzzy=False)
    assert alias.entity_id == france.entity_id


def legacy_kb_path(tmp_path) -> str:
    return write_legacy_kb(
        str(tmp_path / "kb2"),
        [legacy_entity("e1", "place", "Франция"),
         legacy_entity("e2", "place", "Париж")],
        [legacy_statement("s1", "e1", "capital_of", object_entity="e2")])


def test_artifact_import_is_idempotent(tmp_path, dm, store):
    d = write_learned(str(tmp_path / "learned"),
                      predicate_hints={"где столица?": "capital_of"})
    first = apply_plan(build_plan(d), dm, store=store)
    second = apply_plan(build_plan(d), dm, store=store)
    assert first.applied and not second.applied
    assert len(RoutingStore(store.home).hints()) == 1


# --------------------------------------------------------------------------
# robustness
# --------------------------------------------------------------------------

def test_unknown_kind_is_refused(legacy_kb):
    with pytest.raises(ImportError_, match="unknown import kind"):
        build_plan(legacy_kb, kind="telepathy")


def test_sniffing_detects_a_facts_file(tmp_path):
    p = tmp_path / "rows.jsonl"
    p.write_text('{"subject": "A", "relation": "is_a", "object": "B"}\n',
                 encoding="utf-8")
    assert build_plan(str(p)).kind == "facts"


def test_sniffing_detects_an_event_log(tmp_path):
    p = tmp_path / "events.jsonl"
    p.write_text('{"event": "entity_defined", "entity": {"id": "e1", '
                 '"type": "person", "name": "Иван"}}\n', encoding="utf-8")
    assert build_plan(str(p)).kind == "kb"
