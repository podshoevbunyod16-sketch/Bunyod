"""Plan A — the legacy readers must not invent, lose or accept junk silently."""
from __future__ import annotations

import json
import os

import pytest

from conftest import (V11_FACTS, legacy_entity, legacy_statement,
                      write_learned, write_legacy_kb)

from rasm_cim_v17.interop.legacy import (KB_SCHEMA, LegacyFacts,
                                         LegacyFormatError, LegacyKB)
from rasm_cim_v17.interop.learned import LearnedFormatError, LearnedSnapshot


# --------------------------------------------------------------------------
# rasm-cim-kb/1
# --------------------------------------------------------------------------

def test_replay_reads_entities_and_statements(legacy_kb):
    kb = LegacyKB.load(legacy_kb)
    assert kb.schema == KB_SCHEMA
    assert set(kb.entities) == {"ent_fr", "ent_paris", "ent_newton", "ent_physicist"}
    assert len(kb.statements) == 4
    assert kb.events_seen == 8          # 4 entities + 4 statements
    assert kb.errors == [] and kb.torn_tail == 0
    assert kb.sha256 and len(kb.sha256) == 64


def test_replay_preserves_provenance_and_reliability(legacy_kb):
    kb = LegacyKB.load(legacy_kb)
    st = kb.statements["st_1"]
    assert st.provenance["source"] == "user_teach"
    assert st.provenance["reliability"] == "high"
    assert st.provenance["fragment"]
    assert st.object_entity == "ent_paris" and st.object_literal is None


def test_torn_last_line_is_skipped_and_counted(tmp_path):
    path = write_legacy_kb(
        str(tmp_path / "kb"),
        [legacy_entity("e1", "person", "Иван")],
        [legacy_statement("s1", "e1", "is_a", object_literal="человек")],
        torn=True)
    kb = LegacyKB.load(path)
    assert kb.torn_tail == 1
    assert len(kb.statements) == 1
    assert kb.errors == []              # a torn tail is not a protocol error


def test_mid_file_bad_line_is_recorded_with_its_line_number(tmp_path):
    path = str(tmp_path / "kb")
    write_legacy_kb(path, [legacy_entity("e1", "person", "Иван")])
    with open(os.path.join(path, "events.jsonl"), "a", encoding="utf-8") as f:
        f.write("{not json}\n")
        f.write(json.dumps({"event": "entity_defined",
                            "entity": legacy_entity("e2", "person", "Пётр")},
                           ensure_ascii=False) + "\n")
    kb = LegacyKB.load(path)
    assert kb.errors and kb.errors[0]["line"] == 2
    assert "e2" in kb.entities        # later good events still replay


def test_unknown_event_kind_is_reported_not_ignored(tmp_path):
    path = write_legacy_kb(
        str(tmp_path / "kb"), [legacy_entity("e1", "person", "Иван")],
        extra_events=[{"event": "entity_merged", "into": "e1"}])
    kb = LegacyKB.load(path)
    assert kb.unknown_events == ["entity_merged"]
    assert kb.errors and "unknown event" in kb.errors[-1]["error"]


def test_retraction_and_promotion_are_replayed(tmp_path):
    path = write_legacy_kb(
        str(tmp_path / "kb"),
        [legacy_entity("e1", "person", "Иван"),
         legacy_entity("e2", "concept", "человек")],
        [legacy_statement("s1", "e1", "is_a", object_entity="e2",
                          status="active"),
         legacy_statement("s2", "e1", "is_a", object_literal="гипотеза",
                          kind="hypothesis", status="retracted")],
        extra_events=[{"event": "statement_retracted", "statement_id": "s1",
                       "reason": "ошибка"},
                      {"event": "hypothesis_promoted", "statement_id": "s2"}])
    kb = LegacyKB.load(path)
    assert kb.statements["s1"].status == "retracted"
    assert kb.statements["s1"].retract_reason == "ошибка"
    assert kb.statements["s2"].kind == "extracted_fact"
    assert [s.id for s in kb.active_statements()] == []


def test_alias_events_are_folded_into_the_entity(tmp_path):
    path = write_legacy_kb(
        str(tmp_path / "kb"), [legacy_entity("e1", "person", "Исаак Ньютон")],
        extra_events=[{"event": "entity_alias_added", "entity_id": "e1",
                       "alias": "Ньютон"}])
    kb = LegacyKB.load(path)
    assert "Ньютон" in kb.entities["e1"].aliases
    assert kb.alias_events == [("e1", "Ньютон")]


def test_alias_for_unknown_entity_is_an_error(tmp_path):
    path = write_legacy_kb(
        str(tmp_path / "kb"), [legacy_entity("e1", "person", "Иван")],
        extra_events=[{"event": "entity_alias_added", "entity_id": "e9",
                       "alias": "Вася"}])
    kb = LegacyKB.load(path)
    assert any("unknown entity" in e["error"] for e in kb.errors)


def test_wrong_schema_is_refused(tmp_path):
    path = str(tmp_path / "kb")
    os.makedirs(path)
    with open(os.path.join(path, "meta.json"), "w", encoding="utf-8") as f:
        json.dump({"schema": "rasm-cim-kb/2", "format": "events-jsonl"}, f)
    with open(os.path.join(path, "events.jsonl"), "w", encoding="utf-8") as f:
        f.write("")
    with pytest.raises(LegacyFormatError, match="unsupported KB schema"):
        LegacyKB.load(path)


def test_missing_meta_is_refused(tmp_path):
    path = str(tmp_path / "not-a-kb")
    os.makedirs(path)
    with pytest.raises(LegacyFormatError, match="meta.json"):
        LegacyKB.load(path)


def test_missing_path_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        LegacyKB.load(str(tmp_path / "nope"))


def test_statement_with_two_objects_is_rejected(tmp_path):
    path = write_legacy_kb(
        str(tmp_path / "kb"),
        [legacy_entity("e1", "person", "Иван"),
         legacy_entity("e2", "concept", "человек")],
        [legacy_statement("s1", "e1", "is_a", object_entity="e2",
                          object_literal="человек")])
    kb = LegacyKB.load(path)
    assert "s1" not in kb.statements
    assert any("both object_entity" in e["error"] for e in kb.errors)


def test_statements_without_object_are_rejected(tmp_path):
    path = write_legacy_kb(
        str(tmp_path / "kb"), [legacy_entity("e1", "person", "Иван")],
        [legacy_statement("s1", "e1", "is_a")])
    kb = LegacyKB.load(path)
    assert kb.statements == {}
    assert any("no object" in e["error"] for e in kb.errors)


def test_summary_counts(legacy_kb):
    s = LegacyKB.load(legacy_kb).summary()
    assert s["entities"] == 4 and s["statements"] == 4
    assert s["active"] == 4 and s["errors"] == 0
    assert s["schema"] == KB_SCHEMA


# --------------------------------------------------------------------------
# v0.11 demo facts
# --------------------------------------------------------------------------

def test_facts_reader_reads_the_real_v11_corpus():
    facts = LegacyFacts.load(V11_FACTS)
    assert len(facts.rows) == 34
    assert facts.errors == []
    assert facts.rows[0]["subject"] == "Франция"
    assert facts.sha256


def test_facts_materialise_into_entities_and_statements():
    kb = LegacyFacts.load(V11_FACTS).as_legacy_kb()
    names = {e.name for e in kb.entities.values()}
    assert "Франция" in names and "Париж" in names
    # two lines about Дракон (assert + deny) produce ONE entity
    dragons = [e for e in kb.entities.values() if e.name == "Дракон"]
    assert len(dragons) == 1
    assert len(kb.statements) == 34
    st = kb.statements["st_facts_0001"]
    assert st.provenance["source"] == "legacy:demo_facts"
    assert st.provenance["legacy_line"] == 1
    assert st.polarity == "true"


def test_facts_reader_reports_bad_rows(tmp_path):
    p = tmp_path / "facts.jsonl"
    p.write_text('{"subject": "A", "relation": "is_a", "object": "B"}\n'
                 '{"subject": "A"}\n'
                 '[1, 2]\n', encoding="utf-8")
    facts = LegacyFacts.load(str(p))
    assert len(facts.rows) == 1
    assert len(facts.errors) == 2


# --------------------------------------------------------------------------
# rasm-cim-learned/1
# --------------------------------------------------------------------------

def test_learned_verified(tmp_path):
    d = write_learned(str(tmp_path / "learned"),
                      predicate_hints={"где столица франция?": "capital_of"},
                      alias_delta={"ньютон": "Исаак Ньютон"})
    snap = LearnedSnapshot.load(d)
    assert snap.verdict == "VERIFIED" and snap.trustworthy
    assert snap.version == "lv_000"
    assert snap.predicate_hint_for("  Где   столица Франция? ") == "capital_of"
    assert snap.alias_of("Ньютон") == "Исаак Ньютон"
    assert snap.summary()["intents"] == ["ask_fact", "yes_no"]


def test_learned_sha_mismatch_is_not_trusted(tmp_path):
    d = write_learned(str(tmp_path / "learned"), tamper_sha=True)
    snap = LearnedSnapshot.load(d)
    assert snap.verdict == "MISMATCH"
    assert not snap.trustworthy


def test_learned_without_pointer_is_unverified(tmp_path):
    d = write_learned(str(tmp_path / "learned"))
    os.unlink(os.path.join(d, "current.json"))
    snap = LearnedSnapshot.load(d)
    assert snap.verdict == "NO_POINTER"
    assert not snap.trustworthy


def test_learned_wrong_schema_is_refused(tmp_path):
    d = write_learned(str(tmp_path / "learned"))
    path = os.path.join(d, "lv_000.json")
    doc = json.load(open(path, encoding="utf-8"))
    doc["schema"] = "rasm-cim-learned/9"
    json.dump(doc, open(path, "w", encoding="utf-8"))
    with pytest.raises(LearnedFormatError, match="unsupported learned schema"):
        LearnedSnapshot.load(d)


def test_learned_intent_fallback_uses_recorded_counts(tmp_path):
    d = write_learned(str(tmp_path / "learned"))
    snap = LearnedSnapshot.load(d)
    intent, margin = snap.predict_intent("где столица")
    assert intent == "ask_fact"
    assert margin > 0                    # two classes -> a real margin
    assert snap.should_use_fallback("где столица")[0] == "ask_fact"


def test_single_class_artifact_has_no_margin(tmp_path):
    """With one intent the log-probability IS the margin, so it is negative.

    That is v0.15's arithmetic, reproduced rather than "fixed" here: a
    one-class fallback can never clear tau, which is documented behaviour.
    """
    import hashlib
    d = write_learned(str(tmp_path / "learned"))
    path = os.path.join(d, "lv_000.json")
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    doc["intent_fallback"] = {"tokens": {"где": {"ask_fact": 3}},
                              "intent_totals": {"ask_fact": 5}}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False)
    digest = hashlib.sha256(open(path, "rb").read()).hexdigest()
    with open(os.path.join(d, "current.json"), "w", encoding="utf-8") as f:
        json.dump({"current": "lv_000", "sha256": digest}, f)
    snap = LearnedSnapshot.load(d)
    intent, margin = snap.predict_intent("где")
    assert intent == "ask_fact" and margin < 0
    assert snap.should_use_fallback("где")[0] is None


def test_learned_fallback_respects_tau(tmp_path):
    d = write_learned(str(tmp_path / "learned"),
                      config={"alpha": 0.5, "tau": 99.0,
                              "min_margin": 1.0,
                              "use_fallback_on_unknown": True})
    snap = LearnedSnapshot.load(d)
    intent, why = snap.should_use_fallback("где столица")
    assert intent is None and "tau" in why
