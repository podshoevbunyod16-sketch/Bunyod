"""Spec §8 — the four memory tiers, record shape, deduplication, sizes.
"""
from __future__ import annotations

import pytest

from rasm_cim_v16.memory_manager import (TIER_EPISODIC, TIER_PROCEDURAL,
                                         TIER_SEMANTIC, TIER_WORKING, TIERS,
                                         MemoryRecord)


def test_four_tiers_exist():
    assert set(TIERS) == {"working", "episodic", "semantic", "procedural"}


def test_unknown_tier_is_rejected(dm):
    with pytest.raises(ValueError):
        dm.memory.record(kind="x", payload={}, tier="nope")


def test_working_record_has_every_required_field(dm):
    rec, created = dm.memory.record(kind="fact", tier=TIER_WORKING,
                                    payload={"a": 1}, source="test",
                                    confidence=0.9, links=["st-000001"],
                                    when="t1")
    assert created
    assert rec.record_id.startswith("mem-")
    assert rec.source == "test"
    assert rec.created_at == "t1" and rec.modified_at == "t1"
    assert rec.confidence == 0.9
    assert rec.links == ["st-000001"]
    assert rec.tier == TIER_WORKING
    assert rec.kind == "fact"
    assert rec.history == []


def test_working_memory_is_bounded(dm):
    dm.memory.working_cap = 10
    for i in range(25):
        dm.memory.record(kind="fact", tier=TIER_WORKING, payload={"i": i})
    assert len(dm.memory.working()) == 10
    assert dm.memory.evictions == 15


def test_working_memory_can_be_cleared(dm):
    dm.memory.record(kind="fact", tier=TIER_WORKING, payload={"i": 1})
    assert dm.memory.clear_working() == 1
    assert dm.memory.working() == []


def test_episodic_tier_mirrors_the_dialogue_history(dm):
    dm.say("teach Исаак Ньютон — это физик")
    dm.say("Кто такой Исаак Ньютон?")
    ep = dm.memory.episodic()
    assert len(ep) == 4                      # 2 user + 2 system turns
    assert ep[0]["role"] == "user"
    assert ep[1]["role"] == "system"
    assert len(dm.memory.episodic(limit=2)) == 2


def test_semantic_tier_reports_the_graph(dm):
    dm.say("teach Исаак Ньютон — это физик")
    sem = dm.memory.semantic()
    assert sem["statements"] == 1 and sem["entities"] == 2


def test_procedural_tier_exposes_the_reasoning_configuration(dm):
    proc = dm.memory.procedural()
    assert "instance_of" in proc["relations"]
    assert len(proc["inference_rules"]) == 5
    assert proc["max_depth"] == 3
    assert "custom_relations" in proc


def test_procedural_tier_reflects_a_custom_relation(dm):
    dm.schema.register_dict({"name": "teaches_at", "human": "преподаёт"})
    assert "teaches_at" in dm.memory.procedural()["custom_relations"]


def test_all_records_iterates_every_tier(dm):
    dm.say("teach Исаак Ньютон — это физик")
    dm.memory.record(kind="x", tier=TIER_WORKING, payload={})
    tiers = {r.tier for r in dm.memory.all_records()}
    assert tiers == set(TIERS)
    assert len(list(dm.memory.all_records(tier=TIER_SEMANTIC))) == 1


def test_duplicates_are_reported(dm):
    dm.say("teach Исаак Ньютон — это физик")
    dm.say("teach Исаак Ньютон — это физик")
    d = dm.memory.duplicates()
    assert d and d[0]["reinforce"] == 2


def test_conflicts_are_visible_from_the_memory_facade(dm):
    dm.say("teach Исаак Ньютон — это физик")
    dm.say("teach Исаак Ньютон не был физиком")
    assert len(dm.memory.conflicts()) == 1


def test_stats_and_sizes(dm):
    dm.say("teach Исаак Ньютон — это физик")
    s = dm.memory.stats()
    assert s["episodic_turns"] == 2
    assert s["semantic_statements"] == 1
    sizes = dm.memory.size_bytes()
    assert set(sizes) == {"working", "episodic", "semantic", "evidence",
                          "conflicts"}
    assert all(v >= 0 for v in sizes.values())


def test_integrity_check(dm):
    dm.say("teach Исаак Ньютон — это физик")
    rep = dm.memory.integrity_check()
    assert rep["context"] == []
    assert rep["orphan_evidence"] == []
    assert rep["working_over_cap"] is False


def test_record_fingerprint_is_content_based():
    a = MemoryRecord(record_id="1", tier=TIER_WORKING, kind="k", payload={"x": 1})
    b = MemoryRecord(record_id="2", tier=TIER_WORKING, kind="k", payload={"x": 1})
    c = MemoryRecord(record_id="3", tier=TIER_WORKING, kind="k", payload={"x": 2})
    assert a.fingerprint() == b.fingerprint()
    assert a.fingerprint() != c.fingerprint()


def test_memory_roundtrip(dm):
    dm.say("teach Исаак Ньютон — это физик")
    d = dm.memory.to_dict()
    dm.memory.load_dict(d)
    assert dm.memory.stats()["working_records"] == len(dm.memory.working())


def test_memory_does_not_hold_the_whole_kb_in_a_cache(dm):
    """Spec §13: no full-DB reload per query.  The semantic tier is the graph
    itself; the working tier stays small regardless of KB size."""
    for i in range(200):
        dm.say(f"teach Сущность{i} — это класс{i}")
    assert len(dm.kg.all_statements()) == 200
    assert len(dm.memory.working()) <= dm.memory.working_cap
