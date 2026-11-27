"""TEST G + spec §15 — persistence, restart survival, migration, rollback.

Migration is verified against a real legacy file written to disk (not a
mock), on a copy, twice, with record counts checked before and after.
"""
from __future__ import annotations

import json
import os

import pytest

from rasm_cim_v16 import SCHEMA_VERSION
from rasm_cim_v16.migrate import (R1, R2, R3, R4, R5,
                                  classify_legacy_is_a,
                                  legacy_to_v2, migrate_context_file,
                                  migrate_file, write_legacy_sample)
from rasm_cim_v16.persistence import Store, atomic_write_json, read_json

from .conftest import NEWTON_TEACHINGS


# ------------------------------------------------------------------ TEST G
def test_g_knowledge_survives_a_full_restart(home, clock):
    from rasm_cim_v16.dialogue_manager import DialogueManager
    dm = DialogueManager(clock=clock)
    st = Store(home=home, clock=clock)
    st.attach(dm)
    for line in NEWTON_TEACHINGS:
        dm.say(line)
    dm.save()
    n_facts = len(dm.kg.all_statements())
    n_entities = len(dm.kg.resolver)
    del dm

    dm2 = DialogueManager(clock=clock)
    st2 = Store(home=home, clock=clock)
    st2.attach(dm2)
    st2.load(dm2)
    assert len(dm2.kg.all_statements()) == n_facts
    assert len(dm2.kg.resolver) == n_entities
    a = dm2.say("Ньютон — физик?")
    assert a.text.startswith("Да"), a.text
    assert a.status == "INFERRED"


def test_g_context_survives_a_restart(home, clock):
    from rasm_cim_v16.dialogue_manager import DialogueManager
    dm = DialogueManager(clock=clock)
    st = Store(home=home, clock=clock)
    st.attach(dm)
    dm.say("teach Исаак Ньютон открыл закон всемирного тяготения")
    dm.say("Кто такой Исаак Ньютон?")
    dm.save()
    turns = len(dm.context.history)
    focus = dict(dm.context.focus)
    del dm

    dm2 = DialogueManager(clock=clock)
    st2 = Store(home=home, clock=clock)
    st2.attach(dm2)
    st2.load(dm2)
    assert len(dm2.context.history) == turns
    assert dm2.context.focus["subject_id"] == focus["subject_id"]
    # the pronoun still resolves after the restart
    a = dm2.say("Что он открыл?")
    assert "закон всемирного тяготения" in a.text


def test_g_restart_is_repeatable(home, clock):
    from rasm_cim_v16.dialogue_manager import DialogueManager
    dm = DialogueManager(clock=clock)
    st = Store(home=home, clock=clock)
    st.attach(dm)
    dm.say("teach Исаак Ньютон — это физик")
    dm.save()
    for i in range(3):
        d = DialogueManager(clock=clock)
        s = Store(home=home, clock=clock)
        s.attach(d)
        s.load(d)
        assert len(d.kg.all_statements()) == i + 1, \
            [x.statement_id for x in d.kg.all_statements()]
        d.say(f"teach Учёный{i} — это человек")   # a genuinely new fact
        d.save()
    d = DialogueManager(clock=clock)
    s = Store(home=home, clock=clock)
    s.attach(d)
    s.load(d)
    assert len(d.kg.all_statements()) == 4      # 1 + 3 appends
    # re-teaching an OLD fact across a restart must reinforce, not duplicate
    d.say("teach Исаак Ньютон — это физик")
    d.save()
    d2 = DialogueManager(clock=clock)
    s2 = Store(home=home, clock=clock)
    s2.attach(d2)
    s2.load(d2)
    assert len(d2.kg.all_statements()) == 4
    st = d2.kg.all_statements()[0]
    assert st.reinforce >= 1, "the re-taught fact must be marked as reinforced"


def test_g_polarity_oracle_is_rebuilt_after_load(home, clock):
    from rasm_cim_v16.dialogue_manager import DialogueManager
    dm = DialogueManager(clock=clock)
    st = Store(home=home, clock=clock)
    st.attach(dm)
    dm.say("teach Исаак Ньютон — это физик")
    dm.say("teach Исаак Ньютон не был физиком")
    dm.save()
    dm2 = DialogueManager(clock=clock)
    st2 = Store(home=home, clock=clock)
    st2.attach(dm2)
    st2.load(dm2)
    a = dm2.say("Ньютон — физик?")
    assert a.status == "CONTRADICTED"


# -------------------------------------------------------------- file shape
def test_atomic_write_creates_valid_json(tmp_path):
    p = str(tmp_path / "x.json")
    n = atomic_write_json(p, {"a": [1, 2, "строка"]})
    assert n > 0
    assert read_json(p) == {"a": [1, 2, "строка"]}


def test_saved_documents_are_versioned(home, clock):
    from rasm_cim_v16.dialogue_manager import DialogueManager
    dm = DialogueManager(clock=clock)
    st = Store(home=home, clock=clock)
    st.attach(dm)
    dm.say("teach Исаак Ньютон — это физик")
    dm.save()
    assert read_json(st.kb_path)["schema"] == SCHEMA_VERSION
    assert read_json(st.context_path)["schema"] == 2


def test_load_on_an_empty_home_is_not_an_error(home, clock):
    from rasm_cim_v16.dialogue_manager import DialogueManager
    dm = DialogueManager(clock=clock)
    st = Store(home=home, clock=clock)
    st.attach(dm)
    assert st.exists() is False
    notes = st.load(dm)
    assert notes == {"kb": [], "context": []}


def test_home_comes_from_the_environment(tmp_path, monkeypatch):
    d = tmp_path / "custom"
    monkeypatch.setenv("RASM_CIM_HOME", str(d))
    from rasm_cim_v16.persistence import default_home
    assert default_home() == str(d)


# --------------------------------------------------------------- migration
LEGACY_FACTS = [
    ("Ньютон", "is_a", "Исаак Ньютон"),        # R1 -> alias_of
    ("Исаак Ньютон", "is_a", "физик"),          # R2 -> instance_of
    ("Альберт Эйнштейн", "is_a", "Исаак Ньютон"),  # R3 -> related_to
    ("Исаак Ньютон", "discovered", "закон всемирного тяготения"),  # R4
    ("Исаак Ньютон", "no_such_rel", "что-то"),  # R5 -> related_to
]


def test_classify_legacy_is_a_rules():
    assert classify_legacy_is_a("Ньютон", "Исаак Ньютон")[0] == "alias_of"
    assert classify_legacy_is_a("Исаак Ньютон", "физик")[0] == "instance_of"
    assert classify_legacy_is_a("Альберт Эйнштейн", "Исаак Ньютон")[0] == "related_to"


def test_migration_is_lossless(tmp_path):
    p = str(tmp_path / "kb.json")
    write_legacy_sample(p, LEGACY_FACTS,
                        entities=[{"name": "Исаак Ньютон",
                                   "aliases": ["Ньютон", "Isaac Newton"]}])
    before = len(read_json(p)["facts"])
    rep = migrate_file(p)
    assert rep["status"] == "migrated"
    assert rep["records_in"] == before
    assert rep["records_out"] == before
    after = read_json(p)
    assert len(after["graph"]["statements"]) == before
    assert after["schema"] == SCHEMA_VERSION
    assert rep["rules"][R1] == 1 and rep["rules"][R2] == 1
    assert rep["rules"][R3] == 1 and rep["rules"][R4] == 1
    assert rep["rules"][R5] == 1


def test_migration_keeps_provenance(tmp_path):
    p = str(tmp_path / "kb.json")
    write_legacy_sample(p, LEGACY_FACTS[:1])
    migrate_file(p)
    st = read_json(p)["graph"]["statements"][0]
    assert st["source"] == "migration:v1"
    assert st["prov"]["rule"] == R1
    assert st["prov"]["legacy"]["p"] == "is_a"
    assert st["prov"]["original_predicate"] == "is_a"


def test_migration_creates_a_backup_and_can_roll_back(tmp_path):
    p = str(tmp_path / "kb.json")
    write_legacy_sample(p, LEGACY_FACTS)
    original = read_json(p)
    rep = migrate_file(p)
    assert os.path.exists(rep["backup"])
    assert read_json(rep["backup"]) == original
    st = Store(home=str(tmp_path))
    st.rollback(rep["backup"])
    assert read_json(p) == original


def test_migration_is_idempotent(tmp_path):
    p = str(tmp_path / "kb.json")
    write_legacy_sample(p, LEGACY_FACTS)
    first = migrate_file(p)
    snapshot = read_json(p)
    second = migrate_file(p)
    assert second["status"] == "already-current"
    assert read_json(p)["graph"]["statements"] == snapshot["graph"]["statements"]
    assert first["records_out"] == second["records_out"]


def test_migration_dry_run_writes_nothing(tmp_path):
    p = str(tmp_path / "kb.json")
    write_legacy_sample(p, LEGACY_FACTS)
    before = read_json(p)
    rep = migrate_file(p, dry_run=True)
    assert rep["status"] == "dry-run"
    assert read_json(p) == before
    assert not [f for f in os.listdir(tmp_path) if ".bak-" in f]


def test_migration_result_is_usable_by_the_system(tmp_path):
    p = str(tmp_path / "kb.json")
    write_legacy_sample(p, LEGACY_FACTS,
                        entities=[{"name": "Исаак Ньютон",
                                   "aliases": ["Ньютон"]}])
    migrate_file(p)
    from rasm_cim_v16.dialogue_manager import DialogueManager
    dm = DialogueManager(clock=lambda: "t")
    st = Store(home=str(tmp_path), clock=lambda: "t")
    st.attach(dm)
    st.load(dm)
    a = dm.say("Ньютон — физик?")
    assert a.text.startswith("Да"), a.text
    # The legacy alias «Ньютон» was migrated onto the entity record, so the
    # identity is STORED, not derived: DIRECT is the honest status here.  It
    # would be INFERRED only if the alias existed as a separate entity linked
    # by an alias_of edge (that path is TEST D).
    assert a.status == "DIRECT", a.status
    assert a.facts and a.facts[0].status == "direct"
    # the self-referential alias edge the legacy record produced is kept for
    # provenance but never indexed, so it cannot appear as a graph fact
    assert dm.kg.stats()["redundant"] == 1
    assert not [x for x in dm.kg.all_statements()
                if x.subject == x.object], "no self-loop may be queryable"


def test_migration_of_negated_legacy_records(tmp_path):
    p = str(tmp_path / "kb.json")
    atomic_write_json(p, {"version": 1, "entities": [],
                          "facts": [{"s": "Ньютон", "p": "is_a", "o": "химик",
                                     "neg": True}]})
    migrate_file(p)
    st = read_json(p)["graph"]["statements"][0]
    assert st["pol"] == "false"


def test_migration_no_file_is_reported(tmp_path):
    rep = migrate_file(str(tmp_path / "missing.json"))
    assert rep["status"] == "no-file"


def test_context_migration_on_a_real_file(tmp_path):
    p = str(tmp_path / "context.json")
    atomic_write_json(p, {"schema": 1, "turn_no": 2,
                          "entities": [{"id": 1}], "history": []})
    rep = migrate_context_file(p)
    assert rep["status"] == "migrated"
    assert read_json(p)["schema"] == 2
    assert migrate_context_file(p)["status"] == "already-current"


def test_store_detects_a_file_needing_migration(tmp_path):
    p = str(tmp_path / "kb.json")
    write_legacy_sample(p, LEGACY_FACTS[:1])
    st = Store(home=str(tmp_path))
    assert st.needs_migration() is True
    migrate_file(p)
    assert st.needs_migration() is False


def test_upgrade_kb_dict_on_a_current_document_is_a_noop():
    from rasm_cim_v16.migrate import upgrade_kb_dict
    doc = {"schema": SCHEMA_VERSION, "graph": {"statements": []}}
    notes = upgrade_kb_dict(doc)
    assert notes == [f"schema={SCHEMA_VERSION}, nothing to do"]


def test_a_newer_schema_is_not_silently_downgraded():
    from rasm_cim_v16.migrate import upgrade_kb_dict
    doc = {"schema": SCHEMA_VERSION + 1}
    notes = upgrade_kb_dict(doc)
    assert "NEWER" in notes[0]


def test_legacy_to_v2_report_lists_flagged_records():
    kb, rep = legacy_to_v2({"version": 1, "entities": [], "facts": [
        {"s": "A", "p": "weird", "o": "B"}]})
    assert rep["flagged"], "an unknown legacy relation must be flagged"
    assert rep["rules"][R5] == 1
