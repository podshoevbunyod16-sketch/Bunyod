"""Plan A — the integrity check must catch real corruption, not just pass.

Each test damages one invariant of ``kb.json`` the way a bad merge, a partial
migration or a hand edit would, and asserts that ``verify`` refuses to call the
database healthy.  A check that cannot fail is not a check.
"""
from __future__ import annotations

import json
import os

import pytest

from conftest import V11_FACTS, legacy_entity, legacy_statement, write_legacy_kb

from rasm_cim_v16.dialogue_manager import DialogueManager
from rasm_cim_v16.persistence import Store, read_json, atomic_write_json

from rasm_cim_v17.interop import apply_plan, build_plan
from rasm_cim_v17.routing import RoutingStore
from rasm_cim_v17.verify import verify_dm, verify_home


@pytest.fixture()
def imported(tmp_path):
    """A store with the real v0.11 corpus imported into it."""
    home = str(tmp_path / "home")
    dm = DialogueManager()
    store = Store(home=home)
    store.attach(dm)
    apply_plan(build_plan(V11_FACTS), dm, store=store)
    return home


def _edit_kb(home: str, mutate) -> None:
    path = os.path.join(home, "kb.json")
    doc = read_json(path)
    mutate(doc)
    atomic_write_json(path, doc)


def test_a_healthy_import_verifies(imported):
    rep = verify_home(imported)
    assert rep.ok, rep.errors
    assert rep.stats["statements"] >= 34
    assert rep.stats["evidence"] >= 34
    assert rep.stats["custom_relations"] >= 1


def test_empty_home_is_not_an_error(tmp_path):
    rep = verify_home(str(tmp_path / "nothing"))
    assert rep.ok
    assert any("нет базы" in w for w in rep.warnings)


def test_dangling_subject_is_an_error(imported):
    def mutate(doc):
        ents = doc["graph"]["resolver"]["entities"]
        victim = doc["graph"]["statements"][0]["s"]
        doc["graph"]["resolver"]["entities"] = [e for e in ents
                                                if e["id"] != victim]
    _edit_kb(imported, mutate)
    rep = verify_home(imported)
    assert not rep.ok
    assert any("не существует в базе" in e for e in rep.errors)


def test_unregistered_relation_is_an_error(imported):
    _edit_kb(imported, lambda doc: doc["graph"]["statements"][0].update(
        {"p": "телепортация"}))
    rep = verify_home(imported)
    assert not rep.ok
    assert any("не зарегистрировано" in e for e in rep.errors)


def test_duplicate_direct_assertion_is_an_error(imported):
    def mutate(doc):
        sts = doc["graph"]["statements"]
        clone = dict(sts[0])
        clone["id"] = sts[0]["id"] + "-copy"
        sts.append(clone)
    _edit_kb(imported, mutate)
    rep = verify_home(imported)
    assert not rep.ok
    assert any("дубль прямого утверждения" in e for e in rep.errors)


def test_orphan_evidence_is_an_error(imported):
    _edit_kb(imported, lambda doc: doc["evidence"]["evidence"].append(
        {"id": "ev-999999", "statement_id": "st-999999", "kind": "import",
         "utterance": "", "turn_id": "", "source": "", "created_at": "",
         "note": ""}))
    rep = verify_home(imported)
    assert not rep.ok
    assert any("несуществующее" in e for e in rep.errors)


def test_missing_evidence_is_a_warning(imported):
    _edit_kb(imported, lambda doc: doc["evidence"].update({"evidence": []}))
    rep = verify_home(imported)
    assert rep.ok                                  # legal, but worth a look
    assert any("без записи об источнике" in w for w in rep.warnings)


def test_dead_routing_hint_is_a_warning(imported):
    RoutingStore(imported).merge_predicate_hints(
        {"странный вопрос?": "не_существующее_отношение"})
    rep = verify_home(imported)
    assert rep.ok
    assert any("подсказок маршрутизации" in w for w in rep.warnings)


def test_missing_import_source_is_a_warning(tmp_path):
    kb = write_legacy_kb(
        str(tmp_path / "kb"),
        [legacy_entity("e1", "person", "Иван")],
        [legacy_statement("s1", "e1", "is_a", object_literal="человек")])
    home = str(tmp_path / "home")
    dm = DialogueManager()
    store = Store(home=home)
    store.attach(dm)
    apply_plan(build_plan(kb), dm, store=store)
    import shutil
    shutil.rmtree(kb)                 # the whole source directory moved away
    rep = verify_home(home)
    assert rep.ok
    assert any("больше не существует" in w for w in rep.warnings)


def test_in_memory_check_finds_the_same_kind_of_damage():
    dm = DialogueManager()
    dm.say("teach Иван — это человек")
    assert verify_dm(dm).ok
    dm.kg.schema = type(dm.kg.schema)(builtin=False)     # wipe the vocabulary
    rep = verify_dm(dm)
    assert not rep.ok
    assert any("не зарегистрировано" in e for e in rep.errors)


def test_report_renders_without_crashing(imported):
    text = verify_home(imported).render()
    assert text.startswith("ПРОВЕРКА")
    assert "statements" in text
