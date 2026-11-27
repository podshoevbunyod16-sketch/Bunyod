"""Тесты аудиторских инструментов и правок, сделанных по итогам аудита.

Проверяется то, что видно пользователю и что легко сломать незаметно:
нормализация реплик, память диалога, расширенный каталог, составные факты,
учёт упоминаний при непонятой реплике и целостность самих аудиторских наборов.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from audit_v1.harness import load_cases
from audit_v1.second_annotator import agreement
from rasm_cim_v17.discourse.memory import DialogueMemory
from rasm_cim_v17.knowledge.catalog import facts as catalog_facts
from rasm_cim_v17.knowledge.catalog import size as catalog_size
from rasm_cim_v17.pipeline import SemanticPipeline
from rasm_cim_v17.semantics import build_demo
from rasm_cim_v17.semantics.lexicon import FILLERS, PROFESSIONS, normalize

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "audit_v1" / "data"


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("Ну он, типа, физик?", "Он физик?"),
        ("Стало быть, он врач?", "Он врач?"),
        ("Правда ли, что она астроном?", "Она астроном?"),
        ("Он работает геологом?", "Он геолог?"),
        ("Она астрономом работает?", "Она астроном?"),
        ("Он математик по профессии?", "Он математик?"),
        ("А она врач, верно?", "А она врач?"),
        ("Она... историк?", "Она историк?"),
        ("Бале, кто такой Рустам?", "Кто такой Рустам?"),
        ("Слушай, а кто такой Сергей?", "А кто такой Сергей?"),
    ],
)
def test_normalize_rewrites(raw: str, expected: str):
    assert normalize(raw) == expected


def test_normalize_keeps_pronouns_and_content():
    # «он» нельзя удалять как заполнитель: это местоимение, ради которого всё
    assert normalize("Он поэт?") == "Он поэт?"
    for filler in FILLERS:
        assert filler.casefold() not in PROFESSIONS
    # имена из каталога не должны попадать в список заполнителей
    for name, _ in (("Иван", ""), ("Мария", ""), ("Рустам", "")):
        assert name.casefold() not in {f.casefold() for f in FILLERS}


def test_dialogue_memory_counts_and_cap():
    memory = DialogueMemory(max_items=3)
    for turn, entity in enumerate([1, 2, 1, 3, 4]):
        memory.note(entity, f"e{entity}", turn=turn)
    assert len(memory.entries) == 3
    assert memory.dropped >= 1
    assert memory.count(1) == 2
    assert memory.last_turn(1) == 2
    restored = DialogueMemory.from_dict(memory.to_dict())
    assert restored.ids() == memory.ids()
    assert restored.count(1) == 2


def test_expanded_catalog_adds_information():
    default = build_demo()
    expanded = build_demo(expanded=True)
    assert len(expanded.kg.resolver) > len(default.kg.resolver)
    assert catalog_size()["people"] >= 40
    assert len(catalog_facts()) == catalog_size()["facts"]


def test_compound_teach_stores_both_facts():
    dm = build_demo()
    pipe = SemanticPipeline(dm)
    pipe.respond("Камол — механик, Сабрина — юрист.")
    assert pipe.respond("Кто такой Камол?").text.startswith("Камол")
    assert pipe.respond("Кто такая Сабрина?").text.startswith("Сабрина")


def test_mention_is_remembered_even_when_the_turn_is_not_parsed():
    dm = build_demo()
    pipe = SemanticPipeline(dm)
    pipe.respond("Дилшод — это поэт.")
    pipe.respond("Бале, кто такой Дилшод?")      # реплика с таджикским словом
    answer = pipe.respond("Он поэт?")
    focus = {f.surface.casefold(): f for f in pipe.last_meaning.focus}
    assert focus["он"].status == "RESOLVED"
    assert dm.kg.name_of(focus["он"].entity_id) == "Дилшод"
    assert answer.status == "DIRECT"


# ---------------------------------------------------------------- целостность
def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_audit_sets_are_sealed_and_large_enough():
    sealed = {}
    for line in (DATA / "SEALED.sha256").read_text(encoding="utf-8").splitlines():
        digest, name = line.split()
        sealed[Path(name).name] = digest
    for name in ("bind.jsonl", "refuse.jsonl", "mixed.jsonl",
                 "paraphrase.jsonl", "hidden.jsonl"):
        assert _sha256(DATA / name) == sealed[name], f"{name}: изменён после запечатывания"
    visible = sum(len(load_cases(DATA / n)) for n in
                  ("bind.jsonl", "refuse.jsonl", "mixed.jsonl"))
    assert visible >= 150
    long_dialogues = [c for n in ("bind.jsonl", "refuse.jsonl", "mixed.jsonl")
                      for c in load_cases(DATA / n) if len(c.turns) >= 8]
    assert len(long_dialogues) >= 20
    assert len(load_cases(DATA / "paraphrase.jsonl")) >= 50


def test_every_case_declares_author_date_and_method():
    for name in ("bind.jsonl", "refuse.jsonl", "mixed.jsonl",
                 "paraphrase.jsonl", "hidden.jsonl"):
        for case in load_cases(DATA / name):
            assert case.author and case.date and case.method, case.id
            if case.expected == "BIND":
                assert case.entity, case.id
            else:
                assert case.refuse_reason, case.id


def test_second_annotator_agreement_is_substantial():
    cases = []
    for name in ("bind", "refuse", "mixed", "paraphrase"):
        cases += load_cases(DATA / f"{name}.jsonl")
    result = agreement(cases)
    assert result["n"] == round(len(cases) * 0.3)
    assert result["observed"] >= 0.9
    record = json.loads((ROOT / "audit_v1" / "results" /
                         "second_annotator.json").read_text(encoding="utf-8"))
    assert record["pooled"]["kappa"] >= 0.8
