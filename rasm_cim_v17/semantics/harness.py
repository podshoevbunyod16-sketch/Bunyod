"""Build the demo database the metrics run against, through the normal path.

Two rules, both to keep the numbers honest:

1. The database is built **only** by teach utterances and by the Plan A
   importer — never by writing triples directly.  If a fact cannot be taught,
   it must not silently appear in the knowledge base the metrics praise.
2. The store is in memory by default: measuring must not touch a user's
   database (`~/.rasm-cim`) even if the caller forgot a flag.
"""
from __future__ import annotations

import os

from rasm_cim_v16.dialogue_manager import DialogueManager
from rasm_cim_v16.persistence import Store

from ..interop import apply_plan, build_plan
from .corpus import load_seed

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_SEED = os.path.join(HERE, "tests_v17", "corpus", "seed_kb.jsonl")
DEFAULT_CORPUS = os.path.join(HERE, "tests_v17", "corpus",
                              "semantic_cases.jsonl")
DEFAULT_V11 = os.path.join(HERE, "tests_v17", "fixtures",
                           "v11_demo_facts.jsonl")
DEFAULT_DIALOGUES = os.path.join(HERE, "tests_v17", "corpus",
                                 "dialogue_cases.jsonl")


def build_demo(seed_path: str | None = None,
               v11_path: str | None = DEFAULT_V11,
               clock=None, expanded: bool = False) -> DialogueManager:
    """A dialogue manager seeded with the v0.11 corpus + the demo teaches.

    ``expanded=True`` дополнительно применяет `knowledge/catalog.py`: около
    сотни людей с профессиями, города и организации.  По умолчанию база та же,
    что измеряли авторы, — чтобы их числа оставались воспроизводимыми.
    """
    dm = DialogueManager(clock=clock)
    if v11_path and os.path.exists(v11_path):
        store = Store(home=os.path.join(os.path.dirname(v11_path), "unused"))
        store.attach(dm)
        apply_plan(build_plan(v11_path, kind="facts"), dm, store=store,
                   save=False, backup=False, record_ledger=False)
    seed = seed_path or DEFAULT_SEED
    if os.path.exists(seed):
        for line in load_seed(seed):
            dm.say(line)
    if expanded:
        from ..knowledge.catalog import facts as catalog_facts
        for line in catalog_facts():
            dm.say(line)
    return dm


def seed_report(dm) -> dict:
    """What the demo database actually contains — reported with the metrics."""
    return {"entities": len(dm.kg.resolver),
            "statements": len(dm.kg.all_statements()),
            "evidence": len(dm.evidence),
            "relations": len(dm.kg.schema.names())}
