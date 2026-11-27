"""Fixtures for the v0.17 tests.

Every test runs against a store created in ``tmp_path``; nothing reads or
writes ``~/.rasm-cim``.  The legacy stores are written by the fixtures using
the exact on-disk layout of ``rasm-cim-kb/1`` — that is the point: the reader
is tested against the format, not against a hand-built Python object.
"""
from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rasm_cim_v16.dialogue_manager import DialogueManager      # noqa: E402
from rasm_cim_v16.persistence import Store                     # noqa: E402

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
V11_FACTS = os.path.join(FIXTURES, "v11_demo_facts.jsonl")


class FakeClock:
    def __init__(self, start: int = 0):
        self.n = start

    def __call__(self) -> str:
        self.n += 1
        return f"2026-01-01T00:00:{self.n:02d}"


def write_legacy_kb(path: str,
                    entities: list[dict],
                    statements: list[dict] | None = None,
                    extra_events: list[dict] | None = None,
                    torn: bool = False) -> str:
    """Write a ``rasm-cim-kb/1`` directory the way v0.12 would have."""
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "meta.json"), "w", encoding="utf-8") as f:
        json.dump({"schema": "rasm-cim-kb/1",
                   "created_at": "2026-01-01T00:00:00Z",
                   "format": "events-jsonl"}, f)
    lines: list[str] = []
    for e in entities:
        lines.append(json.dumps({"event": "entity_defined", "entity": e,
                                 "schema": "rasm-cim-kb/1"}, ensure_ascii=False))
    for s in (statements or []):
        lines.append(json.dumps({"event": "statement_asserted", "statement": s,
                                 "schema": "rasm-cim-kb/1"}, ensure_ascii=False))
    for ev in (extra_events or []):
        lines.append(json.dumps(ev, ensure_ascii=False))
    body = "\n".join(lines) + ("\n" if lines else "")
    if torn:
        body += '{"event": "statement_asserted", "statement": {"id": "st_tor'
    with open(os.path.join(path, "events.jsonl"), "w", encoding="utf-8") as f:
        f.write(body)
    return path


def legacy_entity(eid: str, type_: str, name: str, aliases=None,
                  status: str = "active") -> dict:
    return {"id": eid, "type": type_, "name": name,
            "aliases": list(aliases or []),
            "provenance": {"source": "user_teach", "doc_id": f"doc:{eid}",
                           "fragment": name, "added_at": "2026-01-01T00:00:00Z",
                           "method": "manual", "parser_version": "teach-0.12.0",
                           "reliability": "high"},
            "status": status}


def legacy_statement(sid: str, subject: str, predicate: str, *,
                     object_entity: str | None = None,
                     object_literal: str | None = None,
                     polarity: str = "assert", kind: str = "user_assertion",
                     reliability: str = "high",
                     status: str = "active") -> dict:
    return {"id": sid, "subject": subject, "predicate": predicate,
            "object_entity": object_entity, "object_literal": object_literal,
            "polarity": polarity, "kind": kind,
            "provenance": {"source": "user_teach", "doc_id": f"doc:{sid}",
                           "fragment": f"{subject} {predicate}",
                           "position": [0, 10],
                           "added_at": "2026-01-01T00:00:00Z",
                           "method": "manual", "parser_version": "teach-0.12.0",
                           "reliability": reliability},
            "created_at": "2026-01-01T00:00:00Z", "status": status,
            "retract_reason": None, "replaces": None}


def write_learned(dirpath: str, *, version: str = "lv_000",
                  predicate_hints: dict | None = None,
                  alias_delta: dict | None = None,
                  config: dict | None = None,
                  tamper_sha: bool = False) -> str:
    """Write a ``rasm-cim-learned/1`` directory with a real current.json."""
    import hashlib
    os.makedirs(dirpath, exist_ok=True)
    snap = {
        "schema": "rasm-cim-learned/1", "version": version, "parent": None,
        "created_at": "2026-01-01T00:00:00Z", "seed": 0,
        "corpus_sha256": "0" * 64,
        "config": config or {"alpha": 0.5, "tau": 0.6, "min_margin": 1.0,
                             "use_fallback_on_unknown": True},
        "intent_fallback": {"tokens": {"где": {"ask_fact": 3},
                                       "столица": {"ask_fact": 2},
                                       "ли": {"yes_no": 2}},
                            "intent_totals": {"ask_fact": 5, "yes_no": 2}},
        "predicate_hints": predicate_hints or {},
        "alias_delta": alias_delta or {},
        "corrections": [], "metrics": {"dev_intent_acc": {"n": 10, "acc": 0.9}},
    }
    path = os.path.join(dirpath, f"{version}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(snap, f, ensure_ascii=False, indent=2)
    digest = hashlib.sha256(open(path, "rb").read()).hexdigest()
    with open(os.path.join(dirpath, "current.json"), "w", encoding="utf-8") as f:
        json.dump({"current": version,
                   "sha256": "f" * 64 if tamper_sha else digest}, f)
    return dirpath


@pytest.fixture()
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture()
def home(tmp_path) -> str:
    return str(tmp_path / "home")


@pytest.fixture()
def dm(clock) -> DialogueManager:
    return DialogueManager(clock=clock)


@pytest.fixture()
def store(home, clock, dm) -> Store:
    s = Store(home=home, clock=clock, autosave=False)
    s.attach(dm)
    return s


@pytest.fixture()
def legacy_kb(tmp_path) -> str:
    """A small but semantically complete legacy store."""
    path = str(tmp_path / "legacy-kb")
    entities = [
        legacy_entity("ent_fr", "place", "Франция", ["Французская Республика"]),
        legacy_entity("ent_paris", "place", "Париж"),
        legacy_entity("ent_newton", "person", "Исаак Ньютон", ["Ньютон"]),
        legacy_entity("ent_physicist", "concept", "физик"),
    ]
    statements = [
        legacy_statement("st_1", "ent_fr", "capital_of", object_entity="ent_paris"),
        legacy_statement("st_2", "ent_newton", "is_a", object_entity="ent_physicist"),
        legacy_statement("st_3", "ent_newton", "может_летать",
                         object_literal="нет"),
        legacy_statement("st_4", "ent_fr", "кипит_при", object_literal="100 цельсия"),
    ]
    return write_legacy_kb(path, entities, statements)


@pytest.fixture()
def v11_facts_path() -> str:
    return V11_FACTS
