"""Concept/Entity/Statement models with stable ids and a versioned schema.

Schema: rasm-cim-kb/1  (explicit; unknown schema → import refuses)
Entity id    : ent_<sha256(type|fold(name))[:16]>   — deterministic across runs/machines
Statement id : st_<sha256(canonical triple+polarity+kind+doc_id)[:16]> — re-import idempotent
Aliases are many-to-many; an alias pointing at ≥2 active entities is AMBIGUOUS by
design and is NEVER auto-merged (spec: do not merge distinct entities on similarity).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import List, Optional

SCHEMA = "rasm-cim-kb/1"
PARSER_VERSION = "teach-0.12.0"

ENTITY_TYPES = ("person", "animal", "place", "organization", "concept", "thing", "other")
STATEMENT_KINDS = ("user_assertion", "extracted_fact", "hypothesis")
POLARITIES = ("assert", "deny")

# predicates with at most one active object per subject (functional semantics)
FUNCTIONAL_PREDICATES = frozenset({
    "capital_of", "population_of", "height_of", "director_of", "ceo_of", "governor_of",
})


def fold(text: str) -> str:
    return (text.replace("ё", "е").replace("Ё", "е")
            .casefold().strip())


def entity_id(type_: str, name: str) -> str:
    if type_ not in ENTITY_TYPES:
        raise ValueError(f"unknown entity type {type_!r}; allowed: {ENTITY_TYPES}")
    h = hashlib.sha256(f"{type_}|{fold(name)}".encode("utf-8")).hexdigest()[:16]
    return f"ent_{h}"


def statement_id(subject: str, predicate: str, obj_repr: str,
                 polarity: str, kind: str, doc_id: str) -> str:
    canon = json.dumps([subject, fold(predicate), fold(obj_repr), polarity, kind, doc_id],
                       ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return "st_" + hashlib.sha256(canon.encode("utf-8")).hexdigest()[:16]


@dataclass
class Entity:
    id: str
    type: str
    name: str
    aliases: List[str] = field(default_factory=list)
    provenance: Optional[dict] = None      # Provenance.to_dict()
    status: str = "active"                  # active | retracted

    def to_dict(self) -> dict:
        return {"id": self.id, "type": self.type, "name": self.name,
                "aliases": list(self.aliases), "provenance": self.provenance,
                "status": self.status}

    @staticmethod
    def from_dict(d: dict) -> "Entity":
        return Entity(id=d["id"], type=d["type"], name=d["name"],
                      aliases=list(d.get("aliases", [])),
                      provenance=d.get("provenance"),
                      status=d.get("status", "active"))


@dataclass
class Statement:
    id: str
    subject: str                 # entity id
    predicate: str
    object_entity: Optional[str] # entity id if object is an entity
    object_literal: Optional[str]# literal otherwise (number, temperature, …)
    polarity: str                # assert | deny
    kind: str                    # user_assertion | extracted_fact | hypothesis
    provenance: dict
    created_at: str
    status: str = "active"       # active | retracted (derived from events)
    retract_reason: Optional[str] = None
    replaces: Optional[str] = None   # statement id this one revises (update flow)

    @property
    def object_repr(self) -> str:
        return f"ent:{self.object_entity}" if self.object_entity else f"lit:{fold(self.object_literal or '')}"

    def to_dict(self) -> dict:
        return {"id": self.id, "subject": self.subject, "predicate": self.predicate,
                "object_entity": self.object_entity, "object_literal": self.object_literal,
                "polarity": self.polarity, "kind": self.kind,
                "provenance": self.provenance, "created_at": self.created_at,
                "status": self.status, "retract_reason": self.retract_reason,
                "replaces": self.replaces}

    @staticmethod
    def from_dict(d: dict) -> "Statement":
        return Statement(id=d["id"], subject=d["subject"], predicate=d["predicate"],
                         object_entity=d.get("object_entity"),
                         object_literal=d.get("object_literal"),
                         polarity=d.get("polarity", "assert"),
                         kind=d.get("kind", "user_assertion"),
                         provenance=d.get("provenance", {}),
                         created_at=d.get("created_at", ""),
                         status=d.get("status", "active"),
                         retract_reason=d.get("retract_reason"),
                         replaces=d.get("replaces"))
