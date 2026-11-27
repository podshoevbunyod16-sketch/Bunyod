"""Integration layer (roadmap Plan A): legacy formats -> v0.16 graph.

Public surface:

    LegacyKB.load(path)          read a rasm-cim-kb/1 store (dir or events.jsonl)
    LegacyFacts.load(path)       read the v0.11 demo-fact JSONL
    LearnedSnapshot.load(dir)    read a rasm-cim-learned/1 artifact + pointer
    build_plan(...)              dry-run: what WOULD be imported, what cannot be
    apply_plan(...)              idempotent write into a v0.16 DialogueManager
    ImportLedger                 append-only record of every applied import
"""
from __future__ import annotations

from .importer import (ImportError_, ImportLedger, ImportPlan, ImportResult,
                       apply_plan, build_plan, plan_from_artifact,
                       plan_from_legacy_kb)
from .learned import LearnedSnapshot
from .legacy import LegacyEntity, LegacyFacts, LegacyKB, LegacyStatement
from .mapping import (ENTITY_KIND_MAP, PREDICATE_MAP, PredicateMapping,
                      map_entity_kind, map_predicate)

__all__ = [
    "LegacyKB", "LegacyFacts", "LegacyEntity", "LegacyStatement",
    "LearnedSnapshot", "ENTITY_KIND_MAP", "PREDICATE_MAP",
    "PredicateMapping", "map_entity_kind", "map_predicate",
    "ImportPlan", "ImportResult", "ImportLedger", "ImportError_",
    "build_plan", "apply_plan", "plan_from_legacy_kb", "plan_from_artifact",
]
