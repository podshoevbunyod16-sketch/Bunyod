"""v0.16 — memory architecture: four tiers, one interface.

    A. working     current dialogue/task state, temporary entities,
                   intermediate results.  Bounded (LRU), lost on reset.
    B. episodic    what happened: turns, questions, answers, actions.
                   Backed by ``ContextEngine.history``, persisted in
                   context.json, capped at ``max_turns``.
    C. semantic    what is true: entities, typed relations, provenance.
                   Backed by ``KnowledgeGraph``, persisted in kb.json.
    D. procedural  how to reason: the inference rule table, the relation
                   schema (including user-added relations) and the NLU
                   thresholds.  Persisted inside kb.json so a knowledge base
                   carries the configuration that produced it.

Every record created through ``MemoryManager.record`` gets the fields spec §8
requires: unique id, source, created/modified timestamps, confidence, kind,
links and an update history.

Duplicate detection is content-hash based, so re-teaching the same fact is
detected here as well as in the graph (defence in depth for TEST K).
"""
from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Iterable, Iterator

from .context_engine import ContextEngine
from .contradiction_detector import ContradictionDetector
from .evidence_manager import EvidenceManager
from .knowledge_graph import KnowledgeGraph
from .language_parser import DEFINITION_PATTERNS

TIER_WORKING = "working"
TIER_EPISODIC = "episodic"
TIER_SEMANTIC = "semantic"
TIER_PROCEDURAL = "procedural"
TIERS = (TIER_WORKING, TIER_EPISODIC, TIER_SEMANTIC, TIER_PROCEDURAL)

WORKING_CAP = 256          # max working records before LRU eviction


@dataclass
class MemoryRecord:
    record_id: str
    tier: str
    kind: str
    payload: dict = field(default_factory=dict)
    source: str = ""
    created_at: str = ""
    modified_at: str = ""
    confidence: float = 1.0
    links: list[str] = field(default_factory=list)
    history: list[dict] = field(default_factory=list)

    def fingerprint(self) -> str:
        blob = json.dumps({"tier": self.tier, "kind": self.kind,
                           "payload": self.payload},
                          sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:24]

    def as_dict(self) -> dict:
        return {"id": self.record_id, "tier": self.tier, "kind": self.kind,
                "payload": self.payload, "source": self.source,
                "created_at": self.created_at, "modified_at": self.modified_at,
                "confidence": round(self.confidence, 4), "links": list(self.links),
                "history": list(self.history)}

    @classmethod
    def from_dict(cls, d: dict) -> "MemoryRecord":
        return cls(record_id=d["id"], tier=d["tier"], kind=d["kind"],
                   payload=dict(d.get("payload", {})), source=d.get("source", ""),
                   created_at=d.get("created_at", ""),
                   modified_at=d.get("modified_at", ""),
                   confidence=float(d.get("confidence", 1.0)),
                   links=list(d.get("links", [])),
                   history=list(d.get("history", [])))


class MemoryManager:
    """Facade over the four tiers.  Owns the working tier; the others are
    views over objects that already exist (no duplicated state)."""

    def __init__(self, kg: KnowledgeGraph, context: ContextEngine,
                 evidence: EvidenceManager, detector: ContradictionDetector,
                 inference=None, parser=None, clock=None,
                 working_cap: int = WORKING_CAP):
        self.kg = kg
        self.context = context
        self.evidence = evidence
        self.detector = detector
        self.inference = inference
        self.parser = parser
        self._clock = clock or (lambda: "")
        self.working_cap = working_cap
        self._working: "OrderedDict[str, MemoryRecord]" = OrderedDict()
        self._fingerprints: dict[str, str] = {}      # fp -> record_id
        self._n = 0
        self.evictions = 0
        self.duplicate_hits = 0

    # -- generic record API ------------------------------------------------
    def record(self, kind: str, payload: dict, tier: str = TIER_WORKING,
               source: str = "", confidence: float = 1.0,
               links: Iterable[str] = (), when: str = "",
               dedupe: bool = True) -> tuple[MemoryRecord, bool]:
        """Create or refresh a record.  Returns ``(record, created)``."""
        when = when or self._clock()
        if tier not in TIERS:
            raise ValueError(f"unknown memory tier {tier!r}")
        rec = MemoryRecord(record_id="", tier=tier, kind=kind,
                           payload=dict(payload), source=source,
                           created_at=when, modified_at=when,
                           confidence=confidence, links=list(links))
        fp = rec.fingerprint()
        if dedupe and tier == TIER_WORKING and fp in self._fingerprints:
            prev = self._working.get(self._fingerprints[fp])
            if prev is not None:
                prev.modified_at = when
                prev.history.append({"at": when, "change": "duplicate record "
                                                           "merged", "source": source})
                self._working.move_to_end(prev.record_id)
                self.duplicate_hits += 1
                return prev, False
        self._n += 1
        rec.record_id = f"mem-{self._n:06d}"
        if tier == TIER_WORKING:
            self._working[rec.record_id] = rec
            self._fingerprints[fp] = rec.record_id
            while len(self._working) > self.working_cap:
                rid, _ = self._working.popitem(last=False)
                self.evictions += 1
        return rec, True

    def working(self) -> list[MemoryRecord]:
        return list(self._working.values())

    def clear_working(self) -> int:
        n = len(self._working)
        self._working.clear()
        self._fingerprints.clear()
        return n

    # -- tier views --------------------------------------------------------
    def episodic(self, limit: int | None = None) -> list[dict]:
        turns = self.context.history
        if limit:
            turns = turns[-limit:]
        return [t.as_dict() for t in turns]

    def semantic(self) -> dict:
        return self.kg.stats()

    def procedural(self) -> dict:
        """The reasoning configuration currently in force."""
        out = {
            "relations": self.kg.schema.names(),
            "custom_relations": self.kg.schema.custom_names(),
            "functional": sorted(self.kg.schema.functional_relations()),
            "transitive": sorted(self.kg.schema.transitive_relations()),
            "symmetric": sorted(self.kg.schema.symmetric_relations()),
        }
        if self.inference is not None:
            out["inference_rules"] = self.inference.rule_report()
            out["max_depth"] = self.inference.max_depth
            out["max_nodes"] = self.inference.max_nodes
        if self.parser is not None:
            out["definition_templates"] = len(DEFINITION_PATTERNS)
        out["disjoint_classes"] = [list(p) for p in
                                   sorted(self.detector.disjoint_classes)]
        return out

    def all_records(self, tier: str | None = None) -> Iterator[MemoryRecord]:
        if tier in (None, TIER_WORKING):
            yield from self._working.values()
        if tier in (None, TIER_SEMANTIC):
            for st in self.kg.all_statements(include_retracted=True):
                yield MemoryRecord(record_id=st.statement_id, tier=TIER_SEMANTIC,
                                   kind=f"statement:{st.predicate}",
                                   payload=st.as_dict(), source=st.source,
                                   created_at=st.created_at,
                                   modified_at=st.modified_at,
                                   confidence=st.confidence,
                                   links=list(st.evidence),
                                   history=list(st.history))
        if tier in (None, TIER_EPISODIC):
            for t in self.context.history:
                yield MemoryRecord(record_id=t.turn_id, tier=TIER_EPISODIC,
                                   kind=f"turn:{t.role}", payload=t.as_dict(),
                                   source=t.intent, created_at=t.created_at,
                                   modified_at=t.created_at,
                                   links=list(t.entity_ids and
                                              [str(x) for x in t.entity_ids] or []))
        if tier in (None, TIER_PROCEDURAL):
            yield MemoryRecord(record_id="proc-000001", tier=TIER_PROCEDURAL,
                               kind="reasoning_config", payload=self.procedural(),
                               source="system", created_at="", modified_at="")

    # -- duplicates & conflicts -------------------------------------------
    def duplicates(self) -> list[dict]:
        """Statements that were reinforced more than once (TEST K evidence)."""
        out = []
        for st in self.kg.all_statements(include_retracted=True):
            if st.reinforce > 1:
                out.append({"statement_id": st.statement_id,
                            "reinforce": st.reinforce,
                            "triple": [st.subject, st.predicate, st.object]})
        return out

    def conflicts(self) -> list[dict]:
        return [c.as_dict() for c in self.detector.conflicts()]

    # -- sizes & health ----------------------------------------------------
    def stats(self) -> dict:
        return {
            "working_records": len(self._working),
            "working_cap": self.working_cap,
            "working_evictions": self.evictions,
            "duplicate_record_hits": self.duplicate_hits,
            "episodic_turns": len(self.context.history),
            "episodic_active_entities": len(self.context.active),
            "episodic_topics": 1 + len(self.context.topic_stack),
            "evidence_records": len(self.evidence),
            "conflicts": len(self.detector.conflicts()),
            "unresolved_conflicts": len(self.detector.unresolved()),
            **{"semantic_" + k: v for k, v in self.kg.stats().items()},
        }

    def size_bytes(self) -> dict:
        """Approximate resident size of the serialisable tiers."""
        def blen(o) -> int:
            return len(json.dumps(o, ensure_ascii=False,
                                  default=str).encode("utf-8"))
        return {"working": blen([r.as_dict() for r in self._working.values()]),
                "episodic": blen(self.context.to_dict()),
                "semantic": blen(self.kg.to_dict()),
                "evidence": blen(self.evidence.to_dict()),
                "conflicts": blen(self.detector.to_dict())}

    def integrity_check(self) -> dict:
        return {"context": self.context.integrity_check(),
                "working_over_cap": len(self._working) > self.working_cap,
                "orphan_evidence": [e.evidence_id for e in self.evidence.all()
                                    if self.kg.statement(e.statement_id) is None]}

    # -- (de)serialisation -------------------------------------------------
    def to_dict(self) -> dict:
        return {"working": [r.as_dict() for r in self._working.values()],
                "counter": self._n,
                "fingerprints": dict(self._fingerprints),
                "evictions": self.evictions,
                "duplicate_hits": self.duplicate_hits}

    def load_dict(self, d: dict) -> None:
        self._working = OrderedDict()
        self._fingerprints = {}
        for rd in d.get("working", []):
            r = MemoryRecord.from_dict(rd)
            self._working[r.record_id] = r
            self._fingerprints[r.fingerprint()] = r.record_id
        self._n = int(d.get("counter", len(self._working)))
        self.evictions = int(d.get("evictions", 0))
        self.duplicate_hits = int(d.get("duplicate_hits", 0))
