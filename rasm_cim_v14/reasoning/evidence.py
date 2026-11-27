"""Evidence model (v0.14) — every answer is bound to its supports.

EvidenceItem.role:
  direct   — a confirmed statement fetched from the KB (user_assertion|extracted_fact)
  derived  — produced by a rule application (carries rule_id; support = statement ids below)
  missing  — a prerequisite that would be needed but was not found (absence is data)

EvidenceSet.confidence is CATEGORICAL and derived from evidence structure,
never a raw model score:
  supported — ≥1 direct (or depth-1 derived from active direct facts) evidence,
              no missing prerequisites, no conflict
  partial   — evidence exists but is derived-only (depth≥2) or incomplete
  unknown   — no usable evidence (absence of facts ≠ wrong answer)
  conflict  — contradictory active statements surfaced; nothing hidden
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

STATUSES = ("supported", "partial", "unknown", "conflict")


@dataclass(frozen=True)
class EvidenceItem:
    role: str                     # direct | derived | missing
    statement_id: Optional[str] = None
    source: Optional[str] = None
    doc_id: Optional[str] = None
    reliability: Optional[str] = None
    rule_id: Optional[str] = None
    support_ids: tuple = ()       # for derived: the statements it was derived from
    depth: int = 0
    note: str = ""


@dataclass
class EvidenceSet:
    intent: str
    subject_names: List[str] = field(default_factory=list)
    predicate: Optional[str] = None
    items: List[EvidenceItem] = field(default_factory=list)
    applied_rules: List[dict] = field(default_factory=list)
    missing_prereqs: List[str] = field(default_factory=list)

    def evidence_ids(self) -> List[str]:
        ids: List[str] = []
        for it in self.items:
            if it.role in ("direct", "derived"):
                ids.append(it.statement_id or f"rule:{it.rule_id}")
                ids.extend(it.support_ids)
        # dedupe, keep order
        seen, out = set(), []
        for i in ids:
            if i not in seen:
                seen.add(i)
                out.append(i)
        return out

    def status(self) -> str:
        roles = [it.role for it in self.items]
        has_conflict_marker = any(it.note == "CONFLICT" for it in self.items)
        if has_conflict_marker:
            return "conflict"
        if not any(r in ("direct", "derived") for r in roles):
            # abstained answers: explicit missing prerequisites stay "partial"
            if self.missing_prereqs or any(it.role == "missing" for it in self.items):
                return "partial"
            return "unknown"
        if self.missing_prereqs:
            return "partial"
        if any(it.role == "derived" and it.depth >= 2 for it in self.items):
            return "partial"
        return "supported"

    def to_dict(self) -> dict:
        return {
            "intent": self.intent,
            "subject_names": list(self.subject_names),
            "predicate": self.predicate,
            "status": self.status(),
            "evidence_ids": self.evidence_ids(),
            "items": [{"role": it.role, "statement_id": it.statement_id,
                       "source": it.source, "doc_id": it.doc_id,
                       "reliability": it.reliability, "rule_id": it.rule_id,
                       "support_ids": list(it.support_ids), "depth": it.depth,
                       "note": it.note} for it in self.items],
            "applied_rules": list(self.applied_rules),
            "missing_prereqs": list(self.missing_prereqs),
        }
