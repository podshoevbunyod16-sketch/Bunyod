"""Evidence audit: for every fact an answer quotes, where does it come from?

    evidence precision   доля фактов ответа, у которых есть запись об источнике
    premise coverage     доля посылок в доказательствах выведенных фактов,
                         у которых тоже есть запись об источнике

The second number is the one that keeps inference honest: a conclusion can only
be as trustworthy as the statements it was derived from, and a proof whose
premises have no source is a chain of assertions.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AnswerAudit:
    facts: int = 0
    direct: int = 0
    inferred: int = 0
    facts_with_evidence: int = 0
    premises: int = 0
    premises_with_evidence: int = 0
    missing_fact_ids: list[str] = field(default_factory=list)
    missing_premise_ids: list[str] = field(default_factory=list)

    @property
    def evidence_precision(self) -> float | None:
        if not self.facts:
            return None
        return round(self.facts_with_evidence / self.facts, 4)

    @property
    def premise_coverage(self) -> float | None:
        if not self.premises:
            return None
        return round(self.premises_with_evidence / self.premises, 4)

    def as_dict(self) -> dict:
        return {"facts": self.facts, "direct": self.direct,
                "inferred": self.inferred,
                "facts_with_evidence": self.facts_with_evidence,
                "evidence_precision": self.evidence_precision,
                "premises": self.premises,
                "premises_with_evidence": self.premises_with_evidence,
                "premise_coverage": self.premise_coverage,
                "missing_fact_ids": self.missing_fact_ids,
                "missing_premise_ids": self.missing_premise_ids}


def audit_answer(dm, answer) -> AnswerAudit:
    """Audit one answer against the evidence log; never mutates anything."""
    evidenced = {e.statement_id for e in dm.evidence.all()}
    audit = AnswerAudit()
    for fact in getattr(answer, "facts", []):
        audit.facts += 1
        if fact.status == "inferred":
            audit.inferred += 1
        else:
            audit.direct += 1
        if fact.statement_id and fact.statement_id in evidenced:
            audit.facts_with_evidence += 1
        elif fact.statement_id:
            audit.missing_fact_ids.append(fact.statement_id)
        for step in fact.proof or []:
            sid = step.get("statement_id") or ""
            if not sid:
                continue
            audit.premises += 1
            if sid in evidenced:
                audit.premises_with_evidence += 1
            else:
                audit.missing_premise_ids.append(sid)
    return audit


def evidence_precision(dm, answers) -> dict:
    """Aggregate audit over several answers."""
    total = AnswerAudit()
    for answer in answers:
        a = audit_answer(dm, answer)
        total.facts += a.facts
        total.direct += a.direct
        total.inferred += a.inferred
        total.facts_with_evidence += a.facts_with_evidence
        total.premises += a.premises
        total.premises_with_evidence += a.premises_with_evidence
        total.missing_fact_ids.extend(a.missing_fact_ids)
        total.missing_premise_ids.extend(a.missing_premise_ids)
    return total.as_dict()


def premise_report(dm, fact) -> dict:
    """Which premises of one fact have a source row."""
    evidenced = {e.statement_id for e in dm.evidence.all()}
    steps = []
    for step in fact.proof or []:
        sid = step.get("statement_id") or ""
        steps.append({"rule": step.get("rule", ""), "statement_id": sid,
                      "evidenced": sid in evidenced})
    return {"statement_id": fact.statement_id, "status": fact.status,
            "depth": fact.depth, "steps": steps,
            "all_premises_evidenced": all(s["evidenced"] for s in steps)
            if steps else None}


def unsupported_premises(dm, answer) -> list[str]:
    return audit_answer(dm, answer).missing_premise_ids
