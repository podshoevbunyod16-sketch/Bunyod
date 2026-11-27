"""Reasoning policy: the bounds that decide what may reach an answer.

v0.16's engine takes ``max_depth``/``max_nodes`` at construction.  That is an
implementation detail; what the roadmap asks for is a *policy* that can be
stated, changed and audited: how deep a chain may go, how many goals may be
explored, how deep a fact may be before it is shown at all, and whether an
answer without a proof is allowed (it is not).
"""
from __future__ import annotations

from dataclasses import dataclass

from rasm_cim_v16.inference_engine import Fact

#: a chain of three is already a claim about the world that a user cannot see;
#: everything below is the default the CLI ships with
DEFAULT_POLICY = None      # set after the dataclass is defined


@dataclass(frozen=True)
class ReasoningPolicy:
    #: how many hops the engine may expand
    max_depth: int = 3
    #: hard budget on expanded goals (protects Termux-class devices)
    max_nodes: int = 400
    #: facts deeper than this are not shown even if derived
    max_answer_depth: int = 3
    #: a derived fact must carry a proof to be quoted
    require_proof: bool = True
    #: confidence below which an inferred fact is not quoted
    min_inferred_confidence: float = 0.35

    def as_dict(self) -> dict:
        return {"max_depth": self.max_depth, "max_nodes": self.max_nodes,
                "max_answer_depth": self.max_answer_depth,
                "require_proof": self.require_proof,
                "min_inferred_confidence": self.min_inferred_confidence}

    def apply(self, dm) -> "ReasoningPolicy":
        """Install the bounds into an existing manager (no rebuild)."""
        dm.inference.max_depth = self.max_depth
        dm.inference.max_nodes = self.max_nodes
        return self


DEFAULT_POLICY = ReasoningPolicy()


def fact_within_policy(fact: Fact, policy: ReasoningPolicy = DEFAULT_POLICY
                       ) -> tuple[bool, str]:
    """``(allowed, reason)`` — the single place that decides admissibility."""
    if fact.depth > policy.max_answer_depth:
        return False, (f"глубина {fact.depth} превышает политику "
                       f"{policy.max_answer_depth}")
    if policy.require_proof and fact.status == "inferred" and not fact.proof:
        return False, "вывод без доказательства"
    if fact.status == "inferred" and \
            fact.confidence < policy.min_inferred_confidence:
        return False, (f"уверенность вывода {fact.confidence:.2f} ниже порога "
                       f"{policy.min_inferred_confidence}")
    return True, "ok"


def explanation_lines(dm, fact: Fact) -> list[str]:
    """Proof chain, one line per step, with the evidence id of each premise.

    Uses ``InferenceEngine.explain`` (v0.16) and appends the availability of the
    source row for every premise, because a proof whose premises are
    unevidenced is a chain of guesses.
    """
    lines = list(dm.inference.explain(fact))
    evidenced = {e.statement_id for e in dm.evidence.all()}
    for i, step in enumerate(fact.proof, 1):
        sid = step.get("statement_id") or ""
        if not sid:
            continue
        mark = "источник есть" if sid in evidenced else "ИСТОЧНИКА НЕТ"
        lines.append(f"   шаг {i}: {sid} — {mark}")
    return lines
