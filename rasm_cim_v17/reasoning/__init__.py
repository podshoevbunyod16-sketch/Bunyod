"""Plan D — reasoning with evidence, as a *policy* + an *audit*.

v0.16 already derives facts under hard bounds and returns a proof for each one
(``InferenceEngine``).  v0.17 does not re-derive anything.  It adds the three
things the roadmap asks for and the core does not have:

1. ``ReasoningPolicy`` — the bounds are named, explicit and checkable, and a
   fact deeper than the policy allows is refused rather than shown;
2. ``answer_with_inference`` — when v0.16 answers "I do not know" but its own
   engine can *prove* the answer, the pipeline answers with the proof instead
   of a shrug (this is why "Кит — это животное?" used to be UNKNOWN while
   "Кто такой Кит?" already returned the derived class);
3. ``audit_answer`` / ``evidence_precision`` — for every fact quoted in an
   answer: is its statement evidenced, and are the premises of its proof?
   An inference is only as good as the statements it started from.

Nothing here may invent a step: a derived answer is assembled from
``Fact.proof``; if there is no proof, there is no answer.
"""
from __future__ import annotations

from .audit import (audit_answer, evidence_precision, premise_report,
                    unsupported_premises)
from .policy import (DEFAULT_POLICY, ReasoningPolicy, explanation_lines,
                     fact_within_policy)
from .use import answer_with_inference

__all__ = ["ReasoningPolicy", "DEFAULT_POLICY", "fact_within_policy",
           "explanation_lines", "answer_with_inference", "audit_answer",
           "evidence_precision", "premise_report", "unsupported_premises"]
