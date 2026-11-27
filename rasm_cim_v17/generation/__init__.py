"""Plan E — coherent generation, as *data* first and text second.

    Answer (v0.16)  ──planner──►  ResponseIR  ──nlg──►  text

The roadmap asks for a discourse planner, a response representation and an NLG
step.  The order matters: a plan you cannot inspect is a string you cannot
test.  So:

* ``ResponseIR`` is a small, versioned record of *what the answer says* — the
  claim, the evidence it rests on, the qualifier that says how sure we are, the
  refusal when we do not know;
* the planner decides the segments, the order and what to leave out (repetition
  against the previous turn, evidence only when the answer is inferred);
* the renderer turns segments into text for three registers — and it is a pure
  function, so the same plan always produces the same sentence.

Nothing here invents content.  Every segment carries the ids it came from
(``fact_ids``, ``statement_ids``); a segment with no ids is only allowed to be a
refusal or a clarification, and the metrics check exactly that.
"""
from __future__ import annotations

from .nlg import MODES, render, render_ir_json
from .planner import plan, plan_from_answer
from .response_ir import (RESPONSE_SCHEMA, ResponseIR, Segment,
                          ResponseIRError, validate)

__all__ = ["RESPONSE_SCHEMA", "ResponseIR", "Segment", "ResponseIRError",
           "validate", "plan", "plan_from_answer", "render", "render_ir_json",
           "MODES"]
