"""Plan B — semantic layer (v0.17).

    rasm_cim_v16.language_parser   text  -> QueryIR   (syntax + intent, frozen)
    rasm_cim_v17.semantics         QueryIR -> MeaningIR (what the user MEANT)
                                            -> metrics (is it right, measured)

The split matters: ``QueryIR`` says *what the words are*, ``MeaningIR`` says
*what the utterance commits to* — which speech act, which slot is being asked
for, which presuppositions the question carries, and what remains genuinely
unknown.  Everything downstream (context, reasoning, generation) consumes
``MeaningIR``, so a decision about meaning is made once, explicitly, and is
testable in isolation.

Nothing here asserts an answer.  The layer is allowed to say "I do not know
what this means" — and that is the honest default when the evidence in the
utterance does not determine a meaning.
"""
from __future__ import annotations

from .meaning import (AS_ASK_DEFINITION, AS_ASK_LIST, AS_ASK_WH, AS_ASK_YESNO,
                      AS_COMMAND, AS_GREETING, AS_RETRACT, AS_TELL, AS_UNKNOWN,
                      FOCUS_AMBIGUOUS, FOCUS_PRONOUN, FOCUS_RESOLVED,
                      FOCUS_UNKNOWN, MeaningIR, Focus, Presupposition,
                      build_meaning, meanings_for)
from .corpus import Case, Corpus, load_cases, load_seed
from .metrics import (evaluate_answers, evaluate_entities, evaluate_intents,
                      render_report, run_all)
from .harness import (DEFAULT_CORPUS, DEFAULT_SEED, DEFAULT_V11,
                      build_demo, seed_report)

__all__ = [
    "MeaningIR", "Focus", "Presupposition", "build_meaning", "meanings_for",
    "AS_ASK_DEFINITION", "AS_ASK_WH", "AS_ASK_YESNO", "AS_ASK_LIST", "AS_TELL",
    "AS_RETRACT", "AS_GREETING", "AS_COMMAND", "AS_UNKNOWN",
    "FOCUS_RESOLVED", "FOCUS_AMBIGUOUS", "FOCUS_UNKNOWN", "FOCUS_PRONOUN",
    "Case", "Corpus", "load_cases", "load_seed",
    "evaluate_intents", "evaluate_entities",
    "evaluate_answers", "run_all", "render_report",
    "build_demo", "seed_report",
    "DEFAULT_CORPUS", "DEFAULT_SEED", "DEFAULT_V11",
]
