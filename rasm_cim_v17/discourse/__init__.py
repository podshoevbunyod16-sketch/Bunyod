"""Plan C — context and coreference.

The heavy lifting already exists in ``rasm_cim_v16.context_engine``: salience
ranking, gender agreement, a tie margin and an explicit refusal with a question.
v0.17 does **not** re-implement it.  It does three things v0.16 cannot:

1. **Binds MeaningIR to the context.**  A pronoun in the meaning layer carries
   no referent; ``bind_pronouns`` asks the v0.16 resolver and writes the answer
   back into the focus, together with the reason — including refusals.
2. **Keeps the context alive for repaired turns.**  A repaired utterance is
   executed by v0.17, outside ``DialogueManager.say``, so the pipeline would
   otherwise leave the salience state untouched and break the *next* pronoun.
   ``record_turn`` performs exactly the bookkeeping ``say`` does.
3. **Measures coreference.**  ``evaluate_coreference`` runs whole dialogues and
   compares the resolved referent with the expected one.

What it still does not do: no ellipsis reconstruction beyond what v0.16 does, no
discourse-level inference, no "the former / the latter".  Those are Plan E.
"""
from __future__ import annotations

from .binding import (apply_context, bind_pronouns, record_turn,
                      resolved_referents)
from .dialogue_corpus import DialogueCase, load_dialogues
from .dialogue_metrics import evaluate_coreference

__all__ = ["apply_context", "bind_pronouns", "record_turn",
           "resolved_referents", "DialogueCase", "load_dialogues",
           "evaluate_coreference"]
