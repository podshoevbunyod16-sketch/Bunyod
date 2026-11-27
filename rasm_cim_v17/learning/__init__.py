"""Plan F — learning without destroying what is already known.

The roadmap's demand is not "the system learns".  It is: **after learning, the
old knowledge is still there, still answerable, and the change is reversible**.
So this package measures exactly that:

* ``curriculum`` — a versioned list of lessons (teach / retract / question) with
  the answer each one must produce, so "learned" has a definition;
* ``session`` — applies the lessons through the semantic pipeline (the same
  path a user takes) and reports expectation failures instead of hiding them;
* ``forgetting`` — a snapshot of the knowledge base before and after, a
  retention rate, new/retracted counts, contradictions, the answers to a fixed
  set of guard questions, and a rollback check that restores the pre-learning
  state and compares it bit for bit.

Nothing here writes knowledge except through ``DialogueManager``; a lesson that
the pipeline did not turn into a fact simply does not become one.
"""
from __future__ import annotations

from .curriculum import (CURRICULUM_SCHEMA, DEFAULT_CURRICULUM, Curriculum,
                         CurriculumError, Lesson, load_curriculum)
from .forgetting import (apply_curriculum, compare_states, evaluate_learning,
                         render_learning, rollback_check, snapshot)
from .session import LessonResult, apply_lessons

__all__ = ["CURRICULUM_SCHEMA", "DEFAULT_CURRICULUM", "Curriculum",
           "CurriculumError", "Lesson",
           "load_curriculum", "LessonResult", "apply_lessons",
           "snapshot", "compare_states", "apply_curriculum",
           "evaluate_learning", "rollback_check", "render_learning"]
