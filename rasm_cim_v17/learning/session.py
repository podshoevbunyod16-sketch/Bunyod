"""Applying lessons through the semantic pipeline, one at a time.

A lesson result records what actually happened — the status, the text, whether
the expectation held — and the failures are returned, not raised: a curriculum
is a measurement, and a measurement that stops at the first surprise tells you
nothing about the rest.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..pipeline import SemanticPipeline
from .curriculum import RETRACT_STATUSES, TEACH_STATUSES, Curriculum, Lesson


@dataclass
class LessonResult:
    id: str
    kind: str
    text: str
    status: str
    answer: str
    ok: bool
    reason: str = ""
    duplicate: bool = False
    statements_before: int = 0
    statements_after: int = 0

    @property
    def delta(self) -> int:
        return self.statements_after - self.statements_before

    def as_dict(self) -> dict:
        return {"id": self.id, "kind": self.kind, "text": self.text,
                "status": self.status, "ok": self.ok, "reason": self.reason,
                "duplicate": self.duplicate,
                "statements_before": self.statements_before,
                "statements_after": self.statements_after,
                "delta": self.delta,
                "answer": self.answer[:400]}


def _check(lesson: Lesson, status: str, text: str,
           delta: int) -> tuple[bool, str, bool]:
    reasons: list[str] = []
    expected = lesson.expected_status
    if expected and status != expected:
        reasons.append(f"статус {status}, ждали {expected}")
    if lesson.kind == "teach" and status not in TEACH_STATUSES:
        reasons.append("урок не записан в базу")
    duplicate = lesson.kind == "teach" and delta == 0
    if duplicate and status not in TEACH_STATUSES:
        reasons.append("факт не записан и не подтверждён как известный")
    if lesson.kind == "retract" and status not in RETRACT_STATUSES:
        reasons.append(f"откат не подтверждён (статус {status})")
    low = (text or "").casefold()
    for needle in lesson.expect_contains:
        if needle.casefold() not in low:
            reasons.append(f"в ответе нет {needle!r}")
    if duplicate:
        reasons.append("факт уже был известен") if reasons else None
    return (not reasons), "; ".join(reasons), duplicate


def apply_lessons(dm, curriculum: Curriculum, respond=None) -> list[LessonResult]:
    pipeline = SemanticPipeline(dm)
    respond = respond or pipeline.respond
    results: list[LessonResult] = []
    for lesson in curriculum.lessons:
        before = len(dm.kg.all_statements())
        answer = respond(lesson.text)
        after = len(dm.kg.all_statements())
        ok, reason, duplicate = _check(lesson, answer.status,
                                       answer.text or "", after - before)
        results.append(LessonResult(id=lesson.id, kind=lesson.kind,
                                    text=lesson.text, status=answer.status,
                                    answer=answer.text or "", ok=ok,
                                    reason=reason, duplicate=duplicate,
                                    statements_before=before,
                                    statements_after=after))
    return results
