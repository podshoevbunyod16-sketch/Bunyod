"""Curriculum — a versioned list of lessons, schema ``rasm-cim-curriculum/1``.

A lesson is a text to say plus the answer it must produce:

    {"id": "l03", "kind": "question", "text": "Где находится Париж?",
     "expect_status": "DIRECT", "expect_contains": ["Франция"]}

``kind`` is ``teach`` (must be recorded), ``retract`` (must remove) or
``question`` (must be answered).  Keeping the expectation next to the lesson is
what makes "обучение" measurable: a lesson without an expectation is a wish.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

CURRICULUM_SCHEMA = "rasm-cim-curriculum/1"

HERE = Path(__file__).resolve().parents[2]
#: the curriculum the CLI measures by default (kept next to the other corpora)
DEFAULT_CURRICULUM = str(HERE / "tests_v17" / "corpus" / "curriculum.jsonl")
KINDS = ("teach", "retract", "question")
#: statuses a recorded lesson may end with
TEACH_STATUSES = ("DIRECT", "CONTEXTUAL")
RETRACT_STATUSES = ("DIRECT", "UNKNOWN")


class CurriculumError(ValueError):
    """The curriculum file is not a valid curriculum."""


@dataclass
class Lesson:
    id: str
    kind: str
    text: str
    expect_status: str = ""
    expect_contains: list[str] = field(default_factory=list)
    note: str = ""

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise CurriculumError("урок без id")
        if self.kind not in KINDS:
            raise CurriculumError(f"{self.id}: неизвестный вид урока {self.kind!r}")
        if not self.text.strip():
            raise CurriculumError(f"{self.id}: урок без текста")
        self.expect_contains = [str(s) for s in self.expect_contains]

    @property
    def expected_status(self) -> str:
        if self.expect_status:
            return self.expect_status
        if self.kind == "teach":
            return "DIRECT"
        return ""

    def as_dict(self) -> dict:
        return {"id": self.id, "kind": self.kind, "text": self.text,
                "expect_status": self.expect_status,
                "expect_contains": list(self.expect_contains),
                "note": self.note}


@dataclass
class Curriculum:
    path: str
    sha256: str
    lessons: list[Lesson]

    @property
    def schema(self) -> str:
        return CURRICULUM_SCHEMA

    def summary(self) -> dict:
        kinds: dict[str, int] = {}
        for lesson in self.lessons:
            kinds[lesson.kind] = kinds.get(lesson.kind, 0) + 1
        return {"schema": CURRICULUM_SCHEMA, "path": self.path,
                "sha256": self.sha256, "size": len(self.lessons),
                "kinds": kinds}


def load_curriculum(path: str | Path) -> Curriculum:
    path = Path(path)
    if not path.exists():
        raise CurriculumError(f"нет файла {path}")
    raw = path.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    lessons: list[Lesson] = []
    seen: set[str] = set()
    for lineno, line in enumerate(raw.decode("utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError as exc:
            raise CurriculumError(f"{path}:{lineno}: не JSON ({exc})") from exc
        if not isinstance(data, dict):
            raise CurriculumError(f"{path}:{lineno}: ожидался объект")
        unknown = set(data) - {"id", "kind", "text", "expect_status",
                               "expect_contains", "note"}
        if unknown:
            raise CurriculumError(f"{path}:{lineno}: неизвестные поля "
                                  f"{sorted(unknown)}")
        try:
            lesson = Lesson(**data)
        except TypeError as exc:
            raise CurriculumError(f"{path}:{lineno}: {exc}") from exc
        if lesson.id in seen:
            raise CurriculumError(f"{path}:{lineno}: повтор id {lesson.id!r}")
        seen.add(lesson.id)
        lessons.append(lesson)
    if not lessons:
        raise CurriculumError(f"{path}: пустой учебный план")
    return Curriculum(path=str(path), sha256=sha, lessons=lessons)
