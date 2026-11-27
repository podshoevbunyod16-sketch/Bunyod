"""Dialogue corpus: turns plus the referent each pronoun must bind to.

Format (``rasm-cim-dialogue-cases/1``), one JSON object per line:

    {"id": "d01",
     "turns": ["Кто такой Исаак Ньютон?", "А он физик?"],
     "refers_to": [{"turn": 1, "pronoun": "он", "entity": "Исаак Ньютон"}],
     "expect_status": {"1": "DIRECT"},
     "expect_contains": {"1": ["да"]},
     "note": "..."}

Why a separate file from the semantic corpus: a coreference case is a
*sequence*, and a turn's meaning depends on the turns before it.  Measuring it
in isolation would measure nothing.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field

DIALOGUE_SCHEMA = "rasm-cim-dialogue-cases/1"


class DialogueCorpusError(ValueError):
    pass


@dataclass
class Reference:
    turn: int
    pronoun: str
    entity: str

    def as_dict(self) -> dict:
        return {"turn": self.turn, "pronoun": self.pronoun,
                "entity": self.entity}


@dataclass
class DialogueCase:
    id: str
    turns: list[str]
    refers_to: list[Reference] = field(default_factory=list)
    expect_status: dict[int, str] = field(default_factory=dict)
    expect_contains: dict[int, list[str]] = field(default_factory=dict)
    note: str = ""

    def __post_init__(self):
        if not self.turns:
            raise DialogueCorpusError(f"{self.id}: dialogue without turns")
        for ref in self.refers_to:
            if not (0 <= ref.turn < len(self.turns)):
                raise DialogueCorpusError(
                    f"{self.id}: refers_to turn {ref.turn} out of range "
                    f"(0..{len(self.turns) - 1})")

    def as_dict(self) -> dict:
        return {"id": self.id, "turns": list(self.turns),
                "refers_to": [r.as_dict() for r in self.refers_to],
                "expect_status": {str(k): v for k, v in
                                  self.expect_status.items()},
                "expect_contains": {str(k): v for k, v in
                                    self.expect_contains.items()},
                "note": self.note}


@dataclass
class DialogueCorpus:
    path: str
    schema: str
    cases: list[DialogueCase]
    sha256: str = ""

    def summary(self) -> dict:
        return {"path": self.path, "schema": self.schema, "size": len(self.cases),
                "turns": sum(len(c.turns) for c in self.cases),
                "references": sum(len(c.refers_to) for c in self.cases),
                "sha256": self.sha256}


def load_dialogues(path: str) -> DialogueCorpus:
    p = os.path.expanduser(path)
    raw = open(p, "rb").read()
    digest = hashlib.sha256(raw).hexdigest()
    cases: list[DialogueCase] = []
    declared: str | None = None
    for i, line in enumerate(raw.decode("utf-8").splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DialogueCorpusError(f"{p}:{i}: invalid JSON: {exc}") from exc
        if row.get("_meta") == "corpus":
            declared = row.get("schema")
            continue
        refs = [Reference(turn=int(r["turn"]), pronoun=r.get("pronoun", ""),
                          entity=r.get("entity", ""))
                for r in row.get("refers_to") or []]
        if any(not r.entity for r in refs):
            raise DialogueCorpusError(f"{p}:{i}: reference without entity")
        cases.append(DialogueCase(
            id=row.get("id") or f"d{i:02d}",
            turns=[str(t) for t in row.get("turns") or []],
            refers_to=refs,
            expect_status={int(k): v for k, v in
                           (row.get("expect_status") or {}).items()},
            expect_contains={int(k): list(v) for k, v in
                             (row.get("expect_contains") or {}).items()},
            note=row.get("note", "")))
    if declared not in (None, DIALOGUE_SCHEMA):
        raise DialogueCorpusError(f"{p}: unsupported schema {declared!r}; "
                                  f"expected {DIALOGUE_SCHEMA!r}")
    if not cases:
        raise DialogueCorpusError(f"{p}: dialogue corpus is empty")
    return DialogueCorpus(path=p, schema=declared or DIALOGUE_SCHEMA,
                          cases=cases, sha256=digest)
