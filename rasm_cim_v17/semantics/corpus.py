"""Evaluation corpus format: one JSON object per line, schema-checked.

Why a file format instead of a Python list: a corpus is *data*, and its size,
composition and provenance must be inspectable without running the code.  The
schema is explicit (``rasm-cim-eval-cases/1``) and an unknown schema is a hard
error — a silently mis-parsed corpus would produce confident nonsense.

Case fields (all optional except ``text``):

    id              stable id, for talking about a miss without quoting it
    text            the utterance
    intent          expected MeaningIR speech act
    entities        canonical names that must end up in the focus
    answer_status   expected answer status (DIRECT / INFERRED / UNKNOWN / …)
    answer_contains substrings that must appear in the answer (case-folded)
    note            why this case exists (e.g. "known parser defect")

Nothing here is generated at import time: the numbers reported by the metrics
module always refer to a corpus file whose sha256 is reported with them.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field

CASES_SCHEMA = "rasm-cim-eval-cases/1"
SEED_SCHEMA = "rasm-cim-eval-seed/1"


class CorpusError(ValueError):
    pass


@dataclass
class Case:
    text: str
    id: str = ""
    intent: str = ""
    #: ``None`` = this case says nothing about entities (not scored);
    #: ``[]`` = the case asserts that NO entity must be resolved
    entities: list[str] | None = None
    answer_status: str = ""
    answer_contains: list[str] = field(default_factory=list)
    note: str = ""

    def __post_init__(self):
        if not self.text or not self.text.strip():
            raise CorpusError("case without text")
        if not self.id:
            self.id = "case_" + hashlib.sha256(
                self.text.encode("utf-8")).hexdigest()[:10]

    def as_dict(self) -> dict:
        return {"id": self.id, "text": self.text, "intent": self.intent,
                "entities": None if self.entities is None
                else list(self.entities),
                "answer_status": self.answer_status,
                "answer_contains": list(self.answer_contains),
                "note": self.note}


@dataclass
class Corpus:
    path: str
    schema: str
    cases: list[Case]
    sha256: str = ""
    meta: dict = field(default_factory=dict)

    def summary(self) -> dict:
        by_intent: dict[str, int] = {}
        for c in self.cases:
            key = c.intent or "—"
            by_intent[key] = by_intent.get(key, 0) + 1
        return {"path": self.path, "schema": self.schema,
                "sha256": self.sha256, "size": len(self.cases),
                "with_expectations": {
                    "intent": len([c for c in self.cases if c.intent]),
                    "entities": len([c for c in self.cases
                                     if c.entities is not None]),
                    "answer": len([c for c in self.cases
                                   if c.answer_status or c.answer_contains]),
                },
                "by_intent": by_intent, "meta": dict(self.meta)}


def _read_jsonl(path: str) -> list[dict]:
    rows: list[dict] = []
    with open(os.path.expanduser(path), "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise CorpusError(f"{path}:{i}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise CorpusError(f"{path}:{i}: expected an object")
            rows.append(row)
    return rows


def load_cases(path: str) -> Corpus:
    """Load a case corpus; refuses an unknown schema."""
    p = os.path.expanduser(path)
    raw = open(p, "rb").read()
    digest = hashlib.sha256(raw).hexdigest()
    rows = _read_jsonl(p)
    meta: dict = {}
    cases: list[Case] = []
    declared: str | None = None
    for row in rows:
        if row.get("_meta") == "corpus":
            declared = row.get("schema")
            meta = {k: v for k, v in row.items()
                    if k not in ("_meta", "schema")}
            continue
        if declared is None:
            declared = row.get("_schema", CASES_SCHEMA)
        cases.append(Case(
            text=row.get("text", ""), id=row.get("id", ""),
            intent=row.get("intent", ""),
            entities=(list(row["entities"]) if "entities" in row else None),
            answer_status=row.get("answer_status", ""),
            answer_contains=list(row.get("answer_contains") or []),
            note=row.get("note", "")))
    if declared not in (None, CASES_SCHEMA):
        raise CorpusError(f"{p}: unsupported corpus schema {declared!r}; "
                          f"expected {CASES_SCHEMA!r}")
    if not cases:
        raise CorpusError(f"{p}: corpus is empty")
    return Corpus(path=p, schema=declared or CASES_SCHEMA, cases=cases,
                  sha256=digest, meta=meta)


def load_seed(path: str) -> list[str]:
    """Load the utterances used to build the demo KB before answering.

    Each line is a teach instruction the system must accept through its normal
    pipeline; the seed is not a shortcut into the graph.
    """
    p = os.path.expanduser(path)
    out: list[str] = []
    for row in _read_jsonl(p):
        if row.get("_meta") == "seed":
            schema = row.get("schema")
            if schema not in (None, SEED_SCHEMA):
                raise CorpusError(f"{p}: unsupported seed schema {schema!r}")
            continue
        line = row.get("teach") or row.get("text")
        if line:
            out.append(str(line))
    return out
