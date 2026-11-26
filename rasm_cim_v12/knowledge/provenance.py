"""Provenance — where every fact comes from (v0.12).

Fields (required unless noted):
  source         — logical source: "user_teach" | "import" | "corpus:<doc_id>" | "rule:<id>" | "correction:<id>"
  doc_id         — id of the source document/record (may equal source for ad-hoc input)
  fragment       — the exact source fragment (original words) when available, else ""
  position       — (start, end) char offsets of fragment in its source, or None
  added_at       — ISO-8601 UTC timestamp of ingestion
  method         — "manual" | "extracted" | "rule" | "correction" | "import"
  parser_version — version of the parser/learner that produced the fact (e.g. "lang-0.11.0", "teach-0.12.0")
  reliability    — categorical: "low" | "medium" | "high" (default "medium")
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional, Tuple

_RELIABILITY = ("low", "medium", "high")
_METHODS = ("manual", "extracted", "rule", "correction", "import")


@dataclass(frozen=True)
class Provenance:
    source: str
    doc_id: str
    fragment: str = ""
    position: Optional[Tuple[int, int]] = None
    added_at: str = ""
    method: str = "manual"
    parser_version: str = "teach-0.12.0"
    reliability: str = "medium"

    def __post_init__(self):
        if self.method not in _METHODS:
            raise ValueError(f"method must be one of {_METHODS}, got {self.method!r}")
        if self.reliability not in _RELIABILITY:
            raise ValueError(f"reliability must be one of {_RELIABILITY}, got {self.reliability!r}")
        if not self.source or not self.doc_id:
            raise ValueError("source and doc_id are required")

    def to_dict(self) -> dict:
        d = asdict(self)
        d["position"] = list(self.position) if self.position else None
        return d

    @staticmethod
    def from_dict(d: dict) -> "Provenance":
        pos = d.get("position")
        return Provenance(source=d["source"], doc_id=d["doc_id"],
                          fragment=d.get("fragment", ""),
                          position=tuple(pos) if pos else None,
                          added_at=d.get("added_at", ""),
                          method=d.get("method", "manual"),
                          parser_version=d.get("parser_version", "teach-0.12.0"),
                          reliability=d.get("reliability", "medium"))
