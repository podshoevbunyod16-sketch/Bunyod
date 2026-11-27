"""v0.16 — evidence records and answer explanations.

An answer without provenance is not an answer, it is a guess with good
punctuation.  Every statement written to the graph gets an Evidence record,
and every answer the dialogue produces gets an Explanation built from those
records plus the inference proof chain.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

# evidence kinds
EV_USER = "user_assertion"        # the user said it
EV_INFERENCE = "inference"        # derived by a rule
EV_MIGRATION = "migration"        # converted from a legacy record
EV_IMPORT = "import"              # bulk loaded


@dataclass
class Evidence:
    evidence_id: str
    statement_id: str
    kind: str = EV_USER
    utterance: str = ""
    turn_id: str = ""
    source: str = ""
    created_at: str = ""
    note: str = ""

    def as_dict(self) -> dict:
        return {"id": self.evidence_id, "statement_id": self.statement_id,
                "kind": self.kind, "utterance": self.utterance,
                "turn_id": self.turn_id, "source": self.source,
                "created_at": self.created_at, "note": self.note}

    @classmethod
    def from_dict(cls, d: dict) -> "Evidence":
        return cls(evidence_id=d["id"], statement_id=d["statement_id"],
                   kind=d.get("kind", EV_USER), utterance=d.get("utterance", ""),
                   turn_id=d.get("turn_id", ""), source=d.get("source", ""),
                   created_at=d.get("created_at", ""), note=d.get("note", ""))


class EvidenceManager:
    """Append-only evidence log indexed by statement id."""

    def __init__(self, clock=None):
        self._clock = clock or (lambda: "")
        self._ev: dict[str, Evidence] = {}
        self._by_statement: dict[str, list[str]] = {}
        self._counter = itertools.count(1)

    def add(self, statement_id: str, kind: str = EV_USER, utterance: str = "",
            turn_id: str = "", source: str = "", when: str = "",
            note: str = "") -> Evidence:
        """Add evidence.  Identical (statement, utterance, turn) triples are
        not duplicated — a re-taught fact gains a reinforce counter on the
        statement, not a second identical evidence row."""
        from .normalization import normalize
        want = normalize(utterance)
        for eid in self._by_statement.get(statement_id, []):
            e = self._ev[eid]
            # compared after normalisation: the same utterance typed with an
            # em dash or a hyphen is one piece of evidence, not two
            if (normalize(e.utterance) == want and e.turn_id == turn_id
                    and e.kind == kind):
                return e
        e = Evidence(evidence_id=f"ev-{next(self._counter):06d}",
                     statement_id=statement_id, kind=kind, utterance=utterance,
                     turn_id=turn_id, source=source,
                     created_at=when or self._clock(), note=note)
        self._ev[e.evidence_id] = e
        self._by_statement.setdefault(statement_id, []).append(e.evidence_id)
        return e

    def for_statement(self, statement_id: str) -> list[Evidence]:
        return [self._ev[i] for i in self._by_statement.get(statement_id, [])]

    def all(self) -> list[Evidence]:
        return [self._ev[i] for i in sorted(self._ev)]

    def __len__(self) -> int:
        return len(self._ev)

    def to_dict(self) -> dict:
        return {"evidence": [e.as_dict() for e in self.all()]}

    def load_dict(self, d: dict) -> None:
        for ed in d.get("evidence", []):
            e = Evidence.from_dict(ed)
            self._ev[e.evidence_id] = e
            self._by_statement.setdefault(e.statement_id, []).append(e.evidence_id)
            try:
                n = int(e.evidence_id.split("-")[1])
            except (IndexError, ValueError):
                n = 0
            self._counter = itertools.count(max(n, next(self._counter)) + 1)


def format_evidence(ev: list[Evidence]) -> list[str]:
    lines = []
    for e in ev:
        src = e.utterance or e.source or e.kind
        lines.append(f"  • {e.evidence_id} [{e.kind}] {src}"
                     + (f"  (реплика {e.turn_id})" if e.turn_id else ""))
    return lines or ["  • доказательств не зарегистрировано"]
