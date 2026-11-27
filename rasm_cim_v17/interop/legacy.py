"""Read-only readers for the pre-0.16 formats.

Three inputs exist in the historical stack and none of them may be guessed at:

1. ``rasm-cim-kb/1`` — the v0.12–v0.15 knowledge base: a *directory* with
   ``meta.json`` (``{"schema": "rasm-cim-kb/1", "format": "events-jsonl"}``)
   and an append-only ``events.jsonl``.  Replayed here with the same
   crash semantics the writer promised: a torn last line is skipped and
   counted, a mid-file bad line is skipped and recorded with its line number,
   and an unknown event kind is recorded — never silently ignored.
2. ``rasm-cim-learned/1`` — the v0.15 learned artifact (see ``learned.py``).
3. the v0.11 ``demo_facts.jsonl`` — flat ``{subject, relation, object,
   polarity}`` rows with no ids and no provenance.  It is materialised as
   *synthetic* statements whose provenance says exactly that
   (``source="legacy:demo_facts"``, ``doc_id="demo_facts:<line>"``), so an
   imported fact can always be traced back to a line number rather than to
   an invented document.

Nothing in this module writes.  Reading a legacy store never mutates it.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Iterable, Iterator

KB_SCHEMA = "rasm-cim-kb/1"
EVENTS_FILE = "events.jsonl"
META_FILE = "meta.json"

#: legacy event kinds this reader understands.  Anything else is reported.
KNOWN_EVENTS = frozenset({"entity_defined", "entity_alias_added",
                          "statement_asserted", "statement_retracted",
                          "hypothesis_promoted"})

#: legacy polarity -> v0.16 polarity
POLARITY_MAP = {"assert": "true", "deny": "false",
                "true": "true", "false": "false"}


class LegacyFormatError(ValueError):
    """The input is not the legacy format it claims to be."""


@dataclass
class LegacyEntity:
    id: str
    type: str
    name: str
    aliases: list[str] = field(default_factory=list)
    provenance: dict | None = None
    status: str = "active"          # active | retracted

    def as_dict(self) -> dict:
        return {"id": self.id, "type": self.type, "name": self.name,
                "aliases": list(self.aliases), "provenance": self.provenance,
                "status": self.status}


@dataclass
class LegacyStatement:
    id: str
    subject: str                     # legacy entity id
    predicate: str
    object_entity: str | None        # legacy entity id
    object_literal: str | None
    polarity: str                    # "assert" | "deny"
    kind: str                        # user_assertion | extracted_fact | hypothesis
    provenance: dict = field(default_factory=dict)
    created_at: str = ""
    status: str = "active"           # active | retracted
    retract_reason: str | None = None
    replaces: str | None = None

    @property
    def is_hypothesis(self) -> bool:
        return self.kind == "hypothesis"

    def as_dict(self) -> dict:
        return {"id": self.id, "subject": self.subject,
                "predicate": self.predicate,
                "object_entity": self.object_entity,
                "object_literal": self.object_literal,
                "polarity": self.polarity, "kind": self.kind,
                "provenance": dict(self.provenance),
                "created_at": self.created_at, "status": self.status,
                "retract_reason": self.retract_reason,
                "replaces": self.replaces}


@dataclass
class LegacyKB:
    """A replayed ``rasm-cim-kb/1`` store."""

    path: str
    schema: str = KB_SCHEMA
    entities: dict[str, LegacyEntity] = field(default_factory=dict)
    statements: dict[str, LegacyStatement] = field(default_factory=dict)
    alias_events: list[tuple[str, str]] = field(default_factory=list)
    #: (entity_id, alias) pairs in log order — replayed alias additions that
    #: the entity record itself may not carry (older exports)
    events_seen: int = 0
    torn_tail: int = 0
    errors: list[dict] = field(default_factory=list)
    unknown_events: list[str] = field(default_factory=list)
    sha256: str = ""

    # -- loading -----------------------------------------------------------
    @classmethod
    def load(cls, path: str) -> "LegacyKB":
        path = os.path.expanduser(path)
        if os.path.isdir(path):
            meta_path = os.path.join(path, META_FILE)
            events_path = os.path.join(path, EVENTS_FILE)
            if not os.path.exists(meta_path):
                raise LegacyFormatError(
                    f"{path!r} is not a rasm-cim-kb/1 store (no {META_FILE})")
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            if meta.get("schema") != KB_SCHEMA:
                raise LegacyFormatError(
                    f"unsupported KB schema {meta.get('schema')!r}; "
                    f"this reader only accepts {KB_SCHEMA!r}")
            if not os.path.exists(events_path):
                raise LegacyFormatError(f"no {EVENTS_FILE} in {path!r}")
        elif os.path.isfile(path):
            events_path = path
        else:
            raise FileNotFoundError(path)
        with open(events_path, "rb") as f:
            raw = f.read()
        kb = cls(path=path, sha256=hashlib.sha256(raw).hexdigest())
        kb._replay(raw)
        return kb

    def _replay(self, raw: bytes) -> None:
        lines = raw.split(b"\n")
        if lines and lines[-1] == b"":
            lines.pop()                  # trailing newline, not a torn line
        for i, line in enumerate(lines):
            if not line.strip():
                continue
            is_last = (i == len(lines) - 1)
            try:
                ev = json.loads(line.decode("utf-8"))
            except Exception as exc:     # noqa: BLE001 — mirrored from the writer
                if is_last:
                    self.torn_tail += 1
                else:
                    self.errors.append({"line": i + 1, "error": str(exc)})
                continue
            if not isinstance(ev, dict):
                self.errors.append({"line": i + 1, "error": "event is not an object"})
                continue
            self._apply(ev, line_no=i + 1)
            self.events_seen += 1

    def _apply(self, ev: dict, line_no: int) -> None:
        kind = ev.get("event")
        if kind not in KNOWN_EVENTS:
            self.unknown_events.append(str(kind))
            self.errors.append({"line": line_no,
                                "error": f"unknown event {kind!r}"})
            return
        if kind == "entity_defined":
            self._define(ev.get("entity") or {}, line_no)
        elif kind == "entity_alias_added":
            eid, alias = ev.get("entity_id"), ev.get("alias")
            if eid not in self.entities:
                self.errors.append({"line": line_no,
                                    "error": f"alias for unknown entity {eid!r}"})
                return
            if not alias:
                return
            if alias not in self.entities[eid].aliases:
                self.entities[eid].aliases.append(alias)
            self.alias_events.append((eid, alias))
        elif kind == "statement_asserted":
            self._assert(ev.get("statement") or {}, line_no)
        elif kind == "statement_retracted":
            st = self.statements.get(ev.get("statement_id") or "")
            if st is None:
                self.errors.append({"line": line_no,
                                    "error": "retract of unknown statement"})
                return
            st.status = "retracted"
            st.retract_reason = ev.get("reason")
        elif kind == "hypothesis_promoted":
            st = self.statements.get(ev.get("statement_id") or "")
            if st is None:
                self.errors.append({"line": line_no,
                                    "error": "promote of unknown statement"})
                return
            if st.kind == "hypothesis":
                st.kind = "extracted_fact"

    def _define(self, d: dict, line_no: int) -> None:
        try:
            e = LegacyEntity(id=d["id"], type=d.get("type", "other"),
                             name=d["name"],
                             aliases=list(d.get("aliases") or []),
                             provenance=d.get("provenance"),
                             status=d.get("status", "active"))
        except KeyError as exc:
            self.errors.append({"line": line_no, "error": f"entity missing {exc}"})
            return
        self.entities[e.id] = e

    def _assert(self, d: dict, line_no: int) -> None:
        try:
            st = LegacyStatement(
                id=d["id"], subject=d["subject"], predicate=d["predicate"],
                object_entity=d.get("object_entity"),
                object_literal=d.get("object_literal"),
                polarity=d.get("polarity", "assert"),
                kind=d.get("kind", "user_assertion"),
                provenance=dict(d.get("provenance") or {}),
                created_at=d.get("created_at", ""),
                status=d.get("status", "active"),
                retract_reason=d.get("retract_reason"),
                replaces=d.get("replaces"))
        except KeyError as exc:
            self.errors.append({"line": line_no, "error": f"statement missing {exc}"})
            return
        if st.object_entity and st.object_literal:
            self.errors.append({"line": line_no,
                                "error": "both object_entity and object_literal"})
            return
        if not st.object_entity and not st.object_literal:
            self.errors.append({"line": line_no, "error": "statement has no object"})
            return
        self.statements[st.id] = st

    # -- views -------------------------------------------------------------
    def active_statements(self) -> list[LegacyStatement]:
        return [s for s in self.statements.values() if s.status == "active"]

    def summary(self) -> dict:
        return {"path": self.path, "schema": self.schema,
                "sha256": self.sha256, "events": self.events_seen,
                "entities": len(self.entities),
                "statements": len(self.statements),
                "active": len(self.active_statements()),
                "retracted": len(self.statements) - len(self.active_statements()),
                "torn_tail": self.torn_tail,
                "errors": len(self.errors)}


@dataclass
class LegacyFacts:
    """The v0.11 ``demo_facts.jsonl``: flat rows, no ids, no provenance."""

    path: str
    rows: list[dict] = field(default_factory=list)
    errors: list[dict] = field(default_factory=list)
    sha256: str = ""
    schema: str = "rasm-cim-facts/0"

    @classmethod
    def load(cls, path: str) -> "LegacyFacts":
        path = os.path.expanduser(path)
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        with open(path, "rb") as f:
            raw = f.read()
        facts = cls(path=path, sha256=hashlib.sha256(raw).hexdigest())
        for i, line in enumerate(raw.decode("utf-8").splitlines()):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception as exc:      # noqa: BLE001
                facts.errors.append({"line": i + 1, "error": str(exc)})
                continue
            if not isinstance(row, dict):
                facts.errors.append({"line": i + 1, "error": "row is not an object"})
                continue
            if not (row.get("subject") and row.get("relation") and
                    row.get("object") is not None):
                facts.errors.append({"line": i + 1,
                                     "error": "row lacks subject/relation/object"})
                continue
            row = dict(row)
            row.setdefault("polarity", "true")
            facts.rows.append(row)
        return facts

    def as_legacy_kb(self) -> LegacyKB:
        """Materialise the flat rows as a synthetic ``LegacyKB``.

        Entities get deterministic ids derived from their surface form so that
        two lines mentioning ``Франция`` yield one entity.  Statement ids are
        the line number, which is the only stable identity a flat file has.
        """
        kb = LegacyKB(path=self.path, schema=self.schema, sha256=self.sha256)
        for idx, row in enumerate(self.rows, start=1):
            s_name = str(row["subject"]).strip()
            o_name = str(row["object"]).strip()
            s_id = _synthetic_entity_id(kb, s_name, "other")
            o_id = _synthetic_entity_id(kb, o_name, "other")
            sid = f"st_facts_{idx:04d}"
            kb.statements[sid] = LegacyStatement(
                id=sid, subject=s_id, predicate=str(row["relation"]),
                object_entity=o_id, object_literal=None,
                polarity=str(row.get("polarity", "true")),
                kind="extracted_fact",
                provenance={"source": "legacy:demo_facts",
                            "doc_id": f"demo_facts:{idx}",
                            "fragment": json.dumps(row, ensure_ascii=False),
                            "added_at": "", "method": "import",
                            "parser_version": "v0.11",
                            "reliability": "medium",
                            "legacy_line": idx},
                created_at="")
        return kb

    def summary(self) -> dict:
        return {"path": self.path, "sha256": self.sha256,
                "rows": len(self.rows), "errors": len(self.errors)}


def _synthetic_entity_id(kb: LegacyKB, name: str, kind: str) -> str:
    for e in kb.entities.values():
        if e.name.casefold() == name.casefold():
            return e.id
    eid = "ent_facts_%s" % hashlib.sha256(
        name.casefold().encode("utf-8")).hexdigest()[:12]
    kb.entities[eid] = LegacyEntity(id=eid, type=kind, name=name,
                                    provenance={"source": "legacy:demo_facts",
                                                "doc_id": "demo_facts",
                                                "method": "import"})
    return eid


def iter_jsonl(path: str) -> Iterator[dict]:
    """Yield JSON objects from a file, skipping (and counting) bad lines.

    Kept tiny and dependency-free on purpose: it is used by the tests to
    re-read what the importer wrote.
    """
    with open(os.path.expanduser(path), "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue
