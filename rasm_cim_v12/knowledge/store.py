"""KnowledgeStore — event-sourced, append-only JSONL knowledge base (v0.12).

Layout (a directory):
  meta.json    {"schema": "rasm-cim-kb/1", "created_at": iso, "format": "events-jsonl"}
  events.jsonl one event per line, fsync after each append

Replay semantics (recovery after interruption):
  * a TORN LAST line (partial write at crash time) is skipped and counted
    (`replay_torn_tail += 1`) — the store still opens and serves committed data;
  * a MID-FILE invalid line is skipped and recorded in `replay_errors` with its
    line number — never silently swallowed, and never blocks the other events;
  * prefix consistency: applying any prefix of the log yields the state that
    prefix implies (property-tested with hypothesis).

Idempotency: re-applying the same event is a no-op (entity by id, statement by
id, retract by id). Statement ids derive from content+doc_id, so re-import of
the same export re-creates identical ids and zero duplicates.
"""
from __future__ import annotations

import json
import os
import time
from typing import Iterable, Iterator, List, Optional

from .models import (ENTITY_TYPES, FUNCTIONAL_PREDICATES, PARSER_VERSION, SCHEMA,
                     STATEMENT_KINDS, Entity, Statement, fold)
from .models import entity_id as _entity_id, statement_id as _statement_id
from .provenance import Provenance

FORMAT = "events-jsonl"


class KnowledgeError(Exception):
    pass


def utcnow() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class KnowledgeStore:
    def __init__(self, path: str):
        self.path = path
        self.events_path = os.path.join(path, "events.jsonl")
        self.meta_path = os.path.join(path, "meta.json")
        self.replay_errors: List[dict] = []
        self.replay_torn_tail = 0
        self.entities: dict[str, Entity] = {}
        self.alias_index: dict[str, set[str]] = {}
        self.statements: dict[str, Statement] = {}
        self._by_subject: dict[str, set[str]] = {}
        self._by_predicate: dict[str, set[str]] = {}
        self._by_object: dict[str, set[str]] = {}
        self._event_count = 0

    # ------------------------------------------------------------- lifecycle
    @classmethod
    def init(cls, path: str) -> "KnowledgeStore":
        os.makedirs(path, exist_ok=True)
        if not os.path.exists(os.path.join(path, "meta.json")):
            meta = {"schema": SCHEMA, "created_at": utcnow(), "format": FORMAT}
            tmp = os.path.join(path, "meta.json.tmp")
            with open(tmp, "w", encoding="utf-8") as fh:
                fh.write(json.dumps(meta, ensure_ascii=False))
                fh.flush(); os.fsync(fh.fileno())
            os.replace(tmp, os.path.join(path, "meta.json"))
        st = cls(path)
        st._check_meta()
        st.replay()
        return st

    def _check_meta(self):
        if not os.path.exists(self.meta_path):
            raise KnowledgeError(f"not a knowledge base dir (no meta.json): {self.path}")
        with open(self.meta_path, encoding="utf-8") as fh:
            meta = json.load(fh)
        if meta.get("schema") != SCHEMA:
            raise KnowledgeError(f"unsupported KB schema {meta.get('schema')!r} (expected {SCHEMA})")

    def replay(self) -> int:
        if not os.path.exists(self.events_path):
            return 0
        with open(self.events_path, "rb") as fh:
            raw = fh.read()
        lines = raw.split(b"\n")
        if lines and lines[-1] == b"":
            lines.pop()
        n = 0
        for i, line in enumerate(lines):
            if not line.strip():
                continue
            is_last = (i == len(lines) - 1)
            try:
                ev = json.loads(line.decode("utf-8"))
            except Exception as exc:
                if is_last:
                    self.replay_torn_tail += 1
                else:
                    self.replay_errors.append({"line": i + 1, "error": str(exc)})
                continue
            self._apply(ev)
            n += 1
        self._event_count = n
        return n

    # ------------------------------------------------------------ replaying
    def _apply(self, ev: dict):
        kind = ev.get("event")
        try:
            if kind == "entity_defined":
                e = Entity.from_dict(ev["entity"])
                self._index_entity(e)
            elif kind == "entity_alias_added":
                e = self.entities.get(ev["entity_id"])
                if e is None:
                    self.replay_errors.append({"event": ev, "error": "alias for unknown entity"})
                    return
                a = ev["alias"]
                if fold(a) not in [fold(x) for x in e.aliases]:
                    e.aliases.append(a)
                self.alias_index.setdefault(fold(a), set()).add(e.id)
            elif kind == "statement_asserted":
                s = Statement.from_dict(ev["statement"])
                self._index_statement(s)
            elif kind == "statement_retracted":
                s = self.statements.get(ev["statement_id"])
                if s is None:
                    self.replay_errors.append({"line_event": ev.get("seq"),
                                               "error": "retract of unknown statement"})
                    return
                s.status = "retracted"
                s.retract_reason = ev.get("reason")
            elif kind == "hypothesis_promoted":
                s = self.statements.get(ev["statement_id"])
                if s is None:
                    self.replay_errors.append({"line_event": ev.get("seq"),
                                               "error": "promote of unknown statement"})
                    return
                if s.kind == "hypothesis":
                    s.kind = "extracted_fact"
            else:
                self.replay_errors.append({"line_event": ev.get("seq"),
                                           "error": f"unknown event {kind!r}"})
        except Exception as exc:  # applied defensively: never crash replay wholesale
            self.replay_errors.append({"line_event": ev.get("seq"), "error": str(exc)})

    def _index_entity(self, e: Entity):
        self.entities[e.id] = e
        self.alias_index.setdefault(fold(e.name), set()).add(e.id)
        for a in e.aliases:
            self.alias_index.setdefault(fold(a), set()).add(e.id)

    def _index_statement(self, s: Statement):
        if s.id in self.statements:
            return  # idempotent re-apply
        self.statements[s.id] = s
        self._by_subject.setdefault(s.subject, set()).add(s.id)
        self._by_predicate.setdefault(fold(s.predicate), set()).add(s.id)
        if s.object_entity:
            self._by_object.setdefault(s.object_entity, set()).add(s.id)

    # --------------------------------------------------------------- writes
    def _append(self, event: dict) -> dict:
        event = dict(event)
        event["schema"] = SCHEMA
        event["seq"] = self._event_count + 1
        event["ts"] = event.get("ts") or utcnow()
        line = json.dumps(event, ensure_ascii=False, separators=(",", ":"))
        with open(self.events_path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
            fh.flush(); os.fsync(fh.fileno())
        self._apply(event)
        self._event_count += 1
        return event

    def define_entity(self, type_: str, name: str, aliases: Optional[List[str]] = None,
                      provenance: Optional[Provenance] = None) -> Entity:
        if type_ not in ENTITY_TYPES:
            raise KnowledgeError(f"unknown entity type {type_!r}")
        prov = provenance or Provenance(source="user_teach", doc_id=f"adhoc:{utcnow()}",
                                        method="manual", parser_version=PARSER_VERSION,
                                        added_at=utcnow())
        eid = _entity_id(type_, name)
        if eid in self.entities:
            e = self.entities[eid]
            for a in (aliases or []):
                self.add_alias(eid, a, provenance=prov)
            return e
        e = Entity(id=eid, type=type_, name=name, aliases=list(aliases or []),
                   provenance=prov.to_dict())
        self._append({"event": "entity_defined", "entity": e.to_dict()})
        return self.entities[eid]

    def add_alias(self, entity_id: str, alias: str,
                  provenance: Optional[Provenance] = None) -> bool:
        e = self.entities.get(entity_id)
        if e is None:
            raise KnowledgeError(f"unknown entity {entity_id}")
        if fold(alias) in [fold(x) for x in e.aliases] or fold(alias) == fold(e.name):
            return False
        self._append({"event": "entity_alias_added", "entity_id": entity_id,
                      "alias": alias,
                      "provenance": (provenance.to_dict() if provenance else
                                     Provenance(source="user_teach", doc_id=f"adhoc:{utcnow()}",
                                                method="manual", added_at=utcnow()).to_dict())})
        return True

    def assert_statement(self, subject: str, predicate: str,
                         object_entity: Optional[str] = None,
                         object_literal: Optional[str] = None,
                         polarity: str = "assert", kind: str = "user_assertion",
                         provenance: Optional[Provenance] = None,
                         created_at: Optional[str] = None,
                         replaces: Optional[str] = None) -> Statement:
        if polarity not in ("assert", "deny"):
            raise KnowledgeError("polarity must be assert|deny")
        if kind not in STATEMENT_KINDS:
            raise KnowledgeError(f"kind must be one of {STATEMENT_KINDS}")
        if bool(object_entity) == bool(object_literal):
            raise KnowledgeError("exactly one of object_entity / object_literal")
        if subject not in self.entities:
            raise KnowledgeError(f"unknown subject entity {subject}")
        if object_entity and object_entity not in self.entities:
            raise KnowledgeError(f"unknown object entity {object_entity}")
        prov = provenance or Provenance(source="user_teach", doc_id=f"adhoc:{utcnow()}",
                                        method="manual", parser_version=PARSER_VERSION,
                                        added_at=created_at or utcnow())
        ts = created_at or utcnow()
        obj_repr = f"ent:{object_entity}" if object_entity else f"lit:{fold(object_literal)}"
        sid = _statement_id(subject, predicate, obj_repr, polarity, kind, prov.doc_id)
        if sid in self.statements:
            return self.statements[sid]      # idempotent
        stmt = Statement(id=sid, subject=subject, predicate=fold(predicate),
                         object_entity=object_entity, object_literal=object_literal,
                         polarity=polarity, kind=kind,
                         provenance=prov.to_dict(), created_at=ts, replaces=replaces)
        self._append({"event": "statement_asserted", "statement": stmt.to_dict()})
        return self.statements[sid]

    def retract_statement(self, statement_id: str, reason: str = "",
                          provenance: Optional[Provenance] = None) -> bool:
        s = self.statements.get(statement_id)
        if s is None:
            raise KnowledgeError(f"unknown statement {statement_id}")
        if s.status == "retracted":
            return False
        prov = provenance or Provenance(source="user_teach", doc_id=f"retract:{utcnow()}",
                                        method="manual", added_at=utcnow())
        self._append({"event": "statement_retracted", "statement_id": statement_id,
                      "reason": reason, "provenance": prov.to_dict()})
        return True

    def promote_hypothesis(self, statement_id: str, provenance: Optional[Provenance] = None) -> bool:
        s = self.statements.get(statement_id)
        if s is None:
            raise KnowledgeError(f"unknown statement {statement_id}")
        if s.kind != "hypothesis":
            return False                       # already confirmed / only hypotheses promote
        prov = provenance or Provenance(source="user_teach", doc_id=f"promote:{utcnow()}",
                                        method="manual", added_at=utcnow(),
                                        parser_version=PARSER_VERSION)
        self._append({"event": "hypothesis_promoted", "statement_id": statement_id,
                      "provenance": prov.to_dict()})
        return True

    # --------------------------------------------------------------- reads
    def get_entity(self, eid: str) -> Optional[Entity]:
        return self.entities.get(eid)

    def find_by_name(self, name: str, type_hint: Optional[str] = None):
        """→ (status, entity|None, candidates). status: found|ambiguous|not_found."""
        ids = set(self.alias_index.get(fold(name), set()))
        if type_hint:
            ids = {i for i in ids if self.entities[i].type == type_hint
                   and self.entities[i].status == "active"}
        else:
            ids = {i for i in ids if self.entities[i].status == "active"}
        if len(ids) == 1:
            return "found", self.entities[next(iter(ids))], []
        if len(ids) > 1:
            return "ambiguous", None, sorted(ids)
        return "not_found", None, []

    def query_statements(self, subject: Optional[str] = None, predicate: Optional[str] = None,
                         object: Optional[str] = None, kind: Optional[str] = None,
                         provenance_source: Optional[str] = None,
                         include_retracted: bool = False,
                         include_hypotheses: bool = True,
                         polarity: Optional[str] = None) -> List[Statement]:
        if subject is not None:
            ids = list(self._by_subject.get(subject, ()))
        elif predicate is not None:
            ids = list(self._by_predicate.get(fold(predicate), ()))
        else:
            ids = list(self.statements)
        out = []
        for sid in ids:
            s = self.statements[sid]
            if not include_retracted and s.status == "retracted":
                continue
            if not include_hypotheses and s.kind == "hypothesis":
                continue
            if predicate is not None and fold(s.predicate) != fold(predicate):
                continue
            if object is not None and s.object_entity != object and s.object_literal != object:
                continue
            if kind is not None and s.kind != kind:
                continue
            if polarity is not None and s.polarity != polarity:
                continue
            if (provenance_source is not None
                    and (s.provenance or {}).get("source") != provenance_source):
                continue
            out.append(s)
        return sorted(out, key=lambda s: (s.created_at, s.id))

    def reverse_links(self, entity_id: str) -> List[Statement]:
        return [self.statements[i] for i in sorted(self._by_object.get(entity_id, ()))
                if self.statements[i].status == "active"]

    @staticmethod
    def _confirmed(s: Statement) -> bool:
        """Hypotheses and retracted statements never count as fact."""
        return s.status == "active" and s.kind in ("user_assertion", "extracted_fact")

    def status_for(self, subject: str, predicate: str,
                   object_entity: Optional[str] = None,
                   object_literal: Optional[str] = None) -> str:
        rows = [s for s in self.query_statements(subject=subject, predicate=predicate)
                if self._confirmed(s)]
        if object_entity or object_literal:
            t = any(s.polarity == "assert" and
                    (s.object_entity == object_entity and s.object_literal == object_literal)
                    for s in rows)
            f = any(s.polarity == "deny" and
                    (s.object_entity == object_entity and s.object_literal == object_literal)
                    for s in rows)
            if t and f: return "CONFLICT"
            if t: return "TRUE"
            if f: return "FALSE"
            return "UNKNOWN"
        # no object: TruthGraph9-compatible functional semantics where applicable
        asserts = [s for s in rows if s.polarity == "assert"]
        denies = [s for s in rows if s.polarity == "deny"]
        by_obj = lambda s: s.object_repr
        if fold(predicate) in FUNCTIONAL_PREDICATES:
            objs = {by_obj(s) for s in asserts}
            if len(objs) > 1:
                return "CONFLICT"
            if {by_obj(s) for s in asserts} & {by_obj(s) for s in denies}:
                return "CONFLICT"
            if len(objs) == 1:
                return "TRUE"
            if denies and not asserts:
                return "FALSE"
            return "UNKNOWN"
        triples_t = {by_obj(s) for s in asserts}
        triples_f = {by_obj(s) for s in denies}
        if triples_t & triples_f:
            return "CONFLICT"
        if triples_t:
            return "TRUE"
        if triples_f:
            return "FALSE"
        return "UNKNOWN"

    def conflicts(self) -> List[dict]:
        """All currently-active conflict groups (never auto-resolved)."""
        out: List[dict] = []
        triple_map: dict[tuple, set] = {}
        for s in self.statements.values():
            if not self._confirmed(s):
                continue
            key = (s.subject, fold(s.predicate), s.object_repr)
            triple_map.setdefault(key, set()).add(s.polarity)
        for (subj, pred, obj), pols in triple_map.items():
            if pols == {"assert", "deny"}:
                ids = [s.id for s in self.statements.values()
                       if (s.subject, fold(s.predicate), s.object_repr) == (subj, pred, obj)
                       and self._confirmed(s)]
                out.append({"type": "polarity_conflict", "subject": subj,
                            "predicate": pred, "object": obj, "statement_ids": sorted(ids)})
        # functional: one subject+predicate, >1 distinct active asserted objects
        func_map: dict[tuple, set] = {}
        for s in self.statements.values():
            if self._confirmed(s) and s.polarity == "assert" and fold(s.predicate) in FUNCTIONAL_PREDICATES:
                func_map.setdefault((s.subject, fold(s.predicate)), set()).add(s.object_repr)
        for (subj, pred), objs in func_map.items():
            if len(objs) > 1:
                ids = [s.id for s in self.statements.values()
                       if s.subject == subj and fold(s.predicate) == pred
                       and s.polarity == "assert" and self._confirmed(s)]
                out.append({"type": "functional_conflict", "subject": subj,
                            "predicate": pred, "objects": sorted(objs),
                            "statement_ids": sorted(ids)})
        return out

    def to_truth_graph(self):
        """Projection into TruthGraph9 (confirmed, active statements only)."""
        from rasm_cim_v11.language.graph_adapter import GraphAdapter
        ad = GraphAdapter()
        for s in self.statements.values():
            if not self._confirmed(s):
                continue
            subj = self.entities[s.subject].name
            if s.object_entity:
                obj = self.entities[s.object_entity].name
            else:
                obj = s.object_literal or ""
            ad.assert_fact(subj, s.predicate, obj,
                           "true" if s.polarity == "assert" else "false")
        return ad

    def stats(self) -> dict:
        active = [s for s in self.statements.values() if s.status == "active"]
        return {"schema": SCHEMA,
                "entities": len(self.entities),
                "statements_total": len(self.statements),
                "statements_active": len(active),
                "hypotheses_active": sum(1 for s in active if s.kind == "hypothesis"),
                "retracted": len(self.statements) - len(active),
                "events": self._event_count,
                "replay_errors": len(self.replay_errors),
                "replay_torn_tail": self.replay_torn_tail}

    def iter_events(self) -> Iterator[dict]:
        if not os.path.exists(self.events_path):
            return iter(())
        with open(self.events_path, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    yield json.loads(line)
