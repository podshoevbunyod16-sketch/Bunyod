"""v0.16 — typed knowledge graph with provenance and evidence.

Reuses, does not replace, two frozen mechanisms:

  * ``rasm_cim_v9.graph.TruthGraph9`` is the *polarity oracle*.  Every direct
    assertion is mirrored into it (relation names interned to ints), so
    TRUE / FALSE / CONFLICT / UNKNOWN and the symmetric-retract semantics are
    exactly the exhaustively tested v0.9 behaviour (12 tests incl. all
    operation sequences up to length 5 against a shadow model).  v0.16 adds
    the layer that v0.9 lacked — string entities, typed predicates,
    provenance, evidence, confidence — on top of it.
  * ``MobileMemory9`` / the v3 store stays the persistence engine for large
    integer-keyed data; the semantic KB itself is a JSON document (see
    ``persistence.py``) because entity names are strings and the KB is small
    (thousands, not millions, of statements).  Both can coexist.

Statement record fields (spec §8):
    id, subject, predicate, object, polarity, status(direct|inferred),
    confidence, source, created_at, modified_at, kind, links, history,
    provenance{utterance, rule, path, depth, source_statement_ids}
"""
from __future__ import annotations

import itertools
import json
from dataclasses import dataclass, field
from typing import Dict, Iterable, Iterator, Sequence

from .entity_resolver import Entity, EntityResolver
from .relations import DEFAULT_SCHEMA, KIND_CLASS, Relation, RelationSchema

KIND_LITERAL = "LITERAL"

# identity relations that carry no graph information when both sides resolve
# to the same entity — the alias then lives on the entity record itself
IDENTITY_RELATIONS = frozenset({"alias_of", "has_alias", "same_as"})

# answer / statement statuses
ST_DIRECT = "direct"
ST_REDUNDANT = "redundant"
ST_INFERRED = "inferred"

POL_TRUE = "true"
POL_FALSE = "false"


@dataclass
class Statement:
    """One typed triple with full provenance."""

    statement_id: str
    subject: int                       # entity id
    predicate: str                     # relation name
    object: int                        # entity id (literals are entities too)
    polarity: str = POL_TRUE
    status: str = ST_DIRECT
    confidence: float = 1.0
    source: str = ""                   # "teach:<utterance id>" / "rule:..." ...
    created_at: str = ""
    modified_at: str = ""
    reinforce: int = 1                 # how many times the same fact was taught
    evidence: list[str] = field(default_factory=list)
    history: list[dict] = field(default_factory=list)
    provenance: dict = field(default_factory=dict)

    # -- (de)serialisation -------------------------------------------------
    def as_dict(self) -> dict:
        return {"id": self.statement_id, "s": self.subject, "p": self.predicate,
                "o": self.object, "pol": self.polarity, "status": self.status,
                "conf": round(self.confidence, 4), "source": self.source,
                "created_at": self.created_at, "modified_at": self.modified_at,
                "reinforce": self.reinforce, "evidence": list(self.evidence),
                "history": list(self.history), "prov": dict(self.provenance)}

    @classmethod
    def from_dict(cls, d: dict) -> "Statement":
        return cls(statement_id=d["id"], subject=int(d["s"]), predicate=d["p"],
                   object=int(d["o"]), polarity=d.get("pol", POL_TRUE),
                   status=d.get("status", ST_DIRECT),
                   confidence=float(d.get("conf", 1.0)),
                   source=d.get("source", ""),
                   created_at=d.get("created_at", ""),
                   modified_at=d.get("modified_at", ""),
                   reinforce=int(d.get("reinforce", 1)),
                   evidence=list(d.get("evidence", [])),
                   history=list(d.get("history", [])),
                   provenance=dict(d.get("prov", {})))

    def key(self) -> tuple:
        """Identity used for duplicate detection."""
        return (self.subject, self.predicate, self.object, self.polarity)


class KnowledgeGraph:
    """Typed triple store.  Direct assertions only — inference never writes
    here (see ``inference_engine``)."""

    def __init__(self, resolver: EntityResolver | None = None,
                 schema: RelationSchema | None = None, clock=None):
        self.resolver = resolver or EntityResolver()
        self.schema = schema or DEFAULT_SCHEMA
        self._clock = clock or (lambda: "")
        self._stmts: Dict[str, Statement] = {}
        self._by_key: Dict[tuple, str] = {}
        #: (s, p, polarity) -> {o -> statement_id}
        #: polarity is part of the key on purpose: the same triple asserted
        #: both true and false is a CONFLICT, and a single-slot index would
        #: silently let the second assertion hide the first.
        self._spo: Dict[tuple, Dict[int, str]] = {}
        #: (p, o, polarity) -> {s -> statement_id}
        self._pos: Dict[tuple, Dict[int, str]] = {}
        self._by_subject: Dict[int, set[str]] = {}
        self._by_object: Dict[int, set[str]] = {}
        self._by_pred: Dict[str, set[str]] = {}
        self._counter = itertools.count(1)
        self._rel_ids: Dict[str, int] = {}
        self._truth = None                      # lazily built TruthGraph9
        self.type_warnings: list[dict] = []

    # -- TruthGraph9 mirror (polarity oracle) ------------------------------
    @property
    def truth(self):
        if self._truth is None:
            from rasm_cim_v9.graph import TruthGraph9
            self._truth = TruthGraph9(
                functional_relations=self.schema.functional_relations(),
                mode="symmetric")
            for st in self._stmts.values():
                sid, rid, oid = self._tids(st)
                if st.polarity == POL_TRUE:
                    self._truth.assert_true(sid, rid, oid)
                else:
                    self._truth.assert_false(sid, rid, oid)
        return self._truth

    def _tids(self, st: Statement) -> tuple[int, int, int]:
        rid = self._rel_ids.get(st.predicate)
        if rid is None:
            rid = len(self._rel_ids) + 1
            self._rel_ids[st.predicate] = rid
        return st.subject, rid, st.object

    # -- literals ----------------------------------------------------------
    def intern_literal(self, text: str, when: str = "") -> Entity:
        return self.resolver.intern(str(text).strip(), kind=KIND_LITERAL,
                                    source="literal", when=when)

    def is_literal(self, entity_id: int) -> bool:
        e = self.resolver.get(entity_id)
        return bool(e and e.kind == KIND_LITERAL)

    def name_of(self, entity_id: int) -> str:
        e = self.resolver.get(entity_id)
        return e.canonical if e else f"#{entity_id}"

    # -- writes ------------------------------------------------------------
    def assert_triple(self, subject: int, predicate: str, obj: int,
                      polarity: str = POL_TRUE, confidence: float = 1.0,
                      source: str = "", evidence: str | None = None,
                      when: str = "", provenance: dict | None = None,
                      status: str = ST_DIRECT) -> tuple[Statement, bool]:
        """Assert a typed triple.  Returns ``(statement, created)``.

        ``created=False`` means an identical statement already existed; it is
        reinforced (counter++, modified_at bumped, history entry appended)
        rather than duplicated — this is the duplicate guard required by
        TEST K.
        """
        when = when or self._clock()
        if self.schema.get(predicate) is None:
            raise KeyError(f"unknown relation {predicate!r} — register it in "
                           f"the RelationSchema first")
        key = (subject, predicate, obj, polarity)
        existing_id = self._by_key.get(key)
        if existing_id is not None:
            st = self._stmts[existing_id]
            st.reinforce += 1
            st.modified_at = when
            if confidence > st.confidence:
                st.confidence = confidence
            if evidence and evidence not in st.evidence:
                st.evidence.append(evidence)
            st.history.append({"at": when, "change": "reasserted (duplicate)",
                               "source": source})
            return st, False

        rel = self.schema.get(predicate)
        conf = confidence
        skind = (self.resolver.get(subject) or Entity(0, "")).kind
        okind = (self.resolver.get(obj) or Entity(0, "")).kind
        if rel is not None and not rel.allows(skind, okind):
            self.type_warnings.append({
                "at": when, "predicate": predicate,
                "subject_kind": skind, "object_kind": okind,
                "note": f"type constraint violated: {rel.subj_types}/"
                        f"{rel.obj_types}"})
            conf = min(conf, 0.6)          # kept, but visibly weakened

        if subject == obj and predicate in IDENTITY_RELATIONS:
            # "Исаак Ньютон alias_of Исаак Ньютон": both sides resolved to the
            # same entity, because the alias is already carried by the entity
            # record itself.  The statement adds no graph edge, so it is stored
            # (nothing is lost, the migrated provenance survives) but marked
            # redundant, which every index and traversal already skips.
            status = ST_REDUNDANT

        st = Statement(statement_id=self._new_id(), subject=subject,
                       predicate=predicate, object=obj, polarity=polarity,
                       status=status, confidence=conf, source=source,
                       created_at=when, modified_at=when,
                       evidence=[evidence] if evidence else [],
                       provenance=dict(provenance or {}))
        self._stmts[st.statement_id] = st
        self._by_key[key] = st.statement_id
        if status != ST_REDUNDANT:
            self._spo.setdefault((subject, predicate, polarity),
                                 {})[obj] = st.statement_id
            self._pos.setdefault((predicate, obj, polarity),
                                 {})[subject] = st.statement_id
        self._by_subject.setdefault(subject, set()).add(st.statement_id)
        self._by_object.setdefault(obj, set()).add(st.statement_id)
        self._by_pred.setdefault(predicate, set()).add(st.statement_id)
        sid, rid, oid = self._tids(st)
        if status != ST_REDUNDANT:
            if polarity == POL_TRUE:
                self.truth.assert_true(sid, rid, oid)
            else:
                self.truth.assert_false(sid, rid, oid)
        return st, True

    def retract(self, subject: int, predicate: str, obj: int, when: str = "",
                source: str = "") -> Statement | None:
        """Explicit forget.  Symmetric (both polarities) — v0.9 default.
        Nothing is deleted silently: a history entry is always appended and
        the statement is kept with ``status="retracted"``."""
        when = when or self._clock()
        for pol in (POL_TRUE, POL_FALSE):
            sid = self._by_key.get((subject, predicate, obj, pol))
            if sid is None:
                continue
            st = self._stmts[sid]
            st.status = "retracted"
            st.modified_at = when
            st.history.append({"at": when, "change": "retracted", "source": source})
            sid2, rid, oid = self._tids(st)
            self.truth.retract(sid2, rid, oid)
            self._index_remove(st)
            del self._by_key[(subject, predicate, obj, pol)]
            return st
        return None

    def _index_remove(self, st: Statement) -> None:
        self._spo.get((st.subject, st.predicate, st.polarity), {}).pop(st.object, None)
        self._pos.get((st.predicate, st.object, st.polarity), {}).pop(st.subject, None)
        self._by_subject.get(st.subject, set()).discard(st.statement_id)
        self._by_object.get(st.object, set()).discard(st.statement_id)
        self._by_pred.get(st.predicate, set()).discard(st.statement_id)

    def _new_id(self) -> str:
        return f"st-{next(self._counter):06d}"

    # -- reads -------------------------------------------------------------
    def statement(self, statement_id: str) -> Statement | None:
        return self._stmts.get(statement_id)

    def all_statements(self, include_retracted: bool = False,
                       include_redundant: bool = False) -> list[Statement]:
        return [s for s in self._stmts.values()
                if (include_retracted or s.status != "retracted")
                and (include_redundant or s.status != ST_REDUNDANT)]

    def objects_of(self, subject: int, predicate: str,
                   polarity: str = POL_TRUE) -> list[Statement]:
        out = []
        for o, sid in sorted(self._spo.get((subject, predicate, polarity), {}).items()):
            st = self._stmts[sid]
            if st.polarity == polarity and st.status == ST_DIRECT:
                out.append(st)
        return out

    def subjects_of(self, predicate: str, obj: int,
                    polarity: str = POL_TRUE) -> list[Statement]:
        out = []
        for s, sid in sorted(self._pos.get((predicate, obj, polarity), {}).items()):
            st = self._stmts[sid]
            if st.polarity == polarity and st.status == ST_DIRECT:
                out.append(st)
        return out

    def neighbours(self, entity_id: int, polarity: str = POL_TRUE) -> list[Statement]:
        """Every active statement touching the entity, either direction."""
        ids = self._by_subject.get(entity_id, set()) | self._by_object.get(entity_id, set())
        return [self._stmts[i] for i in sorted(ids)
                if self._stmts[i].polarity == polarity
                and self._stmts[i].status == ST_DIRECT]

    def find(self, subject: int | None = None, predicate: str | None = None,
             obj: int | None = None, polarity: str = POL_TRUE) -> list[Statement]:
        if subject is not None and predicate is not None:
            cands = [self._stmts[s] for s in
                     self._spo.get((subject, predicate, polarity), {}).values()]
        elif predicate is not None and obj is not None:
            cands = [self._stmts[s] for s in
                     self._pos.get((predicate, obj, polarity), {}).values()]
        elif subject is not None:
            cands = [self._stmts[s] for s in self._by_subject.get(subject, set())]
        elif obj is not None:
            cands = [self._stmts[s] for s in self._by_object.get(obj, set())]
        elif predicate is not None:
            cands = [self._stmts[s] for s in self._by_pred.get(predicate, set())]
        else:
            cands = list(self._stmts.values())
        out = []
        for st in cands:
            if st.status != ST_DIRECT:
                continue
            if subject is not None and st.subject != subject:
                continue
            if predicate is not None and st.predicate != predicate:
                continue
            if obj is not None and st.object != obj:
                continue
            if polarity is not None and st.polarity != polarity:
                continue
            out.append(st)
        return sorted(out, key=lambda s: s.statement_id)

    def polarity_of(self, subject: int, predicate: str, obj: int) -> str:
        """TRUE / FALSE / CONFLICT / UNKNOWN — delegated to TruthGraph9."""
        sid, rid, oid = subject, self._rel_ids.get(predicate), obj
        if rid is None:
            return "UNKNOWN"
        return self.truth.query(sid, rid, oid)

    def functional_conflict(self, subject: int, predicate: str) -> list[Statement]:
        """For functional predicates: >1 distinct true object == conflict."""
        if predicate not in self.schema.functional_relations():
            return []
        objs = self.objects_of(subject, predicate)
        distinct = {s.object for s in objs}
        return objs if len(distinct) > 1 else []

    def stats(self) -> dict:
        return {
            "entities": len(self.resolver),
            "statements": len([s for s in self._stmts.values()
                               if s.status != "retracted"]),
            "retracted": len([s for s in self._stmts.values()
                              if s.status == "retracted"]),
            "predicates": len(self._by_pred),
            "true": len([s for s in self._stmts.values()
                         if s.polarity == POL_TRUE and s.status == ST_DIRECT]),
            "false": len([s for s in self._stmts.values()
                          if s.polarity == POL_FALSE and s.status == ST_DIRECT]),
            "type_warnings": len(self.type_warnings),
            # self-identity statements kept for provenance but not indexed
            "redundant": len([s for s in self._stmts.values()
                              if s.status == ST_REDUNDANT]),
        }

    # -- (de)serialisation -------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "resolver": self.resolver.to_dict(),
            "statements": [s.as_dict() for s in
                           sorted(self._stmts.values(),
                                  key=lambda x: x.statement_id)],
            "custom_relations": self.schema.to_dict(),
            "type_warnings": list(self.type_warnings),
        }

    def load_dict(self, d: dict) -> None:
        self.resolver = EntityResolver.from_dict(d.get("resolver", {}))
        rels = d.get("custom_relations") or {}
        if rels:
            added = self.schema.merge_custom(RelationSchema.from_dict(rels))
            if added:
                self.schema.__dict__.setdefault("_merged", []).extend(added)
        for sd in d.get("statements", []):
            st = Statement.from_dict(sd)
            self._stmts[st.statement_id] = st
            self._by_key[st.key()] = st.statement_id
            if st.status not in ("retracted", ST_REDUNDANT):
                self._spo.setdefault((st.subject, st.predicate, st.polarity),
                                     {})[st.object] = st.statement_id
                self._pos.setdefault((st.predicate, st.object, st.polarity),
                                     {})[st.subject] = st.statement_id
                self._by_subject.setdefault(st.subject, set()).add(st.statement_id)
                self._by_object.setdefault(st.object, set()).add(st.statement_id)
                self._by_pred.setdefault(st.predicate, set()).add(st.statement_id)
            try:
                n = int(st.statement_id.split("-")[1])
            except (IndexError, ValueError):
                n = 0
            self._counter = itertools.count(max(n, next(self._counter)) + 1)
        self.type_warnings = list(d.get("type_warnings", []))
        self._truth = None        # rebuilt lazily from the loaded statements

    # -- traversal ---------------------------------------------------------
    def walk(self, start: int, max_depth: int = 2,
             max_nodes: int = 500) -> Iterator[tuple[int, list[str], int]]:
        """BFS over the *undirected* view of the graph.  Cycle-safe (visited
        set) and bounded by ``max_depth``/``max_nodes``.  Yields
        ``(entity_id, predicate_path, depth)``."""
        from collections import deque
        seen = {start}
        q: deque[tuple[int, list[str], int]] = deque([(start, [], 0)])
        yielded = 0
        while q and yielded < max_nodes:
            node, path, depth = q.popleft()
            if depth > 0:
                yield node, path, depth
                yielded += 1
            if depth >= max_depth:
                continue
            for st in self.neighbours(node):
                nxt = st.object if st.subject == node else st.subject
                label = st.predicate if st.subject == node else f"^{st.predicate}"
                if nxt in seen:
                    continue
                seen.add(nxt)
                q.append((nxt, path + [label], depth + 1))


def render_phrase(kg: KnowledgeGraph, subject: int, predicate: str, obj: int,
                  negated: bool = False) -> str:
    """Natural-language rendering of one triple.

    Uses the relation's own ``template`` (declared in the schema, not
    hardcoded here) and falls back to the gloss form.  Answer text is always
    built from this, so a new relation added to the schema is rendered
    correctly without touching the answer engine.
    """
    rel = kg.schema.get(predicate)
    s, o = kg.name_of(subject), kg.name_of(obj)
    if rel is not None and rel.template:
        try:
            text = rel.template.format(s=s, o=o)
        except (KeyError, IndexError, ValueError):
            text = ""
        if text:
            # "не Исаак Ньютон — физик" would be wrong Russian; a negated
            # fact is rendered as an explicit denial of the positive form,
            # which is grammatical for every template.
            return f"Неверно, что {text}" if negated else text
    gloss = rel.human if rel and rel.human else predicate
    return f"{s} — {'не ' if negated else ''}{gloss} → {o}"


def render_triple(kg: KnowledgeGraph, st: Statement) -> str:
    """Machine-friendly rendering used by :graph and :explain."""
    rel = kg.schema.get(st.predicate)
    s, o = kg.name_of(st.subject), kg.name_of(st.object)
    gloss = rel.human if rel and rel.human else st.predicate
    neg = "не " if st.polarity == POL_FALSE else ""
    return f"{s} — {neg}{gloss} → {o}"
