"""Teach / Remember API (v0.12) — explicit, safe knowledge ingestion.

High-level functions over KnowledgeStore. All provenance is required-ish:
callers may omit it for ad-hoc teaching (defaults to user_teach/manual with
current time), but import/correction flows SHOULD pass explicit provenance.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

from .models import ENTITY_TYPES, FUNCTIONAL_PREDICATES, PARSER_VERSION
from .provenance import Provenance
from .store import KnowledgeStore, utcnow

_PREDICATE_ALIASES = {
    "столица": "capital_of", "столица_страны": "capital_of",
    "автор": "written_by", "написал": "written_by",
    "население": "population_of", "рост": "height_of",
    "такой_же_как": "is_a", "вид": "is_a",
}


def canon_predicate(p: str) -> str:
    p = p.casefold().strip().replace(" ", "_").replace("ё", "е")
    return _PREDICATE_ALIASES.get(p, p)


def teach(store: KnowledgeStore, subject: str, predicate: str,
          object_: str, *, object_is_entity: bool = True,
          subject_type: str = "thing", object_type: str = "thing",
          kind: str = "user_assertion", polarity: str = "assert",
          provenance: Optional[Provenance] = None,
          created_at: Optional[str] = None) -> dict:
    """Define entities (if absent), assert the fact, return a receipt dict."""
    pred = canon_predicate(predicate)
    subj = store.define_entity(subject_type, subject, provenance=provenance)
    if object_is_entity:
        obj = store.define_entity(object_type, object_, provenance=provenance)
        stmt = store.assert_statement(subj.id, pred, object_entity=obj.id,
                                      polarity=polarity, kind=kind,
                                      provenance=provenance, created_at=created_at)
    else:
        stmt = store.assert_statement(subj.id, pred, object_literal=object_,
                                      polarity=polarity, kind=kind,
                                      provenance=provenance, created_at=created_at)
    return {"statement_id": stmt.id, "subject_id": subj.id,
            "object_id": stmt.object_entity, "predicate": pred,
            "status": stmt.status, "kind": stmt.kind, "polarity": stmt.polarity}


# spec-phrased alias
remember = teach


def retract(store: KnowledgeStore, statement_id: str, reason: str = "",
            provenance: Optional[Provenance] = None) -> dict:
    ok = store.retract_statement(statement_id, reason=reason, provenance=provenance)
    return {"retracted": ok, "statement_id": statement_id}


def update_statement(store: KnowledgeStore, statement_id: str, *,
                     subject: Optional[str] = None, predicate: Optional[str] = None,
                     object_: Optional[str] = None,
                     provenance: Optional[Provenance] = None) -> dict:
    """Safe update = append-only: old statement retracted (reason=replaced),
    new statement asserted with `replaces` link. History is never rewritten."""
    old = store.statements.get(statement_id)
    if old is None:
        raise KeyError(statement_id)
    subj_name = store.entities[old.subject].name
    new_s = subject or subj_name
    new_p = predicate or old.predicate
    if object_ is None:
        new_o = (store.entities[old.object_entity].name if old.object_entity
                 else old.object_literal)
        new_obj_is_entity = bool(old.object_entity)
    else:
        # explicit replacement is a (named) entity — documented convention
        new_o = object_
        new_obj_is_entity = True
    prov = provenance or Provenance(source="user_teach",
                                    doc_id=f"update:{utcnow()}",
                                    method="correction",
                                    parser_version=PARSER_VERSION,
                                    added_at=utcnow())
    result = teach(store, new_s, new_p, new_o,
                   object_is_entity=new_obj_is_entity,
                   subject_type=store.entities[old.subject].type,
                   kind=old.kind, polarity=old.polarity, provenance=prov)
    store.retract_statement(statement_id, reason=f"replaced_by:{result['statement_id']}",
                            provenance=prov)
    # link the replacement
    new_stmt = store.statements[result["statement_id"]]
    new_stmt.replaces = statement_id
    return {"old": statement_id, "new": result["statement_id"], "status": "updated"}


def teach_hypothesis(store: KnowledgeStore, subject: str, predicate: str,
                     object_: str, *, provenance: Optional[Provenance] = None,
                     subject_type: str = "thing", object_type: str = "thing") -> dict:
    return teach(store, subject, predicate, object_,
                 subject_type=subject_type, object_type=object_type,
                 kind="hypothesis", provenance=provenance)


def confirm_hypothesis(store: KnowledgeStore, statement_id: str,
                       provenance: Optional[Provenance] = None) -> dict:
    ok = store.promote_hypothesis(statement_id, provenance=provenance)
    return {"promoted": ok, "statement_id": statement_id}
