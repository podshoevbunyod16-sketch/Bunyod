"""Rule registry for the v0.14 reasoning pipeline.

Rules are deliberately small, explicit and total. Each rule declares
  id            — stable id used in evidence (rule:R01 ...)
  kind          — "transitive" | "entail" | "compare"
  depth         — logical depth of one application (1)
  preconditions — enforced by construction (only CONFIRMED, active premises)
  fn            — pure function (items, max_depth) -> [DerivedFact]

`items` is a mixed list of Statement (raw KB facts) and DerivedFact.
A rule never invents facts: every derived fact lists the exact premise ids
in support_ids, so evidence stays traceable to source. Cycle-safety: derived
facts carry a depth counter and the pipeline caps it (default 2, configurable,
max 3) plus a global expansion budget (default 64) — a cyclic rule set would
terminate by exhaustion, not hang.
"""
from __future__ import annotations

from typing import List, Optional

from rasm_cim_v12.knowledge.models import fold


class DerivedFact:
    def __init__(self, rule_id: str, subject: str, predicate: str,
                 object_entity: Optional[str], object_literal: Optional[str],
                 support_ids: tuple, depth: int):
        self.rule_id = rule_id
        self.subject = subject
        self.predicate = predicate
        self.object_entity = object_entity
        self.object_literal = object_literal
        self.support_ids = tuple(sorted(set(support_ids)))
        self.depth = depth

    def key(self):
        return (self.subject, fold(self.predicate),
                f"ent:{self.object_entity}" if self.object_entity
                else f"lit:{fold(self.object_literal or '')}")

    def to_dict(self) -> dict:
        return {"rule_id": self.rule_id, "subject": self.subject,
                "predicate": self.predicate,
                "object_entity": self.object_entity,
                "object_literal": self.object_literal,
                "support_ids": list(self.support_ids), "depth": self.depth}


def _fact_edges(items, predicate_fold):
    """(subject, object, support_ids, depth) edges for a predicate."""
    edges = []
    for it in items:
        if isinstance(it, DerivedFact):
            if fold(it.predicate) == predicate_fold and it.object_entity:
                edges.append((it.subject, it.object_entity,
                              tuple(it.support_ids), it.depth))
            continue
        s = it
        if (s.status == "active" and s.kind in ("user_assertion", "extracted_fact")
                and s.polarity == "assert"
                and fold(s.predicate) == predicate_fold and s.object_entity):
            edges.append((s.subject, s.object_entity, (s.id,), 1))
    return edges


def _transitive(items, rule_id, predicate_fold, out_predicate,
                max_depth):
    edges = _fact_edges(items, predicate_fold)
    by_a = {}
    for a, b, ids, d in edges:
        by_a.setdefault(a, []).append((b, ids, d))
    out = []
    for a, mids in by_a.items():
        for b, ids1, d1 in mids:
            for c, ids2, d2 in by_a.get(b, []):
                depth = max(d1, d2) + 1
                if a != c and depth <= max_depth:
                    out.append(DerivedFact(rule_id, a, out_predicate, c, None,
                                           tuple(ids1) + tuple(ids2), depth))
    return out


def r01_is_a_transitive(items, max_depth, rules_cfg):
    return _transitive(items, "R01", "is_a", "is_a", max_depth)


def r02_capital_gives_located_in(items, max_depth, rules_cfg):
    out = []
    for it in items:
        if isinstance(it, DerivedFact):
            continue
        s = it
        if (s.status == "active" and s.kind in ("user_assertion", "extracted_fact")
                and s.polarity == "assert" and fold(s.predicate) == "capital_of"
                and s.object_entity):
            out.append(DerivedFact("R02", s.object_entity, "located_in",
                                   s.subject, None, (s.id,), 1))
    return out


def r03_located_in_transitive(items, max_depth, rules_cfg):
    return _transitive(items, "R03", "located_in", "located_in", max_depth)


RULES = [
    {"id": "R01", "kind": "transitive", "preconditions":
        "is_a(A,B) и is_a(B,C), оба подтверждены; следствие is_a(A,C)",
     "fn": r01_is_a_transitive},
    {"id": "R02", "kind": "entail", "preconditions":
        "capital_of(Страна, Город) подтверждён ⇒ located_in(Город, Страна)",
     "fn": r02_capital_gives_located_in},
    {"id": "R03", "kind": "transitive", "preconditions":
        "located_in(A,B) и located_in(B,C) ⇒ located_in(A,C)",
     "fn": r03_located_in_transitive},
]

ACTIVE_RULE_IDS = tuple(r["id"] for r in RULES)


def premise_key(it):
    if isinstance(it, DerivedFact):
        return it.key()
    s = it
    return (s.subject, fold(s.predicate),
            f"ent:{s.object_entity}" if s.object_entity
            else f"lit:{fold(s.object_literal or '')}")


_CHAIN_PREDICATES = ("is_a", "located_in")


def kb_closure(kb, premises, max_depth: int = 2, max_expansion: int = 64,
               fetch_cap: int = 128):
    """Fixed-point expansion of transitive/entail rules with KB fetch (BFS).

    Worklist starts with the objects of the premise statements; each frontier
    round fetches their chain-predicate statements (deduplicated, confirmed,
    asserted), applies rules, then enqueues objects of freshly derived facts.
    Bounded by max_depth (max derived depth), max_expansion (global derived
    budget) and fetch_cap (raw KB reads). Deterministic fetch order.
    """
    seen_stmts = set()
    raw = []
    for s in premises:
        if s.id not in seen_stmts:
            seen_stmts.add(s.id)
            raw.append(s)
    derived: List[DerivedFact] = []
    seen_d = set()
    fetch_needed = {s.object_entity for s in raw if s.object_entity}
    fetched_entities = set()
    fetched_total = 0
    truncated = None
    for _frontier in range(max_depth + 2):
        # ---- fetch step (deduplicated per entity) -------------------
        for obj in sorted(fetch_needed or ()):
            if obj in fetched_entities:
                continue
            fetched_entities.add(obj)
            for pred in _CHAIN_PREDICATES:
                for stmt in kb.query_statements(subject=obj, predicate=pred):
                    if fetched_total >= fetch_cap:
                        return derived, f"truncated at fetch_cap={fetch_cap}"
                    if stmt.id in seen_stmts:
                        continue
                    if not (stmt.status == "active"
                            and stmt.kind in ("user_assertion", "extracted_fact")
                            and stmt.polarity == "assert"):
                        continue
                    seen_stmts.add(stmt.id)
                    raw.append(stmt)
                    fetched_total += 1
        fetch_needed = set()
        # ---- rule step ------------------------------------------------
        items = list(raw) + list(derived)
        before = len(derived)
        for r in RULES:
            new_facts = r["fn"](items, max_depth, None)
            for f in new_facts:
                if f.depth > max_depth:
                    continue
                if f.key() in seen_d or f.key() in {premise_key_of(x) for x in raw}:
                    continue
                if len(derived) >= max_expansion:
                    return derived, f"truncated at max_expansion={max_expansion}"
                seen_d.add(f.key())
                derived.append(f)
        for f in derived[before:]:
            if f.object_entity:
                fetch_needed.add(f.object_entity)
        if len(derived) == before and not fetch_needed:
            break
    return derived, truncated


def premise_key_of(s):
    return (s.subject, fold(s.predicate),
            f"ent:{s.object_entity}" if s.object_entity
            else f"lit:{fold(s.object_literal or '')}")
