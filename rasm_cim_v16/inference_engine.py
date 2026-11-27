"""v0.16 — inference engine: bounded, rule-based, proof-carrying.

Every rule is a record with an explicit applicability condition.  Nothing is
propagated "because graphs are transitive" — transitivity is a property of a
*relation* plus a depth cap plus a confidence decay, and each rule states all
three.

    alias_bridge        alias_of / same_as substitute an entity on either end
                        of ANY query.  conf x0.98 per hop.
    symmetric           s P o  =>  o P s      (only for symmetric relations)
    inverse             s Q o  =>  o P s      (only when inverse(Q) == P)
    instance_subclass   instance_of(x,C) & subclass_of(C,D) => instance_of(x,D)
                        conf x0.90 per taxonomy hop
    transitive          a P b & b P c => a P c (only for transitive relations,
                        depth-capped — this is where runaway inference lives)
    functional_negative nothing; functional conflicts are the contradiction
                        detector's job, not inference's

Guarantees the caller can rely on:
  * ``max_depth`` and ``max_nodes`` are hard bounds — the BFS cannot loop
    (visited-goal set) and cannot explode (node budget).
  * every returned ``Fact`` carries ``proof``: the ordered list of
    ``{rule, statement_id, s, p, o}`` steps that produced it, plus ``depth``.
  * ``status`` is ``direct`` for a stored statement and ``inferred`` for
    anything derived.  Derived facts are returned to the caller only — the
    engine NEVER writes them back into the KnowledgeGraph (spec §6: derived
    conclusions must not be persisted as confirmed facts).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from .knowledge_graph import POL_FALSE, POL_TRUE, ST_DIRECT, ST_INFERRED, KnowledgeGraph
from .relations import RelationSchema

# rule names
R_ALIAS = "alias_bridge"
R_SYMMETRIC = "symmetric"
R_INVERSE = "inverse"
R_INSTANCE_SUBCLASS = "instance_subclass"
R_TRANSITIVE = "transitive"

#: per-hop confidence multipliers
CONF = {
    R_ALIAS: 0.98,
    R_SYMMETRIC: 0.99,
    R_INVERSE: 0.95,
    R_INSTANCE_SUBCLASS: 0.90,
    R_TRANSITIVE: 0.85,
}
#: how far each rule is allowed to chain
RULE_DEPTH = {
    R_ALIAS: 2,
    R_SYMMETRIC: 1,
    R_INVERSE: 1,
    R_INSTANCE_SUBCLASS: 3,
    R_TRANSITIVE: 3,
}

ALIAS_RELATIONS = ("alias_of", "has_alias", "same_as")


@dataclass
class Fact:
    subject: int
    predicate: str
    object: int
    status: str = ST_DIRECT
    polarity: str = POL_TRUE
    confidence: float = 1.0
    depth: int = 0
    proof: list[dict] = field(default_factory=list)
    statement_id: str = ""

    def as_dict(self) -> dict:
        return {"s": self.subject, "p": self.predicate, "o": self.object,
                "status": self.status, "polarity": self.polarity,
                "confidence": round(self.confidence, 4), "depth": self.depth,
                "statement_id": self.statement_id, "proof": list(self.proof)}

    def rules_used(self) -> list[str]:
        return [step["rule"] for step in self.proof]


@dataclass
class Rule:
    """A named inference rule.  ``expand`` returns subgoals."""

    name: str
    description: str
    confidence: float
    max_chain: int
    enabled: bool = True

    def as_dict(self) -> dict:
        return {"name": self.name, "description": self.description,
                "confidence": self.confidence, "max_chain": self.max_chain,
                "enabled": self.enabled}


def builtin_rules() -> list[Rule]:
    return [
        Rule(R_ALIAS, "alias_of/has_alias/same_as substitute an entity on "
                      "either end of a query", CONF[R_ALIAS], RULE_DEPTH[R_ALIAS]),
        Rule(R_SYMMETRIC, "s P o => o P s for symmetric relations",
             CONF[R_SYMMETRIC], RULE_DEPTH[R_SYMMETRIC]),
        Rule(R_INVERSE, "s Q o => o P s when inverse(Q) == P",
             CONF[R_INVERSE], RULE_DEPTH[R_INVERSE]),
        Rule(R_INSTANCE_SUBCLASS, "instance_of(x,C) + subclass_of(C,D) => "
                                  "instance_of(x,D)",
             CONF[R_INSTANCE_SUBCLASS], RULE_DEPTH[R_INSTANCE_SUBCLASS]),
        Rule(R_TRANSITIVE, "a P b + b P c => a P c for transitive relations",
             CONF[R_TRANSITIVE], RULE_DEPTH[R_TRANSITIVE]),
    ]


class InferenceEngine:
    """Bounded forward search over the KnowledgeGraph."""

    def __init__(self, kg: KnowledgeGraph, max_depth: int = 3,
                 max_nodes: int = 400, schema: RelationSchema | None = None,
                 enabled_rules: Sequence[str] | None = None):
        self.kg = kg
        self.schema = schema or kg.schema
        self.max_depth = max_depth
        self.max_nodes = max_nodes
        self.rules: dict[str, Rule] = {r.name: r for r in builtin_rules()}
        if enabled_rules is not None:
            for name, r in self.rules.items():
                r.enabled = name in enabled_rules
        self.stats: dict = {"last_goals": 0, "last_expanded": 0}

    # -- rule management ---------------------------------------------------
    def add_rule(self, rule: Rule) -> None:
        self.rules[rule.name] = rule

    def enable(self, name: str, on: bool = True) -> None:
        if name in self.rules:
            self.rules[name].enabled = on

    def rule_report(self) -> list[dict]:
        return [r.as_dict() for r in self.rules.values()]

    def _on(self, name: str) -> bool:
        r = self.rules.get(name)
        return bool(r and r.enabled)

    # -- alias groups ------------------------------------------------------
    def alias_group(self, entity_id: int, max_hops: int = 2) -> dict[int, int]:
        """{entity_id: hop_count} reachable through alias/same_as links.

        Bounded BFS in both directions; the seed maps to hop 0.
        """
        out = {entity_id: 0}
        q = deque([(entity_id, 0)])
        while q:
            node, d = q.popleft()
            if d >= max_hops:
                continue
            for st in self.kg.neighbours(node):
                if st.predicate not in ALIAS_RELATIONS:
                    continue
                nxt = st.object if st.subject == node else st.subject
                if nxt in out:
                    continue
                out[nxt] = d + 1
                q.append((nxt, d + 1))
        return out

    # -- the search --------------------------------------------------------
    def query(self, subject: int | None = None, predicate: str | None = None,
              obj: int | None = None, polarity: str = POL_TRUE,
              max_depth: int | None = None,
              max_nodes: int | None = None) -> list[Fact]:
        """Find every fact matching the pattern.  ``None`` == variable.

        Returns DIRECT facts first (highest confidence), then INFERRED ones
        ordered by (confidence desc, depth asc, subject, object).
        """
        depth_limit = self.max_depth if max_depth is None else max_depth
        budget = self.max_nodes if max_nodes is None else max_nodes
        goals = 0
        expanded = 0
        # goal = (s, p, o, depth, confidence, proof, chain_counts)
        start = (subject, predicate, obj, 0, 1.0, [], {})
        q: deque = deque([start])
        seen_goals: set = {(subject, predicate, obj)}
        facts: dict[tuple, Fact] = {}
        seen_entities: set = set()

        while q:
            s, p, o, d, conf, proof, chains = q.popleft()
            goals += 1
            if goals > budget:
                break

            # ---- direct answers for this goal ----------------------------
            for f in self._direct(s, p, o, polarity, conf, proof):
                key = (f.subject, f.predicate, f.object, f.polarity)
                prev = facts.get(key)
                if prev is None or f.confidence > prev.confidence:
                    if prev is None or f.status == ST_DIRECT:
                        facts[key] = f
                    elif f.depth < prev.depth:
                        facts[key] = f
            if d >= depth_limit:
                continue

            # ---- expansions ---------------------------------------------
            for nxt in self._expand(s, p, o, d, conf, proof, chains, seen_entities):
                ns, np_, no, nd, nc, nproof, nchains, rule = nxt
                sig = (ns, np_, no)
                if sig in seen_goals:
                    continue
                if ns is not None:
                    seen_entities.add(ns)
                if no is not None:
                    seen_entities.add(no)
                if len(seen_entities) > budget:
                    break
                seen_goals.add(sig)
                q.append((ns, np_, no, nd, nc, nproof, nchains))
                expanded += 1

        self.stats = {"last_goals": goals, "last_expanded": expanded,
                      "last_facts": len(facts)}
        out = list(facts.values())
        out.sort(key=lambda f: (0 if f.status == ST_DIRECT else 1,
                                -f.confidence, f.depth, f.subject, f.object))
        return out

    # -- direct lookups ----------------------------------------------------
    def _direct(self, s, p, o, polarity, conf, proof) -> list[Fact]:
        out = []
        stmts = self.kg.find(subject=s, predicate=p, obj=o, polarity=polarity)
        for st in stmts:
            # A stored statement reached WITHOUT any rule step is DIRECT.
            # The same statement reached through an alias hop / inverse / etc.
            # is an answer the system had to *derive*, and is reported as
            # INFERRED — the user must be able to tell the two apart.
            status = ST_DIRECT if not proof else ST_INFERRED
            f = Fact(subject=st.subject, predicate=st.predicate, object=st.object,
                     status=status, polarity=st.polarity,
                     confidence=conf * st.confidence, depth=len(proof),
                     proof=list(proof) + ([{"rule": "lookup",
                                            "statement_id": st.statement_id,
                                            "s": st.subject, "p": st.predicate,
                                            "o": st.object}] if proof else []),
                     statement_id=st.statement_id)
            out.append(f)
        return out

    # -- expansions --------------------------------------------------------
    def _expand(self, s, p, o, d, conf, proof, chains, seen_entities):
        """Yield subgoals: (s, p, o, depth, conf, proof, chains, rule_name)."""
        out = []

        def bump(rule):
            c = dict(chains)
            c[rule] = c.get(rule, 0) + 1
            return c

        # 1. alias bridge on the subject side
        if self._on(R_ALIAS) and s is not None and chains.get(R_ALIAS, 0) < self.rules[R_ALIAS].max_chain:
            for eid, hops in self.alias_group(s, max_hops=1).items():
                if eid == s:
                    continue
                st_id = self._link_statement(s, eid)
                np_ = proof + [{"rule": R_ALIAS, "statement_id": st_id,
                                "s": s, "p": "alias_of", "o": eid}]
                out.append((eid, p, o, d + 1, conf * CONF[R_ALIAS] ** hops,
                            np_, bump(R_ALIAS), R_ALIAS))

        # 2. alias bridge on the object side
        if self._on(R_ALIAS) and o is not None and chains.get(R_ALIAS, 0) < self.rules[R_ALIAS].max_chain:
            for eid, hops in self.alias_group(o, max_hops=1).items():
                if eid == o:
                    continue
                st_id = self._link_statement(o, eid)
                np_ = proof + [{"rule": R_ALIAS, "statement_id": st_id,
                                "s": o, "p": "alias_of", "o": eid}]
                out.append((s, p, eid, d + 1, conf * CONF[R_ALIAS] ** hops,
                            np_, bump(R_ALIAS), R_ALIAS))

        if p is None:
            return out
        rel = self.schema.get(p)

        # 3. symmetric
        if self._on(R_SYMMETRIC) and rel is not None and rel.symmetric \
                and chains.get(R_SYMMETRIC, 0) < self.rules[R_SYMMETRIC].max_chain:
            np_ = proof + [{"rule": R_SYMMETRIC, "statement_id": "",
                            "s": s, "p": p, "o": o}]
            out.append((o, p, s, d + 1, conf * CONF[R_SYMMETRIC], np_,
                        bump(R_SYMMETRIC), R_SYMMETRIC))

        # 4. inverse
        inv = self.schema.resolve_inverse(p)
        if self._on(R_INVERSE) and inv and chains.get(R_INVERSE, 0) < self.rules[R_INVERSE].max_chain:
            np_ = proof + [{"rule": R_INVERSE, "statement_id": "",
                            "s": s, "p": p, "o": o, "via": inv}]
            out.append((o, inv, s, d + 1, conf * CONF[R_INVERSE], np_,
                        bump(R_INVERSE), R_INVERSE))

        # 5. instance_of composed over a subclass_of chain.
        #    Two directions, because the question may bind either side:
        #      o bound  ("Ньютон - человек?")  -> walk DOWN to subclasses of o
        #                 instance_of(x,C) holds if C is a subclass of o
        #      o free   ("Ньютон - это кто/что?") -> walk UP from the classes x
        #                 already has, offering their superclasses
        if (self._on(R_INSTANCE_SUBCLASS) and p == "instance_of"
                and chains.get(R_INSTANCE_SUBCLASS, 0) < self.rules[R_INSTANCE_SUBCLASS].max_chain):
            if o is not None:
                for st in self.kg.subjects_of("subclass_of", o):
                    np_ = proof + [{"rule": R_INSTANCE_SUBCLASS,
                                    "statement_id": st.statement_id,
                                    "s": st.subject, "p": "subclass_of", "o": o}]
                    out.append((s, "instance_of", st.subject, d + 1,
                                conf * CONF[R_INSTANCE_SUBCLASS], np_,
                                bump(R_INSTANCE_SUBCLASS), R_INSTANCE_SUBCLASS))
            elif s is not None:
                for known in self.kg.objects_of(s, "instance_of"):
                    for st in self.kg.objects_of(known.object, "subclass_of"):
                        np_ = proof + [{"rule": R_INSTANCE_SUBCLASS,
                                        "statement_id": st.statement_id,
                                        "s": known.object, "p": "subclass_of",
                                        "o": st.object}]
                        out.append((s, "instance_of", st.object, d + 1,
                                    conf * CONF[R_INSTANCE_SUBCLASS], np_,
                                    bump(R_INSTANCE_SUBCLASS),
                                    R_INSTANCE_SUBCLASS))

        # 6. generic transitivity
        if (self._on(R_TRANSITIVE) and rel is not None and rel.transitive
                and chains.get(R_TRANSITIVE, 0) < self.rules[R_TRANSITIVE].max_chain):
            if o is None and s is not None:
                for st in self.kg.objects_of(s, p):
                    for st2 in self.kg.objects_of(st.object, p):
                        np_ = proof + [
                            {"rule": R_TRANSITIVE, "statement_id": st.statement_id,
                             "s": s, "p": p, "o": st.object},
                            {"rule": R_TRANSITIVE, "statement_id": st2.statement_id,
                             "s": st.object, "p": p, "o": st2.object}]
                        out.append((s, p, st2.object, d + 1,
                                    conf * CONF[R_TRANSITIVE] ** 2, np_,
                                    bump(R_TRANSITIVE), R_TRANSITIVE))
            elif s is None and o is not None:
                for st in self.kg.subjects_of(p, o):
                    for st2 in self.kg.subjects_of(p, st.subject):
                        np_ = proof + [
                            {"rule": R_TRANSITIVE, "statement_id": st.statement_id,
                             "s": st.subject, "p": p, "o": o},
                            {"rule": R_TRANSITIVE, "statement_id": st2.statement_id,
                             "s": st2.subject, "p": p, "o": st.subject}]
                        out.append((st2.subject, p, o, d + 1,
                                    conf * CONF[R_TRANSITIVE] ** 2, np_,
                                    bump(R_TRANSITIVE), R_TRANSITIVE))
            elif s is not None and o is not None:
                # both ends bound ("палец - часть руки?"): search for the
                # middle instead of enumerating everything reachable from s
                for st in self.kg.objects_of(s, p):
                    np_ = proof + [{"rule": R_TRANSITIVE,
                                    "statement_id": st.statement_id,
                                    "s": s, "p": p, "o": st.object}]
                    out.append((st.object, p, o, d + 1,
                                conf * CONF[R_TRANSITIVE], np_,
                                bump(R_TRANSITIVE), R_TRANSITIVE))
        return out

    def _link_statement(self, a: int, b: int) -> str:
        for rel in ALIAS_RELATIONS:
            sts = self.kg.objects_of(a, rel)
            for st in sts:
                if st.object == b:
                    return st.statement_id
        return ""

    # -- convenience -------------------------------------------------------
    def yes_no(self, subject: int, predicate: str, obj: int,
               polarity: str = POL_TRUE) -> tuple[str, Fact | None]:
        """YES / NO / CONFLICT / UNKNOWN with the supporting fact."""
        pos = self.query(subject, predicate, obj, polarity=POL_TRUE)
        neg = self.query(subject, predicate, obj, polarity=POL_FALSE)
        if pos and neg:
            return "CONTRADICTED", pos[0]
        if polarity == POL_TRUE:
            return ("YES", pos[0]) if pos else (("NO", neg[0]) if neg else ("UNKNOWN", None))
        return ("YES", neg[0]) if neg else (("NO", pos[0]) if pos else ("UNKNOWN", None))

    def explain(self, fact: Fact) -> list[str]:
        """Human-readable proof chain, one line per step."""
        lines = []
        for i, step in enumerate(fact.proof, 1):
            s = self.kg.name_of(step.get("s"))
            o = self.kg.name_of(step.get("o"))
            sid = step.get("statement_id") or "-"
            lines.append(f"{i}. [{step['rule']}] {s} — {step.get('p')} → {o} "
                         f"({sid})")
        if not lines:
            lines.append("direct statement "
                         f"{fact.statement_id or '(unknown id)'}")
        lines.append(f"= {self.kg.name_of(fact.subject)} — {fact.predicate} → "
                     f"{self.kg.name_of(fact.object)}  "
                     f"[{fact.status}, depth {fact.depth}, "
                     f"conf {fact.confidence:.2f}]")
        return lines
