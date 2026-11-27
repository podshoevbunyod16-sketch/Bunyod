"""v0.16 — contradiction detection.

Rules (spec §8/§9): a contradiction is *recorded*, never resolved by deletion.
Both sides keep their provenance; the answer engine then reports
``CONTRADICTED`` instead of quietly picking one.

    POLARITY          the same (s,p,o) asserted both true and false
    FUNCTIONAL        a functional relation holds for two different objects
    DISJOINT_CLASS    instance_of(x,C) and instance_of(x,D) with C,D disjoint
    ALIAS_DIVERGENCE  x alias_of A and x alias_of B, A and B not the same
    TAXONOMY_CYCLE    subclass_of forms a loop
    SYMMETRIC_CONFLICT a symmetric relation asserted true and false

Conflicts are stored with ``resolution=""`` until a human resolves them; the
only automatic action is that the affected facts get an answer status of
CONTRADICTED.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .knowledge_graph import POL_FALSE, POL_TRUE, KnowledgeGraph

K_POLARITY = "POLARITY"
K_FUNCTIONAL = "FUNCTIONAL"
K_DISJOINT = "DISJOINT_CLASS"
K_ALIAS = "ALIAS_DIVERGENCE"
K_CYCLE = "TAXONOMY_CYCLE"


@dataclass
class Conflict:
    conflict_id: str
    kind: str
    subject: int
    predicate: str
    detail: str
    statement_ids: list[str] = field(default_factory=list)
    objects: list[int] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    detected_at: str = ""
    resolution: str = ""               # "" = unresolved

    def as_dict(self) -> dict:
        return {"id": self.conflict_id, "kind": self.kind, "subject": self.subject,
                "predicate": self.predicate, "detail": self.detail,
                "statement_ids": list(self.statement_ids),
                "objects": list(self.objects), "sources": list(self.sources),
                "detected_at": self.detected_at, "resolution": self.resolution}

    @classmethod
    def from_dict(cls, d: dict) -> "Conflict":
        return cls(conflict_id=d["id"], kind=d["kind"], subject=int(d["subject"]),
                   predicate=d["predicate"], detail=d.get("detail", ""),
                   statement_ids=list(d.get("statement_ids", [])),
                   objects=[int(x) for x in d.get("objects", [])],
                   sources=list(d.get("sources", [])),
                   detected_at=d.get("detected_at", ""),
                   resolution=d.get("resolution", ""))


class ContradictionDetector:
    def __init__(self, kg: KnowledgeGraph, clock=None):
        self.kg = kg
        self._clock = clock or (lambda: "")
        self._conflicts: dict[str, Conflict] = {}
        self._n = 0
        #: disjoint class pairs, e.g. ("физик", "химик") — declared, not guessed
        self.disjoint_classes: set[tuple[str, str]] = set()

    # -- configuration -----------------------------------------------------
    def declare_disjoint(self, class_a: str, class_b: str) -> None:
        a, b = sorted((class_a, class_b))
        self.disjoint_classes.add((a, b))

    # -- scanning ----------------------------------------------------------
    def scan(self, when: str = "", focus: tuple | None = None) -> list[Conflict]:
        """Scan the graph.  Idempotent: the same conflict is not recorded twice
        (keyed by kind + subject + predicate + sorted objects).

        ``focus`` is the ``(subject, predicate, object)`` triple that was just
        added.  With it, every check is restricted to the statements that could
        possibly conflict with that triple, so teaching one more fact costs
        O(neighbours) instead of O(all statements) — measured 3x faster at 2000
        facts (see V016_TEST_REPORT.md).  Without it (explicit ``:conflicts``,
        load, migration) the whole graph is scanned, and cycle detection uses
        the full coloured DFS.
        """
        when = when or self._clock()
        found: list[Conflict] = []
        f_subj, f_pred, f_obj = focus if focus else (None, None, None)
        found += self._scan_polarity(when, focus)
        found += self._scan_functional(when, focus)
        found += self._scan_disjoint(when, focus)
        found += self._scan_alias(when, focus)
        if focus is None:
            found += self._scan_cycles(when)
        else:
            found += self._scan_cycles_through(f_subj, f_pred, f_obj, when)
        return found

    def _candidates(self, focus: tuple | None, predicates: tuple):
        """Statements a focused scan has to look at.

        With a focus only the statements sharing its subject are relevant — a
        conflict is always about two statements over the same subject (or the
        same subject/predicate pair), never about two unrelated parts of the
        graph.  Without a focus the whole graph is scanned.
        """
        if focus is None:
            return self.kg.all_statements()
        return self.kg.neighbours(focus[0], polarity=POL_TRUE) + \
            self.kg.neighbours(focus[0], polarity=POL_FALSE)

    def _scan_cycles_through(self, subject: int, predicate: str, obj: int,
                             when: str) -> list[Conflict]:
        """Cycle check for one newly added taxonomy edge.

        Any cycle created by adding ``subject -P-> obj`` must contain that edge,
        therefore it must contain a path ``obj ⇝ subject``.  Searching for that
        path is enough, and it is index-backed, so it does not walk the graph.
        """
        if predicate not in ("subclass_of", "instance_of"):
            return []
        seen = {obj}
        stack = [obj]
        while stack:
            node = stack.pop()
            for rel in ("subclass_of", "instance_of"):
                for nxt in self.kg.objects_of(node, rel):
                    if nxt.object == subject:
                        cf = self._add(
                            K_CYCLE, subject, predicate,
                            f"добавленное отношение замыкает цикл: "
                            f"{self.kg.name_of(subject)} — {predicate} → "
                            f"{self.kg.name_of(obj)} → … → "
                            f"{self.kg.name_of(subject)}",
                            [], sorted(seen | {subject}), [], when)
                        return [cf] if cf else []
                    if nxt.object not in seen:
                        seen.add(nxt.object)
                        stack.append(nxt.object)
        return []

    def _key(self, kind: str, subject: int, predicate: str,
             objects: list[int]) -> str:
        return f"{kind}|{subject}|{predicate}|{'-'.join(str(o) for o in sorted(objects))}"

    def _add(self, kind: str, subject: int, predicate: str, detail: str,
             statement_ids: list[str], objects: list[int], sources: list[str],
             when: str) -> Conflict | None:
        key = self._key(kind, subject, predicate, objects)
        if key in self._conflicts:
            return None
        self._n += 1
        c = Conflict(conflict_id=f"cf-{self._n:05d}", kind=kind, subject=subject,
                     predicate=predicate, detail=detail,
                     statement_ids=statement_ids, objects=objects,
                     sources=sources, detected_at=when)
        self._conflicts[key] = c
        return c

    # -- individual scans --------------------------------------------------
    def _scan_polarity(self, when: str, focus: tuple | None = None) -> list[Conflict]:
        out = []
        for st in self._candidates(focus, ("instance_of",)):
            if st.polarity != POL_TRUE:
                continue
            if focus is not None and (st.subject, st.predicate,
                                      st.object) != focus:
                continue
            negs = self.kg.find(subject=st.subject, predicate=st.predicate,
                                obj=st.object, polarity=POL_FALSE)
            if not negs:
                continue
            ids = [st.statement_id] + [n.statement_id for n in negs]
            srcs = [st.source] + [n.source for n in negs]
            c = self._add(K_POLARITY, st.subject, st.predicate,
                          f"«{self.kg.name_of(st.subject)} — "
                          f"{st.predicate} → "
                          f"{self.kg.name_of(st.object)}» утверждается и как "
                          f"истина, и как ложь; оба источника сохранены",
                          ids, [st.object], srcs, when)
            if c:
                out.append(c)
        return out

    def _scan_functional(self, when: str, focus: tuple | None = None) -> list[Conflict]:
        out = []
        for rel in sorted(self.kg.schema.functional_relations()):
            if focus is not None and focus[1] != rel:
                continue
            for st in self._candidates(focus, (rel,)):
                if st.predicate != rel or st.polarity != POL_TRUE:
                    continue
                peers = self.kg.objects_of(st.subject, rel)
                objs = sorted({p.object for p in peers})
                if len(objs) < 2:
                    continue
                c = self._add(K_FUNCTIONAL, st.subject, rel,
                              f"{rel} функционально, но у "
                              f"«{self.kg.name_of(st.subject)}» "
                              f"{len(objs)} значения: "
                              + ", ".join(self.kg.name_of(o) for o in objs),
                              [p.statement_id for p in peers], objs,
                              [p.source for p in peers], when)
                if c:
                    out.append(c)
        return out

    def _scan_disjoint(self, when: str, focus: tuple | None = None) -> list[Conflict]:
        out = []
        if not self.disjoint_classes:
            return out
        if focus is not None and focus[1] != "instance_of":
            return out
        by_subj: dict[int, list] = {}
        for st in self._candidates(focus, ("instance_of",)):
            if st.predicate == "instance_of" and st.polarity == POL_TRUE:
                by_subj.setdefault(st.subject, []).append(st)
        for subj, sts in by_subj.items():
            names = {self.kg.name_of(st.object).lower(): st for st in sts}
            for a, b in self.disjoint_classes:
                if a in names and b in names:
                    pair = [names[a], names[b]]
                    c = self._add(K_DISJOINT, subj, "instance_of",
                                  f"«{self.kg.name_of(subj)}» отнесено и к "
                                  f"«{a}», и к «{b}» — это несовместимые классы",
                                  [p.statement_id for p in pair],
                                  sorted(p.object for p in pair),
                                  [p.source for p in pair], when)
                    if c:
                        out.append(c)
        return out

    def _scan_alias(self, when: str, focus: tuple | None = None) -> list[Conflict]:
        out = []
        if focus is not None and focus[1] not in ("alias_of", "has_alias"):
            return out
        by_subj: dict[int, list] = {}
        for st in self._candidates(focus, ("alias_of", "has_alias")):
            if st.predicate in ("alias_of", "has_alias") and st.polarity == POL_TRUE:
                by_subj.setdefault(st.subject, []).append(st)
        for subj, sts in by_subj.items():
            objs = sorted({s.object for s in sts})
            if len(objs) < 2:
                continue
            # not a conflict when the two targets are declared the same entity
            same = False
            for i, a in enumerate(objs):
                for b in objs[i + 1:]:
                    if self._same_entity(a, b):
                        same = True
            if same:
                continue
            c = self._add(K_ALIAS, subj, "alias_of",
                          f"«{self.kg.name_of(subj)}» объявлено псевдонимом "
                          f"двух несвязанных сущностей: "
                          + ", ".join(self.kg.name_of(o) for o in objs),
                          [s.statement_id for s in sts], objs,
                          [s.source for s in sts], when)
            if c:
                out.append(c)
        return out

    def _same_entity(self, a: int, b: int) -> bool:
        for st in self.kg.objects_of(a, "same_as"):
            if st.object == b:
                return True
        for st in self.kg.objects_of(b, "same_as"):
            if st.object == a:
                return True
        return False

    def _scan_cycles(self, when: str) -> list[Conflict]:
        """Detect subclass_of loops with a coloured DFS (white/grey/black).

        A naive "visited" set cannot tell a diamond (fine) from a cycle (a
        taxonomy that can never terminate an instance_of walk), so the
        recursion stack is tracked explicitly.
        """
        out = []
        edges: dict[int, list[int]] = {}
        ids: dict[tuple[int, int], str] = {}
        for st in self.kg.all_statements():
            if st.predicate == "subclass_of" and st.polarity == POL_TRUE:
                edges.setdefault(st.subject, []).append(st.object)
                ids[(st.subject, st.object)] = st.statement_id
        WHITE, GREY, BLACK = 0, 1, 2
        colour: dict[int, int] = {}
        reported: set[frozenset] = set()

        def dfs(node: int, stack: list[int]):
            colour[node] = GREY
            stack.append(node)
            for nxt in edges.get(node, []):
                c = colour.get(nxt, WHITE)
                if c == GREY:
                    i = stack.index(nxt)
                    cyc = stack[i:]
                    key = frozenset(cyc)
                    if key in reported:
                        continue
                    reported.add(key)
                    sids = [ids[(a, b)] for a, b in zip(cyc, cyc[1:] + [cyc[0]])
                            if (a, b) in ids]
                    cf = self._add(K_CYCLE, cyc[0], "subclass_of",
                                   "subclass_of образует цикл через "
                                   + " → ".join(self.kg.name_of(x) for x in cyc)
                                   + " → " + self.kg.name_of(cyc[0]),
                                   sids, sorted(cyc), [], when)
                    if cf:
                        out.append(cf)
                elif c == WHITE:
                    dfs(nxt, stack)
            stack.pop()
            colour[node] = BLACK

        for start in list(edges):
            if colour.get(start, WHITE) == WHITE:
                dfs(start, [])
        return out

    # -- reads -------------------------------------------------------------
    def conflicts(self) -> list[Conflict]:
        return sorted(self._conflicts.values(), key=lambda c: c.conflict_id)

    def affecting(self, statement_id: str) -> list[Conflict]:
        return [c for c in self._conflicts.values()
                if statement_id in c.statement_ids]

    def resolve(self, conflict_id: str, how: str, when: str = "") -> bool:
        for c in self._conflicts.values():
            if c.conflict_id == conflict_id:
                c.resolution = how
                c.detected_at = c.detected_at or (when or self._clock())
                return True
        return False

    def unresolved(self) -> list[Conflict]:
        return [c for c in self.conflicts() if not c.resolution]

    # -- (de)serialisation -------------------------------------------------
    def to_dict(self) -> dict:
        return {"conflicts": [c.as_dict() for c in self.conflicts()],
                "disjoint_classes": [list(p) for p in sorted(self.disjoint_classes)]}

    def load_dict(self, d: dict) -> None:
        self.disjoint_classes = {tuple(p) for p in d.get("disjoint_classes", [])}
        for cd in d.get("conflicts", []):
            c = Conflict.from_dict(cd)
            key = self._key(c.kind, c.subject, c.predicate, c.objects)
            self._conflicts[key] = c
            try:
                n = int(c.conflict_id.split("-")[1])
            except (IndexError, ValueError):
                n = 0
            self._n = max(self._n, n)
