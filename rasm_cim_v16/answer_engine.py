"""v0.16 — answer engine.

Seven kinds of answer, all of them real:

    1. concise      one line
    2. normal       the answer plus the directly related facts
    3. detailed     the above + evidence, the reasoning chain and an explicit
                    list of what is NOT known
    4. related-fact list
    5. reasoning explanation (the proof chain, step by step)
    6. a clarification question
    7. an honest UNKNOWN

and six answer statuses (spec §10):
    DIRECT / INFERRED / CONTEXTUAL / AMBIGUOUS / UNKNOWN / CONTRADICTED

Every sentence is assembled from statements that exist in the graph.  There is
no template that can produce a fact the KB does not contain — the renderer
only ever iterates over ``Fact`` objects, and the "what is unknown" section is
computed by diffing the relations the schema knows about against the ones that
actually have a statement.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .contradiction_detector import Conflict, ContradictionDetector
from .evidence_manager import Evidence, EvidenceManager, format_evidence
from .inference_engine import ST_DIRECT, ST_INFERRED, Fact, InferenceEngine
from .knowledge_graph import (POL_FALSE, POL_TRUE, KnowledgeGraph,
                              Statement, render_phrase)
from .relations import KIND_CLASS
from .relation_extractor import AskResult

# answer statuses
A_DIRECT = "DIRECT"
A_INFERRED = "INFERRED"
A_CONTEXTUAL = "CONTEXTUAL"
A_AMBIGUOUS = "AMBIGUOUS"
A_UNKNOWN = "UNKNOWN"
A_CONTRADICTED = "CONTRADICTED"

MODE_CONCISE = "concise"
MODE_NORMAL = "normal"
MODE_DETAILED = "detailed"
MODES = (MODE_CONCISE, MODE_NORMAL, MODE_DETAILED)
#: legacy aliases kept for compatibility with earlier naming
MODE_ALIASES = {"short": MODE_CONCISE, "brief": MODE_CONCISE, "min": MODE_CONCISE,
                "full": MODE_DETAILED, "verbose": MODE_DETAILED,
                "long": MODE_DETAILED, "explain": MODE_DETAILED,
                "default": MODE_NORMAL}

#: relations shown in a definition answer, in display order
PROFILE_RELATIONS = ("instance_of", "subclass_of", "alias_of", "has_alias",
                     "discovered", "formulated", "created", "wrote", "known_for",
                     "studied", "located_in", "born_at", "part_of", "has_part",
                     "has_property", "causes", "happened_at", "related_to")
MAX_RELATED_NORMAL = 5
MAX_RELATED_DETAILED = 40


@dataclass
class Answer:
    text: str
    status: str = A_UNKNOWN
    confidence: float = 0.0
    mode: str = MODE_NORMAL
    short: str = ""
    normal: str = ""
    detailed: str = ""
    facts: list[Fact] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)
    explanation: list[str] = field(default_factory=list)
    clarification: str = ""
    missing: dict = field(default_factory=dict)
    conflicts: list[Conflict] = field(default_factory=list)
    used_context: bool = False
    notes: list[str] = field(default_factory=list)

    def render(self, mode: str = MODE_NORMAL) -> str:
        mode = MODE_ALIASES.get(mode, mode)
        return {MODE_CONCISE: self.short or self.text,
                MODE_NORMAL: self.normal or self.text,
                MODE_DETAILED: self.detailed or self.text}.get(mode, self.text)

    def as_dict(self) -> dict:
        return {"status": self.status, "mode": self.mode,
                "confidence": round(self.confidence, 3),
                "concise": self.short, "normal": self.normal,
                "detailed": self.detailed, "text": self.text,
                "facts": [f.as_dict() for f in self.facts],
                "evidence": [e.as_dict() for e in self.evidence],
                "explanation": list(self.explanation),
                "clarification": self.clarification,
                "missing": dict(self.missing),
                "conflicts": [c.as_dict() for c in self.conflicts],
                "used_context": self.used_context, "notes": list(self.notes)}


class AnswerEngine:
    def __init__(self, kg: KnowledgeGraph, inference: InferenceEngine,
                 detector: ContradictionDetector, evidence: EvidenceManager,
                 clock=None):
        self.kg = kg
        self.inf = inference
        self.det = detector
        self.ev = evidence
        self._clock = clock or (lambda: "")

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def name(self, entity_id: int) -> str:
        return self.kg.name_of(entity_id)

    def phrase(self, f: Fact) -> str:
        """One fact as a Russian sentence, via the relation's template."""
        return render_phrase(self.kg, f.subject, f.predicate, f.object,
                             negated=(f.polarity == POL_FALSE))

    def _ev_for(self, facts: list[Fact]) -> list[Evidence]:
        out: list[Evidence] = []
        seen = set()
        for f in facts:
            for sid in [f.statement_id] + [s.get("statement_id", "")
                                           for s in f.proof]:
                if not sid or sid in seen:
                    continue
                seen.add(sid)
                out.extend(self.ev.for_statement(sid))
        return out

    def _conflicts_for(self, facts: list[Fact]) -> list[Conflict]:
        ids = {f.statement_id for f in facts}
        ids |= {s.get("statement_id", "") for f in facts for s in f.proof}
        out = []
        for c in self.det.conflicts():
            if set(c.statement_ids) & ids:
                out.append(c)
        return out

    def _status_from_facts(self, facts: list[Fact], used_context: bool) -> str:
        if not facts:
            return A_UNKNOWN
        if any(f.status == ST_DIRECT and f.depth == 0 for f in facts):
            return A_CONTEXTUAL if used_context else A_DIRECT
        return A_CONTEXTUAL if used_context else A_INFERRED

    def _explain_block(self, facts: list[Fact]) -> list[str]:
        lines = []
        for f in facts[:3]:
            lines.append(f"Цепочка для «{self.name(f.subject)} —"
                         f"{f.predicate}→ {self.name(f.object)}»:")
            lines.extend("  " + l for l in self.inf.explain(f))
        return lines

    # ------------------------------------------------------------------
    # entity profile  (the basis of every definition answer)
    # ------------------------------------------------------------------
    def profile(self, entity_id: int) -> dict:
        """Structured summary of everything the KB really knows."""
        ent = self.kg.resolver.get(entity_id)
        facts: list[Fact] = []
        by_rel: dict[str, list[Fact]] = {}
        for rel in PROFILE_RELATIONS:
            fs = self.inf.query(subject=entity_id, predicate=rel, max_depth=2)
            fs = [f for f in fs if f.object != entity_id]
            if fs:
                by_rel[rel] = fs
                facts.extend(fs)
        rev: dict[str, list[Fact]] = {}
        for rel in ("discovered_by", "formulated_by", "created_by", "written_by",
                    "has_instance", "has_subclass", "contains", "birthplace_of",
                    "event_of", "caused_by"):
            fs = self.inf.query(subject=None, predicate=rel, obj=entity_id,
                                max_depth=1)
            if fs:
                rev[rel] = fs
                facts.extend(fs)
        unknown = [r for r in PROFILE_RELATIONS
                   if r not in by_rel and self.kg.schema.get(r) is not None
                   and r not in ("has_alias", "subclass_of")]
        return {
            "entity_id": entity_id,
            "canonical": ent.canonical if ent else f"#{entity_id}",
            "kind": ent.kind if ent else "",
            "gender": ent.gender if ent else "",
            "aliases": list(ent.aliases) if ent else [],
            "by_relation": by_rel,
            "reverse": rev,
            "facts": facts,
            "unknown_relations": unknown,
        }

    # ------------------------------------------------------------------
    # 1-3: definition / "расскажи о X"
    # ------------------------------------------------------------------
    def answer_definition(self, entity_id: int, mode: str = MODE_NORMAL,
                          used_context: bool = False) -> Answer:
        prof = self.profile(entity_id)
        name = prof["canonical"]
        lines: list[str] = []
        short = ""
        inst = prof["by_relation"].get("instance_of", [])
        if inst:
            classes = []
            for f in inst:
                if self.kg.resolver.get(f.object) and \
                        self.kg.resolver.get(f.object).kind == KIND_CLASS:
                    classes.append(self.name(f.object))
            if classes:
                short = f"{name} — {', '.join(classes[:3])}."
        if not short:
            short = f"{name}: прямых классифицирующих фактов в базе нет."
        lines.append(short)

        alias = prof["by_relation"].get("alias_of", [])
        has_alias = prof["by_relation"].get("has_alias", [])
        aliases = sorted(({self.name(f.object) for f in alias} |
                          {self.name(f.subject) for f in has_alias} |
                          set(prof["aliases"])) - {name})
        if aliases:
            lines.append("Также известен как: " + ", ".join(aliases) + ".")

        limit = MAX_RELATED_NORMAL if mode != MODE_DETAILED else MAX_RELATED_DETAILED
        related: list[str] = []
        shown: set[tuple] = set()          # the reverse pass re-derives the
                                           # same facts through the inverse
                                           # rule; never list one twice
        # facts already used for the headline / alias line are "shown" too —
        # has_instance is the inverse of instance_of and would otherwise echo
        # the very fact the first sentence states.
        for rel in ("instance_of", "subclass_of", "alias_of", "has_alias"):
            for f in prof["by_relation"].get(rel, []):
                shown.add((f.subject, f.predicate, f.object))
        for rel, fs in prof["by_relation"].items():
            if rel in ("instance_of", "alias_of", "has_alias"):
                continue
            for f in fs[:limit]:
                key = (f.subject, f.predicate, f.object)
                if key in shown:
                    continue
                shown.add(key)
                related.append(self.phrase(f)
                               + ("" if f.status == ST_DIRECT else " (вывод)"))
        for rel, fs in prof["reverse"].items():
            for f in fs[:limit]:
                key = (f.subject, f.predicate, f.object)
                if key in shown:
                    continue
                shown.add(key)
                related.append(self.phrase(f)
                               + ("" if f.status == ST_DIRECT else " (вывод)"))
        if related:
            lines.append("Известные факты:")
            lines.extend(f"  • {r}" for r in related[:limit])

        facts = prof["facts"]
        ev = self._ev_for(facts)
        conf = max((f.confidence for f in facts), default=0.0)
        conflicts = self._conflicts_for(facts)
        status = A_UNKNOWN if not facts else self._status_from_facts(
            [f for f in facts if f.predicate == "instance_of"] or facts,
            used_context)
        if conflicts:
            status = A_CONTRADICTED

        normal = "\n".join(lines)
        det_lines = list(lines)
        # the detailed rendering is ALWAYS built, independent of the
        # mode requested for this call: one Answer object carries all three
        # renderings, so `--answer detailed` cannot silently degrade to
        # the normal one when the answer was built in another mode.
        if prof["unknown_relations"]:
            det_lines.append("Чего база НЕ знает:")
            det_lines.extend(f"  • {r}" for r in prof["unknown_relations"][:12])
        det_lines.append("Доказательства:")
        det_lines.extend(format_evidence(ev))
        expl = self._explain_block(facts)
        if expl:
            det_lines.append("Как получен ответ:")
            det_lines.extend(expl)
        if conflicts:
            det_lines.append("Обнаруженные противоречия:")
            det_lines.extend(f"  • {c.kind}: {c.detail}" for c in conflicts)
        detailed = "\n".join(det_lines)

        return Answer(text=self._pick(mode, short, normal, detailed),
                      status=status, confidence=conf, mode=mode,
                      short=short, normal=normal, detailed=detailed,
                      facts=facts, evidence=ev,
                      explanation=self._explain_block(facts),
                      missing={"unknown_relations": prof["unknown_relations"]},
                      conflicts=conflicts, used_context=used_context)

    # ------------------------------------------------------------------
    # wh-questions
    # ------------------------------------------------------------------
    def answer_wh(self, bind: AskResult, mode: str = MODE_NORMAL,
                  used_context: bool = False) -> Answer:
        if bind.clarification:
            return self._clarification(bind.clarification, bind, mode)
        want_subject = bind.wants == "subject"
        s, o = bind.subject_id, bind.object_id
        p = bind.predicate or "related_to"
        facts: list[Fact] = []
        if want_subject and o is not None:
            facts = self.inf.query(subject=None, predicate=p, obj=o)
        elif not want_subject and s is not None:
            facts = self.inf.query(subject=s, predicate=p, obj=o)
        else:
            return self._unknown(bind, mode, used_context,
                                 "в вопросе нет ни одной известной сущности")

        polarity = bind.polarity
        facts = [f for f in facts if f.polarity == polarity]
        conflicts = self._conflicts_for(facts)
        if conflicts:
            return self._contradicted(facts, conflicts, mode, bind, used_context)
        if not facts:
            return self._unknown(bind, mode, used_context,
                                 f"в базе нет фактов «{p}» для этого запроса",
                                 facts=facts)

        if want_subject:
            names = [self.name(f.subject) for f in facts]
        else:
            names = [self.name(f.object) for f in facts]
        uniq = list(dict.fromkeys(names))
        if len(facts) == 1:
            short = self.phrase(facts[0]) + "."
        elif want_subject:
            rel = self.kg.schema.get(p)
            gloss = rel.human if rel and rel.human else p
            short = (f"{gloss}: {', '.join(uniq)} "
                     f"(по отношению к «{self.name(o)}»).")
        else:
            first = self.phrase(facts[0]).rstrip(".")
            short = f"{first}; также: {', '.join(uniq[1:])}."
        lines = [short]
        limit = MAX_RELATED_NORMAL if mode != MODE_DETAILED else MAX_RELATED_DETAILED
        if mode != MODE_CONCISE:
            lines.append("Основания:")
            for f in facts[:limit]:
                tag = "факт" if f.status == ST_DIRECT else f"вывод (глубина {f.depth})"
                lines.append(f"  • {self.phrase(f)}  [{tag}, "
                             f"уверенность {f.confidence:.2f}, "
                             f"{f.statement_id or '-'}]")
        ev = self._ev_for(facts)
        normal = "\n".join(lines)
        det_lines = list(lines)
        # the detailed rendering is ALWAYS built, independent of the
        # mode requested for this call: one Answer object carries all three
        # renderings, so `--answer detailed` cannot silently degrade to
        # the normal one when the answer was built in another mode.
        det_lines.append("Доказательства:")
        det_lines.extend(format_evidence(ev))
        det_lines.append("Как получен ответ:")
        det_lines.extend(self._explain_block(facts))
        status = self._status_from_facts(facts, used_context)
        return Answer(text=self._pick(mode, short, normal, "\n".join(det_lines)),
                      status=status,
                      confidence=max(f.confidence for f in facts), mode=mode,
                      short=short, normal=normal, detailed="\n".join(det_lines),
                      facts=facts, evidence=ev,
                      explanation=self._explain_block(facts),
                      used_context=used_context, notes=list(bind.notes))

    # ------------------------------------------------------------------
    # yes/no
    # ------------------------------------------------------------------
    def answer_yesno(self, bind: AskResult, mode: str = MODE_NORMAL,
                     used_context: bool = False) -> Answer:
        if bind.clarification:
            return self._clarification(bind.clarification, bind, mode)
        if bind.subject_id is None or bind.object_id is None:
            return self._unknown(bind, mode, used_context,
                                 "одна из сторон вопроса неизвестна базе")
        s, o, p = bind.subject_id, bind.object_id, bind.predicate or "instance_of"
        verdict, fact = self.inf.yes_no(s, p, o, polarity=bind.polarity)
        if verdict == "UNKNOWN" and p == "instance_of":
            # "Физик — учёный?" is a classification question; the KB may hold
            # that classification as a taxonomy edge (subclass_of) rather than
            # as membership (instance_of).  Both are an answer to it, so the
            # taxonomy edge is consulted before declaring UNKNOWN.
            v2, f2 = self.inf.yes_no(s, "subclass_of", o,
                                     polarity=bind.polarity)
            if v2 != "UNKNOWN":
                verdict, fact, p = v2, f2, "subclass_of"
        pos = self.inf.query(s, p, o, polarity=POL_TRUE)
        neg = self.inf.query(s, p, o, polarity=POL_FALSE)
        facts = pos + neg
        conflicts = self._conflicts_for(facts)
        ev = self._ev_for(facts)
        base = render_phrase(self.kg, s, p, o)

        if verdict == "CONTRADICTED" or conflicts:
            return self._contradicted(facts, conflicts, mode, bind, used_context)
        if verdict == "UNKNOWN":
            a = self._unknown(bind, mode, used_context,
                              f"в базе нет ни подтверждения, ни опровержения "
                              f"«{base}»", facts=facts)
            known = self.answer_definition(s, MODE_CONCISE)
            if known.facts:
                a.notes.append("что известно о субъекте: " + known.short)
                # the detailed rendering is ALWAYS built, independent of the
                # mode requested for this call: one Answer object carries all three
                # renderings, so `--answer detailed` cannot silently degrade to
                # the normal one when the answer was built in another mode.
                a.detailed += "\nЧто всё-таки известно:\n" + known.normal
            return a

        yes = (verdict == "YES")
        short = ("Да. " if yes else "Нет. ") + base + "."
        tag = ("факт" if fact.status == ST_DIRECT
               else f"вывод, глубина {fact.depth}, правила "
                    f"{'+'.join(fact.rules_used()) or '-'}")
        lines = [short, f"Основание: {self.phrase(fact)} [{tag}, уверенность "
                        f"{fact.confidence:.2f}, {fact.statement_id or '-'}]"]
        normal = "\n".join(lines)
        det_lines = list(lines)
        # the detailed rendering is ALWAYS built, independent of the
        # mode requested for this call: one Answer object carries all three
        # renderings, so `--answer detailed` cannot silently degrade to
        # the normal one when the answer was built in another mode.
        det_lines.append("Доказательства:")
        det_lines.extend(format_evidence(ev))
        det_lines.append("Как получен ответ:")
        det_lines.extend(self.inf.explain(fact))
        status = self._status_from_facts([fact], used_context)
        return Answer(text=self._pick(mode, short, normal, "\n".join(det_lines)),
                      status=status, confidence=fact.confidence, mode=mode,
                      short=short, normal=normal, detailed="\n".join(det_lines),
                      facts=[fact], evidence=ev,
                      explanation=self.inf.explain(fact),
                      used_context=used_context, notes=list(bind.notes))

    # ------------------------------------------------------------------
    # teaching acknowledgement
    # ------------------------------------------------------------------
    def answer_teach(self, result, mode: str = MODE_NORMAL) -> Answer:
        if not result.ok:
            text = ("Не записал: " + result.reason)
            if result.clarification:
                text += "\n" + result.clarification
            return Answer(text=text, status=(A_AMBIGUOUS if result.clarification
                                             else A_UNKNOWN),
                          confidence=result.confidence, mode=mode, short=text,
                          normal=text, detailed=text,
                          clarification=result.clarification,
                          notes=list(result.notes))
        st = result.statement
        if st is None:
            # a retract reports what it removed, not a triple
            short = result.reason or "Готово."
            return Answer(text=short, status=A_DIRECT, mode=mode, short=short,
                          normal=short, detailed=short,
                          notes=list(result.notes))
        triple = render_phrase(self.kg, st.subject, st.predicate, st.object,
                               negated=(st.polarity == POL_FALSE))
        short = (f"Записал: {triple} [уверенность {st.confidence:.2f}, "
                 f"{st.statement_id}]")
        if not result.created:
            short = (f"Это уже известно — повтор усилен, дубликат не создан "
                     f"(повторов: {st.reinforce}) [{st.statement_id}]")
        lines = [short]
        normal = short
        det_lines = [short, f"Отношение: {st.predicate}",
                     f"Полярность: {st.polarity}",
                     f"Источник: {st.source or '-'}",
                     f"Разбор: {st.provenance.get('parse', '-')}"]
        if st.provenance.get("notes"):
            det_lines.append("Заметки разбора:")
            det_lines.extend(f"  • {n}" for n in st.provenance["notes"])
        det_lines.append("Доказательства:")
        det_lines.extend(format_evidence(self.ev.for_statement(st.statement_id)))
        detailed = "\n".join(det_lines)
        return Answer(text=self._pick(mode, short, normal, detailed),
                      status=A_DIRECT, confidence=st.confidence, mode=mode,
                      short=short, normal=normal, detailed=detailed,
                      evidence=self.ev.for_statement(st.statement_id),
                      notes=list(result.notes))

    # ------------------------------------------------------------------
    # UNKNOWN / AMBIGUOUS / CONTRADICTED
    # ------------------------------------------------------------------
    def _unknown(self, bind: AskResult, mode: str, used_context: bool,
                 why: str, facts: list[Fact] | None = None) -> Answer:
        short = "Не знаю. " + (why[:1].upper() + why[1:] if why else
                               "данных недостаточно") + "."
        known: list[str] = []
        for eid, role in ((bind.subject_id, "субъект"), (bind.object_id, "объект")):
            if eid is None:
                continue
            prof = self.profile(eid)
            if prof["facts"]:
                rels = ", ".join(sorted(prof["by_relation"]))
                known.append(f"{role} «{prof['canonical']}»: известно {rels}")
            else:
                known.append(f"{role} «{prof['canonical']}»: фактов нет")
        missing = {"reason": why,
                   "subject_id": bind.subject_id, "object_id": bind.object_id,
                   "predicate": bind.predicate}
        lines = [short]
        if known:
            lines.append("Что всё же известно: " + "; ".join(known) + ".")
        normal = "\n".join(lines)
        det_lines = list(lines)
        # the detailed rendering is ALWAYS built, independent of the
        # mode requested for this call: one Answer object carries all three
        # renderings, so `--answer detailed` cannot silently degrade to
        # the normal one when the answer was built in another mode.
        det_lines.append("Чего не хватает для ответа:")
        det_lines.append(f"  • отношение: {bind.predicate or 'не определено'}")
        det_lines.append(f"  • субъект: "
                         f"{self.name(bind.subject_id) if bind.subject_id else 'не определён'}")
        det_lines.append(f"  • объект: "
                         f"{self.name(bind.object_id) if bind.object_id else 'не определён'}")
        det_lines.append("  • статус: UNKNOWN — ответ не выдуман")
        return Answer(text=self._pick(mode, short, normal, "\n".join(det_lines)),
                      status=A_UNKNOWN, confidence=0.0, mode=mode, short=short,
                      normal=normal, detailed="\n".join(det_lines),
                      facts=list(facts or []), missing=missing,
                      used_context=used_context, notes=list(bind.notes))

    def _clarification(self, question: str, bind: AskResult,
                       mode: str) -> Answer:
        return Answer(text=question, status=A_AMBIGUOUS, confidence=0.0,
                      mode=mode, short=question, normal=question,
                      detailed=question + "\n(статус AMBIGUOUS — система не "
                                          "выбирает вариант наугад)",
                      clarification=question, notes=list(bind.notes))

    def _contradicted(self, facts: list[Fact], conflicts: list[Conflict],
                      mode: str, bind: AskResult, used_context: bool) -> Answer:
        short = ("Противоречие в базе: по этому вопросу есть несовместимые "
                 "записи, обе сохранены.")
        lines = [short]
        for c in conflicts:
            lines.append(f"  • {c.kind}: {c.detail}")
            lines.append(f"    источники: {', '.join(c.sources) or '-'}")
        normal = "\n".join(lines)
        det_lines = list(lines)
        # the detailed rendering is ALWAYS built, independent of the
        # mode requested for this call: one Answer object carries all three
        # renderings, so `--answer detailed` cannot silently degrade to
        # the normal one when the answer was built in another mode.
        det_lines.append("Записи:")
        for f in facts:
            det_lines.append(f"  • {f.statement_id or '-'} "
                             f"{self.name(f.subject)} — {f.predicate} → "
                             f"{self.name(f.object)} "
                             f"[{f.polarity}, {f.confidence:.2f}]")
        det_lines.append("Доказательства:")
        det_lines.extend(format_evidence(self._ev_for(facts)))
        return Answer(text=self._pick(mode, short, normal, "\n".join(det_lines)),
                      status=A_CONTRADICTED, confidence=0.0, mode=mode,
                      short=short, normal=normal, detailed="\n".join(det_lines),
                      facts=facts, conflicts=conflicts,
                      evidence=self._ev_for(facts), used_context=used_context,
                      notes=list(bind.notes))

    # ------------------------------------------------------------------
    @staticmethod
    def _pick(mode: str, short: str, normal: str, detailed: str) -> str:
        mode = MODE_ALIASES.get(mode, mode)
        return {MODE_CONCISE: short, MODE_NORMAL: normal,
                MODE_DETAILED: detailed}.get(mode, normal)
