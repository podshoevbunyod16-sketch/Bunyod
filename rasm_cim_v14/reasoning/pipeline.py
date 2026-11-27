"""Deterministic answering pipeline (v0.14).

Stages (spec §7): parse → retrieve → resolve → reason → verify → compose
→ evidence-check. No generative model is involved anywhere: the compose
stage is pure templates over the verified EvidenceSet.

Inputs: a KnowledgeStore and a raw Russian question string.
Output: dict with verdict (TRUE|FALSE|UNKNOWN|CONFLICT — TruthGraph9-compatible),
categorical confidence status (supported|partial|unknown|conflict),
answer text (each substantive sentence carries its evidence ids), the full
EvidenceSet, and the list of missing prerequisites.
"""
from __future__ import annotations

import functools
import re
from typing import List, Optional, Tuple

from rasm_cim_v11.language import analyze, lemma, normalize
from rasm_cim_v12.knowledge import KnowledgeStore
from rasm_cim_v12.knowledge.models import FUNCTIONAL_PREDICATES, fold

from .evidence import EvidenceItem, EvidenceSet
from .rules import DerivedFact, kb_closure

MAX_DEPTH_HARD_CAP = 3
_STEM_RE = re.compile(r"[А-Яа-яЁёA-Za-z0-9\-]+")

_KW_PRED = [
    ("столиц", "capital_of"),
    ("населени", "population_of"), ("жител", "population_of"),
    ("рост", "height_of"),
    ("написал", "written_by"), ("написан", "written_by"), ("автор", "written_by"),
    ("может_летать", "может_летать"), ("лета", "может_летать"),
    ("расположен", "located_in"), ("находится", "located_in"),
    ("существует", "существует"), ("существов", "существует"),
    ("реальн", "существует"),
]
_DEF_TRIGGERS = ("что такое", "кто так", "что значит", "что означает")
_COMPARE_TRIGGERS = ("больше", "выше", "меньше", "ниже", "быстрее", "медленнее",
                     "старше", "моложе", "лучше", "хуже")


@functools.lru_cache(maxsize=32768)
def _stems(text: str) -> set:
    return {lemma(w) for w in _STEM_RE.findall(normalize(text).text.lower())
            if w.strip()}


def _find_entities(kb: KnowledgeStore, text: str, cap: int = 6) -> List[dict]:
    q = _stems(text)
    found = []
    for e in kb.entities.values():
        if e.status != "active":
            continue
        best = None
        for n in [e.name] + list(e.aliases):
            ns = _stems(n)
            if ns and ns <= q and (best is None or len(n) > len(best)):
                best = n
        if best is not None:
            found.append({"id": e.id, "name": e.name, "type": e.type,
                          "matched_on": best})
    found.sort(key=lambda d: -len(d["name"]))
    seen, out = set(), []
    for d in found:
        if d["id"] not in seen:
            out.append(d)
            seen.add(d["id"])
    return out[:cap]


def _predicate_for(normalized: str) -> Optional[str]:
    for kw, pred in _KW_PRED:
        if kw in normalized:
            return pred
    if any(t in normalized for t in _DEF_TRIGGERS):
        return "is_a"
    return None


def _num(s: Optional[str]) -> Optional[float]:
    if not s:
        return None
    cleaned = re.sub(r"[^\d,.\-]", "", str(s).replace(" ", "").replace("_", ""))
    cleaned = cleaned.replace(",", ".")
    try:
        return float(cleaned)
    except ValueError:
        return None


def _ev_item_for_statement(s, role="direct", rule_id=None,
                           support_ids=(), depth: int = 1, note: str = "") -> EvidenceItem:
    prov = s.provenance or {}
    return EvidenceItem(role=role, statement_id=s.id,
                        source=prov.get("source"), doc_id=prov.get("doc_id"),
                        reliability=prov.get("reliability"),
                        rule_id=rule_id, support_ids=tuple(support_ids),
                        depth=depth, note=note)


def _ev_item_for_derived(f: DerivedFact) -> EvidenceItem:
    return EvidenceItem(role="derived", statement_id=None, rule_id=f.rule_id,
                        support_ids=tuple(sorted(f.support_ids)), depth=f.depth)


def _fmt_entity(kb, s) -> str:
    if s.object_entity:
        e = kb.get_entity(s.object_entity)
        return e.name if e else s.object_entity
    return str(s.object_literal)


class ReasoningPipeline:
    """One-shot question → answer, deterministic and evidence-bound."""

    def __init__(self, kb: KnowledgeStore, *, max_depth: int = 2,
                 max_expansion: int = 64):
        self.kb = kb
        self.max_depth = max(1, min(MAX_DEPTH_HARD_CAP, int(max_depth)))
        self.max_expansion = int(max_expansion)

    # stages ---------------------------------------------------------
    def _seed_superset(self, entities):
        seed = []
        seen = set()
        for e in entities:
            for st in self.kb.query_statements(subject=e["id"],
                                               include_retracted=False):
                if st.id not in seen:
                    seen.add(st.id); seed.append(st)
            for pred in ("is_a", "located_in", "capital_of", "written_by", "написал"):
                for st in self.kb.query_statements(predicate=pred,
                                                   object=e["id"]):
                    if st.id not in seen:
                        seen.add(st.id); seed.append(st)
        return seed

    def ask(self, question: str) -> dict:
        norm = normalize(question)
        analysis = analyze(question)
        intent = analysis.intent.intent
        normalized = analysis.normalized if hasattr(analysis, "normalized") else norm.text.lower()

        entities = _find_entities(self.kb, question)
        es = EvidenceSet(intent=intent,
                         subject_names=[e["name"] for e in entities])

        predicate = _predicate_for(normalized)
        es.predicate = predicate

        if intent == "compare" or any(t in normalized for t in _COMPARE_TRIGGERS):
            return self._compare(question, entities, es, normalized)

        if intent == "ask_definition":
            # definitions are is_a-centred: raw facts + closure chain
            entities = entities or []
            if not entities:
                es.missing_prereqs.append("не найдено сущностей из вопроса в памяти")
                r = self._finalize(question, es,
                                   ["Не могу понять, о чём речь — у меня нет такой сущности."],
                                   verdict="UNKNOWN")
                return r
            raw: List = []
            for e in entities:
                raw.extend(self.kb.query_statements(subject=e["id"],
                                                    include_retracted=False))
                raw.extend(s for s in self.kb.query_statements(
                               predicate="is_a", object=e["id"])
                           if s.status == "active")
            seed = self._seed_superset(entities)
            derived, trunc = kb_closure(self.kb, seed,
                                        self.max_depth,
                                        self.max_expansion)
            return self._definition(es, entities, raw, derived, trunc)

        # retrieve
        raw: List = []
        derived: List[DerivedFact] = []
        if entities and predicate is None:
            for e in entities:
                raw.extend(self.kb.query_statements(subject=e["id"],
                                                    include_retracted=False))
            seed = self._seed_superset(entities)
            derived, trunc = kb_closure(self.kb, seed,
                                        self.max_depth,
                                        self.max_expansion)
            return self._definition(es, entities, raw, derived, trunc)
        if not entities:
            es.missing_prereqs.append("не найдено сущностей из вопроса в памяти")
            return self._finalize(question, es,
                                  ["Не могу понять, о ком или о чём идёт речь."],
                                  verdict="UNKNOWN")
        if predicate is None:
            predicate = "is_a"
            es.predicate = predicate

        raw = []
        for e in entities:
            raw.extend(self.kb.query_statements(subject=e["id"],
                                                predicate=predicate,
                                                include_retracted=False))
            # reverse: e is the OBJECT of the predicate (e.g. written_by)
            rev = [s for s in self.kb.query_statements(predicate=predicate,
                                                       object=e["id"])
                   if s.status == "active"]
            raw.extend(rev)
        seed = self._seed_superset(entities)
        derived, trunc = kb_closure(self.kb, seed,
                                    self.max_depth,
                                    self.max_expansion)

        if intent == "yes_no" or predicate in ("может_летать", "существует", "имеет_спутник"):
            return self._yes_no(question, es, entities, predicate, trunc)
        return self._ask_fact(question, es, entities, predicate, raw, derived, trunc)

    # ----------------------------------------------------------------
    def _collect_confirmed(self, raw):
        return [s for s in raw if s.status == "active"
                and s.kind in ("user_assertion", "extracted_fact")]

    def _add_missing(self, es, predicate, entity_names):
        es.items.append(EvidenceItem(
            role="missing", note=f"нет подтверждённого факта {predicate} "
                                 f"для: {', '.join(entity_names)}"))

    def _ask_fact(self, question, es, entities, predicate, raw, derived, trunc) -> dict:
        confirmed = self._collect_confirmed(raw)
        hyps = [s for s in raw if s.status == "active" and s.kind == "hypothesis"]
        sentences, per_entity_status = [], []
        name_idx = {}

        for e in entities:
            own = [s for s in raw if (s.subject == e["id"] or (s.object_entity == e["id"]))]
            own_confirmed = self._collect_confirmed(own)
            if own_confirmed:
                for s in own_confirmed:
                    es.items.append(_ev_item_for_statement(s))
                # verbalize: reverse-link (e is object) → subject is the answer
                rev = [s for s in own_confirmed if s.object_entity == e["id"]]
                fwd = [s for s in own_confirmed if s.subject == e["id"]]
                if rev:
                    names = sorted({self.kb.get_entity(s.subject).name for s in rev})
                    ids = ", ".join(sorted({s.id for s in rev}))
                    sentences.append(f"{e['name']}: {', '.join(names)} [{ids}]")
                    per_entity_status.append("supported")
                    continue
                if fwd:
                    pols = {s.polarity for s in fwd}
                    if pols == {"assert", "deny"}:
                        ids = ", ".join(s.id for s in fwd)
                        sentences.append(f"{e['name']}: данные противоречат друг другу "
                                         f"({ids})")
                        per_entity_status.append("conflict")
                        es.items.append(_ev_item_for_statement(fwd[0], note="CONFLICT"))
                        continue
                    asserts = [s for s in fwd if s.polarity == "assert"]
                    if asserts:
                        objs = sorted({_fmt_entity(self.kb, s) for s in asserts})
                        ids = ", ".join(sorted({s.id for s in asserts}))
                        sentences.append(f"{e['name']}: {', '.join(objs)} [{ids}]")
                        per_entity_status.append("supported")
                    else:
                        ids = ", ".join(sorted({s.id for s in fwd}))
                        sentences.append(f"{e['name']}: {predicate} — нет [{ids}]")
                        per_entity_status.append("supported")
                continue
            # nothing confirmed: check derived facts (multi-hop)
            mine = [f for f in derived
                    if f.subject == e["id"] and fold(f.predicate) == fold(predicate)]
            if mine:
                best = min(mine, key=lambda f: f.depth)
                for f in mine:
                    item = _ev_item_for_derived(f)
                    if item not in es.items:
                        es.items.append(item)
                obj_list, seen_o = [], set()
                for f in sorted(mine, key=lambda f: (f.depth,
                                self.kb.get_entity(f.object_entity).name
                                if f.object_entity else str(f.object_literal))):
                    if fold(f.predicate) != fold(predicate):
                        continue
                    name = (self.kb.get_entity(f.object_entity).name
                            if f.object_entity else str(f.object_literal))
                    if name not in seen_o:
                        seen_o.add(name); obj_list.append(name)
                objs = obj_list
                rules = ", ".join(sorted({f.rule_id for f in mine}))
                support = sorted({sid for f in mine for sid in f.support_ids})
                es.applied_rules.append({"rule_ids": sorted({f.rule_id for f in mine}),
                                         "support_ids": support,
                                         "depth": best.depth})
                sentences.append(f"{e['name']}: {', '.join(objs)} "
                                 f"[derived via {rules}; "
                                 f"supports: {', '.join(support)}]")
                per_entity_status.append(
                    "partial" if best.depth >= 2 else "supported")
                continue
            if hyps:
                own_h = [s for s in hyps if s.subject == e["id"]
                         and fold(s.predicate) == fold(predicate)]
                if own_h:
                    for s in own_h:
                        es.items.append(_ev_item_for_statement(s, role="derived",
                                                                rule_id=None,
                                                                depth=1))
                    sentences.append(f"{e['name']}: только неподтверждённая гипотеза")
                    per_entity_status.append("partial")
                    continue
            self._add_missing(es, predicate, [e["name"]])
            sentences.append(f"{e['name']}: нет данных ({predicate})")
            per_entity_status.append("unknown")

        # same-name duplicates across types (seed heuristics): a zero-evidence
        # sibling of an answered entity adds noise, not information — drop it.
        name_of = [e["name"] for e in entities]
        answered_names = {n for n, st in zip(name_of, per_entity_status)
                          if st in ("supported", "partial", "conflict")}
        keep = [(sent, st, e) for sent, st, e in zip(
            sentences, per_entity_status, entities)
            if not (st == "unknown"
                    and e["name"] in answered_names)]
        if len(keep) != len(sentences):
            sentences = [k[0] for k in keep]
            per_entity_status = [k[1] for k in keep]
            kept_entities = [k[2] for k in keep]
            # prune stray missing items for dropped names
            dropped = answered_names & {e["name"] for e in entities
                                         if e not in kept_entities}
            if dropped:
                es.items = [it for it in es.items
                            if not (it.role == "missing"
                                    and any(n in (it.note or "")
                                            for n in dropped))]
                es.missing_prereqs = [m for m in es.missing_prereqs
                                      if not any(n in m for n in dropped)]

        if derived and predicate and fold(predicate) in ("is_a", "located_in"):
            covered = {(it.rule_id, it.depth) for it in es.items
                       if it.role == "derived"}
            for f in derived:
                if fold(f.predicate) == fold(predicate):
                    item = _ev_item_for_derived(f)
                    if (item.rule_id, item.depth) not in covered:
                        es.items.append(item)
                        covered.add((item.rule_id, item.depth))

        if trunc:
            es.missing_prereqs.append(f"цепочка обрезана: {trunc}")

        verdict = _aggregate_verdict(per_entity_status, es)
        return self._finalize(question, es, sentences, verdict=verdict)

    def _definition(self, es, entities, raw, derived, trunc) -> dict:
        confirmed = self._collect_confirmed(raw)
        sentences, statuses = [], []
        # is_a chains: from each entity, take raw is_a asserts + derived closure
        for e in entities:
            own_raw = [s for s in confirmed
                       if s.subject == e["id"] and fold(s.predicate) == "is_a"]
            own_der = [f for f in derived
                       if f.subject == e["id"] and fold(f.predicate) == "is_a"]
            for s in own_raw:
                es.items.append(_ev_item_for_statement(s))
            for f in own_der:
                es.items.append(_ev_item_for_derived(f))
            obj_items = [(1, _fmt_entity(self.kb, s)) for s in own_raw]
            obj_items += [(f.depth + 1,
                           self.kb.get_entity(f.object_entity).name
                           if f.object_entity else str(f.object_literal))
                          for f in own_der]
            seen_obj = set()
            objs = []
            for d, name in sorted(obj_items, key=lambda x: (x[0], x[1])):
                if name not in seen_obj:
                    seen_obj.add(name)
                    objs.append(name)
            if objs:
                ids_all = sorted({s.id for s in own_raw} |
                                 {sid for f in own_der for sid in f.support_ids})
                if own_der:
                    rules_used = sorted({f.rule_id for f in own_der})
                    es.applied_rules.append({"rule_ids": rules_used,
                                             "support_ids": ids_all,
                                             "depth": max(f.depth for f in own_der)})
                    suffix = f"[derived via {', '.join(rules_used)}; supports: {', '.join(ids_all)}]"
                else:
                    suffix = f"[{ids_all[0] if len(ids_all)==1 else ', '.join(ids_all)}]"
                sentences.append(f"«{e['name']}» — это {', '.join(objs)} {suffix}")
                if own_raw:
                    # core claim directly supported; depth of closure is tracked
                    # in applied_rules, not subtracted from confidence
                    statuses.append("supported")
                else:
                    statuses.append("partial")
            else:
                own_all = [s for s in confirmed if s.subject == e["id"]]
                if own_all:
                    pols = {s.polarity for s in own_all}
                    if pols == {"assert", "deny"}:
                        for s in own_all:
                            es.items.append(_ev_item_for_statement(s, note="CONFLICT"))
                        ids = ", ".join(sorted({s.id for s in own_all}))
                        sentences.append(f"{e['name']}: данные противоречат друг другу ({ids})")
                        statuses.append("conflict")
                    else:
                        for s in own_all:
                            es.items.append(_ev_item_for_statement(s))
                        payload = "; ".join(
                            f"{s.predicate} = {_fmt_entity(self.kb, s)}"
                            for s in own_all)
                        ids = ", ".join(sorted({s.id for s in own_all}))
                        sentences.append(f"{e['name']}: {payload} [{ids}]")
                        statuses.append("supported")
                else:
                    self._add_missing(es, "is_a", [e["name"]])
                    sentences.append(f"«{e['name']}»: нет данных")
                    statuses.append("unknown")
        if trunc:
            es.missing_prereqs.append(f"цепочка обрезана: {trunc}")
        verdict = _aggregate_verdict(statuses, es)
        return self._finalize("", es, sentences, verdict=verdict)

    def _yes_no(self, question, es, entities, predicate, trunc) -> dict:
        raw: List = []
        for e in entities:
            raw.extend(self.kb.query_statements(subject=e["id"],
                                                predicate=predicate,
                                                include_retracted=False))
        confirmed = self._collect_confirmed(raw)
        sentences, statuses = [], []
        for e in entities:
            own = [s for s in confirmed if s.subject == e["id"]]
            if not own:
                hyps = [s for s in raw if s.kind == "hypothesis"
                        and s.status == "active" and s.subject == e["id"]]
                if hyps:
                    for s in hyps:
                        es.items.append(_ev_item_for_statement(s, role="derived"))
                    sentences.append(f"{e['name']}: только неподтверждённая гипотеза")
                    statuses.append("partial")
                else:
                    self._add_missing(es, predicate, [e["name"]])
                    sentences.append(f"{e['name']}: нет данных")
                    statuses.append("unknown")
                continue
            pols = {s.polarity for s in own}
            if pols == {"assert", "deny"}:
                for s in own:
                    es.items.append(_ev_item_for_statement(s, note="CONFLICT"))
                ids = ", ".join(sorted({s.id for s in own}))
                sentences.append(f"{e['name']}: данные противоречат друг другу "
                                 f"({ids})")
                statuses.append("conflict")
                continue
            for s in own:
                es.items.append(_ev_item_for_statement(s))
            asserts = [s for s in own if s.polarity == "assert"]
            ids = ", ".join(sorted({s.id for s in own}))
            if asserts:
                sentences.append(f"{e['name']}: да [{ids}]")
                statuses.append("supported")
            else:
                sentences.append(f"{e['name']}: нет [{ids}]")
                # proposition is false by stored denial — verdict FALSE
                statuses.append("supported!deny")
        uniq = {s.split("!")[0] for s in statuses}
        if uniq == {"conflict"}:
            verdict = "CONFLICT"
        elif any(s == "supported!deny" for s in statuses):
            verdict = "FALSE" if all(
                s in ("supported!deny",) for s in statuses)                 else "UNKNOWN"
        elif "unknown" in uniq:
            verdict = "UNKNOWN"
        elif "partial" in uniq:
            verdict = "UNKNOWN"
        else:
            verdict = "TRUE"
        return self._finalize(question, es, sentences, verdict=verdict)

    def _compare(self, question, entities, es, normalized) -> dict:
        """Numeric comparison on literal predicates (population/height)."""
        if len(entities) < 2:
            es.missing_prereqs.append("для сравнения нужны две известные сущности")
            return self._finalize(question, es,
                                  ["Для сравнения нужно, чтобы я знал обе сущности."],
                                  verdict="UNKNOWN")
        a, b = entities[0], entities[1]
        measurements = {}
        for pred in ("population_of", "height_of"):
            vals = {}
            for e in (a, b):
                stmts = [s for s in self.kb.query_statements(
                             subject=e["id"], predicate=pred)
                         if s.status == "active"
                         and s.kind in ("user_assertion", "extracted_fact")
                         and s.polarity == "assert"
                         and _num(s.object_literal) is not None]
                if stmts:
                    # latest wins (deterministic: created_at, then id)
                    s = sorted(stmts, key=lambda s: (s.created_at, s.id))[-1]
                    vals[e["id"]] = (_num(s.object_literal), s)
            if len(vals) == 2:
                measurements[pred] = vals
        if not measurements:
            es.missing_prereqs.append(
                "нет числовых population_of/height_of для обеих сущностей")
            return self._finalize(question, es,
                                  ["Числовых данных для сравнения нет — не буду гадать."],
                                  verdict="UNKNOWN")
        sentences = []
        for pred, vals in measurements.items():
            for e in (a, b):
                es.items.append(_ev_item_for_statement(vals[e["id"]][1]))
            va, vb = vals[a["id"]][0], vals[b["id"]][0]
            ids = ", ".join(sorted({vals[x][1].id for x in (a["id"], b["id"])}))
            rel = "больше" if va > vb else ("меньше" if va < vb else "равно")
            pred_ru = "население" if pred == "population_of" else "высота"
            sentences.append(f"{pred_ru}: у {a['name']} {_fmt_ru_num(va)}, "
                             f"у {b['name']} {_fmt_ru_num(vb)} — "
                             f"{a['name']} {rel.replace('равно', 'равно')} "
                             f"({rel} по сравнению с {b['name']}) [{ids}]")
        verdict = "TRUE"
        return self._finalize(question, es, sentences, verdict=verdict,
                              compare=True)

    # ----------------------------------------------------------------
    def _finalize(self, question, es: EvidenceSet, sentences, verdict,
                  compare=False) -> dict:
        """Compose answer text and run the evidence-check gate.

        Evidence-check: every bracketed id set referenced by a sentence must be
        present in the EvidenceSet; any sentence that references a missing id
        or any factual claim with zero support is replaced by an explicit
        abstain phrase. This is the last gate — what survives is what the
        answer may state.
        """
        ids_ok = set(es.evidence_ids())
        clean, verified = [], 0
        for sent in sentences:
            brackets = re.findall(r"\[([^\]]+)\]", sent)
            flat = [tok.strip() for b in brackets for tok in b.split(",")
                    if tok.strip() and not tok.startswith(("derived", "supports"))]
            needs_support = bool(brackets)
            if needs_support:
                unsupported = [tok for tok in flat
                               if tok.startswith(("st_", "rule:"))
                               and tok not in ids_ok]
                if unsupported:
                    continue  # drop the sentence — never assert it
                verified += 1
            clean.append(sent)
        if not clean:
            clean = ["Не хватает подтверждённых данных, чтобы ответить."]
        text = " ".join(s.rstrip(".") + "." for s in clean)
        ed = es.to_dict()
        return {
            "question": question,
            "status": ed["status"],          # supported|partial|unknown|conflict
            "verdict": verdict,              # TRUE|FALSE|UNKNOWN|CONFLICT
            "answer": text,
            "evidence_ids": ed["evidence_ids"],
            "evidence": ed,
            "sentences_with_evidence": verified,
            "missing_prereqs": es.missing_prereqs,
        }


def _aggregate_verdict(statuses: List[str], es: EvidenceSet) -> str:
    if not statuses:
        return "UNKNOWN"
    uniq = set(statuses)
    if "conflict" in uniq:
        return "CONFLICT"
    if "unknown" in uniq:
        return "UNKNOWN"
    if any(s == "partial" for s in statuses):
        return "UNKNOWN"  # composed of derived/partial evidence only
    return "TRUE"


def _fmt_ru_num(n: float) -> str:
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f} млн".replace(".", ",")
    if n >= 1_000:
        return f"{n:,.0f}".replace(",", " ")
    return f"{n:g}"


def ask(kb: KnowledgeStore, question: str, *, max_depth: int = 2) -> dict:
    return ReasoningPipeline(kb, max_depth=max_depth).ask(question)
