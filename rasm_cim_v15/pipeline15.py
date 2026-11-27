"""Question answering for v0.15 — v0.14 pipeline + learned-artifact routing.

Two ADDITIVE learned hooks (artifacts are files, never code):
  1. intent_fallback — when v0.11 analyze() (frozen) returns `unknown`,
     the corpus-trained fallback may classify with a log-prob margin ≥ tau.
  2. predicate_hints — user relation-correction: exact normalized question ⇒
     predicate routing (narrow by design; generalization is the fallback's job).

Answers are still composed ONLY from the verified EvidenceSet
(direct construction from KB statements: evidence ids attached to each
factoid sentence). Neither hook can make the system assert unsupported content.
"""
from __future__ import annotations

import re
from typing import Optional

from rasm_cim_v11.language import analyze, normalize, lemma
from rasm_cim_v12.knowledge import KnowledgeStore
from rasm_cim_v12.knowledge.models import fold
from rasm_cim_v14.reasoning import ask as ask_v14
from rasm_cim_v14.reasoning.evidence import EvidenceItem, EvidenceSet
from rasm_cim_v14.reasoning.pipeline import _find_entities
from rasm_cim_v15.artifacts import LearnedArtifacts, _norm_key

_STEM_RE = re.compile(r"[А-Яа-яЁёA-Za-z0-9\-]+")


def resolve_intent(text: str, arts: Optional[LearnedArtifacts]):
    base = analyze(text)
    if base.intent.intent != "unknown":
        return base.intent.intent, "v0.11", base.intent.intent
    if arts is None:
        return "unknown", "no-artifact", None
    if not arts.s["config"].get("use_fallback_on_unknown", True):
        return "unknown", "fallback-disabled", None
    fb, margin = arts.predict_intent(text)
    if fb is not None and margin >= arts.s["config"]["tau"]:
        return fb, f"fallback(margin={margin})", fb
    return "unknown", "fallback-below-tau", fb


def _answer_with_hint(kb: KnowledgeStore, question: str, predicate: str,
                      *, max_depth: int = 2) -> dict:
    """Narrow but fully-evidence-bound path for user-corrected questions."""
    norm = normalize(question).text.lower()
    entities = _find_entities(kb, question)
    es = EvidenceSet(intent="ask_fact",
                     subject_names=[e["name"] for e in entities],
                     predicate=predicate)
    if not entities:
        es.missing_prereqs.append("не найдено сущностей из вопроса в памяти")
        return _finish(question, es,
                       ["Не могу понять, о ком или о чём идёт речь."],
                       "UNKNOWN")
    sentences, statuses = [], []
    for e in entities:
        fwd = [s for s in kb.query_statements(subject=e["id"], predicate=predicate)
               if s.status == "active"
               and s.kind in ("user_assertion", "extracted_fact")]
        rev = [s for s in kb.query_statements(predicate=predicate, object=e["id"])
               if s.status == "active"
               and s.kind in ("user_assertion", "extracted_fact")]
        if rev:
            for s in rev:
                es.items.append(EvidenceItem(
                    role="direct", statement_id=s.id,
                    source=(s.provenance or {}).get("source"),
                    doc_id=(s.provenance or {}).get("doc_id"),
                    reliability=(s.provenance or {}).get("reliability"), depth=1))
            names = sorted({kb.get_entity(s.subject).name for s in rev})
            ids = ", ".join(sorted({s.id for s in rev}))
            sentences.append(f"{e['name']}: {', '.join(names)} [{ids}]")
            statuses.append("supported")
            continue
        if fwd:
            pols = {s.polarity for s in fwd}
            for s in fwd:
                item_note = "CONFLICT" if pols == {"assert", "deny"} else ""
                es.items.append(EvidenceItem(
                    role="direct", statement_id=s.id,
                    source=(s.provenance or {}).get("source"),
                    doc_id=(s.provenance or {}).get("doc_id"),
                    reliability=(s.provenance or {}).get("reliability"),
                    depth=1, note=item_note))
            if pols == {"assert", "deny"}:
                ids = ", ".join(sorted({s.id for s in fwd}))
                sentences.append(f"{e['name']}: данные противоречат друг другу ({ids})")
                statuses.append("conflict")
                continue
            asserts = [s for s in fwd if s.polarity == "assert"]
            ids = ", ".join(sorted({s.id for s in fwd}))
            if asserts:
                names = sorted({(kb.get_entity(s.object_entity).name
                                 if s.object_entity else s.object_literal)
                                for s in asserts})
                sentences.append(f"{e['name']}: {', '.join(names)} [{ids}]")
                statuses.append("supported")
            else:
                sentences.append(f"{e['name']}: {predicate} — нет [{ids}]")
                statuses.append("supported!deny")
            continue
        hyps = [s for s in kb.query_statements(subject=e["id"], predicate=predicate)
                if s.status == "active" and s.kind == "hypothesis"]
        if hyps:
            for s in hyps:
                es.items.append(EvidenceItem(role="derived", statement_id=s.id,
                                             depth=1))
            sentences.append(f"{e['name']}: только неподтверждённая гипотеза")
            statuses.append("partial")
            continue
        es.missing_prereqs.append(f"нет подтверждённого факта {predicate} для {e['name']}")
        es.items.append(EvidenceItem(role="missing",
                                     note=f"нет факта {predicate} для {e['name']}"))
        sentences.append(f"{e['name']}: нет данных ({predicate})")
        statuses.append("unknown")

    uniq = {st.split("!")[0] for st in statuses}
    if "conflict" in uniq:
        verdict = "CONFLICT"
    elif all(st == "supported!deny" for st in statuses):
        verdict = "FALSE"
    elif "unknown" in uniq:
        verdict = "UNKNOWN"
    elif "partial" in uniq:
        verdict = "UNKNOWN"
    elif not statuses:
        verdict = "UNKNOWN"
    else:
        verdict = "TRUE"
    return _finish(question, es, sentences, verdict)


def _finish(question, es, sentences, verdict) -> dict:
    ids_ok = set(es.evidence_ids())
    ids = sorted(ids_ok)
    return {"question": question, "status": es.status(), "verdict": verdict,
            "answer": " ".join(sentences),
            "evidence_ids": ids, "evidence": es.to_dict(),
            "missing_prereqs": es.missing_prereqs,
            "composed_by": "v0.15-hint-path"}


def ask_v15(kb: KnowledgeStore, question: str, *,
            artifacts: Optional[LearnedArtifacts] = None,
            max_depth: int = 2) -> dict:
    intent, origin, learned = resolve_intent(question, artifacts)
    if artifacts is not None:
        hint = artifacts.predicate_hint_for(normalize(question).text.lower())
        if hint:
            r = _answer_with_hint(kb, question, hint, max_depth=max_depth)
            r["intent_resolved_as"] = intent
            r["intent_origin"] = origin if origin != "unknown" else "correction-hint"
            r["predicate_hint"] = hint
            return r
    r = ask_v14(kb, question, max_depth=max_depth)
    r["intent_resolved_as"] = intent
    r["intent_origin"] = origin
    return r
