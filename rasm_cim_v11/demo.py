"""v0.11 demo answerer — deliberately minimal, honest.

NOT the v0.14 pipeline. It demonstrates: language-layer analysis bound to a
TruthGraph9 via the adapter, using a tiny bundled demo fact set
(data/demo_facts.jsonl). It only answers:
  * capital questions  ("какая столица у Франции?")
  * definition questions ("что такое киты?" / "кто такой Пушкин?")
  * yes/no capability questions over known facts ("умеет ли пингвин летать?")
Everything else returns a clear "not supported yet" message. No fabricated
answers; statuses come straight from TruthGraph9 (TRUE/FALSE/UNKNOWN/CONFLICT).

Entity matching is STE-based (same pseudo-stem function for question and
fact subjects) — never substring matching, so inflected forms still hit.
Ambiguity → the longest subject whose stems all appear in the question wins;
no fuzzy merging of different entities.
"""
from __future__ import annotations

import json
import os
from typing import List, Optional, Tuple

from .language import GraphAdapter, analyze, lemma, normalize

_DATA = os.path.join(os.path.dirname(__file__), "data", "demo_facts.jsonl")

_KW_REL = [
    ("столиц", "capital_of"),
    ("написал", "написал"),
    ("автор", "написал"),
    ("лета", "может_летать"),
    ("существует", "существует"),
    ("спутник", "имеет_спутник"),
    ("кипит", "кипит_при"),
]


def load_demo_facts(path: str = _DATA) -> List[dict]:
    facts = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                facts.append(json.loads(line))
    return facts


def build_demo_graph(path: str = _DATA) -> Tuple[GraphAdapter, List[dict]]:
    facts = load_demo_facts(path)
    ad = GraphAdapter()
    for f in facts:
        ad.assert_fact(f["subject"], f["relation"], f["object"],
                       polarity=f.get("polarity", "true"))
    return ad, facts


def _stems(text: str) -> set[str]:
    return {lemma(w) for w in normalize(text).text.replace("?", " ").split()
            if w.strip()}


def _find_subject(normalized: str, facts: List[dict]) -> Optional[str]:
    """Longest demo subject whose stems are a subset of the question stems."""
    q_stems = _stems(normalized)
    best = None
    for f in facts:
        subj_stems = _stems(f["subject"])
        if subj_stems and subj_stems <= q_stems:
            if best is None or len(f["subject"]) > len(best):
                best = f["subject"]
    return best


def answer_v11(question: str, path: str = _DATA) -> dict:
    """Return a dict with answer, status, intent, evidence (fact indices)."""
    analysis = analyze(question)
    intent = analysis.intent.intent
    ad, facts = build_demo_graph(path)

    if intent == "greeting":
        return {"answer": "Привет! Чем помочь?", "status": "GREETING",
                "intent": intent, "evidence": []}

    subj = _find_subject(analysis.normalized, facts)
    rel = next((relation for kw, relation in _KW_REL
                if kw in analysis.normalized), None)

    if intent == "ask_definition":
        target = subj
        if target is None:
            return {"answer": "Не знаю — такого термина нет в демо-памяти.",
                    "status": "UNKNOWN", "intent": intent, "evidence": []}
        hits = [i for i, f in enumerate(facts)
                if f["subject"] == target and f["relation"] == "is_a"]
        status = ad.query_relation(target, "is_a")
        objs = sorted({facts[i]["object"] for i in hits})
        if status == "UNKNOWN":
            return {"answer": f"Не знаю, что такое «{target}».",
                    "status": "UNKNOWN", "intent": intent, "evidence": []}
        if status == "CONFLICT":
            return {"answer": f"Данные о «{target}» противоречат друг другу.",
                    "status": "CONFLICT", "intent": intent, "evidence": hits}
        return {"answer": f"«{target}» — это {', '.join(objs)}.",
                "status": status, "intent": intent, "evidence": hits}

    if subj and rel:
        hits = [i for i, f in enumerate(facts)
                if f["subject"] == subj and f["relation"] == rel]
        status = ad.query_relation(subj, rel)
        objs = sorted({facts[i]["object"] for i in hits})
        if status == "TRUE":
            return {"answer": f"{subj}: {', '.join(objs)}.",
                    "status": "TRUE", "intent": intent, "evidence": hits}
        if status == "CONFLICT":
            return {"answer": f"По «{subj} / {rel}» данные противоречат друг другу.",
                    "status": "CONFLICT", "intent": intent, "evidence": hits}
        if status == "FALSE":
            return {"answer": f"Нет (по «{subj} / {rel}»).",
                    "status": "FALSE", "intent": intent, "evidence": hits}
        return {"answer": f"Не знаю (нет факта «{subj} / {rel}» в памяти).",
                "status": "UNKNOWN", "intent": intent, "evidence": []}

    if intent == "unknown":
        return {"answer": "Вопрос не поддерживается v0.11 (честный unknown).",
                "status": "UNKNOWN", "intent": intent, "evidence": []}
    return {"answer": "v0.11 отвечает только про столицы, определения и "
                      "да/нет-умения из демо-памяти. Полный pipeline — v0.14.",
            "status": "UNSUPPORTED", "intent": intent, "evidence": []}
