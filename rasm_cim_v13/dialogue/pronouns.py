"""Limited Russian pronoun resolution for short dialogues (v0.13).

Scope honesty (by ТЗ): this is NOT co-reference resolution in the linguistic
sense. It handles personal pronoun-like words with crude grammatical gender
and number, resolved ONLY against entities explicitly seen in recent context
(turns). Rules:

  * singular (он/его/ему/нём/него → m; она/её/ей/ней/неё → f; оно/его → n):
      candidates = context entities of matching gender from the LAST entity-
      bearing turns, most recent first;
      exactly 1 candidate → resolved;
      ≥2 candidates → AMBIGUOUS (clarifying question lists them — never a coin flip);
      0 → UNKNOWN.
  * plural (они/их/им/них/ним): candidates = ALL entities of the latest turn
    (or 3 most recent); ≥2 → resolved as a set; <2 → UNKNOWN.
  * demonstratives этот/эта/это/те → same gender rules, slightly looser.

Gender guess from name ending (эвристика, documented): -а/-я → f,
-о/-е → n, otherwise m. Exceptions: нескл. заимствования (кофе m, пальто n…) —
small closed exception list. Wrong gender guesses yield AMBIGUOUS, not wrong
facts — fail-safe direction.

The resolved text keeps the original surface form; resolutions are STRUCTURED
(pronoun, span, chosen/candidates, status). Answer composition never echoes
substituted text into user-visible Russian.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional

_MAP = {
    # masculine
    "он": ("m", "s"), "его": ("m", "s"), "ему": ("m", "s"),
    "нём": ("m", "s"), "нем": ("m", "s"), "него": ("m", "s"),
    "ним": ("m", "s"),   # инструмент. ед.ч. м.р.
    # feminine
    "она": ("f", "s"), "её": ("f", "s"), "ее": ("f", "s"), "ей": ("f", "s"),
    "ней": ("f", "s"), "неё": ("f", "s"), "нее": ("f", "s"),
    # neuter
    "оно": ("n", "s"),
    # plural
    "они": ("p", "p"), "их": ("p", "p"), "им": ("p", "p"),
    "них": ("p", "p"), "ним": ("p", "p"), "ними": ("p", "p"),
}
_DEMONSTRATIVES = {
    "этот": ("m", "s"), "эта": ("f", "s"),
    "эти": ("p", "p"), "тот": ("m", "s"), "та": ("f", "s"), "те": ("p", "p"),
    # NOTE: «это»/«то» are copula/discourse particles more often than referents;
    # resolving them as entities produced false CLARIFYs — excluded deliberately.
}

_INDECLINABLE = {"кофе": "m", "пальто": "n", "шоссе": "n", "метро": "n",
                 "радио": "n", "кенгуру": "m", "какао": "n"}


def guess_gender(name: str) -> str:
    w = name.strip().casefold().replace("ё", "е")
    if w in _INDECLINABLE:
        return _INDECLINABLE[w]
    last = w[-1] if w else ""
    if last in "ая":
        return "f"
    if last in "ое":
        return "n"
    return "m"


def find_pronouns(text: str) -> List[dict]:
    tokens = re.finditer(r"[А-Яа-яЁёA-Za-z\-]+", text)
    out = []
    for m in tokens:
        w = m.group().casefold().replace("ё", "е")
        if w in _MAP:
            g, num = _MAP[w]
        elif w in _DEMONSTRATIVES:
            g, num = _DEMONSTRATIVES[w]
        else:
            continue
        out.append({"pronoun": m.group(), "start": m.start(), "end": m.end(),
                    "gender": g, "number": num})
    return out


def resolve(pronouns: List[dict], recent_entities: List[dict]):
    """recent_entities: MRU-ordered [{name, gender, ...}] (context store order).

    Returns list of resolutions:
      {pronoun, start, end, status: resolved|ambiguous|unknown,
       chosen: [names], candidates: [names]}
    """
    resolutions = []
    for p in pronouns:
        g, num = p["gender"], p["number"]
        if num == "p":
            cand = [e["name"] for e in recent_entities[:4]]
            if len(cand) >= 2:
                resolutions.append({**p, "status": "resolved", "chosen": cand,
                                    "candidates": cand})
            else:
                resolutions.append({**p, "status": "unknown",
                                    "chosen": [], "candidates": cand})
            continue
        cand_all = [e for e in recent_entities]
        if g != "p":
            cand = [e["name"] for e in cand_all if e.get("gender") == g]
        else:
            cand = [e["name"] for e in cand_all]
        if len(cand) == 1:
            resolutions.append({**p, "status": "resolved",
                                "chosen": cand, "candidates": cand})
        elif len(cand) >= 2:
            resolutions.append({**p, "status": "ambiguous",
                                "chosen": [], "candidates": cand[:4]})
        else:
            # gender miss: if there is exactly one recent entity overall, still
            # better to ask than to silently bind → unknown (fail-safe)
            resolutions.append({**p, "status": "unknown",
                                "chosen": [], "candidates": []})
    return resolutions
