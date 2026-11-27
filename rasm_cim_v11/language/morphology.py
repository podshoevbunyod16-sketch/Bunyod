"""v0.11 minimal rule-based Russian morphology.

Honest scope: this is NOT a dictionary-based lemmatizer and NOT a parser.
It produces a *stable pseudo-stem* by stripping common inflectional endings
from both ends of a comparison. Matching works because queries AND the fact
dictionary are stemmed with the same rules; the pseudo-stem is never shown
to the user as a "lemma" claim.

Layers:
  1. num_to_int(): word-form numerals 0..1000 → int ("два" → 2).
  2. is_stopword(): closed-class function words (deterministic lists).
  3. lemma(): pseudo-stem via ordered longest-suffix stripping, then a small
     irregular table for high-frequency cases where suffix rules fail.
"""
from __future__ import annotations

import re
from typing import Dict, Optional

# ---------------------------------------------------------------- numerals
_UNITS = {"ноль": 0, "один": 1, "одна": 1, "одно": 1, "два": 2, "две": 2,
          "три": 3, "четыре": 4, "пять": 5, "шесть": 6, "семь": 7,
          "восемь": 8, "девять": 9, "десять": 10, "одиннадцать": 11,
          "двенадцать": 12, "тринадцать": 13, "четырнадцать": 14,
          "пятнадцать": 15, "шестнадцать": 16, "семнадцать": 17,
          "восемнадцать": 18, "девятнадцать": 19}
_TENS = {"двадцать": 20, "тридцать": 30, "сорок": 40, "пятьдесят": 50,
         "шестьдесят": 60, "семьдесят": 70, "восемьдесят": 80, "девяносто": 90}


def num_to_int(word: str) -> Optional[int]:
    w = word.casefold()
    if w in _UNITS:
        return _UNITS[w]
    if w in _TENS:
        return _TENS[w]
    if w == "сто":
        return 100
    if w == "тысяча":
        return 1000
    # "двадцать два" style compounds arrive token-wise; handle "двадцать-два"
    m = re.fullmatch(r"(\w+)-(\w+)", w)
    if m:
        tens = _TENS.get(m.group(1))
        unit = _UNITS.get(m.group(2))
        if tens is not None and unit is not None:
            return tens + unit
    return None


# --------------------------------------------------------------- stopwords
STOPWORDS = frozenset(
    "а и или но да же ли ль бы было были был быть есть это этого этому том "
    "о в во на за из изо к ко у с со от ото до по при про для над под "
    "не ни как это так то его их её ей ем им ним ней нём них этот эта эти тот "
    "та те мне мой моя моё твой весь всё все без между около".split()
)

QUESTION_WORDS = frozenset(
    "кто что где когда откуда куда почему зачем сколько какой какая какое какие "
    "чей чья чьё чьи который которая которое которые".split()
)


def is_stopword(word: str) -> bool:
    return word.casefold() in STOPWORDS


# ------------------------------------------------------------------ lemma
# Ordered from longest to shortest; applied once per pass, two passes max.
_SUFFIXES = [
    # nouns / adjectives / participles (instrumental, prepositional, plural…)
    "ами", "ями", "ах", "ях", "ого", "его", "ому", "ему", "ыми", "ими",
    "ией", "ою", "ею", "ой", "ей", "ом", "ем", "ам", "ям",
    "ая", "яя", "ое", "ее", "ые", "ие", "ую", "юю",
    "ов", "ев", "ёв", "ей", "ья", "ье", "ью",
    # verbs
    "ешь", "ишь", "ете", "ите", "ают", "яют", "ует", "уют", "ует",
    "ает", "яет", "ит", "ат", "ят", "ут", "ют",
    "ла", "ло", "ли", "ть", "ти", "чь", "ся", "сь",
    # bare endings
    "а", "я", "ы", "и", "е", "у", "ю", "о",
]
_MIN_STEM = 3

# irregular / suppletion table (high frequency only)
_SPECIAL: Dict[str, str] = {
    "людей": "человек", "человеку": "человек", "человеком": "человек",
    "дети": "ребёнок", "детей": "ребёнок", "детям": "ребёнок", "детьми": "ребёнок",
    "люди": "человек", "людьми": "человек",
    "лучше": "хороший", "хуже": "плохой", "больше": "большой", "меньше": "маленький",
    "летел": "лететь", "летит": "лететь", "летать": "лететь",
    "живёт": "жить", "живет": "жить", "жители": "житель",
}


def lemma(word: str) -> str:
    w = word.casefold().replace("ё", "е")
    if w in _SPECIAL:
        return _SPECIAL[w]
    stem = w
    for _ in range(2):
        for suf in _SUFFIXES:
            if stem.endswith(suf) and len(stem) - len(suf) >= _MIN_STEM:
                stem = stem[: -len(suf)]
                break
        else:
            break
    return stem


def lemmas(words):
    return tuple(lemma(w) for w in words)
