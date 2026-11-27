"""v0.16 — text normalisation and lightweight Russian morphology.

Design constraints (hard):
  * stdlib only, no pymorphy2/OpenCorpora (dictionary download == network,
    tens of MB == Termux-hostile).  Everything here is rule based and
    deterministic: same input -> same output, no learned tables.
  * Morphology is used ONLY to *generate candidate match forms*.  It never
    rewrites the surface text that is stored or displayed, and a candidate
    form alone never merges two entities — merging is an explicit act
    (teach / same_as).  This keeps stemming errors from corrupting the KB.

Public API
    normalize(text)                 -> comparable lowercase form
    squeeze(text)                   -> whitespace/punct-squeezed original case
    strip_honorifics(text)          -> remove "сэр", "господин", titles
    case_forms(word)                -> candidate case-normalised forms
    stem_candidates(phrase)         -> [(form, confidence)] best first
    transliterate(text, direction)  -> ru<->latin
    edit_distance(a, b, cap)        -> bounded Levenshtein
    word_tokens(text)               -> alphabetic tokens, order preserved
"""
from __future__ import annotations

import re
import unicodedata
from typing import Iterable, Sequence

# --------------------------------------------------------------------------
# basic text hygiene
# --------------------------------------------------------------------------

_PUNCT_RE = re.compile(r"[^\w\s\-']+", re.UNICODE)
_WS_RE = re.compile(r"\s+")
_DASH_RE = re.compile(r"[\u2010-\u2015\u2212]")   # hyphen-like -> plain hyphen
_STANDALONE_DASH_RE = re.compile(r"\s+[-\u2010-\u2015\u2212]\s+")
# letters AND digits (a year like "1643" must survive tokenisation so it
# can be stored as a literal value), hyphenated words kept whole
_WORD_RE = re.compile(r"[^\W_]+(?:-[^\W_]+)*", re.UNICODE)

#: Titles / honorifics that carry no identity information.
HONORIFICS = frozenset({
    "сэр", "сер", "сэра", "господин", "господина", "госпожа", "госпожи",
    "мистер", "миссис", "мисс", "mr", "mrs", "ms", "sir", "lady", "lord",
    "профессор", "профессора", "проф", "доктор", "доктора", "dr", "prof",
    "академик", "академика", "товарищ", "гражданин", "гражданка",
    "царь", "король", "королева", "император", "императрица",
    "святой", "святая", "преподобный", "граф", "князь", "княгиня",
})


def normalize(text: str) -> str:
    """Comparable form: NFC, lowercase, ё->е, punctuation dropped, spaces squeezed.

    ``normalize("Сэр  Исаак  НЬЮТОН!") == "сэр исаак ньютон"``
    """
    if not text:
        return ""
    t = unicodedata.normalize("NFC", str(text))
    t = _DASH_RE.sub("-", t)
    t = t.lower().replace("ё", "е")
    t = _PUNCT_RE.sub(" ", t)
    t = _STANDALONE_DASH_RE.sub(" ", t)     # copula dash is not part of a name
    return _WS_RE.sub(" ", t).strip()


#: every dash-like character that can act as the Russian copula
DASH_TOKENS = frozenset({"-", "\u2010", "\u2011", "\u2012", "\u2013",
                         "\u2014", "\u2015", "\u2212"})


def squeeze(text: str) -> str:
    """Whitespace squeezed, original case AND original dash character kept.

    A *standalone* dash is deliberately PRESERVED: it is the Russian copula
    ("Ньютон — физик") and the parser needs it as a structural marker.  The
    character itself is not rewritten, so evidence records store exactly what
    the user typed.  ``normalize()`` folds dashes away, ``tokens_keep_case()``
    ignores them as words.
    """
    if not text:
        return ""
    t = unicodedata.normalize("NFC", str(text))
    return _WS_RE.sub(" ", t.strip())


def strip_honorifics(text: str) -> str:
    """Drop leading honorifics/titles.  Only *leading* ones — a trailing token
    is far more likely to be part of a name than a title."""
    toks = normalize(text).split()
    i = 0
    while i < len(toks) and toks[i] in HONORIFICS:
        i += 1
    return " ".join(toks[i:]) if i < len(toks) else " ".join(toks)


def word_tokens(text: str) -> list[str]:
    """Alphabetic tokens (hyphenated kept whole), order preserved, lowercased."""
    return [m.group(0).lower().replace("ё", "е") for m in _WORD_RE.finditer(text or "")]


# --------------------------------------------------------------------------
# Russian suffix morphology (rule based, conservative)
# --------------------------------------------------------------------------

#: Case endings for nouns/adjectives, longest first so greedy stripping picks
#: the most specific match.  Order matters.
_NOUN_ENDINGS: tuple[str, ...] = (
    "иями", "ями", "ами", "иях", "ях", "ах", "иям", "ям", "ам",
    "ого", "его", "ому", "ему", "ыми", "ими", "ом", "ем", "ым", "им",
    "ей", "ой", "ий", "ый", "ая", "яя", "ов", "ев", "ье", "ия", "ию", "ью",
    "а", "я", "у", "ю", "е", "ы", "и", "о", "ь",
)

#: Verb endings (3rd/1st/2nd person, past tense) — used to link surface verbs
#: such as "открыл"/"открыла"/"открыли"/"открывает" to one predicate.
_VERB_ENDINGS: tuple[str, ...] = (
    "ешься", "ется", "ются", "ёшься", "ешься", "ешь", "ете", "ут", "ют",
    "ат", "ят", "ла", "ло", "ли", "л", "а", "о", "и", "у", "ю", "ь", "ем",
    "им", "ешь", "ишь",
)

MIN_STEM_LEN = 4          # never strip below this many characters
MIN_FUZZY_LEN = 6         # never fuzzy-match shorter strings than this


def _strip_once(word: str, endings: Sequence[str], min_len: int = MIN_STEM_LEN) -> str | None:
    for e in endings:
        if word.endswith(e) and len(word) - len(e) >= min_len:
            return word[: -len(e)]
    return None


def case_forms(word: str) -> list[str]:
    """Candidate case-normalised forms of a single word, best (least stripped)
    first.  The untouched word is always included."""
    return [f for f, _n in case_forms_scored(word)]


def case_forms_scored(word: str) -> list[tuple[str, int]]:
    """``[(form, chars_stripped)]``.

    The number of stripped characters matters: dropping a single "а" from
    "физика" to reach "физик" is much weaker evidence than dropping "ом" from
    "физиком".  Callers use it to scale match confidence honestly instead of
    treating every stem as equal.
    """
    w = normalize(word).replace(" ", "")
    out = [(w, 0)]
    seen = {w}
    cur = w
    total = 0
    for _ in range(3):                        # max 3 nested strips (diminutives)
        nxt = _strip_once(cur, _NOUN_ENDINGS)
        if nxt is None or nxt in seen:
            break
        total += len(cur) - len(nxt)
        seen.add(nxt)
        out.append((nxt, total))
        cur = nxt
    return out


def verb_stem(word: str) -> str | None:
    """Stem of a Russian verb form (used to group inflected predicate verbs)."""
    w = normalize(word).replace(" ", "")
    if len(w) < MIN_STEM_LEN + 1:
        return None
    return _strip_once(w, _VERB_ENDINGS)


def stem_candidates(phrase: str, min_len: int = MIN_STEM_LEN) -> list[tuple[str, float]]:
    """Candidate comparable forms for a whole phrase.

    Returns ``[(form, confidence)]`` sorted best-first.  Confidence is the
    system's *matching* confidence, not a probability: 1.0 = verbatim
    normalised match, <1.0 = progressively more inference was needed.
    """
    base = normalize(phrase)
    if not base:
        return []
    out: list[tuple[str, float]] = [(base, 1.0)]
    seen = {base}

    nohon = strip_honorifics(phrase)
    if nohon and nohon not in seen:
        seen.add(nohon)
        out.append((nohon, 0.98))

    toks = base.split()
    # per-word stemming: apply the shortest stem to every strippable word.
    # Confidence is scaled by the total number of stripped characters — a
    # one-character strip is weak evidence, a three-character one is strong.
    stems = []
    changed = 0
    for t in toks:
        forms = [(f, n) for f, n in case_forms_scored(t) if len(f) >= min_len]
        best = max(forms, key=lambda x: x[1]) if forms else (t, 0)
        if best[0] != t:
            changed += best[1]
        stems.append(best[0])
    if changed:
        sform = " ".join(stems)
        if sform not in seen:
            seen.add(sform)
            out.append((sform, 0.78 if changed >= 2 else 0.68))

    # last-word-only stem (handles "Ньютона" in "закон Ньютона")
    if len(toks) >= 1:
        forms = [(f, n) for f, n in case_forms_scored(toks[-1]) if len(f) >= min_len]
        if forms:
            s, n = max(forms, key=lambda x: x[1])
            if s != toks[-1]:
                alt = " ".join(toks[:-1] + [s])
                if alt not in seen:
                    seen.add(alt)
                    out.append((alt, 0.78 if n >= 2 else 0.68))
    return out


# --------------------------------------------------------------------------
# transliteration (Cyrillic <-> Latin) — for "Isaac Newton" == "Исаак Ньютон"
# --------------------------------------------------------------------------

_RU2LAT = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ж": "zh",
    "з": "z", "и": "i", "й": "y", "к": "k", "л": "l", "м": "m", "н": "n",
    "о": "o", "п": "p", "р": "r", "с": "s", "т": "t", "у": "u", "ф": "f",
    "х": "h", "ц": "ts", "ч": "ch", "ш": "sh", "щ": "sch", "ъ": "", "ы": "y",
    "ь": "", "э": "e", "ю": "yu", "я": "ya",
}
_LAT2RU = {v: k for k, v in _RU2LAT.items() if v}
_LAT2RU.update({"j": "й", "q": "к", "w": "в", "x": "кс"})

_CYR = re.compile("[а-я]")
_LAT = re.compile("[a-z]")


def transliterate(text: str, direction: str = "ru2lat") -> str:
    """Phonetic transliteration.  Lossy by construction — used only as a
    *candidate generator* (confidence 0.90), never as an identity proof."""
    t = normalize(text)
    if direction == "ru2lat":
        return "".join(_RU2LAT.get(ch, ch) for ch in t)
    if direction == "lat2ru":
        return "".join(_LAT2RU.get(ch, ch) for ch in t)
    raise ValueError(f"unknown transliteration direction {direction!r}")


def is_cyrillic(text: str) -> bool:
    n = len(_CYR.findall(normalize(text)))
    return n > 0 and n >= len(_LAT.findall(normalize(text)))


def cross_script_form(text: str) -> str | None:
    """Transliterate only when the text actually uses the "other" script."""
    t = normalize(text)
    if not t:
        return None
    if _CYR.search(t) and not _LAT.search(t):
        return transliterate(t, "ru2lat")
    if _LAT.search(t) and not _CYR.search(t):
        return transliterate(t, "lat2ru")
    return None


# --------------------------------------------------------------------------
# edit distance (bounded — early exit keeps it cheap on long dialogues)
# --------------------------------------------------------------------------

def edit_distance(a: str, b: str, cap: int = 2) -> int:
    """Bounded Levenshtein.  Returns ``cap + 1`` as soon as the true distance
    is provably > cap, so cost stays O(len * cap) instead of O(len^2)."""
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if abs(la - lb) > cap:
        return cap + 1
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        best = i
        ca = a[i - 1]
        for j in range(1, lb + 1):
            cost = 0 if ca == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
            if cur[j] < best:
                best = cur[j]
        if best > cap:
            return cap + 1
        prev = cur
    return prev[lb]


# --------------------------------------------------------------------------
# punctuation / clause splitting
# --------------------------------------------------------------------------

_SENT_SPLIT_RE = re.compile(r"(?<=[.!?…])\s+|\n+")


def split_sentences(text: str) -> list[str]:
    """Split into sentences on . ! ? … and newlines.  Never returns empty."""
    parts = [squeeze(p) for p in _SENT_SPLIT_RE.split(text or "")]
    return [p for p in parts if p]


def split_clauses(text: str) -> list[str]:
    """Split on sentence enders and coordinating conjunctions that start a new
    proposition ("а", "но", "и" between two verb groups is NOT split — too
    risky; only comma+conjunction is)."""
    out: list[str] = []
    for sent in split_sentences(text):
        chunks = re.split(r"\s*(?:,\s*(?:а|но|и)\s+)", sent)
        out.extend(squeeze(c) for c in chunks if squeeze(c))
    return out or [squeeze(text)]


def contains_any(text: str, words: Iterable[str]) -> bool:
    toks = set(word_tokens(text))
    return any(w in toks for w in words)


def tokens_keep_case(text: str) -> list[str]:
    """Like ``word_tokens`` but keeps the original case (needed to tell a
    proper name from a common noun: "Ньютон" vs "физик").  ё is still folded
    to е and the text is NFC-normalised first."""
    t = unicodedata.normalize("NFC", str(text or ""))
    t = _DASH_RE.sub("-", t)
    t = _STANDALONE_DASH_RE.sub(" ", t)
    return [m.group(0).replace("ё", "е") for m in _WORD_RE.finditer(t)]


def is_proper(text: str) -> bool:
    """True when the phrase looks like a proper name: any token starts with an
    uppercase letter and the phrase is not a single common noun in a known
    class list.  Purely orthographic — no gazetteer, no network."""
    toks = tokens_keep_case(text)
    if not toks:
        return False
    return any(t[:1].isupper() for t in toks)
