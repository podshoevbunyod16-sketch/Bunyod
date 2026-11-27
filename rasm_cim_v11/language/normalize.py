"""v0.11 normalization — deterministic, reversible-where-it-matters.

DECISION: NFKC (not NFC). Justification:
  * NFKC folds compatibility duplicates that real user input contains when
    text is pasted from documents/chat (e.g. ﬁ→fi, full-width digits,
    superscripts). For a *matching* layer this reduces missed matches.
  * NFC only composes/decomposes canonical sequences and would leave those
    duplicates in place, splitting identical facts across two byte forms.
  * Risk of NFKC (folding ²→2, ¼→1⁄4) is acceptable here because
    normalization happens at INPUT time only; stored facts are never
    silently rewritten, and `original_text` is always preserved verbatim.
  * Either way the choice is documented, deterministic and versioned
    (this file is part of format lang-0.11; changes require a new version).

Other decisions:
  * casefold() instead of lower() — Unicode-correct caseless matching.
  * Quotes/dashes/space-likes are unified; invisible/format chars
    (zero-width space, BOM, word joiner, soft hyphen) are REMOVED with a flag.
  * Runs of repeated terminal punctuation (!!!, ???) collapse to one.
  * Internal punctuation of words (hyphenated compounds) is kept.
  * The original string is NEVER mutated; callers get NormalizedText with
    both forms plus the list of applied transformation flags.
"""
from __future__ import annotations

import re
import unicodedata

from .models import NormalizedText

_QUOTE_MAP = {
    "«": '"', "»": '"', "„": '"', "“": '"', "”": '"', "‟": '"',
    "‘": "'", "’": "'", "‚": "'", "‛": "'", "`": "'", "´": "'",
}
_DASH_CHARS = "‐‑‒–—―−"          # hyphen-likes (NOT ascii '-')
_INVISIBLES = ["\u200b", "\u200c", "\u200d", "\ufeff", "\u2060", "\u00ad",
               "\u200e", "\u200f"]
_SPACE_LIKE = "\u00a0\u1680\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008" \
              "\u2009\u200a\u202f\u205f\u3000\t\r\n"
_RE_SPACES = re.compile(r"[ ]{2,}")
_RE_REPEAT_TERMINAL = re.compile(r"([!?.]){2,}")
_CYRILLIC = re.compile(r"[А-Яа-яЁё]")


def _strip_invisibles(s: str):
    flags = []
    for ch in _INVISIBLES:
        if ch in s:
            s = s.replace(ch, "")
            flags.append("invisibles_removed")
    return s, flags


def normalize(text: str) -> NormalizedText:
    if not isinstance(text, str):
        raise TypeError(f"normalize() expects str, got {type(text).__name__}")
    original = text
    flags: list[str] = []

    s, inv_flags = _strip_invisibles(text)
    flags.extend(inv_flags)

    nfkc = unicodedata.normalize("NFKC", s)
    if nfkc != s:
        flags.append("nfkc")
    s = nfkc

    for src, dst in _QUOTE_MAP.items():
        if src in s:
            s = s.replace(src, dst)
            flags.append("quotes_unified")
    for dash in _DASH_CHARS:
        if dash in s:
            s = s.replace(dash, "-")
            flags.append("dashes_unified")
    for sp in _SPACE_LIKE:
        if sp in s:
            s = s.replace(sp, " ")
            flags.append("space_likes_unified")

    cf = s.casefold()
    if cf != s:
        flags.append("casefold")
    s = cf

    # trim isolated single quotes/brackets noise at edges; keep internal punctuation
    s = s.strip()

    collapsed = _RE_SPACES.sub(" ", s)
    if collapsed != s:
        flags.append("spaces_collapsed")
    s = collapsed

    t2 = _RE_REPEAT_TERMINAL.sub(r"\1", s)
    if t2 != s:
        flags.append("terminal_punct_collapsed")
    flags = list(dict.fromkeys(flags))  # dedupe, keep order

    return NormalizedText(
        original=original,
        text=t2,
        flags=tuple(flags),
        ended_with_question=t2.rstrip().endswith("?"),
        has_cyrillic=bool(_CYRILLIC.search(t2)),
    )


def fold_yo(text: str) -> str:
    """Matching-only fold: ё→е. NEVER use to rewrite stored data."""
    return text.replace("ё", "е").replace("Ё", "е")
