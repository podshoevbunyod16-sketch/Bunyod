"""Russian case recovery for mentions, without a morphology dictionary.

The frozen resolver in v0.16 handles some forms (``Ивана`` -> ``Иван``) but not
others (``Анну`` -> ``Анна``, ``Толстого`` -> ``Толстой``).  Rewriting it is out
of scope, so this module recovers a *candidate base form* for a mention and lets
the existing resolver decide:

    я про Анну            -> «Анну» не резолвится
                          -> кандидат «Анн»? нет; «Анна»? да, EXACT

Rules are endings, not a lexicon; they are ordered from most to least specific
and a candidate is accepted only if the resolver returns an exact/alias/
normalised match.  Fuzzy matches are **not** used here: this path exists to say
"the user means that entity", and a fuzzy guess would be exactly the kind of
unjustified binding the benchmark counts as an error.
"""
from __future__ import annotations

#: (ending, replacement) — accusative/genitive forms of names and nouns
_ENDINGS: tuple[tuple[str, str], ...] = (
    ("ого", "ой"),      # Толстого -> Толстой
    ("его", "ий"),      # Королёва… (редко), оставлено для единообразия
    ("ую", "ая"),
    ("юю", "яя"),
    ("у", "а"),         # Анну -> Анна, Марию -> Мария (через «ю» -> «я»)
    ("ю", "я"),         # Марию -> Мария
    ("а", ""),          # Ивана -> Иван
    ("я", ""),          # Игоря -> Игорь (мягкий знак не восстановить — ниже)
    ("ы", ""),
    ("и", ""),
)

#: accepted resolution statuses — anything weaker is not a binding
_TRUSTED = ("EXACT", "ALIAS", "NORMALIZED", "MORPHOLOGICAL")

#: how many candidate forms are tried before giving up
MAX_CANDIDATES = 6


def base_forms(mention: str) -> list[str]:
    """Candidate dictionary forms for a mention, most specific first."""
    mention = (mention or "").strip()
    if not mention:
        return []
    lowered = mention.casefold()
    out: list[str] = []
    for ending, replacement in _ENDINGS:
        if not lowered.endswith(ending) or len(lowered) <= len(ending) + 1:
            continue
        stem = mention[: len(mention) - len(ending)]
        candidate = stem + replacement
        if candidate.casefold() != lowered and candidate not in out:
            out.append(candidate)
    # «ё» пишут как «е» — это не падеж, а орфография
    for candidate in list(out):
        swapped = candidate.replace("ё", "е")
        if swapped != candidate and swapped not in out:
            out.append(swapped)
    return out[:MAX_CANDIDATES]


def resolve_form(mention: str, resolver) -> tuple[int | None, str]:
    """``(entity_id, matched_form)`` — direct form first, then recovered forms.

    Returns ``(None, "")`` when nothing trustworthy is found, so the caller can
    refuse rather than bind.
    """
    direct = resolver.resolve(mention)
    if direct.entity_id is not None and direct.status in _TRUSTED:
        return direct.entity_id, mention
    for candidate in base_forms(mention):
        res = resolver.resolve(candidate)
        if res.entity_id is None:
            continue
        if res.status not in _TRUSTED:
            continue
        # two different entities matching the same recovered form = ambiguity
        ids = {c.entity_id for c in res.candidates
               if c.status in _TRUSTED}
        if len(ids) > 1:
            return None, ""
        return res.entity_id, candidate
    return None, ""
