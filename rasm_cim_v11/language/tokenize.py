"""v0.11 tokenizer — offsets refer to the ORIGINAL string.

Manual char-scan (no regex over the whole line): Unicode letters
(str.isalpha, covers Cyrillic/Latin/fullwidth…) group into "word" tokens
with intra-word hyphen compounds; Unicode digits group into "number"
tokens (optional single . or , decimal separator); every other non-space
symbol is one "punct" token. Whitespace is skipped, nothing else dropped —
token coverage of the original is complete except whitespace.

Each token carries `norm` = NFKC+casefold of its surface form, so matching
works cross-width/case while `start`/`end` always index the ORIGINAL
string (and slice back to exactly `text`).
"""
from __future__ import annotations

import unicodedata
from typing import List, Tuple

from .models import Token

_HYPHENS = "-‐‑‒–—―"


def _norm_token(raw: str) -> str:
    return unicodedata.normalize("NFKC", raw).casefold()


def tokenize(text: str) -> Tuple[Token, ...]:
    if not isinstance(text, str):
        raise TypeError(f"tokenize() expects str, got {type(text).__name__}")
    tokens: List[Token] = []
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if ch.isspace():
            i += 1
            continue
        if ch.isalpha():
            j = i + 1
            while j < n:
                c = text[j]
                if c.isalpha() or c in _HYPHENS and j + 1 < n and text[j + 1].isalpha():
                    if c in _HYPHENS:
                        # consume hyphen only if followed by a letter; keep it in token
                        j += 1
                        continue
                    j += 1
                else:
                    break
            tokens.append(Token(text[i:j], _norm_token(text[i:j]), i, j, "word"))
            i = j
            continue
        if ch.isdigit():
            j = i + 1
            while j < n:
                c = text[j]
                if c.isdigit():
                    j += 1
                elif c in ".," and j + 1 < n and text[j + 1].isdigit():
                    j += 2  # decimal separator + digit run continued next iters
                else:
                    break
            tokens.append(Token(text[i:j], _norm_token(text[i:j]), i, j, "number"))
            i = j
            continue
        tokens.append(Token(ch, _norm_token(ch), i, i + 1, "punct"))
        i += 1
    return tuple(tokens)


def token_text(tokens) -> str:
    return " ".join(t.norm for t in tokens)
