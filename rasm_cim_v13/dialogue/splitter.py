"""Split one user message into ordered sub-questions (v0.13).

Rules (deterministic, order-preserving):
  1. split on '?' — each kept fragment retains its '?';
  2. each fragment split on ';';
  3. numbered lists "1) ... 2) ..." / "1. ... 2. ..." are split FIRST (before ?),
     keeping the enumeration stripped;
  4. conjunctions ", и " / ", а " / " и " split a fragment ONLY when both sides
     look like questions (question word, 'ли', trailing '?', or yes/no aux);
  5. a leading greeting ("Привет, ...") is emitted as its own item.

Everything else stays whole. Max 8 sub-questions (extra merged into the 8th).
"""
from __future__ import annotations

import re
from typing import List

_QWORD = re.compile(
    r"\b(кто|что|где|когда|откуда|куда|почему|зачем|сколько|как(ой|ая|ое|ие)?|чей|чья|"
    r"который|которая|есть ли|правда ли|верно ли|умеет ли|может ли|бывает ли|можно ли|"
    r"назови|перечисли)\b", re.IGNORECASE)
_GREETING_LEAD = re.compile(
    r"^\s*(привет(ик)?|здравствуй(те)?|доброе утро|добрый день|добрый вечер|салют|хай)\b",
    re.IGNORECASE)
_NUM_SPLIT = re.compile(r"\s*(?:^|(?<=\s))(\d{1,2})[).]\s+")
_LEADING_CONJ = re.compile(r"^(и|а|но|да)\s+", re.IGNORECASE)


def _looks_like_question(s: str) -> bool:
    s = s.strip()
    if not s:
        return False
    if s.rstrip().endswith("?"):
        return True
    return bool(_QWORD.search(s))


def _split_numbered(text: str) -> List[str]:
    parts = _NUM_SPLIT.split(text)
    if len(parts) <= 2:
        return [text]
    # parts: [pre, "1", body1, "2", body2, ...]
    out: List[str] = []
    if parts[0].strip():
        out.append(parts[0].strip())
    it = iter(parts[1:])
    for _, body in zip(it, it):
        body = body.strip()
        if body:
            out.append(body)
    return out or [text]


def _split_conjunctions(fragment: str) -> List[str]:
    for sep in (", и ", ", а ", " и "):
        if sep not in fragment.casefold():
            continue
        idx = fragment.casefold().index(sep)
        left, right = fragment[:idx], fragment[idx + len(sep):]
        # both sides must look like questions, else keep whole
        if _looks_like_question(left) and _looks_like_question(right):
            return (_split_conjunctions(left) + _split_conjunctions(right))
    return [fragment]


def split_questions(text: str) -> List[str]:
    if not text or not text.strip():
        return []
    items: List[str] = []
    # leading greeting?
    m = _GREETING_LEAD.match(text)
    rest = text
    if m and "," in text[: m.end() + 2]:
        comma = text.index(",", m.start())
        greeting, rest = text[:comma].strip(), text[comma + 1:].strip()
        if greeting:
            items.append(greeting)
    for block in _split_numbered(rest):
        for piece in block.split(";"):
            piece = piece.strip()
            if not piece:
                continue
            marked = piece if piece.endswith("?") else piece
            for sub in marked.split("?"):
                sub = sub.strip()
                if sub:
                    items.append(sub + "?")
    expanded: List[str] = []
    for item in items:
        if item.endswith("?"):
            core = item[:-1]
            parts = _split_conjunctions(core)
            for p in parts:
                p = p.strip()
                if p:
                    expanded.append(p + "?")
        else:
            expanded.extend(_split_conjunctions(item))
    # dedup adjacent exact duplicates, cap at 8
    out: List[str] = []
    for e in expanded:
        e = _LEADING_CONJ.sub("", e).strip()
        if not e:
            continue
        if not out or out[-1].casefold() != e.casefold():
            out.append(e)
    if len(out) > 8:
        out = out[:7] + [" ".join(out[7:])]
    return out
