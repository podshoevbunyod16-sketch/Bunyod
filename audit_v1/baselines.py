"""Базовые стратегии аудита: B1 всегда отказывать, B2 последнее имя,
B3 последнее существительное нужного рода.

Они не используют ни базу знаний проекта, ни разметку: только текст диалога и
окончание слова (род по `-а/-я` → женский, иначе мужской). Это заведомо
грубее системы, и именно поэтому цифры обязательны: без них нельзя сказать,
есть ли от системы польза.
"""
from __future__ import annotations

import re

TOKEN = re.compile(r"[А-ЯЁA-Z][а-яёa-z\-]+")
PRONOUNS = {"он", "она", "они", "его", "её", "ее", "ему", "ей", "их", "им"}

#: слова, которые стоят в начале предложения и не являются именами
STARTERS = {
    "привет", "привт", "здравствуй", "здравствуйте", "салам", "салом", "бале",
    "хуб", "кто", "что", "как", "где", "когда", "почему", "зачем", "а", "ну",
    "ладно", "хорошо", "спасибо", "спасиба", "понятно", "значит", "так",
    "слушай", "слушай-ка", "короче", "начнём", "начинаем", "итак", "расскажи",
    "скажи", "скажи-ка", "напомни", "правда", "не", "то", "выходит", "ага",
    "опять", "снова", "поговорим", "говорили", "есть", "тут", "мы", "и",
    "если", "наверное", "стало", "быть", "ин", "чӣ", "кӣ", "дуст", "хона",
    "kitob", "не", "нет", "да", "давай", "хватит", "снова", "про", "это",
    "может", "верно", "ведь", "разве", "неужели", "именно", "теперь", "затем",
}


def _mentions(turns: list[str], *, skip_last: bool = True) -> list[str]:
    """Имена собственные из реплик (по умолчанию — без последней)."""
    source = turns[:-1] if skip_last else list(turns)
    found: list[str] = []
    for text in source:
        for sentence in re.split(r"[.!?…]+", text):
            words = TOKEN.findall(sentence)
            for word in words:
                low = word.casefold()
                # имя определяем по заглавной букве и по тому, что слово не
                # служебное: список основ предложений вынесен в STARTERS
                if low in PRONOUNS or low in STARTERS:
                    continue
                found.append(word)
    return found


def _gender(word: str) -> str:
    low = word.casefold()
    return "f" if low.endswith(("а", "я")) else "m"


def _pronoun_of(case) -> str:
    return (case.pronoun or "").casefold()


def _pronoun_gender(case) -> str:
    pronoun = _pronoun_of(case)
    if pronoun in {"она", "её", "ее", "ей"}:
        return "f"
    if pronoun in {"они", "их"}:
        return "p"
    return "m"


def always_refuse(case) -> dict:
    return {"status": "REFUSE", "bound_to": "", "refused": True}


def last_name(case) -> dict:
    names = _mentions(list(case.turns))
    if not names:
        return {"status": "REFUSE", "bound_to": "", "refused": True}
    return {"status": "BIND", "bound_to": names[-1], "refused": False}


def last_matching_gender(case) -> dict:
    wanted = _pronoun_gender(case)
    if wanted == "p":
        return {"status": "REFUSE", "bound_to": "", "refused": True}
    names = [n for n in _mentions(list(case.turns)) if _gender(n) == wanted]
    if not names:
        return {"status": "REFUSE", "bound_to": "", "refused": True}
    return {"status": "BIND", "bound_to": names[-1], "refused": False}


BASELINES = {
    "B1: всегда отказывать": always_refuse,
    "B2: последнее имя": last_name,
    "B3: последнее имя нужного рода": last_matching_gender,
}
