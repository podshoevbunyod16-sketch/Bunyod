"""Работа с текстом для агентного слоя: разбиение, отбор, сжатие.

Никакой генерации «из головы»: всё, что слой отвечает своими словами, — это
**сжатие** отобранных предложений источника (выбор, сокращение, порядок) с
обязательной ссылкой на источник. Поэтому «своими словами» здесь означает
«пересказ отобранного», а не свободную формулировку; в отчёте это сказано
прямо.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

STOPWORDS = {
    "и", "в", "во", "не", "что", "он", "на", "я", "с", "со", "как", "а", "то",
    "все", "она", "так", "его", "но", "да", "ты", "к", "у", "же", "вы", "за",
    "бы", "по", "только", "ее", "мне", "было", "вот", "от", "меня", "еще",
    "нет", "о", "из", "ему", "теперь", "когда", "даже", "ну", "вдруг", "ли",
    "если", "уже", "или", "ни", "быть", "был", "него", "до", "вас", "нибудь",
    "опять", "уж", "вам", "ведь", "там", "потом", "себя", "ничего", "ей",
    "может", "они", "тут", "где", "есть", "надо", "ней", "для", "мы", "тебя",
    "их", "чем", "была", "сам", "чтоб", "без", "будто", "чего", "раз", "тоже",
    "себе", "под", "будет", "ж", "тогда", "кто", "этот", "того", "потому",
    "этого", "какой", "совсем", "ним", "здесь", "этом", "один", "почти",
    "мой", "тем", "чтобы", "нее", "были", "куда", "зачем", "всех", "никогда",
    "можно", "при", "наконец", "два", "об", "другой", "хоть", "после", "над",
    "больше", "тот", "через", "эти", "нас", "про", "всего", "них", "какая",
    "много", "разве", "три", "эту", "моя", "впрочем", "хорошо", "свою", "этой",
    "перед", "иногда", "лучше", "чуть", "том", "нельзя", "такой", "им", "более",
    "всегда", "конечно", "всю", "между", "это", "который", "которая", "которые",
    "которое", "также", "это", "кто", "такой", "такая", "такое", "был", "является",
}

_SENT_SPLIT = re.compile(r"(?<=[.!?…])\s+(?=[«\"(]?[А-ЯЁA-Z0-9])")
_HYPHENS = str.maketrans({"–": "-", "—": "-"})


def tokens(text: str) -> list[str]:
    """Значимые слова в нижнем регистре (ё → е, без стоп-слов)."""
    words = re.findall(r"[А-Яа-яЁёA-Za-z][А-Яа-яЁёA-Za-z\-]{1,}", text)
    out = []
    for word in words:
        low = word.casefold().replace("ё", "е")
        if len(low) < 3 or low in STOPWORDS:
            continue
        out.append(low)
    return out


def sentences(text: str) -> list[str]:
    """Нарезать текст на предложения, отбросив обрывки короче 25 символов."""
    text = re.sub(r"\s+", " ", text or "").strip()
    if not text:
        return []
    parts = _SENT_SPLIT.split(text)
    return [part.strip() for part in parts if len(part.strip()) >= 25]


def content_terms(question: str) -> list[str]:
    """Что именно спрашивают: значимые слова вопроса без вопросительных слов."""
    skip = {"кто", "что", "такой", "такая", "такое", "это", "расскажи",
            "расскажите", "поведай", "про", "объясни", "объясните", "почему",
            "зачем", "как", "когда", "где", "сколько", "чем", "какой", "какая",
            "какое", "какие", "ли", "мне", "пожалуйста", "кратко", "подробно"}
    return [term for term in tokens(question) if term not in skip]


@dataclass(frozen=True)
class Scored:
    sentence: str
    score: float


def rank(sentences_: list[str], question: str, *, limit: int = 3) -> list[Scored]:
    """Отобрать предложения по пересечению с вопросом.

    Вес: совпадение значимого слова вопроса — 1.0, вхождение слова вопроса как
    части слова — 0.4. Предложения без совпадений не считаются релевантными:
    лучше сказать «в источнике не нашлось», чем пересказать случайный абзац.
    """
    terms = content_terms(question)
    if not terms:
        return []
    expanded: list[str] = []
    for item in sentences_:
        parts = sentences(item)
        expanded.extend(parts or [item])
    sentences_ = expanded
    seen = set()
    scored: list[Scored] = []
    for sentence in sentences_:
        key = sentence.casefold()[:60]
        if key in seen:
            continue
        seen.add(key)
        low = sentence.casefold().replace("ё", "е")
        score = 0.0
        for term in terms:
            if re.search(r"\b" + re.escape(term), low):
                score += 1.0
            elif len(term) >= 5 and term[:5] in low:
                score += 0.4
        if score <= 0:
            continue
        # ранняя позиция чуть важнее: в summary определение обычно первым
        score += 0.2 / (1 + sentences_.index(sentence))
        scored.append(Scored(sentence, round(score, 3)))
    scored.sort(key=lambda item: item.score, reverse=True)
    return scored[:limit]


def shorten(sentence: str, *, max_chars: int = 260) -> str:
    """Обрезать длинное предложение по границе придаточного, не по слову."""
    sentence = sentence.strip()
    if len(sentence) <= max_chars:
        return sentence
    cut = sentence[:max_chars]
    for marker in ("; ", " — ", ", который", ", которая", ", которые", " ("):
        index = cut.rfind(marker)
        if index > max_chars // 2:
            return cut[:index].rstrip(" ,;—-") + "…"
    return cut.rsplit(" ", 1)[0] + "…"


def paraphrase(selected: list[Scored], question: str, *,
               max_chars: int = 420) -> str:
    """Собрать ответ «своими словами» из отобранных предложений.

    Правила: не больше двух предложений, каждое сокращается, повторы
    выбрасываются, связка — по типу вопроса (кто/что → «Это», иначе «Вот что
    известно:»). Ни одного слова, которого нет в источнике, слой не добавляет,
    кроме этой связки.
    """
    if not selected:
        return ""
    picked: list[str] = []
    used: set[str] = set()
    for item in selected:
        piece = shorten(item.sentence)
        key = " ".join(tokens(piece))
        if not key or key in used:
            continue
        # предложение, целиком повторяющее предыдущее по значимым словам, не берём
        if any(set(key.split()) <= set(old.split()) for old in used):
            continue
        used.add(key)
        picked.append(piece)
        if len(picked) == 2:
            break
    if not picked:
        return ""
    body = " ".join(picked)
    if len(body) > max_chars:
        body = body[:max_chars].rsplit(" ", 1)[0] + "…"
    low_question = question.casefold()
    lead = "Это " if re.search(r"\b(кто|что)\b", low_question) else "Вот что известно: "
    if lead == "Это " and picked[0][:1].isupper() and not re.match(
            r"^(Это|Он|Она|Оно|Они)\b", picked[0]):
        lead = ""
    return (lead + body).strip()


def is_hedged(sentence: str) -> bool:
    """Источник сам сомневается — такой факт в память не пишем."""
    low = sentence.casefold()
    markers = ("по слухам", "якобы", "возможно,", "вероятно,", "предположительно",
               "некоторые считают", "считается, что", "по некоторым данным",
               "возможно ", "может быть")
    return any(marker in low for marker in markers)


def looks_like_definition(sentence: str) -> bool:
    """Предложение похоже на определение: «X — это Y» / «X — Y» / «X является Y»."""
    if "?" in sentence or is_hedged(sentence):
        return False
    sentence = sentence.strip().rstrip(".!;…")
    pattern = re.compile(
        r"^\s*([А-ЯЁA-Z][\w\-]+(?:\s+[А-ЯЁA-Z][\w\-]+){0,3})\s+"
        r"(?:—|-|–)\s+(?:это\s+)?([^.;!?]{2,60})$|"
        r"^\s*([А-ЯЁA-Z][\w\-]+(?:\s+[А-ЯЁA-Z][\w\-]+){0,3})\s+является\s+([^.;!?]{2,60})$")
    return bool(pattern.match(sentence.strip()))


_DEF_PATTERNS = (
    re.compile(r"^\s*(?P<subject>[А-ЯЁA-Z][\w\-]+(?:\s+[\w\-]+){0,3})\s+"
               r"(?:—|-|–)\s+(?:это\s+)?(?P<object>[^.;!?]{2,60})\s*[.;!?]?$"),
    re.compile(r"^\s*(?P<subject>[А-ЯЁA-Z][\w\-]+(?:\s+[\w\-]+){0,3})\s+"
               r"является\s+(?P<object>[^.;!?]{2,60})\s*[.;!?]?$"),
)


def parse_definition(sentence: str) -> tuple[str, str] | None:
    """Вытащить «X — это Y» из предложения источника, если оно определение."""
    sentence = (sentence or "").strip()
    if not looks_like_definition(sentence):
        return None
    for pattern in _DEF_PATTERNS:
        match = pattern.match(sentence)
        if not match:
            continue
        subject = re.sub(r"\s+", " ", match.group("subject")).strip()
        obj = re.sub(r"\s+", " ", match.group("object")).strip(" ,—-«»\"")
        if not subject or not obj:
            continue
        if len(obj.split()) > 6:            # длинная фраза — это не термин
            continue
        return subject, obj
    return None
