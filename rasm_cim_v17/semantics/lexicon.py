"""Словарь частиц, рамок вопроса и профессий + нормализация реплики.

Задача: не менять ядро v0.16, но довести до него реплики в той форме, в какой
люди реально пишут.  «Ну он, типа, физик?», «Правда ли, что она врач?»,
«Он работает инженером?» — это те же вопросы о классе, что и «Он физик?»,
только с частицами, рамкой и творительным падежом.

Правила общие (частицы, рамки, падежное окончание), а не список тестовых фраз:
любое имя и любой класс проходят через те же шаги.  Словарь профессий нужен,
чтобы расширять базу знаний (`knowledge/catalog.py`) и знать род класса.
"""
from __future__ import annotations

import re

#: вводные частицы и слова-заполнители, которые не несут смысла для разбора
FILLERS = (
    "ну", "ага", "угу", "так", "итак", "значит", "короче", "слушай",
    "слушай-ка", "слышь", "ладно", "хорошо", "окей", "ок", "пожалуйста",
    "скажи", "скажи-ка", "напомни", "подскажи", "пожалуй", "в общем",
    "вообще", "типа", "как бы", "просто", "давай", "давайте", "и вот",
    "стало быть", "выходит", "получается", "следовательно", "значит так",
    # таджикские вставки-заполнители.  «он» здесь быть не может: в русском это
    # местоимение, и его удаление ломало разбор («Он поэт?» → «Поэт?»).
    "бале", "хуб", "салам", "салом", "чӣ", "кӣ", "ҳа", "ха", "дуст",
)
#: частицы, которые могут стоять внутри вопроса («он ведь физик?»)
INLINE_PARTICLES = ("ведь", "же", "ж", "таки", "всё-таки", "все-таки")
#: хвосты, которые превращают утверждение в вопрос-переспрос
TRAILING_TAGS = (
    "верно", "правда", "не так ли", "да", "так ведь", "не правда ли",
    "наверное", "кажется", "вроде", "по-моему", "или нет", "я прав",
    "я права", "ага",
    # оговорки в конце утверждения: «Пётр — инженер, как я слышал»
    "как я слышал", "как я слышала", "если не ошибаюсь", "если честно",
    "как говорят", "как известно", "по слухам", "как мне кажется",
    "я думаю", "вроде бы", "это точно", "насколько я знаю",
)
#: рамки вопроса об истинности: «правда ли, что X» → «X»
FRAME_PATTERNS = (
    re.compile(r"^\s*(?:а\s+)?(?:правда\s+ли|верно\s+ли|разве|неужели)\s*,?\s*что\s+",
               re.IGNORECASE),
    re.compile(r"^\s*(?:а\s+)?(?:ты\s+)?(?:не\s+)?(?:подскажешь|скажешь|знаешь)\s*,?\s*",
               re.IGNORECASE),
    re.compile(r"^\s*(?:а\s+)?(?:можно\s+)?(?:уточнить|спросить)\s*[:,]?\s*",
               re.IGNORECASE),
    re.compile(r"^\s*(?:вот\s+)?(?:это\s+)?(?:самое|сам)\s+", re.IGNORECASE),
)
#: «Он работает физиком?» / «Она трудится врачом?» → «Он физик?»
WORK_PATTERN = re.compile(
    r"\b(работает|работал|работала|трудится|трудился|трудилась|"
    r"является|был|была|будет)\b\s+", re.IGNORECASE)
#: «Он физик по профессии?» → «Он физик?»
BY_TRADE = re.compile(r"\s+по\s+(?:профессии|специальности|образованию)\b",
                      re.IGNORECASE)
#: окончания творительного падежа, которые снимаются перед классом
INSTRUMENTAL = re.compile(r"(ом|ем|ой|ей|ью|ю)$", re.IGNORECASE)

#: профессии/классы: род класса и область — для базы знаний и для проверки
PROFESSIONS: dict[str, dict[str, str]] = {
    "физик": {"gender": "m", "sphere": "наука"},
    "химик": {"gender": "m", "sphere": "наука"},
    "математик": {"gender": "m", "sphere": "наука"},
    "биолог": {"gender": "m", "sphere": "наука"},
    "геолог": {"gender": "m", "sphere": "наука"},
    "астроном": {"gender": "m", "sphere": "наука"},
    "историк": {"gender": "m", "sphere": "наука"},
    "медик": {"gender": "m", "sphere": "медицина"},
    "врач": {"gender": "m", "sphere": "медицина"},
    "инженер": {"gender": "m", "sphere": "техника"},
    "механик": {"gender": "m", "sphere": "техника"},
    "пилот": {"gender": "m", "sphere": "транспорт"},
    "водитель": {"gender": "m", "sphere": "транспорт"},
    "строитель": {"gender": "m", "sphere": "техника"},
    "учитель": {"gender": "m", "sphere": "образование"},
    "преподаватель": {"gender": "m", "sphere": "образование"},
    "переводчик": {"gender": "m", "sphere": "языки"},
    "юрист": {"gender": "m", "sphere": "право"},
    "бухгалтер": {"gender": "m", "sphere": "экономика"},
    "архитектор": {"gender": "m", "sphere": "искусство"},
    "поэт": {"gender": "m", "sphere": "искусство"},
    "писатель": {"gender": "m", "sphere": "искусство"},
    "художник": {"gender": "m", "sphere": "искусство"},
    "музыкант": {"gender": "m", "sphere": "искусство"},
    "актёр": {"gender": "m", "sphere": "искусство"},
    "актер": {"gender": "m", "sphere": "искусство"},
    "швея": {"gender": "f", "sphere": "производство"},
    "повар": {"gender": "m", "sphere": "производство"},
    "программист": {"gender": "m", "sphere": "техника"},
    "экономист": {"gender": "m", "sphere": "экономика"},
    "журналист": {"gender": "m", "sphere": "медиа"},
    "фотограф": {"gender": "m", "sphere": "искусство"},
    "ветеринар": {"gender": "m", "sphere": "медицина"},
    "агроном": {"gender": "m", "sphere": "сельское хозяйство"},
}


def _strip_fillers(text: str) -> str:
    words = text.strip().split()
    while words and words[0].casefold().strip(",—-") in FILLERS:
        words.pop(0)
    return " ".join(words)


#: многословные заполнители: убираются до поштучного разбора
PHRASE_FILLERS = ("как бы", "в общем", "стало быть", "значит так", "и вот",
                  "всё-таки", "все-таки", "то есть")


def _strip_inline(text: str) -> str:
    """Убрать частицы и заполнители в любом месте реплики, не только в начале.

    «Ну он, типа, физик?» и «Стало быть, он врач?» — тот же вопрос о классе;
    удаляются только слова из FILLERS/INLINE_PARTICLES, ни одно имя или класс
    в этих списках не встречается (проверяется тестом).
    """
    for phrase in PHRASE_FILLERS:
        text = re.sub(r"\b" + re.escape(phrase) + r"\b", " ", text,
                      flags=re.IGNORECASE)
    drop = {w.casefold() for w in FILLERS} | {w.casefold() for w in INLINE_PARTICLES}
    words = [w for w in text.split() if w.casefold().strip(",.!?—-") not in drop]
    return " ".join(words)


def _strip_trailing(text: str) -> str:
    trimmed = text.strip().rstrip(".!?…")
    changed = True
    while changed:
        changed = False
        for tag in TRAILING_TAGS:
            pattern = re.compile(r"[,\s—-]+" + re.escape(tag) + r"$", re.IGNORECASE)
            if pattern.search(trimmed):
                trimmed = pattern.sub("", trimmed)
                changed = True
    return trimmed


def _strip_frames(text: str) -> str:
    for pattern in FRAME_PATTERNS:
        text = pattern.sub("", text)
    return text


#: обратный порядок: «астрономом работает» → «астроном»
REVERSED_WORK = re.compile(
    r"\b([А-Яа-яЁё]+(?:ом|ем|ой|ей|ью))\s+(?:работает|работал|работала|"
    r"трудится|трудился|трудилась)\b", re.IGNORECASE)


def _reversed_work(text: str) -> str:
    def repl(match: re.Match) -> str:
        word = match.group(1)
        base = INSTRUMENTAL.sub("", word.casefold())
        return base if base in PROFESSIONS else match.group(0)

    return REVERSED_WORK.sub(repl, text)


def _instrumental_to_nominative(text: str) -> str:
    """«работает физиком» → «физик»: снимаем глагол и падежное окончание."""
    if not WORK_PATTERN.search(text):
        return text
    text = WORK_PATTERN.sub("", text)
    words = []
    for word in text.split():
        low = word.casefold().strip(",!?.")
        base = INSTRUMENTAL.sub("", low)
        if base != low and base in PROFESSIONS:
            words.append(base)
        else:
            words.append(word)
    return " ".join(words)


def normalize(text: str) -> str:
    """Привести реплику к канонической форме, понятной разбору v0.16."""
    original = (text or "").strip()
    if not original:
        return original
    result = _strip_frames(original)
    result = _strip_fillers(result)
    result = _reversed_work(result)
    result = _instrumental_to_nominative(result)
    result = BY_TRADE.sub("", result)
    result = _strip_inline(result)
    result = _strip_trailing(result)
    result = _strip_fillers(result)
    # остатки пунктуации после удаления частиц: «, он врач?», «Он, физик?»
    result = re.sub(r"\s*,\s*(?=,)", ",", result)
    result = re.sub(r"^[\s,;—-]+", "", result)
    result = re.sub(r"\s*,\s*([,.!?…])", r"\1", result)
    result = re.sub(r"\b(он|она|они|его|её|ее|ей|их)\s*,\s*", r"\1 ", result,
                    flags=re.IGNORECASE)
    result = re.sub(r"\.{2,}\s*\?", "?", result)
    # многоточие внутри реплики («Она... историк?») разбор не понимает
    result = re.sub(r"\.{2,}", " ", result)
    result = re.sub(r"\s+([,.!?…])", r"\1", result)
    result = re.sub(r"\s{2,}", " ", result).strip()
    if not result:
        return original
    first = result[0]
    if first.islower():
        result = first.upper() + result[1:]
    # знак вопроса сохраняем: разбор различает вопрос и утверждение
    if original.rstrip().endswith("?") and not result.endswith("?"):
        result += "?"
    return result


def profession_of(word: str) -> dict[str, str] | None:
    return PROFESSIONS.get((word or "").casefold().strip(",.!?"))


def class_gender(word: str) -> str:
    info = profession_of(word)
    return info["gender"] if info else ""
