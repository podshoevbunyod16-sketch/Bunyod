"""Агентный диалог: уточняющие вопросы, поиск, ответ своими словами, память.

Чем это отличается от `ask`:

* `ask` — вопрос к базе. Нет сущности — «Не знаю», и всё;
* `AgentSession.step()` ведёт **диалог**: здоровается, спрашивает уточнение,
  предлагает поискать, ищет (если сеть включена), пересказывает найденное
  своими словами со ссылкой на источник и предлагает запомнить проверяемое —
  но записывает в граф только после подтверждения.

Границы честности заданы кодом, а не обещаниями:

* свободной генерации нет — ответ «своими словами» собран из предложений
  источника (`agent/text.py`), всё, чего нет в источнике, не появляется;
* без сети агент **не** делает вид, что искал: он говорит «сеть выключена»;
* ошибка инструмента возвращается как ошибка, а не превращается в ответ;
* в память (граф и словарь) факт попадает после «да» или после второго
  независимого источника.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from rasm_cim_v16.answer_engine import A_DIRECT, A_INFERRED, A_UNKNOWN

from ..pipeline import SemanticPipeline
from ..semantics.morphology import resolve_form
from .memory import AgentMemory
from .text import (content_terms, parse_definition, paraphrase, rank,
                   sentences, tokens)
from .tools import Web

GREETING = re.compile(r"^(привет\w*|привт|здравствуй\w*|здорово|салам\w*|салом|"
                      r"добрый (день|вечер)|доброе утро|хай|hello|hi)\b",
                      re.IGNORECASE)
THANKS = re.compile(r"^(спасибо|спасиба|благодарю|рахмат)\b", re.IGNORECASE)
BYE = re.compile(r"^(пока|до свидания|бай|до встречи)\b", re.IGNORECASE)
META = re.compile(r"(что ты умеешь|кто ты|как тебя зовут|что ты такое|"
                  r"твои возможности|как ты работаешь|ты бот|ты человек)",
                  re.IGNORECASE)
SEARCHISH = re.compile(r"\b(поищи|поискать|найди|найти|погугли|загугли|"
                       r"посмотри в интернете|поищи в интернете|из интернета|"
                       r"в интернете|search|google)\b", re.IGNORECASE)
YES = re.compile(r"^(да|ага|давай|конечно|верно|запомни|запиши|угу|ок|окей|"
                 r"yes|подтверждаю)\b", re.IGNORECASE)
NO = re.compile(r"^(нет|не надо|не нужно|не запоминай|no|отмена)\b", re.IGNORECASE)
QUESTION_START = re.compile(r"^(кто|что|какой|какая|какое|какие|где|когда|"
                            r"почему|зачем|сколько|чей|чем|расскажи|расскажите|"
                            r"объясни|объясните|опиши)\b", re.IGNORECASE)
COPULA = re.compile(r"\s[—–-]\s")

CAPABILITIES = (
    "Я работаю с фактами вида «X — это Y» и умею:\n"
    "• отвечать на вопросы по своей базе и связывать местоимения;\n"
    "• задавать уточняющий вопрос, если непонятно, о ком или о чём речь;\n"
    "• искать в сети (когда включён флаг --online) и пересказывать найденное "
    "своими словами со ссылкой;\n"
    "• запоминать проверяемое: сначала показываю, что собираюсь записать, "
    "сохраняю после «да».\n"
    "Чего я не умею: свободно рассуждать и сочинять, как большие языковые "
    "модели. Если ответа нет в базе и нет сети — я так и скажу."
)


@dataclass
class Reply:
    text: str
    kind: str = "answer"          # answer|clarify|offer|learn|smalltalk|error
    source: str = ""
    proposed: list[dict] = field(default_factory=list)
    learned: list[dict] = field(default_factory=list)
    meta: dict = field(default_factory=dict)


class AgentSession:
    """Диалог одного пользователя с инструментами и памятью."""

    def __init__(self, dm, home: str, *, online: bool = False,
                 endpoint: str | None = None, mode: str = "normal",
                 web: Web | None = None):
        self.dm = dm
        self.home = home
        self.mode = mode
        self.pipe = SemanticPipeline(dm)
        self.memory = AgentMemory(home)
        self.web = web or Web(home, online=online, endpoint=endpoint)
        self.topic = ""
        self.last_source = ""
        self._pending_definition: dict | None = None

    # ------------------------------------------------------------------ API
    @property
    def online(self) -> bool:
        return self.web.online

    def set_online(self, value: bool) -> None:
        self.web.online = bool(value)

    def step(self, line: str) -> Reply:
        text = (line or "").strip()
        if not text:
            return Reply("Напиши вопрос или фразу — я отвечу или переспрошу.",
                         kind="clarify")

        # 1. подтверждение или отказ по предложенным фактам
        if self.memory.pending_facts():
            if YES.match(text):
                return self._confirm()
            if NO.match(text):
                self.memory.drop_pending()
                self._pending_definition = None
                return Reply("Хорошо, ничего не записываю.", kind="learn")

        # 2. светская часть и вопросы о себе
        if GREETING.match(text):
            return Reply("Привет. Спроси о чём-нибудь или научи факту: "
                         "«Кулибин — это механик».", kind="smalltalk")
        if THANKS.match(text):
            return Reply("Пожалуйста. Что ещё разобрать?", kind="smalltalk")
        if BYE.match(text):
            return Reply("До связи.", kind="smalltalk")
        if META.search(text):
            return Reply(CAPABILITIES, kind="smalltalk")

        # 3. явная просьба поискать
        if SEARCHISH.search(text):
            topic = SEARCHISH.sub(" ", text).strip(" .,:;—-")
            topic = re.sub(r"^(про|о|об|насчёт|насчет)\s+", "", topic,
                           flags=re.IGNORECASE).strip()
            if not topic:
                return Reply("Что именно искать? Назови тему.",
                             kind="clarify")
            return self._research(topic, question=text)

        # 4. обучение фактом
        if "?" not in text and COPULA.search(text) and not QUESTION_START.match(text):
            return self._teach(text)

        # 5. вопрос к базе, затем — к сети
        if text.endswith("?") or QUESTION_START.match(text):
            return self._question(text)

        # 6. непонятно — уточняем, но с конкретным примером
        self.memory.note_word(text.split()[0])
        return Reply(
            "Не понял, что сделать. Уточни, пожалуйста: спросить («Кто такой "
            "Иван?»), научить («Иван — это физик») или поискать в сети "
            "(«поищи про Кулибина»)?", kind="clarify")

    # ------------------------------------------------------------- обучение
    def _teach(self, text: str) -> Reply:
        answer = self.pipe.respond(text, self.mode)
        rendered = answer.text or ""
        if answer.status == A_DIRECT and rendered.startswith("Записал"):
            parts = re.split(r"\s+[—–-]\s+(?:это\s+)?", text.strip(" ."), maxsplit=1)
            if len(parts) == 2:
                subject, obj = parts[0].strip(), parts[1].strip()
                self.memory.define(obj, f"родовой термин для «{subject}»",
                                   source="пользователь", confidence=0.8,
                                   kind="class")
                self.topic = subject
            for word in tokens(text):
                self.memory.note_word(word)
            clean = rendered.split("[")[0].strip()
            clean = re.sub(r"^(Записал|Понял|Запомнил)\s*:\s*", "", clean)
            return Reply(f"Запомнил: {clean}. Спроси про это или научи ещё.",
                         kind="learn")
        if answer.clarification:
            guess = self._guess_statement(text)
            if guess:
                subject, obj = guess
                # источник пустой: факт пришёл от пользователя, а не из сети
                self.memory.propose(subject, obj, source="", sentence=text)
                return Reply(
                    f"Моё ядро не разобрало эту формулировку само. "
                    f"Записать как «{subject} — это {obj}»? Ответь «да» или «нет».",
                    kind="offer", proposed=[{"subject": subject, "object": obj,
                                             "source": "пользователь"}])
            return Reply(answer.clarification, kind="clarify")
        return Reply(f"Не записал: {rendered or 'не понял формулировку'}. "
                     "Нужна форма «X — это Y».", kind="error")

    @staticmethod
    def _guess_statement(text: str) -> tuple[str, str] | None:
        """«Москва — столица России» → («Москва», «столица России»)."""
        clean = text.strip().rstrip(".!")
        parts = re.split(r"\s+[—–-]\s+", clean, maxsplit=1)
        if len(parts) != 2:
            return None
        subject = re.sub(r"^(это|вот|значит)\s+", "", parts[0].strip(),
                         flags=re.IGNORECASE)
        obj = re.sub(r"^это\s+", "", parts[1].strip(), flags=re.IGNORECASE)
        if not subject or not obj:
            return None
        if len(subject.split()) > 4 or len(obj.split()) > 5:
            return None
        if "?" in subject + obj:
            return None
        return subject, obj

    # -------------------------------------------------------------- вопросы
    def _question(self, text: str) -> Reply:
        answer = self.pipe.respond(text, self.mode)
        if answer.clarification:
            self.topic = self.topic or ""
            return Reply(answer.clarification, kind="clarify")
        if answer.status in (A_DIRECT, A_INFERRED) and (answer.text or "").strip():
            self.topic = text
            return Reply(answer.text, kind="answer",
                         meta={"status": answer.status,
                               "confidence": answer.confidence})
        topic = self._topic_of(text)
        if topic:
            lookup = self._kb_lookup(topic)
            if lookup is not None:
                return lookup
        vocab = self._vocab_lookup(text)
        if vocab is not None:
            return vocab
        if not topic:
            return Reply("О ком или о чём вопрос? Назови сущность целиком.",
                         kind="clarify")
        if self.online:
            return self._research(topic, question=text)
        return Reply(
            f"В базе про «{topic}» ничего нет. Могу поискать в интернете — "
            "для этого перезапусти меня с флагом --online (или скажи "
            "«поищи», если сеть уже включена).", kind="offer",
            meta={"topic": topic})

    # ------------------------------------------------- ответ по своей базе
    def _kb_lookup(self, term: str) -> Reply | None:
        """Ответить по базе, когда ядро сказало «нет»: класс, число, связи.

        Ядро v0.16 отвечает на вопрос об отдельной сущности, но «Кто такие
        физики?» для него — неизвестное слово. Здесь мы разрешаем слово
        морфологией (физики/физиков → физик) и отвечаем тем, что реально
        записано: состав класса или классификацию сущности.
        """
        if not term:
            return None
        eid, _form = resolve_form(term, self.dm.kg.resolver)
        if not eid:
            return None
        kg = self.dm.kg
        name = kg.name_of(eid)
        statements = [st for st in kg.all_statements()
                      if (st.polarity or "true") == "true"]
        members = [kg.name_of(st.subject) for st in statements
                   if st.object == eid and st.predicate == "instance_of"]
        classes = [kg.name_of(st.object) for st in statements
                   if st.subject == eid and st.predicate == "instance_of"]
        members = [m for m in dict.fromkeys(members) if m != name]
        classes = [c for c in dict.fromkeys(classes) if c != name]

        if members and not classes:
            shown = ", ".join(members[:8])
            tail = "" if len(members) <= 8 else f" и ещё {len(members) - 8}"
            return Reply(f"{name} — это класс. Что у меня записано: {shown}{tail} "
                         f"(всего {len(members)}).", kind="answer",
                         meta={"entity_id": eid, "members": members})
        if classes:
            about = f"{name} — {classes[0]}."
            if members:
                about += (f" Кроме того, {name} — класс, известные: "
                          + ", ".join(members[:6]))
            return Reply(about, kind="answer",
                         meta={"entity_id": eid, "classes": classes})
        others = [(st.predicate, kg.name_of(st.object)) for st in statements
                  if st.subject == eid and st.predicate != "instance_of"]
        if others:
            facts = "; ".join(f"{pred} → {obj}" for pred, obj in others[:5])
            return Reply(f"Про «{name}» у меня есть связи: {facts}. "
                         "Больше ничего не записано.", kind="answer",
                         meta={"entity_id": eid})
        return None

    def _vocab_lookup(self, text: str) -> Reply | None:
        """Ответ из словарного запаса: фраза целиком, затем отдельные слова."""
        terms = content_terms(text)
        if not terms:
            return None
        candidates = [" ".join(terms)] + sorted(terms, key=len, reverse=True)
        for candidate in candidates:
            entry = self.memory.describe(candidate)
            if entry:
                source = f" (источник: {entry['source']})" if entry.get("source") else ""
                return Reply(f"{entry['word']} — {entry['definition']}{source}.",
                             kind="answer", source=entry.get("source", ""),
                             meta={"vocab": entry})
        return None

    @staticmethod
    def _topic_of(text: str) -> str:
        terms = content_terms(text)
        terms = [t for t in terms if t not in {"столица", "город", "страна"}]
        if not terms:
            return ""
        words = re.findall(r"[А-ЯЁA-Z][\w\-]+|[а-яёa-z][\w\-]+", text)
        named = [w for w in words if w[:1].isupper()
                 and w.casefold().replace("ё", "е") in terms]
        if named:
            # имя собственное: берём самое длинное (Иван Кулибин лучше «Иван»)
            return max(named, key=len)
        return max(terms, key=len)

    # ------------------------------------------------------------------ сеть
    def _research(self, topic: str, *, question: str | None = None) -> Reply:
        question = question or topic
        results = self.web.search(topic)
        if not results:
            return Reply("Поиск не дал результатов.", kind="error")
        first = results[0]
        if not first.ok:
            return Reply(
                f"Не смог получить информацию: {first.error}. "
                "Отвечать по памяти не буду — в базе этого нет.",
                kind="error", meta={"tool_error": first.error})
        page = first
        body = first.text or ""
        if len(body) < 400 and first.source:
            fetched = self.web.fetch(first.source)
            if fetched.ok and fetched.text:
                page = fetched
                body = f"{body}\n{fetched.text}"
        if not body.strip():
            return Reply("Источник открылся, но текста в нём нет.",
                         kind="error", source=first.source)

        selected = rank(sentences(body), question)
        if not selected:
            return Reply(
                f"Нашёл источник ({first.title or first.source}), но ответа на "
                f"вопрос «{topic}» в нём нет. Не буду выдумывать.",
                kind="error", source=first.source)

        answer_text = paraphrase(selected, question)
        source = page.source or first.source
        self.last_source = source
        self.topic = topic

        proposed = self._propose_from(selected, topic, source)
        self._pending_definition = {"word": topic, "definition": answer_text,
                                    "source": source}
        reply_text = f"{answer_text}\n\nИсточник: {source}"
        if proposed:
            lines = "\n".join(f"• {item['subject']} — {item['object']}"
                              for item in proposed)
            reply_text += ("\n\nМогу запомнить это как факты:\n" + lines +
                           "\nОтветь «да», и сохраню (или «нет»).")
            return Reply(reply_text, kind="offer", source=source,
                         proposed=proposed)
        return Reply(reply_text, kind="answer", source=source)

    def _propose_from(self, selected, topic: str, source: str) -> list[dict]:
        """Кандидаты в память: только определения, где субъект — тема запроса."""
        topic_terms = set(tokens(topic))
        proposed = []
        for item in selected:
            parsed = parse_definition(item.sentence)
            if not parsed:
                continue
            subject, obj = parsed
            subject_terms = set(tokens(subject))
            if not (subject_terms & topic_terms):
                continue
            if self.memory.define is None:      # pragma: no cover - защита
                continue
            entry = self.memory.propose(subject, obj, source=source,
                                        sentence=item.sentence)
            if entry:
                proposed.append({"subject": subject, "object": obj,
                                 "source": source})
        return proposed

    # ---------------------------------------------------------- подтверждение
    def _confirm(self) -> Reply:
        written, rejected = self.memory.confirm(self._teach_fact)
        if self._pending_definition and written:
            self.memory.define(self._pending_definition["word"],
                               self._pending_definition["definition"],
                               source=self._pending_definition["source"],
                               confidence=0.55, kind="topic")
        self._pending_definition = None
        if not written:
            return Reply("Не получилось записать — формулировка не подошла "
                         "к моей модели фактов. Ничего не сохранил.",
                         kind="error")
        lines = "\n".join(
            f"• {item['subject']} — {item['object']}"
            + (f" (полная формулировка «{item['full_phrase']}» — в словаре)"
               if item.get("full_phrase") else "")
            for item in written)
        note = "" if not rejected else f"\nНе записал: {len(rejected)}."
        return Reply(f"Записал в память:\n{lines}{note}\n"
                     "Теперь это есть в базе и в словарном запасе.",
                     kind="learn", learned=written)

    def _teach_fact(self, text: str) -> bool:
        answer = self.pipe.respond(text, self.mode)
        return bool(answer.status == A_DIRECT and
                    (answer.text or "").startswith("Записал"))

    # -------------------------------------------------------------- словарь
    def vocabulary(self) -> Reply:
        words = self.memory.words()
        if not words:
            return Reply("Словарный запас пока пуст: научи факту или подтверди "
                         "найденное в сети.", kind="answer")
        lines = [f"• {item['word']} — {item['definition']} "
                 f"[{item['source'] or 'без источника'}]" for item in words]
        return Reply("Словарный запас:\n" + "\n".join(lines), kind="answer")


def unknown_status(status: str) -> bool:
    return (status or "").upper() in {"UNKNOWN", "AMBIGUOUS", "NOT_FOUND"}
