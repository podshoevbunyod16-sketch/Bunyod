"""MeaningIR — «что человек имел в виду», отдельно от «какие это слова».

``QueryIR`` (v0.16) — контракт синтаксиса: интент, wh-слово, упоминания,
предикат, отрицание, время, модальность.  Он не отвечает на вопросы:

    * какой речевой акт перед нами на самом деле (и не ошибся ли парсер);
    * какой слот спрашивают и какого типа ответ ожидается;
    * какие пресуппозиции несёт вопрос («сущность существует», «отношение
      известно», «кандидат единственный»);
    * что осталось *неизвестным* — и должно быть озвучено, а не додумано.

``MeaningIR`` отвечает ровно на них и ничего не утверждает о мире.  Он не
обращается к графу за фактами: только к резолверу сущностей и к схеме
отношений.  Поэтому его можно тестировать по одному предложению, без базы.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from rasm_cim_v16.entity_resolver import RES_ALIAS, RES_AMBIGUOUS, RES_EXACT, \
    RES_NORMALIZED
from rasm_cim_v16.language_parser import (INT_ASK_DEFINITION, INT_ASK_WH,
                                          INT_ASK_YESNO, INT_COMMAND,
                                          INT_GREETING, INT_RETRACT, INT_TEACH,
                                          INT_UNKNOWN, WH_WANTS, QueryIR)
from rasm_cim_v16.relations import KIND_CLASS, KIND_PERSON, KIND_PLACE

from .refine import refine_speech_act

# -- speech acts -----------------------------------------------------------
AS_ASK_DEFINITION = "ASK_DEFINITION"
AS_ASK_WH = "ASK_WH"
AS_ASK_YESNO = "ASK_YESNO"
AS_ASK_LIST = "ASK_LIST"
AS_TELL = "TELL"
AS_RETRACT = "RETRACT"
AS_GREETING = "GREETING"
AS_COMMAND = "COMMAND"
AS_UNKNOWN = "UNKNOWN"

SPEECH_ACTS = (AS_ASK_DEFINITION, AS_ASK_WH, AS_ASK_YESNO, AS_ASK_LIST,
               AS_TELL, AS_RETRACT, AS_GREETING, AS_COMMAND, AS_UNKNOWN)

# -- focus statuses --------------------------------------------------------
FOCUS_RESOLVED = "RESOLVED"
FOCUS_AMBIGUOUS = "AMBIGUOUS"
FOCUS_UNKNOWN = "UNKNOWN"
FOCUS_PRONOUN = "PRONOUN"          # needs the dialogue context to resolve
FOCUS_SURFACE = "SURFACE"          # teaching: the surface is not required to exist

#: expected answer type by wh-word — from the schema's own question words
WH_ANSWER_TYPE = {
    "кто": "person", "кем": "person", "кого": "person",
    "что": "entity", "чего": "entity", "чем": "entity",
    "какой": "class", "какая": "class", "какое": "class", "какие": "class",
    "какую": "class", "какого": "class",
    "когда": "time", "где": "place", "куда": "place", "откуда": "place",
    "почему": "cause", "зачем": "cause", "сколько": "quantity",
}

#: which slot each wh-word asks for
WH_SLOT = {
    "кто": "subject", "кем": "subject", "кого": "subject",
    "что": "object", "чего": "object", "чем": "object",
    "когда": "object", "где": "object", "куда": "object",
    "откуда": "object", "почему": "object", "зачем": "object",
    "сколько": "object",
}

MAX_FOCUS_CANDIDATES = 3

#: the frozen parser calls teaching ``TEACH``; MeaningIR calls the act ``TELL``.
#: One normalisation table, applied to both sides of the comparison, so that a
#: rename can never masquerade as a "repair".
PARSER_ACT_ALIASES = {INT_TEACH: AS_TELL, "TEACH": AS_TELL}


def normalise_act(intent: str) -> str:
    return PARSER_ACT_ALIASES.get(intent, intent)


@dataclass
class Focus:
    """One mention the utterance is about, with its resolution state."""

    surface: str
    role: str = "subject"          # subject | object | topic
    entity_id: int | None = None
    status: str = FOCUS_UNKNOWN
    confidence: float = 0.0
    candidates: list[int] = field(default_factory=list)
    note: str = ""

    def as_dict(self) -> dict:
        return {"surface": self.surface, "role": self.role,
                "entity_id": self.entity_id, "status": self.status,
                "confidence": round(self.confidence, 3),
                "candidates": list(self.candidates),
                "note": self.note}

    @property
    def resolved(self) -> bool:
        return self.entity_id is not None


@dataclass
class Presupposition:
    """What the utterance takes for granted, and whether that holds."""

    kind: str                 # entity_exists | relation_known | unique_candidate
    text: str
    holds: bool | None = None  # None: cannot be decided before execution

    def as_dict(self) -> dict:
        return {"kind": self.kind, "text": self.text, "holds": self.holds}


#: complementizers the frozen parser leaves glued to a mention ("что Кит")
MENTION_PREFIXES = ("что ", "чтобы ", "как ", "будто ", "словно ")


def clean_mention(mention: str) -> tuple[str, str]:
    """Strip a leading complementizer from a mention.

    ``"Забудь, что Кит — это млекопитающее"`` arrives with subject mention
    ``"что Кит"``, which the resolver cannot match, so the retraction fails
    with "unknown entity" while looking like it worked.  Removing the
    complementizer is a repair of the *surface*, not of the meaning, and it is
    recorded as such.
    """
    text = (mention or "").strip().strip(",;:.!?")
    low = text.casefold()
    for prefix in MENTION_PREFIXES:
        if low.startswith(prefix):
            cleaned = text[len(prefix):].strip()
            if cleaned:
                return cleaned, f"mention cleaned: {mention!r} -> {cleaned!r}"
    return text, ""


@dataclass
class MeaningIR:
    text: str
    norm: str
    speech_act: str = AS_UNKNOWN
    asks: str = ""                 # subject | object | definition | related |
                                   # confirmation | listing | ""
    expected_answer_type: str = ""  # person | place | class | entity | boolean |
                                    # list | time | quantity | cause | ""
    relation: str = ""
    subject_mention: str = ""      # cleaned surface, as executed
    object_mention: str = ""
    negation: bool = False
    focus: list[Focus] = field(default_factory=list)
    presuppositions: list[Presupposition] = field(default_factory=list)
    unknowns: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    confidence: float = 0.0
    query: QueryIR | None = None
    parser_intent: str = ""

    # -- views -------------------------------------------------------------
    @property
    def is_question(self) -> bool:
        return self.speech_act in (AS_ASK_WH, AS_ASK_YESNO, AS_ASK_DEFINITION,
                                   AS_ASK_LIST)

    @property
    def resolved_focus(self) -> list[Focus]:
        return [f for f in self.focus if f.resolved]

    @property
    def unresolved_focus(self) -> list[Focus]:
        return [f for f in self.focus if not f.resolved]

    @property
    def repaired(self) -> bool:
        """True when this meaning differs from what the parser alone said.

        The pipeline uses this to decide whether the v0.16 executor can be
        trusted with the utterance as-is: an unrepaired meaning delegates,
        a repaired one is executed here, so the decision is never silently
        dropped between the two layers.
        """
        return (self.speech_act != self.parser_intent
                or any(f.note for f in self.focus)
                or any(n.startswith("mention cleaned") for n in self.notes))

    @property
    def interpretable(self) -> bool:
        """Enough meaning to act on: a known act and (if it names something)
        at least one resolved focus, or an act that needs none."""
        if self.speech_act in (AS_GREETING, AS_COMMAND, AS_UNKNOWN):
            return self.speech_act != AS_UNKNOWN
        if self.speech_act in (AS_ASK_WH, AS_ASK_DEFINITION, AS_ASK_YESNO):
            if not self.focus:
                return True       # e.g. "Сколько планет?" — no named entity
            return bool(self.resolved_focus)
        return True               # TEACH/RETRACT/ASK_LIST act on surfaces

    def as_dict(self) -> dict:
        return {"text": self.text, "speech_act": self.speech_act,
                "asks": self.asks,
                "expected_answer_type": self.expected_answer_type,
                "relation": self.relation, "negation": self.negation,
                "focus": [f.as_dict() for f in self.focus],
                "presuppositions": [p.as_dict() for p in self.presuppositions],
                "unknowns": list(self.unknowns), "notes": list(self.notes),
                "confidence": round(self.confidence, 3),
                "parser_intent": self.parser_intent,
                "interpretable": self.interpretable}

    def describe(self) -> str:
        bits = [self.speech_act]
        if self.asks:
            bits.append(f"asks={self.asks}")
        if self.expected_answer_type:
            bits.append(f"expects={self.expected_answer_type}")
        if self.relation:
            bits.append(f"rel={self.relation}")
        for f in self.focus:
            bits.append(f"{f.role}:{f.surface}->{f.status}")
        if self.unknowns:
            bits.append("unknown=" + "; ".join(self.unknowns))
        return " ".join(bits)


# --------------------------------------------------------------------------
# building
# --------------------------------------------------------------------------

def _focus_for(mention: str, role: str, resolver, teaching: bool) -> Focus:
    if not mention:
        return Focus(surface="", role=role, status=FOCUS_UNKNOWN)
    res = resolver.resolve(mention)
    if res.status == RES_AMBIGUOUS or len(res.candidates) > 1:
        strong = [c for c in res.candidates
                  if c.confidence >= (res.best().confidence - 0.05)] \
            if res.best() else []
        if len(strong) > 1:
            return Focus(surface=mention, role=role, status=FOCUS_AMBIGUOUS,
                         confidence=0.0,
                         candidates=[c.entity_id for c in
                                     strong[:MAX_FOCUS_CANDIDATES]])
    if res.entity_id is not None:
        status = FOCUS_RESOLVED if res.status in (RES_EXACT, RES_ALIAS,
                                                  RES_NORMALIZED) else \
            FOCUS_RESOLVED
        return Focus(surface=mention, role=role, entity_id=res.entity_id,
                     status=status, confidence=res.confidence,
                     candidates=[c.entity_id for c in
                                 res.candidates[:MAX_FOCUS_CANDIDATES]])
    if teaching:
        return Focus(surface=mention, role=role, status=FOCUS_SURFACE,
                     confidence=0.0)
    # The whole phrase is not an entity ("спутников у Марса").  A question
    # word plus a case ending can swallow a real name; try the longest
    # *suffix* of the phrase before giving up.  Reported as a repair, with a
    # confidence penalty, because a span is weaker evidence than a whole
    # mention.
    recovered = _recover_from_span(mention, resolver)
    if recovered is not None:
        eid, surface, conf = recovered
        return Focus(surface=mention, role=role, entity_id=eid,
                     status=FOCUS_RESOLVED, confidence=round(conf * 0.9, 3),
                     note=f"span-recovered from {surface!r}")
    return Focus(surface=mention, role=role, status=FOCUS_UNKNOWN,
                 candidates=[c.entity_id for c in
                             res.candidates[:MAX_FOCUS_CANDIDATES]])


#: relational nouns: in "отец Исаака Ньютона" the suffix names somebody ELSE.
#: Recovering it would answer a question about the father with facts about the
#: son — false precision dressed as recall, so these prefixes veto recovery.
RELATIONAL_PREFIXES = frozenset({
    "отец", "мать", "сын", "дочь", "жена", "муж", "брат", "сестра", "внук",
    "дед", "бабушка", "родители", "ребёнок", "ребенок", "дети",
    "столица", "население", "президент", "директор", "глава", "автор",
    "создатель", "основатель", "учитель", "ученик",
})


def _recover_from_span(mention: str, resolver):
    """Longest suffix of a multi-token mention that resolves, or ``None``.

    Only the *leading* words are dropped, so the recovered span is always the
    tail of the phrase — and if any dropped word is a relational noun, no
    recovery happens at all.
    """
    words = [w for w in mention.split() if w]
    if len(words) < 2:
        return None
    for i in range(1, len(words)):
        dropped = {w.casefold().strip(",.;:!?") for w in words[:i]}
        if dropped & RELATIONAL_PREFIXES:
            return None
        candidate = " ".join(words[i:])
        res = resolver.resolve(candidate)
        if res.entity_id is not None and res.confidence >= 0.8:
            return res.entity_id, candidate, res.confidence
    return None


def build_meaning(ir: QueryIR, resolver=None, schema=None,
                  teaching: bool | None = None) -> MeaningIR:
    """``QueryIR -> MeaningIR``.  Pure: reads the resolver, never writes."""
    if resolver is None:
        from rasm_cim_v16.entity_resolver import EntityResolver
        resolver = EntityResolver()
    if teaching is None:
        teaching = ir.intent in (INT_TEACH, INT_RETRACT)

    act, refine_notes = refine_speech_act(normalise_act(ir.intent),
                                          ir.raw or ir.norm,
                                          explicit_teach=ir.explicit_teach)
    m = MeaningIR(text=ir.raw or ir.norm, norm=ir.norm, speech_act=act,
                  relation=ir.predicate, negation=ir.negation,
                  parser_intent=normalise_act(ir.intent), query=ir,
                  notes=list(ir.notes))
    m.notes.extend(refine_notes)

    # -- what is being asked ----------------------------------------------
    if act == AS_ASK_DEFINITION:
        m.asks = "related" if ir.wants == "related" else "definition"
        m.expected_answer_type = "class" if ir.wants != "related" else "list"
    elif act == AS_ASK_WH and not ir.predicate and ir.wh in ("кем", "кто", "что"):
        # "Кем был Ньютон?" asks what the entity IS.  The parser sets
        # wants=subject with no relation, which the executor can only answer
        # with "no related_to facts".  Treating it as a definition request is
        # a repair, so it is recorded as one.
        m.speech_act = AS_ASK_DEFINITION
        m.asks = "definition"
        m.expected_answer_type = "class"
        m.notes.append(f"refine: {ir.wh!r} without a relation -> "
                       f"ASK_DEFINITION")
    elif act == AS_ASK_WH:
        m.asks = WH_SLOT.get(ir.wh, ir.wants if ir.wants != "none" else "object")
        m.expected_answer_type = WH_ANSWER_TYPE.get(ir.wh, "entity")
        if ir.wants == "subject":
            m.asks = "subject"
        elif ir.wants == "related":
            m.asks = "related"
            m.expected_answer_type = "list"
    elif act == AS_ASK_YESNO:
        m.asks = "confirmation"
        m.expected_answer_type = "boolean"
    elif act == AS_ASK_LIST:
        m.asks = "listing"
        m.expected_answer_type = "list"
    elif act == AS_TELL:
        m.asks = ""
        m.expected_answer_type = ""
    elif act == AS_UNKNOWN:
        m.unknowns.append("речевой акт не распознан")
        m.notes.append("parser returned UNKNOWN and no refinement rule applied")

    # -- mentions (cleaned) -------------------------------------------------
    m.subject_mention, subj_note = clean_mention(ir.subject_mention)
    m.object_mention, obj_note = clean_mention(ir.object_mention)
    for note in (subj_note, obj_note):
        if note:
            m.notes.append(note)

    # -- focus -------------------------------------------------------------
    if m.subject_mention:
        m.focus.append(_focus_for(m.subject_mention, "subject", resolver,
                                  teaching))
    if ir.subject_pronoun:
        m.focus.append(Focus(surface=ir.subject_pronoun, role="subject",
                             status=FOCUS_PRONOUN))
        m.unknowns.append(f"местоимение {ir.subject_pronoun!r} без контекста "
                          f"не разрешается")
    if m.object_mention:
        m.focus.append(_focus_for(m.object_mention, "object", resolver,
                                  teaching))
    if ir.object_pronoun:
        m.focus.append(Focus(surface=ir.object_pronoun, role="object",
                             status=FOCUS_PRONOUN))

    repairs: list[str] = []
    for f in m.focus:
        if f.note:
            repairs.append(f"{f.role}: {f.note}")
    m.notes.extend(repairs)

    for f in m.focus:
        if f.status == FOCUS_AMBIGUOUS:
            m.unknowns.append(
                f"{f.surface!r}: несколько кандидатов "
                f"({', '.join('#' + str(c) for c in f.candidates)}), "
                f"выбор без уточнения запрещён")
        elif f.status == FOCUS_UNKNOWN and not teaching:
            m.unknowns.append(f"{f.surface!r}: сущность неизвестна")

    # -- presuppositions ---------------------------------------------------
    for f in m.focus:
        if f.status == FOCUS_PRONOUN:
            continue
        m.presuppositions.append(Presupposition(
            kind="entity_exists",
            text=f"{f.surface!r} существует в базе",
            holds=True if f.resolved else None))
    if ir.predicate:
        known = schema is not None and ir.predicate in schema
        m.presuppositions.append(Presupposition(
            kind="relation_known", text=f"отношение {ir.predicate!r} известно",
            holds=bool(known)))
    elif act in (AS_ASK_WH, AS_ASK_YESNO):
        m.presuppositions.append(Presupposition(
            kind="relation_recognised",
            text="глагол/предлог вопроса распознан как отношение",
            holds=False))
        m.unknowns.append("отношение не распознано: вопрос без известного "
                          "предиката")
    if act == AS_ASK_YESNO:
        m.presuppositions.append(Presupposition(
            kind="has_evidence",
            text="в базе есть данные об этом утверждении", holds=None))
    if act in (AS_TELL, AS_RETRACT) and m.focus:
        m.presuppositions.append(Presupposition(
            kind="speaker_authority",
            text="говорящий сообщает факт, а не спрашивает", holds=True))

    # -- confidence: the parser's own, capped by unresolved meaning --------
    conf = float(ir.confidence or 0.0)
    if m.unknowns:
        conf = min(conf, 0.5)
    if act == AS_UNKNOWN:
        conf = min(conf, 0.2)
    if refine_notes:
        conf = min(conf, 0.9)      # a refinement is a repair, not a certainty
    m.confidence = conf
    return m


def meanings_for(text: str, resolver=None, schema=None, parser=None) -> list[MeaningIR]:
    """Parse ``text`` and build one ``MeaningIR`` per clause.

    Convenience used by the CLI and the metrics harness.  The parser is
    v0.16's — this function adds nothing to the syntax layer.
    """
    if parser is None:
        from rasm_cim_v16.language_parser import LanguageParser
        parser = LanguageParser(schema) if schema is not None \
            else LanguageParser()
    if resolver is None:
        resolver = getattr(parser, "resolver", None)
        if resolver is None:
            from rasm_cim_v16.entity_resolver import EntityResolver
            resolver = EntityResolver()
    return [build_meaning(ir, resolver=resolver, schema=schema)
            for ir in parser.parse(text)]
