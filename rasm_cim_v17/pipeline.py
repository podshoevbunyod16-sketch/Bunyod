"""The semantic pipeline: MeaningIR decides, the v0.16 executor answers.

    text
      └─ LanguageParser (v0.16, frozen)          -> [QueryIR]
          └─ build_meaning (v0.17)               -> MeaningIR
              ├─ не изменено?  -> DialogueManager.say()      (делегирование)
              └─ исправлено?   -> SemanticPipeline._respond_* (исполнение здесь)

Why this split exists.  ``MeaningIR.repaired`` is true exactly when the v0.17
semantic layer disagrees with the syntax layer: the parser called an imperative
a ``TEACH``, or a question word swallowed a real entity name.  In those cases
handing the original text back to v0.16 would execute the *wrong* act — it
would teach a fact nobody asserted.  So the pipeline takes over, and uses the
same ``AnswerEngine``/graph APIs v0.16 exposes; it never invents an answer.

For every unrepaired utterance the behaviour is byte-for-byte v0.16: this
module adds no reinterpretation to what already works, which is what keeps the
348 core tests meaningful.
"""
from __future__ import annotations

from rasm_cim_v16.answer_engine import (A_DIRECT, A_INFERRED, A_UNKNOWN,
                                        MODE_ALIASES, MODE_NORMAL, Answer)
from rasm_cim_v16.evidence_manager import EV_INFERENCE
from rasm_cim_v16.inference_engine import ST_DIRECT, Fact
from rasm_cim_v16.knowledge_graph import POL_TRUE, render_triple
from rasm_cim_v16.relation_extractor import AskResult

from dataclasses import replace as _dc_replace

from rasm_cim_v16.language_parser import INT_RETRACT, INT_TEACH

from .discourse.binding import apply_context, record_turn
from .reasoning.audit import audit_answer
from .reasoning.use import answer_with_inference
from .semantics.meaning import (AS_ASK_DEFINITION, AS_ASK_LIST, AS_ASK_WH,
                                AS_ASK_YESNO, AS_RETRACT, AS_TELL, AS_UNKNOWN,
                                FOCUS_AMBIGUOUS, FOCUS_PRONOUN, MeaningIR,
                                meanings_for)

MAX_LIST_ITEMS = 12


class SemanticPipeline:
    """Thin, explicit router around a v0.16 ``DialogueManager``."""

    def __init__(self, dm, semantic: bool = True, use_inference: bool = True):
        self.dm = dm
        self.semantic = semantic
        #: Plan D — when v0.16 answers "не знаю" but its own engine can prove
        #: the answer (inverse reading, wh orientation), answer with the proof
        self.use_inference = use_inference
        self.last_meaning: MeaningIR | None = None
        self.last_delegated: bool = False
        self.last_audit = None
        #: расширенная память диалога: кто и когда упоминался (не затухает)
        if getattr(dm, "v17_memory", None) is None:
            from .discourse.memory import DialogueMemory
            dm.v17_memory = DialogueMemory()

    # -- public ------------------------------------------------------------
    def respond(self, text: str, mode: str | None = None) -> Answer:
        mode = MODE_ALIASES.get(mode or self.dm.default_mode,
                                mode or self.dm.default_mode)
        if not (text or "").strip():
            # пустой или пробельный ввод: v0.16 падает IndexError, поэтому
            # отказ формируется здесь, до парсера
            self.last_meaning = None
            self.last_delegated = False
            return Answer(text="Не знаю: пустой ввод.", status=A_UNKNOWN,
                          confidence=0.0, mode=mode, facts=[], evidence=[])
        from .semantics.lexicon import normalize
        from .discourse.binding import note_mention

        # 1) память диалога пополняется по исходному тексту: даже если разбор
        #    ниже не поймёт реплику, названные сущности остаются известными
        self._note_mentions(text, note_mention)
        # 2) к разбору идёт каноническая форма (частицы, рамки вопроса,
        #    творительный падеж), исходный текст остаётся для исправлений
        parsed_text = normalize(text)
        from .discourse.correction import (ACK_TEMPLATE, apply_correction,
                                          detect as detect_correction)
        from .discourse.correction import (apply_topic_switch,
                                          detect_topic_switch)
        if detect_topic_switch(text) and not detect_correction(text):
            switched = apply_topic_switch(self.dm, text)
            if switched is not None:
                meaning = self._correction_meaning(text)
                meaning.correction = switched["surface"]
                self.last_meaning = meaning
                self.last_delegated = False
                self.last_audit = None
                answer = Answer(text=ACK_TEMPLATE.format(name=switched["name"]),
                                status=A_DIRECT, confidence=0.9, mode=mode,
                                facts=[], evidence=[])
                record_turn(self.dm, text, meaning, answer)
                return answer
        if detect_correction(text):
            # «Нет, не Иван, а Мария» парсер режет на два смысла, поэтому
            # исправление распознаётся по исходной реплике, а не по каждому
            # фрагменту отдельно
            applied = apply_correction(self.dm, text)
            if applied is not None:
                meanings = meanings_for(parsed_text,
                                        resolver=self.dm.kg.resolver,
                                        schema=self.dm.kg.schema,
                                        parser=self.dm.parser)
                meaning = (meanings[0] if len(meanings) == 1
                           else self._correction_meaning(text))
                self.last_meaning = meaning
                self.last_delegated = False
                self.last_audit = None      # утверждения о мире нет
                answer = Answer(text=ACK_TEMPLATE.format(name=applied["name"]),
                                status=A_DIRECT, confidence=0.9, mode=mode,
                                facts=[], evidence=[])
                record_turn(self.dm, text, meaning, answer)
                return answer
        meanings = meanings_for(parsed_text, resolver=self.dm.kg.resolver,
                                schema=self.dm.kg.schema, parser=self.dm.parser)
        if not self.semantic or len(meanings) != 1:
            self.last_meaning = meanings[0] if meanings else None
            self.last_delegated = True
            return self.dm.say(parsed_text, mode)
        m = meanings[0]
        compound = self._compound_teach(text, mode)
        if compound is not None:
            self.last_meaning = m
            self.last_delegated = False
            return compound
        if any(f.status == FOCUS_PRONOUN for f in m.focus):
            apply_context(m, self.dm)          # Plan C: bind «он/она/оно»
        self.last_meaning = m
        if not m.repaired:
            self.last_delegated = True
            answer = self.dm.say(parsed_text, mode)
            if self.use_inference and answer.status == A_UNKNOWN:
                better = answer_with_inference(self.dm, m, mode)
                if better is not None:
                    self.last_delegated = False
                    self.last_audit = audit_answer(self.dm, better)
                    record_turn(self.dm, text, m, better)
                    return better
            return answer
        self.last_delegated = False
        answer = self._respond_repaired(m, mode)
        self.last_audit = audit_answer(self.dm, answer)
        record_turn(self.dm, text, m, answer)
        return answer

    COMPOUND = None   # заполняется ниже, см. _compound_teach

    def _compound_teach(self, text: str, mode: str):
        """«Иван — физик, Пётр — инженер»: два факта в одной реплике.

        Парсер v0.16 видит здесь одно предложение и не находит ни субъекта, ни
        предиката, поэтому вторая половина терялась.  Правило общее: если
        реплика состоит ровно из двух частей, и каждая — «X — это/— Y», обе
        части исполняются как обычные TEACH-реплики ядра.
        """
        import re

        if "?" in text or "!" in text:
            return None
        parts = [p.strip() for p in re.split(r"[;,]\s*", text) if p.strip()]
        if len(parts) != 2:
            return None
        copula = re.compile(r"^[^—–-]+\s+[—–-]\s+(?:это\s+)?[^—–-]+$")
        if not all(copula.match(part) for part in parts):
            return None
        answers = []
        for part in parts:
            meanings = meanings_for(part, resolver=self.dm.kg.resolver,
                                    schema=self.dm.kg.schema,
                                    parser=self.dm.parser)
            if len(meanings) != 1:
                return None
            piece = meanings[0]
            if piece.speech_act not in (AS_TELL, AS_RETRACT):
                return None
            answers.append(self._respond_repaired(piece, mode))
        last = answers[-1]
        last.notes = list(last.notes) + ["semantic:compound-teach"]
        return last

    def _note_mentions(self, text: str, note) -> None:
        """Отметить в памяти диалога все известные имена из реплики.

        Работает по исходному тексту — до и независимо от разбора акта: имя,
        названное в реплике, которую парсер не понял, всё равно прозвучало, и
        следующее «он» имеет право на него ссылаться.
        """
        import re

        from .semantics.morphology import resolve_form

        for word in re.findall(r"[А-ЯЁ][а-яё\-]{2,}", text or ""):
            entity_id, matched = resolve_form(word, self.dm.kg.resolver)
            if entity_id is not None:
                note(self.dm, entity_id, matched)

    def _correction_meaning(self, text: str) -> MeaningIR:
        """A minimal MeaningIR for a correction the parser split into clauses."""
        from .discourse.correction import detect as detect_correction

        return MeaningIR(text=text, norm=(text or "").strip().casefold(),
                         speech_act="CORRECTION",
                         correction=detect_correction(text),
                         notes=["correction: парсер разбил реплику, смысл "
                                "собран на уровне pipeline"])

    # -- repaired acts -----------------------------------------------------
    def _respond_repaired(self, m: MeaningIR, mode: str) -> Answer:
        resolver = self.dm.kg.resolver
        if m.speech_act == AS_ASK_LIST:
            return self._answer_list(m, mode)
        if m.speech_act == AS_ASK_DEFINITION:
            return self._answer_definition(m, mode)
        if m.speech_act == AS_ASK_WH:
            return self._answer_wh_with_focus(m, mode)
        if m.speech_act == AS_ASK_YESNO:
            return self._answer_yesno_with_focus(m, mode)
        if m.speech_act in (AS_TELL, AS_RETRACT):
            return self._execute_teach(m, mode)
        self.last_delegated = True
        return self.dm.say(m.text, mode)

    # -- TEACH / RETRACT with cleaned mentions ------------------------------
    def _execute_teach(self, m: MeaningIR, mode: str) -> Answer:
        """Run the core teach path on the cleaned surface.

        Only the *mention strings* are replaced; the act, the predicate and
        every downstream step stay v0.16's, so evidence, conflict scanning,
        topic switch and memory records are produced exactly as usual.
        """
        ir = m.query
        if ir is None:
            self.last_delegated = True
            return self.dm.say(m.text, mode)
        fixed = _dc_replace(ir, subject_mention=m.subject_mention,
                            object_mention=m.object_mention)
        answer = self.dm._do_teach(fixed, mode, turn_id="semantic")  # noqa: SLF001
        answer.notes = list(answer.notes) + list(m.notes) + ["semantic:teach"]
        return answer

    # -- ASK_WH with a repaired focus --------------------------------------
    def _answer_wh_with_focus(self, m: MeaningIR, mode: str) -> Answer:
        """Ask the v0.16 engine, but with the focus this layer recovered.

        The recovered span is real evidence (a suffix of the mention resolves
        to a known entity with high confidence), so the question is put to the
        same ``AnswerEngine`` the core uses — no separate answer path, and no
        chance of the two layers disagreeing about what is known.
        """
        subject = next((f for f in m.focus if f.resolved
                        and f.role == "subject"), None)
        obj = next((f for f in m.focus if f.resolved and f.role == "object"),
                   None)
        if subject is None and obj is None:
            return self._unknown(m, mode, "в вопросе нет известной сущности")
        bind = AskResult(
            ok=True,
            subject_id=subject.entity_id if subject else None,
            object_id=obj.entity_id if obj else None,
            predicate=m.relation,
            wants="subject" if m.asks == "subject" else "object",
            polarity="false" if m.negation else "true",
            confidence=max(m.confidence, 0.5),
            notes=list(m.notes))
        answer = self.dm.answers.answer_wh(bind, mode, used_context=False)
        answer.notes = list(answer.notes) + list(m.notes) + \
            ["semantic:ask_wh"]
        return answer

    # -- ASK_YESNO with a repaired or context-bound focus -------------------
    def _answer_yesno_with_focus(self, m: MeaningIR, mode: str) -> Answer:
        subject = next((f for f in m.focus if f.status != FOCUS_PRONOUN
                        and f.entity_id is not None and f.role == "subject"),
                       None)
        obj = next((f for f in m.focus if f.status != FOCUS_PRONOUN
                    and f.entity_id is not None and f.role == "object"), None)
        if subject is None or obj is None:
            # a pronoun we could not bind: ask, never guess
            if m.clarification:
                return Answer(text=m.clarification, status=A_UNKNOWN,
                              confidence=0.3, mode=mode,
                              short=m.clarification, normal=m.clarification,
                              detailed=m.clarification,
                              clarification=m.clarification,
                              notes=list(m.notes) + ["semantic:clarification"])
            return self._unknown(m, mode,
                                 "непонятно, о какой сущности идёт речь")
        predicate = m.relation
        if not predicate:
            # the parser recognised no relation: use the one that actually
            # holds between the two resolved entities, or admit there is none
            between = self.dm.kg.find(subject=subject.entity_id,
                                      obj=obj.entity_id)
            if not between:
                reverse = self.dm.kg.find(subject=obj.entity_id,
                                          obj=subject.entity_id)
                if reverse:
                    st = reverse[0]
                    subject, obj = obj, subject
                    subject_id, obj_id = obj.entity_id, subject.entity_id
                else:
                    return self._unknown(m, mode,
                                         "в базе нет связи между этими "
                                         "сущностями")
            else:
                predicate = between[0].predicate
        bind = AskResult(ok=True, subject_id=subject.entity_id,
                         object_id=obj.entity_id, predicate=predicate,
                         wants="none",
                         polarity="false" if m.negation else "true",
                         confidence=max(m.confidence, 0.5),
                         notes=list(m.notes))
        answer = self.dm.answers.answer_yesno(bind, mode, used_context=False)
        answer.notes = list(answer.notes) + list(m.notes) + \
            ["semantic:ask_yesno"]
        return answer

    # -- ASK_LIST ----------------------------------------------------------
    def _answer_list(self, m: MeaningIR, mode: str) -> Answer:
        kg = self.dm.kg
        target = next((f for f in m.focus if f.resolved
                       and f.role in ("subject", "object")), None)
        if target is None:
            return self._unknown(m, mode, "в вопросе нет известной сущности")
        predicate = m.relation or ""
        statements = [st for st in kg.neighbours(target.entity_id)
                      if st.polarity == POL_TRUE
                      and (not predicate or st.predicate == predicate)]
        if not statements:
            rel_note = f" по отношению {predicate!r}" if predicate else ""
            return self._unknown(
                m, mode, f"в базе нет фактов о «{kg.name_of(target.entity_id)}»"
                         f"{rel_note}")
        facts = self._facts_from(statements)
        lines = [render_triple(kg, st) for st in statements[:MAX_LIST_ITEMS]]
        head = f"{kg.name_of(target.entity_id)}: {len(lines)} факт(ов)"
        more = ("" if len(statements) <= MAX_LIST_ITEMS
                else f" (показаны первые {MAX_LIST_ITEMS})")
        text = head + more + ".\n" + "\n".join(f"  • {l}" for l in lines)
        answer = Answer(text=text, status=self._status(facts),
                        confidence=round(min((f.confidence for f in facts),
                                             default=0.5), 3),
                        mode=mode, facts=facts,
                        evidence=self.dm.answers._ev_for(facts),
                        short=text.splitlines()[0],
                        normal=text, detailed=text,
                        notes=list(m.notes) + ["semantic:ask_list"])
        return answer

    def _answer_definition(self, m: MeaningIR, mode: str) -> Answer:
        kg = self.dm.kg
        target = next((f for f in m.focus if f.resolved), None)
        if target is None:
            return self._unknown(m, mode, "в вопросе нет известной сущности")
        answer = self.dm.answers.answer_definition(target.entity_id, mode,
                                                  used_context=False)
        answer.notes = list(answer.notes) + list(m.notes) + \
            ["semantic:ask_definition"]
        return answer

    # -- helpers -----------------------------------------------------------
    def _facts_from(self, statements) -> list[Fact]:
        facts: list[Fact] = []
        for st in statements:
            facts.append(Fact(subject=st.subject, predicate=st.predicate,
                              object=st.object, status=ST_DIRECT,
                              polarity=POL_TRUE, confidence=st.confidence,
                              depth=0, statement_id=st.statement_id))
        return facts

    def _status(self, facts: list[Fact]) -> str:
        if not facts:
            return A_UNKNOWN
        return A_DIRECT if all(f.status == ST_DIRECT for f in facts) \
            else A_INFERRED

    def _unknown(self, m: MeaningIR, mode: str, why: str) -> Answer:
        text = f"Не знаю. {why[0].upper() + why[1:] if why else why}."
        detail = text
        if m.unknowns:
            detail += "\nНеясно: " + "; ".join(m.unknowns) + "."
        return Answer(text=text, status=A_UNKNOWN, confidence=0.1, mode=mode,
                      short=text, normal=text, detailed=detail,
                      notes=list(m.notes) + ["semantic:unknown"])
