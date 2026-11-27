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

from .semantics.meaning import (AS_ASK_DEFINITION, AS_ASK_LIST, AS_ASK_WH,
                                AS_ASK_YESNO, AS_RETRACT, AS_TELL, AS_UNKNOWN,
                                MeaningIR, meanings_for)

MAX_LIST_ITEMS = 12


class SemanticPipeline:
    """Thin, explicit router around a v0.16 ``DialogueManager``."""

    def __init__(self, dm, semantic: bool = True):
        self.dm = dm
        self.semantic = semantic
        self.last_meaning: MeaningIR | None = None
        self.last_delegated: bool = False

    # -- public ------------------------------------------------------------
    def respond(self, text: str, mode: str | None = None) -> Answer:
        mode = MODE_ALIASES.get(mode or self.dm.default_mode,
                                mode or self.dm.default_mode)
        meanings = meanings_for(text, resolver=self.dm.kg.resolver,
                                schema=self.dm.kg.schema, parser=self.dm.parser)
        if not self.semantic or len(meanings) != 1:
            self.last_meaning = meanings[0] if meanings else None
            self.last_delegated = True
            return self.dm.say(text, mode)
        m = meanings[0]
        self.last_meaning = m
        if not m.repaired:
            self.last_delegated = True
            return self.dm.say(text, mode)
        self.last_delegated = False
        return self._respond_repaired(m, mode)

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
            # a question the parser did not recognise as one: the honest move
            # is to say so and ask for a reformulation, never to teach
            return Answer(text="Не понял вопрос. Переформулируй, пожалуйста: "
                               "например «Кто такой Исаак Ньютон?» или "
                               "«Что открыл Исаак Ньютон?».",
                          status=A_UNKNOWN, confidence=0.2, mode=mode,
                          short="Не понял вопрос.",
                          normal="Не понял вопрос.",
                          detailed="Не понял вопрос. " + "; ".join(m.unknowns)
                          if m.unknowns else "Не понял вопрос.",
                          notes=list(m.notes) + ["semantic:unresolved"])
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
