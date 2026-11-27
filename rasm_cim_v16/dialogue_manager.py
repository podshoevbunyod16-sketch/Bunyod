"""v0.16 — dialogue manager: the pipeline from user input to answer.

    say(text)
      ├─ LanguageParser.parse            text -> [QueryIR]
      ├─ ContextEngine                   pronouns, ellipsis, topics
      ├─ RelationExtractor               bind / teach (never both blindly)
      ├─ InferenceEngine                 bounded derivation with proofs
      ├─ ContradictionDetector           conflicts recorded, not deleted
      ├─ EvidenceManager                 provenance per statement
      └─ AnswerEngine                    DIRECT/INFERRED/CONTEXTUAL/
                                         AMBIGUOUS/UNKNOWN/CONTRADICTED

Nothing here invents content: the manager only routes, and every Answer it
returns is built from graph statements by the AnswerEngine.

Commands (``:name arg``) — additive, older CLI verbs still work:
    :inspect :evidence :context :memory :entities :graph :explain
    :conflicts :rules :merge :reset :save :stats :help
"""
from __future__ import annotations

from typing import Sequence

from .answer_engine import (A_AMBIGUOUS, A_CONTRADICTED, A_CONTEXTUAL,
                            A_DIRECT, A_INFERRED,
                            A_UNKNOWN, MODE_ALIASES, MODE_NORMAL, Answer,
                            AnswerEngine)
from .context_engine import (PR_AMBIGUOUS, PR_RESOLVED, ContextEngine,
                             PR_UNRESOLVED)
from .contradiction_detector import ContradictionDetector
from .evidence_manager import EvidenceManager, EV_USER, format_evidence
from .inference_engine import InferenceEngine
from .knowledge_graph import KnowledgeGraph, render_triple
from .language_parser import (INT_ASK_DEFINITION, INT_ASK_WH, INT_ASK_YESNO,
                              INT_COMMAND, INT_GREETING, INT_RETRACT, INT_TEACH,
                              INT_UNKNOWN, LanguageParser, QueryIR)
from .memory_manager import (TIER_EPISODIC, TIER_PROCEDURAL, TIER_SEMANTIC,
                             TIER_WORKING, MemoryManager)
from .relation_extractor import AskResult, RelationExtractor, TeachResult
from .relations import DEFAULT_SCHEMA, RelationSchema


class DialogueManager:
    """The whole system, wired once."""

    def __init__(self, clock=None, schema: RelationSchema | None = None,
                 max_depth: int = 3, max_nodes: int = 400,
                 autosave: bool = False, store=None):
        self._clock = clock or (lambda: "")
        self.schema = schema or RelationSchema()
        self.kg = KnowledgeGraph(schema=self.schema, clock=self._clock)
        self.parser = LanguageParser(self.schema)
        self.extractor = RelationExtractor(self.kg, clock=self._clock)
        self.inference = InferenceEngine(self.kg, max_depth=max_depth,
                                         max_nodes=max_nodes, schema=self.schema)
        self.context = ContextEngine(resolver=self.kg.resolver,
                                     clock=self._clock)
        self.evidence = EvidenceManager(clock=self._clock)
        self.detector = ContradictionDetector(self.kg, clock=self._clock)
        self.memory = MemoryManager(self.kg, self.context, self.evidence,
                                    self.detector, inference=self.inference,
                                    parser=self.parser, clock=self._clock)
        self.answers = AnswerEngine(self.kg, self.inference, self.detector,
                                    self.evidence, clock=self._clock)
        self.autosave = autosave
        self.store = store                 # persistence handle, set by load()
        self.last_answer: Answer | None = None
        self.last_ir: QueryIR | None = None
        self.default_mode = MODE_NORMAL
        self._turn_budget = 0

    # ------------------------------------------------------------------
    # main entry point
    # ------------------------------------------------------------------
    def say(self, text: str, mode: str | None = None) -> Answer:
        mode = MODE_ALIASES.get(mode or self.default_mode,
                                mode or self.default_mode)
        irs = self.parser.parse(text)
        user_turn = self.context.record_user_turn(text, intent=(
            irs[0].intent if irs else INT_UNKNOWN), when=self._clock())
        out: list[Answer] = []
        for ir in irs:
            a = self._handle(ir, mode, turn_id=user_turn.turn_id)
            out.append(a)
        ans = out[0] if len(out) == 1 else self._merge_answers(out, mode)
        self.last_answer, self.last_ir = ans, (irs[0] if irs else None)
        self.context.record_system_turn(ans.text, intent=ans.status,
                                        entity_ids=[f.subject for f in ans.facts][:8],
                                        statement_ids=[f.statement_id for f in
                                                       ans.facts if f.statement_id][:8],
                                        answer_status=ans.status,
                                        when=self._clock())
        self.memory.record(kind="answer", tier=TIER_WORKING,
                           payload={"status": ans.status,
                                    "confidence": ans.confidence,
                                    "text": ans.text[:400]},
                           source="dialogue", confidence=ans.confidence,
                           when=self._clock())
        if self.autosave and self.store is not None:
            self.save()
        return ans

    def _handle(self, ir: QueryIR, mode: str, turn_id: str = "") -> Answer:
        if ir.intent == INT_COMMAND:
            return self.command(ir.command, ir.command_arg, mode)
        if ir.intent == INT_GREETING:
            return self._simple("Привет. Спроси что-нибудь или научи меня факту "
                                "(`teach X — это Y`). Команды: :help.",
                                A_DIRECT, mode)
        if ir.intent in (INT_TEACH, INT_RETRACT):
            return self._do_teach(ir, mode, turn_id)
        if ir.intent == INT_ASK_DEFINITION:
            return self._do_definition(ir, mode)
        if ir.intent == INT_ASK_WH:
            return self._do_wh(ir, mode)
        if ir.intent == INT_ASK_YESNO:
            return self._do_yesno(ir, mode)
        return Answer(text="Не понял. Переформулируй, пожалуйста: например "
                           "«teach Исаак Ньютон — это физик» или «Кто такой "
                           "Исаак Ньютон?».",
                      status=A_UNKNOWN, mode=mode,
                      short="Не понял.", normal="Не понял.",
                      detailed="Не понял.\nРазбор: " + ir.describe() +
                               "\nЗаметки: " + "; ".join(ir.notes),
                      notes=list(ir.notes))

    # ------------------------------------------------------------------
    # teach
    # ------------------------------------------------------------------
    def _do_teach(self, ir: QueryIR, mode: str, turn_id: str) -> Answer:
        res = self.extractor.teach(ir, when=self._clock(), source=f"turn:{turn_id}")
        if res.ok and res.statement is not None:
            self.evidence.add(res.statement.statement_id, kind=EV_USER,
                              utterance=ir.raw, turn_id=turn_id,
                              source="user", when=self._clock())
            # focused scan: only the new triple can create a NEW conflict, so
            # the whole graph does not have to be re-walked on every fact
            new = self.detector.scan(
                when=self._clock(),
                focus=(res.statement.subject, res.statement.predicate,
                       res.statement.object))
            if new:
                res.notes.append(f"обнаружено противоречий: {len(new)}")
            self._maybe_switch_topic(
                [res.subject_id, res.object_id],
                roles={res.subject_id: "subject", res.object_id: "object"})
            self.context.set_focus(subject_id=res.subject_id,
                                   predicate=res.predicate,
                                   object_id=res.object_id,
                                   statement_id=res.statement.statement_id,
                                   when=self._clock())
            if res.predicate in ("discovered", "formulated", "created", "wrote"):
                self.context.set_last_event(res.statement.statement_id,
                                            res.subject_id, res.predicate,
                                            res.object_id, when=self._clock())
            self.memory.record(kind=f"fact:{res.predicate}", tier=TIER_WORKING,
                               payload={"statement": res.statement.statement_id,
                                        "s": res.subject_id, "o": res.object_id},
                               source="teach", confidence=res.confidence,
                               links=[res.statement.statement_id],
                               when=self._clock())
        ans = self.answers.answer_teach(res, mode)
        if res.clarification:
            self.context.set_pending(res.clarification, "relation",
                                     {"subject_id": res.subject_id,
                                      "object_id": res.object_id},
                                     when=self._clock())
        return ans

    # ------------------------------------------------------------------
    # questions
    # ------------------------------------------------------------------
    def _do_definition(self, ir: QueryIR, mode: str) -> Answer:
        eid, used, clar = self._resolve_side(ir, "subject")
        if clar:
            return clar
        if eid is None:
            res = AskResult(wants="definition", predicate=ir.predicate,
                            reason=f"{ir.subject_mention!r} неизвестен")
            return self.answers._unknown(res, mode, used,
                                         f"«{ir.subject_mention}» нет в базе")
        self._maybe_switch_topic([eid], roles={eid: "subject"})
        self.context.set_focus(subject_id=eid, predicate="instance_of",
                               when=self._clock())
        return self.answers.answer_definition(eid, mode, used_context=used)

    def _ask_clarification(self, bind: AskResult, mode: str) -> Answer:
        """Return the clarifying question AND record it as pending.

        Recording matters: spec §7 requires the dialogue to remember that it
        asked, so the next user turn can answer it and so ``:context`` shows
        the open question after a restart.
        """
        self.context.set_pending(bind.clarification,
                                 "entity" if bind.ambiguous_mentions
                                 else "relation",
                                 {"mentions": list(bind.ambiguous_mentions),
                                  "predicate": bind.predicate},
                                 when=self._clock())
        return self.answers._clarification(bind.clarification, bind, mode)

    def _do_wh(self, ir: QueryIR, mode: str) -> Answer:
        if ir.is_elliptical:
            return self._do_elliptical(ir, mode)
        bind = self._bind(ir)
        if bind.clarification:
            return self._ask_clarification(bind, mode)
        used = any("context" in n for n in bind.notes)
        self._maybe_switch_topic([bind.subject_id, bind.object_id],
                                 roles={bind.subject_id: "subject",
                                        bind.object_id: "object"})
        ans = self.answers.answer_wh(bind, mode, used_context=used)
        self._after_question(ir, bind, ans)
        return ans

    def _do_yesno(self, ir: QueryIR, mode: str) -> Answer:
        bind = self._bind(ir)
        if bind.clarification:
            return self._ask_clarification(bind, mode)
        used = any("context" in n for n in bind.notes)
        self._maybe_switch_topic([bind.subject_id, bind.object_id],
                                 roles={bind.subject_id: "subject",
                                        bind.object_id: "object"})
        ans = self.answers.answer_yesno(bind, mode, used_context=used)
        self._after_question(ir, bind, ans)
        return ans

    def _do_elliptical(self, ir: QueryIR, mode: str) -> Answer:
        """«А когда?» — bind the missing side from the dialogue context."""
        focus = self.context.focus
        ev = self.context.last_event
        subject_id = focus.get("subject_id")
        pred = ir.predicate or focus.get("predicate") or ""
        notes = [f"elliptical question bound from context focus={focus!r}"]
        if subject_id is None:
            bind = AskResult(wants=ir.wants, predicate=pred,
                             reason="нет фокуса в контексте")
            return self.answers._unknown(
                bind, mode, True,
                "не к чему привязать вопрос — предыдущая тема не определена")
        bind = AskResult(ok=True, subject_id=subject_id, predicate=pred,
                         wants=ir.wants, confidence=0.7, notes=notes,
                         reason="эллиптический вопрос разрешён через контекст")
        self._maybe_switch_topic([subject_id], roles={subject_id: "subject"})
        ans = self.answers.answer_wh(bind, mode, used_context=True)
        if ans.status == A_UNKNOWN and ev:
            # try the event's object too: "Ньютон открыл закон" + "А когда?"
            alt = AskResult(ok=True, subject_id=ev.get("object_id"),
                            predicate=pred, wants=ir.wants, confidence=0.6,
                            notes=notes + ["проверен также объект последнего "
                                           "события"],
                            reason="")
            if alt.subject_id is not None:
                ans2 = self.answers.answer_wh(alt, mode, used_context=True)
                if ans2.status != A_UNKNOWN:
                    ans = ans2
        if ans.status == A_UNKNOWN:
            ans.notes.append("событие определено ("
                             f"{self.kg.name_of(subject_id)} — {pred} → …), но "
                             "значения для этого вопроса в базе нет")
        return ans

    # ------------------------------------------------------------------
    # resolution helpers
    # ------------------------------------------------------------------
    def _resolve_side(self, ir: QueryIR, side: str):
        """-> (entity_id, used_context, clarification_answer_or_None)"""
        pron = getattr(ir, f"{side}_pronoun")
        mention = getattr(ir, f"{side}_mention").strip()
        if pron:
            r = self.context.resolve_pronoun(pron)
            if r.status == PR_RESOLVED:
                return r.entity_id, True, None
            bind = AskResult(wants=ir.wants, predicate=ir.predicate,
                             reason=r.reason)
            self.context.set_pending(r.question, "pronoun",
                                     {"pronoun": pron, "side": side,
                                      "candidates": r.candidates},
                                     when=self._clock())
            return None, True, self.answers._clarification(r.question, bind,
                                                          MODE_NORMAL)
        if not mention:
            return None, False, None
        res = self.kg.resolver.resolve(mention)
        if res.status == "AMBIGUOUS":
            from .relation_extractor import _ambiguity_question
            q = _ambiguity_question(mention, res)
            bind = AskResult(wants=ir.wants, predicate=ir.predicate,
                             ambiguous_mentions=[mention], reason=res.status)
            self.context.set_pending(q, "entity",
                                     {"mention": mention,
                                      "candidates": res.candidates and
                                      [c.entity_id for c in res.candidates]},
                                     when=self._clock())
            return None, False, self.answers._clarification(q, bind, MODE_NORMAL)
        if res.status == "UNKNOWN":
            return None, False, None
        return res.entity_id, False, None

    def _bind(self, ir: QueryIR) -> AskResult:
        """bind_question + pronoun resolution from the context."""
        work = QueryIR(**{f: getattr(ir, f) for f in ir.__dataclass_fields__})
        used = []
        for side in ("subject", "object"):
            pron = getattr(work, f"{side}_pronoun")
            if not pron:
                continue
            r = self.context.resolve_pronoun(pron)
            if r.status == PR_RESOLVED:
                setattr(work, f"{side}_pronoun", "")
                setattr(work, f"{side}_mention",
                        self.kg.name_of(r.entity_id))
                used.append(f"{side} pronoun {pron!r} -> "
                            f"{self.kg.name_of(r.entity_id)} via context")
            else:
                bind = AskResult(wants=ir.wants, predicate=ir.predicate,
                                 reason=r.reason,
                                 ambiguous_mentions=[pron])
                bind.clarification = r.question
                self.context.set_pending(r.question, "pronoun",
                                         {"pronoun": pron, "side": side},
                                         when=self._clock())
                return bind
        bind = self.extractor.bind_question(work, when=self._clock())
        bind.notes.extend(used)
        return bind

    def _after_question(self, ir: QueryIR, bind: AskResult, ans: Answer) -> None:
        if bind.subject_id is not None:
            self.context.set_focus(subject_id=bind.subject_id,
                                   predicate=bind.predicate,
                                   object_id=bind.object_id, when=self._clock())
        for f in ans.facts[:1]:
            if f.predicate in ("discovered", "formulated", "created", "wrote"):
                self.context.set_last_event(f.statement_id, f.subject,
                                            f.predicate, f.object,
                                            when=self._clock())
        if ans.clarification:
            self.context.set_pending(ans.clarification, "answer",
                                     {"predicate": bind.predicate},
                                     when=self._clock())

    def _maybe_switch_topic(self, entity_ids: Sequence[int],
                            roles: dict | None = None) -> str:
        """Decide the topic FIRST, then update salience.

        Order matters: ``ContextEngine.touch`` appends the entity to the
        current topic, so touching before classifying would make every new
        entity look like part of the old topic and topic switching would
        never fire (that was the TEST F failure mode).
        """
        ids = [e for e in entity_ids if e is not None]
        if not ids:
            return "same"
        action = "same"
        kind = self.context.classify_topic(ids)
        if kind in ("new", "resume"):
            action = self.context.switch_topic(
                ids, label=self.kg.name_of(ids[0]), when=self._clock())
        for e in ids:
            self.context.touch(e, role=(roles or {}).get(e, "mention"),
                               when=self._clock())
        return action

    # ------------------------------------------------------------------
    # commands
    # ------------------------------------------------------------------
    def command(self, name: str, arg: str = "", mode: str = MODE_NORMAL) -> Answer:
        name = (name or "").lower()
        arg = (arg or "").strip()
        h = getattr(self, f"_cmd_{name}", None)
        if h is None:
            return self._simple(f"Неизвестная команда :{name}. Доступны: "
                                + ", ".join(sorted(
                                    a[5:] for a in dir(self)
                                    if a.startswith("_cmd_"))) + ".",
                                A_UNKNOWN, mode)
        return h(arg, mode)

    def _simple(self, text: str, status: str, mode: str) -> Answer:
        return Answer(text=text, status=status, mode=mode, short=text,
                      normal=text, detailed=text)

    def _cmd_help(self, arg, mode) -> Answer:
        t = """Команды RASM-CIM v0.16:
  :inspect            общее состояние системы
  :entities           сущности: канонические имена, виды, псевдонимы, род
  :graph [сущность]   факты вокруг сущности (без аргумента — все)
  :evidence [id]      доказательства для факта (или все)
  :explain [id]       цепочка вывода последнего ответа / конкретного факта
  :context            темы, активные сущности, фокус, незакрытые уточнения
  :memory             состояние четырёх уровней памяти
  :conflicts          обнаруженные противоречия
  :rules              правила вывода (процедурная память)
  :merge A | B        явно объявить A и B одной сущностью (same_as)
  :reset              очистить рабочую память и контекст (база знаний цела)
  :save               сохранить kb.json и context.json
  :stats              счётчики и размеры
Режимы ответа: --answer concise | normal | detailed."""
        return self._simple(t, A_DIRECT, mode)

    def _cmd_inspect(self, arg, mode) -> Answer:
        st = self.kg.stats()
        lines = [
            f"Сущностей: {st['entities']}   фактов: {st['statements']}   "
            f"отозвано: {st['retracted']}",
            f"Истинных: {st['true']}   отрицаний: {st['false']}   "
            f"предикатов: {st['predicates']}",
            f"Противоречий: {len(self.detector.conflicts())} "
            f"(нерешённых {len(self.detector.unresolved())})",
            f"Реплик в контексте: {len(self.context.history)}   "
            f"активных сущностей: {len(self.context.active)}   "
            f"тем: {1 + len(self.context.topic_stack)}",
            f"Текущая тема: {self.context.topic.label or '(не задана)'}",
            f"Доказательств: {len(self.evidence)}   "
            f"предупреждений о типах: {st['type_warnings']}",
        ]
        problems = self.memory.integrity_check()
        if problems["context"]:
            lines.append("Проблемы целостности контекста: "
                         + "; ".join(problems["context"]))
        if problems["orphan_evidence"]:
            lines.append(f"Висячих доказательств: "
                         f"{len(problems['orphan_evidence'])}")
        return self._simple("\n".join(lines), A_DIRECT, mode)

    def _cmd_entities(self, arg, mode) -> Answer:
        ents = self.kg.resolver.all_entities()
        if arg:
            r = self.kg.resolver.resolve(arg)
            ents = [e for e in ents if e.entity_id == r.entity_id] if r.entity_id else []
        lines = [f"#{e.entity_id} {e.canonical!r} вид={e.kind or '-'} "
                 f"род={e.gender or '-'}({e.gender_source or '-'}) "
                 f"псевдонимы={e.aliases or '-'}" for e in ents]
        return self._simple("\n".join(lines) or "Сущностей нет.",
                            A_DIRECT if lines else A_UNKNOWN, mode)

    def _cmd_graph(self, arg, mode) -> Answer:
        if arg:
            r = self.kg.resolver.resolve(arg)
            if r.entity_id is None:
                return self._simple(f"«{arg}» неизвестен.", A_UNKNOWN, mode)
            sts = self.kg.neighbours(r.entity_id)
        else:
            sts = self.kg.all_statements()
        lines = [f"{s.statement_id} {render_triple(self.kg, s)} "
                 f"[{s.polarity}, conf {s.confidence:.2f}, x{s.reinforce}]"
                 for s in sts]
        return self._simple("\n".join(lines) or "Фактов нет.",
                            A_DIRECT if lines else A_UNKNOWN, mode)

    def _cmd_evidence(self, arg, mode) -> Answer:
        if arg:
            ev = self.evidence.for_statement(arg)
            st = self.kg.statement(arg)
            head = render_triple(self.kg, st) if st else f"(факт {arg} не найден)"
            return self._simple(head + "\n" + "\n".join(format_evidence(ev)),
                                A_DIRECT if ev else A_UNKNOWN, mode)
        lines = []
        for e in self.evidence.all():
            lines.append(f"{e.evidence_id} -> {e.statement_id} [{e.kind}] "
                         f"{e.utterance or e.source}")
        return self._simple("\n".join(lines) or "Доказательств нет.",
                            A_DIRECT if lines else A_UNKNOWN, mode)

    def _cmd_explain(self, arg, mode) -> Answer:
        if arg and arg.startswith("st-"):
            st = self.kg.statement(arg)
            if st is None:
                return self._simple(f"Факт {arg} не найден.", A_UNKNOWN, mode)
            fs = self.inference.query(st.subject, st.predicate, st.object)
            lines = [render_triple(self.kg, st)]
            if fs:
                lines += self.inference.explain(fs[0])
            lines += format_evidence(self.evidence.for_statement(arg))
            cf = self.detector.affecting(arg)
            if cf:
                lines += [f"Противоречие {c.conflict_id}: {c.detail}" for c in cf]
            return self._simple("\n".join(lines), A_DIRECT, mode)
        if self.last_answer is None:
            return self._simple("Пока не было ответа, который можно объяснить.",
                                A_UNKNOWN, mode)
        a = self.last_answer
        lines = [f"Статус: {a.status}   уверенность {a.confidence:.2f}",
                 f"Вопрос: {self.last_ir.raw if self.last_ir else '-'}"]
        if self.last_ir is not None:
            lines.append("Разбор: " + self.last_ir.describe())
            lines.append("Заметки разбора: " + ("; ".join(self.last_ir.notes) or "-"))
        lines.append(f"Почему такой ответ: найдено фактов {len(a.facts)}")
        lines += a.explanation or ["  (вывод не требовался — прямой факт или "
                                   "данных нет)"]
        if a.facts:
            lines.append("Использованные факты:")
            lines += [f"  • {f.statement_id or '-'} "
                      f"{self.kg.name_of(f.subject)} — {f.predicate} → "
                      f"{self.kg.name_of(f.object)} [{f.status}]"
                      for f in a.facts[:10]]
        if a.missing:
            lines.append(f"Чего не хватает: {a.missing}")
        lines.append("Не подтверждено: " + (
            "ничего — всё опирается на записанные факты"
            if a.status in (A_DIRECT, A_INFERRED, A_CONTEXTUAL)
            else "весь ответ: данных недостаточно"))
        return self._simple("\n".join(lines), a.status, mode)

    def _cmd_context(self, arg, mode) -> Answer:
        c = self.context
        lines = [f"Сессия {c.session_id}, реплика #{c.turn_no}",
                 f"Текущая тема: {c.topic.label or '(не задана)'} "
                 f"(сущности: {[self.kg.name_of(e) for e in c.topic.entity_ids]})",
                 f"Стек тем: {[t.label or '(без имени)' for t in c.topic_stack]}"]
        lines.append("Активные сущности (по значимости):")
        for a in c.salience_ranked()[:10]:
            lines.append(f"  • {self.kg.name_of(a.entity_id)} "
                         f"sal={a.salience:.2f} роль={a.role or '-'} "
                         f"род={a.gender or '-'}")
        lines.append(f"Фокус: {c.focus or '-'}")
        lines.append(f"Последнее событие: {c.last_event or '-'}")
        lines.append(f"Незакрытое уточнение: "
                     f"{c.pending_clarification.get('question', '-')}")
        lines.append("Последние реплики:")
        for t in c.history[-6:]:
            lines.append(f"  {t.turn_id} [{t.role}/{t.intent}] {t.text[:80]}")
        problems = c.integrity_check()
        lines.append("Целостность: " + ("OK" if not problems
                                        else "; ".join(problems)))
        return self._simple("\n".join(lines), A_DIRECT, mode)

    def _cmd_memory(self, arg, mode) -> Answer:
        s = self.memory.stats()
        sizes = self.memory.size_bytes()
        lines = [
            "A. Рабочая память (working): "
            f"{s['working_records']}/{s['working_cap']} записей, "
            f"вытеснено {s['working_evictions']}, "
            f"дублей слито {s['duplicate_record_hits']}, "
            f"{sizes['working']} Б",
            "B. Эпизодическая память (episodic): "
            f"{s['episodic_turns']} реплик, "
            f"{s['episodic_active_entities']} активных сущностей, "
            f"{s['episodic_topics']} тем, {sizes['episodic']} Б",
            "C. Семантическая память (semantic): "
            f"{s['semantic_statements']} фактов, "
            f"{s['semantic_entities']} сущностей, "
            f"{sizes['semantic']} Б",
            "D. Процедурная память (procedural): "
            f"{len(self.memory.procedural()['relations'])} отношений, "
            f"{len(self.memory.procedural().get('inference_rules', []))} "
            f"правил вывода, свои отношения: "
            f"{self.memory.procedural()['custom_relations'] or 'нет'}",
            f"Доказательств: {s['evidence_records']} ({sizes['evidence']} Б); "
            f"противоречий: {s['conflicts']} ({sizes['conflicts']} Б)",
        ]
        if arg == "procedural":
            import json as _json
            lines.append(_json.dumps(self.memory.procedural(),
                                     ensure_ascii=False, indent=1)[:4000])
        return self._simple("\n".join(lines), A_DIRECT, mode)

    def _cmd_conflicts(self, arg, mode) -> Answer:
        cs = self.detector.conflicts()
        lines = [f"{c.conflict_id} [{c.kind}] {c.detail}\n"
                 f"   записи: {c.statement_ids}\n"
                 f"   источники: {c.sources or '-'}\n"
                 f"   решение: {c.resolution or 'не решено'}" for c in cs]
        return self._simple("\n".join(lines) or "Противоречий не обнаружено.",
                            A_DIRECT, mode)

    def _cmd_rules(self, arg, mode) -> Answer:
        lines = ["Правила вывода (процедурная память):"]
        for r in self.inference.rule_report():
            lines.append(f"  • {r['name']} [{'вкл' if r['enabled'] else 'выкл'}] "
                         f"conf x{r['confidence']} max_chain={r['max_chain']}: "
                         f"{r['description']}")
        lines.append(f"Границы поиска: max_depth={self.inference.max_depth}, "
                     f"max_nodes={self.inference.max_nodes}")
        lines.append(f"Транзитивные отношения: "
                     f"{sorted(self.schema.transitive_relations())}")
        lines.append(f"Симметричные: {sorted(self.schema.symmetric_relations())}")
        lines.append(f"Функциональные: {sorted(self.schema.functional_relations())}")
        return self._simple("\n".join(lines), A_DIRECT, mode)

    def _cmd_merge(self, arg, mode) -> Answer:
        parts = [p.strip() for p in arg.replace("|", "|").split("|") if p.strip()]
        if len(parts) != 2:
            return self._simple("Использование: :merge A | B", A_UNKNOWN, mode)
        ra, rb = self.kg.resolver.resolve(parts[0]), self.kg.resolver.resolve(parts[1])
        if ra.entity_id is None or rb.entity_id is None:
            return self._simple("Обе сущности должны уже существовать.",
                                A_UNKNOWN, mode)
        if ra.entity_id == rb.entity_id:
            return self._simple("Это уже одна сущность.", A_DIRECT, mode)
        when = self._clock()
        st, created = self.kg.assert_triple(ra.entity_id, "same_as", rb.entity_id,
                                            confidence=1.0, source=":merge",
                                            when=when,
                                            provenance={"utterance": arg})
        self.evidence.add(st.statement_id, kind=EV_USER, utterance=arg,
                          source=":merge", when=when)
        self.kg.resolver.link_alias(ra.entity_id, rb.entity_id, when=when,
                                    source=":merge")
        self.detector.scan(when=when)
        return self._simple(f"Объявлено: {parts[0]} same_as {parts[1]} "
                            f"({st.statement_id}); псевдонимы объединены.",
                            A_DIRECT, mode)

    def _cmd_reset(self, arg, mode) -> Answer:
        n = self.memory.clear_working()
        self.context.reset()
        return self._simple(f"Рабочая память очищена ({n} записей), контекст "
                            f"сброшен. База знаний не тронута "
                            f"({self.kg.stats()['statements']} фактов).",
                            A_DIRECT, mode)

    def _cmd_save(self, arg, mode) -> Answer:
        if self.store is None:
            return self._simple("Хранилище не подключено (нет --home).",
                                A_UNKNOWN, mode)
        info = self.store.save()
        return self._simple("Сохранено: " + ", ".join(
            f"{k}={v}" for k, v in info.items()), A_DIRECT, mode)

    def _cmd_stats(self, arg, mode) -> Answer:
        import json as _json
        d = {"kg": self.kg.stats(), "memory": self.memory.stats(),
             "bytes": self.memory.size_bytes(),
             "inference": self.inference.stats,
             "context": {"turns": len(self.context.history),
                         "active": len(self.context.active),
                         "topics": 1 + len(self.context.topic_stack),
                         "pronoun_log": len(self.context.pronoun_log)}}
        return self._simple(_json.dumps(d, ensure_ascii=False, indent=1),
                            A_DIRECT, mode)

    # ------------------------------------------------------------------
    # persistence wiring
    # ------------------------------------------------------------------
    def attach_store(self, store) -> None:
        self.store = store

    def save(self) -> dict:
        if self.store is None:
            raise RuntimeError("no store attached")
        return self.store.save()

    def _merge_answers(self, out: list[Answer], mode: str) -> Answer:
        text = "\n---\n".join(a.render(mode) for a in out)
        status = out[0].status
        order = [A_CONTRADICTED, A_AMBIGUOUS, A_UNKNOWN, A_CONTEXTUAL,
                 A_INFERRED, A_DIRECT]
        for a in out:
            if order.index(a.status) < order.index(status):
                status = a.status
        return Answer(text=text, status=status,
                      confidence=max(a.confidence for a in out), mode=mode,
                      short="\n".join(a.short for a in out),
                      normal="\n".join(a.normal for a in out),
                      detailed="\n".join(a.detailed for a in out),
                      facts=[f for a in out for f in a.facts],
                      evidence=[e for a in out for e in a.evidence],
                      explanation=[l for a in out for l in a.explanation],
                      clarification=next((a.clarification for a in out
                                          if a.clarification), ""),
                      conflicts=[c for a in out for c in a.conflicts],
                      notes=[n for a in out for n in a.notes])
