"""Discourse planner — decides what the answer says, in which order.

Inputs: the ``Answer`` from the core, the ``MeaningIR`` that produced it, and
optionally the previous plan (to avoid repeating a sentence the user just read).
Output: ``ResponseIR``.

Decisions the planner makes, each with a reason recorded in ``plan_notes``:

* **refusal** — the core did not know (or refused a pronoun): one refusal
  segment, plus the clarification question when the discourse layer produced
  one.  No claim is ever attached to a refusal.
* **claim** — the first sentence of the core's answer, carrying the ids of the
  facts it rests on.
* **evidence** — only for inferred facts; a direct statement does not need its
  own citation in a conversation (it is still auditable via ``explain``).
* **qualifier** — when the fact is inferred or the confidence is below 0.5, so
  the sentence itself says how sure the system is.
* **repetition** — a claim identical to the previous plan's claim is replaced by
  a short acknowledgement (``already_said``); the facts stay in the plan.
"""
from __future__ import annotations

from rasm_cim_v16.answer_engine import A_AMBIGUOUS, A_UNKNOWN, Answer
from rasm_cim_v16.inference_engine import ST_DIRECT

from ..semantics.meaning import MeaningIR
from .response_ir import (REGISTERS, ResponseIR, Segment, ResponseIRError)

LOW_CONFIDENCE = 0.5


def _sentences(text: str) -> list[str]:
    """Content sentences of a core answer, in order.

    Skips the core's own "Основания:" block and its bullets — the planner
    writes evidence itself, from the facts, not from formatted text.
    """
    out: list[str] = []
    for line in (text or "").splitlines():
        line = line.strip()
        if not line or line.startswith(("•", "-", "Основан", "Что всё же")):
            continue
        for piece in line.split(". "):
            piece = piece.strip().rstrip(".")
            if piece and not piece.endswith(":"):   # заголовок, а не мысль
                out.append(piece)
    return out


def _first_sentence(text: str) -> str:
    parts = _sentences(text)
    return (parts[0] + ".") if parts else ""


def _ids(facts) -> tuple[list[str], list[str]]:
    fact_ids, statement_ids = [], []
    for f in facts:
        fid = f"{f.subject}-{f.predicate}-{f.object}"
        fact_ids.append(fid)
        if f.statement_id:
            statement_ids.append(f.statement_id)
    return fact_ids, statement_ids


def plan_from_answer(answer: Answer, meaning: MeaningIR | None = None,
                     register: str = "normal",
                     previous: ResponseIR | None = None) -> ResponseIR:
    if register not in REGISTERS:
        raise ResponseIRError(f"неизвестный регистр: {register!r}")
    speech_act = getattr(meaning, "speech_act", "") or "UNKNOWN"
    notes: list[str] = []
    referents = [f.surface for f in getattr(meaning, "focus", [])
                 if f.entity_id is not None]

    clarification = getattr(meaning, "clarification", "") or ""
    if answer.status in (A_UNKNOWN, A_AMBIGUOUS):
        notes.append("ядро не знает: отказ без утверждения")
        segments = []
        if clarification:
            segments.append(Segment(kind="clarification", text=clarification))
            notes.append("вопрос-уточнение взят из MeaningIR")
        text = _first_sentence(answer.text) or (answer.text or "").strip()
        if text:
            segments.append(Segment(kind="refusal", text=text))
        if not segments:
            segments.append(Segment(kind="refusal",
                                    text="Не знаю: в базе нет фактов."))
        return ResponseIR(speech_act=speech_act, status=answer.status,
                          segments=segments, register=register,
                          referents=referents, plan_notes=notes)

    facts = list(answer.facts or [])
    fact_ids, statement_ids = _ids(facts)
    sentences = _sentences(answer.text)
    if not sentences:
        raise ResponseIRError("ядро вернуло ответ без текста и без факта")
    sentences = [s if s[-1] in ".!?" else s + "." for s in sentences]

    if previous is not None and previous.claims and \
            previous.claims[0].text == sentences[0]:
        notes.append("already_said: тот же ответ, что и в предыдущей реплике")
        sentences[0] = "Это уже говорилось: " + sentences[0]

    if not facts:
        # приветствие, уточнение, служебная реплика: утверждения о мире нет,
        # и выдавать её за подтверждённый факт нельзя
        notes.append("реплика без фактов (не утверждение о мире)")
        return ResponseIR(speech_act=speech_act, status=answer.status,
                          segments=[Segment(kind="reply", text=t)
                                    for t in sentences],
                          register=register, referents=referents,
                          plan_notes=notes)

    segments = [Segment(kind="claim", text=sentences[0], fact_ids=fact_ids,
                        statement_ids=statement_ids)]
    for extra in sentences[1:]:
        segments.append(Segment(kind="claim", text=extra, fact_ids=fact_ids,
                                statement_ids=statement_ids))
        notes.append("дополнительное утверждение: ядро сообщило больше одной мысли")

    inferred = [f for f in facts if f.status != ST_DIRECT]
    if inferred:
        evidence_ids, evidence_statements = _ids(inferred)
        lines = []
        for f in inferred:
            stmt = f.statement_id or "?"
            lines.append(f"вывод (глубина {f.depth}, уверенность "
                         f"{f.confidence:.2f}, {stmt})")
        segments.append(Segment(kind="evidence", text="; ".join(lines),
                                fact_ids=evidence_ids,
                                statement_ids=evidence_statements))
        notes.append("доказательство включено: факт выведен, а не записан")

    if inferred or (facts and min(f.confidence for f in facts) < LOW_CONFIDENCE):
        conf = min(f.confidence for f in facts) if facts else 0.0
        segments.append(Segment(kind="qualifier",
                                text=f"уверенность {conf:.2f}"
                                     + (", это вывод" if inferred else ""),
                                fact_ids=fact_ids,
                                statement_ids=statement_ids))
        notes.append("квалификатор: неуверенность названа в тексте")

    return ResponseIR(speech_act=speech_act, status=answer.status,
                      segments=segments, register=register,
                      referents=referents, plan_notes=notes)


def plan(pipeline, text: str, register: str = "normal",
         previous: ResponseIR | None = None) -> tuple[Answer, ResponseIR]:
    """Ask the shipped pipeline, then plan its answer."""
    answer = pipeline.respond(text, register if register in REGISTERS else None)
    return answer, plan_from_answer(answer, pipeline.last_meaning, register,
                                    previous)
