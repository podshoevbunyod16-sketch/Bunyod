"""Plan E tests — ResponseIR, planner, NLG, metrics.

The claims under test are structural, not aesthetic: a plan carries the ids it
came from, a refusal cannot smuggle a claim, a register may only shorten, and
the same plan always renders to the same sentence.
"""
from __future__ import annotations

import json
import pathlib
import subprocess
import sys

import pytest

from rasm_cim_v16.answer_engine import A_AMBIGUOUS, A_DIRECT, A_INFERRED, Answer
from rasm_cim_v16.inference_engine import ST_DIRECT, Fact
from rasm_cim_v17.discourse import load_dialogues
from rasm_cim_v17.generation import (MODES, RESPONSE_SCHEMA, ResponseIR,
                                     ResponseIRError, Segment, plan,
                                     plan_from_answer, render, validate)
from rasm_cim_v17.generation.metrics import evaluate_plans, render_generation
from rasm_cim_v17.pipeline import SemanticPipeline
from rasm_cim_v17.semantics import build_demo, meanings_for
from rasm_cim_v17.semantics.harness import DEFAULT_DIALOGUES

REPO = pathlib.Path(__file__).resolve().parents[1]


@pytest.fixture()
def pipe():
    dm = build_demo()
    return SemanticPipeline(dm)


def _answer(text, status=A_DIRECT):
    return Answer(text=text, status=status, confidence=0.9, mode="normal",
                  facts=[], evidence=[])


def _fact(status=ST_DIRECT, conf=0.9, depth=0, sid="st-000001", proof=None):
    return Fact(subject=1, predicate="located_in", object=2, status=status,
                confidence=conf, depth=depth, statement_id=sid,
                proof=proof or [])


# --------------------------------------------------------------------------
# schema
# --------------------------------------------------------------------------
def test_response_schema_is_versioned():
    assert RESPONSE_SCHEMA == "rasm-cim-response-ir/1"


def test_segment_rejects_unknown_kind():
    with pytest.raises(ResponseIRError):
        Segment(kind="opinion", text="что-то")


def test_segment_rejects_empty_text():
    with pytest.raises(ResponseIRError):
        Segment(kind="claim", text="   ")


def test_plan_requires_segments():
    with pytest.raises(ResponseIRError):
        ResponseIR(speech_act="ASK_WH", status=A_DIRECT, segments=[])


def test_plan_rejects_unknown_register():
    seg = Segment(kind="claim", text="x", fact_ids=["f"])
    with pytest.raises(ResponseIRError):
        ResponseIR(speech_act="ASK_WH", status=A_DIRECT, segments=[seg],
                   register="poetic")


def test_claim_without_source_is_rejected():
    seg = Segment(kind="claim", text="Париж в Франции.")
    with pytest.raises(ResponseIRError):
        ResponseIR(speech_act="ASK_WH", status=A_DIRECT, segments=[seg])


def test_refusal_plan_cannot_carry_a_claim():
    seg = Segment(kind="claim", text="Париж в Франции.", fact_ids=["f"])
    refusal = Segment(kind="refusal", text="Не знаю.")
    with pytest.raises(ResponseIRError):
        ResponseIR(speech_act="ASK_WH", status="UNKNOWN",
                   segments=[seg, refusal])


# --------------------------------------------------------------------------
# planner
# --------------------------------------------------------------------------
def test_unknown_answer_becomes_a_refusal(pipe):
    answer, ir = plan(pipe, "Кто президент Марса?")
    assert ir.status == "UNKNOWN"
    assert ir.of_kind("refusal")
    assert not ir.claims


def test_ambiguous_answer_is_not_turned_into_a_claim():
    meaning = None
    answer = _answer("Не уверен, о ком речь.", A_AMBIGUOUS)
    ir = plan_from_answer(answer, meaning)
    assert ir.status == A_AMBIGUOUS
    assert ir.of_kind("refusal")


def test_direct_answer_has_sourced_claim(pipe):
    answer, ir = plan(pipe, "Кто написал Евгений Онегин?")
    assert answer.status == "DIRECT"
    claim = ir.claims[0]
    assert claim.fact_ids and claim.statement_ids
    assert not ir.of_kind("evidence")     # прямой факт не нуждается в выводе


def test_inferred_fact_brings_evidence_and_qualifier():
    meaning = None
    fact = _fact(status="inferred", conf=0.55, depth=1,
                 proof=[{"rule": "transitive", "statement_id": "st-000009"}])
    answer = Answer(text="Париж находится в Европе.", status=A_INFERRED,
                    confidence=0.55, mode="normal", facts=[fact], evidence=[])
    ir = plan_from_answer(answer, meaning)
    assert ir.of_kind("evidence") and ir.of_kind("qualifier")
    assert any("выведен" in n for n in ir.plan_notes)


def test_low_confidence_direct_fact_gets_a_qualifier():
    fact = _fact(conf=0.3)
    answer = Answer(text="Париж находится в Франция.", status=A_DIRECT,
                    confidence=0.3, mode="normal", facts=[fact], evidence=[])
    ir = plan_from_answer(answer, None)
    qualifier = ir.of_kind("qualifier")[0]
    assert "0.30" in qualifier.text


def test_repeated_answer_is_marked():
    fact = _fact()
    answer = Answer(text="Париж находится в Франция.", status=A_DIRECT,
                    confidence=0.9, mode="normal", facts=[fact], evidence=[])
    first = plan_from_answer(answer, None)
    second = plan_from_answer(answer, None, previous=first)
    assert second.claims[0].text.startswith("Это уже говорилось")
    assert any("already_said" in n for n in second.plan_notes)


def test_greeting_becomes_a_reply_without_facts(pipe):
    answer, ir = plan(pipe, "Привет")
    assert answer.facts == []
    assert ir.of_kind("reply")
    assert not ir.claims


def test_second_sentence_is_kept_as_an_extra_claim():
    fact = _fact()
    answer = Answer(text="Париж находится в Франция; также: Европа.",
                    status=A_DIRECT, confidence=0.9, mode="normal",
                    facts=[fact], evidence=[])
    ir = plan_from_answer(answer, None)
    assert len(ir.claims) == 1        # одна мысль с ";", а не два предложения
    answer2 = Answer(text="Париж находится в Франция. Известные факты:",
                     status=A_DIRECT, confidence=0.9, mode="normal",
                     facts=[fact], evidence=[])
    assert len(plan_from_answer(answer2, None).claims) == 1   # заголовок не мысль


# --------------------------------------------------------------------------
# NLG
# --------------------------------------------------------------------------
def test_registers_only_shorten(pipe):
    _, ir = plan(pipe, "Расскажи про Пушкина")
    short, normal, detailed = (render(ir, m) for m in MODES)
    assert len(short) <= len(normal) <= len(detailed)
    assert short and normal and detailed


def test_render_is_deterministic(pipe):
    _, ir = plan(pipe, "Где находится Париж?")
    assert render(ir) == render(ir)
    assert render(ir, "detailed") == render(ir, "detailed")


def test_refusal_render_has_no_evidence_words(pipe):
    _, ir = plan(pipe, "Кто президент Марса?")
    text = render(ir, "detailed")
    assert "Основание" not in text
    assert text.strip()


def test_unknown_register_raises(pipe):
    _, ir = plan(pipe, "Где находится Париж?")
    with pytest.raises(ResponseIRError):
        render(ir, "verbose")


# --------------------------------------------------------------------------
# serialisation
# --------------------------------------------------------------------------
def test_plan_roundtrips_through_json(pipe):
    _, ir = plan(pipe, "Где находится Париж?")
    again = ResponseIR.from_json(ir.to_json())
    assert again.as_dict() == ir.as_dict()
    validate(again)


def test_bad_json_is_reported_not_ignored():
    with pytest.raises(ResponseIRError):
        ResponseIR.from_json("{not json}")
    with pytest.raises(ResponseIRError):
        ResponseIR.from_dict({"segments": []})


# --------------------------------------------------------------------------
# metrics + CLI
# --------------------------------------------------------------------------
def test_metrics_on_the_dialogue_corpus(pipe):
    corpus = load_dialogues(DEFAULT_DIALOGUES)
    plans, prev = [], None
    for case in corpus.cases:
        for turn in case.turns:
            _, ir = plan(pipe, turn, "normal", prev)
            plans.append(ir)
            prev = ir
    report = evaluate_plans(plans)
    assert report["n"] == len(plans)
    assert report["sourced_claim_rate"] == 1.0
    assert report["deterministic"] is True
    assert report["avg_sentences"] > 0
    assert "Связность" in render_generation(report)


def test_planning_does_not_write_to_the_database(pipe):
    before_statements = len(pipe.dm.kg.all_statements())
    before_evidence = len(pipe.dm.evidence.all())
    plan(pipe, "Где находится Париж?")
    plan(pipe, "Кто президент Марса?")
    assert len(pipe.dm.kg.all_statements()) == before_statements
    assert len(pipe.dm.evidence.all()) == before_evidence


def test_cli_plan_json_is_a_valid_plan():
    out = subprocess.run([sys.executable, "-m", "rasm_cim", "plan",
                          "Кто президент Марса?", "--json"],
                         cwd=REPO, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    payload = json.loads(out.stdout)
    assert payload["schema"] == RESPONSE_SCHEMA
    ir = ResponseIR.from_dict(payload)
    assert ir.status == "UNKNOWN"


def test_cli_generation_prints_metrics():
    out = subprocess.run([sys.executable, "-m", "rasm_cim", "generation"],
                         cwd=REPO, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert "Связность" in out.stdout
    assert "детерминизм: да" in out.stdout
