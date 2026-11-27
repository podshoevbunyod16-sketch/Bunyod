"""Plan B — metrics: arithmetic that can be checked by hand, plus a baseline guard.

The baseline test is deliberately a *regression guard*, not a quality claim:
the numbers it asserts are the measured values of this corpus on this code, and
the corpus is small and written by the same team that wrote the code.  Any
change that lowers them fails loudly instead of quietly.
"""
from __future__ import annotations

import json
import os

import pytest

from rasm_cim_v16.dialogue_manager import DialogueManager
from rasm_cim_v16.evidence_manager import EvidenceManager

from rasm_cim_v17.pipeline import SemanticPipeline
from rasm_cim_v17.semantics import (Case, build_demo, evaluate_answers,
                                    evaluate_entities, evaluate_intents,
                                    load_cases, run_all, seed_report)
from rasm_cim_v17.semantics.corpus import CASES_SCHEMA, CorpusError


@pytest.fixture(scope="module")
def corpus():
    from rasm_cim_v17.semantics import DEFAULT_CORPUS
    return load_cases(DEFAULT_CORPUS)


@pytest.fixture()
def demo():
    dm = build_demo()
    return dm, SemanticPipeline(dm)


# --------------------------------------------------------------------------
# corpus format
# --------------------------------------------------------------------------

def test_corpus_meta_and_size(corpus):
    assert corpus.schema == CASES_SCHEMA
    assert len(corpus.cases) >= 30
    assert corpus.sha256
    summary = corpus.summary()
    assert summary["with_expectations"]["intent"] == len(corpus.cases)


def test_unknown_schema_is_refused(tmp_path):
    p = tmp_path / "c.jsonl"
    p.write_text('{"_meta": "corpus", "schema": "какая-то/9"}\n'
                 '{"text": "Привет"}\n', encoding="utf-8")
    with pytest.raises(CorpusError, match="unsupported corpus schema"):
        load_cases(str(p))


def test_invalid_json_names_the_line(tmp_path):
    p = tmp_path / "c.jsonl"
    p.write_text('{"text": "ok"}\n{сломано}\n', encoding="utf-8")
    with pytest.raises(CorpusError, match=":2:"):
        load_cases(str(p))


def test_empty_corpus_is_refused(tmp_path):
    p = tmp_path / "c.jsonl"
    p.write_text('{"_meta": "corpus", "schema": "rasm-cim-eval-cases/1"}\n',
                 encoding="utf-8")
    with pytest.raises(CorpusError, match="empty"):
        load_cases(str(p))


def test_case_without_text_is_refused():
    with pytest.raises(CorpusError, match="without text"):
        Case(text="   ")


def test_entities_field_distinguishes_none_from_empty(tmp_path):
    p = tmp_path / "c.jsonl"
    p.write_text('{"text": "Привет", "intent": "GREETING"}\n'
                 '{"text": "Пока", "intent": "GREETING", "entities": []}\n',
                 encoding="utf-8")
    cases = load_cases(str(p)).cases
    assert cases[0].entities is None        # not scored
    assert cases[1].entities == []          # scored: no entity allowed


# --------------------------------------------------------------------------
# arithmetic
# --------------------------------------------------------------------------

def test_intent_accuracy_is_the_plain_ratio():
    dm = DialogueManager()
    cases = [Case(text="Привет", intent="GREETING"),
             Case(text="Привет", intent="ASK_WH"),      # deliberately wrong
             Case(text="Привет", intent="GREETING")]
    result = evaluate_intents(cases, dm)
    assert result["n"] == 3 and result["correct"] == 2
    assert result["accuracy"] == pytest.approx(2 / 3, abs=1e-4)
    assert sum(result["confusion"]["GREETING"].values()) == 2


def test_entity_precision_recall_follow_the_confusion_counts():
    dm = DialogueManager()
    dm.say("Исаак Ньютон — это физик")
    cases = [Case(text="Кто такой Исаак Ньютон?", entities=["Исаак Ньютон"]),
             Case(text="Кто такой Наполеон Бонапарт?", entities=[])]
    result = evaluate_entities(cases, dm)
    assert result["tp"] == 1 and result["fp"] == 0 and result["fn"] == 0
    assert result["precision"] == 1.0 and result["recall"] == 1.0


def test_missing_entity_lowers_recall():
    dm = DialogueManager()
    cases = [Case(text="Кто такой Наполеон Бонапарт?", entities=["Наполеон"])]
    result = evaluate_entities(cases, dm)
    assert result["tp"] == 0 and result["fn"] == 1
    assert result["recall"] == 0.0


def test_answer_accuracy_counts_status_and_content():
    dm, pipe = build_demo(), None
    pipe = SemanticPipeline(dm)
    cases = [Case(text="Что открыл Исаак Ньютон?", answer_status="DIRECT",
                  answer_contains=["закон тяготения"]),
             Case(text="Что открыл Исаак Ньютон?", answer_status="DIRECT",
                  answer_contains=["этого нет в базе"])]
    result = evaluate_answers(cases, dm, respond=pipe.respond)
    assert result["correct"] == 1 and result["n"] == 2


def test_unsupported_rate_detects_answers_without_sources(corpus):
    """Wipe the evidence log: every answer fact becomes unsupported."""
    dm = build_demo()
    pipe = SemanticPipeline(dm)
    dm.evidence = EvidenceManager()
    result = evaluate_answers(corpus.cases, dm, respond=pipe.respond)
    assert result["facts_in_answers"] > 0
    assert result["unsupported_rate"] == 1.0
    assert result["unsupported_facts"] == result["facts_in_answers"]


def test_no_answer_expectations_yields_an_explicit_note():
    result = evaluate_answers([Case(text="Привет")], DialogueManager())
    assert result["n"] == 0 and "note" in result


# --------------------------------------------------------------------------
# the baseline (regression guard, not a quality claim)
# --------------------------------------------------------------------------

def test_baseline_metrics_do_not_regress(corpus, demo):
    dm, pipe = demo
    report = run_all(corpus, dm, respond=pipe.respond)
    intents, entities, answers = (report["intents"], report["entities"],
                                  report["answers"])
    assert intents["accuracy"] >= 0.95, intents["misses"]
    assert entities["f1"] >= 0.95, entities["misses"]
    assert answers["accuracy"] >= 0.92, answers["misses"]
    assert answers["unsupported_rate"] == 0.0
    assert report["corpus"]["size"] == len(corpus.cases)
    assert seed_report(dm)["statements"] > 0


def test_report_mentions_the_corpus_hash(corpus, demo):
    dm, pipe = demo
    text = __import__("rasm_cim_v17.semantics", fromlist=["render_report"]) \
        .render_report(run_all(corpus, dm, respond=pipe.respond))
    assert corpus.sha256[:16] in text
    assert "sha256" in text


def test_known_misses_are_documented_in_the_corpus(corpus):
    """The misses we keep must be cases that say why they are hard."""
    hard = [c for c in corpus.cases if c.id.startswith("h")]
    assert hard, "hard cases missing: the corpus would only contain easy ones"
    assert all(c.note for c in hard), \
        "каждый трудный случай должен объяснять, что именно он проверяет"
