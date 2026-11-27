"""Plan D tests — reasoning with evidence.

Every test measures a number or a behaviour the docs quote.  Nothing here
asserts that the system "understands" anything; the claims are:

  * the bounds are explicit and enforced where a fact reaches an answer;
  * a derived answer is built from the engine's proof, never assembled by hand;
  * evidence precision and premise coverage are computed from the evidence log;
  * two measured gaps of the frozen layer (inverse reading, wh orientation) are
    filled *only* where the core itself says "не знаю".
"""
from __future__ import annotations

import json
import pathlib
import subprocess
import sys

import pytest

from rasm_cim_v16.dialogue_manager import DialogueManager
from rasm_cim_v16.inference_engine import ST_DIRECT, Fact
from rasm_cim_v17.pipeline import SemanticPipeline
from rasm_cim_v17.reasoning import (DEFAULT_POLICY, ReasoningPolicy, audit_answer,
                                    answer_with_inference, evidence_precision,
                                    explanation_lines, fact_within_policy,
                                    premise_report)
from rasm_cim_v17.semantics import (build_demo, evaluate_evidence,
                                    load_cases, meanings_for, run_all)
from rasm_cim_v17.semantics.harness import DEFAULT_CORPUS

REPO = pathlib.Path(__file__).resolve().parents[1]


def _chain_dm():
    """A tiny KB with one transitive chain and one inverse pair."""
    dm = DialogueManager()
    pipe = SemanticPipeline(dm)
    for line in ("Париж находится в Франция",
                 "Франция находится в Европа",
                 "Пушкин написал Евгений Онегин"):
        pipe.respond(line)
    return dm, pipe


# --------------------------------------------------------------------------
# policy
# --------------------------------------------------------------------------
def test_default_policy_is_bounded():
    assert DEFAULT_POLICY.max_depth == 3
    assert DEFAULT_POLICY.max_nodes == 400
    assert DEFAULT_POLICY.require_proof is True
    assert 0 < DEFAULT_POLICY.min_inferred_confidence < 1


def test_policy_apply_touches_only_bounds():
    dm = DialogueManager()
    ReasoningPolicy(max_depth=1, max_nodes=25).apply(dm)
    assert dm.inference.max_depth == 1
    assert dm.inference.max_nodes == 25


def test_fact_within_policy_rejects_deep_facts():
    deep = Fact(subject=1, predicate="located_in", object=2, status="inferred",
                confidence=0.9, depth=4,
                proof=[{"rule": "transitive", "statement_id": "st-1"}])
    allowed, why = fact_within_policy(deep, ReasoningPolicy(max_answer_depth=3))
    assert allowed is False
    assert "глубина" in why


def test_fact_within_policy_rejects_proofless_inference():
    bare = Fact(subject=1, predicate="located_in", object=2, status="inferred",
                confidence=0.9, depth=1, proof=[])
    allowed, why = fact_within_policy(bare)
    assert allowed is False
    assert "доказательств" in why


def test_fact_within_policy_rejects_low_confidence_inference():
    weak = Fact(subject=1, predicate="located_in", object=2, status="inferred",
                confidence=0.10, depth=1,
                proof=[{"rule": "transitive", "statement_id": "st-1"}])
    allowed, why = fact_within_policy(weak)
    assert allowed is False
    assert "уверенность" in why


def test_fact_within_policy_allows_direct_and_reasonable_inference():
    direct = Fact(subject=1, predicate="located_in", object=2, status=ST_DIRECT,
                  confidence=0.7, depth=0, proof=[])
    assert fact_within_policy(direct)[0] is True
    inferred = Fact(subject=1, predicate="located_in", object=2,
                    status="inferred", confidence=0.6, depth=1,
                    proof=[{"rule": "transitive", "statement_id": "st-1"}])
    assert fact_within_policy(inferred)[0] is True


# --------------------------------------------------------------------------
# audit
# --------------------------------------------------------------------------
def test_audit_counts_and_precision_on_a_real_answer():
    dm, pipe = _chain_dm()
    answer = pipe.respond("Где находится Париж?")
    audit = audit_answer(dm, answer)
    assert audit.facts >= 1
    assert audit.evidence_precision == 1.0
    assert audit.missing_fact_ids == []


def test_audit_flags_unknown_statement_id():
    dm, _ = _chain_dm()
    ghost = Fact(subject=1, predicate="located_in", object=2, status=ST_DIRECT,
                 confidence=0.5, depth=0, proof=[], statement_id="st-999999")

    class _Ans:
        facts = [ghost]

    audit = audit_answer(dm, _Ans())
    assert audit.facts_with_evidence == 0
    assert audit.missing_fact_ids == ["st-999999"]
    assert audit.evidence_precision == 0.0


def test_premise_coverage_of_an_inferred_fact():
    dm, pipe = _chain_dm()
    answer = pipe.respond("Где находится Париж?")
    inferred = [f for f in answer.facts if f.status != ST_DIRECT]
    audit = audit_answer(dm, answer)
    assert audit.premises >= 0
    if inferred:
        report = premise_report(dm, inferred[0])
        assert report["all_premises_evidenced"] is True


def test_evidence_precision_aggregates_several_answers():
    dm, pipe = _chain_dm()
    answers = [pipe.respond(q) for q in ("Где находится Париж?",
                                         "Кто написал Евгений Онегин?",
                                         "Что находится в Европе?")]
    agg = evidence_precision(dm, answers)
    assert agg["facts"] >= 1
    assert agg["evidence_precision"] == 1.0


def test_audit_does_not_write_anything():
    """A question (and its audit) must not add statements or evidence rows."""
    dm, pipe = _chain_dm()
    statements = len(dm.kg.all_statements())
    evidence = len(dm.evidence.all())
    answer = pipe.respond("Где находится Париж?")
    audit_answer(dm, answer)
    assert len(dm.kg.all_statements()) == statements
    assert len(dm.evidence.all()) == evidence


def test_explanation_lines_mark_sources():
    dm, pipe = _chain_dm()
    answer = pipe.respond("Где находится Париж?")
    inferred = [f for f in answer.facts if f.status != ST_DIRECT]
    assert inferred, "ожидался хотя бы один выведенный факт"
    lines = explanation_lines(dm, inferred[0])
    assert any("источник есть" in l for l in lines)
    assert not any("ИСТОЧНИКА НЕТ" in l for l in lines)


# --------------------------------------------------------------------------
# putting inference to work (the two measured gaps)
# --------------------------------------------------------------------------
def test_inverse_reading_is_answered_with_a_proof():
    dm = build_demo()
    pipe = SemanticPipeline(dm)
    answer = pipe.respond("Евгений Онегин написан Пушкиным?")
    assert answer.status != "UNKNOWN"
    assert answer.facts
    assert "Пушкин" in answer.text


def test_inverse_reading_gap_exists_in_the_frozen_core():
    """The improvement is real: with the fallback off, v0.16 says «не знаю»."""
    dm = build_demo()
    pipe = SemanticPipeline(dm, use_inference=False)
    answer = pipe.respond("Евгений Онегин написан Пушкиным?")
    assert answer.status == "UNKNOWN"


def test_wh_orientation_is_answered():
    dm, pipe = _chain_dm()
    answer = pipe.respond("Что находится в Европе?")
    assert answer.status != "UNKNOWN"
    assert any("Франция" in (answer.text or "") for _ in [0])


def test_wh_orientation_gap_exists_in_the_frozen_core():
    dm, pipe = _chain_dm()
    assert dm.say("Что находится в Европе?").status == "UNKNOWN"


def test_fallback_never_overrides_a_confident_core_answer():
    dm = build_demo()
    plain = DialogueManager()
    for line in ("Пушкин написал Евгений Онегин",):
        plain.say(line)
    semantic = SemanticPipeline(dm)
    with_flag = semantic.respond("Кто написал Евгений Онегин?")
    assert with_flag.status == "DIRECT"
    assert semantic.last_delegated is True      # the core answered, we did not


def test_unknown_stays_unknown_without_a_proof():
    dm, pipe = _chain_dm()
    answer = pipe.respond("Кто президент Марса?")
    assert answer.status == "UNKNOWN"


def test_answer_with_inference_returns_none_when_it_cannot():
    dm, pipe = _chain_dm()
    m = meanings_for("Кто президент Марса?", resolver=dm.kg.resolver,
                     schema=dm.kg.schema, parser=dm.parser)[0]
    assert answer_with_inference(dm, m) is None


# --------------------------------------------------------------------------
# metrics + CLI
# --------------------------------------------------------------------------
def test_run_all_reports_evidence_family():
    dm = build_demo()
    corpus = load_cases(DEFAULT_CORPUS)
    report = run_all(corpus, dm, respond=SemanticPipeline(dm).respond)
    evidence = report["evidence"]
    assert evidence["evidence_precision"] is not None
    assert evidence["combined"]["facts"] > 0
    assert evidence["premise_coverage"] is not None


def test_cli_metrics_has_evidence_line():
    out = subprocess.run([sys.executable, "-m", "rasm_cim", "metrics"],
                         cwd=REPO, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert "Источники:" in out.stdout


def test_cli_explain_prints_proof_and_sources(tmp_path):
    home = tmp_path / "home"
    env = {"RASM_CIM_HOME": str(home), "PATH": "/usr/bin:/bin"}
    for line in ("Париж находится в Франция", "Франция находится в Европа"):
        subprocess.run([sys.executable, "-m", "rasm_cim", "ask", line],
                       cwd=REPO, capture_output=True, text=True, env=env)
    out = subprocess.run([sys.executable, "-m", "rasm_cim", "explain",
                          "Где находится Париж?", "--json"],
                         cwd=REPO, capture_output=True, text=True, env=env)
    assert out.returncode == 0, out.stderr
    payload = json.loads(out.stdout)
    assert payload["status"] != "UNKNOWN"
    assert payload["audit"]["facts_with_evidence"] >= 1
    assert payload["facts"]
    assert payload["facts"][0]["within_policy"] is True


def test_cli_explain_exit_zero_on_unknown():
    out = subprocess.run([sys.executable, "-m", "rasm_cim", "explain",
                          "Кто президент Марса?"],
                         cwd=REPO, capture_output=True, text=True)
    assert out.returncode == 0
    assert "доказательств нет" in out.stdout
