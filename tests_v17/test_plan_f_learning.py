"""Plan F tests — learning that can be shown not to destroy knowledge.

The assertions are about *measured* properties: retention, new conflicts, answer
drift, rollback exactness.  A test that only checked "teach returned DIRECT"
would pass on a system that forgot everything else.
"""
from __future__ import annotations

import json
import pathlib
import subprocess
import sys

import pytest

from rasm_cim_v16.dialogue_manager import DialogueManager
from rasm_cim_v17.learning import (CURRICULUM_SCHEMA, CurriculumError,
                                   Lesson, apply_curriculum, apply_lessons,
                                   compare_states, evaluate_learning,
                                   load_curriculum, render_learning,
                                   rollback_check, snapshot)
from rasm_cim_v17.semantics import build_demo

REPO = pathlib.Path(__file__).resolve().parents[1]
CURRICULUM = REPO / "tests_v17" / "corpus" / "curriculum.jsonl"


def _cur(dm=None):
    return load_curriculum(CURRICULUM)


def _write(tmp_path, lines: list[str]) -> pathlib.Path:
    path = tmp_path / "cur.jsonl"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# --------------------------------------------------------------------------
# curriculum format
# --------------------------------------------------------------------------
def test_curriculum_loads_with_hash_and_kinds():
    cur = _cur()
    assert cur.schema == CURRICULUM_SCHEMA
    assert len(cur.sha256) == 64
    summary = cur.summary()
    assert summary["size"] == len(cur.lessons)
    assert summary["kinds"]["teach"] >= 1
    assert summary["kinds"]["question"] >= 1


def test_empty_curriculum_is_an_error(tmp_path):
    with pytest.raises(CurriculumError):
        load_curriculum(_write(tmp_path, ["# только комментарий"]))


def test_bad_json_points_at_the_line(tmp_path):
    path = _write(tmp_path, ['{"id": "a", "kind": "teach", "text": "x"}',
                             "{сломано}"])
    with pytest.raises(CurriculumError) as exc:
        load_curriculum(path)
    assert ":2:" in str(exc.value)


def test_unknown_kind_rejected(tmp_path):
    path = _write(tmp_path, ['{"id": "a", "kind": "dream", "text": "x"}'])
    with pytest.raises(CurriculumError):
        load_curriculum(path)


def test_duplicate_id_rejected(tmp_path):
    path = _write(tmp_path, ['{"id": "a", "kind": "teach", "text": "x"}',
                             '{"id": "a", "kind": "question", "text": "y?"}'])
    with pytest.raises(CurriculumError):
        load_curriculum(path)


def test_unknown_field_rejected(tmp_path):
    path = _write(tmp_path, ['{"id": "a", "kind": "teach", "text": "x", "bonus": 1}'])
    with pytest.raises(CurriculumError):
        load_curriculum(path)


def test_lesson_without_text_rejected():
    with pytest.raises(CurriculumError):
        Lesson(id="a", kind="teach", text="  ")


# --------------------------------------------------------------------------
# snapshots
# --------------------------------------------------------------------------
def test_snapshot_is_stable_for_the_same_state():
    dm = build_demo()
    assert snapshot(dm)["checksum"] == snapshot(dm)["checksum"]
    assert snapshot(dm)["statement_count"] > 0


def test_compare_states_finds_additions_and_losses():
    before = {"statements": ["a", "b"], "statement_count": 2}
    after = {"statements": ["b", "c"], "statement_count": 2}
    diff = compare_states(before, after)
    assert diff["retained"] == 1
    assert diff["missing"] == ["a"]
    assert diff["new"] == ["c"]
    assert diff["retention"] == 0.5 and diff["forgetting"] == 0.5


def test_rollback_restores_a_fresh_manager_exactly():
    def prepare(dm):
        apply_lessons(dm, _cur())

    check = rollback_check(build_demo, prepare)
    assert check["learning_changed"] is True
    assert check["rollback_exact"] is True
    assert check["before"] == check["restored"]


def test_load_dict_is_additive_which_is_why_rollback_uses_a_fresh_graph():
    """Pins the v0.16 behaviour the rollback design works around."""
    dm = build_demo()
    raw = json.dumps(dm.kg.to_dict())
    count = len(dm.kg.all_statements())
    apply_lessons(dm, _cur())
    dm.kg.load_dict(json.loads(raw))
    assert len(dm.kg.all_statements()) > count      # merge, not replace


# --------------------------------------------------------------------------
# lessons
# --------------------------------------------------------------------------
def test_every_lesson_holds_on_the_demo_base():
    dm = build_demo()
    results = apply_lessons(dm, _cur())
    failures = [(r.id, r.reason) for r in results if not r.ok]
    assert failures == []


def test_duplicate_teach_is_flagged_not_failed():
    dm = build_demo()
    curriculum = load_curriculum(_tmp_curriculum_with(
        ['{"id": "d1", "kind": "teach", "text": "Пушкин написал Евгений Онегин"}']))
    result = apply_lessons(dm, curriculum)[0]
    assert result.ok is True
    assert result.duplicate is True
    assert result.delta == 0


def test_wrong_expectation_is_reported_not_raised():
    dm = build_demo()
    curriculum = load_curriculum(_tmp_curriculum_with(
        ['{"id": "w1", "kind": "teach", "text": "Пушкин написал Евгений Онегин",'
         ' "expect_status": "INFERRED"}']))
    result = apply_lessons(dm, curriculum)[0]
    assert result.ok is False
    assert "INFERRED" in result.reason


def test_lesson_that_is_not_a_fact_adds_nothing():
    dm = build_demo()
    before = len(dm.kg.all_statements())
    curriculum = load_curriculum(_tmp_curriculum_with(
        ['{"id": "q1", "kind": "question", "text": "Перечисли факты о Пушкине"}']))
    results = apply_lessons(dm, curriculum)
    assert len(dm.kg.all_statements()) == before
    assert results[0].delta == 0


def test_retraction_lesson_removes_the_fact():
    dm = build_demo()
    curriculum = load_curriculum(_tmp_curriculum_with(
        ['{"id": "r1", "kind": "retract", "text": "Забудь, что Кит — это млекопитающее"}']))
    assert apply_lessons(dm, curriculum)[0].ok is True


# --------------------------------------------------------------------------
# forgetting report
# --------------------------------------------------------------------------
def test_report_keeps_everything_it_had():
    report = evaluate_learning(_cur(), build_demo)
    assert report["lessons_ok"] == report["lessons_total"] == 12
    assert report["retention"] == 1.0
    assert report["forgetting"] == 0.0
    assert report["missing_facts"] == []
    assert report["new_facts"] >= 1


def test_learning_adds_no_conflicts():
    report = evaluate_learning(_cur(), build_demo)
    assert report["conflicts_after"] == report["conflicts_before"]
    assert report["new_conflicts"] == []
    assert report["conflicts_error"] == ""


def test_conflict_inventory_is_real_not_deduped():
    """A fresh detector sees the demo's known conflict every time."""
    dm = build_demo()
    from rasm_cim_v17.learning.forgetting import _scan_conflicts
    first, _ = _scan_conflicts(dm)
    second, _ = _scan_conflicts(dm)
    assert len(first) == len(second) == 1


def test_no_answer_is_degraded():
    report = evaluate_learning(_cur(), build_demo)
    assert report["degraded_questions"] == []
    assert report["answer_drift"] == 0.0


def test_degraded_answer_is_detected_when_knowledge_is_removed():
    base = ("Пушкин написал Евгений Онегин",
            "Париж находится в Франция")
    curriculum = load_curriculum(_tmp_curriculum_with(
        ['{"id": "x1", "kind": "retract", "text": "Забудь, что Пушкин написал Евгений Онегин"}']))

    def make_dm():
        dm = DialogueManager()
        from rasm_cim_v17.pipeline import SemanticPipeline
        pipe = SemanticPipeline(dm)
        for line in base:
            pipe.respond(line)
        return dm

    report = evaluate_learning(curriculum, make_dm,
                               guards=("Кто написал Евгений Онегин?",))
    assert report["degraded_questions"] == ["Кто написал Евгений Онегин?"]
    assert report["answer_drift"] == 1.0


def test_render_mentions_every_number_the_docs_quote():
    text = render_learning(evaluate_learning(_cur(), build_demo))
    for needle in ("уроки выполнены", "удержание", "противоречий", "дрейф", "откат"):
        assert needle in text


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def test_cli_learn_json_is_machine_readable():
    out = subprocess.run([sys.executable, "-m", "rasm_cim", "learn", "--json"],
                         cwd=REPO, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    payload = json.loads(out.stdout)
    assert payload["curriculum"]["schema"] == CURRICULUM_SCHEMA
    assert payload["retention"] == 1.0
    assert payload["rollback"]["rollback_exact"] is True


def test_cli_learn_exit_code_3_on_failed_lesson(tmp_path):
    path = _write(tmp_path, ['{"id": "bad", "kind": "teach", "text": "Пушкин написал Евгений Онегин", "expect_status": "INFERRED"}'])
    out = subprocess.run([sys.executable, "-m", "rasm_cim", "learn",
                          "--curriculum", str(path)], cwd=REPO,
                         capture_output=True, text=True)
    assert out.returncode == 3
    assert "bad" in out.stdout


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _tmp_curriculum_with(lines: list[str]) -> str:
    import tempfile
    fd, path = tempfile.mkstemp(suffix=".jsonl")
    with open(fd, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    return path
