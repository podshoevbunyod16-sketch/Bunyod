"""Этап 1: запечатанные наборы кореференции, их целостность и поведение.

Тесты делятся на три группы:

* **целостность** — manifest совпадает с файлами, hidden не переписан, генератор
  детерминирован (один seed → те же байты);
* **поведение** — отдельные механизмы (исправление, «они», класс-кандидат,
  отсутствие антецедента) проверяются прямо, а не через агрегат;
* **регрессионный порог** — цифры на dev/validation, полученные до запечатывания
  hidden, зафиксированы как нижняя граница.  Порог не «подгоняется»: если
  изменение ухудшает связку, тест падает.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from rasm_cim_v17.discourse import binding, correction
from rasm_cim_v17.discourse.coref_bench import (
    build_cases,
    build_cases_v2,
    evaluate_split,
    load_split,
    verify_sealed,
    write_split,
)
from rasm_cim_v17.pipeline import SemanticPipeline
from rasm_cim_v17.semantics import build_demo

ROOT = Path(__file__).resolve().parents[1]
CORPUS_V1 = ROOT / "tests_v17" / "corpus" / "coref"
CORPUS_V2 = ROOT / "tests_v17" / "corpus" / "coref2"
PROBE = ROOT / "tests_v17" / "corpus" / "coref_paraphrase.jsonl"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest(directory: Path) -> dict:
    return json.loads((directory / "manifest.json").read_text(encoding="utf-8"))


# --------------------------------------------------------------- целостность
@pytest.mark.parametrize("directory", [CORPUS_V1, CORPUS_V2])
def test_manifest_matches_files(directory: Path):
    manifest = _manifest(directory)
    for name, entry in manifest["splits"].items():
        current = _sha256(directory / f"{name}.jsonl")
        assert current == entry["sha256"], f"{name}: файл изменён после запечатывания"
        assert entry["size"] == sum(
            1 for line in (directory / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#") and '"id"' in line)


@pytest.mark.parametrize("directory", [CORPUS_V1, CORPUS_V2])
def test_hidden_is_sealed(directory: Path):
    manifest = _manifest(directory)
    verify_sealed(load_split(directory / "hidden.jsonl", "hidden"), manifest)


@pytest.mark.parametrize(
    "builder, size",
    [(build_cases, 12), (build_cases_v2, 12)],
)
def test_generator_is_deterministic(builder, size, tmp_path):
    first = tmp_path / "a.jsonl"
    second = tmp_path / "b.jsonl"
    write_split(first, builder("dev", size, 424242))
    write_split(second, builder("dev", size, 424242))
    assert first.read_bytes() == second.read_bytes()
    other = tmp_path / "c.jsonl"
    write_split(other, builder("dev", size, 424243))
    assert other.read_bytes() != first.read_bytes()


def test_splits_have_declared_sizes():
    assert load_split(CORPUS_V1 / "dev.jsonl", "dev").summary()["size"] == 150
    assert load_split(CORPUS_V1 / "validation.jsonl", "validation").summary()["size"] == 100
    assert load_split(CORPUS_V1 / "hidden.jsonl", "hidden").summary()["size"] >= 50
    assert load_split(CORPUS_V2 / "dev.jsonl", "dev").summary()["size"] == 60
    assert load_split(CORPUS_V2 / "validation.jsonl", "validation").summary()["size"] == 40
    assert load_split(CORPUS_V2 / "hidden.jsonl", "hidden").summary()["size"] == 25


# ------------------------------------------------------------------ поведение
def _pipeline(*turns: str) -> tuple[SemanticPipeline, list]:
    dm = build_demo()
    pipe = SemanticPipeline(dm)
    answers = [pipe.respond(turn) for turn in turns]
    return pipe, answers


def test_existing_topic_survives_a_class_question():
    pipe, answers = _pipeline("Пушкин — это поэт", "Кто такой Пушкин?",
                              "Что такое поэт?", "А он поэт?")
    focus = {f.surface: f for f in pipe.last_meaning.focus}
    assert focus["он"].status == binding.FOCUS_RESOLVED
    assert pipe.dm.kg.name_of(focus["он"].entity_id) == "Пушкин"


def test_plural_pronoun_over_two_people_is_refused():
    pipe, answers = _pipeline("Тютчев — это поэт", "Фет — это поэт",
                              "Кто такой Тютчев?", "Кто такой Фет?",
                              "Они поэты?")
    focus = {f.surface: f for f in pipe.last_meaning.focus}
    assert focus["они"].status == binding.FOCUS_AMBIGUOUS
    assert pipe.last_meaning.clarification
    assert any("групп" in u for u in pipe.last_meaning.unknowns)


def test_pronoun_without_antecedent_is_refused():
    pipe, _ = _pipeline("Тютчев — это поэт", "Она поэт?")
    focus = {f.surface: f for f in pipe.last_meaning.focus}
    assert focus["она"].status == binding.FOCUS_AMBIGUOUS


@pytest.mark.parametrize(
    "utterance, target",
    [
        ("Нет, я про Анну", "Анну"),
        ("Я говорил про Анну", "Анну"),
        ("Я имел в виду Анну", "Анну"),
        ("Точнее, Анна", "Анна"),
        ("Нет, не Иван, а Анна", "Анна"),
        ("речь о Анне", "Анне"),
    ],
)
def test_correction_markers(utterance: str, target: str):
    assert correction.detect(utterance) == target


def test_correction_moves_focus_and_answers_with_ack():
    pipe, answers = _pipeline("Иван — это физик", "Анна — это физик",
                              "Кто такой Иван?", "Нет, я про Анну")
    ack = answers[-1]
    assert "Анна" in ack.text
    assert pipe.last_meaning.correction
    pipe.respond("Она физик?")
    focus = {f.surface: f for f in pipe.last_meaning.focus}
    assert pipe.dm.kg.name_of(focus["она"].entity_id) == "Анна"


def test_topic_switch_introducer_moves_focus():
    pipe, answers = _pipeline("Блок — это поэт", "Куприн — это писатель",
                              "Кто такой Блок?", "А теперь про Куприна",
                              "Он писатель?")
    focus = {f.surface: f for f in pipe.last_meaning.focus}
    assert pipe.dm.kg.name_of(focus["он"].entity_id) == "Куприн"


# ------------------------------------------------------------ пороги (нижняя)
def _score(path: Path, split: str) -> dict:
    corpus = load_split(path, split)
    return evaluate_split(corpus, build_demo, lambda dm: SemanticPipeline(dm).respond)


@pytest.mark.parametrize(
    "path, split",
    [(CORPUS_V1 / "dev.jsonl", "dev"), (CORPUS_V1 / "validation.jsonl", "validation"),
     (CORPUS_V2 / "dev.jsonl", "dev"), (CORPUS_V2 / "validation.jsonl", "validation")],
)
def test_no_wrong_binding_on_development_splits(path: Path, split: str):
    score = _score(path, split)
    assert score["wrong_bindings"] == 0
    assert score["ungrounded_rate"]["successes"] == 0
    assert score["refusals_on_unique"] == 0    # ни одного отказа при живом референте
    assert score["safe_accuracy"]["successes"] == score["safe_accuracy"]["total"]


def test_paraphrase_probe_is_reported_not_hidden():
    """Парафразы — диагностика: они написаны после запечатывания hidden.

    Порог низкий намеренно: он фиксирует, что механизм не разваливается на
    других формулировках, и не выдаёт себя за оценку точности.
    """
    score = _score(PROBE, "probe")
    assert score["wrong_bindings"] == 0
    assert score["ungrounded_rate"]["successes"] == 0
    assert score["safe_accuracy"]["successes"] >= 10
