"""v0.13 unit tests: context store, splitter, pronouns."""
import json
import os
import time

import pytest

from rasm_cim_v13.dialogue import (ContextStore, find_pronouns, guess_gender,
                                    resolve, split_questions)


@pytest.fixture()
def ctx(tmp_path):
    return ContextStore(str(tmp_path / "context.json"), max_turns=5, ttl_seconds=1800)


# ---------------------------------------------------------------- context
def test_context_add_and_serialize(ctx, tmp_path):
    ctx.add_turn("user", "привет", intent="greeting")
    ctx.add_turn("system", "Привет!")
    path = ctx.path
    ctx2 = ContextStore(path, max_turns=5, ttl_seconds=1800)
    assert len(ctx2.state["turns"]) == 2
    assert ctx2.state["turns"][0]["text"] == "привет"


def test_context_max_turns_fifo(ctx):
    for i in range(8):
        ctx.add_turn("user", f"вопрос {i}")
    assert len(ctx.state["turns"]) == 5
    assert ctx.state["turns"][0]["text"] == "вопрос 3"


def test_context_ttl_expiry(tmp_path):
    path = str(tmp_path / "context.json")
    ctx = ContextStore(path, max_turns=5, ttl_seconds=1)
    ctx.add_turn("user", "старое")
    # pretend it was last updated long ago: rewrite updated_at directly
    data = json.load(open(path, encoding="utf-8"))
    data["updated_at"] = "2000-01-01T00:00:00Z"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh)
    ctx2 = ContextStore(path, max_turns=5, ttl_seconds=1)
    assert ctx2.state["turns"] == []
    assert ctx2.state["active_topic"] is None


def test_context_corrupt_file_does_not_crash(tmp_path):
    path = str(tmp_path / "context.json")
    with open(path, "w") as fh:
        fh.write("{not valid json")
    ctx = ContextStore(path)
    assert ctx.state["turns"] == []
    assert os.path.exists(path + ".corrupt")


def test_context_wrong_schema_discarded(tmp_path):
    path = str(tmp_path / "context.json")
    with open(path, "w") as fh:
        json.dump({"schema": "rasm-cim-context/999", "turns": [1, 2, 3]}, fh)
    ctx = ContextStore(path)
    assert ctx.state["turns"] == []


def test_recent_entities_mru_and_cap(ctx):
    for name in ["А", "Б", "В", "А"]:
        ctx.record_entities([{"name": name, "gender": "m"}])
    names = [e["name"] for e in ctx.state["recent_entities"]]
    assert names[0] == "А" and names.count("А") == 1
    assert set(names) == {"А", "Б", "В"}


def test_open_questions_add_and_clear(ctx):
    ctx.add_open_question("о ком речь?", missing="referent", candidates=["Франция", "Германия"])
    assert len(ctx.state["open_questions"]) == 1
    assert ctx.clear_open_questions() == 1
    assert ctx.state["open_questions"] == []


# -------------------------------------------------------------- splitter
@pytest.mark.parametrize("text,expected", [
    ("Какая столица у Франции?", ["Какая столица у Франции?"]),
    ("Что такое киты? И кто написал Войну и мир?",
     ["Что такое киты?", "кто написал Войну и мир?"]),
    ("Привет, как дела?", ["Привет", "как дела?"]),
    ("1) какая столица у Франции? 2) кто написал Войну и мир?",
     ["какая столица у Франции?", "кто написал Войну и мир?"]),
    ("Где живёт Пушкин? А Лермонтов?", ["Где живёт Пушкин?", "Лермонтов?"]),  # leading А — discourse particle, stripped
    ("", []),
])
def test_split_questions(text, expected):
    assert split_questions(text) == expected


def test_split_conjunction_both_sides_questions():
    out = split_questions("Кто написал Пушкина и где он жил?")
    assert out == ["Кто написал Пушкина?", "где он жил?"]


def test_split_conjunction_one_side_not_question_keeps_whole():
    out = split_questions("какие есть планеты и спутники?")
    assert out == ["какие есть планеты и спутники?"]


def test_split_preserves_order():
    out = split_questions("Первый вопрос? Второй вопрос? Третий вопрос?")
    assert out == ["Первый вопрос?", "Второй вопрос?", "Третий вопрос?"]


# ---------------------------------------------------------------- pronouns
def test_guess_gender():
    assert guess_gender("Франция") == "f"
    assert guess_gender("Берлин") == "m"
    assert guess_gender("Япония") == "f"
    assert guess_gender("пальто") == "n"
    assert guess_gender("кофе") == "m"


def test_find_pronouns_spans():
    text = "Он был в её доме"
    found = find_pronouns(text)
    spans = [(p["pronoun"], p["gender"]) for p in found]
    assert ("Он", "m") in spans and ("её", "f") in spans
    for p in found:
        assert text[p["start"]:p["end"]] == p["pronoun"]


def test_resolve_singular_unique_candidate():
    prons = find_pronouns("а её столица?")
    recent = [{"name": "Франция", "gender": "f"}, {"name": "Берлин", "gender": "m"}]
    res = resolve(prons, recent)
    assert res[0]["status"] == "resolved" and res[0]["chosen"] == ["Франция"]


def test_resolve_singular_ambiguous_two_candidates():
    prons = find_pronouns("а её столица?")
    recent = [{"name": "Франция", "gender": "f"}, {"name": "Германия", "gender": "f"}]
    res = resolve(prons, recent)
    assert res[0]["status"] == "ambiguous"
    assert set(res[0]["candidates"]) == {"Франция", "Германия"}


def test_resolve_singular_unknown_when_zero_candidates():
    prons = find_pronouns("а её столица?")
    res = resolve(prons, [{"name": "Берлин", "gender": "m"}])
    assert res[0]["status"] == "unknown"


def test_resolve_plural_set():
    prons = find_pronouns("а их столицы?")
    recent = [{"name": "Франция", "gender": "f"}, {"name": "Германия", "gender": "f"}]
    res = resolve(prons, recent)
    assert res[0]["status"] == "resolved"
    assert set(res[0]["chosen"]) == {"Франция", "Германия"}
