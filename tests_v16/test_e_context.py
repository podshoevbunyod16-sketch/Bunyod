"""TEST E / F / L + spec §7 — dialogue context.

Pronouns must be resolved by gender and salience, never by "the last noun".
Topic switching must isolate two subjects and allow a clean return.
"""
from __future__ import annotations

import pytest

from rasm_cim_v16.context_engine import (PR_AMBIGUOUS, PR_RESOLVED,
                                         PR_UNRESOLVED, ContextEngine,
                                         migrate_context_dict)


# ------------------------------------------------------------------ TEST E
def test_e_pronoun_resolves_to_the_subject_just_discussed(dm):
    dm.say("teach Исаак Ньютон открыл закон всемирного тяготения")
    a = dm.say("Кто такой Исаак Ньютон?")
    b = dm.say("Что он открыл?")
    assert b.status == "CONTEXTUAL"
    assert "закон всемирного тяготения" in b.text


def test_e_pronoun_gender_prevents_a_wrong_referent(dm):
    dm.say("teach Исаак Ньютон — это физик")
    dm.say("teach Мария Кюри — это химик")
    dm.say("Кто такая Мария Кюри?")
    a = dm.say("Что она открыла?")
    # the masculine Newton must not be picked for "она"
    assert "Ньютон" not in a.text


def test_e_pronoun_with_no_referent_asks(dm):
    a = dm.say("Что он открыл?")
    assert a.status == "AMBIGUOUS"
    assert a.clarification


def test_e_ambiguous_pronoun_asks_instead_of_guessing(dm):
    """Two masculine entities at equal salience -> must ask."""
    dm.say("teach Исаак Ньютон открыл закон всемирного тяготения")
    dm.say("teach Альберт Эйнштейн открыл теорию относительности")
    ce = dm.context
    # force an exact tie so the engine cannot pick
    for ae in ce.active.values():
        ae.salience = 1.0
        ae.gender = "masculine"
    res = ce.resolve_pronoun("он")
    assert res.status == PR_AMBIGUOUS
    assert len(res.candidates) >= 2
    assert res.question


def test_e_pronoun_resolution_is_logged(dm):
    dm.say("teach Исаак Ньютон открыл закон всемирного тяготения")
    dm.say("Что он открыл?")
    assert dm.context.pronoun_log


# ------------------------------------------------------------- ellipsis
def test_elliptical_when_asks_for_clarification_when_no_time_is_known(dm):
    dm.say("teach Исаак Ньютон открыл закон всемирного тяготения")
    dm.say("Что он открыл?")
    a = dm.say("А когда?")
    assert a.status == "UNKNOWN"
    assert "Не знаю" in a.text
    assert not any(c in a.text for c in ("1687", "1643")), \
        "a date must never be invented"


def test_elliptical_when_uses_a_known_time(dm):
    dm.say("teach Исаак Ньютон открыл закон всемирного тяготения")
    dm.say("teach Закон всемирного тяготения был открыт в 1687 году")
    dm.say("Что он открыл?")
    a = dm.say("А когда?")
    assert "1687" in a.text


def test_elliptical_without_any_focus_is_honest(dm):
    a = dm.say("А когда?")
    assert a.status == "UNKNOWN"


# ------------------------------------------------------------------ TEST F
def test_f_topic_switch_and_return(dm):
    for t in ["teach Исаак Ньютон — это физик",
              "teach Ньютон — это Исаак Ньютон"]:
        dm.say(t)
    a1 = dm.say("Кто такой Исаак Ньютон?")
    assert "физик" in a1.text
    newton_topic = dm.context.topic.entity_ids[:]

    dm.say("teach Компьютер — это устройство")
    a2 = dm.say("Что такое компьютер?")
    assert "устройство" in a2.text
    assert "физик" not in a2.text, "topics must not be mixed"
    assert set(newton_topic) & set(dm.context.topic.entity_ids) == set()

    a3 = dm.say("Расскажи о Ньютоне")
    assert "физик" in a3.text
    assert "устройство" not in a3.text
    assert dm.context.topic.label != "Компьютер"


def test_f_topic_stack_is_maintained(dm):
    dm.say("teach Исаак Ньютон — это физик")
    dm.say("Кто такой Исаак Ньютон?")
    dm.say("teach Компьютер — это устройство")
    dm.say("Что такое компьютер?")
    assert len(dm.context.topic_stack) >= 1
    labels = {t.label for t in dm.context.topic_stack}
    assert any("Ньютон" in (l or "") for l in labels)


def test_f_classify_topic_before_touching():
    ce = ContextEngine()
    assert ce.classify_topic([]) == "same"
    assert ce.classify_topic([1, 2]) == "new"
    ce.switch_topic([1, 2], label="A")
    assert ce.classify_topic([1]) == "same"
    assert ce.classify_topic([3]) == "new"
    ce.switch_topic([3], label="B")
    assert ce.classify_topic([1]) == "resume"


def test_f_return_to_previous():
    ce = ContextEngine()
    ce.switch_topic([1], label="A")
    ce.switch_topic([2], label="B")
    assert ce.topic.label == "B"
    prev = ce.return_to_previous()
    assert prev.label == "A"
    assert ce.topic.label == "A"


# ------------------------------------------------------------------ TEST L
LONG_SCRIPT = [
    "teach Исаак Ньютон — это физик",
    "teach Ньютон — это Исаак Ньютон",
    "teach Исаак Ньютон открыл закон всемирного тяготения",
    "Кто такой Исаак Ньютон?",
    "Ньютон — физик?",
    "Что он открыл?",
    "teach Компьютер — это устройство",
    "Что такое компьютер?",
    "Компьютер — устройство?",
    "teach Мария Кюри — это химик",
    "Кто такая Мария Кюри?",
    "Мария Кюри — химик?",
    "Расскажи о Ньютоне",
    "Ньютон — учёный?",
    "teach Физик — это учёный",
    "Ньютон — учёный?",
    "Что такое компьютер?",
    "Кто такая Мария Кюри?",
    "Расскажи о Ньютоне",
    "Что он открыл?",
    "Ньютон — физик?",
    "Кто написал Войну и мир?",
]


def test_l_long_dialogue_keeps_context_clean(dm):
    assert len(LONG_SCRIPT) >= 20
    for line in LONG_SCRIPT:
        ans = dm.say(line)
        assert ans.text, f"empty answer for {line!r}"

    # Newton answers must never be polluted by the computer or Curie topics
    a = dm.say("Расскажи о Ньютоне")
    assert "физик" in a.text
    assert "устройство" not in a.text
    assert "химик" not in a.text

    c = dm.say("Что такое компьютер?")
    assert "устройство" in c.text
    assert "Ньютон" not in c.text

    # every turn is in the episodic memory, in order
    turns = [t for t in dm.context.history if t.role == "user"]
    assert [t.text for t in turns] == LONG_SCRIPT + ["Расскажи о Ньютоне",
                                                     "Что такое компьютер?"]


def test_l_no_runaway_growth(dm):
    for _ in range(3):
        for line in LONG_SCRIPT:
            dm.say(line)
    assert len(dm.context.history) <= dm.context.max_turns
    assert len(dm.context.active) <= 40
    assert len(dm.context.topic_stack) <= 20
    assert len(dm.memory.working()) <= dm.memory.working_cap


# ----------------------------------------------------------- context API
def test_context_reset_keeps_the_knowledge_base(dm):
    dm.say("teach Исаак Ньютон — это физик")
    n = len(dm.kg.all_statements())
    dm.command("reset", "")
    assert len(dm.kg.all_statements()) == n
    assert dm.context.history == []
    assert dm.context.active == {}


def test_context_integrity_check(dm):
    dm.say("teach Исаак Ньютон — это физик")
    dm.say("Кто такой Исаак Ньютон?")
    assert dm.context.integrity_check() == []


def test_context_integrity_detects_an_orphan():
    from rasm_cim_v16.context_engine import ActiveEntity
    ce = ContextEngine()
    ce.active[999] = ActiveEntity(entity_id=999, salience=1.0, last_turn=0)
    ce.resolver = None
    assert ce.integrity_check() == []          # no resolver -> cannot judge
    from rasm_cim_v16.entity_resolver import EntityResolver
    ce.resolver = EntityResolver()
    problems = ce.integrity_check()
    assert any("not in the KB" in p for p in problems)


def test_context_migration_from_v1():
    old = {"schema": 1, "turn_no": 3,
           "entities": [{"id": 1}, {"id": 2}],
           "history": [{"id": "u-1", "role": "user", "text": "привет"}]}
    notes = migrate_context_dict(old)
    assert notes, "a migration must report what it did"
    assert old["schema"] == 2
    assert [a["id"] for a in old["active"]] == [1, 2]
    assert old["topic"]["entity_ids"] == [1, 2]
    assert len(old["history"]) == 1
    assert migrate_context_dict(old) == [], "migration must be idempotent"


def test_context_roundtrip(dm):
    dm.say("teach Исаак Ньютон открыл закон всемирного тяготения")
    dm.say("Что он открыл?")
    d = dm.context.to_dict()
    ce2 = ContextEngine.from_dict(d)
    assert ce2.turn_no == dm.context.turn_no
    assert [t.turn_id for t in ce2.history] == [t.turn_id for t in dm.context.history]
    assert ce2.topic.label == dm.context.topic.label
    assert ce2.last_event == dm.context.last_event
    assert ce2.integrity_check() == []
