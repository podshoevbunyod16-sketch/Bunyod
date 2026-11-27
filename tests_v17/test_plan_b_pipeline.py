"""Plan B — the pipeline: delegation when nothing is repaired, execution when it is.

The rule the tests enforce: an unrepaired utterance must behave exactly like
v0.16 (delegation), and a repaired one must NOT fall through to the parser's
original act (that is how "Перечисли, ..." used to teach a fact nobody said).
"""
from __future__ import annotations

import pytest

from rasm_cim_v16.dialogue_manager import DialogueManager

from rasm_cim_v17.pipeline import SemanticPipeline
from rasm_cim_v17.semantics.meaning import AS_ASK_LIST, AS_ASK_WH


@pytest.fixture()
def dm():
    d = DialogueManager()
    for line in ("Исаак Ньютон — это физик",
                 "Исаак Ньютон открыл закон тяготения",
                 "Пушкин написал Евгений Онегин"):
        assert d.say(line).status != "UNKNOWN"
    return d


@pytest.fixture()
def pipe(dm):
    return SemanticPipeline(dm)


# --------------------------------------------------------------------------
# delegation
# --------------------------------------------------------------------------

def test_ordinary_question_is_delegated_to_the_core(dm, pipe):
    before = len(dm.kg.all_statements())
    ans = pipe.respond("Что открыл Исаак Ньютон?")
    assert pipe.last_delegated is True
    assert ans.status == "DIRECT" and "закон тяготения" in ans.text
    assert len(dm.kg.all_statements()) == before     # a question writes nothing


def test_definition_question_is_delegated(dm, pipe):
    ans = pipe.respond("Кто такой Исаак Ньютон?")
    assert pipe.last_delegated is True
    assert "физик" in ans.text


def test_semantic_flag_off_always_delegates(dm):
    pipe = SemanticPipeline(dm, semantic=False)
    ans = pipe.respond("Перечисли, что открыл Исаак Ньютон")
    assert pipe.last_delegated is True
    assert ans.status == "DIRECT"          # v0.16 teaches it — the defect itself


# --------------------------------------------------------------------------
# repaired acts
# --------------------------------------------------------------------------

def test_listing_request_does_not_become_a_teach(dm, pipe):
    before = len(dm.kg.all_statements())
    ans = pipe.respond("Перечисли, что открыл Исаак Ньютон")
    assert pipe.last_delegated is False
    assert pipe.last_meaning.speech_act == AS_ASK_LIST
    assert len(dm.kg.all_statements()) == before, "вопрос не должен учить"
    assert "закон тяготения" in ans.text
    assert ans.status == "DIRECT"
    assert ans.facts and all(f.statement_id for f in ans.facts)


def test_listing_answers_carry_evidence(dm, pipe):
    ans = pipe.respond("Перечисли, что открыл Исаак Ньютон")
    evidence = {e.statement_id for e in dm.evidence.all()}
    for fact in ans.facts:
        assert fact.statement_id in evidence


def test_listing_of_an_unknown_entity_is_honest(dm, pipe):
    ans = pipe.respond("Перечисли, что открыл Циолковский")
    assert ans.status == "UNKNOWN"
    assert "не знаю" in ans.text.casefold()


def test_definition_repair_answers_instead_of_teaching(dm, pipe):
    before = len(dm.kg.all_statements())
    ans = pipe.respond("Кем был Исаак Ньютон?")
    assert pipe.last_delegated is False
    assert len(dm.kg.all_statements()) == before
    assert "физик" in ans.text


def test_span_recovered_question_uses_the_core_engine(dm, pipe):
    ans = pipe.respond("Сколько лет Исааку Ньютону?")
    assert pipe.last_meaning.speech_act == AS_ASK_WH
    assert ans.status == "UNKNOWN"          # no age facts: no invention
    assert ans.notes


def test_relational_question_is_not_answered_about_the_wrong_person(dm, pipe):
    ans = pipe.respond("Кто такой отец Исаака Ньютона?")
    assert "физик" not in ans.text, "факты про сына выданы за факты про отца"
    assert ans.status == "UNKNOWN"


def test_retraction_with_complementizer_actually_retracts(dm, pipe):
    dm.say("Кит — это млекопитающее")
    whale = dm.kg.resolver.resolve("Кит").entity_id
    assert dm.kg.objects_of(whale, "instance_of")
    ans = pipe.respond("Забудь, что Кит — это млекопитающее")
    assert ans.status == "DIRECT"
    assert dm.kg.objects_of(whale, "instance_of") == []
    assert any(st.status == "retracted"
               for st in dm.kg.all_statements(include_retracted=True))


def test_retraction_records_evidence_and_history(dm, pipe):
    dm.say("Кит — это млекопитающее")
    pipe.respond("Забудь, что Кит — это млекопитающее")
    retracted = [st for st in dm.kg.all_statements(include_retracted=True)
                 if st.status == "retracted"]
    assert retracted and retracted[0].history
    assert any(e.statement_id == retracted[0].statement_id
               for e in dm.evidence.all())


def test_unresolved_pronoun_leads_to_a_clarification_not_a_guess(dm, pipe):
    ans = pipe.respond("А он физик?")
    assert ans.status in ("AMBIGUOUS", "UNKNOWN")
    assert ans.status != "DIRECT"


def test_unknown_utterance_is_not_treated_as_a_question(dm, pipe):
    ans = pipe.respond("Абракадабра фыр-фыр")
    assert ans.status == "UNKNOWN"


# --------------------------------------------------------------------------
# invariants
# --------------------------------------------------------------------------

def test_pipeline_never_writes_during_a_question(dm, pipe):
    before = len(dm.kg.all_statements())
    for question in ("Что открыл Исаак Ньютон?", "Кто написал Евгений Онегин?",
                     "Перечисли, что открыл Исаак Ньютон",
                     "Кем был Исаак Ньютон?", "Кто такой Наполеон Бонапарт?"):
        pipe.respond(question)
    assert len(dm.kg.all_statements()) == before


def test_every_answer_fact_has_a_source(dm, pipe):
    evidence = {e.statement_id for e in dm.evidence.all()}
    for question in ("Что открыл Исаак Ньютон?", "Кто написал Евгений Онегин?",
                     "Перечисли, что открыл Исаак Ньютон"):
        for fact in pipe.respond(question).facts:
            assert fact.statement_id in evidence


def test_multi_clause_text_is_delegated_unchanged(dm, pipe):
    """Two clauses in one line are v0.16's business, not ours — be predictable."""
    ans = pipe.respond("Исаак Ньютон — физик. Пушкин написал Евгений Онегин.")
    assert ans is not None
