"""Spec §5 — natural language understanding.

The same relation must be reachable through several surface forms, without a
single rigid template.  These tests pin the intermediate representation
(``QueryIR``) so that a regression in segmentation is visible even when an
end-to-end answer still happens to look right.
"""
from __future__ import annotations

import pytest

from rasm_cim_v16.language_parser import (INT_ASK_DEFINITION, INT_ASK_WH,
                                           INT_ASK_YESNO, INT_COMMAND,
                                           INT_GREETING, INT_TEACH,
                                           INT_UNKNOWN, LanguageParser)


@pytest.fixture
def p():
    return LanguageParser()


def one(p, text):
    irs = p.parse(text)
    assert len(irs) == 1, f"expected one clause, got {len(irs)}"
    return irs[0]


# ------------------------------------------------------- the spec examples
def test_declarative_discovery(p):
    ir = one(p, "Ньютон открыл закон всемирного тяготения")
    assert ir.intent == INT_TEACH
    assert ir.predicate == "discovered"
    assert ir.subject_mention == "Ньютон"
    assert ir.object_mention == "закон всемирного тяготения"


def test_who_question(p):
    ir = one(p, "Кто открыл закон всемирного тяготения?")
    assert ir.intent == INT_ASK_WH
    assert ir.predicate == "discovered"
    assert ir.wants == "subject"
    assert ir.object_mention == "закон всемирного тяготения"
    assert ir.subject_mention == ""


def test_what_question(p):
    ir = one(p, "Что открыл Ньютон?")
    assert ir.intent == INT_ASK_WH
    assert ir.predicate == "discovered"
    assert ir.wants == "object"
    assert ir.subject_mention == "Ньютон"


def test_kakoy_question_drops_the_class_noun(p):
    ir = one(p, "Какой закон сформулировал Ньютон?")
    assert ir.predicate == "formulated"
    assert ir.subject_mention == "Ньютон"
    assert "закон" not in ir.subject_mention


def test_known_for(p):
    ir = one(p, "Ньютон известен благодаря своим работам по физике")
    assert ir.predicate == "known_for"
    assert ir.subject_mention == "Ньютон"
    assert "работам по физике" in ir.object_mention


def test_tell_me_about(p):
    ir = one(p, "Расскажи о Ньютоне")
    assert ir.intent == INT_ASK_DEFINITION
    assert ir.wants == "related"
    assert ir.subject_mention == "Ньютоне"


def test_who_is(p):
    ir = one(p, "Кто такой Исаак Ньютон?")
    assert ir.intent == INT_ASK_DEFINITION
    assert ir.wants == "definition"
    assert ir.subject_mention == "Исаак Ньютон"


def test_what_is(p):
    ir = one(p, "Что такое компьютер?")
    assert ir.intent == INT_ASK_DEFINITION
    assert ir.subject_mention == "компьютер"


# --------------------------------------------------------- intent coverage
@pytest.mark.parametrize("text,intent", [
    ("Ньютон — это физик", INT_TEACH),
    ("teach Ньютон — это физик", INT_TEACH),
    ("Ньютон — физик?", INT_ASK_YESNO),
    ("Ньютон открыл закон?", INT_ASK_YESNO),
    ("Кто такой Ньютон?", INT_ASK_DEFINITION),
    ("Что открыл Ньютон?", INT_ASK_WH),
    ("забудь факт про Ньютона", "RETRACT"),
    (":inspect", INT_COMMAND),
    ("привет", INT_GREETING),
])
def test_intents(p, text, intent):
    assert one(p, text).intent == intent


def test_command_parsing(p):
    ir = one(p, ":evidence st-000001")
    assert ir.intent == INT_COMMAND
    assert ir.command == "evidence"
    assert ir.command_arg == "st-000001"


# -------------------------------------------------------------- negation
@pytest.mark.parametrize("text", [
    "Ньютон не был физиком",
    "Ньютон не является физиком",
    "Ньютон никогда не был химиком",
])
def test_negation_is_detected(p, text):
    ir = one(p, text)
    assert ir.negation is True, text


def test_positive_sentence_is_not_negated(p):
    assert one(p, "Ньютон был физиком").negation is False


# ------------------------------------------------------- tense / modality
def test_tense_past(p):
    assert one(p, "Ньютон открыл закон").tense == "past"


def test_tense_future(p):
    assert one(p, "Это будет физик").tense == "future"


def test_modality_possible(p):
    assert one(p, "Ньютон мог открыть закон").modality == "possible"


# ------------------------------------------------------------- pronouns
def test_pronoun_is_flagged_not_guessed(p):
    ir = one(p, "Что он открыл?")
    assert ir.subject_pronoun == "он"
    assert ir.subject_mention == ""
    assert ir.predicate == "discovered"


def test_feminine_pronoun(p):
    ir = one(p, "Что она открыла?")
    assert ir.object_pronoun == ""
    assert ir.subject_pronoun == "она"


# -------------------------------------------------------------- ellipsis
def test_elliptical_question(p):
    ir = one(p, "А когда?")
    assert ir.intent == INT_ASK_WH
    assert ir.is_elliptical is True
    assert ir.predicate == "happened_at"
    assert ir.subject_mention == ""


def test_elliptical_where(p):
    ir = one(p, "А где?")
    assert ir.is_elliptical is True
    assert ir.predicate == "located_in"


# --------------------------------------------------- multiple questions
def test_two_sentences_produce_two_irs(p):
    irs = p.parse("Кто такой Ньютон? Что он открыл?")
    assert len(irs) == 2
    assert irs[0].intent == INT_ASK_DEFINITION
    assert irs[1].intent == INT_ASK_WH


# ----------------------------------------------------------- robustness
def test_empty_input(p):
    assert p.parse("") == []
    assert p.parse("   ") == []


@pytest.mark.parametrize("text", [
    "asdf qwer zxcv",
    "…",
    "???",
])
def test_unparseable_is_honest(p, text):
    irs = p.parse(text)
    assert irs
    assert irs[0].intent in (INT_UNKNOWN, INT_TEACH)
    assert irs[0].confidence <= 0.5


def test_case_is_preserved_in_mentions(p):
    """Orthography is the signal that separates a proper name from a class."""
    ir = one(p, "Исаак Ньютон — это физик")
    assert ir.subject_mention == "Исаак Ньютон"
    assert ir.object_mention == "физик"


def test_em_dash_and_plain_dash_behave_alike(p):
    a = one(p, "Ньютон — физик?")
    b = one(p, "Ньютон - физик?")
    assert (a.intent, a.predicate, a.subject_mention, a.object_mention) == \
           (b.intent, b.predicate, b.subject_mention, b.object_mention)


def test_parser_never_mutates_state(p):
    """The parser has no KB access; calling it twice must be identical."""
    a = one(p, "Ньютон открыл закон").as_dict()
    b = one(p, "Ньютон открыл закон").as_dict()
    assert a == b


def test_low_confidence_parse_is_marked(p):
    ir = one(p, "ну такое")
    assert ir.confidence < 0.6
