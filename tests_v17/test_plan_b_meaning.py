"""Plan B — MeaningIR: the meaning layer must be explicit, and must refuse.

Every test here states a semantic claim and checks it without an answer being
produced: the layer is about *what was meant*, not about what is true.
"""
from __future__ import annotations

import pytest

from rasm_cim_v16.dialogue_manager import DialogueManager
from rasm_cim_v16.language_parser import LanguageParser

from rasm_cim_v17.semantics.meaning import (AS_ASK_DEFINITION, AS_ASK_LIST,
                                            AS_ASK_WH, AS_ASK_YESNO, AS_TELL,
                                            AS_UNKNOWN, FOCUS_AMBIGUOUS,
                                            FOCUS_PRONOUN, FOCUS_RESOLVED,
                                            FOCUS_SURFACE, FOCUS_UNKNOWN,
                                            build_meaning, clean_mention,
                                            meanings_for, normalise_act)


@pytest.fixture()
def dm():
    d = DialogueManager()
    d.say("Исаак Ньютон — это физик")
    d.say("Ньютон открыл закон тяготения")
    d.say("Пушкин написал Евгений Онегин")
    return d


def meaning_of(dm, text):
    return meanings_for(text, resolver=dm.kg.resolver, schema=dm.kg.schema,
                        parser=dm.parser)[0]


# --------------------------------------------------------------------------
# speech acts and slots
# --------------------------------------------------------------------------

def test_definition_question_asks_for_a_class(dm):
    m = meaning_of(dm, "Кто такой Исаак Ньютон?")
    assert m.speech_act == AS_ASK_DEFINITION
    assert m.asks == "definition"
    assert m.expected_answer_type == "class"
    assert m.is_question and m.interpretable


def test_related_request_expects_a_list(dm):
    m = meaning_of(dm, "Расскажи про Исаака Ньютона")
    assert m.speech_act == AS_ASK_DEFINITION
    assert m.asks == "related" and m.expected_answer_type == "list"


def test_wh_question_records_the_slot_and_the_type(dm):
    m = meaning_of(dm, "Что открыл Исаак Ньютон?")
    assert (m.speech_act, m.asks, m.expected_answer_type) == \
        (AS_ASK_WH, "object", "entity")
    assert m.relation == "discovered"


def test_who_question_asks_for_the_subject(dm):
    m = meaning_of(dm, "Кто написал Евгений Онегин?")
    assert m.asks == "subject" and m.relation == "wrote"


def test_yes_no_question_asks_for_confirmation(dm):
    m = meaning_of(dm, "Исаак Ньютон — физик?")
    assert m.speech_act == AS_ASK_YESNO
    assert m.asks == "confirmation" and m.expected_answer_type == "boolean"


def test_teach_act_is_normalised_not_renamed(dm):
    m = meaning_of(dm, "Исаак Ньютон — это физик")
    assert m.speech_act == AS_TELL
    assert m.parser_intent == AS_TELL       # both sides normalised
    assert not m.repaired                   # a rename is not a repair
    assert normalise_act("TEACH") == AS_TELL


def test_greeting_and_unknown_are_distinguished(dm):
    assert meaning_of(dm, "Привет").speech_act == "GREETING"
    m = meaning_of(dm, "Абракадабра фыр-фыр")
    assert m.speech_act == AS_UNKNOWN
    assert m.unknowns and m.confidence <= 0.2


def test_imperative_listing_is_repaired_from_teach(dm):
    m = meaning_of(dm, "Перечисли, что открыл Исаак Ньютон")
    assert m.speech_act == AS_ASK_LIST
    assert m.parser_intent == AS_TELL
    assert m.repaired
    assert any("ASK_LIST" in n for n in m.notes)


def test_explicit_teach_marker_vetoes_every_refinement(dm):
    m = meaning_of(dm, "Запомни: перечисли, что открыл Исаак Ньютон")
    assert m.speech_act == AS_TELL
    assert not m.repaired


def test_wh_without_a_relation_becomes_a_definition(dm):
    m = meaning_of(dm, "Кем был Исаак Ньютон?")
    assert m.speech_act == AS_ASK_DEFINITION
    assert m.asks == "definition"
    assert m.repaired


# --------------------------------------------------------------------------
# focus
# --------------------------------------------------------------------------

def test_focus_has_both_sides_of_a_yes_no_question(dm):
    m = meaning_of(dm, "Исаак Ньютон — физик?")
    roles = {f.role: f for f in m.focus}
    assert roles["subject"].status == FOCUS_RESOLVED
    assert roles["object"].status == FOCUS_RESOLVED


def test_unknown_entity_is_reported_not_guessed(dm):
    m = meaning_of(dm, "Кто такой Наполеон Бонапарт?")
    assert m.focus[0].status == FOCUS_UNKNOWN
    assert m.focus[0].entity_id is None
    assert any("неизвестна" in u for u in m.unknowns)
    assert not m.interpretable


def test_pronoun_without_context_is_an_explicit_unknown(dm):
    m = meaning_of(dm, "А он физик?")
    pronouns = [f for f in m.focus if f.status == FOCUS_PRONOUN]
    assert pronouns, [f.as_dict() for f in m.focus]
    assert any("местоимение" in u for u in m.unknowns)


def test_teaching_marks_the_surface_rather_than_requiring_an_entity(dm):
    m = meaning_of(dm, "Циолковский — это изобретатель")
    subj = [f for f in m.focus if f.role == "subject"][0]
    assert subj.status in (FOCUS_SURFACE, FOCUS_RESOLVED)


def test_span_recovery_finds_a_name_inside_a_phrase(dm):
    m = meaning_of(dm, "Сколько лет Исааку Ньютону?")
    assert m.focus[0].status == FOCUS_RESOLVED
    assert "span-recovered" in m.focus[0].note
    assert m.focus[0].confidence < 1.0      # a span is weaker evidence


def test_span_recovery_is_vetoed_by_a_relational_noun(dm):
    """«отец Исаака Ньютона» is a different person from Исаак Ньютон."""
    m = meaning_of(dm, "Кто такой отец Исаака Ньютона?")
    assert m.focus[0].status == FOCUS_UNKNOWN
    assert m.focus[0].note == ""


def test_mention_cleaning_strips_the_complementizer():
    cleaned, note = clean_mention("что Кит")
    assert cleaned == "Кит" and "cleaned" in note
    assert clean_mention("Кит") == ("Кит", "")
    assert clean_mention("  ,что Кит. ") == ("Кит", "mention cleaned: "
                                                    "'  ,что Кит. ' -> 'Кит'")


def test_retract_keeps_the_real_subject_after_cleaning(dm):
    m = meaning_of(dm, "Забудь, что Кит — это млекопитающее")
    assert m.speech_act == "RETRACT"
    surfaces = {f.surface for f in m.focus}
    assert "млекопитающее" in surfaces
    assert any(n.startswith("mention cleaned") for n in m.notes)
    assert m.repaired


# --------------------------------------------------------------------------
# presuppositions and confidence
# --------------------------------------------------------------------------

def test_presuppositions_name_what_the_question_assumes(dm):
    m = meaning_of(dm, "Что открыл Исаак Ньютон?")
    kinds = {p.kind for p in m.presuppositions}
    assert "entity_exists" in kinds and "relation_known" in kinds
    rel = [p for p in m.presuppositions if p.kind == "relation_known"][0]
    assert rel.holds is True


def test_unrecognised_relation_is_declared_as_such(dm):
    m = meaning_of(dm, "Что телепортировал Исаак Ньютон?")
    rels = [p for p in m.presuppositions
            if p.kind in ("relation_known", "relation_recognised")]
    assert rels and rels[0].holds is False
    assert any("отношение не распознано" in u for u in m.unknowns)


def test_unresolved_meaning_caps_the_confidence(dm):
    known = meaning_of(dm, "Кто такой Исаак Ньютон?")
    unknown = meaning_of(dm, "Кто такой Наполеон Бонапарт?")
    assert unknown.confidence < known.confidence


def test_meaning_serialises_without_losing_the_verdict(dm):
    payload = meaning_of(dm, "Кто написал Евгений Онегин?").as_dict()
    assert payload["speech_act"] == AS_ASK_WH
    assert payload["interpretable"] is True
    assert payload["focus"][0]["entity_id"] is not None
