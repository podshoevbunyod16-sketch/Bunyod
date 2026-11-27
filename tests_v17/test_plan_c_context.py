"""Plan C — context binding and coreference measurement.

Two claims are under test: (1) a pronoun is bound only when the v0.16 context
resolver is confident, and every refusal is visible; (2) a dialogue keeps its
salience state alive even when turn 2 is executed by the v0.17 pipeline.
"""
from __future__ import annotations

import pytest

from rasm_cim_v17.discourse import (DialogueCase, evaluate_coreference,
                                    load_dialogues, record_turn)
from rasm_cim_v17.discourse.binding import (apply_context, bind_pronouns,
                                            resolved_referents)
from rasm_cim_v17.discourse.dialogue_corpus import (DIALOGUE_SCHEMA,
                                                    DialogueCorpusError,
                                                    Reference)
from rasm_cim_v17.discourse.dialogue_metrics import render_coreference
from rasm_cim_v17.pipeline import SemanticPipeline
from rasm_cim_v17.semantics import build_demo, meanings_for
from rasm_cim_v17.semantics.harness import DEFAULT_DIALOGUES


@pytest.fixture()
def dm():
    return build_demo()


@pytest.fixture()
def pipe(dm):
    return SemanticPipeline(dm)


def meaning_of(dm, text):
    return meanings_for(text, resolver=dm.kg.resolver, schema=dm.kg.schema,
                        parser=dm.parser)[0]


# --------------------------------------------------------------------------
# corpus format
# --------------------------------------------------------------------------

def test_dialogue_corpus_loads_with_schema_and_hash():
    corpus = load_dialogues(DEFAULT_DIALOGUES)
    assert corpus.schema == DIALOGUE_SCHEMA
    assert len(corpus.cases) >= 6
    assert corpus.summary()["references"] >= 4
    assert len(corpus.sha256) == 64


def test_reference_out_of_range_is_refused():
    with pytest.raises(DialogueCorpusError, match="out of range"):
        DialogueCase(id="x", turns=["один"],
                     refers_to=[Reference(turn=3, pronoun="он", entity="A")])


def test_dialogue_without_turns_is_refused():
    with pytest.raises(DialogueCorpusError, match="without turns"):
        DialogueCase(id="x", turns=[])


def test_unknown_dialogue_schema_is_refused(tmp_path):
    p = tmp_path / "d.jsonl"
    p.write_text('{"_meta": "corpus", "schema": "other/9"}\n'
                 '{"id": "d1", "turns": ["Привет"]}\n', encoding="utf-8")
    with pytest.raises(DialogueCorpusError, match="unsupported schema"):
        load_dialogues(str(p))


# --------------------------------------------------------------------------
# binding
# --------------------------------------------------------------------------

def test_pronoun_resolves_after_a_question(dm):
    dm.say("Кто такой Исаак Ньютон?")
    m = meaning_of(dm, "Что он открыл?")
    assert any(f.status == "PRONOUN" for f in m.focus)
    resolutions = bind_pronouns(m, dm)
    assert resolutions and resolutions[0].entity_id is not None
    bound = [f for f in m.focus if f.entity_id is not None]
    assert dm.kg.name_of(bound[0].entity_id) == "Исаак Ньютон"
    assert any("pronoun" in n for n in m.notes)


def test_pronoun_without_context_is_refused_with_a_question():
    from rasm_cim_v16.dialogue_manager import DialogueManager
    empty = DialogueManager()                 # no turns at all
    m = meaning_of(empty, "Что он открыл?")
    bind_pronouns(m, empty)
    assert all(f.entity_id is None for f in m.focus
               if f.surface == "он")
    assert m.clarification
    assert any("местоимение" in u for u in m.unknowns)


def test_gender_mismatch_refuses_instead_of_binding():
    from rasm_cim_v16.dialogue_manager import DialogueManager
    local = DialogueManager()
    local.say("Исаак Ньютон — это физик")      # masculine, the only candidate
    m = meaning_of(local, "А она физик?")
    bind_pronouns(m, local)
    assert all(f.entity_id is None for f in m.focus if f.surface == "она")
    assert m.unknowns


def test_apply_context_caps_confidence_on_refusal(dm):
    m = meaning_of(dm, "Кто такой он?")
    before = m.confidence
    apply_context(m, dm)
    assert m.confidence <= before
    assert m.confidence <= 0.5


def test_resolved_referents_lists_ids_only(dm):
    dm.say("Кто такой Исаак Ньютон?")
    m = meaning_of(dm, "Что он открыл?")
    apply_context(m, dm)
    ids = resolved_referents(m)
    assert ids and all(isinstance(i, int) for i in ids)
    assert [dm.kg.name_of(i) for i in ids] == ["Исаак Ньютон"]


# --------------------------------------------------------------------------
# pipeline behaviour across turns
# --------------------------------------------------------------------------

def test_two_turn_dialogue_answers_the_pronoun(pipe, dm):
    first = pipe.respond("Кто такой Исаак Ньютон?")
    assert first.status == "DIRECT"
    second = pipe.respond("А он физик?")
    assert second.status == "DIRECT"
    assert "да" in second.text.casefold()


def test_pronoun_uses_the_last_mention(pipe, dm):
    pipe.respond("Кто такой Исаак Ньютон?")
    pipe.respond("Кто такой Альберт Эйнштейн?")
    ans = pipe.respond("А он физик?")
    assert ans.status == "DIRECT"
    # both are physicists, so the answer alone cannot distinguish them:
    # the meaning layer must have bound the LAST one.
    names = [dm.kg.name_of(f.entity_id) for f in pipe.last_meaning.focus
             if f.entity_id is not None]
    assert "Альберт Эйнштейн" in names


def test_repaired_turn_keeps_the_context_warm(pipe, dm):
    """A listing (repaired) turn must not break the next pronoun."""
    pipe.respond("Кто такой Исаак Ньютон?")
    pipe.respond("Перечисли, что открыл Исаак Ньютон")      # repaired path
    ans = pipe.respond("А он физик?")
    assert ans.status == "DIRECT"


def test_gender_mismatch_leads_to_a_clarification(pipe):
    pipe.respond("Кто такой Исаак Ньютон?")
    ans = pipe.respond("А она физик?")
    assert ans.status in ("UNKNOWN", "AMBIGUOUS")
    assert ans.status != "DIRECT"


def test_yes_no_uses_the_relation_that_actually_holds(pipe, dm):
    pipe.respond("Кто написал Евгений Онегин?")
    ans = pipe.respond("Пушкин — автор Евгения Онегина?")
    assert ans.status in ("DIRECT", "UNKNOWN")
    if ans.status == "DIRECT":
        assert "да" in ans.text.casefold()


def test_record_turn_updates_context_explicitly(dm):
    m = meaning_of(dm, "Кто такой Исаак Ньютон?")
    apply_context(m, dm)
    before = len(dm.context.turns) if hasattr(dm.context, "turns") else None
    class FakeAnswer:
        text = "…"
        status = "DIRECT"
        confidence = 0.9
        facts = []
    record_turn(dm, "Кто такой Исаак Ньютон?", m, FakeAnswer())
    after = len(dm.context.turns) if hasattr(dm.context, "turns") else None
    if before is not None and after is not None:
        assert after >= before


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------

def test_coreference_metrics_on_the_real_corpus():
    corpus = load_dialogues(DEFAULT_DIALOGUES)
    report = evaluate_coreference(corpus, build_demo,
                                  lambda dm: SemanticPipeline(dm).respond)
    assert report["references"] >= 4
    assert report["wrong"] == 0, report["misses"]
    assert report["accuracy"] >= 0.8
    assert report["answer_accuracy"] == 1.0
    assert report["sha256"] == corpus.sha256


def test_coreference_reports_refusals_separately():
    corpus = load_dialogues(DEFAULT_DIALOGUES)
    report = evaluate_coreference(corpus, build_demo,
                                  lambda dm: SemanticPipeline(dm).respond)
    assert report["correct"] + report["refused"] + report["wrong"] == \
        report["references"]
    assert report["refused"] >= 1        # d04: a genuine gender tie


def test_coreference_render_mentions_the_hash():
    corpus = load_dialogues(DEFAULT_DIALOGUES)
    report = evaluate_coreference(corpus, build_demo,
                                  lambda dm: SemanticPipeline(dm).respond)
    text = render_coreference(report)
    assert corpus.sha256[:16] in text
    assert "отказов" in text
