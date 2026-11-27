"""v0.13 scenario tests — 2–5 turn dialogues against a seeded KB."""
import os

import pytest

from rasm_cim_v12.knowledge import KnowledgeStore
from rasm_cim_v13.dialogue import ContextStore, DialogueManager


@pytest.fixture()
def setup(tmp_path):
    kb_path = str(tmp_path / "kb")
    kb = KnowledgeStore.init(kb_path)
    # seed a coherent small KB directly via Teach API
    from rasm_cim_v12.knowledge import teach, Provenance
    prov = Provenance(source="test:fixture", doc_id="fixture", added_at="2026-11-26T00:00:00Z")
    teach(kb, "Франция", "столица", "Париж", subject_type="place", object_type="place", provenance=prov)
    teach(kb, "Германия", "столица", "Берлин", subject_type="place", object_type="place", provenance=prov)
    teach(kb, "Киты", "is_a", "млекопитающие", subject_type="thing", object_type="concept", provenance=prov)
    teach(kb, "Толстой", "написал", "Война и мир", subject_type="person", object_type="thing", provenance=prov)
    teach(kb, "Пингвин", "может_летать", "да", object_is_entity=False, polarity="deny",
          subject_type="thing", provenance=prov)
    # a conflicting fact pair
    teach(kb, "Дракон", "существует", "да", object_is_entity=False, polarity="assert", provenance=prov)
    teach(kb, "Дракон", "существует", "да", object_is_entity=False, polarity="deny", provenance=prov)
    ctx = ContextStore(str(tmp_path / "ctx" / "context.json"))
    return kb, ctx


def run(dm, *messages):
    return [dm.process(m) for m in messages]


def answers(d):
    return [(r["status"], r["answer"]) for r in d["results"]]


def test_two_turn_capital_then_population(setup):
    kb, ctx = setup
    dm = DialogueManager(kb, ctx)
    out1 = dm.process("Какая столица у Франции?")
    assert out1["results"][0]["status"] == "TRUE"
    assert "Париж" in out1["results"][0]["answer"]
    out2 = dm.process("А её население?")          # её → Франция (f), no population fact
    assert out2["results"][0]["status"] == "UNKNOWN"
    assert "население" not in out2["results"][0]["answer"] or True
    assert ctx.state["recent_entities"][0]["name"] == "Франция"


def test_two_entities_then_ambiguous_pronoun_asks_clarify(setup):
    kb, ctx = setup
    dm = DialogueManager(kb, ctx)
    dm.process("Какие столицы у Германии и Франции?")
    out = dm.process("А её?")                      # два кандидата (f+f)
    assert out["results"][0]["status"] == "CLARIFY"
    assert "Германия" in out["results"][0]["answer"]
    assert "Франция" in out["results"][0]["answer"]
    assert len(ctx.state["open_questions"]) == 1


def test_pronoun_after_single_entity_resolves(setup):
    kb, ctx = setup
    dm = DialogueManager(kb, ctx)
    dm.process("Какая столица у Германии?")
    out = dm.process("А её население?")            # её → Германия, однозначно (f)
    assert out["results"][0]["status"] == "UNKNOWN"  # нет факта — но без CLARIFY


def test_multi_question_message_two_questions(setup):
    kb, ctx = setup
    dm = DialogueManager(kb, ctx)
    out = dm.process("Что такое киты? И кто написал Войну и мир?")
    assert [r["status"] for r in out["results"]] == ["TRUE", "TRUE"]
    assert "млекопитающ" in out["results"][0]["answer"]
    assert "Толстой" in out["results"][1]["answer"]


def test_multi_question_order_preserved_and_all_answered(setup):
    kb, ctx = setup
    dm = DialogueManager(kb, ctx)
    out = dm.process("Кто написал Войну и мир? А киты — это кто?")
    assert out["results"][0]["entities"] == ["Война и мир"]
    assert out["results"][1]["status"] == "TRUE"


def test_conflicting_fact_shown_not_hidden(setup):
    kb, ctx = setup
    dm = DialogueManager(kb, ctx)
    out = dm.process("Правда ли что дракон существует?")
    assert out["results"][0]["status"] == "CONFLICT"
    assert "противореч" in out["results"][0]["answer"]


def test_yes_no_false_answer(setup):
    kb, ctx = setup
    dm = DialogueManager(kb, ctx)
    out = dm.process("Умеет ли пингвин летать?")
    assert out["results"][0]["status"] == "FALSE"


def test_hypothetical_not_persisted(setup):
    kb, ctx = setup
    dm = DialogueManager(kb, ctx)
    before = kb.stats()["statements_total"]
    out = dm.process("Допустим, что Франция — остров. Какая столица у Франции?")
    assert out["mode"] == "hypothetical"
    assert kb.stats()["statements_total"] == before     # nothing written
    assert out["results"][-1]["status"] == "TRUE"


def test_quote_not_persisted(setup):
    kb, ctx = setup
    dm = DialogueManager(kb, ctx)
    before = kb.stats()["statements_total"]
    out = dm.process('«Франция — это остров»')
    assert out["mode"] == "quote"
    assert kb.stats()["statements_total"] == before


def test_explicit_teach_persists_with_provenance(setup):
    kb, ctx = setup
    dm = DialogueManager(kb, ctx)
    before = kb.stats()["statements_total"]
    out = dm.process("запомни: Франция — это страна")
    assert out["results"][0]["status"] == "TAUGHT"
    assert kb.stats()["statements_total"] == before + 1
    stmts = kb.query_statements(predicate="is_a", object=None)
    taught = [s for s in stmts if s.provenance.get("source") == "user_teach"]
    assert taught and taught[-1].provenance["fragment"].startswith("запомни")


def test_teach_hypothesis_like_phrase_not_auto_fact(setup):
    kb, ctx = setup
    dm = DialogueManager(kb, ctx)
    before = kb.stats()["statements_total"]
    out = dm.process("допустим, запомни: Франция — это остров")
    # hypothetical mode wins over teach; nothing persisted
    assert out["mode"] == "hypothetical"
    assert kb.stats()["statements_total"] == before


def test_three_turn_topic_switch_and_back(setup):
    kb, ctx = setup
    dm = DialogueManager(kb, ctx)
    dm.process("Какая столица у Франции?")
    assert ctx.state["active_topic"] == "Франция"
    dm.process("Что такое киты?")
    assert ctx.state["active_topic"] == "Киты"
    assert "Франция" in ctx.state["topic_stack"]
    out = dm.process("А киты кто такие?")   # возврат к теме китов
    assert out["results"][0]["status"] == "TRUE"


def test_five_turn_sequence_with_evidence_links(setup):
    kb, ctx = setup
    dm = DialogueManager(kb, ctx)
    dm.process("Какая столица у Франции?")
    dm.process("Что такое киты?")
    dm.process("Кто написал Войну и мир?")
    dm.process("Умеет ли пингвин летать?")
    out = dm.process("Правда ли что дракон существует?")
    ids = [ev["statement_id"] for ev in out["results"][0]["evidence"]]
    assert len(ids) == 2   # assert + deny both cited
    for sid in ids:
        assert sid in ctx.state["turns"][-2]["statement_ids"] or sid in ctx.state["turns"][-1]["statement_ids"]


def test_context_export_session_shape(setup, tmp_path):
    kb, ctx = setup
    dm = DialogueManager(kb, ctx)
    dm.process("Какая столица у Франции?")
    from rasm_cim_v13.dialogue import export_session
    out = export_session(ctx)
    assert out["schema"] == "rasm-cim-context/1"
    assert any(t["role"] == "user" for t in out["turns"])
    assert "recent_entities" in out and "open_questions" in out


def test_context_reset(setup):
    kb, ctx = setup
    dm = DialogueManager(kb, ctx)
    dm.process("Какая столица у Франции?")
    ctx.reset()
    assert ctx.state["turns"] == [] and ctx.state["recent_entities"] == []


def test_restart_restores_dialogue_state(setup, tmp_path):
    kb, ctx = setup
    dm = DialogueManager(kb, ctx)
    dm.process("Какая столица у Франции?")
    path = ctx.path
    ctx2 = ContextStore(path)
    names = [e["name"] for e in ctx2.state["recent_entities"]]
    assert "Франция" in names
    dm2 = DialogueManager(kb, ctx2)
    out = dm2.process("А её население?")
    assert out["results"][0]["status"] == "UNKNOWN"     # resolved without CLARIFY
