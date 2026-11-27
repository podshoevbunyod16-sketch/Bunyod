"""v0.14 — reasoning pipeline tests: direct facts, multi-hop rules,
negative/adversarial cases, hypothesis isolation, evidence gate, determinism."""
import json

import pytest

from rasm_cim_v12.knowledge import KnowledgeStore, teach, Provenance
from rasm_cim_v14.reasoning import ask, ReasoningPipeline
from rasm_cim_v14.reasoning.pipeline import ReasoningPipeline as RP
from rasm_cim_v14.reasoning.evidence import EvidenceSet, EvidenceItem


def _prov():
    return Provenance(source="test_fixture", doc_id="tests_v14",
                      fragment="fixture", method="manual",
                      parser_version="tests-0.14.0")


def _kb(path):
    kb = KnowledgeStore.init(str(path))
    teach(kb, "Франция", "capital_of", "Париж", subject_type="place",
          object_type="place", provenance=_prov())
    teach(kb, "Франция", "located_in", "Европа", subject_type="place",
          object_type="concept", provenance=_prov())
    teach(kb, "Киты", "is_a", "млекопитающее", subject_type="animal",
          object_type="concept", provenance=_prov())
    teach(kb, "млекопитающее", "is_a", "животное", subject_type="concept",
          object_type="concept", provenance=_prov())
    teach(kb, "животное", "is_a", "живое существо", subject_type="concept",
          object_type="concept", provenance=_prov())
    teach(kb, "Пингвин", "может_летать", "да", object_is_entity=False,
          subject_type="animal", polarity="deny", provenance=_prov())
    teach(kb, "Дракон", "существует", "да", object_is_entity=False,
          subject_type="animal", polarity="assert", provenance=_prov())
    teach(kb, "Дракон", "существует", "да", object_is_entity=False,
          subject_type="animal", polarity="deny", provenance=_prov())
    teach(kb, "Венера", "is_a", "планета", subject_type="thing",
          object_type="concept", kind="hypothesis", provenance=_prov())
    teach(kb, "Франция", "population_of", "68 000 000", object_is_entity=False,
          subject_type="place", provenance=_prov())
    teach(kb, "Германия", "population_of", "84 000 000", object_is_entity=False,
          subject_type="place", provenance=_prov())
    return kb


@pytest.fixture()
def kb(tmp_path):
    return _kb(tmp_path / "kb")


class TestDirectAnswers:
    def test_capital_exact_with_evidence_id(self, kb):
        r = ask(kb, "Какая столица у Франции?")
        assert r["verdict"] == "TRUE"
        assert "Париж" in r["answer"]
        assert r["evidence_ids"], "substantive sentence must carry evidence ids"
        assert all(i.startswith("st_") for i in r["evidence_ids"])
        assert "[st_" in r["answer"]

    def test_definition_mentions_chain_when_asked(self, kb):
        r = ask(kb, "Что такое Киты?", max_depth=3)
        assert "млекопитающ" in r["answer"]
        assert "животное" in r["answer"]          # depth-2 derived
        assert "живое существо" in r["answer"]    # depth-3 derived
        assert set(r["evidence_ids"]) >= {i for i in r["evidence_ids"]}
        assert r["evidence"]["applied_rules"][0]["rule_ids"] == ["R01"]

    def test_yes_no_deny_verdict_false(self, kb):
        r = ask(kb, "Умеет ли пингвин летать?")
        assert r["verdict"] == "FALSE"
        assert r["evidence_ids"]

    def test_conflict_surfaces(self, kb):
        r = ask(kb, "Правда ли что дракон существует?")
        assert r["verdict"] == "CONFLICT"
        assert r["status"] == "conflict"
        assert len(r["evidence_ids"]) == 2

    def test_hypothesis_only_never_becomes_fact(self, kb):
        r = ask(kb, "Что такое Венера?")
        assert r["verdict"] == "UNKNOWN"
        assert r["status"] in ("partial", "unknown")
        assert "планета" not in r["answer"] or "гипотез" in r["answer"]


class TestMultiHop:
    def test_located_in_via_capital_rule(self, kb):
        r = ask(kb, "Где расположен Париж?", max_depth=2)
        assert "Франция" in r["answer"]
        assert any(rid == "R02" for rid in
                   [x for a in r["evidence"]["applied_rules"] for x in a["rule_ids"]])

    def test_located_in_transitive_depth_2(self, kb):
        r = ask(kb, "Где расположен Париж?", max_depth=3)
        assert "Европа" in r["answer"]           # Париж→Франция→Европа via R02+R03
        assert r["status"] in ("partial", "supported")

    def test_depth_cap_respected(self, kb):
        r = ask(kb, "Что такое Киты?", max_depth=1)
        assert "животное" not in r["answer"]     # chain cut at depth 1
        assert "млекопитающ" in r["answer"]

    def test_is_a_cycle_terminates(self, kb):
        teach(kb, "А", "is_a", "Б", subject_type="concept",
              object_type="concept", provenance=_prov())
        teach(kb, "Б", "is_a", "А", subject_type="concept",
              object_type="concept", provenance=_prov())
        r = ask(kb, "Что такое А?", max_depth=3)   # must not hang
        assert r["answer"]

    def test_width_cap_terminates(self, kb):
        for i in range(20):
            teach(kb, f"Сущ{i}", "is_a", "Животное", subject_type="thing",
                  object_type="concept", provenance=_prov())
            teach(kb, "Животное", "is_a", f"Категория{i}", subject_type="concept",
                  object_type="concept", provenance=_prov())
        r = ask(kb, "Что такое Сущ0?", max_depth=3)
        assert r["answer"]


class TestAbstainAndAdversarial:
    @pytest.mark.parametrize("q", ["Какая столица у Атлантиды?",
                                    "Что такое Голубица?",
                                    "asdfgh jkl"])
    def test_unknown_never_hallucinates(self, kb, q):
        before_stats = kb.stats()
        r = ask(kb, q)
        assert r["verdict"] in ("UNKNOWN", "CONFLICT")
        assert r["evidence_ids"] == [] or r["verdict"] == "CONFLICT"
        after_stats = kb.stats()
        assert before_stats["statements_total"] == after_stats["statements_total"]
        assert before_stats["entities"] == after_stats["entities"]

    def test_compare_abstains_without_both_numbers(self, kb):
        kb2 = KnowledgeStore.init(str(kb.path) + "_alone") if False else kb
        r = kb2 and ask(kb2, "У кого больше население: Франция или Атлантида?")
        assert r["verdict"] == "UNKNOWN"
        assert r["missing_prereqs"]

    def test_compare_picks_larger(self, kb):
        r = ask(kb, "У кого население больше: Франция или Германия?")
        assert r["verdict"] == "TRUE"
        assert "Германия" in r["answer"]
        assert r["evidence_ids"]

    def test_empty_kb(self, tmp_path):
        empty = KnowledgeStore.init(str(tmp_path / "empty"))
        r = ask(empty, "Какая столица у Франции?")
        assert r["verdict"] == "UNKNOWN"
        assert r["status"] in ("unknown", "partial")

    def test_determinism_same_input_twice(self, kb):
        r1 = ask(kb, "Что такое Киты?", max_depth=3)
        r2 = ask(kb, "Что такое Киты?", max_depth=3)
        assert r1 == r2


class TestEvidenceGate:
    def test_finalize_drops_unsupported_sentence(self):
        es = EvidenceSet(intent="ask_fact", subject_names=["X"], predicate="is_a")
        # no items at all -> nothing supports the bracketed claim
        out = RP(KnowledgeStore.__new__(KnowledgeStore))._finalize(
            "q", es,
            ["Парламент принят [st_0000000000000000].",
             "Нет данных вообще."],
            verdict="UNKNOWN")
        assert "st_0000000000000000" not in out["answer"]
        assert out["missing_prereqs"] == []
        assert "Нет данных" in out["answer"]

    def test_missing_item_status_partial(self):
        es = EvidenceSet(intent="ask_fact", subject_names=["X"], predicate="is_a")
        es.items.append(EvidenceItem(role="missing", note="нет факта"))
        assert es.status() == "partial"

    def test_conflict_status_aggregation(self):
        es = EvidenceSet(intent="ask_fact", subject_names=["X"], predicate="существует")
        es.items.append(EvidenceItem(role="direct", statement_id="st_a",
                                     note="CONFLICT"))
        assert es.status() == "conflict"

    def test_all_sentences_cited_or_abstained(self, kb):
        r = ask(kb, "Какая столица у Франции?")
        for sent in r["answer"].split(". "):
            if sent.strip():
                assert "[st_" in sent or "нет данных" in sent
