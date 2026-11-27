"""Regression: _find_entities re-stemmed EVERY entity/alias on EVERY ask (O(N)
morphology per query — ~19 ms @ +1000 facts, ~81 ms @ +5000 facts measured on
host during release verification; maxsize sized for ~30k distinct stem keys). _stems is pure (normalize+lemma), so it is
now lru_cached. This test locks the cache behavior without timing asserts."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from rasm_cim_v14.reasoning import pipeline as P


def test_stems_is_cached():
    a = P._stems("Какая столица у Франции?")
    b = P._stems("Какая столица у Франции?")
    assert a == b
    assert a is b, "_stems must return the cached object for identical input"
    assert P._stems.cache_info().hits >= 1


def test_find_entities_same_result_twice(tmp_path):
    from rasm_cim_v12.knowledge import KnowledgeStore, teach, Provenance
    PROV = Provenance(source="t", doc_id="t", fragment="t",
                      method="manual", parser_version="t")
    kb = KnowledgeStore.init(str(tmp_path / "kb"))
    teach(kb, "Франция", "capital_of", "Париж", subject_type="place",
          object_type="place", provenance=PROV)
    r1 = P._find_entities(kb, "Какая столица у Франции?")
    r2 = P._find_entities(kb, "Какая столица у Франции?")
    assert [d["id"] for d in r1] == [d["id"] for d in r2]
    assert r1 and r1[0]["name"] == "Франция"
