"""v0.12 property-based tests (hypothesis, derandomized for CI stability).

Seed is fixed via derandomize=True; minimal counter-examples are printed
by hypothesis on failure (no shrinking budget games).
"""
import os

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from rasm_cim_v12.knowledge import KnowledgeStore, teach, retract

SETTINGS = dict(max_examples=25, derandomize=True,
                suppress_health_check=list(HealthCheck))

name_st = st.sampled_from(["Франция", "Германия", "Япония", "Италия", "Испания",
                           "Канада", "Бразилия", "Австралия", "Польша", "Португалия"])
obj_st = st.sampled_from(["Париж", "Берлин", "Токио", "Рим", "Мадрид",
                          "Оттава", "Бразилиа", "Канберра", "Варшава", "Лиссабон"])


@given(pairs=st.lists(st.tuples(name_st, obj_st), min_size=1, max_size=12))
@settings(**SETTINGS)
def test_add_then_query_true(tmp_path_factory, pairs):
    kb = KnowledgeStore.init(str(tmp_path_factory.mktemp("kb")))
    seen = set()
    for subj, obj in pairs:
        teach(kb, subj, "capital_of", obj, created_at="2026-11-26T00:00:00Z")
        seen.add((subj, obj))
    for subj, obj in seen:
        st_ids = kb.query_statements(predicate="capital_of")
        assert st_ids


@given(seed_pairs=st.lists(st.tuples(name_st, obj_st), min_size=2, max_size=8, unique=True))
@settings(**SETTINGS)
def test_reimport_twice_is_idempotent(tmp_path_factory, seed_pairs):
    from rasm_cim_v12.knowledge import export_jsonl, import_jsonl
    src_path = str(tmp_path_factory.mktemp("src"))
    kb = KnowledgeStore.init(src_path)
    for subj, obj in seed_pairs:
        teach(kb, subj, "capital_of", obj, created_at="2026-11-26T00:00:00Z")
    exp = str(src_path + "/export.jsonl")
    export_jsonl(kb, exp)
    dst = KnowledgeStore.init(str(tmp_path_factory.mktemp("dst")))
    r1 = import_jsonl(dst, exp)
    n1 = dst.stats()["statements_total"]
    r2 = import_jsonl(dst, exp)
    assert r1["imported"] >= n1 and n1 == len(seed_pairs)
    assert r2["imported"] == 0
    assert dst.stats()["statements_total"] == n1


@given(pairs=st.lists(st.tuples(name_st, obj_st), min_size=1, max_size=8))
@settings(**SETTINGS)
def test_assert_retract_cycle_ends_unknown(tmp_path_factory, pairs):
    kb = KnowledgeStore.init(str(tmp_path_factory.mktemp("kb")))
    made = []
    for subj, obj in pairs:
        r = teach(kb, subj, "capital_of", obj, created_at="2026-11-26T00:00:00Z")
        made.append((r, subj, obj))
    for r, subj, obj in made:
        retract(kb, r["statement_id"], reason="property-test")
    for r, subj, obj in made:
        assert kb.status_for(r["subject_id"], "capital_of",
                             object_entity=r["object_id"]) == "UNKNOWN"


@given(n=st.integers(min_value=1, max_value=40))
@settings(**SETTINGS)
def test_prefix_replay_never_crashes(tmp_path_factory, n):
    """Interrupted write ≈ any prefix of the event log; replay must survive."""
    import json
    path = str(tmp_path_factory.mktemp("kb"))
    kb = KnowledgeStore.init(path)
    base_events = 0
    for i, subj in enumerate(["Франция", "Германия", "Япония", "Италия"]):
        teach(kb, subj, "capital_of", "Париж",
              created_at="2026-11-26T00:00:00Z")   # distinct subjects, same object
    raw = open(os.path.join(path, "events.jsonl"), "rb").read()
    lines = [l for l in raw.split(b"\n") if l.strip()]
    cut = lines[: max(1, min(n, len(lines)))]
    p2 = str(tmp_path_factory.mktemp("kb_prefix"))
    os.makedirs(p2, exist_ok=True)
    import shutil
    shutil.copy(os.path.join(path, "meta.json"), os.path.join(p2, "meta.json"))
    with open(os.path.join(p2, "events.jsonl"), "wb") as fh:
        fh.write(b"\n".join(cut) + b"\n")
    kb2 = KnowledgeStore.init(p2)
    assert kb2.stats()["events"] == len(cut)


@given(flips=st.lists(st.booleans(), min_size=1, max_size=10))
@settings(**SETTINGS)
def test_polarity_conflicts_detected_then_resolved(tmp_path_factory, flips):
    kb = KnowledgeStore.init(str(tmp_path_factory.mktemp("kb")))
    r1 = teach(kb, "Дракон", "существует", "да", object_is_entity=False,
               polarity="assert", created_at="2026-11-26T00:00:00Z")
    r2 = teach(kb, "Дракон", "существует", "да", object_is_entity=False,
               polarity="deny", created_at="2026-11-26T00:00:01Z")
    assert kb.status_for(r1["subject_id"], "существует", object_literal="да") == "CONFLICT"
    for sid in (r1["statement_id"], r2["statement_id"]):
        retract(kb, sid, reason="property-test")
    assert kb.status_for(r1["subject_id"], "существует", object_literal="да") == "UNKNOWN"
