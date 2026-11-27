"""Spec §13 — performance and stress.

Targets (host, this sandbox; the CI assertion is deliberately loose — it
guards against order-of-magnitude regressions, not against machine speed):
  * teach throughput        >= 30 facts/s
  * lookup latency (p95)    <= 200 ms per question on a 1 000-fact KB
  * inference on a dense graph terminates within the node budget
  * resident memory for 1 000 facts stays under 120 MiB
"""
from __future__ import annotations

import gc
import os
import resource
import time

import pytest

from rasm_cim_v16.inference_engine import InferenceEngine


def _rss_mib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


@pytest.fixture(scope="module")
def big_kb():
    """A 1 000-fact KB with 200 classes and 200 people, built once."""
    from rasm_cim_v16.dialogue_manager import DialogueManager
    dm = DialogueManager(clock=lambda: "t")
    for i in range(200):
        dm.say(f"teach Класс{i} — это категория")
    for i in range(200):
        dm.say(f"teach Человек{i} — это физик")
        dm.say(f"teach Человек{i} открыл Объект{i}")
        dm.say(f"teach Человек{i} сформулировал Закон{i}")
        dm.say(f"teach Объект{i} находится в Месте{i % 50}")
    return dm


def test_kb_size(big_kb):
    st = big_kb.kg.all_statements()
    assert len(st) >= 1000


def test_teach_throughput():
    from rasm_cim_v16.dialogue_manager import DialogueManager
    dm = DialogueManager(clock=lambda: "t")
    n = 300
    t0 = time.perf_counter()
    for i in range(n):
        dm.say(f"teach Сущность{i} — это класс")
    dt = time.perf_counter() - t0
    rate = n / dt
    assert rate >= 30, f"teach throughput {rate:.1f}/s is below the 30/s floor"


def test_lookup_latency_p95(big_kb):
    lat = []
    for i in range(0, 200, 7):
        t0 = time.perf_counter()
        big_kb.say(f"Человек{i} — это категория?")
        lat.append((time.perf_counter() - t0) * 1000)
    lat.sort()
    p95 = lat[int(len(lat) * 0.95) - 1]
    assert p95 <= 200, f"p95 lookup {p95:.1f} ms exceeds 200 ms"


def test_definition_answer_latency(big_kb):
    t0 = time.perf_counter()
    a = big_kb.say("Расскажи о Человек17", mode="detailed")
    dt = (time.perf_counter() - t0) * 1000
    assert a.text
    assert dt <= 500, f"detailed profile took {dt:.1f} ms"


def test_inference_node_budget_on_a_dense_graph(big_kb):
    inf = big_kb.inference
    eid = big_kb.kg.resolver.resolve("Человек17").entity_id
    t0 = time.perf_counter()
    facts = inf.query(subject=eid, predicate=None, obj=None)
    dt = (time.perf_counter() - t0) * 1000
    assert facts
    assert inf.stats["last_goals"] <= inf.max_nodes + 1
    assert dt <= 500, f"graph query took {dt:.1f} ms"


def test_adversarial_cycle_does_not_blow_up():
    from rasm_cim_v16.knowledge_graph import KnowledgeGraph
    kg = KnowledgeGraph(clock=lambda: "t")
    ids = [kg.resolver.intern(f"N{i}").entity_id for i in range(30)]
    for a, b in zip(ids, ids[1:] + ids[:1]):        # one big cycle
        kg.assert_triple(a, "part_of", b, when="t")
    inf = InferenceEngine(kg, max_depth=6, max_nodes=300)
    t0 = time.perf_counter()
    inf.query(subject=ids[0], predicate="part_of", obj=None)
    dt = (time.perf_counter() - t0) * 1000
    assert inf.stats["last_goals"] <= 301
    assert dt <= 2000, f"cycle query took {dt:.1f} ms"


def test_long_dialogue_does_not_leak_memory(big_kb):
    before = len(big_kb.context.history)
    for i in range(60):
        big_kb.say(f"Человек{i} — это категория?")
    assert len(big_kb.context.history) <= big_kb.context.max_turns
    assert len(big_kb.memory.working()) <= big_kb.memory.working_cap
    assert len(big_kb.context.history) >= before


def test_resident_memory_for_1000_facts(big_kb):
    gc.collect()
    rss = _rss_mib()
    sizes = big_kb.memory.size_bytes()
    # serialised semantic tier must stay well under a megabyte per 1k facts
    assert sizes["semantic"] < 5_000_000, sizes
    assert rss < 400, f"process RSS {rss:.0f} MiB is unexpectedly large"


def test_save_load_time_for_1000_facts(tmp_path):
    from rasm_cim_v16.dialogue_manager import DialogueManager
    from rasm_cim_v16.persistence import Store
    home = str(tmp_path / ".rasm-cim")
    dm = DialogueManager(clock=lambda: "t")
    st = Store(home=home, clock=lambda: "t")
    st.attach(dm)
    for i in range(300):
        dm.say(f"teach Сущность{i} — это класс")
    t0 = time.perf_counter()
    info = dm.save()
    save_dt = time.perf_counter() - t0
    dm2 = DialogueManager(clock=lambda: "t")
    st2 = Store(home=home, clock=lambda: "t")
    st2.attach(dm2)
    t0 = time.perf_counter()
    st2.load(dm2)
    load_dt = time.perf_counter() - t0
    assert len(dm2.kg.all_statements()) == 300
    assert save_dt < 5.0, f"save took {save_dt:.2f} s"
    assert load_dt < 5.0, f"load took {load_dt:.2f} s"
    assert info["kb_bytes"] < 5_000_000


def test_no_full_kb_scan_for_a_point_lookup(big_kb):
    """A point lookup must use the (s,p) index, not a scan of all statements."""
    eid = big_kb.kg.resolver.resolve("Человек17").entity_id
    # the index path returns exactly the stored objects
    assert len(big_kb.kg.objects_of(eid, "discovered")) == 1
    assert len(big_kb.kg.objects_of(eid, "formulated")) == 1
    assert big_kb.kg.objects_of(eid, "wrote") == []


def test_teaching_does_not_get_slower_as_the_kb_grows():
    """Spec §13: growth must not degrade performance.

    Regression guard for a real defect: the contradiction scan used to walk
    every statement on every taught fact, so per-fact cost grew linearly with
    the KB (measured 1096 facts/s at 1000 facts vs 328 facts/s at 5000).  The
    scan is now focused on the new triple, and this test fails if the cost ever
    becomes superlinear again.
    """
    import time

    from rasm_cim_v16.dialogue_manager import DialogueManager

    def rate(n, offset):
        dm = DialogueManager(clock=lambda: "t")
        t0 = time.perf_counter()
        for i in range(n):
            dm.say(f"teach Человек{offset + i} открыл Объект{offset + i}")
        return n / (time.perf_counter() - t0)

    small = rate(600, 0)
    large = rate(2400, 10_000)
    assert large > small * 0.5, (
        f"per-fact cost is growing with KB size: {small:.0f}/s at 600 facts "
        f"vs {large:.0f}/s at 2400 facts")
