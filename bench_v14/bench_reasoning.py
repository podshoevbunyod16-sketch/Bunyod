"""v0.14 bench: latency (p50/p95/p99), peak RSS, evidence-bound accuracy metrics.

Fixed fixture, fixed question set, fixed machine (Linux host, tmpfs).
Run: python bench_v14/bench_reasoning.py
"""
import json
import os
import resource
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rasm_cim_v12.knowledge import KnowledgeStore, teach, Provenance
from rasm_cim_v14.reasoning import ask

FIX = "/tmp/bench_kb_v14"
os.system(f"rm -rf {FIX}")
kb = KnowledgeStore.init(FIX)
prov = Provenance(source="bench_fixture", doc_id="bench_v14", fragment="fixture",
                  method="manual", parser_version="bench-0.14.0")

teach(kb, "Франция", "capital_of", "Париж", subject_type="place", object_type="place", provenance=prov)
teach(kb, "Франция", "located_in", "Европа", subject_type="place", object_type="concept", provenance=prov)
teach(kb, "Киты", "is_a", "млекопитающее", subject_type="animal", object_type="concept", provenance=prov)
teach(kb, "млекопитающее", "is_a", "животное", subject_type="concept", object_type="concept", provenance=prov)
teach(kb, "животное", "is_a", "живое существо", subject_type="concept", object_type="concept", provenance=prov)
teach(kb, "Пингвин", "может_летать", "да", object_is_entity=False, subject_type="animal", polarity="deny", provenance=prov)
teach(kb, "Дракон", "существует", "да", object_is_entity=False, subject_type="animal", polarity="assert", provenance=prov)
teach(kb, "Дракон", "существует", "да", object_is_entity=False, subject_type="animal", polarity="deny", provenance=prov)
teach(kb, "Венера", "is_a", "планета", subject_type="thing", object_type="concept", kind="hypothesis", provenance=prov)
teach(kb, "Франция", "population_of", "68 000 000", object_is_entity=False, subject_type="place", provenance=prov)
teach(kb, "Германия", "population_of", "84 000 000", object_is_entity=False, subject_type="place", provenance=prov)

# (question, expected_verdict, expects_evidence, expects_abstain)
LABELED = [
    ("Какая столица у Франции?", "TRUE", True, False),
    ("Что такое Киты?", "TRUE", True, False),
    ("Где расположен Париж?", "TRUE", True, False),
    ("Кто написал несуществующее произведение?", "UNKNOWN", False, True),
    ("Умеет ли пингвин летать?", "FALSE", True, False),
    ("Какая столица у Атлантиды?", "UNKNOWN", False, True),
    ("Правда ли что дракон существует?", "CONFLICT", True, False),
    ("Что такое Венера?", "UNKNOWN", False, True),
    ("У кого население больше: Франция или Германия?", "TRUE", True, False),
    ("asdfgh jkl", "UNKNOWN", False, True),
    ("Что такое Голубица?", "UNKNOWN", False, True),
]
LAT_Q = ["Какая столица у Франции?", "Что такое Киты?",
         "Где расположен Париж?", "Правда ли что дракон существует?",
         "У кого население больше: Франция или Германия?"]

def metric_block():
    n = len(LABELED)
    correct = sum(1 for q, ev, _, _ in LABELED
                  for r in [ask(kb, q)] if r["verdict"] == ev)
    unsupported = sum(1 for q, _, _, _ in LABELED
                      for r in [ask(kb, q)]
                      if r["evidence_ids"] == [] and r["verdict"] not in ("UNKNOWN",))
    abstain_ok = sum(1 for q, _, _, ab in LABELED
                     for r in [ask(kb, q)]
                     if (r["verdict"] == "UNKNOWN") == ab or not ab)
    abstain_total = sum(1 for _, _, _, ab in LABELED if ab)
    abstain_correct = sum(1 for q, _, _, ab in LABELED
                          for r in [ask(kb, q)]
                          if ab and r["verdict"] == "UNKNOWN")
    return {
        "answer_accuracy": round(correct / n, 3),
        "unsupported_answer_rate": round(unsupported / n, 3),
        "abstention_accuracy": round(abstain_correct / max(1, abstain_total), 3),
        "labeled_questions": n,
    }

def latency_block():
    out = {}
    for q in LAT_Q:
        for _ in range(30):
            ask(kb, q)
        ts = []
        for _ in range(300):
            t0 = time.perf_counter()
            ask(kb, q)
            ts.append((time.perf_counter() - t0) * 1e6)
        ts.sort()
        out[q] = {"n": len(ts),
                  "p50_us": round(ts[len(ts)//2], 1),
                  "p95_us": round(ts[int(len(ts)*.95)], 1),
                  "p99_us": round(ts[int(len(ts)*.99)], 1)}
    return out

rss_mb = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)
payload = {
    "python": sys.version.split()[0], "machine": os.uname().machine,
    "method": "300 reps after 30 warm, fixed fixture KB, tmpfs host; not measured on device",
    "latency": latency_block(),
    "quality": metric_block(),
    "peak_rss_mb_process": rss_mb,
}
os.makedirs("results_v14", exist_ok=True)
with open("results_v14/bench_reasoning.json", "w", encoding="utf-8") as fh:
    json.dump(payload, fh, ensure_ascii=False, indent=2)
print(json.dumps(payload["quality"], ensure_ascii=False, indent=2))
for k, v in payload["latency"].items():
    print(f"  p50 {v['p50_us']}us  p95 {v['p95_us']}us  p99 {v['p99_us']}us  {k}")
print(f"peak RSS {rss_mb} MB")
