"""v0.11 intent classification benchmark on the committed corpus.

Uses the REAL pipeline (normalize → classify_intent). Writes
results_v11/intent_eval.json with per-split accuracy, confusion matrix and
the full error list. Never used to tune rules; numbers reported as-is.

Run: python3 bench_v11/eval_intent.py
"""
import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rasm_cim_v11.language import normalize, classify_intent

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS = os.path.join(ROOT, "rasm_cim_v11", "data", "intent_corpus.jsonl")
OUT = os.path.join(ROOT, "results_v11", "intent_eval.json")

INTENTS = ["greeting", "ask_fact", "ask_definition", "yes_no", "compare", "list", "unknown"]


def main():
    rows = [json.loads(l) for l in open(CORPUS, encoding="utf-8") if l.strip()]
    report = {"corpus": "rasm_cim_v11/data/intent_corpus.jsonl",
              "n_examples": len(rows),
              "by_split": {}, "errors": []}
    overall_ok = 0
    for split in ("train", "dev", "test"):
        sub = [r for r in rows if r["split"] == split]
        per_class = defaultdict(lambda: [0, 0])   # cls -> [correct, total]
        confusion = Counter()
        ok = 0
        for r in sub:
            norm = normalize(r["text"])
            pred = classify_intent(norm.text, norm.has_cyrillic).intent
            per_class[r["intent"]][1] += 1
            confusion[(r["intent"], pred)] += 1
            if pred == r["intent"]:
                ok += 1
                per_class[r["intent"]][0] += 1
            else:
                report["errors"].append({"id": r["id"], "text": r["text"],
                                          "expected": r["intent"], "predicted": pred})
        report["by_split"][split] = {
            "n": len(sub), "correct": ok,
            "accuracy": round(ok / len(sub), 4),
            "per_class": {c: {"n": t, "correct": k,
                              "accuracy": round(k / t, 4) if t else None}
                          for c, (k, t) in sorted(per_class.items())},
            "confusion": {f"{a}->{b}": n for (a, b), n in sorted(confusion.items())},
        }
        overall_ok += ok
    report["overall"] = {"n": len(rows), "correct": overall_ok,
                          "accuracy": round(overall_ok / len(rows), 4)}
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    print(json.dumps({s: report["by_split"][s]["accuracy"] for s in ("train", "dev", "test")},
                     ensure_ascii=False), "overall", report["overall"]["accuracy"],
          "errors", len(report["errors"]))


if __name__ == "__main__":
    main()
