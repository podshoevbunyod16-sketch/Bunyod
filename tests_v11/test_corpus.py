"""Corpus integrity + CLI tests for v0.11.

Leakage policy enforced here as tests (not just docs):
  * no normalized text appears in two splits;
  * no template family appears in two splits (paraphrase families are split
    as units, per spec);
  * schema fields present; ids unique.
"""
import json
import os
import subprocess
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS = os.path.join(ROOT, "rasm_cim_v11", "data", "intent_corpus.jsonl")

sys.path.insert(0, ROOT)
from rasm_cim_v11.language import normalize  # noqa: E402


def _rows():
    return [json.loads(l) for l in open(CORPUS, encoding="utf-8") if l.strip()]


def test_corpus_schema_and_unique_ids():
    rows = _rows()
    assert len(rows) >= 100
    ids = [r["id"] for r in rows]
    assert len(ids) == len(set(ids))
    required = {"id", "text", "intent", "template", "split", "source", "schema"}
    for r in rows:
        assert required <= set(r)
        assert r["split"] in ("train", "dev", "test")
        assert r["schema"] == "rasm-cim-corpus/0"


def test_corpus_no_cross_split_duplicates():
    rows = _rows()
    seen = {}
    for r in rows:
        key = normalize(r["text"]).text
        assert key not in seen, f"duplicate across splits: {key!r} ({seen.get(key)} vs {r['split']})"
        seen[key] = r["split"]


def test_corpus_template_families_disjoint_per_split():
    rows = _rows()
    fam_split = {}
    for r in rows:
        prev = fam_split.setdefault(r["template"], r["split"])
        assert prev == r["split"], f"template {r['template']} in two splits"


def test_corpus_intents_supported():
    rows = _rows()
    assert {r["intent"] for r in rows} <= {
        "greeting", "ask_fact", "ask_definition", "yes_no", "compare", "list", "unknown"}


def test_cli_language_check_json():
    r = subprocess.run([sys.executable, "-m", "rasm_cim_v11.cli",
                        "language-check", "Какая столица у Франции?"],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0
    out = json.loads(r.stdout)
    assert out["intent"] == "ask_fact"
    assert out["tokens"] and out["original"] == "Какая столица у Франции?"


def test_cli_ask_text_capital():
    r = subprocess.run([sys.executable, "-m", "rasm_cim_v11.cli",
                        "ask-text", "Какая столица у Германии?"],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0
    out = json.loads(r.stdout)
    assert out["status"] == "TRUE" and "Берлин" in out["answer"]


def test_cli_language_check_offsets_point_into_original():
    text = "Где находится Эрмитаж?"
    r = subprocess.run([sys.executable, "-m", "rasm_cim_v11.cli",
                        "language-check", text],
                       capture_output=True, text=True, cwd=ROOT)
    out = json.loads(r.stdout)
    for t in out["tokens"]:
        assert text[t["start"]:t["end"]] == t["text"]
