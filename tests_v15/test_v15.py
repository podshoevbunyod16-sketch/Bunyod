"""v0.15 tests — corpus splits, leakage control, artifacts, corrections, rollback."""
import json
import os
import shutil
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from rasm_cim_v12.knowledge import KnowledgeStore, teach, Provenance
from rasm_cim_v15 import corpus_build
from rasm_cim_v15.artifacts import LearnedArtifacts
from rasm_cim_v15 import corrections as C
from rasm_cim_v15.pipeline15 import ask_v15, resolve_intent

PROV = Provenance(source="tests_v15", doc_id="tests_v15", fragment="test",
                  method="manual", parser_version="tests-0.15.0")


@pytest.fixture()
def kb(tmp_path):
    store = KnowledgeStore.init(str(tmp_path / "kb"))
    teach(store, "Франция", "capital_of", "Париж", subject_type="place",
          object_type="place", provenance=PROV)
    teach(store, "Германия", "capital_of", "Берлин", subject_type="place",
          object_type="place", provenance=PROV)
    return store


@pytest.fixture()
def arts_dir(tmp_path):
    d = tmp_path / "learned"
    d.mkdir()
    yield str(d)


def _split_rows(name):
    path = os.path.join(corpus_build.CORPUS_DIR, f"{name}.jsonl")
    return [json.loads(l) for l in open(path, encoding="utf-8")]


class TestCorpus:
    def test_counts_in_expected_ranges(self):
        for name, lo, hi in (("train", 200, 500), ("dev", 50, 100), ("test", 50, 100)):
            n = len(_split_rows(name))
            assert lo <= n <= hi, f"{name} has {n}"

    def test_no_family_straddles_splits(self):
        fams = {s: {r["family"] for r in _split_rows(s)} for s in ("train", "dev", "test")}
        assert fams["train"].isdisjoint(fams["dev"])
        assert fams["train"].isdisjoint(fams["test"])
        assert fams["dev"].isdisjoint(fams["test"])

    def test_no_identical_example_across_splits(self):
        texts = {s: {r["text"] for r in _split_rows(s)} for s in ("train", "dev", "test")}
        assert texts["train"].isdisjoint(texts["dev"])
        assert texts["train"].isdisjoint(texts["test"])
        assert texts["dev"].isdisjoint(texts["test"])

    def test_manifest_checksum_matches_files(self):
        manifest = json.load(open(os.path.join(corpus_build.CORPUS_DIR,
                                               "manifest.json"),
                                  encoding="utf-8"))
        import hashlib
        for split, meta in manifest["files"].items():
            data = open(os.path.join(corpus_build.CORPUS_DIR,
                                     f"{split}.jsonl"), "rb").read()
            assert hashlib.sha256(data).hexdigest() == meta["sha256"]

    def test_corpus_rebuild_is_deterministic(self, tmp_path):
        out = tmp_path / "rebuilt"
        m1 = corpus_build.write_corpus(str(out))
        manifest2 = json.load(open(os.path.join(corpus_build.CORPUS_DIR,
                                                "manifest.json"), encoding="utf-8"))
        assert m1["corpus_sha256"] == manifest2["corpus_sha256"]


class TestArtifacts:
    def test_train_load_roundtrip(self, arts_dir):
        train = _split_rows("train")
        arts = LearnedArtifacts.train(train, "deadbeef", parent=None)
        path = arts.save(arts_dir)
        loaded = LearnedArtifacts.load(arts.s["version"], arts_dir)
        assert loaded.s["intent_fallback"] == arts.s["intent_fallback"]
        assert loaded.s["corpus_sha256"] == "deadbeef"

    def test_train_deterministic_counts(self):
        train = _split_rows("train")
        a = LearnedArtifacts.train(train, "sha", parent=None)
        b = LearnedArtifacts.train(train, "sha", parent=None)
        assert a.s["intent_fallback"] == b.s["intent_fallback"]

    def test_save_does_not_touch_repo_pointer(self, arts_dir):
        # Regression: LearnedArtifacts.save(out_dir=tmp) must write current.json into
        # out_dir, not into the repo's ARTIFACTS_DIR — otherwise every test run leaks
        # a dangling lv_XXX pointer into the tracked rasm_cim_v15/learned/current.json.
        from rasm_cim_v15.artifacts import ARTIFACTS_DIR, CURRENT_POINTER
        before = open(CURRENT_POINTER, "rb").read() if os.path.exists(CURRENT_POINTER) else None
        train = _split_rows("train")
        arts = LearnedArtifacts.train(train, "pointer-leak-check", parent=None)
        arts.save(arts_dir)
        after = open(CURRENT_POINTER, "rb").read() if os.path.exists(CURRENT_POINTER) else None
        assert before == after, "repo rasm_cim_v15/learned/current.json was modified"
        assert os.path.exists(os.path.join(arts_dir, "current.json"))

    def test_rollback_restores_parent(self, arts_dir):
        train = _split_rows("train")
        base = LearnedArtifacts.train(train, "sha", parent=None)
        base.save(arts_dir)
        v1 = base.s["version"]
        child = LearnedArtifacts.train(train, "sha", parent=base,
                                       correction_events=[{
                                           "correction_id": "cor_test1",
                                           "kind": "relation",
                                           "wrong": "", "correct": "capital_of",
                                           "target_text": "Столицу франции зовут?",
                                       }])
        child.save(arts_dir)
        assert child.s["predicate_hints"]["столицу франции зовут?"] == "capital_of"
        rolled = LearnedArtifacts.rollback(v1, arts_dir)
        assert rolled.s["predicate_hints"] == base.s["predicate_hints"]
        assert LearnedArtifacts.current(arts_dir).s["version"] == v1


class TestCorrections:
    def test_chain_tamper_detected(self, tmp_path):
        path = str(tmp_path / "corr.jsonl")
        e1 = C.record_correction("intent", "", "ask_fact", "столица Франции",
                                 session_id="t", path=path)
        assert C.verify_chain(path)["chain_ok"]
        raw = open(path, encoding="utf-8").read()
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(raw.replace('"ask_fact"', '"list"', 1))
        assert not C.verify_chain(path)["chain_ok"]

    def test_relation_correction_e2e_and_undo(self, kb, tmp_path, monkeypatch):
        arts_dir = str(tmp_path / "learned")
        os.makedirs(arts_dir)
        train = _split_rows("train")
        before = LearnedArtifacts.train(train, "sha", parent=None)
        before.save(arts_dir)
        question = "Столицу Франции зовут?"     # нет предикативного ключа из _KW_PRED,
                                                # но содержит токен "столицу"…
        assert resolve_intent(question, before)[0] == "unknown"
        base_r = ask_v15(kb, question, artifacts=before)
        # baseline: predicate hint absent -> generic definition route or no fact
        corr_path = str(tmp_path / "corr.jsonl")
        ev = C.record_correction("relation", "", "capital_of", question,
                                 session_id="test", path=corr_path)
        monkeypatch.setattr(C, "CORRECTIONS_PATH", corr_path)
        report = C.apply_corrections(kb, [ev], artifacts_dir=arts_dir)
        after = LearnedArtifacts.current(arts_dir)
        assert after.s["version"] == report["new_artifact"]
        fixed = ask_v15(kb, question, artifacts=after)
        assert fixed["verdict"] == "TRUE"
        assert "Париж" in fixed["answer"]
        assert fixed["evidence_ids"], "corrected path must carry evidence ids"
        # undo: rollback
        LearnedArtifacts.rollback(before.s["version"], arts_dir)
        restored = ask_v15(kb, question, artifacts=LearnedArtifacts.current(arts_dir))
        assert restored["verdict"] == base_r["verdict"] == "UNKNOWN" or \
               (restored["verdict"], restored["evidence_ids"]) == \
               (base_r["verdict"], base_r["evidence_ids"])

    def test_conflicting_corrections_surfaced_not_hidden(self, kb, tmp_path, monkeypatch):
        corr_path = str(tmp_path / "corr.jsonl")
        monkeypatch.setattr(C, "CORRECTIONS_PATH", corr_path)
        q = "Проверочный вопрос о столицах?"
        C.record_correction("relation", "", "capital_of", q, session_id="t", path=corr_path)
        C.record_correction("relation", "", "located_in", q, session_id="t", path=corr_path)
        events = C.load_corrections(corr_path)
        conflicts = C.find_conflicts(events)
        assert len(conflicts) == 1
        assert sorted(conflicts[0]["correction_ids"]) == sorted(e["correction_id"] for e in events)

    def test_alias_correction_refused_on_ambiguity(self, kb, tmp_path):
        # Берлин already names the city entity; rebinding to Париж must be refused
        ev = {"correction_id": "cor_amb", "kind": "alias",
              "wrong": "Берлин", "correct": "Париж",
              "target_text": "Берлин — alias", "session_id": "t",
              "artifact_version": None, "prev_hash": "GENESIS",
              "event_hash": ""}
        errs = C.validate_event(kb, ev)
        assert any("уже связано" in e or "Paris" in e or "Берлин" in e for e in errs)

    def test_intent_correction_affects_fallback(self, kb, tmp_path, monkeypatch):
        corr_path = str(tmp_path / "corr.jsonl")
        monkeypatch.setattr(C, "CORRECTIONS_PATH", corr_path)
        arts_dir = str(tmp_path / "learned"); os.makedirs(arts_dir)
        train = _split_rows("train")
        base = LearnedArtifacts.train(train, "sha", parent=None); base.save(arts_dir)
        # user correction: this NOUN-ish phrase is a fact question
        ev = C.record_correction("intent", "unknown", "ask_fact",
                                 "крым гора столица",
                                 session_id="test", path=corr_path)
        report = C.apply_corrections(kb, [ev], artifacts_dir=arts_dir)
        cur = LearnedArtifacts.current(arts_dir)
        assert ev["correction_id"] in cur.s["corrections"]
        assert "dev_intent_acc_after" in cur.s["metrics"]

    def test_retrain_keeps_old_versions(self, arts_dir):
        train = _split_rows("train")
        base = LearnedArtifacts.train(train, "sha", parent=None); base.save(arts_dir)
        child = LearnedArtifacts.train(train, "sha", parent=base); child.save(arts_dir)
        assert os.path.exists(os.path.join(arts_dir, base.s["version"] + ".json"))
        assert os.path.exists(os.path.join(arts_dir, child.s["version"] + ".json"))
        assert child.s["parent"] == base.s["version"]
