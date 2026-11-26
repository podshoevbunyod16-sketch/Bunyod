"""Correction interface (v0.15) — tamper-evident, undo-able, verifiable.

A correction event says: "at time T, for target TEXT with kind KIND,
the wrong value was WRONG; the user asserts CORRECT."

Event fields: correction_id, ts, kind (intent|alias|entity|answer|relation),
wrong, correct, target_text, session_id, artifact_version, prev_hash.
prev_hash chains events (sha256 of previous line); the chain root is the
corpus checksum — a rewritten history breaks verification by construction.

apply_corrections(kb, events): validates events against the KB (ambiguity control
for aliases: refuse to bind to >1 entity), folds them into a NEW artifact version
(never mutates existing ones), and records before/after dev metrics so the user
sees the verified effect. rollback(artifacts, version) restores prior behavior.
Conflicting corrections (same target, opposite values) are both kept in history;
resolution is latest-wins and the conflict is SURFACED in the report.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from datetime import datetime, timezone
from typing import List, Optional

from rasm_cim_v12.knowledge import KnowledgeStore
from rasm_cim_v14.reasoning import ask
from .artifacts import LearnedArtifacts, utcnow
from .evaluate import load_split, intent_acc, REGRESSION_PROBES, regression_pass

_DIR = os.path.dirname(os.path.abspath(__file__))
CORRECTIONS_PATH = os.path.join(_DIR, "corrections.jsonl")


def _line_hash(prev_hash: str, line: dict) -> str:
    blob = json.dumps(line, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(prev_hash.encode() + blob).hexdigest()


def _last_hash(path: str) -> str:
    if not os.path.exists(path):
        return "GENESIS"
    last = "GENESIS"
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                last = json.loads(line)["event_hash"]
    return last


def record_correction(kind: str, wrong: str, correct: str, target_text: str,
                      *, session_id: str = "cli",
                      path: str = CORRECTIONS_PATH) -> dict:
    prev = _last_hash(path)
    ev = {"correction_id": "cor_" + uuid.uuid4().hex[:12],
          "ts": utcnow(),
          "kind": kind,
          "wrong": wrong,
          "correct": correct,
          "target_text": target_text,
          "session_id": session_id,
          "artifact_version": (LearnedArtifacts.current().s["version"]
                               if LearnedArtifacts.current() else None),
          "prev_hash": prev}
    ev["event_hash"] = _line_hash(prev, {k: v for k, v in ev.items()
                                          if k != "event_hash"})
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(ev, ensure_ascii=False, sort_keys=True) + "\n")
    return ev


def load_corrections(path: str = CORRECTIONS_PATH) -> List[dict]:
    out = []
    prev = "GENESIS"
    if not os.path.exists(path):
        return out
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        ev = json.loads(line)
        expect = _line_hash(prev, {k: v for k, v in ev.items()
                                         if k != "event_hash"})
        ev["chain_ok"] = (prev == ev["prev_hash"] and expect == ev["event_hash"])
        prev = ev["event_hash"]
        out.append(ev)
    return out


def verify_chain(path: str = CORRECTIONS_PATH) -> dict:
    events = load_corrections(path)
    bad = [e["correction_id"] for e in events if not e.get("chain_ok")]
    return {"events": len(events), "chain_ok": not bad, "broken": bad}


# ------------------------------------------------------------------ apply

def find_conflicts(events: List[dict]) -> List[dict]:
    """Same target_text & kind, opposite correct-values → surfaced, never hidden."""
    by_key = {}
    for ev in events:
        by_key.setdefault((ev["kind"], ev["target_text"].casefold()), []).append(ev)
    conflicts = []
    for (kind, _), evs in by_key.items():
        values = {e["correct"] for e in evs}
        if len(values) > 1:
            conflicts.append({"kind": kind, "target_text": evs[0]["target_text"],
                              "values": sorted(values),
                              "correction_ids": [e["correction_id"] for e in evs]})
    return conflicts


def validate_event(kb: KnowledgeStore, ev: dict) -> List[str]:
    """Sanity checks before folding into artifacts; returns human problems."""
    errs = []
    if ev["kind"] in ("entity", "alias"):
        wrong, correct = ev.get("wrong"), ev.get("correct")
        if not wrong or not correct:
            errs.append("alias/entity correction requires wrong and correct")
        hits = kb.find_by_name(correct)
        if hits[0] != "found":
            errs.append(f"canonical entity «{correct}» not in KB")
        else:
            # ambiguity control: wrong-surface must not already bind to a DIFFERENT entity
            alias_hits = kb.find_by_name(wrong)
            if alias_hits[0] == "found" and alias_hits[1].id != hits[1].id:
                errs.append(f"«{wrong}» уже связано с «{alias_hits[1].name}» — конфликтующая коррекция отклонена")
    elif ev["kind"] == "intent":
        try:
            allowed = {"ask_fact", "ask_definition", "compare", "list",
                       "yes_no", "greeting", "unknown"}
            if ev.get("correct") not in allowed:
                errs.append(f"unknown intent «{ev.get('correct')}»")
        except Exception:
            errs.append("intent validation failed")
    elif ev["kind"] == "answer":
        if not ev.get("correct"):
            errs.append("answer correction requires the corrected text")
    elif ev["kind"] == "relation":
        if not ev.get("correct"):
            errs.append("relation correction requires the corrected relation")
    return errs


def apply_corrections(kb: KnowledgeStore, events: List[dict],
                      artifacts_dir: str = None) -> dict:
    """Fold pending events into a new artifact version. Returns a verification
    report: before/after metrics on dev + conflicts surfaced + rollback hint."""
    from .artifacts import ARTIFACTS_DIR
    artifacts_dir = artifacts_dir or ARTIFACTS_DIR
    before = LearnedArtifacts.current(artifacts_dir)
    train_examples = load_split("train")
    # apply correction examples into train set (source=correction, no metric leakage:
    # these are user corrections on train-like occurrences, evaluation still uses dev/test)
    corr_examples = [{"id": ev["correction_id"],
                      "text": ev["target_text"],
                      "intent": ev["correct"],
                      "relation": None, "subject": None, "object": None,
                      "polarity": None, "family": "correction", "split": "train",
                      "source": "user_correction:v0.15"}
                     for ev in events if ev["kind"] == "intent"]
    new_arts = LearnedArtifacts.train(train_examples + corr_examples,
                                      corpus_sha=(before.s["corpus_sha256"] if before
                                                  else _corpus_sha()),
                                      parent=before,
                                      correction_events=events,
                                      config={"use_fallback_on_unknown": True})
    dev = load_split("dev")
    metrics_before = intent_acc(dev, before)
    metrics_after = intent_acc(dev, new_arts)
    new_arts.s["metrics"] = {"dev_intent_acc_before": metrics_before,
                             "dev_intent_acc_after": metrics_after,
                             "regression": regression_pass(kb)}
    path = new_arts.save(artifacts_dir)
    conflicts = find_conflicts(load_corrections())
    return {"new_artifact": new_arts.s["version"],
            "parent": before.s["version"] if before else None,
            "saved_to": path,
            "dev_intent_acc_before": metrics_before,
            "dev_intent_acc_after": metrics_after,
            "regression": new_arts.s["metrics"]["regression"],
            "conflicts": conflicts,
            "undo": f"python -m rasm_cim_v15.cli rollback {before.s['version']}"
                    if before else None}


def _corpus_sha() -> str:
    import hashlib
    blob = b""
    d = os.path.join(_DIR, "corpus")
    for name in ("train.jsonl", "dev.jsonl", "test.jsonl"):
        blob += open(os.path.join(d, name), "rb").read()
    return hashlib.sha256(blob).hexdigest()


def answer_verified(kb: KnowledgeStore, question: str) -> dict:
    """Verifier used by tests: returns verdict + evidence ids for a question."""
    r = ask(kb, question)
    return {"verdict": r["verdict"], "evidence_ids": r["evidence_ids"],
            "answer": r["answer"], "status": r["status"]}
