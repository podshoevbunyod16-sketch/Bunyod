"""Learned artifacts (v0.15) — versioned, checksumed, rollback-able.

Schema: rasm-cim-learned/1

An artifact version = full snapshot of learnable state:
  intent_fallback  — naive token→intent counts trained on train+corrections
  predicate_hints  — normalized-question-substring → predicate (relation corrections)
  alias_delta      — user-corrected alias→canonical-name bindings (check ambiguity on insert)
  config           — alpha (Laplace), tau (fallback threshold), built from dev sweep
  corrections      — ids of correction events folded in
  provenance       — parent version, seed, corpus checksum, timestamps

Rollback = loading an earlier artifact file by its version id and re-basing the
`current` pointer file on it. History is never rewritten: versions persist.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from typing import List, Optional

LEARNED_SCHEMA = "rasm-cim-learned/1"
_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(_DIR, "learned")
CURRENT_POINTER = os.path.join(ARTIFACTS_DIR, "current.json")


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _norm_key(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").casefold().strip())


def _slug(obj) -> str:
    blob = json.dumps(obj, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def _token_counts_intent_totals(examples):
    tok = {}
    tot = {}
    for ex in examples:
        intent = ex["intent"]
        if intent == "unknown":
            continue  # don't teach the fallback "unknown" — absence IS unknown
        tot[intent] = tot.get(intent, 0) + 1
        for w in re.findall(r"[А-Яа-яЁёA-Za-z0-9]+", ex["text"].lower()):
            tok.setdefault(w, {}).setdefault(intent, 0)
            tok[w][intent] += 1
    return tok, tot


class LearnedArtifacts:
    """In-memory snapshot + persistence."""

    def __init__(self, snapshot: dict):
        self.s = snapshot

    # ------------------------------------------------------------ build
    @classmethod
    def baseline(cls, corpus_sha: str) -> "LearnedArtifacts":
        snap = {
            "schema": LEARNED_SCHEMA,
            "version": "lv_000",
            "parent": None,
            "created_at": utcnow(),
            "seed": 0,
            "corpus_sha256": corpus_sha,
            "config": {"alpha": 0.5, "tau": 0.60, "min_margin": 1.0,
                       "use_fallback_on_unknown": True},
            "intent_fallback": {"tokens": {}, "intent_totals": {}},
            "predicate_hints": {},
            "alias_delta": {},
            "corrections": [],
            "metrics": None,
        }
        return cls(snap)

    @classmethod
    def train(cls, train_examples, corpus_sha: str, *, parent: Optional["LearnedArtifacts"],
              correction_events: Optional[List[dict]] = None,
              config: Optional[dict] = None) -> "LearnedArtifacts":
        tok, tot = _token_counts_intent_totals(train_examples)
        cfg = {"alpha": 0.5, "tau": 0.60, "min_margin": 1.0,
               "use_fallback_on_unknown": True}
        if config:
            cfg.update(config)
        parent_s = parent.s if parent else None
        n = 0
        if parent_s:
            m = re.match(r"lv_(\d+)$", parent_s.get("version", "lv_000"))
            n = int(m.group(1)) + 1 if m else 1
        snap = {
            "schema": LEARNED_SCHEMA,
            "version": f"lv_{n:03d}",
            "parent": parent_s.get("version") if parent_s else None,
            "created_at": utcnow(),
            "seed": 0,
            "corpus_sha256": corpus_sha,
            "config": cfg,
            "intent_fallback": {"tokens": tok, "intent_totals": tot},
            "predicate_hints": dict(parent_s.get("predicate_hints", {})) if parent_s else {},
            "alias_delta": dict(parent_s.get("alias_delta", {})) if parent_s else {},
            "corrections": list(parent_s.get("corrections", [])) if parent_s else [],
            "metrics": None,
        }
        arts = cls(snap)
        if correction_events:
            for ev in correction_events:
                arts.fold_correction(ev)
        return arts

    def fold_correction(self, ev: dict) -> None:
        """Fold a correction event into the snapshot (idempotent by event id)."""
        cid = ev.get("correction_id")
        if not cid or cid in self.s["corrections"]:
            return
        kind = ev.get("kind")
        if kind == "alias":
            wrong, correct = ev.get("wrong"), ev.get("correct")
            if wrong and correct:
                self.s["alias_delta"][wrong.casefold().strip()] = correct.strip()
        elif kind == "intent":
            # synthetic train example folded into correction_intents; folded at
            # retrain time by apply_corrections() into the fallback's counts
            self.s.setdefault("correction_intents", {})
            self.s["correction_intents"][ev.get("target_text", "").casefold()] = ev.get("correct")
        elif kind == "relation":
            # e.g. target_text "Столицу Франции зовут?", correct="capital_of":
            # exact normalized question → predicate routing hint (see pipeline15)
            key = _norm_key(ev.get("target_text", ""))
            if key and ev.get("correct"):
                self.s["predicate_hints"][key] = ev["correct"].strip()
        self.s["corrections"].append(cid)

    # ------------------------------------------------------------ persist
    def artifact_id(self) -> str:
        return self.s["version"]

    def content_hash(self) -> str:
        return hashlib.sha256(
            json.dumps(self.s, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":")).encode("utf-8")).hexdigest()

    def save(self, out_dir: str = ARTIFACTS_DIR) -> str:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"{self.s['version']}.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.s, fh, ensure_ascii=False, indent=2)
        with open(CURRENT_POINTER, "w", encoding="utf-8") as fh:
            json.dump({"current": self.s["version"], "sha256":
                       hashlib.sha256(open(path, "rb").read()).hexdigest()},
                      fh)
        return path

    # ------------------------------------------------------------ load
    @classmethod
    def load(cls, version: str, out_dir: str = ARTIFACTS_DIR) -> "LearnedArtifacts":
        path = os.path.join(out_dir, f"{version}.json")
        with open(path, encoding="utf-8") as fh:
            return cls(json.load(fh))

    @classmethod
    def current(cls, out_dir: str = ARTIFACTS_DIR) -> Optional["LearnedArtifacts"]:
        if not os.path.exists(CURRENT_POINTER):
            cands = sorted(f for f in os.listdir(out_dir)
                           if re.fullmatch(r"lv_\d+\.json", f)) if os.path.isdir(out_dir) else []
            return cls.load(cands[-1][:-5], out_dir) if cands else None
        with open(CURRENT_POINTER, encoding="utf-8") as fh:
            cur = json.load(fh)
        return cls.load(cur["current"], out_dir)

    @classmethod
    def rollback(cls, version: str, out_dir: str = ARTIFACTS_DIR) -> "LearnedArtifacts":
        """Re-base the `current` pointer on an earlier version; history preserved."""
        target = cls.load(version, out_dir)
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"{version}.json")
        with open(CURRENT_POINTER, "w", encoding="utf-8") as fh:
            json.dump({"current": version,
                       "sha256": hashlib.sha256(open(path, "rb").read()).hexdigest()},
                      fh)
        return target

    # ------------------------------------------------------- inference
    def intent_scores(self, text: str) -> dict:
        import math
        tok = self.s["intent_fallback"]["tokens"]
        tot = self.s["intent_fallback"]["intent_totals"]
        if not tot:
            return {}
        alpha = self.s["config"]["alpha"]
        vocab = len(tok) + 1
        intents = list(tot)
        total_tokens = {i: sum(tv.get(i, 0) for tv in tok.values()) for i in intents}
        logp = {i: math.log((tot[i] + alpha) / (sum(tot.values()) + alpha * len(intents)))
                for i in intents}
        for w in re.findall(r"[А-Яа-яЁёA-Za-z0-9]+", text.lower()):
            dist = tok.get(w)
            for i in intents:
                c = dist.get(i, 0) if dist else 0
                logp[i] += math.log((c + alpha) / (total_tokens[i] + alpha * vocab))
        return logp

    def predict_intent(self, text: str):
        logp = self.intent_scores(text)
        if not logp:
            return None, 0.0
        best = max(logp, key=logp.get)
        ordered = sorted(logp.values(), reverse=True)
        margin = (ordered[0] - ordered[1]) if len(ordered) > 1 else ordered[0]
        return best, round(margin, 3)

    def alias_of(self, name: str) -> Optional[str]:
        return self.s["alias_delta"].get(name.casefold().strip())

    def predicate_hint_for(self, normalized_text: str) -> Optional[str]:
        return self.s.get("predicate_hints", {}).get(_norm_key(normalized_text))
