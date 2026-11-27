"""Read a ``rasm-cim-learned/1`` artifact (v0.15) without executing v0.15 code.

The v0.15 learned artifact is a *data* file: token→intent counts, predicate
hints, an alias delta, a config and the metrics measured when it was built.
v0.17 reads it as data.  Two consequences that matter:

  * the artifact is accepted **only if** its sha256 matches the pointer
    ``current.json`` — a corrupted or hand-edited artifact is reported, never
    silently used;
  * the intent fallback is re-implemented here from the recorded counts with
    the same Laplace smoothing (alpha=0.5) as v0.15, so nothing from the
    previous package is imported at runtime and the numbers stay comparable.

Nothing in this module writes to the artifact directory: rollback of learned
state stays a v0.15 concern until Plan F gives v0.17 its own versioned store.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field

LEARNED_SCHEMA = "rasm-cim-learned/1"
POINTER = "current.json"
_VERSION_RE = re.compile(r"^lv_\d+\.json$")
_TOKEN_RE = re.compile(r"[А-Яа-яЁёA-Za-z0-9]+")


class LearnedFormatError(ValueError):
    pass


def norm_key(text: str) -> str:
    """Normalisation of a question key — identical to v0.15 ``_norm_key``."""
    return re.sub(r"\s+", " ", (text or "").casefold().strip())


@dataclass
class LearnedSnapshot:
    path: str
    version: str
    parent: str | None = None
    schema: str = LEARNED_SCHEMA
    config: dict = field(default_factory=dict)
    intent_fallback: dict = field(default_factory=dict)
    predicate_hints: dict = field(default_factory=dict)
    alias_delta: dict = field(default_factory=dict)
    corrections: list = field(default_factory=list)
    metrics: dict | None = None
    sha256_actual: str = ""
    sha256_expected: str = ""
    verdict: str = "UNVERIFIED"      # VERIFIED | MISMATCH | NO_POINTER | MISSING

    # -- loading -----------------------------------------------------------
    @classmethod
    def load(cls, path: str) -> "LearnedSnapshot":
        """Load an artifact from a directory (with ``current.json``) or a file."""
        path = os.path.expanduser(path)
        pointer: dict = {}
        if os.path.isdir(path):
            pointer_path = os.path.join(path, POINTER)
            if os.path.exists(pointer_path):
                with open(pointer_path, "r", encoding="utf-8") as f:
                    pointer = json.load(f)
                version = pointer.get("current")
                if not version:
                    raise LearnedFormatError(
                        f"{pointer_path!r} has no 'current' pointer")
                artifact_path = os.path.join(path, f"{version}.json")
            else:
                candidates = sorted(f for f in os.listdir(path)
                                    if _VERSION_RE.match(f))
                if not candidates:
                    raise LearnedFormatError(
                        f"no lv_*.json artifacts in {path!r}")
                artifact_path = os.path.join(path, candidates[-1])
                version = candidates[-1][:-5]
        else:
            artifact_path = path
            version = os.path.basename(path)[:-5]

        if not os.path.isfile(artifact_path):
            raise FileNotFoundError(artifact_path)
        with open(artifact_path, "rb") as f:
            raw = f.read()
        try:
            doc = json.loads(raw.decode("utf-8"))
        except Exception as exc:          # noqa: BLE001
            raise LearnedFormatError(f"{artifact_path!r} is not JSON: {exc}") from exc
        if doc.get("schema") != LEARNED_SCHEMA:
            raise LearnedFormatError(
                f"unsupported learned schema {doc.get('schema')!r}; "
                f"expected {LEARNED_SCHEMA!r}")

        snap = cls(
            path=artifact_path,
            version=doc.get("version", version),
            parent=doc.get("parent"),
            schema=LEARNED_SCHEMA,
            config=dict(doc.get("config") or {}),
            intent_fallback=dict(doc.get("intent_fallback") or {}),
            predicate_hints=dict(doc.get("predicate_hints") or {}),
            alias_delta=dict(doc.get("alias_delta") or {}),
            corrections=list(doc.get("corrections") or []),
            metrics=doc.get("metrics"),
            sha256_actual=hashlib.sha256(raw).hexdigest())
        expected = pointer.get("sha256", "")
        snap.sha256_expected = expected
        if not expected:
            snap.verdict = "NO_POINTER"
        elif expected == snap.sha256_actual:
            snap.verdict = "VERIFIED"
        else:
            snap.verdict = "MISMATCH"
        return snap

    # -- views -------------------------------------------------------------
    @property
    def trustworthy(self) -> bool:
        """Only a sha-verified artifact may feed routing or metrics."""
        return self.verdict == "VERIFIED"

    def summary(self) -> dict:
        return {"path": self.path, "version": self.version,
                "parent": self.parent, "schema": self.schema,
                "verdict": self.verdict, "sha256": self.sha256_actual,
                "config": dict(self.config),
                "intents": sorted(self.intent_fallback.get("intent_totals", {})),
                "tokens": len(self.intent_fallback.get("tokens", {})),
                "predicate_hints": len(self.predicate_hints),
                "alias_delta": len(self.alias_delta),
                "corrections": len(self.corrections),
                "metrics": self.metrics}

    def alias_of(self, name: str) -> str | None:
        return self.alias_delta.get((name or "").casefold().strip())

    def predicate_hint_for(self, text: str) -> str | None:
        return self.predicate_hints.get(norm_key(text))

    # -- intent fallback (re-implemented from the recorded counts) ---------
    def intent_scores(self, text: str) -> dict[str, float]:
        import math
        tok = self.intent_fallback.get("tokens") or {}
        tot = self.intent_fallback.get("intent_totals") or {}
        if not tot:
            return {}
        alpha = float(self.config.get("alpha", 0.5))
        vocab = len(tok) + 1
        intents = list(tot)
        sums = {i: sum(tv.get(i, 0) for tv in tok.values()) for i in intents}
        total = sum(tot.values())
        logp = {i: math.log((tot[i] + alpha) / (total + alpha * len(intents)))
                for i in intents}
        for w in _TOKEN_RE.findall((text or "").lower()):
            dist = tok.get(w)
            for i in intents:
                c = dist.get(i, 0) if dist else 0
                logp[i] += math.log((c + alpha) / (sums[i] + alpha * vocab))
        return logp

    def predict_intent(self, text: str) -> tuple[str | None, float]:
        logp = self.intent_scores(text)
        if not logp:
            return None, 0.0
        best = max(logp, key=logp.get)
        ordered = sorted(logp.values(), reverse=True)
        margin = (ordered[0] - ordered[1]) if len(ordered) > 1 else ordered[0]
        return best, round(margin, 3)

    def should_use_fallback(self, text: str) -> tuple[str | None, str]:
        """v0.15 policy: use only when the margin clears ``tau``."""
        if not self.config.get("use_fallback_on_unknown", True):
            return None, "fallback-disabled"
        tau = float(self.config.get("tau", 0.6))
        intent, margin = self.predict_intent(text)
        if intent is None:
            return None, "no-counts"
        if margin < tau:
            return None, f"margin {margin} < tau {tau}"
        return intent, f"fallback(margin={margin})"
