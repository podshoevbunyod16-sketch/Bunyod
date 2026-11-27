"""Data models for the v0.11 language layer.

All models are plain dataclasses, stdlib-only, offline. They carry both the
original (untouched) user text and the normalized representation together —
normalization NEVER replaces the original.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass(frozen=True)
class NormalizedText:
    """Result of normalization. `original` is exactly what the user sent."""
    original: str
    text: str                      # normalized text (NFKC + casefold + tidy)
    flags: Tuple[str, ...] = ()    # names of applied transformations
    ended_with_question: bool = False
    has_cyrillic: bool = True


@dataclass(frozen=True)
class Token:
    """One token. `start`/`end` are offsets into the ORIGINAL string."""
    text: str        # as it appears in the original string
    norm: str        # normalized (casefolded) form of this token
    start: int
    end: int
    kind: str        # "word" | "number" | "punct"


@dataclass(frozen=True)
class IntentResult:
    intent: str                  # greeting|ask_fact|ask_definition|yes_no|compare|list|unknown
    confidence: float            # 0.0–0.99; low confidence MUST yield "unknown"
    matched: Tuple[str, ...]     # rule ids that fired (explainability)
    scores: Tuple[Tuple[str, float], ...] = ()   # all non-zero intent scores (sorted)
    reason: str = ""


@dataclass
class AnalysisResult:
    """Full v0.11 analysis of one user message (single question granularity)."""
    original: str
    normalized: str
    tokens: Tuple[Token, ...]
    lemmas: Tuple[str, ...]
    intent: IntentResult
    has_cyrillic: bool = True
    version: str = "0.11.0"

    def token_spans(self):
        """(start, end) spans of word tokens, for provenance later."""
        return [(t.start, t.end) for t in self.tokens if t.kind in ("word", "number")]
