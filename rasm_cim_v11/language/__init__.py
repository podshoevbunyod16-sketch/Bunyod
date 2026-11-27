"""RASM-CIM v0.11 — Russian language layer (experimental, offline).

Public API:
    analyze(text)      → AnalysisResult (normalize + tokenize + lemmas + intent)
    normalize(text)    → NormalizedText
    tokenize(text)     → tuple[Token, ...]
    lemma(word)        → str pseudo-stem
    classify_intent(normalized_text, has_cyrillic=True) → IntentResult
    GraphAdapter       → bridge to TruthGraph9 (no semantics change)

Scope honesty: Russian only, rule-based, no ML, no claims of understanding.
"""
from __future__ import annotations

from .models import AnalysisResult, IntentResult, NormalizedText, Token
from .normalize import fold_yo, normalize
from .tokenize import tokenize
from .morphology import is_stopword, lemma, lemmas, num_to_int
from .intent import INTENTS, classify_intent
from .graph_adapter import GraphAdapter, stable_id

__version__ = "0.11.0"

__all__ = [
    "AnalysisResult", "IntentResult", "NormalizedText", "Token",
    "normalize", "fold_yo", "tokenize", "lemma", "lemmas", "is_stopword",
    "num_to_int", "classify_intent", "INTENTS", "analyze",
    "GraphAdapter", "stable_id", "__version__",
]


def analyze(text: str) -> AnalysisResult:
    norm = normalize(text)
    toks = tokenize(text)                      # offsets into the ORIGINAL
    lem = lemmas(t.norm for t in toks)
    intent = classify_intent(norm.text, norm.has_cyrillic)
    return AnalysisResult(original=text, normalized=norm.text,
                          tokens=toks, lemmas=lem, intent=intent,
                          has_cyrillic=norm.has_cyrillic,
                          version=__version__)
