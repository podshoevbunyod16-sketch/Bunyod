"""RASM-CIM v0.16 — semantic / language / context layer.

v0.10 froze the storage core (residue-addressable streaming store, journal,
hypervectors) and TruthGraph9 (polarised triple semantics).  v0.10's own
LIMITATIONS.md lists everything language-related as *not implemented*
(v0.11 normalisation, v0.12 entities/relations+provenance, v0.13 dialogue
context+pronouns, v0.14 evidence-linked pipeline, v0.15 corpus).

v0.16 implements that missing stack.  It is additive: no frozen module is
modified, the storage core is reused read/write through its public API, and
TruthGraph9 is reused as the polarity/conflict engine (interned ids).

Layout (one responsibility per module, independently testable):

    normalization.py        text -> comparable forms (ru morphology, translit)
    relations.py            typed, extensible relation schema
    entity_resolver.py      canonical entities, aliases, confidence, ambiguity
    language_parser.py      intent + QueryIR (subject/predicate/object/neg/...)
    relation_extractor.py   declarative sentence -> typed statement candidate
    knowledge_graph.py      typed triple store, indexes, evidence, provenance
    inference_engine.py     rule registry, bounded BFS, proof chains
    context_engine.py       topics, active entities, pronoun resolution, history
    memory_manager.py       working / episodic / semantic / procedural tiers
    contradiction_detector.py conflict classes, both sides retained
    evidence_manager.py     evidence records, explanations
    answer_engine.py        concise | normal | detailed, status-classified
    dialogue_manager.py     orchestration input -> classified answer
    persistence.py          kb.json / context.json, atomic, versioned
    migrate.py              legacy is_a migration, backup, rollback, idempotent
    cli.py                  python -m rasm_cim_v16  (teach/ask/chat/:inspect...)

Runtime dependencies: Python stdlib only (>=3.10).  No LLM, no network.
"""
from __future__ import annotations

__version__ = "0.16.0"

SCHEMA_VERSION = 2          # kb.json / context.json schema written by v0.16
LEGACY_SCHEMA_VERSION = 1   # the is_a-era format migrate.py converts from

__all__ = ["__version__", "SCHEMA_VERSION", "LEGACY_SCHEMA_VERSION"]
