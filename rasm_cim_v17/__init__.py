"""RASM-CIM v0.17 — integration + cognitive layer on top of v0.16.

v0.16 (``rasm_cim_v16``) is the frozen semantic core: typed graph, provenance,
evidence, bounded inference, answers.  This package never edits it.  It adds,
in separate modules, what the v0.17 roadmap asks for:

    interop/    Plan A — read-only readers for the v0.11–v0.15 formats
                (``rasm-cim-kb/1`` event log, ``rasm-cim-learned/1`` artifact,
                the v0.11 demo-fact JSONL) and a dry-run / idempotent /
                rollback-able importer that materialises them into the v0.16
                graph with their provenance intact.
    verify      integrity self-check of a persisted v0.16 database.
    cli         the single entry point: ``python -m rasm_cim_v17`` and
                ``python -m rasm_cim`` (added in v0.17) both land here.

Later plans (B–G: semantics, discourse, learning, robustness) land in their own
subpackages; nothing in this file promises a capability that does not exist
yet — see ``LIMITATIONS`` in docs_v17/.
"""
from __future__ import annotations

__version__ = "0.17.0"

#: schema of the v0.17 auxiliary documents (import ledger, routing hints)
AUX_SCHEMA_VERSION = 1

__all__ = ["__version__", "AUX_SCHEMA_VERSION"]
