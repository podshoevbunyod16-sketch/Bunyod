"""Learned routing hints as a v0.17 document (``routing.json``).

The v0.15 artifact carried two things that are *not* knowledge and must never
live inside ``kb.json``:

  * ``predicate_hints`` — normalized question -> relation name (a user
    correction of the *routing*, not of the facts);
  * ``alias_delta`` — folded into the entity resolver instead, because an alias
    is knowledge about an entity.

Keeping hints in their own file means: a hint can be dropped without touching
the KB, a KB can be restored without losing hints, and the provenance of a hint
(the artifact version + sha256 it came from) is recorded per entry.
"""
from __future__ import annotations

import os
from typing import Iterable

from rasm_cim_v16.persistence import atomic_write_json, read_json

from . import AUX_SCHEMA_VERSION
from .interop.learned import norm_key

ROUTING_FILENAME = "routing.json"
ROUTING_SCHEMA = "rasm-cim-routing/1"


class RoutingStore:
    def __init__(self, home: str):
        self.home = os.path.expanduser(home)
        self.path = os.path.join(self.home, ROUTING_FILENAME)

    # -- io ---------------------------------------------------------------
    def load(self) -> dict:
        if not os.path.exists(self.path):
            return {"schema": ROUTING_SCHEMA, "version": AUX_SCHEMA_VERSION,
                    "predicate_hints": {}, "provenance": {}}
        doc = read_json(self.path)
        if doc.get("schema") != ROUTING_SCHEMA:
            raise ValueError(f"unsupported routing schema {doc.get('schema')!r}")
        doc.setdefault("predicate_hints", {})
        doc.setdefault("provenance", {})
        return doc

    def save(self, doc: dict) -> None:
        os.makedirs(self.home, exist_ok=True)
        atomic_write_json(self.path, doc)

    # -- writes -----------------------------------------------------------
    def merge_predicate_hints(self, hints: dict[str, str],
                              origin: str = "") -> int:
        """Add hints; an existing key keeps its value (first writer wins).

        Returns how many *new* hints were written.  Overwriting would make the
        same import non-idempotent, which is exactly what Plan A forbids.
        """
        doc = self.load()
        added = 0
        for question, predicate in hints.items():
            key = norm_key(question)
            if not key or not predicate:
                continue
            if key in doc["predicate_hints"]:
                continue
            doc["predicate_hints"][key] = predicate
            doc["provenance"][key] = origin
            added += 1
        if added:
            self.save(doc)
        return added

    # -- reads ------------------------------------------------------------
    def hint_for(self, text: str) -> str | None:
        return self.load()["predicate_hints"].get(norm_key(text))

    def hints(self) -> dict[str, str]:
        return dict(self.load()["predicate_hints"])

    def __len__(self) -> int:
        return len(self.hints())
