"""v0.11 → TruthGraph9 adapter.

Maps Russian surface forms to the integer triples TruthGraph9 uses.
IDs are deterministic (sha256 of fold(text), first 8 bytes, big-endian) —
offline, stable across runs and machines, no state required. The adapter
does NOT change TruthGraph9 semantics: TRUE/FALSE/UNKNOWN/CONFLICT and the
symmetric retract mode are used as-is.
"""
from __future__ import annotations

import hashlib

from rasm_cim_v9.graph import TruthGraph9

from .normalize import fold_yo

_FUNCTIONAL = {"capital_of", "population_of", "height_of"}


def stable_id(text: str) -> int:
    norm = fold_yo(text).casefold().strip()
    return int.from_bytes(hashlib.sha256(norm.encode("utf-8")).digest()[:8], "big")


class GraphAdapter:
    """Thin typed wrapper over TruthGraph9 for natural-language facts."""

    def __init__(self, mode: str = "symmetric"):
        self.graph = TruthGraph9(functional_relations={stable_id(r) for r in _FUNCTIONAL},
                                 mode=mode)

    @staticmethod
    def id_for(name: str) -> int:
        return stable_id(name)

    def assert_fact(self, subject: str, relation: str, object_: str,
                    polarity: str = "true") -> None:
        s, r, o = stable_id(subject), stable_id(relation), stable_id(object_)
        if polarity == "true":
            self.graph.assert_true(s, r, o)
        elif polarity == "false":
            self.graph.assert_false(s, r, o)
        else:
            raise ValueError(f"polarity must be 'true'|'false', got {polarity!r}")

    def retract_fact(self, subject: str, relation: str, object_: str) -> None:
        self.graph.retract(stable_id(subject), stable_id(relation), stable_id(object_))

    def query_fact(self, subject: str, relation: str, object_: str) -> str:
        return self.graph.query(stable_id(subject), stable_id(relation), stable_id(object_))

    def query_relation(self, subject: str, relation: str) -> str:
        return self.graph.query(stable_id(subject), stable_id(relation))

    def objects_of(self, subject: str, relation: str) -> list[str]:
        # TruthGraph9 stores int ids only; we cannot invert arbitrary hashes.
        # Kept deliberately explicit: use the v0.12 KB for surface-form recall.
        raise NotImplementedError(
            "TruthGraph9 is id-based; surface-form recall lives in the v0.12 KB")
