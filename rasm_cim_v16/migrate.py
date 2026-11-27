"""v0.16 — migration from the legacy single-relation format.

WHY THIS EXISTS
---------------
Every version up to v0.10 stored at most one undifferentiated relation.  A
system built on that could only express ``Ньютон is_a Исаак Ньютон``, which
conflates two different facts:

    "Ньютон" and "Исаак Ньютон" are two names for one thing   -> alias_of
    "Исаак Ньютон" is a member of the class "физик"           -> instance_of

Spec §15 requires the conversion to be safe, documented, lossless, reversible
and idempotent.  The rules below are the ONLY transformations applied; nothing
else is rewritten, and every converted statement keeps the original record
verbatim inside its ``prov.legacy`` field.

DOCUMENTED RULES
----------------
    R1 ALIAS       p == "is_a" and the subject name is contained in the object
                   name and the object is written as a proper name
                   -> alias_of, confidence 0.95
    R2 INSTANCE    p == "is_a" and the object is a common noun
                   -> instance_of, object kind := CLASS, confidence 0.90
    R3 PRESERVE    p == "is_a", both sides proper names, no name evidence
                   -> related_to, confidence 0.50, flagged ``identity_unknown``
                   (never silently promoted to same_as / alias_of)
    R4 KEEP        p is a known v0.16 relation and not "is_a" -> kept as is
    R5 FALLBACK    p is unknown -> related_to, confidence 0.50, flagged
    R6 NEGATION    ``neg``/``polarity == "false"`` -> polarity false
                   (a legacy negation never becomes a positive fact)
    R7 PROVENANCE  source := "migration:v1", the original record is stored
                   under prov.legacy together with the rule id

LEGACY DOCUMENT SHAPE (v1)
--------------------------
    {"version": 1,                        # or "schema": 1, or absent
     "entities": [{"name": "...", "aliases": [...], "type": "..."}],
     "facts": [{"s": "...", "p": "is_a", "o": "...",
                "neg": false, "conf": 1.0, "source": "...", "ts": "..."}],
     "context": {"entities": [...], "history": [...]}}

Idempotency: converting an already-v2 document is a no-op that reports
``schema=N, nothing to do``.
"""
from __future__ import annotations

import json
import os
import time

from . import LEGACY_SCHEMA_VERSION, SCHEMA_VERSION
from .knowledge_graph import POL_FALSE, POL_TRUE, KnowledgeGraph
from .normalization import is_proper, word_tokens
from .relations import KIND_CLASS, RelationSchema

R1, R2, R3, R4, R5, R6, R7 = "R1_ALIAS", "R2_INSTANCE", "R3_PRESERVE", \
    "R4_KEEP", "R5_FALLBACK", "R6_NEGATION", "R7_PROVENANCE"


def _stems(phrase: str) -> set[str]:
    from .normalization import case_forms
    out = set()
    for t in word_tokens(phrase):
        forms = [f for f in case_forms(t) if len(f) >= 4]
        out.add(min(forms, key=len) if forms else t)
    return out


def name_overlap(a: str, b: str) -> bool:
    ta, tb = set(word_tokens(a)), set(word_tokens(b))
    if not ta or not tb:
        return False
    if ta == tb or ta < tb or tb < ta:
        return True
    sa, sb = _stems(a), _stems(b)
    return bool(sa and sb) and (sa < sb or sb < sa or sa == sb)


def classify_legacy_is_a(subject: str, obj: str) -> tuple[str, float, str]:
    """The documented is_a disambiguation.  Returns (predicate, conf, rule)."""
    obj_proper = is_proper(obj)
    if obj_proper and name_overlap(subject, obj):
        return "alias_of", 0.95, R1
    if not obj_proper:
        return "instance_of", 0.90, R2
    return "related_to", 0.50, R3


# --------------------------------------------------------------------------
# document conversion
# --------------------------------------------------------------------------

def legacy_to_v2(legacy: dict, schema: RelationSchema | None = None,
                 when: str = "") -> tuple[dict, dict]:
    """Convert a v1 document into a v2 document.  Returns ``(kb_dict, report)``.

    Lossless by construction: ``report["records_in"] == report["records_out"]``
    is asserted by the test-suite, and every output statement carries the
    original record inside ``prov.legacy``.
    """
    schema = schema or RelationSchema()
    when = when or time.strftime("%Y-%m-%dT%H:%M:%S")
    kg = KnowledgeGraph(schema=schema, clock=lambda: when)
    report = {"schema_in": int(legacy.get("schema", legacy.get("version",
                                                              LEGACY_SCHEMA_VERSION))),
              "schema_out": SCHEMA_VERSION, "rules": {}, "records_in": 0,
              "records_out": 0, "notes": [], "flagged": []}

    # entities (canonical names + alias lists)
    for ed in legacy.get("entities", []) or []:
        if isinstance(ed, str):
            ed = {"name": ed}
        name = str(ed.get("name", "")).strip()
        if not name:
            continue
        ent = kg.resolver.intern(name, kind=ed.get("type", ""),
                                 source="migration:v1", when=when)
        for a in ed.get("aliases", []) or []:
            kg.resolver.add_alias(ent.entity_id, a, when=when,
                                  source="migration:v1")
        if ed.get("gender"):
            kg.resolver.set_gender(ent.entity_id, ed["gender"], when=when,
                                   source="migration")

    facts = legacy.get("facts", legacy.get("statements", [])) or []
    for fd in facts:
        report["records_in"] += 1
        s = str(fd.get("s", fd.get("subject", ""))).strip()
        o = str(fd.get("o", fd.get("object", ""))).strip()
        p = str(fd.get("p", fd.get("predicate", ""))).strip()
        if not s:
            report["notes"].append(f"record #{report['records_in']} has no "
                                   f"subject; kept as flagged, not dropped")
            report["flagged"].append(fd)
            continue
        neg = bool(fd.get("neg", False)) or str(fd.get("polarity", "")).lower() == "false"
        subj = kg.resolver.intern(s, source="migration:v1", when=when)
        obj_ent = kg.resolver.intern(o, source="migration:v1", when=when) \
            if o else subj

        if p == "is_a" or not p:
            pred, conf, rule = classify_legacy_is_a(s, o or s)
        elif schema.get(p) is not None:
            pred, conf, rule = p, float(fd.get("conf", 1.0)), R4
        else:
            pred, conf, rule = "related_to", 0.50, R5
            report["flagged"].append({"record": fd, "rule": R5,
                                      "note": f"unknown relation {p!r}"})

        if pred == "instance_of" and o and not is_proper(o):
            kg.resolver.set_kind(obj_ent.entity_id, KIND_CLASS, when=when)

        conf = min(conf, float(fd.get("conf", 1.0))) if rule == R4 else conf
        st, _created = kg.assert_triple(
            subj.entity_id, pred, obj_ent.entity_id,
            polarity=POL_FALSE if neg else POL_TRUE, confidence=conf,
            source="migration:v1", when=when,
            provenance={"legacy": fd, "rule": rule,
                        "original_source": fd.get("source", ""),
                        "original_predicate": p})
        report["rules"][rule] = report["rules"].get(rule, 0) + 1
        if neg:
            report["rules"][R6] = report["rules"].get(R6, 0) + 1
        report["records_out"] += 1
        if rule in (R3, R5):
            report["notes"].append(
                f"{st.statement_id}: legacy {p or '(empty)'} "
                f"{s!r}->{o!r} kept as {pred} (confidence {conf:.2f}); "
                f"identity deliberately NOT asserted")

    report["rules"][R7] = report["records_out"]
    kb = {"schema": SCHEMA_VERSION, "generator": "rasm-cim v0.16 migration",
          "saved_at": when, "graph": kg.to_dict(),
          "evidence": {"evidence": []}, "conflicts": {"conflicts": [],
                                                      "disjoint_classes": []},
          "memory": {"working": [], "counter": 0, "fingerprints": {}},
          "migration": report}
    return kb, report


def upgrade_kb_dict(kb: dict) -> list[str]:
    """In-place schema upgrade used by ``Store.load``.

    A v2+ document is returned untouched (idempotent).  A v1 document is
    converted by rebuilding it through ``legacy_to_v2``.
    """
    ver = int(kb.get("schema", kb.get("version", LEGACY_SCHEMA_VERSION)))
    if ver >= SCHEMA_VERSION:
        return [f"schema={ver}, nothing to do"] if ver == SCHEMA_VERSION else \
               [f"schema={ver} is NEWER than this build "
                f"({SCHEMA_VERSION}); loaded read-only"]
    fresh, report = legacy_to_v2(kb)
    kb.clear()
    kb.update(fresh)
    notes = [f"migrated v{ver}->v{SCHEMA_VERSION}: "
             f"{report['records_in']} records in, {report['records_out']} out, "
             f"rules {report['rules']}"]
    notes += report["notes"]
    return notes


# --------------------------------------------------------------------------
# file-level migration with backup + rollback
# --------------------------------------------------------------------------

def migrate_file(kb_path: str, dry_run: bool = False,
                 schema: RelationSchema | None = None,
                 backup: bool = True) -> dict:
    """Migrate ``kb_path`` in place.

    1. read + validate
    2. backup (unless ``backup=False`` or ``dry_run``)
    3. convert
    4. verify record count and re-read
    5. atomic replace

    Returns a report dict.  Raises on any inconsistency — a half-migrated file
    is never written.
    """
    from .persistence import atomic_write_json, read_json
    if not os.path.exists(kb_path):
        return {"status": "no-file", "path": kb_path}
    legacy = read_json(kb_path)
    ver = int(legacy.get("schema", legacy.get("version", LEGACY_SCHEMA_VERSION)))
    if ver >= SCHEMA_VERSION:
        return {"status": "already-current", "path": kb_path, "schema": ver,
                "records_in": len(legacy.get("graph", {}).get("statements", [])),
                "records_out": len(legacy.get("graph", {}).get("statements", []))}
    fresh, report = legacy_to_v2(legacy, schema=schema)
    report["status"] = "migrated" if not dry_run else "dry-run"
    report["path"] = kb_path
    if report["records_in"] != report["records_out"]:
        raise RuntimeError(
            f"migration would lose records: {report['records_in']} in vs "
            f"{report['records_out']} out — refusing to write")
    if dry_run:
        return report
    bak = None
    if backup:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        bak = f"{kb_path}.bak-{stamp}"
        atomic_write_json(bak, legacy)
        report["backup"] = bak
    atomic_write_json(kb_path, fresh)
    reread = read_json(kb_path)
    n_after = len(reread.get("graph", {}).get("statements", []))
    report["verify_reread_statements"] = n_after
    report["verify_schema"] = int(reread.get("schema", 0))
    if n_after < report["records_out"]:
        raise RuntimeError(f"post-write verification failed: {n_after} < "
                           f"{report['records_out']}")
    return report


def migrate_context_file(context_path: str, dry_run: bool = False) -> dict:
    """Upgrade a legacy context document (flat entity list -> topics+salience)."""
    from .context_engine import migrate_context_dict
    from .persistence import atomic_write_json, read_json
    if not os.path.exists(context_path):
        return {"status": "no-file", "path": context_path}
    doc = read_json(context_path)
    ver = int(doc.get("schema", LEGACY_SCHEMA_VERSION))
    if ver >= SCHEMA_VERSION:
        return {"status": "already-current", "path": context_path, "schema": ver}
    notes = migrate_context_dict(doc)
    out = {"status": "migrated" if not dry_run else "dry-run",
           "path": context_path, "schema_in": ver,
           "schema_out": doc.get("schema"), "notes": notes,
           "entities_kept": len(doc.get("active", [])),
           "history_kept": len(doc.get("history", []))}
    if not dry_run:
        atomic_write_json(context_path, doc)
    return out


def write_legacy_sample(path: str, facts: list[tuple[str, str, str]],
                        entities: list[dict] | None = None,
                        version: int = LEGACY_SCHEMA_VERSION) -> str:
    """Write a v1 document.  Used by the test-suite and by the migration
    dry-run so the legacy path is exercised against a real file, not a mock."""
    from .persistence import atomic_write_json
    doc = {"version": version,
           "entities": entities or [],
           "facts": [{"s": s, "p": p, "o": o, "source": "legacy-sample",
                      "ts": "2026-01-01T00:00:00"} for s, p, o in facts]}
    atomic_write_json(path, doc)
    return path
