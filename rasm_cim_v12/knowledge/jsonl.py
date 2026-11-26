"""JSONL import/export for the KB (schema rasm-cim-kb/1).

Export format: one JSON object per line.
  line 1 : {"type": "header", "schema": "rasm-cim-kb/1", "exported_at": iso,
            "counts": {...}}
  then   : raw events (same objects as events.jsonl lines)

Import rules (spec §v0.12):
  * unknown schema / missing header → refuse with ImportReport(fatal=True), nothing applied;
  * each line is validated independently: unknown event, missing fields, bad enum
    → recorded in report["errors"] as {line, error} and SKIPPED; other lines apply;
  * --strict → on the first bad line, abort with NOTHING applied
    (load-then-swap: lines are validated fully before any event is appended);
  * re-import is idempotent: entities/statements dedup by deterministic ids,
    retract/promote of an already-applied state is a no-op;
  * provenance of imported events is preserved verbatim (never rewritten).
"""
from __future__ import annotations

import json
import os
from typing import Iterator, List, Optional

from .models import SCHEMA
from .store import KnowledgeStore, utcnow

_APPLICABLE = {"entity_defined", "entity_alias_added", "statement_asserted",
               "statement_retracted", "hypothesis_promoted"}


def export_jsonl(store: KnowledgeStore, out_path: Optional[str] = None) -> str:
    lines: List[str] = []
    header = {"type": "header", "schema": SCHEMA, "exported_at": utcnow(),
              "counts": store.stats()}
    lines.append(json.dumps(header, ensure_ascii=False))
    for ev in store.iter_events():
        ev = dict(ev)
        ev.pop("schema", None)   # re-emitted by importer after validation
        ev.pop("seq", None)      # seq is store-local; importers re-assign
        ev.pop("ts", None)
        lines.append(json.dumps(ev, ensure_ascii=False))
    payload = "\n".join(lines) + "\n"
    if out_path:
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(payload)
    return payload


class ImportReport(dict):
    pass


def import_jsonl(store: KnowledgeStore, path: str, strict: bool = False) -> ImportReport:
    errors: List[dict] = []
    events: List[dict] = []
    with open(path, encoding="utf-8") as fh:
        raw_lines = fh.readlines()
    if not raw_lines:
        return ImportReport(fatal=True, reason="empty file", errors=[],
                            imported=0, skipped=0, total_lines=0)

    def _fatal(reason):
        return ImportReport(fatal=True, reason=reason, errors=errors,
                            imported=0, skipped=len(raw_lines), total_lines=len(raw_lines))

    try:
        header = json.loads(raw_lines[0])
    except Exception as exc:
        return _fatal(f"bad header line: {exc}")
    if not isinstance(header, dict) or header.get("type") != "header":
        return _fatal("missing header (type=header)")
    if header.get("schema") != SCHEMA:
        return _fatal(f"schema mismatch: {header.get('schema')!r} (expected {SCHEMA})")

    for i, line in enumerate(raw_lines[1:], start=2):
        line = line.strip()
        if not line:
            continue
        try:
            ev = json.loads(line)
        except Exception as exc:
            errors.append({"line": i, "error": f"json: {exc}"})
            if strict:
                return _fatal(f"strict: line {i}")
            continue
        if not isinstance(ev, dict) or ev.get("event") not in _APPLICABLE:
            errors.append({"line": i, "error": f"unknown/invalid event {ev.get('event') if isinstance(ev, dict) else type(ev)!s}"})
            if strict:
                return _fatal(f"strict: line {i}")
            continue
        try:
            _validate_event(ev)
        except ValueError as exc:
            errors.append({"line": i, "error": str(exc)})
            if strict:
                return _fatal(f"strict: line {i}")
            continue
        events.append(ev)

    if strict and errors:
        return _fatal("strict validation failed")

    before = store._event_count
    imported = 0
    skipped_dup = 0
    for ev in events:
        key = _event_key(ev)
        if key and _already_applied(store, key, ev):
            skipped_dup += 1
            continue
        store._append(ev)      # re-assigns seq/ts; provenance untouched
        imported += 1
    return ImportReport(fatal=False, errors=errors, imported=imported,
                        skipped_duplicates=skipped_dup, total_lines=len(raw_lines),
                        seq_start=before + 1)


def _validate_event(ev: dict):
    kind = ev["event"]
    if kind == "entity_defined":
        e = ev.get("entity") or {}
        for k in ("id", "type", "name"):
            if not e.get(k):
                raise ValueError(f"entity_defined missing {k}")
    elif kind == "entity_alias_added":
        if not ev.get("entity_id") or not ev.get("alias"):
            raise ValueError("entity_alias_added missing entity_id/alias")
    elif kind == "statement_asserted":
        s = ev.get("statement") or {}
        for k in ("id", "subject", "predicate", "polarity", "kind"):
            if not s.get(k):
                raise ValueError(f"statement_asserted missing {k}")
        if s["polarity"] not in ("assert", "deny"):
            raise ValueError("statement polarity must be assert|deny")
        if s["kind"] not in ("user_assertion", "extracted_fact", "hypothesis"):
            raise ValueError(f"bad kind {s['kind']!r}")
    elif kind == "statement_retracted":
        if not ev.get("statement_id"):
            raise ValueError("statement_retracted missing statement_id")
    elif kind == "hypothesis_promoted":
        if not ev.get("statement_id"):
            raise ValueError("hypothesis_promoted missing statement_id")


def _event_key(ev: dict):
    kind = ev["event"]
    if kind == "entity_defined":
        return ("e", ev["entity"]["id"])
    if kind == "statement_asserted":
        return ("s", ev["statement"]["id"])
    if kind == "statement_retracted":
        return ("r", ev["statement_id"])
    if kind == "hypothesis_promoted":
        return ("p", ev["statement_id"])
    if kind == "entity_alias_added":
        return ("a", ev["entity_id"], ev["alias"])
    return None


def _already_applied(store: KnowledgeStore, key, ev) -> bool:
    kind = key[0]
    if kind == "e":
        return key[1] in store.entities
    if kind == "s":
        return key[1] in store.statements
    if kind == "r":
        s = store.statements.get(key[1])
        return bool(s) and s.status == "retracted"
    if kind == "p":
        s = store.statements.get(key[1])
        return bool(s) and s.kind != "hypothesis"
    if kind == "a":
        e = store.entities.get(key[1])
        return bool(e) and ev["alias"] in e.aliases
    return False
