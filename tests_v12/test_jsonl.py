"""JSONL import/export tests — validation, idempotency, corruption isolation."""
import json

import pytest

from rasm_cim_v12.knowledge import (KnowledgeStore, export_jsonl, import_jsonl,
                                     teach)


@pytest.fixture()
def seeded(tmp_path):
    kb = KnowledgeStore.init(str(tmp_path / "src"))
    teach(kb, "Франция", "столица", "Париж", subject_type="place", object_type="place",
          created_at="2026-11-26T00:00:00Z")
    teach(kb, "Германия", "столица", "Берлин", subject_type="place", object_type="place",
          created_at="2026-11-26T00:00:01Z")
    return kb


def test_export_import_roundtrip(tmp_path, seeded):
    out = str(tmp_path / "export.jsonl")
    export_jsonl(seeded, out)
    kb2 = KnowledgeStore.init(str(tmp_path / "dst"))
    report = import_jsonl(kb2, out)
    assert report["fatal"] is False
    assert report["imported"] > 0
    assert kb2.stats()["statements_active"] == 2
    assert kb2.status_for(
        next(iter(kb2._by_subject)), "capital_of") == "TRUE"


def test_reimport_is_idempotent(tmp_path, seeded):
    out = str(tmp_path / "export.jsonl")
    export_jsonl(seeded, out)
    kb2 = KnowledgeStore.init(str(tmp_path / "dst"))
    import_jsonl(kb2, out)
    first = kb2.stats()["statements_total"]
    report = import_jsonl(kb2, out)
    assert report["imported"] == 0
    assert report["skipped_duplicates"] > 0
    assert kb2.stats()["statements_total"] == first


def test_corrupted_line_skipped_others_imported(tmp_path, seeded):
    out = str(tmp_path / "export.jsonl")
    export_jsonl(seeded, out)
    lines = open(out, encoding="utf-8").read().splitlines()
    lines.insert(3, "{not json at all")
    lines.insert(5, json.dumps({"event": "statement_asserted",
                                "statement": {"id": "st_x", "predicate": "capital_of"}}))  # missing fields
    corrupted = str(tmp_path / "corrupted.jsonl")
    open(corrupted, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    kb2 = KnowledgeStore.init(str(tmp_path / "dst"))
    report = import_jsonl(kb2, corrupted, strict=False)
    assert report["fatal"] is False
    assert len(report["errors"]) == 2
    assert kb2.stats()["statements_active"] == 2        # good lines still landed


def test_strict_aborts_without_applying(tmp_path, seeded):
    out = str(tmp_path / "export.jsonl")
    export_jsonl(seeded, out)
    lines = open(out, encoding="utf-8").read().splitlines()
    lines.insert(2, "{broken json")
    corrupted = str(tmp_path / "corrupted.jsonl")
    open(corrupted, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    kb2 = KnowledgeStore.init(str(tmp_path / "dst"))
    report = import_jsonl(kb2, corrupted, strict=True)
    assert report["fatal"] is True
    assert kb2.stats()["statements_total"] == 0          # nothing applied


def test_unknown_schema_refused(tmp_path, seeded):
    bad = str(tmp_path / "bad_schema.jsonl")
    with open(bad, "w", encoding="utf-8") as fh:
        fh.write(json.dumps({"type": "header", "schema": "rasm-cim-kb/999"}) + "\n")
        fh.write(json.dumps({"event": "entity_defined",
                             "entity": {"id": "ent_x", "type": "thing", "name": "X"}}) + "\n")
    kb2 = KnowledgeStore.init(str(tmp_path / "dst"))
    report = import_jsonl(kb2, bad)
    assert report["fatal"] is True and "schema" in report["reason"]
    assert kb2.stats()["entities"] == 0


def test_missing_header_refused(tmp_path):
    f = str(tmp_path / "no_header.jsonl")
    with open(f, "w", encoding="utf-8") as fh:
        fh.write(json.dumps({"event": "entity_defined",
                             "entity": {"id": "ent_x", "type": "thing", "name": "X"}}) + "\n")
    kb2 = KnowledgeStore.init(str(tmp_path / "dst"))
    assert import_jsonl(kb2, f)["fatal"] is True


def test_unknown_event_type_rejected(tmp_path):
    f = str(tmp_path / "unknown_event.jsonl")
    with open(f, "w", encoding="utf-8") as fh:
        fh.write(json.dumps({"type": "header", "schema": "rasm-cim-kb/1"}) + "\n")
        fh.write(json.dumps({"event": "nuke_everything"}) + "\n")
    kb2 = KnowledgeStore.init(str(tmp_path / "dst"))
    report = import_jsonl(kb2, f)
    assert len(report["errors"]) == 1 and report["imported"] == 0


def test_provenance_preserved_through_import(tmp_path, seeded):
    out = str(tmp_path / "export.jsonl")
    export_jsonl(seeded, out)
    kb2 = KnowledgeStore.init(str(tmp_path / "dst"))
    import_jsonl(kb2, out)
    srcs = {s.provenance.get("source") for s in kb2.statements.values()}
    assert srcs  # imported provenance is the original, not rewritten
    assert all("user_teach" in s for s in srcs)
