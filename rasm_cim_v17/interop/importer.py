"""Plan A — dry-run / idempotent / rollback-able import of legacy data.

Contract (each bullet is tested in ``tests_v17/test_plan_a_importer.py``):

* **Dry run first.**  ``build_plan`` is pure: it reads the legacy file, decides
  per record what would happen, and reports counts, unmapped predicates,
  skipped records with reasons, and conflicts that the import would introduce.
  Nothing is written until ``apply_plan`` is called.
* **No guessing.**  An unknown predicate, a hypothesis statement, a record
  without an object — each is *skipped with a reason* and surfaced, never
  coerced into ``related_to``.
* **Provenance is carried, not regenerated.**  Every imported statement keeps
  the legacy statement id, source, doc_id, fragment and reliability inside
  ``Statement.provenance["legacy"]``; the evidence log gets an
  ``EV_IMPORT`` row pointing at the same source.
* **Idempotent.**  The ledger is keyed by ``(realpath, sha256(source))``.
  Re-running the same import is a no-op that reports
  ``duplicate_of=<ledger id>``.  Additionally the v0.16 graph itself
  reinforces an identical triple instead of duplicating it, so even an import
  whose source file was touched (new sha256, same content) adds no statements.
* **Reversible in one step.**  Before writing, ``kb.json`` and ``context.json``
  are copied to ``*.bak-<stamp>-pre-import`` and the paths are recorded in the
  ledger entry; ``restore`` copies them back.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from typing import Iterable

from rasm_cim_v16.evidence_manager import EV_IMPORT
from rasm_cim_v16.persistence import Store, atomic_write_json, read_json
from rasm_cim_v16.relations import KIND_ANY

from .legacy import (LegacyFacts, LegacyFormatError, LegacyKB, LegacyStatement)
from .learned import LearnedSnapshot
from .mapping import (BOOLEAN_GLOSS, KIND_POLICY, boolean_value, custom_relations,
                      map_entity_kind, map_predicate)

LEDGER_FILENAME = "import_ledger.json"
LEDGER_SCHEMA = "rasm-cim-import-ledger/1"
PLAN_SCHEMA = "rasm-cim-import-plan/1"

#: how much a legacy record's evidence is worth on import
RELIABILITY_CONFIDENCE = {"high": 0.95, "medium": 0.90, "low": 0.75}
DEFAULT_RELIABILITY_CONFIDENCE = 0.90


class ImportError_(RuntimeError):
    """Raised when an import cannot proceed (never for skippable records)."""


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


# --------------------------------------------------------------------------
# plan records
# --------------------------------------------------------------------------

@dataclass
class PlannedEntity:
    legacy_id: str
    name: str
    kind: str
    source_type: str
    kind_mapped: bool = True
    aliases: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"legacy_id": self.legacy_id, "name": self.name,
                "kind": self.kind, "source_type": self.source_type,
                "kind_mapped": self.kind_mapped, "aliases": list(self.aliases)}


@dataclass
class PlannedStatement:
    legacy_id: str
    subject_legacy_id: str
    subject_name: str
    predicate: str                    # target v0.16 relation
    legacy_predicate: str
    object_kind: str                  # "entity" | "literal"
    object_legacy_id: str = ""
    object_name: str = ""
    polarity: str = "true"
    confidence: float = DEFAULT_RELIABILITY_CONFIDENCE
    source: str = ""
    provenance: dict = field(default_factory=dict)
    note: str = ""
    skipped_reason: str = ""          # "" == will be imported

    @property
    def will_import(self) -> bool:
        return not self.skipped_reason

    def as_dict(self) -> dict:
        return {"legacy_id": self.legacy_id, "subject": self.subject_name,
                "predicate": self.predicate,
                "legacy_predicate": self.legacy_predicate,
                "object_kind": self.object_kind,
                "object": self.object_name,
                "polarity": self.polarity, "confidence": self.confidence,
                "source": self.source, "note": self.note,
                "skipped_reason": self.skipped_reason}


@dataclass
class ImportPlan:
    source_path: str
    source_schema: str
    source_sha256: str
    kind: str = "kb"                  # kb | facts | learned
    entities: list[PlannedEntity] = field(default_factory=list)
    statements: list[PlannedStatement] = field(default_factory=list)
    relations: list[dict] = field(default_factory=list)
    alias_delta: dict[str, str] = field(default_factory=dict)
    predicate_hints: dict[str, str] = field(default_factory=dict)
    learned_summary: dict = field(default_factory=dict)
    warnings: list[dict] = field(default_factory=list)
    reader_errors: list[dict] = field(default_factory=list)

    # -- views -------------------------------------------------------------
    @property
    def importable(self) -> list[PlannedStatement]:
        return [s for s in self.statements if s.will_import]

    @property
    def skipped(self) -> list[PlannedStatement]:
        return [s for s in self.statements if not s.will_import]

    def unmapped_predicates(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for s in self.statements:
            if s.skipped_reason.startswith("unmapped predicate"):
                out[s.legacy_predicate] = out.get(s.legacy_predicate, 0) + 1
        return out

    def skip_reasons(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for s in self.skipped:
            key = s.skipped_reason.split(":")[0]
            out[key] = out.get(key, 0) + 1
        return out

    def digest(self) -> str:
        """Stable hash of the *decisions*, used to detect a changed plan."""
        payload = json.dumps(
            {"source_sha256": self.source_sha256, "kind": self.kind,
             "statements": [s.as_dict() for s in self.statements],
             "entities": [e.as_dict() for e in self.entities],
             "relations": sorted(r["name"] for r in self.relations)},
            ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def summary(self) -> dict:
        return {
            "schema": PLAN_SCHEMA, "source": self.source_path,
            "source_schema": self.source_schema,
            "source_sha256": self.source_sha256, "kind": self.kind,
            "entities": len(self.entities),
            "literal_objects": len([s for s in self.importable
                                    if s.object_kind == "literal"]),
            "statements_total": len(self.statements),
            "statements_importable": len(self.importable),
            "statements_skipped": len(self.skipped),
            "skip_reasons": self.skip_reasons(),
            "unmapped_predicates": self.unmapped_predicates(),
            "custom_relations": sorted(r["name"] for r in self.relations),
            "alias_delta": len(self.alias_delta),
            "predicate_hints": len(self.predicate_hints),
            "warnings": len(self.warnings),
            "reader_errors": len(self.reader_errors),
            "digest": self.digest(),
        }

    def render(self) -> str:
        """Human-readable dry-run report (no colours, no dependency)."""
        s = self.summary()
        lines = [
            f"ИМПОРТ (dry-run): {s['source']}",
            f"  формат: {s['source_schema']}  sha256: {s['source_sha256'][:16]}…",
            f"  сущностей: {s['entities']} (+{s['literal_objects']} литералов)   "
            f"утверждений: {s['statements_total']}"
            f" (к импорту {s['statements_importable']}, пропуск {s['statements_skipped']})",
        ]
        if s["custom_relations"]:
            lines.append("  новые отношения: " + ", ".join(s["custom_relations"]))
        if s["skip_reasons"]:
            lines.append("  причины пропуска:")
            for k, v in sorted(s["skip_reasons"].items()):
                lines.append(f"    - {k}: {v}")
        if s["unmapped_predicates"]:
            lines.append("  НЕИЗВЕСТНЫЕ ПРЕДИКАТЫ (нужно расширить mapping.py):")
            for k, v in sorted(s["unmapped_predicates"].items()):
                lines.append(f"    - {k}: {v}")
        if s["warnings"]:
            lines.append(f"  предупреждений: {s['warnings']} (см. --json)")
        if s["reader_errors"]:
            lines.append(f"  ошибок чтения: {s['reader_errors']} (см. --json)")
        if self.learned_summary:
            ls = self.learned_summary
            lines.append(f"  обучающий артефакт: {ls.get('version')} "
                         f"({ls.get('verdict')}), alias_delta={ls.get('alias_delta')}, "
                         f"predicate_hints={ls.get('predicate_hints')}")
        lines.append(f"  digest: {s['digest'][:16]}…")
        return "\n".join(lines)


@dataclass
class ImportResult:
    ledger_id: str
    applied: bool
    duplicate_of: str = ""
    entities_before: int = 0
    entities_after: int = 0
    statements_before: int = 0
    statements_after: int = 0
    statements_added: int = 0
    statements_reinforced: int = 0
    statements_skipped: int = 0
    aliases_added: int = 0
    relations_registered: list[str] = field(default_factory=list)
    predicate_hints_written: int = 0
    alias_delta_applied: int = 0
    backup: dict = field(default_factory=dict)
    saved: dict = field(default_factory=dict)

    def summary(self) -> dict:
        return {k: getattr(self, k) for k in
                ("ledger_id", "applied", "duplicate_of", "entities_before",
                 "entities_after", "statements_before", "statements_after",
                 "statements_added", "statements_reinforced",
                 "statements_skipped", "aliases_added",
                 "relations_registered", "predicate_hints_written",
                 "alias_delta_applied", "backup", "saved")}

    def render(self) -> str:
        d = self.summary()
        if not d["applied"]:
            return (f"ИМПОРТ не выполнялся: источник уже импортирован "
                    f"(ledger {d['duplicate_of']})")
        return "\n".join([
            f"ИМПОРТ выполнен, ledger {d['ledger_id']}",
            f"  сущности: {d['entities_before']} -> {d['entities_after']}",
            f"  утверждения: {d['statements_before']} -> {d['statements_after']}"
            f" (+{d['statements_added']} новых, "
            f"{d['statements_reinforced']} подтверждено повторно, "
            f"{d['statements_skipped']} пропущено)",
            f"  алиасов добавлено: {d['aliases_added']}",
            f"  отношений зарегистрировано: {len(d['relations_registered'])}"
            + (f" ({', '.join(d['relations_registered'])})"
               if d["relations_registered"] else ""),
            f"  резервные копии: {d['backup']}",
        ])


# --------------------------------------------------------------------------
# plan building
# --------------------------------------------------------------------------

def build_plan(source: str, kind: str = "auto") -> ImportPlan:
    """Read ``source`` and decide, without writing, what the import would do."""
    src = os.path.expanduser(source)
    if kind == "auto":
        if os.path.isdir(src) and os.path.exists(os.path.join(src, "current.json")):
            kind = "learned"
        elif os.path.isdir(src):
            kind = "kb"
        elif src.endswith(".jsonl"):
            kind = _sniff_jsonl(src)
        else:
            raise ImportError_(f"cannot guess the format of {src!r}; pass --kind")
    if kind == "kb":
        return plan_from_legacy_kb(LegacyKB.load(src), kind="kb")
    if kind == "facts":
        return plan_from_legacy_kb(LegacyFacts.load(src).as_legacy_kb(),
                                   kind="facts")
    if kind == "learned":
        return plan_from_artifact(LearnedSnapshot.load(src))
    raise ImportError_(f"unknown import kind {kind!r}")


def _sniff_jsonl(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "event" in row:
                return "kb"
            if "relation" in row and "subject" in row:
                return "facts"
            if "schema" in row and "intent_fallback" in row:
                raise ImportError_("a learned artifact must be a directory or "
                                   "a .json file, not JSONL")
            break
    raise ImportError_(f"cannot sniff the JSONL format of {path!r}; pass --kind")


def plan_from_legacy_kb(kb: LegacyKB, kind: str = "kb") -> ImportPlan:
    plan = ImportPlan(source_path=kb.path, source_schema=kb.schema,
                      source_sha256=kb.sha256, kind=kind,
                      reader_errors=list(kb.errors))
    for kind_seen in kb.unknown_events:
        plan.warnings.append({"kind": "unknown_event",
                              "detail": f"legacy event {kind_seen!r} skipped"})

    # -- entities actually referenced by importable statements -------------
    used: set[str] = set()
    planned_statements: list[PlannedStatement] = []
    needed_relations: dict[str, dict] = {}

    for st in sorted(kb.statements.values(), key=lambda s: s.id):
        planned = _plan_statement(kb, st, plan, needed_relations)
        if planned.object_kind == "entity" and planned.object_legacy_id:
            used.add(planned.object_legacy_id)
        used.add(planned.subject_legacy_id)
        planned_statements.append(planned)

    for eid in sorted(used):
        ent = kb.entities.get(eid)
        if ent is None:
            plan.warnings.append({"kind": "dangling_entity", "detail": eid})
            continue
        kind, mapped = map_entity_kind(ent.type)
        if not mapped:
            plan.warnings.append({"kind": "unknown_entity_type",
                                  "detail": f"{ent.name!r}: type {ent.type!r} "
                                            f"-> {KIND_ANY}"})
        plan.entities.append(PlannedEntity(
            legacy_id=ent.id, name=ent.name, kind=kind,
            source_type=ent.type, kind_mapped=mapped,
            aliases=list(ent.aliases)))

    plan.statements = planned_statements
    plan.relations = sorted(needed_relations.values(), key=lambda d: d["name"])
    _detect_planned_conflicts(plan)
    return plan


def plan_from_artifact(snap: LearnedSnapshot) -> ImportPlan:
    """Learned artifacts contribute routing hints and aliases, never facts."""
    plan = ImportPlan(source_path=snap.path, source_schema=snap.schema,
                      source_sha256=snap.sha256_actual, kind="learned",
                      learned_summary=snap.summary())
    if not snap.trustworthy:
        plan.warnings.append({
            "kind": "artifact_unverified",
            "detail": f"sha256 verdict {snap.verdict}: routing hints and alias "
                      f"delta are NOT applied"})
        return plan
    plan.alias_delta = {str(k): str(v) for k, v in snap.alias_delta.items()}
    plan.predicate_hints = {str(k): str(v) for k, v in snap.predicate_hints.items()}
    unmapped = {}
    for q, predicate in plan.predicate_hints.items():
        if map_predicate(predicate) is None:
            unmapped[predicate] = unmapped.get(predicate, 0) + 1
    if unmapped:
        plan.warnings.append({"kind": "unmapped_hint_predicate",
                              "detail": unmapped})
    return plan


def _plan_statement(kb: LegacyKB, st: LegacyStatement, plan: ImportPlan,
                    needed: dict[str, dict]) -> PlannedStatement:
    subject = kb.entities.get(st.subject)
    subject_name = subject.name if subject else ""
    obj_entity = kb.entities.get(st.object_entity) if st.object_entity else None
    obj_name = obj_entity.name if obj_entity else (st.object_literal or "")
    base = PlannedStatement(
        legacy_id=st.id, subject_legacy_id=st.subject, subject_name=subject_name,
        predicate=st.predicate, legacy_predicate=st.predicate,
        object_kind="entity" if st.object_entity else "literal",
        object_legacy_id=st.object_entity or "", object_name=obj_name,
        provenance={"imported_with": "rasm_cim_v17", "legacy": st.as_dict(),
                    "legacy_schema": kb.schema, "legacy_sha256": kb.sha256})

    if subject is None:
        base.skipped_reason = "dangling subject: statement references an unknown entity"
        return base
    if st.object_entity and obj_entity is None:
        base.skipped_reason = "dangling object: statement references an unknown entity"
        return base

    policy = KIND_POLICY.get(st.kind, "import")
    if policy == "skip":
        base.skipped_reason = (f"statement kind {st.kind!r} is not imported: "
                               f"v0.16 has no hypothesis status and importing it "
                               f"as a fact would overstate confidence")
        return base

    mapping = map_predicate(st.predicate)
    if mapping is None:
        base.skipped_reason = (f"unmapped predicate {st.predicate!r}: add it to "
                               f"rasm_cim_v17/interop/mapping.py")
        return base

    source_label = (st.provenance or {}).get("source", "legacy")
    base.source = f"import:{kb.schema}:{source_label}"
    base.confidence = RELIABILITY_CONFIDENCE.get(
        (st.provenance or {}).get("reliability", ""),
        DEFAULT_RELIABILITY_CONFIDENCE)
    base.polarity = {"assert": "true", "deny": "false",
                     "true": "true", "false": "false"}.get(st.polarity, "true")

    # -- boolean properties: "может_летать -> да" is not an edge --------------
    if mapping.boolean_object:
        value = boolean_value(obj_name)
        gloss = BOOLEAN_GLOSS.get(st.predicate, st.predicate.replace("_", " "))
        base.predicate = "has_property"
        base.legacy_predicate = st.predicate
        base.object_kind = "literal"
        base.object_legacy_id = ""
        base.object_name = gloss
        base.note = f"boolean legacy predicate normalised to has_property({gloss!r})"
        if value is None:
            plan.warnings.append({"kind": "non_boolean_object",
                                  "detail": f"{st.id}: {st.predicate} -> {obj_name!r}"})
        elif value is False:
            base.polarity = "false" if base.polarity == "true" else "true"
        return base

    # -- legacy is_a used for aliases ---------------------------------------
    if st.predicate == "is_a" and obj_name and subject_name and \
            obj_name.casefold().strip() == subject_name.casefold().strip():
        base.predicate = "alias_of"
        base.note = "legacy is_a between identical surfaces -> alias_of"
        return base

    base.predicate = mapping.target
    if mapping.definition:
        needed[mapping.target] = mapping.definition

    if mapping.object_is_literal:
        base.object_kind = "literal"
        base.object_legacy_id = ""
        if st.object_literal is None:
            base.note = ("legacy object was an entity record but the relation "
                         "takes a measurement; imported as a literal")
    elif st.object_entity is None:
        base.object_kind = "literal"
        if mapping.target == "has_property":
            plan.warnings.append({"kind": "property_literal",
                                  "detail": f"{st.id}: literal object on "
                                            f"has_property"})
    return base


def _detect_planned_conflicts(plan: ImportPlan) -> None:
    """Functional relations with >1 object for one subject WILL be conflicts.

    Reported in the dry run so nobody is surprised by the contradiction
    detector firing immediately after an import.
    """
    functional = {r["name"] for r in plan.relations if r.get("functional")}
    if not functional:
        return
    seen: dict[tuple[str, str], set[str]] = {}
    for s in plan.importable:
        if s.predicate in functional:
            seen.setdefault((s.subject_name, s.predicate), set()).add(s.object_name)
    for (subject, predicate), objs in sorted(seen.items()):
        if len(objs) > 1:
            plan.warnings.append({"kind": "planned_conflict",
                                  "detail": f"{subject} —{predicate}-> "
                                            f"{sorted(objs)}"})


# --------------------------------------------------------------------------
# ledger
# --------------------------------------------------------------------------

class ImportLedger:
    """Append-only record of applied imports, stored next to the KB."""

    def __init__(self, home: str | None = None):
        self.home = os.path.expanduser(home or Store().home)
        self.path = os.path.join(self.home, LEDGER_FILENAME)

    # -- io ---------------------------------------------------------------
    def load(self) -> dict:
        if not os.path.exists(self.path):
            return {"schema": LEDGER_SCHEMA, "entries": []}
        doc = read_json(self.path)
        if doc.get("schema") != LEDGER_SCHEMA:
            raise ImportError_(f"unsupported ledger schema "
                               f"{doc.get('schema')!r} in {self.path!r}")
        return doc

    def entries(self) -> list[dict]:
        return list(self.load().get("entries", []))

    def find(self, source_path: str, source_sha256: str) -> dict | None:
        key = (os.path.realpath(os.path.expanduser(source_path)), source_sha256)
        for e in self.entries():
            if (e.get("source_path"), e.get("source_sha256")) == key:
                return e
        return None

    def append(self, entry: dict) -> dict:
        doc = self.load()
        doc["entries"].append(entry)
        atomic_write_json(self.path, doc)
        return entry

    def get(self, ledger_id: str) -> dict | None:
        for e in self.entries():
            if e.get("id") == ledger_id:
                return e
        return None


def _new_ledger_id(plan: ImportPlan, entries: Iterable[dict]) -> str:
    n = sum(1 for _ in entries) + 1
    return f"imp_{n:04d}_{plan.digest()[:8]}"


# --------------------------------------------------------------------------
# apply
# --------------------------------------------------------------------------

def apply_plan(plan: ImportPlan, dm, store: Store | None = None,
               when: str = "", save: bool = True,
               backup: bool = True) -> ImportResult:
    """Materialise ``plan`` into ``dm`` (a v0.16 DialogueManager)."""
    when = when or _now()
    store = store or Store()
    ledger = ImportLedger(store.home)
    real_source = os.path.realpath(os.path.expanduser(plan.source_path))

    prior = ledger.find(real_source, plan.source_sha256)
    if prior is not None and prior.get("status") == "applied":
        return ImportResult(ledger_id=prior["id"], applied=False,
                            duplicate_of=prior["id"],
                            entities_before=len(dm.kg.resolver),
                            entities_after=len(dm.kg.resolver),
                            statements_before=len(dm.kg.all_statements()),
                            statements_after=len(dm.kg.all_statements()))

    before_entities = len(dm.kg.resolver)
    before_statements = len(dm.kg.all_statements())
    backup_paths: dict = {}
    if backup and store.exists():
        kb_bak = store.backup(tag="pre-import")
        backup_paths["kb"] = kb_bak
    if backup and store.context_exists():
        ctx_bak = store.backup(path=store.context_path, tag="pre-import")
        backup_paths["context"] = ctx_bak

    # -- 1. schema: register only the relations this plan needs ------------
    registered = _register_relations(dm, plan)

    # -- 2. entities -------------------------------------------------------
    id_map: dict[str, int] = {}
    for pe in plan.entities:
        ent = dm.kg.resolver.intern(pe.name, kind=pe.kind,
                                    source="import:v17", when=when)
        if not ent.kind:
            dm.kg.resolver.set_kind(ent.entity_id, pe.kind, when=when)
        id_map[pe.legacy_id] = ent.entity_id

    aliases_added = 0
    for pe in plan.entities:
        eid = id_map.get(pe.legacy_id)
        if eid is None:
            continue
        for alias in pe.aliases:
            ent = dm.kg.resolver.get(eid)
            if ent and alias and alias not in ent.aliases and alias != ent.canonical:
                dm.kg.resolver.add_alias(eid, alias, when=when, source="import:v17")
                aliases_added += 1

    # -- 3. statements -----------------------------------------------------
    added = reinforced = skipped = 0
    for ps in plan.statements:
        if not ps.will_import:
            skipped += 1
            continue
        subject_id = id_map.get(ps.subject_legacy_id)
        if subject_id is None:
            skipped += 1
            continue
        if ps.object_kind == "literal":
            obj_id = dm.kg.intern_literal(ps.object_name, when=when).entity_id
        else:
            obj_id = id_map.get(ps.object_legacy_id)
            if obj_id is None:
                skipped += 1
                continue
        if ps.predicate not in dm.kg.schema:
            skipped += 1        # registered above, but never assume
            continue
        st, created = dm.kg.assert_triple(
            subject=subject_id, predicate=ps.predicate, obj=obj_id,
            polarity=ps.polarity, confidence=ps.confidence,
            source=ps.source, when=when,
            provenance={**ps.provenance,
                        "legacy_statement_id": ps.legacy_id,
                        "legacy_predicate": ps.legacy_predicate,
                        "note": ps.note})
        dm.evidence.add(st.statement_id, kind=EV_IMPORT,
                        utterance=(ps.provenance.get("legacy") or {}).get(
                            "provenance", {}).get("fragment", ""),
                        turn_id=f"import:{ledger.home}",
                        source=ps.source, when=when,
                        note=f"legacy {ps.legacy_id}")
        if created:
            added += 1
        else:
            reinforced += 1

    # -- 4. learned routing (artifact imports only) ------------------------
    hints_written = 0
    alias_applied = 0
    if plan.kind == "learned" and (plan.predicate_hints or plan.alias_delta):
        from ..routing import RoutingStore
        routing = RoutingStore(store.home)
        hints_written = routing.merge_predicate_hints(plan.predicate_hints)
        for alias, canonical in plan.alias_delta.items():
            res = dm.kg.resolver.resolve(canonical, allow_fuzzy=False)
            if res.entity_id is None:
                continue                     # nothing to attach the name to
            surface = dm.kg.resolver.resolve(alias, allow_fuzzy=False)
            if surface.entity_id is None:
                # the corrected surface is a NAME, not an entity: register it
                # as an alias of the canonical record
                dm.kg.resolver.add_alias(res.entity_id, alias, when=when,
                                         source="import:artifact")
                alias_applied += 1
            elif surface.entity_id != res.entity_id:
                # ambiguity control: never fold two existing entities because
                # an artifact says so — that would need an explicit act_teach
                plan.warnings.append({
                    "kind": "alias_delta_ambiguous",
                    "detail": f"{alias!r} already designates entity "
                              f"#{surface.entity_id}; not re-pointed to "
                              f"#{res.entity_id}"})
            else:
                alias_applied += 1

    if save:
        dm.store = dm.store or store
        saved = store.save(dm)
    else:
        saved = {}

    result = ImportResult(
        ledger_id="", applied=True,
        entities_before=before_entities, entities_after=len(dm.kg.resolver),
        statements_before=before_statements,
        statements_after=len(dm.kg.all_statements()),
        statements_added=added, statements_reinforced=reinforced,
        statements_skipped=skipped, aliases_added=aliases_added,
        relations_registered=registered,
        predicate_hints_written=hints_written,
        alias_delta_applied=alias_applied,
        backup=backup_paths, saved=saved)

    ledger_id = _new_ledger_id(plan, ledger.entries())
    result.ledger_id = ledger_id
    ledger.append({
        "id": ledger_id,
        "schema": LEDGER_SCHEMA,
        "status": "applied",
        "applied_at": when,
        "source_path": real_source,
        "source_sha256": plan.source_sha256,
        "source_schema": plan.source_schema,
        "plan_digest": plan.digest(),
        "plan": plan.summary(),
        "result": result.summary(),
        "backup": backup_paths,
    })
    return result


def _register_relations(dm, plan: ImportPlan) -> list[str]:
    """Register the plan's relations and the inverses they name.

    Registering the inverse as well is not cosmetic: ``RelationSchema``
    resolves ``inverse`` by name, and a name that is not registered silently
    disables inverse traversal for that relation.
    """
    catalog = {d["name"]: d for d in custom_relations()}
    registered: list[str] = []
    queue = [dict(d) for d in plan.relations]
    seen: set[str] = set()
    while queue:
        definition = queue.pop(0)
        name = definition["name"]
        if name in seen:
            continue
        seen.add(name)
        if name not in dm.kg.schema:
            dm.kg.schema.register_dict(dict(definition))
            registered.append(name)
        inverse = definition.get("inverse") or ""
        if inverse and inverse not in dm.kg.schema and inverse in catalog:
            queue.append(dict(catalog[inverse]))
    dm.inference.schema = dm.kg.schema
    dm.parser.schema = dm.kg.schema
    return registered


def restore(ledger_id: str, store: Store, home: str | None = None) -> dict:
    """Undo one import by copying its pre-import backup back into place."""
    ledger = ImportLedger(home or store.home)
    entry = ledger.get(ledger_id)
    if entry is None:
        raise ImportError_(f"unknown ledger id {ledger_id!r}")
    backup = entry.get("backup") or {}
    if not backup.get("kb"):
        raise ImportError_(f"ledger {ledger_id!r} has no pre-import backup")
    restored = {}
    for target, bak in backup.items():
        if not bak or not os.path.exists(bak):
            continue
        path = store.kb_path if target == "kb" else store.context_path
        atomic_write_json(path, read_json(bak))
        restored[target] = path
    doc = ledger.load()
    for e in doc["entries"]:
        if e.get("id") == ledger_id:
            e["status"] = "restored"
            e["restored_at"] = _now()
    atomic_write_json(ledger.path, doc)
    return restored
