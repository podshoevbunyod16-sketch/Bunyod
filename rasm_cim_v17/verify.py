"""Integrity self-check for a persisted v0.16/v0.17 database.

The check is *structural*, not statistical: it can prove that the database is
internally consistent (no dangling ids, no unregistered relation, no evidence
row orphaned from its statement) and it can prove that something is wrong.  It
cannot prove that the facts are true — nothing can.

    errors    — the document contradicts itself; an answer built on it may be
                wrong for reasons that have nothing to do with the question
    warnings  — policy issues that are legal but worth a human look
                (missing evidence, type-constraint violations, hints pointing
                at relations that do not exist, imports whose source vanished)
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

from rasm_cim_v16.persistence import Store, read_json

from .interop.importer import ImportLedger
from .routing import RoutingStore


@dataclass
class VerifyReport:
    ok: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)
        self.ok = False

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)

    def render(self) -> str:
        head = "ПРОВЕРКА: OK" if self.ok else "ПРОВЕРКА: ОШИБКИ"
        lines = [head]
        for k, v in sorted(self.stats.items()):
            lines.append(f"  {k}: {v}")
        if self.errors:
            lines.append(f"  ошибки ({len(self.errors)}):")
            lines.extend(f"    ! {e}" for e in self.errors[:40])
            if len(self.errors) > 40:
                lines.append(f"    … ещё {len(self.errors) - 40}")
        if self.warnings:
            lines.append(f"  предупреждения ({len(self.warnings)}):")
            lines.extend(f"    - {w}" for w in self.warnings[:40])
            if len(self.warnings) > 40:
                lines.append(f"    … ещё {len(self.warnings) - 40}")
        return "\n".join(lines)

    def as_dict(self) -> dict:
        return {"ok": self.ok, "errors": self.errors,
                "warnings": self.warnings, "stats": self.stats}


def verify_home(home: str | None = None) -> VerifyReport:
    """Open the database at ``home`` and check it without mutating it."""
    from rasm_cim_v16.dialogue_manager import DialogueManager

    store = Store(home=home)
    dm = DialogueManager()
    rep = VerifyReport()
    if not store.exists():
        rep.add_warning(f"нет базы по пути {store.kb_path} — проверять нечего")
        rep.stats["kb_exists"] = False
        return rep
    kb = read_json(store.kb_path)
    rep.stats["schema"] = kb.get("schema")
    rep.stats["kb_bytes"] = os.path.getsize(store.kb_path)
    if not isinstance(kb.get("schema"), int):
        rep.add_error(f"kb.json: поле schema не целое ({kb.get('schema')!r})")

    notes = store.load(dm)
    for bucket, items in notes.items():
        for n in items:
            rep.add_warning(f"{bucket}: {n}")

    kg, ev = dm.kg, dm.evidence
    resolver_ids = {e.entity_id for e in kg.resolver.all_entities()}
    statements = kg.all_statements(include_retracted=True,
                                   include_redundant=True)
    direct = [s for s in statements if s.status == "direct"]

    rep.stats.update({
        "kb_exists": True,
        "entities": len(resolver_ids),
        "statements": len(statements),
        "direct": len(direct),
        "retracted": len([s for s in statements if s.status == "retracted"]),
        "redundant": len([s for s in statements if s.status == "redundant"]),
        "evidence": len(ev),
        "relations_registered": len(kg.schema.names()),
        "custom_relations": len(kg.schema.custom_names()),
    })

    # -- 1. referential integrity -----------------------------------------
    for st in statements:
        if st.subject not in resolver_ids:
            rep.add_error(f"{st.statement_id}: субъект #{st.subject} не существует в базе")
        if st.object not in resolver_ids:
            rep.add_error(f"{st.statement_id}: объект #{st.object} не существует в базе")
        if st.predicate not in kg.schema:
            rep.add_error(f"{st.statement_id}: отношение {st.predicate!r} не зарегистрировано")

    # -- 2. duplicate keys -------------------------------------------------
    seen: dict[tuple, str] = {}
    for st in direct:
        key = st.key()
        if key in seen:
            rep.add_error(f"дубль прямого утверждения {key}: {seen[key]} и {st.statement_id}")
        else:
            seen[key] = st.statement_id

    # -- 3. evidence -------------------------------------------------------
    statement_ids = {s.statement_id for s in statements}
    with_evidence: set[str] = set()
    for e in ev.all():
        if e.statement_id not in statement_ids:
            rep.add_error(f"источник {e.evidence_id} ссылается на несуществующее "
                          f"утверждение {e.statement_id}")
        else:
            with_evidence.add(e.statement_id)
    unevidenced = [s for s in direct if s.statement_id not in with_evidence]
    if unevidenced:
        rep.stats["direct_without_evidence"] = len(unevidenced)
        rep.add_warning(f"{len(unevidenced)} прямых утверждений без записи об "
                        f"источнике (например {unevidenced[0].statement_id})")

    # -- 4. type constraints ----------------------------------------------
    violations = 0
    for st in direct:
        rel = kg.schema.get(st.predicate)
        if rel is None:
            continue
        skind = (kg.resolver.get(st.subject) or None)
        okind = (kg.resolver.get(st.object) or None)
        if skind and okind and not rel.allows(skind.kind, okind.kind):
            violations += 1
    rep.stats["type_violations"] = violations
    if violations:
        rep.add_warning(f"{violations} утверждений нарушают ограничения типов "
                        f"своего отношения (уверенность уже понижена при записи)")

    # -- 5. routing hints ---------------------------------------------------
    try:
        routing = RoutingStore(store.home)
        hints = routing.hints()
        rep.stats["routing_hints"] = len(hints)
        dead = {q: p for q, p in hints.items() if p not in kg.schema}
        if dead:
            rep.add_warning(f"{len(dead)} подсказок маршрутизации ссылаются на "
                            f"незарегистрированные отношения: "
                            f"{sorted(set(dead.values()))}")
    except (OSError, ValueError) as exc:
        rep.add_warning(f"routing.json не читается: {exc}")

    # -- 6. import ledger ---------------------------------------------------
    try:
        ledger = ImportLedger(store.home)
        entries = ledger.entries()
        rep.stats["imports"] = len([e for e in entries
                                    if e.get("status") == "applied"])
        for e in entries:
            if e.get("status") != "applied":
                continue
            src = e.get("source_path", "")
            if src and not os.path.exists(src):
                rep.add_warning(f"источник импорта {e.get('id')} больше не "
                                f"существует: {src}")
    except (OSError, ValueError) as exc:
        rep.add_warning(f"import_ledger.json не читается: {exc}")

    # -- 7. context ---------------------------------------------------------
    if store.context_exists():
        ctx = read_json(store.context_path)
        active = ctx.get("active_entities") or ctx.get("focus") or []
        if isinstance(active, list):
            dangling = [a for a in active if isinstance(a, int)
                        and a not in resolver_ids]
            if dangling:
                rep.add_warning(f"context.json ссылается на "
                                f"{len(dangling)} несуществующих сущностей")
    return rep


def verify_dm(dm) -> VerifyReport:
    """Same checks for an in-memory manager (used by tests and by the CLI in
    ``--no-save`` runs)."""
    rep = VerifyReport()
    resolver_ids = {e.entity_id for e in dm.kg.resolver.all_entities()}
    statements = dm.kg.all_statements(include_retracted=True,
                                      include_redundant=True)
    rep.stats = {"entities": len(resolver_ids), "statements": len(statements),
                 "evidence": len(dm.evidence)}
    for st in statements:
        if st.subject not in resolver_ids:
            rep.add_error(f"{st.statement_id}: субъект #{st.subject} "
                          f"не существует в базе")
        if st.object not in resolver_ids:
            rep.add_error(f"{st.statement_id}: объект #{st.object} "
                          f"не существует в базе")
        if st.predicate not in dm.kg.schema:
            rep.add_error(f"{st.statement_id}: отношение {st.predicate!r} не зарегистрировано")
    return rep
