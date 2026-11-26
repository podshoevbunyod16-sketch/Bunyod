"""v0.12 CLI — KB management (additive; nothing from v0.10/v0.11 removed).

  python -m rasm_cim_v12.cli kb-init    PATH
  python -m rasm_cim_v12.cli teach      PATH --subject Франция --predicate столица --object Париж
  python -m rasm_cim_v12.cli kb-query   PATH --subject Франция [--predicate capital_of] [--object ...]
  python -m rasm_cim_v12.cli conflicts  PATH
  python -m rasm_cim_v12.cli aliases    PATH "Париж"
  python -m rasm_cim_v12.cli import-jsonl PATH FILE [--strict]
  python -m rasm_cim_v12.cli export-jsonl PATH [OUT]
  python -m rasm_cim_v12.cli kb-info    PATH
  python -m rasm_cim_v12.cli seed-demo  PATH     # offline demo facts (v0.11 bundle)

Exit codes: 0 ok · 2 input/usage error · 3 not found · 4 ambiguous reference
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from .knowledge import (KnowledgeError, KnowledgeStore, Provenance, export_jsonl,
                        import_jsonl, teach, retract, canon_predicate)


def _open(path, init=True):
    # init() is idempotent: creates dir+meta if absent, then replays events
    return KnowledgeStore.init(path)


def _entity_name(store, eid):
    e = store.get_entity(eid)
    return e.name if e else eid


def cmd_kb_init(a):
    st = _open(a.path, init=True)
    print(json.dumps(st.stats(), ensure_ascii=False))
    return 0


def cmd_teach(a):
    st = _open(a.path)
    prov = Provenance(source=a.source, doc_id=a.doc_id or f"cli:{a.subject}",
                      fragment=a.fragment or "", method=a.method,
                      reliability=a.reliability,
                      added_at=__import__("time").strftime("%Y-%m-%dT%H:%M:%SZ",
                                                           __import__("time").gmtime()))
    try:
        res = teach(st, a.subject, a.predicate, a.object,
                    object_is_entity=not a.object_literal,
                    subject_type=a.subject_type, object_type=a.object_type,
                    kind=a.kind, polarity=("deny" if a.deny else "assert"),
                    provenance=prov)
    except KnowledgeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(res, ensure_ascii=False, indent=2))
    return 0


def cmd_query(a):
    st = _open(a.path)
    subj_id = None
    if a.subject:
        status, e, cand = st.find_by_name(a.subject, a.subject_type)
        if status == "ambiguous":
            print(f"ambiguous subject {a.subject!r}; candidates: "
                  + ", ".join(f"{c} ({_entity_name(st, c)}/{st.entities[c].type})" for c in cand),
                  file=sys.stderr)
            return 4
        if status == "not_found":
            print(f"entity not found: {a.subject!r}", file=sys.stderr)
            return 3
        subj_id = e.id
    obj_id = None
    if a.object:
        status, e, cand = st.find_by_name(a.object)
        if status == "found":
            obj_id = e.id
    rows = st.query_statements(subject=subj_id, predicate=a.predicate,
                               object=obj_id or a.object_literal,
                               include_retracted=a.all,
                               include_hypotheses=True)
    out = []
    for s in rows:
        out.append({"id": s.id, "subject": _entity_name(st, s.subject),
                    "predicate": s.predicate,
                    "object": _entity_name(st, s.object_entity) if s.object_entity else s.object_literal,
                    "polarity": s.polarity, "kind": s.kind, "status": s.status,
                    "provenance": s.provenance})
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


def cmd_conflicts(a):
    st = _open(a.path)
    conflicts = st.conflicts()
    print(json.dumps(conflicts, ensure_ascii=False, indent=2))
    return 0


def cmd_aliases(a):
    st = _open(a.path)
    status, e, cand = st.find_by_name(a.name)
    out = {"status": status,
           "entity": e.to_dict() if e else None,
           "candidates": [st.entities[c].to_dict() for c in cand]}
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 4 if status == "ambiguous" else (3 if status == "not_found" else 0)


def cmd_import(a):
    st = _open(a.path)
    report = import_jsonl(st, a.file, strict=a.strict)
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    return 0 if not report.get("fatal") else 2


def cmd_export(a):
    st = _open(a.path)
    payload = export_jsonl(st, a.out)
    if not a.out:
        sys.stdout.write(payload)
    else:
        print(json.dumps({"written": a.out, "bytes": len(payload.encode("utf-8"))}))
    return 0


def cmd_info(a):
    st = _open(a.path)
    stats = st.stats()
    stats["path"] = a.path
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


def cmd_retract(a):
    st = _open(a.path)
    try:
        res = retract(st, a.statement_id, reason=a.reason)
    except KnowledgeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(res, ensure_ascii=False))
    return 0


def cmd_seed_demo(a):
    st = _open(a.path)
    demo_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "rasm_cim_v11", "data", "demo_facts.jsonl")
    type_map = {"capital_of": ("place", "place"), "is_a": ("thing", "concept"),
                "написал": ("person", "thing"), "может_летать": ("thing", "concept"), "является_столицей": ("place", "place")}
    # v0.14 consistency fix: same entity name must get ONE type across the
    # whole seed (person names were typed thing in is_a and person in написал,
    # producing silent duplicates). Seed is demo data hygiene, not a merge.
    persons = {"Пушкин", "Толстой", "Достоевский", "Ломоносов", "Гагарин"}
    n = 0
    for line in open(demo_path, encoding="utf-8"):
        f = json.loads(line)
        st_types = type_map.get(f["relation"], ("thing", "thing"))
        st_types = ("person", st_types[1]) if f["subject"] in persons else st_types
        st_types = (st_types[0], "person") if f["object"] in persons else st_types
        obj = f["object"]
        obj_is_entity = not obj[0].isdigit()
        try:
            teach(st, f["subject"], f["relation"], obj,
                  object_is_entity=obj_is_entity,
                  subject_type=st_types[0], object_type=st_types[1],
                  polarity="assert" if f["polarity"] == "true" else "deny",
                  provenance=Provenance(source="demo_corpus:v0.11",
                                        doc_id="demo_facts.jsonl",
                                        fragment=json.dumps(f, ensure_ascii=False),
                                        method="import",
                                        parser_version="seed-0.12.0"))
            n += 1
        except KnowledgeError as exc:
            print(f"seed skip: {exc}", file=sys.stderr)
    print(json.dumps({"seeded": n, "path": a.path}, ensure_ascii=False))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="rasm_cim_v12.cli",
                                 description="RASM-CIM v0.12 KB (offline)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("kb-init"); p.add_argument("path"); p.set_defaults(f=cmd_kb_init)

    p = sub.add_parser("teach"); p.add_argument("path")
    p.add_argument("--subject", required=True); p.add_argument("--predicate", required=True)
    p.add_argument("--object", required=True); p.add_argument("--deny", action="store_true")
    p.add_argument("--kind", default="user_assertion",
                   choices=["user_assertion", "extracted_fact", "hypothesis"])
    p.add_argument("--subject-type", default="thing"); p.add_argument("--object-type", default="thing")
    p.add_argument("--object-literal", action="store_true")
    p.add_argument("--source", default="user_teach"); p.add_argument("--doc-id", default=None)
    p.add_argument("--fragment", default=None); p.add_argument("--method", default="manual")
    p.add_argument("--reliability", default="medium", choices=["low", "medium", "high"])
    p.set_defaults(f=cmd_teach)

    p = sub.add_parser("kb-query"); p.add_argument("path")
    p.add_argument("--subject"); p.add_argument("--subject-type", default=None)
    p.add_argument("--predicate", default=None); p.add_argument("--object", default=None)
    p.add_argument("--object-literal", default=None)
    p.add_argument("--all", action="store_true", help="include retracted")
    p.set_defaults(f=cmd_query)

    p = sub.add_parser("conflicts"); p.add_argument("path"); p.set_defaults(f=cmd_conflicts)
    p = sub.add_parser("aliases"); p.add_argument("path"); p.add_argument("name"); p.set_defaults(f=cmd_aliases)
    p = sub.add_parser("import-jsonl"); p.add_argument("path"); p.add_argument("file")
    p.add_argument("--strict", action="store_true"); p.set_defaults(f=cmd_import)
    p = sub.add_parser("export-jsonl"); p.add_argument("path"); p.add_argument("out", nargs="?", default=None)
    p.set_defaults(f=cmd_export)
    p = sub.add_parser("kb-info"); p.add_argument("path"); p.set_defaults(f=cmd_info)
    p = sub.add_parser("retract"); p.add_argument("path"); p.add_argument("statement_id")
    p.add_argument("--reason", default=""); p.set_defaults(f=cmd_retract)
    p = sub.add_parser("seed-demo"); p.add_argument("path"); p.set_defaults(f=cmd_seed_demo)

    a = ap.parse_args(argv)
    return a.f(a)


if __name__ == "__main__":
    sys.exit(main())
