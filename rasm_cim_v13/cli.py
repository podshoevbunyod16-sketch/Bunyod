"""v0.13 CLI — dialogue commands (additive).

  python -m rasm_cim_v13.cli ask CTX_DIR --kb KB_DIR "Какая столица у Франции?"
  python -m rasm_cim_v13.cli show-context CTX_DIR
  python -m rasm_cim_v13.cli reset-context CTX_DIR
  python -m rasm_cim_v13.cli export-session CTX_DIR [OUT]

Exit codes: 0 ok · 2 input · 3 not found · 4 ambiguous/clarify (needs user input)
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from rasm_cim_v12.knowledge import KnowledgeStore

from .dialogue import ContextStore, DialogueManager, export_session


def _ctx(path: str) -> ContextStore:
    return ContextStore(os.path.join(path, "context.json"))


def cmd_ask(a):
    kb = KnowledgeStore.init(a.kb)
    ctx = _ctx(a.ctx)
    dm = DialogueManager(kb, ctx)
    out = dm.process(a.text)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    needs = any(r["status"] == "CLARIFY" for r in out["results"])
    return 4 if needs else 0


def cmd_show(a):
    print(json.dumps(_ctx(a.ctx).summary(), ensure_ascii=False, indent=2))
    return 0


def cmd_reset(a):
    _ctx(a.ctx).reset()
    print("context reset")
    return 0


def cmd_export(a):
    out = export_session(_ctx(a.ctx), a.out)
    if not a.out:
        print(json.dumps(out, ensure_ascii=False, indent=2))
    else:
        print(json.dumps({"written": a.out}))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="rasm_cim_v13.cli",
                                 description="RASM-CIM v0.13 dialogue (offline)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("ask"); p.add_argument("ctx"); p.add_argument("text")
    p.add_argument("--kb", required=True); p.set_defaults(f=cmd_ask)
    p = sub.add_parser("show-context"); p.add_argument("ctx"); p.set_defaults(f=cmd_show)
    p = sub.add_parser("reset-context"); p.add_argument("ctx"); p.set_defaults(f=cmd_reset)
    p = sub.add_parser("export-session"); p.add_argument("ctx"); p.add_argument("out", nargs="?", default=None)
    p.set_defaults(f=cmd_export)
    a = ap.parse_args(argv)
    return a.f(a)


if __name__ == "__main__":
    sys.exit(main())
