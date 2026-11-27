"""v0.16 — command line interface.

    python -m rasm_cim_v16 chat [--answer normal] [--home ~/.rasm-cim]
    python -m rasm_cim_v16 ask  "Кто такой Исаак Ньютон?" --answer detailed
    python -m rasm_cim_v16 teach "Исаак Ньютон — это физик"
    python -m rasm_cim_v16 say  "Ньютон — физик?"
    python -m rasm_cim_v16 migrate [--dry-run] [--home DIR]
    python -m rasm_cim_v16 rollback --backup kb.json.bak-YYYYmmdd-HHMMSS
    python -m rasm_cim_v16 inspect | stats | entities | graph | memory | rules
    python -m rasm_cim_v16 demo            guided scenario (Newton corpus)

Answer modes: --answer concise | normal | detailed
              (legacy aliases short/brief/full/verbose/long also accepted)

Storage: ~/.rasm-cim/{kb.json,context.json}, overridable with --home or
$RASM_CIM_HOME.  No network, no LLM, stdlib only.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)          # computed, never hardcoded

from rasm_cim_v16 import __version__                      # noqa: E402
from rasm_cim_v16.answer_engine import MODES, MODE_ALIASES  # noqa: E402
from rasm_cim_v16.dialogue_manager import DialogueManager  # noqa: E402
from rasm_cim_v16.persistence import Store, default_home   # noqa: E402


def _clock() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def _build(home: str | None, autosave: bool, max_depth: int,
           max_nodes: int) -> tuple[DialogueManager, Store]:
    dm = DialogueManager(clock=_clock, max_depth=max_depth,
                         max_nodes=max_nodes)
    store = Store(home=home, clock=_clock, autosave=autosave)
    store.attach(dm)
    if store.exists() or store.context_exists():
        notes = store.load(dm)
        for k, v in notes.items():
            for n in v:
                if "migrated" in n:
                    print(f"[migration:{k}] {n}", file=sys.stderr)
    return dm, store


def _norm_mode(mode: str | None) -> str:
    if not mode:
        return "normal"
    return MODE_ALIASES.get(mode.lower(), mode.lower())


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------

def cmd_chat(a) -> int:
    dm, store = _build(a.home, not a.no_autosave, a.max_depth, a.max_nodes)
    mode = _norm_mode(a.answer)
    dm.default_mode = mode
    print(f"RASM-CIM v{__version__} — режим {mode}, база: {store.home}")
    print("Команды: :help   Выход: :quit / Ctrl-D")
    while True:
        try:
            line = input("ты> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line.lower() in (":quit", ":exit", "exit", "quit"):
            break
        ans = dm.say(line)
        print(f"rasm[{ans.status}]> {ans.render(mode)}")
        if a.show_status:
            print(f"    (уверенность {ans.confidence:.2f}, фактов "
                  f"{len(ans.facts)}, доказательств {len(ans.evidence)})")
    try:
        info = dm.save()
        print(f"Сохранено: {info.get('kb')} ({info.get('kb_bytes')} Б), "
              f"{info.get('context')} ({info.get('context_bytes')} Б)")
    except Exception as e:                              # noqa: BLE001
        print(f"Не удалось сохранить: {e}", file=sys.stderr)
        return 1
    return 0


def cmd_say(a) -> int:
    dm, store = _build(a.home, not a.no_autosave, a.max_depth, a.max_nodes)
    mode = _norm_mode(a.answer)
    # the mode must reach the builder, not only the renderer: an Answer is
    # constructed with all three renderings, and building it in the default
    # mode while printing "detailed" silently degrades the output
    ans = dm.say(a.text, mode)
    print(ans.render(mode))
    if a.json:
        print(json.dumps(ans.as_dict(), ensure_ascii=False, indent=1))
    dm.save()
    return 0 if ans.status != "AMBIGUOUS" else 3


cmd_ask = cmd_say
cmd_teach = cmd_say


def cmd_migrate(a) -> int:
    from rasm_cim_v16.migrate import migrate_context_file, migrate_file
    store = Store(home=a.home, clock=_clock)
    rep = migrate_file(store.kb_path, dry_run=a.dry_run)
    crep = migrate_context_file(store.context_path, dry_run=a.dry_run)
    print(json.dumps({"kb": rep, "context": crep}, ensure_ascii=False, indent=1))
    return 0


def cmd_rollback(a) -> int:
    store = Store(home=a.home, clock=_clock)
    bak = a.backup
    if not os.path.isabs(bak):
        bak = os.path.join(store.home, os.path.basename(bak))
    n = store.rollback(bak)
    print(f"Восстановлено {n} Б из {bak} в {store.kb_path}")
    return 0


def cmd_inspect(a) -> int:
    dm, _ = _build(a.home, False, a.max_depth, a.max_nodes)
    parts = list(a.arg or [])
    name = a.what
    # `inspect ":entities"` and `entities` must reach the same handler: the
    # subcommand name is the default, a leading-colon argument overrides it,
    # and the rest is the command's own argument (`graph Ньютон`)
    if parts and parts[0].startswith(":"):
        name = parts[0][1:]
        parts = parts[1:]
    print(dm.command(name, " ".join(parts), _norm_mode(a.answer)).text)
    return 0


def cmd_demo(a) -> int:
    """Guided scenario: teach, alias, multi-step inference, pronouns, topic
    switch, contradiction, UNKNOWN.  Deterministic, no network."""
    import tempfile
    home = a.home or os.path.join(tempfile.mkdtemp(prefix="rasm-demo-"),
                                  ".rasm-cim")
    dm, store = _build(home, True, a.max_depth, a.max_nodes)
    mode = _norm_mode(a.answer)
    script = [
        "teach Исаак Ньютон — это физик",
        "teach Ньютон — это Исаак Ньютон",
        "teach Исаак Ньютон открыл закон всемирного тяготения",
        "teach Исаак Ньютон сформулировал законы движения",
        "teach Физик — это учёный",
        "Кто такой Исаак Ньютон?",
        "Ньютон — физик?",
        "Ньютон — учёный?",
        "Что он открыл?",
        "teach Компьютер — это устройство",
        "Что такое компьютер?",
        "Расскажи о Ньютоне",
        "teach Исаак Ньютон не был физиком",
        "Ньютон — физик?",
        "Кто написал Войну и мир?",
    ]
    for line in script:
        ans = dm.say(line)
        print(f"\nты> {line}\nrasm[{ans.status}]> {ans.render(mode)}")
    print("\n--- :context ---")
    print(dm.command("context", "", mode).text)
    print("\n--- :memory ---")
    print(dm.command("memory", "", mode).text)
    info = dm.save()
    print(f"\nСохранено в {info['kb']} ({info['kb_bytes']} Б)")
    return 0


# --------------------------------------------------------------------------

#: Defaults for the options shared by every subcommand.  They are applied on
#: the top-level parser, NOT as action defaults, because argparse re-applies a
#: subparser's defaults over values the parent already parsed — with a real
#: default here, `--answer detailed ask "..."` would silently become "normal".
_SHARED_DEFAULTS = {
    "home": None,
    "answer": "normal",
    "json": False,
    "show_status": False,
    "no_autosave": False,
    "max_depth": 3,
    "max_nodes": 400,
}


def _common() -> argparse.ArgumentParser:
    """Options accepted both before and after the subcommand."""
    c = argparse.ArgumentParser(add_help=False)
    c.add_argument("--home", default=argparse.SUPPRESS,
                   help=f"storage directory (default {default_home()})")
    c.add_argument("--answer", default=argparse.SUPPRESS,
                   help="concise | normal | detailed (aliases accepted)")
    c.add_argument("--json", action="store_true", default=argparse.SUPPRESS,
                   help="also dump the raw answer as JSON")
    c.add_argument("--show-status", action="store_true",
                   default=argparse.SUPPRESS,
                   help="print confidence/fact counts after each reply")
    c.add_argument("--no-autosave", action="store_true",
                   default=argparse.SUPPRESS,
                   help="do not write after every turn (chat saves on exit)")
    c.add_argument("--max-depth", type=int, default=argparse.SUPPRESS,
                   help="inference depth cap (default 3)")
    c.add_argument("--max-nodes", type=int, default=argparse.SUPPRESS,
                   help="inference node budget (default 400)")
    return c


def build_parser() -> argparse.ArgumentParser:
    common = _common()
    ap = argparse.ArgumentParser(
        prog="rasm_cim_v16", parents=[common],
        description=f"RASM-CIM v{__version__} — typed knowledge graph, "
                    f"inference and dialogue context (stdlib only, offline)")
    ap.add_argument("--version", action="version",
                    version=f"rasm-cim {__version__}")
    sub = ap.add_subparsers(dest="cmd")

    sub.add_parser("chat", parents=[common],
                   help="interactive dialogue").set_defaults(f=cmd_chat)
    for name, helpt in (("ask", "answer one question"),
                        ("say", "handle one utterance (question or fact)"),
                        ("teach", "record one fact")):
        q = sub.add_parser(name, parents=[common], help=helpt)
        q.add_argument("text")
        q.set_defaults(f=cmd_say)
    m = sub.add_parser("migrate", parents=[common],
                       help="migrate a legacy kb.json to v2")
    m.add_argument("--dry-run", action="store_true")
    m.set_defaults(f=cmd_migrate)
    r = sub.add_parser("rollback", parents=[common],
                       help="restore kb.json from a backup")
    r.add_argument("--backup", required=True)
    r.set_defaults(f=cmd_rollback)
    for what in ("inspect", "stats", "entities", "graph", "memory", "rules",
                 "context", "conflicts", "evidence", "explain", "help"):
        q = sub.add_parser(what, parents=[common],
                           help=f"same as the :{what} command")
        q.add_argument("arg", nargs="*", default=[])
        q.set_defaults(f=cmd_inspect, what=what)
    sub.add_parser("demo", parents=[common],
                   help="guided scenario on a temporary KB").set_defaults(f=cmd_demo)
    return ap


def main(argv=None) -> int:
    ap = build_parser()
    a = ap.parse_args(argv)
    for k, v in _SHARED_DEFAULTS.items():
        if not hasattr(a, k):
            setattr(a, k, v)
    if getattr(a, "f", None) is None:
        a.f = cmd_chat
    if _norm_mode(a.answer) not in MODES:
        print(f"Неизвестный режим ответа {a.answer!r}; допустимы {MODES}",
              file=sys.stderr)
        return 2
    try:
        return a.f(a)
    except KeyboardInterrupt:
        print("\nпрервано", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
