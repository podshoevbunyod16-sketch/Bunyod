"""v0.17 — the single command line for the whole stack.

    python -m rasm_cim_v17 chat                     интерактивный диалог
    python -m rasm_cim_v17 ask  "Кто такой Ньютон?"
    python -m rasm_cim_v17 teach "Ньютон — это физик"
    python -m rasm_cim_v17 say  "Ньютон — физик?"
    python -m rasm_cim_v17 stats | entities | graph | evidence | conflicts
    python -m rasm_cim_v17 import SOURCE [--dry-run] [--kind auto|kb|facts|learned]
    python -m rasm_cim_v17 imports                 журнал импортов
    python -m rasm_cim_v17 restore LEDGER_ID      откат одного импорта
    python -m rasm_cim_v17 verify [--json]         проверка целостности
    python -m rasm_cim_v17 learned DIR             сводка обучающего артефакта
    python -m rasm_cim_v17 migrate [--dry-run]     миграция схемы v1 -> v2
    python -m rasm_cim_v17 version

``python -m rasm_cim`` also lands here (v0.17 added that entry point).  Storage:
``~/.rasm-cim/`` (``$RASM_CIM_HOME`` or ``--home``).  No network, stdlib only.
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

from rasm_cim_v16 import __version__ as CORE_VERSION            # noqa: E402
from rasm_cim_v16.answer_engine import MODE_ALIASES             # noqa: E402
from rasm_cim_v16.dialogue_manager import DialogueManager       # noqa: E402
from rasm_cim_v16.persistence import Store, default_home        # noqa: E402
from rasm_cim_v17 import __version__                            # noqa: E402
from rasm_cim_v17.interop import build_plan, apply_plan         # noqa: E402
from rasm_cim_v17.interop.importer import ImportLedger, restore  # noqa: E402
from rasm_cim_v17.interop.learned import LearnedSnapshot        # noqa: E402
from rasm_cim_v17.pipeline import SemanticPipeline              # noqa: E402
from rasm_cim_v17.semantics import (DEFAULT_CORPUS, DEFAULT_SEED,  # noqa: E402
                                    DEFAULT_V11, build_demo, load_cases,
                                    render_report, run_all, seed_report)
from rasm_cim_v17.verify import verify_home                     # noqa: E402


def _clock() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def _build(home: str | None, mode: str = "normal") -> tuple[DialogueManager, Store]:
    dm = DialogueManager(clock=_clock)
    dm.default_mode = MODE_ALIASES.get(mode, mode)
    store = Store(home=home, clock=_clock, autosave=False)
    store.attach(dm)
    if store.exists() or store.context_exists():
        dm.autosave = True
        store.load(dm)
    return dm, store


def _responder(dm, semantic: bool):
    """Return ``(respond, pipeline)`` — one place that decides delegation."""
    if not semantic:
        return dm.say, None
    pipe = SemanticPipeline(dm)
    return pipe.respond, pipe


def _print_json(obj) -> None:
    print(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=False))


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------

def cmd_version(a) -> int:
    print(f"rasm-cim v{__version__} (ядро v{CORE_VERSION}, схема kb 2)")
    print(f"python {sys.version.split()[0]} · stdlib only · offline")
    print(f"дом: {os.path.expanduser(a.home or default_home())}")
    return 0


def cmd_chat(a) -> int:
    dm, store = _build(a.home, a.answer)
    respond, _ = _responder(dm, not getattr(a, "core", False))
    print(f"RASM-CIM v{__version__} — режим {dm.default_mode}, база: {store.home}")
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
        ans = respond(line)
        print(f"система> {ans.render(dm.default_mode)}")
    if store.exists() or dm.kg.all_statements():
        store.save(dm)
        print(f"[сохранено] {store.kb_path}")
    return 0


def cmd_ask(a) -> int:
    dm, store = _build(a.home, a.answer)
    respond, pipe = _responder(dm, not getattr(a, "core", False))
    ans = respond(a.text)
    if pipe is not None and pipe.last_meaning is not None and a.explain_meaning:
        print(f"[смысл] {pipe.last_meaning.describe()}", file=sys.stderr)
        print(f"[делегировано ядру: {'да' if pipe.last_delegated else 'нет'}]",
              file=sys.stderr)
    if a.json:
        _print_json({"text": ans.text, "status": ans.status,
                     "confidence": ans.confidence,
                     "facts": [f.statement_id for f in ans.facts],
                     "evidence": [e.evidence_id for e in ans.evidence],
                     "explanation": ans.explanation,
                     "missing": ans.missing,
                     "clarification": ans.clarification})
    else:
        print(ans.render(dm.default_mode))
    if dm.kg.all_statements() and a.save:
        store.save(dm)
    return 0


def cmd_say(a) -> int:
    return cmd_ask(a)


def cmd_teach(a) -> int:
    dm, store = _build(a.home, a.answer)
    respond, _ = _responder(dm, not getattr(a, "core", False))
    ans = respond(a.text)
    print(ans.render(dm.default_mode))
    if a.save:
        store.save(dm)
        print(f"[сохранено] {store.kb_path}")
    return 0


def cmd_stats(a) -> int:
    dm, store = _build(a.home)
    stats = dm.kg.stats()
    stats["evidence"] = len(dm.evidence)
    stats["conflicts"] = len(dm.detector.conflicts()) if hasattr(
        dm.detector, "conflicts") else 0
    stats["home"] = store.home
    stats["kb_bytes"] = store.sizes()
    stats["version"] = __version__
    _print_json(stats)
    return 0


def cmd_import(a) -> int:
    plan = build_plan(a.source, a.kind)
    if a.dry_run:
        if a.json:
            _print_json(plan.summary())
        else:
            print(plan.render())
        if a.verbose and not a.json:
            for s in plan.statements:
                state = "OK  " if s.will_import else "SKIP"
                extra = s.skipped_reason or s.note
                print(f"  [{state}] {s.legacy_id}: {s.subject_name} "
                      f"—{s.predicate}-> {s.object_name}"
                      + (f"   # {extra}" if extra else ""))
        return 0
    if not a.json:
        print(plan.render())
    dm, store = _build(a.home)
    result = apply_plan(plan, dm, store=store, save=not a.no_save)
    if a.json:
        _print_json({"plan": plan.summary(), "result": result.summary()})
    else:
        print(result.render())
    return 0


def cmd_imports(a) -> int:
    ledger = ImportLedger(os.path.expanduser(a.home or default_home()))
    entries = ledger.entries()
    if a.json:
        _print_json({"path": ledger.path, "entries": entries})
        return 0
    if not entries:
        print(f"импортов нет ({ledger.path})")
        return 0
    print(f"журнал импортов: {ledger.path}")
    for e in entries:
        print(f"  {e['id']}  {e.get('status','?'):8s} {e.get('applied_at','')}  "
              f"{e.get('source_schema')}  {e.get('source_path')}")
        r = e.get("result") or {}
        print(f"       +{r.get('statements_added', 0)} утверждений, "
              f"{r.get('statements_skipped', 0)} пропущено, "
              f"{r.get('aliases_added', 0)} алиасов")
    return 0


def cmd_restore(a) -> int:
    store = Store(home=a.home, clock=_clock)
    restored = restore(a.ledger_id, store)
    if not restored:
        print("нечего восстанавливать: резервные копии не найдены", file=sys.stderr)
        return 2
    for kind, path in restored.items():
        print(f"восстановлено {kind}: {path}")
    return 0


def cmd_verify(a) -> int:
    rep = verify_home(a.home)
    if a.json:
        _print_json(rep.as_dict())
    else:
        print(rep.render())
    return 0 if rep.ok else 1


def cmd_metrics(a) -> int:
    dm = build_demo(seed_path=a.seed,
                    v11_path=None if a.no_v11 else DEFAULT_V11)
    corpus = load_cases(a.corpus)
    pipe = SemanticPipeline(dm)
    report = run_all(corpus, dm, name=a.corpus, respond=pipe.respond)
    report["seed"] = seed_report(dm)
    report["pipeline"] = {"semantic": True, "version": __version__}
    if a.json:
        _print_json(report)
    else:
        print(render_report(report))
        sd = report["seed"]
        print(f"  база: {sd['statements']} утверждений, {sd['entities']} "
              f"сущностей, {sd['evidence']} записей об источниках")
    return 0


def cmd_coref(a) -> int:
    from rasm_cim_v17.discourse import evaluate_coreference, load_dialogues
    from rasm_cim_v17.discourse.dialogue_corpus import DIALOGUE_SCHEMA
    from rasm_cim_v17.discourse.dialogue_metrics import render_coreference
    from rasm_cim_v17.semantics.harness import DEFAULT_DIALOGUES
    path = a.corpus or DEFAULT_DIALOGUES
    corpus = load_dialogues(path)
    report = evaluate_coreference(corpus, build_demo,
                                  lambda dm: SemanticPipeline(dm).respond)
    report["schema"] = DIALOGUE_SCHEMA
    report["corpus"] = corpus.summary()
    if a.json:
        _print_json(report)
    else:
        print(render_coreference(report))
    return 0


def cmd_meaning(a) -> int:
    dm, _ = _build(a.home)
    from rasm_cim_v17.semantics import meanings_for
    meanings = meanings_for(a.text, resolver=dm.kg.resolver,
                            schema=dm.kg.schema, parser=dm.parser)
    if a.json:
        _print_json([m.as_dict() for m in meanings])
    else:
        for m in meanings:
            print(m.describe())
            for p in m.presuppositions:
                print(f"  пресуппозиция {p.kind}: {p.text} -> {p.holds}")
            for note in m.notes:
                print(f"  · {note}")
    return 0


def cmd_learned(a) -> int:
    snap = LearnedSnapshot.load(a.path)
    _print_json(snap.summary())
    return 0 if snap.verdict in ("VERIFIED", "NO_POINTER") else 2


def cmd_migrate(a) -> int:
    store = Store(home=a.home, clock=_clock)
    if not store.exists():
        print("мигрировать нечего: базы нет", file=sys.stderr)
        return 1
    if not store.needs_migration():
        print("схема уже актуальна")
        return 0
    if a.dry_run:
        print(f"требуется миграция: {store.kb_path}")
        return 0
    dm = DialogueManager(clock=_clock)
    store.attach(dm)
    store.load(dm)
    saved = store.save(dm)
    print(f"миграция выполнена: {saved}")
    return 0


def cmd_entities(a) -> int:
    dm, _ = _build(a.home)
    for e in dm.kg.resolver.all_entities():
        aliases = f"  (алиасы: {', '.join(e.aliases)})" if e.aliases else ""
        print(f"#{e.entity_id} [{e.kind or '?'}] {e.canonical}{aliases}")
    return 0


def cmd_graph(a) -> int:
    from rasm_cim_v16.knowledge_graph import render_triple
    dm, _ = _build(a.home)
    for st in dm.kg.all_statements():
        print(f"{st.statement_id}  {render_triple(dm.kg, st)}  "
              f"[{st.status}, conf={st.confidence:.2f}]")
    return 0


def cmd_evidence(a) -> int:
    dm, _ = _build(a.home)
    for e in dm.evidence.all():
        print(f"{e.evidence_id}  {e.kind:16s} {e.statement_id}  "
              f"{e.utterance or e.note}".rstrip())
    return 0


def cmd_conflicts(a) -> int:
    dm, _ = _build(a.home)
    conflicts = dm.detector.conflicts() if hasattr(dm.detector, "conflicts") else []
    if not conflicts:
        print("конфликтов нет")
        return 0
    for c in conflicts:
        print(c if isinstance(c, str) else json.dumps(c, ensure_ascii=False,
                                                      default=str))
    return 0


# --------------------------------------------------------------------------
# parser
# --------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="rasm-cim",
        description="RASM-CIM v0.17 — offline Russian-language cognitive system")
    p.add_argument("--home", default=None,
                   help="каталог базы (по умолчанию $RASM_CIM_HOME или ~/.rasm-cim)")
    sub = p.add_subparsers(dest="cmd")

    def add(name, func, help_):
        sp = sub.add_parser(name, help=help_)
        sp.add_argument("--home", default=argparse.SUPPRESS)
        sp.set_defaults(func=func)
        return sp

    add("version", cmd_version, "версии и окружение")

    sp = add("chat", cmd_chat, "интерактивный диалог")
    sp.add_argument("--answer", default="normal")
    sp.add_argument("--core", action="store_true")

    sp = add("ask", cmd_ask, "задать вопрос")
    sp.add_argument("text")
    sp.add_argument("--answer", default="normal")
    sp.add_argument("--json", action="store_true")
    sp.add_argument("--no-save", dest="save", action="store_false")
    sp.add_argument("--core", action="store_true",
                    help="выполнить без семантического слоя (чистый v0.16)")
    sp.add_argument("--explain-meaning", action="store_true",
                    help="показать MeaningIR и факт делегирования ядру")

    sp = add("say", cmd_say, "любая реплика (вопрос или обучение)")
    sp.add_argument("text")
    sp.add_argument("--answer", default="normal")
    sp.add_argument("--json", action="store_true")
    sp.add_argument("--no-save", dest="save", action="store_false")
    sp.add_argument("--core", action="store_true")
    sp.add_argument("--explain-meaning", action="store_true")

    sp = add("teach", cmd_teach, "научить факту")
    sp.add_argument("text")
    sp.add_argument("--answer", default="normal")
    sp.add_argument("--no-save", dest="save", action="store_false")
    sp.add_argument("--core", action="store_true")

    sp = add("import", cmd_import, "импорт данных v0.11–v0.15")
    sp.add_argument("source")
    sp.add_argument("--kind", default="auto",
                    choices=("auto", "kb", "facts", "learned"))
    sp.add_argument("--dry-run", action="store_true")
    sp.add_argument("--json", action="store_true")
    sp.add_argument("--verbose", "-v", action="store_true")
    sp.add_argument("--no-save", action="store_true")

    sp = add("imports", cmd_imports, "журнал импортов")
    sp.add_argument("--json", action="store_true")
    sp = add("restore", cmd_restore, "откатить импорт по ledger id")
    sp.add_argument("ledger_id")

    sp = add("verify", cmd_verify, "проверка целостности базы")
    sp.add_argument("--json", action="store_true")

    sp = add("learned", cmd_learned, "сводка обучающего артефакта v0.15")
    sp.add_argument("path")

    sp = add("metrics", cmd_metrics, "метрики семантического слоя (План B)")
    sp.add_argument("--corpus", default=DEFAULT_CORPUS)
    sp.add_argument("--seed", default=DEFAULT_SEED)
    sp.add_argument("--no-v11", action="store_true",
                    help="не подмешивать корпус v0.11")
    sp.add_argument("--json", action="store_true")

    sp = add("coref", cmd_coref, "кореференция: метрика на корпусе диалогов")
    sp.add_argument("--corpus", default=None)
    sp.add_argument("--json", action="store_true")

    sp = add("meaning", cmd_meaning, "показать MeaningIR разбора")
    sp.add_argument("text")
    sp.add_argument("--json", action="store_true")

    sp = add("migrate", cmd_migrate, "миграция схемы базы")
    sp.add_argument("--dry-run", action="store_true")

    add("stats", cmd_stats, "сводная статистика")
    add("entities", cmd_entities, "список сущностей")
    add("graph", cmd_graph, "все утверждения")
    add("evidence", cmd_evidence, "журнал источников")
    add("conflicts", cmd_conflicts, "противоречия")
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    a = parser.parse_args(argv)
    if not getattr(a, "cmd", None):
        parser.print_help()
        return 0
    if not hasattr(a, "home"):
        a.home = None
    try:
        return int(a.func(a) or 0)
    except FileNotFoundError as exc:
        print(f"файл не найден: {exc}", file=sys.stderr)
        return 2
    except (ValueError, KeyError) as exc:
        print(f"ошибка: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":       # pragma: no cover
    raise SystemExit(main())
