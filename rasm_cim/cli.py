"""`python -m rasm_cim` — единый offline entry point RASM-CIM v0.15.

Режимы (реально реализованные поверх слоёв v0.11…v0.15):
  chat        — интерактивный диалог: ввод → v0.13 DialogueManager → ответ;
                команды: help, exit, reset, teach, inspect/evidence
  ask         — одиночный вопрос; печатает JSON или компактный текст; v0.14+v0.15
  teach       — «запомни …» как user_teach с полным provenance
  memory      — KB: stats | list | query | export | import | retract | conflicts | aliases
  evaluate    — offline eval (intent/verdict/unsupported/latency/RSS/regression probes)
  system-info — версия, платформа, KB path/stats, freeze-манифесты, artifact version

Streams: результаты → stdout, диагностика → stderr.
Exit codes: 0 ok · 2 usage · 3 KB/data problem · 4 conflict/CLARIFY · 5 evaluate DEGRADED.
Demo corpus — НИКОГДА не загружается молча: только `--demo` при создании KB.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import sys
from datetime import datetime, timezone

# repo-layout import shim (uninstalled runs)
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

__version__ = "0.15.0"
DEFAULT_KB = os.path.expanduser(os.environ.get("RASM_CIM_KB",
                                               "~/.rasm-cim/kb.json"))
DEFAULT_CTX = os.path.expanduser(os.environ.get("RASM_CIM_CTX",
                                                "~/.rasm-cim/context.json"))


def _log(*args):
    print(*args, file=sys.stderr)


def _open_kb(path: str, *, allow_create: bool = True):
    from rasm_cim_v12.knowledge import KnowledgeStore
    kb = KnowledgeStore.init(path)
    _log(f"[kb] {path} (entities={kb.stats()['entities']}, "
         f"statements={kb.stats()['statements_active']})")
    return kb


def _seed_demo(kb):
    import subprocess
    env = dict(os.environ)
    env["PYTHONPATH"] = _ROOT + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run([sys.executable, "-m", "rasm_cim_v12.cli", "seed-demo", kb.path],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                   env=env)


def _open_ctx(path: str):
    from rasm_cim_v13.dialogue import ContextStore
    return ContextStore(path)


# --------------------------------------------------------------- modes
def cmd_ask(a) -> int:
    from rasm_cim_v15.pipeline15 import ask_v15
    kb = _open_kb(a.kb)
    try:
        from rasm_cim_v15.artifacts import LearnedArtifacts
        arts = LearnedArtifacts.current()
    except Exception:
        arts = None
    r = ask_v15(kb, a.question, artifacts=arts, max_depth=a.max_depth)
    if a.json:
        print(json.dumps(r, ensure_ascii=False, indent=2))
    else:
        line = r["answer"]
        ids = ", ".join(r["evidence_ids"])
        suffix = f" [{r['verdict']}]" + (f" [{ids}]" if ids else "")
        print(line + suffix)
    return 4 if r["verdict"] == "CONFLICT" else 0


def cmd_teach(a) -> int:
    from rasm_cim_v13.dialogue import DialogueManager, ContextStore
    kb = _open_kb(a.kb)
    ctx = ContextStore(a.ctx)
    mgr = DialogueManager(kb, ctx)
    msg = a.fact if a.fact.lower().startswith(("запомни", "запиши")) else f"запомни: {a.fact}"
    out = mgr.process(msg)
    r = out["results"][0]
    print(json.dumps({"status": r.get("status"), "answer": r.get("answer"),
                      "evidence": r.get("evidence")}, ensure_ascii=False, indent=2))
    return 0 if r.get("status") in ("TAUGHT", "TEACH_HELP") else 2


_CHAT_HELP = """Команды:
  help, ?            — эта справка
  exit, quit, :q     — выйти (контекст сохраняется)
  reset              — очистить контекст диалога (и topic/open questions)
  teach <факт>       — «запомни: факт» с provenance user_teach
  inspect / evidence — показать evidence последнего ответа (statement_id → provenance)
  entities           — последние сущности в контексте
  mode ask|chat      — подсказка текущего режима
Любой другой текст — вопрос/реплика: v0.13 dialogue + v0.14 reasoning.
Пример: «Какая столица у Франции?», «А её население?», «1) … 2) …»"""


def cmd_chat(a) -> int:
    from rasm_cim_v13.dialogue import DialogueManager, ContextStore
    kb = _open_kb(a.kb)
    ctx = _open_ctx(a.ctx)
    mgr = DialogueManager(kb, ctx)
    if kb.stats()["entities"] == 0 and not a.quiet:
        _log("KB пуст. Ответы будут «нет данных». Демо-корпус: --demo (opt-in).")
    print(f"RASM-CIM v{__version__} — чат (offline). :help для команд, :exit для выхода.")
    last_evidence = []
    last_answer = {}
    pending_help = None
    while True:
        try:
            line = input("Вы> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            _log("bye")
            return 0
        if not line:
            continue
        low = line.casefold()
        if low in ("exit", "quit", ":q", ":exit", ":quit"):
            _log("контекст сохранён:", a.ctx)
            return 0
        if low in ("help", "?", ":help", ":?"):
            print(_CHAT_HELP)
            continue
        if low in ("reset", ":reset"):
            ctx.reset()
            print("Контекст очищен.")
            continue
        if low in ("entities", ":entities"):
            recent = ctx.state.get("recent_entities", [])
            print(json.dumps([e["name"] for e in recent], ensure_ascii=False))
            continue
        if low.startswith(("teach ", ":teach ")):
            fact = line.split(" ", 1)[1].strip()
            msg = fact if fact.lower().startswith(("запомни", "запиши")) else f"запомни: {fact}"
            out = mgr.process(msg)
            r = out["results"][0]
            print(f"Бот> {r.get('answer')} [{r.get('status')}]")
            continue
        if low in ("inspect", "evidence", ":inspect", ":evidence"):
            if not last_evidence:
                print("Evidence последнего ответа пуст.")
                continue
            print("Evidence последнего ответа:")
            for ev in last_evidence:
                print(json.dumps(ev, ensure_ascii=False))
            continue
        if low == "mode ask|chat" or low == "mode":
            print("chat")
            continue
        out = mgr.process(line)
        for r in out["results"]:
            ev = r.get("evidence", [])
            ids = ", ".join(e.get("statement_id") for e in ev if e.get("statement_id"))
            print(f"Бот> {r.get('answer')} [{r.get('status')}]" +
                  (f" [{ids}]" if ids else ""))
            last_evidence = ev
            last_answer = r
        if isinstance(out.get("mode"), str):
            last_answer["mode"] = out["mode"]


def cmd_memory(a) -> int:
    from rasm_cim_v12.knowledge import export_jsonl, import_jsonl
    from rasm_cim_v12.knowledge.store import KnowledgeStore
    kb = _open_kb(a.kb)
    cmd = a.sub
    if cmd == "stats":
        print(json.dumps(kb.stats(), ensure_ascii=False, indent=2))
        return 0
    if cmd == "list":
        limit = a.limit
        rows = []
        for s in sorted(kb.statements.values(), key=lambda s: (s.created_at, s.id)):
            if s.status != "active":
                continue
            if len(rows) >= limit:
                break
            subj = kb.get_entity(s.subject)
            obj_e = kb.get_entity(s.object_entity) if s.object_entity else None
            rows.append({"id": s.id, "subject": subj.name if subj else s.subject,
                         "predicate": s.predicate,
                         "object": obj_e.name if obj_e else s.object_literal,
                         "polarity": s.polarity, "kind": s.kind,
                         "source": (s.provenance or {}).get("source")})
        print(json.dumps(rows, ensure_ascii=False, indent=2))
        return 0
    if cmd == "query":
        print(json.dumps([s.to_dict() for s in kb.query_statements(
            subject=None, predicate=a.predicate)[:a.limit]],
            ensure_ascii=False, indent=2))
        return 0
    if cmd == "export":
        payload = export_jsonl(kb)
        with open(a.out, "w", encoding="utf-8") as fh:
            fh.write(payload)
        _log(f"exported → {a.out} ({len(payload)} bytes)")
        return 0
    if cmd == "import":
        res = import_jsonl(kb, a.file, strict=a.strict)
        # ImportReport IS a dict subclass — serialize the mapping itself,
        # not res.__dict__ (always empty for dict subclasses).
        print(json.dumps(dict(res), ensure_ascii=False, indent=2, default=str))
        return 0
    if cmd == "retract":
        from rasm_cim_v12.knowledge import retract
        print(json.dumps(retract(kb, a.statement_id, reason=a.reason),
                         ensure_ascii=False, indent=2))
        return 0
    if cmd == "conflicts":
        print(json.dumps(kb.conflicts(), ensure_ascii=False, indent=2))
        return 0
    if cmd == "aliases":
        out = [{"id": e.id, "name": e.name, "aliases": e.aliases}
               for e in kb.entities.values() if e.aliases]
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return 0
    _log(f"unknown memory sub-command: {cmd}")
    return 2


def cmd_evaluate(a) -> int:
    from rasm_cim_v12.knowledge import KnowledgeStore
    from rasm_cim_v15.evaluate import (load_split, intent_acc, verdict_acc,
                                         unsupported_rate, latency_p50_ms,
                                         regression_pass, peak_rss_mb)
    from rasm_cim_v15.artifacts import LearnedArtifacts
    kb = _open_kb(a.kb)
    split = a.split
    examples = load_split(split)
    arts = LearnedArtifacts.current() if a.learned else None
    report = {
        "split": split, "artifact": arts.s["version"] if arts else None,
        "intent_acc": intent_acc(examples, arts),
        "intent_acc_baseline": intent_acc(examples, None),
        "verdict_acc": verdict_acc(kb, examples),
        "unsupported": unsupported_rate(kb, examples),
        "latency_p50_ms": latency_p50_ms(kb, examples),
        "regression": regression_pass(kb),
        "peak_rss_mb": peak_rss_mb(),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    degraded = (not report["regression"]["ok"]
                or (report["verdict_acc"]["acc"] or 0) < 0.5)
    if degraded:
        _log("evaluate: DEGRADED (см. regression/verdict_acc)")
        return 5
    return 0


def cmd_system_info(a) -> int:
    from rasm_cim_v12.knowledge import KnowledgeStore
    import hashlib
    kb_line = {}
    try:
        kb = KnowledgeStore.init(a.kb)
        kb_line = {"path": a.kb, **kb.stats()}
    except Exception as exc:
        kb_line = {"path": a.kb, "error": str(exc)}
    try:
        from rasm_cim_v15.artifacts import LearnedArtifacts
        arts = LearnedArtifacts.current()
        learned = arts.s["version"] if arts else None
    except Exception:
        learned = None
    freeze_ok = None
    base = os.path.join(_ROOT, "results_v11", "freeze_manifest_baseline_v0.10.md5")
    if os.path.exists(base):
        import subprocess
        out = subprocess.run(["md5sum", "-c", base], capture_output=True,
                             text=True, cwd=_ROOT)
        freeze_ok = ("OK" in out.stdout and out.stdout.count("OK") ==
                     sum(1 for l in open(base) if l.strip()))
    info = {
        "package": "rasm-cim", "version": __version__,
        "python": sys.version.split()[0], "platform": platform.platform(),
        "machine": platform.machine(),
        "kb": kb_line,
        "learned_artifact": learned,
        "freeze_baseline_v0.10": "OK" if freeze_ok else ("UNKNOWN" if freeze_ok is None else "MISMATCH"),
        "frozen_layers": ["v0.1-v0.8 (sources, tests, results)", "rust v0.7"],
        "time_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    print(json.dumps(info, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="rasm_cim",
        description="RASM-CIM v0.15 — русский offline CIM-комплекс "
                    "(диалог/KB/reasoning/коррекции). Demo-корпус: opt-in.")
    ap.add_argument("--version", action="version", version=f"rasm-cim {__version__}")
    ap.add_argument("--kb", default=DEFAULT_KB, help="путь к KB (default ~/.rasm-cim/kb.json)")
    ap.add_argument("--ctx", default=DEFAULT_CTX, help="путь к контексту диалога")
    ap.add_argument("--demo", action="store_true",
                    help="при создании KB наполнить демо-фактами (demo corpus)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("chat", help="интерактивный диалог с контекстом")
    p.add_argument("--quiet", action="store_true")
    p.set_defaults(f=cmd_chat)

    p = sub.add_parser("ask", help="одиночный вопрос")
    p.add_argument("question")
    p.add_argument("--json", action="store_true", help="полный JSON сevidence")
    p.add_argument("--max-depth", type=int, default=2)
    p.set_defaults(f=cmd_ask)

    p = sub.add_parser("teach", help="«запомни» факт с provenance")
    p.add_argument("fact")
    p.set_defaults(f=cmd_teach)

    p = sub.add_parser("memory", help="работа с KB")
    mem = p.add_subparsers(dest="sub", required=True)
    mem.add_parser("stats").set_defaults(sub_cmd="stats")
    pl = mem.add_parser("list")
    pl.add_argument("--limit", type=int, default=20)
    pq = mem.add_parser("query")
    pq.add_argument("--predicate", default=None)
    pq.add_argument("--limit", type=int, default=20)
    pe = mem.add_parser("export"); pe.add_argument("out")
    pi = mem.add_parser("import")
    pi.add_argument("file"); pi.add_argument("--strict", action="store_true")
    pr = mem.add_parser("retract"); pr.add_argument("statement_id"); pr.add_argument("--reason", default="user")
    mem.add_parser("conflicts")
    mem.add_parser("aliases")
    p.set_defaults(f=cmd_memory)

    p = sub.add_parser("evaluate", help="offline evaluation")
    p.add_argument("--split", default="dev", choices=["dev", "test", "train"])
    p.add_argument("--learned", action="store_true", help="с learned artifacts")
    p.set_defaults(f=cmd_evaluate)

    p = sub.add_parser("system-info", help="версия/окружение/KB/freeze")
    p.set_defaults(f=cmd_system_info)

    return ap


def main(argv=None) -> int:
    ap = build_parser()
    a = ap.parse_args(argv)
    if not os.path.exists(a.kb):
        os.makedirs(os.path.dirname(a.kb) or ".", exist_ok=True)
        if a.demo:
            from rasm_cim_v12.knowledge import KnowledgeStore
            kb = KnowledgeStore.init(a.kb)
            _seed_demo(kb)
            _log("demo corpus загружен (opt-in)")
        else:
            from rasm_cim_v12.knowledge import KnowledgeStore
            KnowledgeStore.init(a.kb)
            _log(f"создан пустой KB: {a.kb} (hint: --demo загрузит демо-факты)")
    return a.f(a)


if __name__ == "__main__":
    sys.exit(main())
