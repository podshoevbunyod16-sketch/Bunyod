"""v0.15 CLI — corpus/training/correction/rollback/report (offline)."""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rasm_cim_v12.knowledge import KnowledgeStore
from rasm_cim_v15 import corpus_build
from rasm_cim_v15.artifacts import LearnedArtifacts, ARTIFACTS_DIR
from rasm_cim_v15 import corrections as C
from rasm_cim_v15.evaluate import (load_split, intent_acc, verdict_acc,
                                     unsupported_rate, latency_p50_ms,
                                     regression_pass, peak_rss_mb)


def _kb(path: str):
    return KnowledgeStore.init(path)


def _cmd_build_corpus(a):
    print(json.dumps(corpus_build.write_corpus(), ensure_ascii=False, indent=2))
    return 0


def _cmd_corpus_info(a):
    manifest = json.load(open(os.path.join(corpus_build.CORPUS_DIR, "manifest.json"),
                              encoding="utf-8"))
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


def _cmd_train(a):
    import hashlib
    train = load_split("train")
    corpus_sha = hashlib.sha256(b"".join(
        open(os.path.join(corpus_build.CORPUS_DIR, f"{s}.jsonl"), "rb").read()
        for s in ("train", "dev", "test"))).hexdigest()
    parent = LearnedArtifacts.current()
    arts = LearnedArtifacts.train(train, corpus_sha, parent=parent,
                                  correction_events=None)
    dev = load_split("dev")
    arts.s["metrics"] = {"dev_intent_acc": intent_acc(dev, arts),
                         "dev_intent_acc_baseline": intent_acc(dev, None)}
    path = arts.save()
    print(json.dumps({"artifact": arts.s["version"], "saved_to": path,
                      "dev_intent_acc": arts.s["metrics"]["dev_intent_acc"],
                      "dev_intent_acc_baseline": arts.s["metrics"]["dev_intent_acc_baseline"]},
                     ensure_ascii=False, indent=2))
    return 0


def _cmd_eval(a):
    kb = _kb(a.kb)
    split = a.split
    examples = load_split(split)
    arts = LearnedArtifacts.current() if a.learned else None
    report = {
        "split": split,
        "artifact": arts.s["version"] if arts else None,
        "intent_acc": intent_acc(examples, arts),
        "intent_acc_baseline": intent_acc(examples, None),
        "verdict_acc": verdict_acc(kb, examples),
        "unsupported": unsupported_rate(kb, examples),
        "latency_p50_ms": latency_p50_ms(kb, examples),
        "regression": regression_pass(kb),
        "peak_rss_mb": peak_rss_mb(),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def _cmd_correct(a):
    kinds_ok = {"intent", "alias", "entity", "answer", "relation"}
    if a.kind not in kinds_ok:
        print(f"error: kind must be one of {sorted(kinds_ok)}", file=sys.stderr)
        return 2
    kb = _kb(a.kb)
    ev = C.record_correction(a.kind, a.wrong, a.correct, a.target_text,
                             session_id=a.session)
    errs = C.validate_event(kb, ev)
    for e in errs:
        print(f"warning: {e}", file=sys.stderr)
    if a.apply:
        from rasm_cim_v15 import corrections as CC
        evs = [e for e in CC.load_corrections()
               if e["correction_id"] == ev["correction_id"]]
        report = CC.apply_corrections(kb, evs)
        print(json.dumps({"recorded": ev["correction_id"], "apply": report},
                         ensure_ascii=False, indent=2))
    else:
        print(json.dumps({"recorded": ev["correction_id"],
                          "hint": "run: python -m rasm_cim_v15.cli apply-corrections --kb KB"},
                         ensure_ascii=False))
    return 0


def _cmd_apply(a):
    kb = _kb(a.kb)
    pending = C.load_corrections()
    if a.ids:
        keep = set(a.ids.split(","))
        pending = [e for e in pending if e["correction_id"] in keep]
    print(json.dumps(C.apply_corrections(kb, pending), ensure_ascii=False, indent=2))
    return 0


def _cmd_corrections(a):
    events = C.load_corrections(a.path)
    print(json.dumps({"events": len(events), "chain": C.verify_chain(a.path)["chain_ok"],
                      "conflicts": C.find_conflicts(events),
                      "list": events}, ensure_ascii=False, indent=2))
    return 0


def _cmd_rollback(a):
    arts = LearnedArtifacts.rollback(a.version)
    print(json.dumps({"rolled_back_to": arts.s["version"],
                      "created_at": arts.s["created_at"]}, ensure_ascii=False))
    return 0


def _cmd_ask(a):
    kb = _kb(a.kb)
    from rasm_cim_v15.pipeline15 import ask_v15
    arts = LearnedArtifacts.current()
    r = ask_v15(kb, a.question, artifacts=arts, max_depth=a.max_depth)
    print(json.dumps(r, ensure_ascii=False, indent=2))
    return 0 if r["verdict"] != "CONFLICT" else 4


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="rasm_cim_v15.cli",
                                 description="RASM-CIM v0.15 (offline)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("build-corpus"); p.set_defaults(f=_cmd_build_corpus)
    p = sub.add_parser("corpus-info"); p.set_defaults(f=_cmd_corpus_info)
    p = sub.add_parser("train"); p.set_defaults(f=_cmd_train)
    p = sub.add_parser("eval")
    p.add_argument("--kb", required=True); p.add_argument("--split", default="dev",
                                                          choices=["train", "dev", "test"])
    p.add_argument("--learned", action="store_true")
    p.set_defaults(f=_cmd_eval)
    p = sub.add_parser("correct")
    p.add_argument("--kb", required=True)
    p.add_argument("--kind", required=True)
    p.add_argument("--wrong", default=""); p.add_argument("--correct", required=True)
    p.add_argument("--target-text", required=True); p.add_argument("--session", default="cli")
    p.add_argument("--apply", action="store_true")
    p.set_defaults(f=_cmd_correct)
    p = sub.add_parser("apply-corrections")
    p.add_argument("--kb", required=True); p.add_argument("--ids", default="")
    p.set_defaults(f=_cmd_apply)
    p = sub.add_parser("corrections")
    p.add_argument("--path", default=C.CORRECTIONS_PATH)
    p.set_defaults(f=_cmd_corrections)
    p = sub.add_parser("rollback"); p.add_argument("version"); p.set_defaults(f=_cmd_rollback)
    p = sub.add_parser("ask")
    p.add_argument("--kb", required=True); p.add_argument("question")
    p.add_argument("--max-depth", type=int, default=2)
    p.set_defaults(f=_cmd_ask)

    a = ap.parse_args(argv)
    return a.f(a)


if __name__ == "__main__":
    sys.exit(main())
