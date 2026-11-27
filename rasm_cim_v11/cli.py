"""v0.11 CLI — additive to v0.10 (old commands untouched).

python -m rasm_cim_v11.cli language-check "Какая столица у Франции?"
python -m rasm_cim_v11.cli ask-text "Умеет ли пингвин летать?"
"""
from __future__ import annotations

import argparse
import json
import sys

from .language import analyze, INTENTS
from .demo import answer_v11


def cmd_language_check(a):
    res = analyze(a.text)
    out = {
        "original": res.original,
        "normalized": res.normalized,
        "tokens": [{"text": t.text, "norm": t.norm, "start": t.start,
                    "end": t.end, "kind": t.kind} for t in res.tokens],
        "lemmas": list(res.lemmas),
        "intent": res.intent.intent,
        "confidence": res.intent.confidence,
        "matched": list(res.intent.matched),
        "scores": dict(res.intent.scores),
        "has_cyrillic": res.has_cyrillic,
        "version": res.version,
    }
    json.dump(out, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")
    return 0


def cmd_ask_text(a):
    res = answer_v11(a.text)
    out = dict(res)
    out["question"] = a.text
    json.dump(out, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="rasm_cim_v11.cli",
                                 description="RASM-CIM v0.11 language layer (offline)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    lc = sub.add_parser("language-check", help="analyze text: normalize/tokens/lemmas/intent")
    lc.add_argument("text")
    lc.set_defaults(f=cmd_language_check)
    at = sub.add_parser("ask-text", help="ask against bundled demo facts (TruthGraph9)")
    at.add_argument("text")
    at.set_defaults(f=cmd_ask_text)
    a = ap.parse_args(argv)
    return a.f(a)


if __name__ == "__main__":
    sys.exit(main())
