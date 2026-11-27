"""Прогон аудита: система и базовые стратегии на наборах A–C и E.

Hidden (D) этим скриптом не читается: для него есть отдельный флаг `--hidden`,
который вызывается один раз, после чего результат фиксируется в отчёте.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .baselines import BASELINES
from .harness import (Case, by_category, fmt, load_cases, run_system, score,
                      verdict)

ROOT = Path(__file__).resolve().parent
SETS = ("bind", "refuse", "mixed", "paraphrase")


def evaluate(cases: list[Case], strategy: str) -> list[tuple[Case, str]]:
    results = []
    for case in cases:
        if strategy == "system":
            observed = run_system(case)
        else:
            observed = BASELINES[strategy](case)
        results.append((case, verdict(case, observed)))
    return results


def table_row(name: str, stats: dict) -> str:
    return ("| {name} | {na} | {ba} | {wr} | {lo} | {ra} | {un} |".format(
        name=name,
        na=stats["n_bind"],
        ba=fmt(stats["binding_accuracy"]),
        wr=fmt(stats["wrong_binding"]),
        lo=fmt(stats["lost_referent"]),
        ra=fmt(stats["refusal_accuracy"]) + f" (n={stats['n_refuse']})",
        un=fmt(stats["ungrounded"]),
    ))


HEADER = ("| стратегия | n(BIND) | точность связки | ошибок | потеряно из-за отказа "
          "| точность отказа | необоснованных |\n"
          "|---|---|---|---|---|---|---|")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sets", nargs="*", default=list(SETS))
    parser.add_argument("--strategies", nargs="*", default=["system", *BASELINES])
    parser.add_argument("--hidden", action="store_true")
    parser.add_argument("--out", default="audit_v1/results/audit.json")
    args = parser.parse_args()

    sets = list(args.sets)
    if args.hidden:
        sets = ["hidden"]

    report: dict = {}
    lines: list[str] = []
    for name in sets:
        cases = load_cases(ROOT / "data" / f"{name}.jsonl")
        report[name] = {}
        lines.append(f"\n### набор `{name}` (n={len(cases)})\n")
        lines.append(HEADER)
        for strategy in args.strategies:
            verdicts = evaluate(cases, strategy)
            stats = score(verdicts)
            report[name][strategy] = {"summary": stats,
                                      "by_category": by_category(verdicts),
                                      "verdicts": [
                                          {"id": c.id, "category": c.category,
                                           "expected": c.expected,
                                           "entity": c.entity, "verdict": v}
                                          for c, v in verdicts]}
            lines.append(table_row(strategy, stats))
        # по категориям — только система
        lines.append("\nПо категориям (система):\n")
        lines.append("| категория | n(BIND) | связка | ошибок | потеряно | отказ | необосн. |")
        lines.append("|---|---|---|---|---|---|---|")
        for category, stats in report[name]["system"]["by_category"].items():
            lines.append("| {c} | {n} | {b} | {w} | {l} | {r} | {u} |".format(
                c=category.split("/", 1)[1], n=stats["n_bind"],
                b=fmt(stats["binding_accuracy"]), w=fmt(stats["wrong_binding"]),
                l=fmt(stats["lost_referent"]),
                r=fmt(stats["refusal_accuracy"]) + f" (n={stats['n_refuse']})",
                u=fmt(stats["ungrounded"])))

    text = "\n".join(lines)
    print(text)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    Path(str(out).replace(".json", ".md")).write_text(text, encoding="utf-8")
    print(f"\nсохранено: {out}")


if __name__ == "__main__":
    main()
