"""«Второй размечающий»: формальное правило + каппа Коэна с метками автора.

Второго человека у аудита нет. Поэтому второй размечающий — скрипт, который
применяет к тексту диалога формальное правило (см. PROTOCOL.md, раздел 2) и
**не читает** метку автора. Согласие считается как «человек ↔ правило», не
«человек ↔ человек», и в отчёте названо именно так.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from .baselines import _gender, _mentions, _pronoun_gender
from .harness import Case, load_cases

SAMPLE_SEED = 20261129
SAMPLE_SHARE = 0.30


def annotate(case: Case) -> dict:
    """Метка правила: BIND с именем либо REFUSE с причиной."""
    wanted = _pronoun_gender(case)
    if wanted == "p":
        mentioned = _mentions(list(case.turns))
        if len(mentioned) >= 2:
            return {"label": "REFUSE", "reason": "group", "entity": ""}
        if not mentioned:
            return {"label": "REFUSE", "reason": "no_antecedent", "entity": ""}
        return {"label": "BIND", "reason": "", "entity": mentioned[-1]}

    mentioned = _mentions(list(case.turns))
    matching = [m for m in mentioned if _gender(m) == wanted]
    if not matching:
        reason = "no_antecedent" if not mentioned else "gender_mismatch"
        return {"label": "REFUSE", "reason": reason, "entity": ""}
    # «равнозначны» = названы вместе, в одной реплике, и позже выбора не было;
    # иначе действует обычное для русского текста правило «последнее упоминание»
    last_turn = _last_mention_turn(case, matching)
    if len(last_turn) > 1:
        return {"label": "REFUSE", "reason": "ambiguous", "entity": ""}
    return {"label": "BIND", "reason": "", "entity": matching[-1]}


def _last_mention_turn(case, matching: list[str]) -> set[str]:
    wanted = {m.casefold() for m in matching}
    for turn in reversed(list(case.turns)):
        found = {m.casefold() for m in _mentions([turn], skip_last=False)
                 if m.casefold() in wanted}
        if found:
            return found
    return set()


def cohen_kappa(pairs: list[tuple[str, str]]) -> float:
    if not pairs:
        return 0.0
    labels = sorted({label for pair in pairs for label in pair})
    total = len(pairs)
    observed = sum(1 for a, b in pairs if a == b) / total
    expected = 0.0
    for label in labels:
        pa = sum(1 for a, _ in pairs if a == label) / total
        pb = sum(1 for _, b in pairs if b == label) / total
        expected += pa * pb
    if expected >= 1.0:
        return 1.0
    return (observed - expected) / (1.0 - expected)


def sample(cases: list[Case], seed: int = SAMPLE_SEED,
           share: float = SAMPLE_SHARE) -> list[Case]:
    rng = random.Random(seed)
    count = max(1, round(len(cases) * share))
    return rng.sample(cases, count)


def agreement(cases: list[Case]) -> dict:
    chosen = sample(cases)
    pairs = []
    detail = []
    for case in chosen:
        second = annotate(case)
        pairs.append((case.expected, second["label"]))
        detail.append({"id": case.id, "author": case.expected,
                       "second": second["label"],
                       "second_entity": second["entity"],
                       "author_entity": case.entity})
    observed = sum(1 for a, b in pairs if a == b) / len(pairs)
    return {"n": len(pairs), "observed": observed,
            "kappa": cohen_kappa(pairs), "detail": detail,
            "labels": {"author": [a for a, _ in pairs],
                       "second": [b for _, b in pairs]}}


def main() -> None:
    root = Path(__file__).resolve().parent
    report = {}
    for name in ("bind", "refuse", "mixed", "paraphrase"):
        cases = load_cases(root / "data" / f"{name}.jsonl")
        result = agreement(cases)
        report[name] = {k: v for k, v in result.items() if k != "detail"}
        print(f"{name:11s} n={result['n']:3d} согласие={result['observed']:.3f} "
              f"каппа={result['kappa']:.3f}")
    pooled_pairs = []
    for name in ("bind", "refuse", "mixed", "paraphrase"):
        cases = load_cases(root / "data" / f"{name}.jsonl")
        pooled_pairs += [(c.expected, annotate(c)["label"]) for c in cases]
    pooled = {"n": len(pooled_pairs),
              "observed": sum(1 for a, b in pooled_pairs if a == b) / len(pooled_pairs),
              "kappa": cohen_kappa(pooled_pairs)}
    report["pooled"] = pooled
    print(f"{'pooled':11s} n={pooled['n']:3d} согласие={pooled['observed']:.3f} "
          f"каппа={pooled['kappa']:.3f}")
    (root / "results").mkdir(exist_ok=True)
    (root / "results" / "second_annotator.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
