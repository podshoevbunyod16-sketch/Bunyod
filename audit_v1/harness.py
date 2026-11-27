"""Независимый аудит кореференции RASM-CIM: загрузка кейсов и прогон.

Кейсы — диалоги, написанные вручную (см. `authoring/`): факты сообщаются внутри
реплик пользователя, отдельного предобучения нет. Оценивается последняя реплика.

Отказы фиксируются формально: местоимение осталось неразрешённым (AMBIGUOUS)
или система задала уточняющий вопрос. Связка с классом/свойством («он» →
«физик») считается ошибочной связкой, а не отказом.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data"

#: рода местоимений, которые аудит считает местоимениями 3-го лица
PRONOUN_GENDER = {"он": "m", "его": "m", "ему": "m", "им": "m",
                  "она": "f", "её": "f", "ее": "f", "ей": "f",
                  "они": "p", "их": "p"}


@dataclass(frozen=True)
class Case:
    id: str
    set: str
    category: str
    turns: tuple[str, ...]
    expected: str            # BIND | REFUSE
    entity: str = ""
    pronoun: str = ""
    refuse_reason: str = ""
    author: str = ""
    date: str = ""
    method: str = ""

    @property
    def target(self) -> str:
        return self.turns[-1]


def load_cases(path: Path | str) -> list[Case]:
    path = Path(path)
    cases: list[Case] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        raw = json.loads(line)
        cases.append(Case(
            id=raw["id"], set=raw.get("set", path.stem), category=raw["category"],
            turns=tuple(raw["turns"]), expected=raw["expected"],
            entity=raw.get("entity", ""), pronoun=raw.get("pronoun", ""),
            refuse_reason=raw.get("refuse_reason", ""),
            author=raw.get("author", ""), date=raw.get("date", ""),
            method=raw.get("method", ""),
        ))
    return cases


def name_key(name: str) -> str:
    return (name or "").strip().casefold().replace("ё", "е")


# --------------------------------------------------------------------- система
def run_system(case: Case) -> dict:
    """Прогон одного диалога через v0.17 и наблюдение по последней реплике."""
    from rasm_cim_v17.discourse.binding import FOCUS_AMBIGUOUS, FOCUS_RESOLVED  # noqa: F401
    from rasm_cim_v17.pipeline import SemanticPipeline
    from rasm_cim_v17.semantics import build_demo

    dm = build_demo()
    pipe = SemanticPipeline(dm)
    answer = None
    for turn in case.turns:
        answer = pipe.respond(turn)

    meaning = getattr(pipe, "last_meaning", None)
    bound_to, status = "", "no_focus"
    if meaning is not None:
        pronouns = [f for f in meaning.focus
                    if (f.surface or "").strip().casefold() in PRONOUN_GENDER]
        if pronouns:
            focus = pronouns[-1]
            status = focus.status
            if focus.status == "RESOLVED" and focus.entity_id:
                bound_to = dm.kg.name_of(focus.entity_id)
    clarified = bool(getattr(meaning, "clarification", "") or "")
    answered_unknown = getattr(answer, "status", "") == "UNKNOWN"
    refused = status == "AMBIGUOUS" or clarified or (
        status in ("PRONOUN", "no_focus") and answered_unknown)
    if status == "no_focus":
        refused = True
    return {"status": status, "bound_to": bound_to, "refused": refused,
            "clarified": clarified, "answer": getattr(answer, "text", "")[:200]}


def verdict(case: Case, observed: dict) -> str:
    """correct / wrong / refused / ungrounded."""
    bound = observed["bound_to"]
    if case.expected == "BIND":
        if bound and name_key(bound) == name_key(case.entity):
            return "correct"
        if observed["refused"] and not bound:
            return "refused"
        return "wrong"
    if observed["refused"] and not bound:
        return "correct"
    return "ungrounded"


# ------------------------------------------------------------------ статистика
def wilson(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    p = successes / total
    denom = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    spread = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return max(0.0, centre - spread), min(1.0, centre + spread)


def rate(successes: int, total: int) -> dict:
    lo, hi = wilson(successes, total)
    return {"successes": successes, "total": total,
            "rate": (successes / total) if total else 0.0, "ci95": (lo, hi)}


def fmt(stat: dict) -> str:
    return (f"{stat['successes']}/{stat['total']} = {stat['rate'] * 100:.1f}% "
            f"({stat['ci95'][0] * 100:.1f}–{stat['ci95'][1] * 100:.1f})")


def score(verdicts: list[tuple[Case, str]]) -> dict:
    """Метрики раздельно: связки и отказы не смешиваются."""
    bind = [(c, v) for c, v in verdicts if c.expected == "BIND"]
    refuse = [(c, v) for c, v in verdicts if c.expected == "REFUSE"]
    return {
        "binding_accuracy": rate(sum(1 for _, v in bind if v == "correct"), len(bind)),
        "wrong_binding": rate(sum(1 for _, v in bind if v == "wrong"), len(bind)),
        "lost_referent": rate(sum(1 for _, v in bind if v == "refused"), len(bind)),
        "refusal_accuracy": rate(sum(1 for _, v in refuse if v == "correct"), len(refuse)),
        "ungrounded": rate(sum(1 for _, v in refuse if v == "ungrounded"), len(refuse)),
        "n_bind": len(bind), "n_refuse": len(refuse),
    }


def by_category(verdicts: list[tuple[Case, str]]) -> dict[str, dict]:
    groups: dict[str, list[tuple[Case, str]]] = {}
    for case, verdict in verdicts:
        groups.setdefault(f"{case.set}/{case.category}", []).append((case, verdict))
    return {name: score(items) for name, items in sorted(groups.items())}
