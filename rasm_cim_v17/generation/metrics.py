"""Generation metrics — the numbers the roadmap asks to be measured.

    sourced_claim_rate    доля утверждений с id факта/утверждения
    avg_sentences         средняя длина ответа в предложениях
    evidence_share        доля планов, где доказательство включено
    repeat_rate           доля подряд идущих реплик с тем же утверждением
    determinism           одинаковый план -> одинаковый текст (проверяется
                          двумя рендерами, а не обещанием)

Small numbers, but they answer "стало ли связнее" without asking anyone to
believe a subjective judgement.
"""
from __future__ import annotations

import re

from .nlg import render

SENTENCE_RE = re.compile(r"[.!?]+(?:\s|$)")


def _sentences(text: str) -> int:
    parts = [p for p in SENTENCE_RE.split(text or "") if p.strip()]
    return len(parts)


def evaluate_plans(plans: list, mode: str = "normal") -> dict:
    if not plans:
        return {"n": 0, "note": "нет планов"}
    claims = [s for ir in plans for s in ir.segments if s.kind == "claim"]
    sourced = [s for s in claims if s.sourced]
    with_evidence = [ir for ir in plans if ir.of_kind("evidence")]
    texts = [render(ir, mode) for ir in plans]
    again = [render(ir, mode) for ir in plans]
    repeats = 0
    for prev, cur in zip(plans, plans[1:]):
        if prev.claims and cur.claims and prev.claims[0].text == cur.claims[0].text:
            repeats += 1
    return {
        "n": len(plans),
        "mode": mode,
        "claims": len(claims),
        "sourced_claims": len(sourced),
        "sourced_claim_rate": round(len(sourced) / len(claims), 4) if claims else None,
        "evidence_share": round(len(with_evidence) / len(plans), 4),
        "avg_sentences": round(sum(_sentences(t) for t in texts) / len(texts), 2),
        "avg_chars": round(sum(len(t) for t in texts) / len(texts), 1),
        "repeat_rate": round(repeats / max(1, len(plans) - 1), 4),
        "deterministic": texts == again,
        "empty": sum(1 for t in texts if not t.strip()),
    }


def render_generation(report: dict) -> str:
    lines = [f"Связность (регистр {report['mode']}, планов {report['n']}):"]
    lines.append(f"  утверждения с источником: {report['sourced_claims']}/"
                 f"{report['claims']} = {report['sourced_claim_rate']:.1%}"
                 if report["claims"] else "  утверждения: нет")
    lines.append(f"  с доказательством: {report['evidence_share']:.1%} планов")
    lines.append(f"  длина: {report['avg_sentences']} предложений, "
                 f"{report['avg_chars']} символов в среднем")
    lines.append(f"  повторы подряд: {report['repeat_rate']:.1%}")
    lines.append(f"  детерминизм: {'да' if report['deterministic'] else 'НЕТ'}")
    return "\n".join(lines)
