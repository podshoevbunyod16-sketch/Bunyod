"""NLG — a pure function from ResponseIR to text, in three registers.

    short     только утверждение (или отказ)
    normal    утверждение + квалификатор
    detailed  утверждение + квалификатор + доказательство

No randomness, no templates that fight the content: segments already carry
their text, the renderer only joins them and decides punctuation.  That is why
``render(ir) == render(ir)`` holds for the same plan and why the tests can
compare two runs instead of asserting "it looks nice".
"""
from __future__ import annotations

from .response_ir import ResponseIR, ResponseIRError

MODES = ("short", "normal", "detailed")

_PREFIX = {"evidence": "Основание", "qualifier": "Оговорка",
           "clarification": "Уточнение", "correction": "Поправка"}

_ORDER = {"claim": 0, "reply": 0, "correction": 1, "qualifier": 2,
          "evidence": 3, "clarification": 4, "refusal": 0}


def _sentence(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    return text if text[-1] in ".!?…" else text + "."


def render(ir: ResponseIR, mode: str | None = None) -> str:
    mode = mode or ir.register
    if mode not in MODES:
        raise ResponseIRError(f"неизвестный регистр: {mode!r}")
    if ir.status in ("UNKNOWN", "AMBIGUOUS"):
        parts = []
        for seg in ir.segments:
            prefix = _PREFIX.get(seg.kind, "")
            body = _sentence(seg.text)
            parts.append(f"{prefix}: {body}" if prefix else body)
        return " ".join(p for p in parts if p)

    claims = [s for s in ir.segments if s.kind in ("claim", "reply")]
    if mode == "short":
        claims = claims[:1]
    kept = claims + [s for s in ir.segments
                     if s.kind == "correction"
                     or (mode in ("normal", "detailed") and s.kind == "qualifier")
                     or (mode == "detailed" and s.kind == "evidence")]
    kept.sort(key=lambda s: _ORDER[s.kind])
    if not kept:
        raise ResponseIRError("после отбора не осталось сегментов")

    out = ""
    for seg in [s for s in kept if s.kind in ("claim", "reply")]:
        out = f"{out} {_sentence(seg.text)}".strip()
    for seg in [s for s in kept if s.kind not in ("claim", "reply")]:
        prefix = _PREFIX.get(seg.kind, "")
        body = _sentence(seg.text)
        piece = f"{prefix}: {body}" if prefix else body
        out = f"{out} {piece}".strip()
    return out


def render_ir_json(ir: ResponseIR) -> str:
    return ir.to_json()
