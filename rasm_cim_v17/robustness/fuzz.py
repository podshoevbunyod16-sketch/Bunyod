"""Deterministic fuzzing of the input path.

The generator is seeded, so a crash found in CI is reproducible with the seed
and the index printed in the report.  What is exercised is the whole shipped
chain: parse → MeaningIR → answer → plan → render.

Failure means an exception escaped.  A wrong answer is *not* a fuzz failure —
that is what the measured corpora are for.
"""
from __future__ import annotations

import random
import time
import traceback

FUZZ_SEED = 20261127

_ALPHABETS = (
    "абвгдеёжзийклмнопрстуфхцчшщъыьэюя",
    "abcdefghijklmnopqrstuvwxyz",
    "0123456789 !?.,:;-—«»()[]{}<>/\\|@#$%^&*_+=~`'\"",
    "😀🙂🙃🤖🔥💡🕳️",
    "\u0301\u200b\u200e\u202e\ufeff",      # combining, zero-width, bidi, BOM
)


def generate_inputs(n: int = 200, seed: int = FUZZ_SEED) -> list[str]:
    """``n`` reproducible inputs, each one hostile in a different way."""
    rng = random.Random(seed)
    out: list[str] = [
        "", " ", "\n\n\t ", "\x00", "?", "?!?!", "«»", "—",
        "Кто такой ?", "Забудь, что", "Запомни:", "он",
        # по одному гарантированному образцу каждой категории, чтобы отчёт
        # не зависел от того, что выпало генератору
        "Кто такой " + "а" * 2000,
        "😀🙂🙃🤖" + "\u200b" + "Кит" + "\ufeff",
        "Кит — это млекопитающее. " * 3,
        bytes(range(0, 40)).decode("utf-8", errors="replace"),
    ]
    for _ in range(max(0, n - len(out))):
        kind = rng.choice(("short", "words", "long", "bytes", "mixed", "repeat"))
        if kind == "short":
            s = "".join(rng.choice(rng.choice(_ALPHABETS))
                        for _ in range(rng.randint(1, 6)))
        elif kind == "words":
            s = " ".join("".join(rng.choice(_ALPHABETS[0])
                                 for _ in range(rng.randint(1, 9)))
                         for _ in range(rng.randint(1, 6)))
        elif kind == "long":
            s = "Кто такой " + "а" * rng.randint(1000, 5000)
        elif kind == "bytes":
            raw = bytes(rng.randrange(0, 256) for _ in range(rng.randint(1, 64)))
            s = raw.decode("utf-8", errors="replace")
        elif kind == "repeat":
            s = rng.choice(("Кит — это млекопитающее", "Где находится Париж?",
                            "он физик")) * rng.randint(2, 8)
        else:
            s = "".join(rng.choice("".join(_ALPHABETS))
                        for _ in range(rng.randint(5, 40)))
        out.append(s)
    return out[:n]


def fuzz_pipeline(make_dm, n: int = 200, seed: int = FUZZ_SEED) -> dict:
    from ..generation import plan as make_plan
    from ..generation.nlg import render
    from ..pipeline import SemanticPipeline

    dm = make_dm()
    pipeline = SemanticPipeline(dm)
    inputs = generate_inputs(n, seed)
    crashes, empty = [], 0
    timings: list[float] = []
    for index, text in enumerate(inputs):
        started = time.perf_counter()
        try:
            answer = pipeline.respond(text)
            _, ir = make_plan(pipeline, text, "normal")
            rendered = render(ir, "detailed")
            if not (answer.text or "").strip() or not rendered.strip():
                empty += 1
        except Exception:                       # noqa: BLE001 — это и есть замер
            crashes.append({"index": index, "input": text[:200],
                            "trace": traceback.format_exc().strip().splitlines()[-1]})
        timings.append((time.perf_counter() - started) * 1000)
    timings.sort()
    return {
        "seed": seed,
        "n": len(inputs),
        "crashes": crashes,
        "crash_count": len(crashes),
        "empty_answers": empty,
        "max_ms": round(timings[-1], 2) if timings else 0.0,
        "p95_ms": round(timings[int(len(timings) * 0.95)], 2) if timings else 0.0,
        "median_ms": round(timings[len(timings) // 2], 2) if timings else 0.0,
    }


def render_fuzz(report: dict) -> str:
    lines = [f"Фаззинг входа (seed {report['seed']}, {report['n']} строк):"]
    lines.append(f"  падений: {report['crash_count']}")
    for crash in report["crashes"][:5]:
        lines.append(f"    - #{crash['index']} {crash['input']!r}: {crash['trace']}")
    lines.append(f"  пустых ответов: {report['empty_answers']}")
    lines.append(f"  время: медиана {report['median_ms']} мс, "
                 f"p95 {report['p95_ms']} мс, максимум {report['max_ms']} мс")
    return "\n".join(lines)
