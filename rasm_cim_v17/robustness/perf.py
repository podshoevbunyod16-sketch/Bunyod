"""Where the time and memory go — measured, on this machine, with the caveat.

``benchmark`` answers one answer's latency (median of repeated runs) and the
peak memory of a whole question set via ``tracemalloc``.  ``scale_test`` grows
the knowledge base and reports how the per-query cost moves, because "works on
Termux" is a claim about growth, not about a single lucky measurement.

These are numbers from the CI/desktop machine.  The Termux device is not
available here — see ``docs_v17/TERMUX.md`` and ``LIMITATIONS.md``.
"""
from __future__ import annotations

import statistics
import time
import tracemalloc

DEFAULT_QUERIES = (
    "Кто написал Евгений Онегин?",
    "Где находится Париж?",
    "Кто такой Пушкин?",
    "Кит — это млекопитающее?",
    "Кто президент Марса?",
)


def _time_call(fn, repeat: int) -> tuple[float, float, float]:
    times: list[float] = []
    for _ in range(max(1, repeat)):
        started = time.perf_counter()
        fn()
        times.append((time.perf_counter() - started) * 1000)
    return statistics.median(times), sum(times) / len(times), max(times)


def benchmark(make_dm, queries=DEFAULT_QUERIES, repeat: int = 5) -> dict:
    from ..pipeline import SemanticPipeline

    started = time.perf_counter()
    dm = make_dm()
    cold_ms = (time.perf_counter() - started) * 1000
    pipeline = SemanticPipeline(dm)
    rows = []
    for query in queries:
        median, mean, worst = _time_call(lambda q=query: pipeline.respond(q), repeat)
        rows.append({"query": query, "median_ms": round(median, 3),
                     "mean_ms": round(mean, 3), "max_ms": round(worst, 3)})
    medians = sorted(r["median_ms"] for r in rows)

    tracemalloc.start()
    for query in queries:
        pipeline.respond(query)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    statements = len(dm.kg.all_statements())
    return {
        "cold_start_ms": round(cold_ms, 2),
        "queries": rows,
        "p50_ms": round(medians[len(medians) // 2], 3) if medians else 0.0,
        "p95_ms": round(medians[min(len(medians) - 1, int(len(medians) * 0.95))], 3)
        if medians else 0.0,
        "max_ms": round(max(medians), 3) if medians else 0.0,
        "peak_kb": round(peak / 1024, 1),
        "current_kb": round(current / 1024, 1),
        "statements": statements,
        "python": __import__("sys").version.split()[0],
    }


def scale_test(make_dm, sizes=(50, 100, 200, 400), query="Где находится Париж?") -> dict:
    """Teach ``size`` extra facts, then time the same question."""
    from ..pipeline import SemanticPipeline

    dm = make_dm()
    pipeline = SemanticPipeline(dm)
    rows = []
    added = 0
    for size in sizes:
        while added < size:
            name = f"Город{added}"
            pipeline.respond(f"{name} находится в Страна{added % 7}")
            pipeline.respond(f"Страна{added % 7} находится в Регион{added % 3}")
            added += 2
        median, _, _ = _time_call(lambda: pipeline.respond(query), 5)
        rows.append({"statements": len(dm.kg.all_statements()),
                     "median_ms": round(median, 3)})
    first = rows[0]["median_ms"] or 0.001
    last = rows[-1]["median_ms"]
    return {"rows": rows, "growth_factor": round(last / first, 2),
            "query": query}


def render_benchmark(report: dict) -> str:
    lines = [f"Производительность (Python {report['python']}):"]
    lines.append(f"  холодный старт демо-базы: {report['cold_start_ms']} мс, "
                 f"{report['statements']} утверждений")
    lines.append(f"  запрос: p50 {report['p50_ms']} мс, p95 {report['p95_ms']} мс, "
                 f"худший {report['max_ms']} мс")
    lines.append(f"  пиковая память на серию вопросов: {report['peak_kb']} КБ")
    for row in report["queries"]:
        lines.append(f"    {row['median_ms']:>6} мс  {row['query']}")
    return "\n".join(lines)
