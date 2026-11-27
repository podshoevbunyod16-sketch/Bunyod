"""Plan G tests — fuzzing, offline guarantees, latency and memory.

Two of these tests exist because the fuzzer and the scale run found real crashes
in earlier plans (empty input; `Resolution.best` called as a method).  They stay
as regressions.
"""
from __future__ import annotations

import json
import pathlib
import subprocess
import sys

import pytest

from rasm_cim_v16.dialogue_manager import DialogueManager
from rasm_cim_v17.pipeline import SemanticPipeline
from rasm_cim_v17.robustness import (FUZZ_SEED, benchmark, fuzz_pipeline,
                                     generate_inputs, render_benchmark,
                                     render_fuzz, scale_test, scan_offline,
                                     scan_paths)
from rasm_cim_v17.semantics import build_demo

REPO = pathlib.Path(__file__).resolve().parents[1]
PKG = REPO / "rasm_cim_v17"


# --------------------------------------------------------------------------
# generator
# --------------------------------------------------------------------------
def test_generator_is_deterministic():
    assert generate_inputs(30, 7) == generate_inputs(30, 7)
    assert generate_inputs(30, 7) != generate_inputs(30, 8)


def test_generator_covers_hostile_categories():
    inputs = generate_inputs(60, FUZZ_SEED)
    assert "" in inputs
    assert any(len(x) > 1000 for x in inputs)          # длинные строки
    assert any(any(ch in x for ch in "\x00\u200b\ufeff") for x in inputs)
    assert any(any(ord(ch) > 0x1F000 for ch in x) for x in inputs)  # эмодзи
    assert any(x.count("Кит") > 1 for x in inputs)      # повторы


def test_generator_respects_n():
    assert len(generate_inputs(25)) == 25
    assert len(generate_inputs(5)) == 5


# --------------------------------------------------------------------------
# fuzzing
# --------------------------------------------------------------------------
def test_fuzz_finds_no_crashes():
    report = fuzz_pipeline(build_demo, n=60)
    assert report["crash_count"] == 0, report["crashes"]
    assert report["empty_answers"] == 0
    assert report["n"] == 60


def test_fuzz_report_is_self_describing():
    text = render_fuzz(fuzz_pipeline(build_demo, n=20))
    for needle in ("Фаззинг", "падений:", "пустых", "время"):
        assert needle in text


def test_empty_input_is_answered_not_raised():
    """Regression: v0.16 raises IndexError on '' — the pipeline must not."""
    pipe = SemanticPipeline(build_demo())
    for text in ("", " ", "\n\t "):
        answer = pipe.respond(text)
        assert answer.status in ("UNKNOWN", "AMBIGUOUS")
        assert "пустой ввод" in answer.text


def test_ambiguous_mention_does_not_crash():
    """Regression: `Resolution.best` is a property; calling it raised TypeError."""
    dm = DialogueManager()
    pipe = SemanticPipeline(dm)
    for name in ("Страна0", "Страна1", "Страна2"):
        pipe.respond(f"{name} находится в Регион")
    answer = pipe.respond("Страна2 находится в Регион?")
    assert answer.status in ("UNKNOWN", "AMBIGUOUS", "DIRECT", "INFERRED")
    assert pipe.last_meaning is not None


def test_fuzz_inputs_never_leave_a_mark():
    dm = build_demo()
    statements = len(dm.kg.all_statements())
    fuzz_pipeline(lambda: dm, n=40)
    # какой-то мусор может быть принят за обучение, но база не должна
    # разрушаться: все исходные утверждения на месте
    assert len(dm.kg.all_statements()) >= statements


# --------------------------------------------------------------------------
# offline scan
# --------------------------------------------------------------------------
def test_package_has_no_network_imports():
    report = scan_offline(PKG)
    assert report["clean"] is True, report["violations"]
    assert report["files"] > 20


def test_scan_detects_a_planted_violation(tmp_path):
    bad = tmp_path / "bad.py"
    bad.write_text("import socket\nimport urllib.request\n"
                   "import subprocess\n"
                   "subprocess.run(['curl', 'http://example.com'])\n",
                   encoding="utf-8")
    report = scan_paths([bad])
    kinds = {(f["kind"], f["detail"]) for f in report["violations"]}
    assert ("import", "socket") in kinds
    assert any("curl" in detail for _, detail in kinds)
    assert report["clean"] is False


def test_comment_about_network_is_not_a_violation(tmp_path):
    good = tmp_path / "good.py"
    good.write_text("# здесь нет socket, http и requests\nx = 1\n",
                    encoding="utf-8")
    assert scan_paths([good])["clean"] is True


def test_syntax_error_is_reported_separately(tmp_path):
    broken = tmp_path / "broken.py"
    broken.write_text("def (:\n", encoding="utf-8")
    report = scan_paths([broken])
    assert report["clean"] is True                     # не сетевое нарушение
    assert report["findings"][0]["kind"] == "syntax"


# --------------------------------------------------------------------------
# performance
# --------------------------------------------------------------------------
def test_benchmark_numbers_are_sane():
    report = benchmark(build_demo, repeat=3)
    assert report["queries"] and len(report["queries"]) == 5
    assert report["p50_ms"] < 50, report
    assert report["max_ms"] < 200, report
    assert report["peak_kb"] < 20_000, report
    assert report["statements"] > 0
    assert "Производительность" in render_benchmark(report)


def test_benchmark_is_repeatable_in_order_of_magnitude():
    first = benchmark(build_demo, repeat=3)["p50_ms"]
    second = benchmark(build_demo, repeat=3)["p50_ms"]
    assert second <= max(first * 20, first + 5)        # тайминги не «прыгают»


def test_scale_test_shows_bounded_growth():
    report = scale_test(build_demo, sizes=(40, 80, 160))
    assert len(report["rows"]) == 3
    assert report["rows"][-1]["statements"] > report["rows"][0]["statements"]
    assert report["growth_factor"] < 10, report


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def test_cli_stress_json_is_complete():
    out = subprocess.run([sys.executable, "-m", "rasm_cim", "stress",
                          "--n", "20", "--json"],
                         cwd=REPO, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    payload = json.loads(out.stdout)
    assert set(payload) == {"fuzz", "offline", "perf", "scale"}
    assert payload["fuzz"]["crash_count"] == 0
    assert payload["offline"]["clean"] is True
    assert payload["perf"]["p50_ms"] > 0


def test_cli_stress_text_mentions_network_and_time():
    out = subprocess.run([sys.executable, "-m", "rasm_cim", "stress", "--n", "10"],
                         cwd=REPO, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert "Сеть:" in out.stdout and "Производительность" in out.stdout
