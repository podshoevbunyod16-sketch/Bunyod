"""Plan B — CLI surface for the semantic layer."""
from __future__ import annotations

import json
import os

from rasm_cim_v17.cli import main
from rasm_cim_v17.semantics import DEFAULT_CORPUS


def run(*argv):
    return main(list(argv))


def test_metrics_json_contains_every_family(tmp_path, capsys):
    assert run("metrics", "--json") == 0
    report = json.loads(capsys.readouterr().out)
    assert set(report) >= {"corpus", "intents", "entities", "answers", "seed"}
    assert report["corpus"]["size"] >= 30
    assert report["intents"]["accuracy"] > 0.9
    assert report["entities"]["f1"] > 0.9
    assert report["answers"]["unsupported_rate"] == 0.0
    assert report["seed"]["statements"] > 0


def test_metrics_text_report_names_the_corpus_hash(capsys):
    assert run("metrics") == 0
    out = capsys.readouterr().out
    assert "МЕТРИКИ" in out and "sha256" in out
    assert "речевой акт" in out and "сущности" in out and "ответы" in out


def test_metrics_accepts_a_custom_corpus(tmp_path, capsys):
    p = tmp_path / "mini.jsonl"
    p.write_text('{"text": "Привет", "intent": "GREETING"}\n', encoding="utf-8")
    assert run("metrics", "--corpus", str(p), "--json") == 0
    report = json.loads(capsys.readouterr().out)
    assert report["corpus"]["size"] == 1
    assert report["intents"]["accuracy"] == 1.0


def test_metrics_without_the_v11_corpus_still_runs(capsys):
    assert run("metrics", "--no-v11", "--json") == 0
    report = json.loads(capsys.readouterr().out)
    assert report["seed"]["statements"] > 0


def test_meaning_json_explains_the_parse(capsys):
    assert run("meaning", "Кем был Исаак Ньютон?", "--json") == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload[0]["speech_act"] == "ASK_DEFINITION"
    assert any("refine" in n for n in payload[0]["notes"])


def test_meaning_text_lists_presuppositions(capsys):
    assert run("meaning", "Что открыл Исаак Ньютон?") == 0
    out = capsys.readouterr().out
    assert "пресуппозиция" in out


def test_core_flag_keeps_the_frozen_behaviour(tmp_path, capsys):
    home = str(tmp_path / "home")
    assert run("--home", home, "ask", "Перечисли, что открыл Исаак Ньютон",
               "--core", "--json") == 0
    payload = json.loads(capsys.readouterr().out)
    # v0.16 treats the imperative as a teach and reports it as DIRECT
    assert payload["status"] == "DIRECT"
    assert "Записал" in payload["text"]


def test_default_path_does_not_teach_a_question(tmp_path, capsys):
    home = str(tmp_path / "home")
    assert run("--home", home, "ask", "Перечисли, что открыл Исаак Ньютон",
               "--json") == 0
    payload = json.loads(capsys.readouterr().out)
    assert "Записал" not in payload["text"]
    assert payload["status"] == "UNKNOWN"       # nothing is known in this home


def test_explain_meaning_goes_to_stderr(tmp_path, capsys):
    home = str(tmp_path / "home")
    assert run("--home", home, "ask", "Кто такой Ньютон?",
               "--explain-meaning") == 0
    captured = capsys.readouterr()
    assert "[смысл]" in captured.err
    assert captured.out


def test_parser_lists_the_new_commands():
    from rasm_cim_v17.cli import build_parser
    sub = [a for a in build_parser()._actions if a.dest == "cmd"][0]
    assert {"metrics", "meaning"} <= set(sub.choices)
