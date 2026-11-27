"""Plan A — one command line for the whole stack, including ``python -m rasm_cim``."""
from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from conftest import FIXTURES, V11_FACTS, write_legacy_kb, legacy_entity, legacy_statement

from rasm_cim_v17.cli import build_parser, main

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run(*argv: str) -> int:
    return main(list(argv))


def test_parser_exposes_the_documented_commands():
    sub = [a for a in build_parser()._actions if a.dest == "cmd"][0]
    names = set(sub.choices)
    assert {"chat", "ask", "teach", "say", "import", "imports", "restore",
            "verify", "learned", "migrate", "stats", "entities", "graph",
            "evidence", "conflicts", "version"} <= names


def test_version(capsys):
    assert run("version") == 0
    out = capsys.readouterr().out
    assert "rasm-cim v0.17.0" in out and "offline" in out


def test_dry_run_prints_a_report_and_writes_nothing(tmp_path, capsys):
    home = str(tmp_path / "home")
    kb = write_legacy_kb(
        str(tmp_path / "kb"),
        [legacy_entity("e1", "place", "Франция"),
         legacy_entity("e2", "place", "Париж")],
        [legacy_statement("s1", "e1", "capital_of", object_entity="e2")])
    assert run("--home", home, "import", kb, "--dry-run") == 0
    out = capsys.readouterr().out
    assert "ИМПОРТ (dry-run)" in out and "capital_of" in out
    assert not os.path.exists(os.path.join(home, "kb.json"))


def test_import_then_ask_then_verify(tmp_path, capsys):
    home = str(tmp_path / "home")
    assert run("--home", home, "import", V11_FACTS) == 0
    assert "ИМПОРТ выполнен" in capsys.readouterr().out

    assert run("--home", home, "ask", "Где столица Франция?", "--json") == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] in ("DIRECT", "INFERRED", "CONTEXTUAL")
    assert payload["text"]

    assert run("--home", home, "verify", "--json") == 0
    report = json.loads(capsys.readouterr().out)
    assert report["ok"] is True
    assert report["stats"]["statements"] >= 34


def test_second_import_reports_duplicate(tmp_path, capsys):
    home = str(tmp_path / "home")
    assert run("--home", home, "import", V11_FACTS) == 0
    capsys.readouterr()
    assert run("--home", home, "import", V11_FACTS) == 0
    assert "уже импортирован" in capsys.readouterr().out


def test_imports_listing_and_restore(tmp_path, capsys):
    home = str(tmp_path / "home")
    assert run("--home", home, "teach", "Иван — это человек") == 0
    capsys.readouterr()
    assert run("--home", home, "import", V11_FACTS) == 0
    capsys.readouterr()

    assert run("--home", home, "imports", "--json") == 0
    entries = json.loads(capsys.readouterr().out)["entries"]
    assert len(entries) == 1
    ledger_id = entries[0]["id"]

    assert run("--home", home, "restore", ledger_id) == 0
    assert "восстановлено kb" in capsys.readouterr().out


def test_entities_and_graph_listings(tmp_path, capsys):
    home = str(tmp_path / "home")
    run("--home", home, "teach", "Исаак Ньютон — это физик")
    capsys.readouterr()
    assert run("--home", home, "entities") == 0
    assert "Исаак Ньютон" in capsys.readouterr().out
    assert run("--home", home, "graph") == 0
    assert "физик" in capsys.readouterr().out


def test_learned_summary_cli(tmp_path, capsys):
    from conftest import write_learned
    d = write_learned(str(tmp_path / "learned"))
    assert run("learned", d) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["verdict"] == "VERIFIED" and summary["version"] == "lv_000"


def test_unknown_command_is_a_usage_error(capsys):
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(["teleport"])
    assert exc.value.code == 2


def test_missing_file_is_reported_not_traced(tmp_path, capsys):
    code = run("--home", str(tmp_path / "h"), "import",
               str(tmp_path / "nope.jsonl"))
    assert code == 2
    assert "не найден" in capsys.readouterr().err


# --------------------------------------------------------------------------
# real module entry points, in a subprocess
# --------------------------------------------------------------------------

def _py(*argv: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", *argv], cwd=REPO,
                          capture_output=True, text=True, timeout=120)


def test_python_m_rasm_cim_v17_works():
    p = _py("rasm_cim_v17", "version")
    assert p.returncode == 0, p.stderr
    assert "v0.17.0" in p.stdout


def test_python_m_rasm_cim_works():
    """The unified entry point that v0.16 lacked."""
    p = _py("rasm_cim", "version")
    assert p.returncode == 0, p.stderr
    assert "v0.17.0" in p.stdout


def test_two_processes_share_one_database(tmp_path):
    home = str(tmp_path / "home")
    p1 = _py("rasm_cim", "--home", home, "teach", "Париж — это город")
    assert p1.returncode == 0, p1.stderr
    p2 = _py("rasm_cim", "--home", home, "ask", "Что такое Париж?")
    assert p2.returncode == 0, p2.stderr
    assert "город" in p2.stdout.lower()


def test_python_m_rasm_cim_imports_the_v11_corpus(tmp_path):
    home = str(tmp_path / "home")
    p = _py("rasm_cim", "--home", home, "import", V11_FACTS, "--json")
    assert p.returncode == 0, p.stderr
    payload = json.loads(p.stdout)
    assert payload["plan"]["statements_importable"] == 34
    assert payload["result"]["statements_added"] == 34
