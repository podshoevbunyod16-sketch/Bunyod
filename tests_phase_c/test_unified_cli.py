"""Phase C tests — unified `python -m rasm_cim` entry."""
import json
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ENV = dict(os.environ)


def run(args, *, cwd=ROOT, stdin=None, timeout=180):
    return subprocess.run([sys.executable, "-m", "rasm_cim", *args],
                          capture_output=True, text=True, cwd=cwd,
                          input=stdin, timeout=timeout, env=_ENV)


def _kb(tmp_path):
    kb = str(tmp_path / "kb.json")
    subprocess.run([sys.executable, "-m", "rasm_cim_v12.cli", "seed-demo", kb],
                   capture_output=True, check=True, cwd=ROOT)
    return kb


class TestAskAndTeach:
    def test_ask_json_and_text(self, tmp_path):
        kb = _kb(tmp_path)
        r = run(["--kb", kb, "ask", "Какая столица у Франции?"])
        assert r.returncode == 0
        assert "Париж" in r.stdout
        assert r.stderr.strip(), "diagnostics should go to stderr"
        r = run(["--kb", kb, "ask", "--json", "Какая столица у Франции?"])
        out = json.loads(r.stdout)
        assert out["verdict"] == "TRUE"
        assert out["evidence_ids"], "evidence ids must be present"

    def test_ask_conflict_exit_code(self, tmp_path):
        kb = _kb(tmp_path)
        r = run(["--kb", kb, "ask", "Правда ли что дракон существует?"])
        assert r.returncode == 4

    def test_teach_roundtrip(self, tmp_path):
        kb = str(tmp_path / "empty.json")          # start empty
        r = run(["--kb", kb, "ask", "Что такое виртуализация?"])
        before = json.loads(r.stdout) if r.stdout.strip().startswith("{") else None
        r = run(["--kb", kb, "teach", "Виртуализация — это абстракция ресурсов"])
        assert r.returncode == 0
        r = run(["--kb", kb, "memory", "stats"])
        stats = json.loads(r.stdout)
        assert stats["entities"] >= 2


class TestMemoryMode:
    def test_stats_list_conflicts_aliases(self, tmp_path):
        kb = _kb(tmp_path)
        r = run(["--kb", kb, "memory", "stats"])
        stats = json.loads(r.stdout)
        assert stats["entities"] > 0 and stats["statements_active"] > 0
        r = run(["--kb", kb, "memory", "list", "--limit", "5"])
        rows = json.loads(r.stdout)
        assert 1 <= len(rows) <= 5
        r = run(["--kb", kb, "memory", "conflicts"])
        assert isinstance(json.loads(r.stdout), list)
        r = run(["--kb", kb, "memory", "aliases"])
        assert isinstance(json.loads(r.stdout), list)

    def test_export_import_roundtrip(self, tmp_path):
        kb = _kb(tmp_path)
        out = str(tmp_path / "dump.jsonl")
        r = run(["--kb", kb, "memory", "export", out])
        assert r.returncode == 0
        assert os.path.exists(out)
        kb2 = str(tmp_path / "kb2.json")
        r = run(["--kb", kb2, "memory", "import", out])
        assert r.returncode == 0
        r = run(["--kb", kb2, "memory", "stats"])
        assert json.loads(r.stdout)["statements_active"] >= 1


class TestEvaluateAndSystemInfo:
    def test_evaluate_exit_0(self, tmp_path):
        kb = _kb(tmp_path)
        r = run(["--kb", kb, "evaluate", "--split", "dev"])
        assert r.returncode == 0
        out = json.loads(r.stdout)
        assert out["regression"]["ok"] is True

    def test_system_info_reports_freeze_and_version(self, tmp_path):
        kb = _kb(tmp_path)
        r = run(["--kb", kb, "system-info"])
        assert r.returncode == 0
        out = json.loads(r.stdout)
        assert out["version"].startswith("0.15")
        assert "freeze_baseline_v0.10" in out

    def test_legacy_entry_point_not_broken(self):
        r = subprocess.run([sys.executable, "-m", "rasm_cim_v9.cli", "--help"],
                           capture_output=True, text=True, cwd=ROOT)
        assert r.returncode == 0


class TestChatInteractive:
    def test_scripted_dialog(self, tmp_path):
        kb = _kb(tmp_path)
        ctx = str(tmp_path / "ctx.json")
        script = "Какая столица у Франции?\n:inspect\nreset\n:exit\n"
        r = run(["--kb", kb, "--ctx", ctx, "chat"], stdin=script)
        assert r.returncode == 0
        assert "Париж" in r.stdout
        assert "st_" in r.stdout          # evidence inspect shows statement ids

    def test_chat_reset_clears_context(self, tmp_path):
        kb = _kb(tmp_path)
        ctx = str(tmp_path / "ctx.json")
        r = run(["--kb", kb, "--ctx", ctx, "chat"],
                stdin="Какая столица у Франции?\nreset\n:exit\n")
        assert r.returncode == 0
