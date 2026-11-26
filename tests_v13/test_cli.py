"""v0.13 CLI smoke tests."""
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _seed(tmp_path):
    kb = str(tmp_path / "kb")
    subprocess.run([sys.executable, "-m", "rasm_cim_v12.cli", "seed-demo", kb],
                   capture_output=True, check=True, cwd=ROOT)
    return kb


def test_cli_ask_and_context_roundtrip(tmp_path):
    kb = _seed(tmp_path)
    ctx = str(tmp_path / "ctx")
    r = subprocess.run([sys.executable, "-m", "rasm_cim_v13.cli", "ask", ctx,
                        "--kb", kb, "Какая столица у Германии?"],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0
    out = json.loads(r.stdout)
    assert out["results"][0]["status"] == "TRUE"
    r = subprocess.run([sys.executable, "-m", "rasm_cim_v13.cli", "show-context", ctx],
                       capture_output=True, text=True, cwd=ROOT)
    summary = json.loads(r.stdout)
    assert summary["turns"] >= 2 and "Германия" in summary["recent_entities"]
    r = subprocess.run([sys.executable, "-m", "rasm_cim_v13.cli", "export-session", ctx],
                       capture_output=True, text=True, cwd=ROOT)
    session = json.loads(r.stdout)
    assert session["schema"] == "rasm-cim-context/1"
    r = subprocess.run([sys.executable, "-m", "rasm_cim_v13.cli", "reset-context", ctx],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0
    r = subprocess.run([sys.executable, "-m", "rasm_cim_v13.cli", "show-context", ctx],
                       capture_output=True, text=True, cwd=ROOT)
    assert json.loads(r.stdout)["turns"] == 0


def test_cli_ambiguous_returns_exit_4(tmp_path):
    kb = _seed(tmp_path)
    ctx = str(tmp_path / "ctx")
    subprocess.run([sys.executable, "-m", "rasm_cim_v13.cli", "ask", ctx,
                    "--kb", kb, "Какие столицы у Германии и Франции?"],
                   capture_output=True, cwd=ROOT)
    r = subprocess.run([sys.executable, "-m", "rasm_cim_v13.cli", "ask", ctx,
                        "--kb", kb, "А её?"],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 4
    assert "Германия" in r.stdout and "Франция" in r.stdout
