"""End-to-end API (REST + WebSocket) and CLI + restart persistence tests."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from modelforge.backend.api import create_app
from modelforge.cli.main import main as cli_main
from modelforge.core.runtime import ModelForgeCore


def test_fastapi_rest_and_websocket_e2e(tmp_path: Path):
    home = tmp_path / "api_home"
    core = ModelForgeCore(home_dir=home)
    app = create_app(core=core)

    try:
        with TestClient(app) as client:
            # Web Studio HTML
            html = client.get("/")
            assert html.status_code == 200
            assert "ModelForge AI" in html.text

            # Health & Doctor
            h = client.get("/api/v1/health").json()
            assert h["status"] == "ok"

            # WebSocket live events
            with client.websocket_connect("/ws/events") as ws:
                hello = ws.receive_json()
                assert hello["topic"] == "ws.connected"

                proj = client.post("/api/v1/projects", json={"name": "API Lab"}).json()
                evt = ws.receive_json()
                assert evt["topic"] == "project.created"

            # Model creation -> training -> generation -> prompt edit -> rollback
            mdl = client.post(
                "/api/v1/models/from-prompt",
                json={"project_id": proj["id"], "prompt": "Компактная модель для русского и таджикского языка"},
            ).json()

            tr = client.post(
                "/api/v1/training/run",
                json={"model_id": mdl["id"], "epochs": 3, "learning_rate": 0.005, "wait": True},
            ).json()
            assert tr["status"] == "completed"
            assert tr["result"]["metrics"]["weights_changed"] is True

            gen = client.post(
                "/api/v1/inference/generate",
                json={"model_id": mdl["id"], "prompt": "Привет", "max_tokens": 12},
            ).json()
            assert gen["tokens_generated"] > 0

            # Remote unreachable handling does not crash
            mode_resp = client.post(
                "/api/v1/mode",
                json={"mode": "remote", "remote_server_url": "http://127.0.0.1:59999"},
            ).json()
            assert mode_resp["remote_status"]["reachable"] is False
    finally:
        core.shutdown()


def test_cli_commands_and_persistence_across_restart(tmp_path: Path, capsys):
    home = str(tmp_path / "cli_home")

    assert cli_main(["--home", home, "--json", "create", "project", "--name", "CLI Project"]) == 0
    proj_out = json.loads(capsys.readouterr().out)
    pid = proj_out["id"]

    assert cli_main(["--home", home, "--json", "create", "model", "--project", pid, "--prompt", "Tiny GRU model for Russian and Tajik"]) == 0
    mdl_out = json.loads(capsys.readouterr().out)
    mid = mdl_out["id"]
    assert mdl_out["architecture"] == "gru"

    assert cli_main(["--home", home, "--json", "train", "--model", mid, "--epochs", "3"]) == 0
    train_out = json.loads(capsys.readouterr().out)
    assert train_out["metrics"]["weights_changed"] is True

    assert cli_main(["--home", home, "--json", "model", "generate", "--model", mid, "--prompt", "Салом", "--max-tokens", "10"]) == 0
    gen_out = json.loads(capsys.readouterr().out)
    assert gen_out["tokens_generated"] > 0

    assert cli_main(["--home", home, "--json", "export", "--model", mid, "--format", "gguf"]) == 0
    exp_out = json.loads(capsys.readouterr().out)
    assert Path(exp_out["path"]).exists()

    assert cli_main(["--home", home, "--json", "create", "agent", "--project", pid, "--prompt", "Research agent"]) == 0
    agt_out = json.loads(capsys.readouterr().out)
    aid = agt_out["id"]

    assert cli_main(["--home", home, "--json", "agent", "run", "--agent", aid, "--prompt", "Inspect workspace"]) == 0
    run_out = json.loads(capsys.readouterr().out)
    assert run_out["status"] == "completed"

    assert cli_main(["--home", home, "--json", "project", "edit", "--project", pid, "--prompt", "Добавь таджикский язык"]) == 0
    edit_out = json.loads(capsys.readouterr().out)
    assert edit_out["test_report"]["passed"] is True

    assert cli_main(["--home", home, "--json", "project", "rollback", "--edit-id", edit_out["id"]]) == 0
    rb_out = json.loads(capsys.readouterr().out)
    assert rb_out["rolled_back"] is True

    # Fresh restart verification: a brand-new ModelForgeCore instance sees persisted project, model, agent & checkpoint
    assert cli_main(["--home", home, "--json", "doctor"]) == 0
    doc_out = json.loads(capsys.readouterr().out)
    assert doc_out["healthy"] is True
    assert doc_out["counts"]["projects"] == 1
    assert doc_out["counts"]["models"] == 1
    assert doc_out["counts"]["agents"] == 1
