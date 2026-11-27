"""CLI агентного режима: команды agent, vocab и чат."""
from __future__ import annotations


def test_cli_agent_command(tmp_path, capsys):
    from rasm_cim_v17.cli import main

    home = str(tmp_path / "home")
    assert main(["agent", "привет", "--home", home]) == 0
    out = capsys.readouterr().out
    assert "Привет" in out
    assert main(["agent", "Иван — это физик", "--home", home]) == 0
    capsys.readouterr()
    assert main(["agent", "Кто такой Иван?", "--home", home]) == 0
    assert "Иван" in capsys.readouterr().out
    assert main(["vocab", "--home", home]) == 0
    assert "физик" in capsys.readouterr().out


def test_cli_agent_json(tmp_path, capsys):
    import json

    from rasm_cim_v17.cli import main

    home = str(tmp_path / "home")
    assert main(["agent", "Кто такой Некто?", "--json", "--home", home]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["kind"] in ("offer", "clarify", "error")
    assert "text" in payload and "source" in payload


def test_cli_chat_classic_still_available(tmp_path, capsys, monkeypatch):
    from rasm_cim_v17.cli import main

    answers = iter(["Иван — это физик", ":quit"])
    monkeypatch.setattr("builtins.input", lambda *_: next(answers))
    home = str(tmp_path / "home")
    assert main(["chat", "--classic", "--home", home]) == 0
    assert "режим normal" in capsys.readouterr().out


def test_cli_chat_agent_mode_learns(tmp_path, capsys, monkeypatch):
    from rasm_cim_v17.cli import main

    answers = iter(["Иван — это физик", "Кто такой Иван?", ":quit"])
    monkeypatch.setattr("builtins.input", lambda *_: next(answers))
    home = str(tmp_path / "home")
    assert main(["chat", "--home", home]) == 0
    out = capsys.readouterr().out
    assert "агентный режим" in out
    assert "Запомнил" in out and "физик" in out
