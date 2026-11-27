"""Состояние диалога v0.17 должно переживать запуск CLI.

Исправление («Нет, я про Анну») и память упоминаний раньше жили только в
памяти процесса: второй вызов `rasm-cim say` их не видел, и местоимение
теряло антецедент. Тесты проверяют общее поведение через два отдельных
процесса, а не конкретную фразу.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile

from rasm_cim_v16.dialogue_manager import DialogueManager
from rasm_cim_v17.discourse.memory import DialogueMemory
from rasm_cim_v17.discourse.persistence import load_state, save_state

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run(home: str, *args: str) -> str:
    env = dict(os.environ)
    env["RASM_CIM_HOME"] = home
    proc = subprocess.run([sys.executable, "-m", "rasm_cim", *args, "--home", home],
                          cwd=ROOT, capture_output=True, text=True, env=env,
                          timeout=120)
    return proc.stdout


def test_save_load_roundtrip():
    class Holder:
        pass

    dm = Holder()
    dm.v17_explicit_focus = {"entity_id": 7, "role": "correction", "turn": 3}
    dm.v17_mentioned = {7, 9}
    dm.v17_memory = DialogueMemory()
    dm.v17_memory.note(7, "Анна", turn=3)
    with tempfile.TemporaryDirectory() as home:
        assert save_state(dm, home) is True
        assert save_state(dm, home) is False        # без изменений не переписываем
        restored = Holder()
        assert load_state(restored, home) is True
        assert restored.v17_explicit_focus["entity_id"] == 7
        assert restored.v17_mentioned == {7, 9}
        assert restored.v17_memory.count(7) == 1


def test_correction_survives_two_cli_invocations():
    with tempfile.TemporaryDirectory() as home:
        _run(home, "say", "Иван — это физик.")
        _run(home, "say", "Мария — это врач.")
        _run(home, "say", "Кто такой Иван?")
        ack = _run(home, "say", "Нет, я про Марию.")
        assert "Мария" in ack
        answer = _run(home, "say", "Она врач?")
        assert "Мария" in answer
        assert "Да" in answer


def test_mention_memory_survives_in_cli():
    with tempfile.TemporaryDirectory() as home:
        _run(home, "say", "Рустам — это геолог.")
        _run(home, "say", "Кто такой Рустам?")
        answer = _run(home, "say", "Он геолог?")
        assert "Рустам" in answer
