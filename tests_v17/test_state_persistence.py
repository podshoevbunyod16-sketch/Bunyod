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


# --- извлечение фактов из разговорной реплики ---------------------------------
def _graph_names(text: str) -> set[tuple[str, str]]:
    from rasm_cim_v16.dialogue_manager import DialogueManager as DM
    from rasm_cim_v16.persistence import Store
    from rasm_cim_v17.pipeline import SemanticPipeline

    home = tempfile.mkdtemp()
    dm = DM()
    Store(home=home, autosave=False).attach(dm)
    SemanticPipeline(dm).respond(text, "normal")
    return {(dm.kg.name_of(st.subject), dm.kg.name_of(st.object))
            for st in dm.kg.all_statements()}


def test_trailing_hedges_do_not_break_fact():
    for text in ("Пётр — это инженер, как я слышал.",
                 "Анна — это учитель, если не ошибаюсь.",
                 "Рустам — это геолог, это точно."):
        names = {(s.replace("ё", "е"), o) for s, o in _graph_names(text)}
        assert len(names) == 1, text


def test_two_facts_with_lead_in_and_trailing_question():
    names = {(s.replace("ё", "е"), o) for s, o in
             _graph_names("Слушай, Иван — это физик, а Мария — врач, понял?")}
    assert names == {("Иван", "физик"), ("Мария", "врач")}
    names = {(s.replace("ё", "е"), o) for s, o in
             _graph_names("Значит так: Пётр — это инженер, Анна — это учитель.")}
    assert names == {("Петр", "инженер"), ("Анна", "учитель")}


def test_broken_compound_does_not_half_teach():
    # если вторая часть не факт, первая тоже не должна попасть в базу:
    # иначе реплика, признанная неразобранной, всё равно меняет знания
    assert _graph_names("Иван — это физик, Мария читает книгу.") == set()


def test_fact_before_trailing_question_is_kept():
    # «..., Мария тоже?» — утверждение перед вопросом остаётся фактом
    names = {(s.replace("ё", "е"), o) for s, o in _graph_names("Иван — это физик, Мария тоже?")}
    assert names == {("Иван", "физик")}
