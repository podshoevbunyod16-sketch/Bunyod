"""TEST M / N + spec §9/§10 — answer content, modes, statuses, explanations.
"""
from __future__ import annotations

import pytest

from rasm_cim_v16.answer_engine import (A_AMBIGUOUS, A_CONTEXTUAL,
                                        A_CONTRADICTED, A_DIRECT,
                                        A_INFERRED, A_UNKNOWN, MODE_CONCISE,
                                        MODE_DETAILED, MODE_NORMAL,
                                        MODE_ALIASES, AnswerEngine)


# ------------------------------------------------------------------ TEST M
def test_m_unknown_instead_of_an_invented_answer(dm):
    a = dm.say("Кто написал Войну и мир?")
    assert a.status == A_UNKNOWN
    assert a.text.startswith("Не знаю")
    assert a.confidence == 0.0
    assert a.facts == []


def test_m_unknown_names_what_is_missing(dm):
    dm.say("teach Исаак Ньютон — это физик")
    a = dm.say("Кто открыл закон всемирного тяготения?")
    assert a.status == A_UNKNOWN
    assert a.missing["reason"]
    assert a.missing["predicate"] == "discovered"


def test_m_unknown_still_says_what_is_known(dm):
    dm.say("teach Исаак Ньютон — это физик")
    a = dm.say("Где находится Исаак Ньютон?")
    assert a.status == A_UNKNOWN
    assert "Известно" in a.detailed or "известно" in a.normal


def test_m_no_numbers_appear_out_of_nowhere(dm):
    dm.say("teach Исаак Ньютон — это физик")
    a = dm.say("Когда родился Исаак Ньютон?", mode=MODE_DETAILED)
    assert a.status == A_UNKNOWN
    assert not any(ch.isdigit() for ch in a.text.replace("2026", ""))


# ------------------------------------------------------------------ TEST N
def test_n_detailed_answer_is_built_from_real_facts(dm):
    for t in ["teach Исаак Ньютон — это физик",
              "teach Ньютон — это Исаак Ньютон",
              "teach Исаак Ньютон открыл закон всемирного тяготения",
              "teach Исаак Ньютон сформулировал законы движения"]:
        dm.say(t)
    a = dm.say("Расскажи о Ньютоне", mode=MODE_DETAILED)
    assert a.status in (A_INFERRED, A_CONTEXTUAL)
    assert "закон всемирного тяготения" in a.detailed
    assert "законы движения" in a.detailed
    assert "физик" in a.detailed
    assert len(a.facts) >= 3
    assert a.evidence, "a detailed answer must cite evidence"
    assert len(a.detailed) > len(a.normal) > len(a.short)


def test_n_answer_is_not_a_single_line_when_data_exists(dm):
    for t in ["teach Исаак Ньютон — это физик",
              "teach Исаак Ньютон открыл закон всемирного тяготения"]:
        dm.say(t)
    a = dm.say("Кто такой Исаак Ньютон?", mode=MODE_NORMAL)
    assert "\n" in a.normal
    assert "открыл" in a.normal


def test_n_answer_lists_only_true_facts(dm):
    dm.say("teach Исаак Ньютон открыл закон всемирного тяготения")
    a = dm.say("Расскажи о Исаак Ньютоне", mode=MODE_DETAILED)
    for line in a.detailed.splitlines():
        if line.strip().startswith("•"):
            assert "закон всемирного тяготения" in line or \
                   "Чего база" in line or "•" in line


def test_n_unknown_relations_are_reported(dm):
    dm.say("teach Исаак Ньютон — это физик")
    a = dm.say("Расскажи о Исаак Ньютоне", mode=MODE_DETAILED)
    assert "Чего база НЕ знает" in a.detailed
    assert a.missing["unknown_relations"]


# ------------------------------------------------------------------- modes
def test_three_modes_are_supported(dm):
    dm.say("teach Исаак Ньютон — это физик")
    dm.say("teach Исаак Ньютон открыл закон всемирного тяготения")
    c = dm.say("Расскажи о Исаак Ньютоне", mode=MODE_CONCISE)
    n = dm.say("Расскажи о Исаак Ньютоне", mode=MODE_NORMAL)
    d = dm.say("Расскажи о Исаак Ньютоне", mode=MODE_DETAILED)
    assert len(c.text) <= len(n.text) <= len(d.text)


@pytest.mark.parametrize("alias,canon", sorted(MODE_ALIASES.items()))
def test_legacy_mode_aliases(alias, canon):
    assert MODE_ALIASES[alias] == canon
    assert MODE_ALIASES[alias] in (MODE_CONCISE, MODE_NORMAL, MODE_DETAILED)


def test_render_picks_the_requested_mode(dm):
    dm.say("teach Исаак Ньютон — это физик")
    a = dm.say("Кто такой Исаак Ньютон?")
    assert a.render(MODE_CONCISE) == a.short
    assert a.render(MODE_NORMAL) == a.normal
    assert a.render(MODE_DETAILED) == a.detailed


# ---------------------------------------------------------------- statuses
def test_status_direct_for_a_stored_fact(dm):
    dm.say("teach Исаак Ньютон — это физик")
    assert dm.say("Исаак Ньютон — физик?").status == A_DIRECT


def test_status_inferred_for_a_derived_fact(dm):
    dm.say("teach Исаак Ньютон — это физик")
    dm.say("teach Ньютон — это Исаак Ньютон")
    assert dm.say("Ньютон — физик?").status == A_INFERRED


def test_status_contextual_when_a_pronoun_was_resolved(dm):
    dm.say("teach Исаак Ньютон открыл закон всемирного тяготения")
    assert dm.say("Что он открыл?").status == A_CONTEXTUAL


def test_status_ambiguous(dm):
    dm.say("teach Исаак Ньютон — это физик")
    dm.say("teach Джон Ньютон — это богослов")
    assert dm.say("Ньютон — физик?").status == A_AMBIGUOUS


def test_status_contradicted(dm):
    dm.say("teach Исаак Ньютон — это физик")
    dm.say("teach Исаак Ньютон не был физиком")
    assert dm.say("Ньютон — физик?").status == "CONTRADICTED"


# ------------------------------------------------------------ explanations
def test_explain_command_after_a_question(dm):
    dm.say("teach Исаак Ньютон — это физик")
    dm.say("teach Ньютон — это Исаак Ньютон")
    dm.say("Ньютон — физик?")
    out = dm.command("explain", "").text
    assert "Статус: INFERRED" in out
    assert "alias_bridge" in out
    assert "Использованные факты" in out
    assert "Не подтверждено" in out


def test_explain_command_for_a_statement(dm):
    dm.say("teach Исаак Ньютон — это физик")
    out = dm.command("explain", "st-000001").text
    assert "st-000001" in out


def test_explain_without_a_previous_answer(dm):
    assert "Пока не было ответа" in dm.command("explain", "").text


def test_answer_carries_evidence_records(dm):
    dm.say("teach Исаак Ньютон — это физик")
    a = dm.say("Исаак Ньютон — физик?")
    assert a.evidence
    assert a.evidence[0].utterance == "Исаак Ньютон — это физик"


def test_answer_serialises(dm):
    dm.say("teach Исаак Ньютон — это физик")
    d = dm.say("Исаак Ньютон — физик?").as_dict()
    assert set(d) >= {"status", "mode", "confidence", "concise", "normal",
                      "detailed", "facts", "evidence", "explanation",
                      "clarification", "missing", "conflicts", "used_context"}


# ---------------------------------------------------------------- commands
def test_inspect_command(dm):
    dm.say("teach Исаак Ньютон — это физик")
    out = dm.command("inspect", "").text
    assert "Сущностей: 2" in out
    assert "фактов: 1" in out


def test_entities_command(dm):
    dm.say("teach Исаак Ньютон — это физик")
    out = dm.command("entities", "").text
    assert "Исаак Ньютон" in out and "физик" in out


def test_graph_command(dm):
    dm.say("teach Исаак Ньютон — это физик")
    out = dm.command("graph", "").text
    assert "st-000001" in out
    assert "instance_of" in out or "экземпляр" in out


def test_graph_command_for_one_entity(dm):
    dm.say("teach Исаак Ньютон — это физик")
    assert "st-000001" in dm.command("graph", "Исаак Ньютон").text


def test_evidence_command(dm):
    dm.say("teach Исаак Ньютон — это физик")
    out = dm.command("evidence", "st-000001").text
    assert "user_assertion" in out


def test_context_command(dm):
    dm.say("teach Исаак Ньютон — это физик")
    dm.say("Кто такой Исаак Ньютон?")
    out = dm.command("context", "").text
    assert "Текущая тема" in out and "Активные сущности" in out
    assert "Целостность: OK" in out


def test_memory_command_lists_four_tiers(dm):
    dm.say("teach Исаак Ньютон — это физик")
    out = dm.command("memory", "").text
    for tier in ("Рабочая память", "Эпизодическая память",
                 "Семантическая память", "Процедурная память"):
        assert tier in out


def test_rules_command(dm):
    out = dm.command("rules", "").text
    for rule in ("alias_bridge", "symmetric", "inverse",
                 "instance_subclass", "transitive"):
        assert rule in out


def test_conflicts_command(dm):
    dm.say("teach Исаак Ньютон — это физик")
    dm.say("teach Исаак Ньютон не был физиком")
    out = dm.command("conflicts", "").text
    assert "POLARITY" in out


def test_stats_command_is_json(dm):
    import json
    dm.say("teach Исаак Ньютон — это физик")
    d = json.loads(dm.command("stats", "").text)
    assert d["kg"]["statements"] == 1


def test_help_command_lists_every_command(dm):
    out = dm.command("help", "").text
    for cmd in (":inspect", ":evidence", ":context", ":memory", ":entities",
                ":graph", ":explain", ":conflicts", ":rules", ":merge",
                ":reset", ":save", ":stats"):
        assert cmd in out


def test_unknown_command_is_reported(dm):
    assert "Неизвестная команда" in dm.command("nosuch", "").text


def test_merge_command(dm):
    dm.say("teach Исаак Ньютон — это физик")
    dm.say("teach Сэр Исаак Ньютон открыл закон всемирного тяготения")
    out = dm.command("merge", "Сэр Исаак Ньютон | Исаак Ньютон")
    assert out.status == A_DIRECT
    a = dm.say("Исаак Ньютон — физик?")
    assert a.text.startswith("Да")


def test_greeting(dm):
    assert dm.say("привет").status == A_DIRECT


def test_unparseable_input_is_honest(dm):
    a = dm.say("asdf qwer")
    assert a.status == A_UNKNOWN
    assert "Не понял" in a.text


def test_two_questions_in_one_message(dm):
    dm.say("teach Исаак Ньютон — это физик")
    dm.say("teach Компьютер — это устройство")
    a = dm.say("Кто такой Исаак Ньютон? Что такое компьютер?")
    assert "физик" in a.text and "устройство" in a.text


# ------------------------------------------------ answer modes are distinct
def test_all_three_renderings_exist_on_every_answer(dm):
    """An Answer must carry concise, normal AND detailed text no matter which
    mode it was built in — otherwise asking in one mode and printing another
    silently degrades the output (regression: `--answer detailed` printed the
    normal text because the detailed block was only built on demand)."""
    for t in ["teach Исаак Ньютон — это физик",
              "teach Ньютон — это Исаак Ньютон",
              "teach Физик — это учёный",
              "teach Учёный — это человек"]:
        dm.say(t)
    for built_in in (MODE_CONCISE, MODE_NORMAL, MODE_DETAILED):
        a = dm.say("Ньютон — учёный?", mode=built_in)
        assert len(a.detailed) > len(a.normal) > len(a.short), built_in
        assert "Доказательства:" in a.detailed, built_in
        assert "Как получен ответ:" in a.detailed, built_in
        assert a.render(MODE_DETAILED) == a.detailed
        assert a.render(MODE_CONCISE) == a.short


def test_cli_answer_mode_reaches_the_builder(tmp_path):
    """End-to-end through the real CLI: the same question must produce three
    different outputs, and the flag must work both before and after the
    subcommand (regression: argparse subparser defaults overwrote it)."""
    import os
    import subprocess
    import sys
    home = str(tmp_path)
    env = dict(os.environ)
    py = sys.executable
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for t in ["Исаак Ньютон — это физик", "Ньютон — это Исаак Ньютон"]:
        subprocess.run([py, "-m", "rasm_cim_v16", "--home", home, "teach", t],
                       check=True, capture_output=True, text=True, cwd=root,
                       env=env)

    def ask(*pre):
        r = subprocess.run([py, "-m", "rasm_cim_v16", "--home", home, *pre],
                           capture_output=True, text=True, cwd=root, env=env,
                           timeout=120)
        assert r.returncode == 0, r.stderr
        return r.stdout

    concise = ask("--answer", "concise", "ask", "Ньютон — физик?")
    normal = ask("ask", "Ньютон — физик?", "--answer", "normal")
    detailed = ask("--answer", "detailed", "ask", "Ньютон — физик?")
    assert len(concise) < len(normal) < len(detailed), (
        concise, normal, detailed)
    assert "Доказательства:" in detailed
