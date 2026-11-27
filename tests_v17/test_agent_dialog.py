"""Агентный слой: диалог, инструменты, память.

Тесты проверяют общее поведение, а не конкретные фразы:

* без сети агент честно говорит, что сеть выключена, и ничего не выдумывает;
* с сетью — отвечает по источнику, показывает ссылку и предлагает запомнить;
* в память факт попадает только после подтверждения;
* если в источнике нет ответа, агент так и говорит;
* инструмент отказывает хостам вне списка разрешённых.
"""
from __future__ import annotations

import json
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

import pytest

from rasm_cim_v16.dialogue_manager import DialogueManager
from rasm_cim_v16.persistence import Store
from rasm_cim_v17.agent.session import AgentSession
from rasm_cim_v17.agent.text import paraphrase, parse_definition, rank
from rasm_cim_v17.agent.tools import Web, host_allowed

PAGE = """<html><head><title>Кулибин</title></head><body>
<p>Иван Кулибин — это русский механик-изобретатель.</p>
<p>Кулибин построил знаменитые часы «яйцо» и служил при дворе.</p>
<p>Он родился в Нижнем Новгороде в семье торговца.</p>
</body></html>"""

UNRELATED = """<html><body>
<p>Погода в Душанбе сегодня солнечная, температура около двадцати градусов.</p>
<p>Завтра ожидается небольшой ветер и ясное небо.</p>
</body></html>"""


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):                                   # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/search":
            query = parse_qs(parsed.query).get("q", [""])[0]
            mode = parse_qs(parsed.query).get("mode", ["ok"])[0]
            port = self.server.server_address[1]
            if mode == "unrelated":
                body = {"results": [{"title": "Погода", "text": "Погода на сегодня",
                                     "url": f"http://127.0.0.1:{port}/weather"}]}
            elif mode == "empty":
                body = {"results": []}
            else:
                body = {"results": [{
                    "title": "Кулибин",
                    "url": f"http://127.0.0.1:{port}/wiki/Kulibin",
                    "text": "Иван Кулибин — это русский механик-изобретатель.",
                    "_query": query}]}
            self._send(json.dumps(body, ensure_ascii=False), "application/json")
        elif parsed.path == "/wiki/Kulibin":
            self._send(PAGE, "text/html; charset=utf-8")
        elif parsed.path == "/weather":
            self._send(UNRELATED, "text/html; charset=utf-8")
        elif parsed.path == "/broken":
            self.send_response(500)
            self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()

    def _send(self, body: str, ctype: str) -> None:
        data = body.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):                        # тишина в выводе тестов
        pass


@pytest.fixture()
def stub():
    server = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    yield f"http://127.0.0.1:{port}"
    server.shutdown()
    thread.join(timeout=5)


def _session(home: str, **kwargs) -> AgentSession:
    dm = DialogueManager()
    Store(home=home, autosave=False).attach(dm)
    return AgentSession(dm, home, mode="normal", **kwargs)


def _facts(session: AgentSession) -> set[tuple[str, str]]:
    kg = session.dm.kg
    return {(kg.name_of(st.subject), kg.name_of(st.object))
            for st in kg.all_statements()}


# ---------------------------------------------------------------- текст
def test_rank_and_paraphrase_use_source_only():
    source = ["Иван Кулибин — это русский механик-изобретатель.",
              "Он родился в Нижнем Новгороде в семье торговца.",
              "Погода сегодня хорошая."]
    selected = rank(source, "Кто такой Кулибин?")
    assert selected, "релевантное предложение должно быть найдено"
    text = paraphrase(selected, "Кто такой Кулибин?")
    assert "Кулибин" in text
    assert "Погода" not in text           # нерелевантное не попадает в ответ


def test_definition_parse_ignores_hedges():
    assert parse_definition("Иван Кулибин — это русский механик.") == \
        ("Иван Кулибин", "русский механик")
    assert parse_definition("Кулибин — якобы механик.") is None
    assert parse_definition("Он родился в Нижнем Новгороде.") is None


def test_host_allowlist():
    assert host_allowed("http://127.0.0.1:8000/x")
    assert not host_allowed("https://example.com/x")
    web = Web(tempfile.mkdtemp(), online=True)
    assert web.fetch("https://example.com/x").ok is False


# ---------------------------------------------------------------- офлайн
def test_offline_agent_does_not_pretend_to_search():
    home = tempfile.mkdtemp()
    session = _session(home)
    reply = session.step("Кто такой Кулибин?")
    assert reply.kind == "offer"
    assert "--online" in reply.text
    assert _facts(session) == set()


def test_explicit_search_offline_is_an_honest_error():
    home = tempfile.mkdtemp()
    reply = _session(home).step("поищи про Кулибина")
    assert reply.kind == "error"
    assert "сеть выключена" in reply.text


def test_smalltalk_and_capabilities():
    home = tempfile.mkdtemp()
    session = _session(home)
    assert session.step("привет").kind == "smalltalk"
    capabilities = session.step("Что ты умеешь?")
    assert capabilities.kind == "smalltalk"
    assert "не умею" in capabilities.text.lower()


def test_teach_then_answer():
    home = tempfile.mkdtemp()
    session = _session(home)
    learned = session.step("Иван — это физик")
    assert learned.kind == "learn"
    answer = session.step("Кто такой Иван?")
    assert answer.kind == "answer"
    assert "Иван" in answer.text


# ------------------------------------------------------------------ онлайн
def test_online_answer_has_source_and_can_be_confirmed(stub):
    home = tempfile.mkdtemp()
    session = _session(home)
    session.set_online(True)
    session.web.endpoint = f"{stub}/search?mode=ok&q={{query}}"
    reply = session.step("Кто такой Кулибин?")
    assert reply.kind == "offer", reply.text
    assert "Кулибин" in reply.text
    assert reply.source.startswith("http://127.0.0.1")
    assert reply.proposed, "из определения должен появиться кандидат в память"
    assert _facts(session) == set(), "до подтверждения в граф ничего не пишем"

    confirmed = session.step("да")
    assert confirmed.kind == "learn"
    assert ("Иван Кулибин", "русский механик-изобретатель") in _facts(session)
    vocab = session.vocabulary()
    assert "механик" in vocab.text or "Кулибин" in vocab.text


def test_refusal_of_proposal_does_not_write(stub):
    home = tempfile.mkdtemp()
    session = _session(home)
    session.set_online(True)
    session.web.endpoint = f"{stub}/search?mode=ok&q={{query}}"
    session.step("Кто такой Кулибин?")
    reply = session.step("нет")
    assert reply.kind == "learn"
    assert _facts(session) == set()


def test_irrelevant_source_is_not_summarised(stub):
    home = tempfile.mkdtemp()
    session = _session(home)
    session.set_online(True)
    session.web.endpoint = f"{stub}/search?mode=unrelated&q={{query}}"
    reply = session.step("Кто такой Кулибин?")
    assert reply.kind == "error"
    assert "не буду выдумывать" in reply.text.lower()
    assert _facts(session) == set()
    assert session.memory.pending_facts() == []


def test_empty_search_results_are_reported(stub):
    home = tempfile.mkdtemp()
    session = _session(home)
    session.set_online(True)
    session.web.endpoint = f"{stub}/search?mode=empty&q={{query}}"
    reply = session.step("Кто такой Кулибин?")
    assert reply.kind == "error"


def test_failed_fetch_is_reported(stub):
    home = tempfile.mkdtemp()
    web = Web(home, online=True)
    result = web.fetch(f"{stub}/broken")
    assert result.ok is False
    assert "нет доступа" in result.error


# ------------------------------------------------- диалог по своей базе
def test_class_and_plural_questions_use_own_kb():
    home = tempfile.mkdtemp()
    session = _session(home)
    session.step("Иван — это физик")
    session.step("Пётр — это физик")
    session.step("Мария — это врач")
    plural = session.step("Кто такие физики?")
    assert plural.kind == "answer"
    low = plural.text.casefold().replace("ё", "е")
    assert "иван" in low and "петр" in low
    singular = session.step("Что такое врач?")
    assert singular.kind == "answer" and "Мария" in singular.text
    person = session.step("Кто такой Иван?")
    assert person.kind == "answer" and "физик" in person.text
    assert "Мария" not in plural.text


def test_statement_without_parsable_copula_is_offered_then_learned():
    home = tempfile.mkdtemp()
    session = _session(home)
    offered = session.step("Москва — столица России")
    assert offered.kind == "offer" and "да" in offered.text
    learned = session.step("да")
    assert learned.kind == "learn"
    assert ("Москва", "столица") in _facts(session)
    # полная формулировка не потерялась: она в словаре
    definition = session.step("Что такое столица России?")
    assert definition.kind == "answer"
    assert "Москва" in definition.text


def test_unknown_entity_asks_and_does_not_invent():
    home = tempfile.mkdtemp()
    session = _session(home)
    reply = session.step("Кто такой Зюзин?")
    assert reply.kind in ("offer", "clarify")
    assert _facts(session) == set()


def test_vocabulary_lists_learned_classes():
    home = tempfile.mkdtemp()
    session = _session(home)
    session.step("Анна — это учитель")
    vocab = session.vocabulary()
    assert "учитель" in vocab.text
    assert "Анна" in vocab.text
