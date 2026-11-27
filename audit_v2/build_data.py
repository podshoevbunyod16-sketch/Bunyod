"""Собрать наборы T1–T5 из рукописных источников и запечатать hidden.

T1/T2 переиспользуют уже сданные независимые наборы этапа аудита
(`audit_v1/data`) и дополняются рукописными кейсами до 100 на задачу.
T3–T5 целиком рукописные. Генераторов и шаблонных подстановок нет: каждый кейс
— литерал в `authoring/`.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from audit_v1.harness import load_cases

from .authoring.coref_supplement import HIDDEN as HIDDEN_COREF
from .authoring.coref_supplement import T1_NEW, T2_NEW
from .authoring.qa_extract import EDGE, EXTRACT, QA

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data"
AUTHOR = "arena-agent (исследователь-инженер)"
DATE = "2026-11-27"
METHOD = "рукописный кейс; разметка автора, сверена со вторым разметчиком (см. отчёт)"


def _row(case_id: str, task: str, **fields) -> dict:
    row = {"id": case_id, "task": task, "author": AUTHOR, "date": DATE,
           "method": METHOD}
    row.update(fields)
    return row


def build_t1() -> list[dict]:
    rows = []
    for case in load_cases(Path("audit_v1/data/bind.jsonl")):
        rows.append(_row(f"t1-a-{case.id}", "t1", category=case.category,
                         turns=list(case.turns), expected="BIND",
                         entity=case.entity, pronoun=case.pronoun))
    for i, (category, entity, pronoun, turns) in enumerate(T1_NEW, start=1):
        rows.append(_row(f"t1-s-{i:03d}", "t1", category=category,
                         turns=list(turns), expected="BIND", entity=entity,
                         pronoun=pronoun))
    return rows


def build_t2() -> list[dict]:
    rows = []
    for case in load_cases(Path("audit_v1/data/refuse.jsonl")):
        rows.append(_row(f"t2-a-{case.id}", "t2", category=case.category,
                         turns=list(case.turns), expected="REFUSE", entity="",
                         pronoun=case.pronoun, refuse_reason=case.refuse_reason))
    for case in load_cases(Path("audit_v1/data/mixed.jsonl")):
        if case.expected != "REFUSE":
            continue
        rows.append(_row(f"t2-m-{case.id}", "t2", category=case.category,
                         turns=list(case.turns), expected="REFUSE", entity="",
                         pronoun=case.pronoun, refuse_reason=case.refuse_reason))
    for i, (category, pronoun, reason, turns) in enumerate(T2_NEW, start=1):
        rows.append(_row(f"t2-s-{i:03d}", "t2", category=category,
                         turns=list(turns), expected="REFUSE", entity="",
                         pronoun=pronoun, refuse_reason=reason))
    return rows


def build_t3() -> list[dict]:
    rows = []
    for i, (facts, question, expected, answerable) in enumerate(QA, start=1):
        rows.append(_row(f"t3-{i:03d}", "t3", category="qa",
                         text=list(facts), question=question,
                         answerable=answerable, expected=expected))
    return rows


def build_t4() -> list[dict]:
    rows = []
    for i, (sentences, triples, note) in enumerate(EXTRACT, start=1):
        rows.append(_row(f"t4-{i:03d}", "t4", category=note,
                         sentences=list(sentences),
                         expected=[list(pair) for pair in triples]))
    return rows


def build_t5() -> list[dict]:
    rows = []
    for case in QA:
        rows.append(None)
    rows = []
    for i, (facts, question, _expected, _answerable) in enumerate(QA, start=1):
        rows.append(_row(f"t5-{i:03d}", "t5", category="from_qa",
                         text=list(facts) if facts else [], question=question,
                         required_fields=["answer", "status", "evidence"]))
    for i, (text, question) in enumerate(EDGE, start=1):
        rows.append(_row(f"t5-e-{i:03d}", "t5", category="edge",
                         text=[text] if text else [], question=question,
                         required_fields=["answer", "status", "evidence"]))
    return rows


def build_hidden() -> list[dict]:
    rows = []
    for i, (task, category, expected, entity, pronoun, reason, turns) in \
            enumerate(HIDDEN_COREF, start=1):
        row = _row(f"h-{i:03d}", task, category=category, turns=list(turns),
                   expected=expected, entity=entity, pronoun=pronoun)
        if expected == "REFUSE":
            row["refuse_reason"] = reason
        rows.append(row)
    qa_hidden = [(["Фирдавс — это механик.", "Ситора — это переводчик."], "Кто такой Фирдавс?", "Фирдавс", True),
                 (["Фирдавс — это механик.", "Ситора — это переводчик."], "Кто такая Ситора?", "Ситора", True),
                 (["Олег — это пилот."], "Олег — это пилот?", "Олег", True),
                 (["Малика — это повар.", "Ахмад — это агроном."], "Кто такая Малика?", "Малика", True),
                 (["Тимур — это экономист."], "Кто такой Тимур?", "Тимур", True),
                 (["Фирдавс — это механик."], "Кто такой Олег?", "", False),
                 (["Ситора — это переводчик."], "Кто такая Малика?", "", False),
                 (["Олег — это пилот."], "Кто такой Фирдавс?", "", False),
                 (["Малика — это повар."], "Кто такая Ситора?", "", False),
                 (["Тимур — это экономист."], "Что такое экономика?", "", False),
                 (["Привет.", "Как дела?"], "Кто такой Фирдавс?", "", False),
                 (["Салам, дуст.", "Хуб."], "Кто такая Ситора?", "", False),
                 (["Фирдавс — это механик, Ситора — это переводчик."], "Кто такой Фирдавс?", "Фирдавс", True),
                 (["Олег — это пилот, Малика — это повар."], "Кто такая Малика?", "Малика", True),
                 (["Тимур — это экономист."], "Ахмад — это агроном?", "", False)]
    for i, (facts, question, expected, answerable) in enumerate(qa_hidden, start=1):
        rows.append(_row(f"h-t3-{i:03d}", "t3", category="qa", text=list(facts),
                         question=question, answerable=answerable,
                         expected=expected))
    ex_hidden = [
        (["Фирдавс — это механик."], [("Фирдавс", "механик")]),
        (["Ситора — это переводчик."], [("Ситора", "переводчик")]),
        (["Олег — это пилот, Малика — это повар."], [("Олег", "пилот"), ("Малика", "повар")]),
        (["Привет, как дела?"], []),
        (["Бале, Тимур — это экономист."], [("Тимур", "экономист")]),
        (["Ахмад — это агроном."], [("Ахмад", "агроном")]),
        (["Ситора — переводчик."], [("Ситора", "переводчик")]),
        (["Олег — это пилот?", "Малика — это повар?"], []),
        (["Фирдавс — это механик, а Олег — это пилот."], [("Фирдавс", "механик"), ("Олег", "пилот")]),
        (["Мария — это врач."], [("Мария", "врач")]),
        (["Спасибо большое."], []),
        (["Иван — это физик."], [("Иван", "физик")]),
        (["Пётр — это инженр."], [("Пётр", "инженр")]),
        (["Анна — это учитель."], [("Анна", "учитель")]),
        (["Хуб, Мария — это врач."], [("Мария", "врач")]),
    ]
    for i, (sentences, triples) in enumerate(ex_hidden, start=1):
        rows.append(_row(f"h-t4-{i:03d}", "t4", category="hidden",
                         sentences=list(sentences),
                         expected=[list(pair) for pair in triples]))
    for i, (text, question) in enumerate([
        ("Фирдавс — это механик.", "Кто такой Фирдавс?"),
        (None, "Кто такой Фирдавс?"),
        ("Ситора — это переводчик.", "Она переводчик?"),
        ("Олег — это пилот.", "Сколько лет Олегу?"),
        ("Малика — это повар.", ""),
        (None, ""),
        ("Тимур — это экономист.", "Тимур — это кто?"),
        ("Фирдавс — это механик, Ситора — это переводчик.", "Кто из них механик?"),
        ("Олег — это пилот.", "КТО ТАКОЙ ОЛЕГ?"),
        ("Малика — это повар.", "Кто такая Малика? Спасибо."),
        ("Ситора — это переводчик.", "«Ситора» — это имя?"),
        ("Тимур — это экономист.", "Расскажи всё про Тимура подробно."),
        ("Ахмад — это агроном.", "Ахмад — это агроном?"),
        ("Фирдавс — это механик.", "Кто такой Олег?"),
        ("Олег — это пилот.", "Кто такой Олег и что такое авиация?"),
    ], start=1):
        rows.append(_row(f"h-t5-{i:03d}", "t5", category="hidden",
                         text=[text] if text else [], question=question,
                         required_fields=["answer", "status", "evidence"]))
    return rows


def write() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    sets = {"t1": build_t1(), "t2": build_t2(), "t3": build_t3(),
            "t4": build_t4(), "t5": build_t5()}
    for name, rows in sets.items():
        path = OUT / f"{name}.jsonl"
        path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
                        encoding="utf-8")
        print(f"{name}: {len(rows)}")
    hidden = build_hidden()
    (OUT / "hidden.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in hidden) + "\n",
        encoding="utf-8")
    print(f"hidden: {len(hidden)}")
    files = sorted(list(OUT.glob("t*.jsonl")) + [OUT / "hidden.jsonl"])
    lines = [f"{hashlib.sha256(f.read_bytes()).hexdigest()}  {f.name}" for f in files]
    (OUT / "SEALED.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("sealed:", len(lines))


if __name__ == "__main__":
    write()
