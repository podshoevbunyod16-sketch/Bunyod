"""Deterministic Russian corpus builder (v0.15).

Content pilot built from the demo corpus (34 English-Russian facts) + hand-written
Russian TEMPLATES. Split by TEMPLATE FAMILY (paraphrases of one fact live in exactly
one split) — this is the anti-leakage boundary, verified by tests.

Counts are what they are (~340 train / ~80 dev / ~80 test): a seed for experiments,
NOT linguistic competence evidence.

Outputs: rasm_cim_v15/corpus/{train,dev,test}.jsonl + manifest.json
Deterministic: same inputs -> same sha256 for every file.
"""
from __future__ import annotations

import hashlib
import json
import os
from collections import defaultdict

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEMO_FACTS = os.path.join(_ROOT, "rasm_cim_v11", "data", "demo_facts.jsonl")
CORPUS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "corpus")

# Template families: (family_id, split, intent, template)
# {s} = subject, {o} = object. Split assigned BEFORE generation; families never straddle.
FAMILIES = [
    # capital_of — 9 families
    ("cap_q1", "train", "ask_fact", "Какая столица у {s}?"),
    ("cap_q2", "train", "ask_fact", "Столица {s} — это что?"),
    ("cap_q3", "dev",   "ask_fact", "Назови столицу {s}."),
    ("cap_q4", "train", "ask_fact", "Столица {s}?"),
    ("cap_q5", "test",  "ask_fact", "Где находится столица {s}?"),
    ("cap_q6", "train", "ask_fact", "Какая столица у {s}, скажите?"),
    ("cap_q7", "dev",   "ask_fact", "Какую столицу имеет {s}?"),
    ("cap_q8", "test",  "ask_fact", "Подскажите, какая столица у {s}"),
    ("cap_q9", "train", "ask_fact", "А столица {s}?"),
    ("cap_q10", "train", "ask_fact", "Напомни столицу {s}"),
    ("cap_q11", "train", "ask_fact", "Столицей {s} является?"),
    ("cap_q12", "train", "ask_fact", "Какой у {s} главный город?"),
    # is_a — 8 families (definition questions)
    ("isa_q1", "train", "ask_definition", "Что такое {o}?"),
    ("isa_q2", "train", "ask_definition", "Кто такой {o}?"),
    ("isa_q3", "dev",   "ask_definition", "Что значит {o}?"),
    ("isa_q4", "train", "ask_definition", "Что означает {o}?"),
    ("isa_q5", "test",  "ask_definition", "Определи, что такое {o}."),
    ("isa_q6", "train", "ask_definition", "Что представляет собой {o}?"),
    ("isa_q7", "dev",   "ask_definition", "Кто такие {o}?"),
    ("isa_q8", "test",  "ask_definition", "Дай определение: {o}"),
    ("isa_q9",  "train", "ask_definition", "Расскажи что такое {o}"),
    ("isa_q10", "train", "ask_definition", "Объясни, что такое {o}"),
    ("isa_q11", "train", "ask_definition", "Кем или чем является {o}?"),
    ("isa_q12", "train", "ask_definition", "Что знаешь про {o}? Это кто?"),
    # wrote — 6 families
    ("wrt_q1", "train", "ask_fact", "Кто написал {o}?"),
    ("wrt_q2", "train", "ask_fact", "Автор {o}?"),
    ("wrt_q3", "dev",   "ask_fact", "Кто автор произведения {o}?"),
    ("wrt_q4", "test",  "ask_fact", "Кем написана {o}?"),
    ("wrt_q5", "train", "ask_fact", "Автором {o} является?"),
    ("wrt_q6", "test",  "ask_fact", "Укажи автора: {o}"),
    ("wrt_q7",  "train", "ask_fact", "Кто автор «{o}»?"),
    # maybe can fly — 5 families (yes_no)
    ("fly_q1", "train", "yes_no", "Умеет ли {s} летать?"),
    ("fly_q2", "train", "yes_no", "Может ли {s} летать?"),
    ("fly_q3", "dev",   "yes_no", "{s} летает?"),
    ("fly_q4", "test",  "yes_no", "Способен ли {s} летать?"),
    ("fly_q5", "train", "yes_no", "Правда ли что {s} летает?"),
    ("fly_q6",  "train", "yes_no", "{s} способен летать?"),
    # exists — 4 families (contradiction: both polarities in KB)
    ("exs_q1", "train", "yes_no", "Правда ли что {s} существует?"),
    ("exs_q2", "dev",   "yes_no", "Существует ли {s}?"),
    ("exs_q3", "test",  "yes_no", "{s} реален?"),
    ("exs_q4", "train", "yes_no", "Верно ли что {s} существует?"),
    # population-ish (quantity) — 3 families (unknown in demo KB; tests abstain honestly)
    ("pop_q1", "dev",   "ask_fact", "Какое население у {s}?"),
    ("pop_q2", "test",  "ask_fact", "Сколько жителей в {s}?"),
    ("pop_q3", "train", "ask_fact", "Какая численность населения у {s}?"),
    # greetings — 4 families (no KB facts; intent only)
    ("grt_1",  "train", "greeting", "Привет!"),
    ("grt_2",  "train", "greeting", "Здравствуйте"),
    ("grt_3",  "dev",   "greeting", "Добрый день"),
    ("grt_4",  "test",  "greeting", "Доброе утро!"),
    ("grt_5",  "train", "greeting", "Хай!"),
    ("grt_6",  "train", "greeting", "Добрый вечер!"),
    # compare — 4 families (no KB facts; intent only)
    ("cmp_1",  "train", "compare", "Сравни {a} и {b}"),
    ("cmp_2",  "train", "compare", "Чем отличается {a} от {b}?"),
    ("cmp_3",  "dev",   "compare", "В чём разница между {a} и {b}?"),
    ("cmp_4",  "test",  "compare", "{a} лучше или хуже, чем {b}?"),
    ("cmp_5",  "train", "compare", "Сравнение: {a} и {b}"),
    # list — 3 families (intent only)
    ("lst_1",  "train", "list", "Перечисли известные тебе {cat}"),
    ("lst_2",  "dev",   "list", "какие есть {cat}?"),
    ("lst_3",  "test",  "list", "Назови все {cat}"),
    # unknown hard negatives — 5 families
    ("unk_1",  "train", "unknown", "asdfghjkl"),
    ("unk_2",  "dev",   "unknown", "кто-нибудь где-нибудь когда-нибудь"),
    ("unk_3",  "test",  "unknown", "!!!!!!!!"),
    ("unk_4",  "train", "unknown", "29 47 91"),
    ("unk_5",  "test",  "unknown", "эЭ?«»»"),
]

COMPARE_SUBJECTS = [("Франция", "Германия"), ("Киты", "Акула"), ("Пушкин", "Толстой"),
                    ("Марс", "Земля")]
LIST_CATS = ["планеты", "поэты", "животные", "страны"]


def _family_split(family):
    return family[1]


def build_corpus():
    facts = [json.loads(l) for l in open(DEMO_FACTS, encoding="utf-8")]
    by_rel = defaultdict(list)
    for f in facts:
        by_rel[f["relation"]].append(f)

    rows = []
    for fid, split, intent, template in FAMILIES:
        if intent == "greeting":
            rows.append({"id": f"{fid}-1", "text": template, "intent": intent,
                         "relation": None, "subject": None, "object": None,
                         "polarity": None, "family": fid, "split": split,
                         "source": "template:v0.15"})
            continue
        if intent == "compare":
            for i, (a, b) in enumerate(COMPARE_SUBJECTS * 2):   # 8 examples per family
                rows.append({"id": f"{fid}-{i}",
                             "text": template.format(a=a, b=b),
                             "intent": intent, "relation": None,
                             "subject": f"{a}|{b}", "object": None,
                             "polarity": None, "family": fid, "split": split,
                             "source": "template:v0.15"})
            continue
        if intent == "list":
            for i, cat in enumerate(LIST_CATS * 2):
                rows.append({"id": f"{fid}-{i}",
                             "text": template.format(cat=cat),
                             "intent": intent, "relation": None,
                             "subject": cat, "object": None,
                             "polarity": None, "family": fid, "split": split,
                             "source": "template:v0.15"})
            continue
        if intent == "unknown":
            rows.append({"id": f"{fid}-1", "text": template, "intent": intent,
                         "relation": None, "subject": None, "object": None,
                         "polarity": None, "family": fid, "split": split,
                         "source": "template:v0.15"})
            continue
        # fact-bound families
        for fact in facts:
            rel = fact["relation"]
            if fid.startswith("cap_") and rel == "capital_of":
                t = template.format(s=fact["subject"], o=fact["object"])
            elif fid.startswith("isa_") and rel == "is_a":
                t = template.format(s=fact["subject"], o=fact["object"])
            elif fid.startswith("wrt_") and rel == "написал":
                t = template.format(s=fact["subject"], o=fact["object"])
            elif fid.startswith("fly_") and rel == "может_летать":
                t = template.format(s=fact["subject"], o=fact["object"])
            elif fid.startswith("exs_") and rel == "существует":
                t = template.format(s=fact["subject"], o=fact["object"])
            elif fid.startswith("pop_") and rel == "capital_of":
                t = template.format(s=fact["subject"], o=fact["object"])
            else:
                continue
            rows.append({"id": f"{fid}-{fact['subject']}-{fact['object']}",
                         "text": t, "intent": intent, "relation": rel,
                         "subject": fact["subject"], "object": fact["object"],
                         "polarity": fact["polarity"], "family": fid,
                         "split": split, "source": "demo_facts+template:v0.15"})
    return rows


def write_corpus(out_dir: str = CORPUS_DIR) -> dict:
    rows = build_corpus()
    os.makedirs(out_dir, exist_ok=True)
    splits = {"train": [], "dev": [], "test": []}
    for r in rows:
        splits[r["split"]].append(r)
    manifest = {"schema": "rasm-cim-corpus/1", "split_policy":
                "template_family", "seed": 0, "files": {}}
    texts = []
    for split, items in splits.items():
        path = os.path.join(out_dir, f"{split}.jsonl")
        with open(path, "w", encoding="utf-8") as fh:
            for it in items:
                fh.write(json.dumps(it, ensure_ascii=False, sort_keys=True) + "\n")
        data = open(path, "rb").read()
        manifest["files"][split] = {"count": len(items),
                                    "sha256": hashlib.sha256(data).hexdigest()}
        texts.append(data)
    all_blob = b"".join(texts)
    manifest["corpus_sha256"] = hashlib.sha256(all_blob).hexdigest()
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)
    return manifest


if __name__ == "__main__":
    print(json.dumps(write_corpus(), ensure_ascii=False, indent=2))
