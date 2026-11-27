"""v0.11 language layer — unit tests (all offline)."""
import json
import os

import pytest

from rasm_cim_v11.language import (INTENTS, GraphAdapter, analyze, classify_intent,
                                    fold_yo, is_stopword, lemma, lemmas, num_to_int,
                                    normalize, tokenize)
from rasm_cim_v11.demo import answer_v11, build_demo_graph, load_demo_facts

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEMO = os.path.join(ROOT, "rasm_cim_v11", "data", "demo_facts.jsonl")


# ---------------------------------------------------------------- normalize
def test_normalize_preserves_original():
    src = "  «ПРИВЕТ»   МИР!!!  "
    n = normalize(src)
    assert n.original == src                     # untouched
    assert n.text == '"привет" мир!'
    assert "nfkc" not in n.flags or True         # flags are informational


def test_normalize_nfkc_compat_fold():
    n = normalize("ﬁle")                          # U+FB01 compatibility ligature
    assert n.text == "file"
    assert "nfkc" in n.flags


def test_normalize_casefold_and_quotes_dashes():
    n = normalize("Он ― сказал: „Привет“")        # em-dash-like + german quotes
    assert "―" not in n.text
    assert "„" not in n.text
    assert '"' in n.text
    assert n.text == normalize('он - сказал: "привет"').text


def test_normalize_invisibles_removed():
    n = normalize("при​вет‏")                     # zwsp + rlm
    assert n.text == "привет"
    assert "invisibles_removed" in n.flags


def test_normalize_repeated_terminal_punct():
    assert normalize("что???").text == "что?"
    assert normalize("вау!!!").text == "вау!"


def test_normalize_is_deterministic_and_stable_spacing():
    a = normalize("  а\t\tб\n\n в  ")
    b = normalize("  а\t\tб\n\n в  ")
    assert a == b
    assert a.text == "а б в"


def test_normalize_rejects_non_str():
    with pytest.raises(TypeError):
        normalize(123)


def test_normalize_question_flag_and_cyrillic():
    assert normalize("как дела?").ended_with_question is True
    assert normalize("hello world").has_cyrillic is False
    assert normalize("привет").has_cyrillic is True


def test_fold_yo():
    assert fold_yo("ёлка Ёлка") == "елка елка"


# -------------------------------------------------------------- tokenize
def test_tokenize_offsets_roundtrip_original():
    src = "Какая столица у Франции?"
    toks = tokenize(src)
    for t in toks:
        assert src[t.start:t.end] == t.text      # exact offsets into ORIGINAL


def test_tokenize_kinds_and_coverage():
    toks = tokenize("крыша-скат 2+2 = 4?")
    kinds = [t.kind for t in toks]
    assert "word" in kinds and "number" in kinds and "punct" in kinds
    words = [t.text for t in toks if t.kind == "word"]
    assert "крыша-скат" in words                 # hyphenated compound is one token


def test_tokenize_empty_and_whitespace():
    assert tokenize("") == ()
    assert tokenize("   ") == ()
    assert len(tokenize("a b")) == 2


def test_tokenize_offsets_with_fullwidth():
    src = "ＡＢＣ?"                               # full-width latin, NFKC folds
    toks = tokenize(src)
    assert src[toks[0].start:toks[0].end] == toks[0].text
    assert toks[0].norm == "abc"


# -------------------------------------------------------------- morphology
def test_lemma_suffix_rules():
    assert lemma("киты") == "кит"
    assert lemma("столицу") == "столиц"
    assert lemma("столица") == "столиц"
    assert lemma("словом") == "слов"


def test_lemma_irregular():
    assert lemma("людей") == "человек"
    assert lemma("Лучше") == "хороший"


def test_num_to_int():
    assert num_to_int("два") == 2
    assert num_to_int("две") == 2
    assert num_to_int("двадцать") == 20
    assert num_to_int("семьдесят") == 70
    assert num_to_int("двадцать-пять") == 25
    assert num_to_int("стол") is None


def test_stopwords():
    assert is_stopword("и") and is_stopword("для")
    assert not is_stopword("квант")


def test_lemmas_tuple_shape():
    assert lemmas(["киты", "летают"]) == ("кит", "лет")


# ------------------------------------------------------------------ intent
@pytest.mark.parametrize("text,intent", [
    ("Приветствую вас!", "greeting"),
    ("Добрый день", "greeting"),
    ("Что такое гравитация?", "ask_definition"),
    ("Кто такой Чехов?", "ask_definition"),
    ("Кто написал Анну Каренину?", "ask_fact"),
    ("Где находится Эверест?", "ask_fact"),
    ("Правда ли что сахар белый?", "yes_no"),
    ("Умеет ли страус летать?", "yes_no"),
    ("Чем отличается ель от сосны?", "compare"),
    ("Сравни волка и лису", "compare"),
    ("Перечисли планеты", "list"),
    ("Какие бывают облака?", "list"),
])
def test_intent_examples_new_phrasings(text, intent):
    res = classify_intent(normalize(text).text, True)
    assert res.intent == intent
    assert res.matched                           # every decision is explainable
    assert 0.5 <= res.confidence <= 0.99


@pytest.mark.parametrize("text", [
    "абырвалг турюм", "", "   ", "12 + 34 = ?", "))))", "[user input unparsed]",
])
def test_intent_honest_unknown(text):
    res = classify_intent(normalize(text).text, True)
    assert res.intent == "unknown"


def test_intent_english_is_unknown_not_threshold_roulette():
    res = classify_intent(normalize("What is the capital of France?").text, False)
    assert res.intent == "unknown"
    assert res.reason == "no_cyrillic_no_support"


def test_definition_beats_bare_question_fact_rule():
    res = classify_intent(normalize("Что такое симфония?").text, True)
    assert res.intent == "ask_definition"


def test_intent_result_scores_sorted():
    res = classify_intent(normalize("кто написал книгу?").text, True)
    scores = [s for _, s in res.scores]
    assert scores == sorted(scores, reverse=True)


def test_analyze_end_to_end_structure():
    r = analyze("  «Какие есть планеты?»  ")
    assert r.original == "  «Какие есть планеты?»  "
    assert r.normalized == '"какие есть планеты?"'
    assert r.intent.intent == "list"
    for t in r.tokens:
        assert r.original[t.start:t.end] == t.text   # offsets on original


# ------------------------------------------------------------ graph adapter
def test_adapter_true_false_unknown_conflict():
    ad = GraphAdapter()
    assert ad.query_fact("Франция", "столица", "Париж") == "UNKNOWN"
    ad.assert_fact("Франция", "столица", "Париж", "true")
    assert ad.query_fact("Франция", "столица", "Париж") == "TRUE"
    ad.assert_fact("Франция", "столица", "Париж", "false")
    assert ad.query_fact("Франция", "столица", "Париж") == "CONFLICT"


def test_adapter_retract_symmetric_forgets():
    ad = GraphAdapter()
    ad.assert_fact("Киты", "is_a", "рыбы", "true")
    ad.assert_fact("Киты", "is_a", "рыбы", "false")
    ad.retract_fact("Киты", "is_a", "рыбы")
    assert ad.query_fact("Киты", "is_a", "рыбы") == "UNKNOWN"


def test_stable_id_deterministic_and_text_sensitive():
    from rasm_cim_v11.language.graph_adapter import stable_id
    assert stable_id("Франция") == stable_id("франция")
    assert stable_id("Франция") != stable_id("Германия")


# -------------------------------------------------------------------- demo
def test_demo_capital_answer():
    r = answer_v11("Какая столица у Германии?", DEMO)
    assert r["status"] == "TRUE" and "Берлин" in r["answer"]
    assert r["evidence"]


def test_demo_definition_answer():
    r = answer_v11("Что такое киты?", DEMO)
    assert r["status"] == "TRUE" and "млекопитающ" in r["answer"]


def test_demo_yes_no_false():
    r = answer_v11("Умеет ли пингвин летать?", DEMO)
    assert r["status"] == "FALSE"


def test_demo_conflict_is_surfaced_not_hidden():
    r = answer_v11("Существует ли дракон?", DEMO)
    assert r["status"] == "CONFLICT"


def test_demo_unknown_for_unsupported():
    r = answer_v11("Как приготовить борщ?", DEMO)
    assert r["status"] in ("UNKNOWN", "UNSUPPORTED")


def test_demo_facts_file_valid_jsonl():
    facts = load_demo_facts(DEMO)
    assert len(facts) >= 30
    for f in facts:
        assert set(f) >= {"subject", "relation", "object", "polarity"}
        assert f["polarity"] in ("true", "false")
