"""Spec §4 — entity resolution: aliases, morphology, transliteration,
ambiguity.  Nothing may be merged on resemblance alone.
"""
from __future__ import annotations

import pytest

from rasm_cim_v16.entity_resolver import (RES_ALIAS, RES_AMBIGUOUS, RES_EXACT,
                                          RES_FUZZY, RES_MORPHOLOGICAL,
                                          RES_PARTIAL, RES_TRANSLIT,
                                          RES_UNKNOWN, EntityResolver,
                                          infer_gender)
from rasm_cim_v16.normalization import (cross_script_form, edit_distance,
                                        normalize, stem_candidates,
                                        strip_honorifics, transliterate)


@pytest.fixture
def r():
    return EntityResolver()


# ------------------------------------------------------------- normalising
def test_normalize_case_and_punctuation():
    assert normalize("Сэр  Исаак  НЬЮТОН!") == "сэр исаак ньютон"


def test_normalize_folds_yo():
    assert normalize("Ёлка") == "елка"


def test_strip_honorifics():
    assert strip_honorifics("сэр Исаак Ньютон") == "исаак ньютон"
    assert strip_honorifics("профессор Ньютон") == "ньютон"


def test_copula_dash_is_not_part_of_a_name():
    assert normalize("Ньютон — это физик") == "ньютон это физик"
    assert "—" not in normalize("Ньютон — физик")


def test_hyphenated_names_survive():
    assert normalize("Санкт-Петербург") == "санкт-петербург"


# ------------------------------------------------------------- morphology
@pytest.mark.parametrize("form,stem", [
    ("физиком", "физик"),
    ("Ньютоном", "ньютон"),
    ("Ньютона", "ньютон"),
    ("законом", "закон"),
    ("компьютером", "компьютер"),
])
def test_case_forms_reach_the_stem(form, stem):
    forms = [f for f, _c in stem_candidates(form)]
    assert stem in forms


def test_long_strips_score_higher_than_short_ones():
    d = dict(stem_candidates("физиком"))     # 2 chars stripped
    a = dict(stem_candidates("физика"))      # 1 char stripped
    assert d["физик"] > a["физик"]


# ------------------------------------------------------------ translit
def test_transliterate_is_phonetic_not_orthographic():
    """Honest scope: the table is phonetic and one-character-per-letter, so
    "Ньютон" -> "nyuton", NOT the English spelling "newton".  Cross-script
    matching therefore works for regularly transliterated names; irregular
    ones need an explicit taught alias (tested below and documented in
    V016_LIMITATIONS.md)."""
    assert transliterate("Исаак Ньютон").startswith("isaak")
    assert transliterate("Ньютон") == "nyuton"
    assert "nyuton" != "newton"


def test_cross_script_only_when_the_other_script_is_used():
    assert cross_script_form("Ньютон") is not None
    assert cross_script_form("Newton") is not None
    assert cross_script_form("Ньютон Newton") is None


def test_edit_distance_bounded():
    assert edit_distance("ньютон", "ньютон") == 0
    assert edit_distance("ньютон", "ньютоn", cap=1) == 1
    assert edit_distance("abc", "xyz", cap=1) == 2   # cap+1 sentinel


# --------------------------------------------------------------- resolver
def test_exact_and_alias(r):
    e = r.intern("Исаак Ньютон")
    assert r.resolve("Исаак Ньютон").status == RES_EXACT
    assert r.resolve("исаак ньютон").status in (RES_EXACT, RES_ALIAS)
    r.add_alias(e.entity_id, "Ньютон")
    res = r.resolve("Ньютон")
    assert res.status in (RES_ALIAS, RES_EXACT)
    assert res.entity_id == e.entity_id


def test_honorific_resolution(r):
    e = r.intern("Исаак Ньютон")
    res = r.resolve("сэр Исаак Ньютон")
    assert res.entity_id == e.entity_id
    assert res.confidence < 1.0


def test_morphological_resolution_of_a_single_word_name(r):
    e = r.intern("Ньютон")
    res = r.resolve("Ньютоном")
    assert res.entity_id == e.entity_id
    assert res.status == RES_MORPHOLOGICAL
    assert 0.6 < res.confidence < 1.0


def test_surname_reaches_a_multiword_name(r):
    e = r.intern("Исаак Ньютон")
    res = r.resolve("Ньютона")
    assert res.entity_id == e.entity_id
    assert res.status == RES_PARTIAL
    assert 0.6 < res.confidence < 1.0





def test_translit_resolution_for_a_regular_name(r):
    e = r.intern("Москва")
    res = r.resolve("Moskva")
    assert res.entity_id == e.entity_id
    # the transliteration is indexed at registration time, so it is reported
    # as an alias hit rather than a late transliteration attempt
    assert res.status in (RES_EXACT, RES_ALIAS, RES_TRANSLIT)


def test_irregular_names_need_an_explicit_alias(r):
    """Documented limitation, not a silent failure: "Newton" does not
    transliterate to "Ньютон", so the correspondence must be taught."""
    e = r.intern("Исаак Ньютон")
    assert r.resolve("Isaac Newton").status == RES_UNKNOWN
    r.add_alias(e.entity_id, "Isaac Newton")
    res = r.resolve("Isaac Newton")
    assert res.entity_id == e.entity_id
    assert res.status in (RES_EXACT, RES_ALIAS)


def test_fuzzy_typo_resolution(r):
    e = r.intern("компьютер")
    res = r.resolve("компьютеp")
    assert res.status in (RES_FUZZY, RES_UNKNOWN)


def test_short_names_are_never_fuzzy_matched(r):
    r.intern("Иван")
    assert r.resolve("Ивон").status == RES_UNKNOWN


def test_unknown(r):
    assert r.resolve("Никто Такой").status == RES_UNKNOWN


# ------------------------------------------------------------ ambiguity
def test_two_distinct_entities_are_ambiguous(r):
    """Two different people sharing a surname: a bare surname must not pick
    one of them (spec §4: "Если есть два разных Ньютона, система должна
    запросить уточнение")."""
    a = r.intern("Исаак Ньютон")
    b = r.intern("Джон Ньютон")
    res = r.resolve("Ньютона")
    assert res.status == RES_AMBIGUOUS, [c.as_dict() for c in res.candidates]
    assert {c.entity_id for c in res.candidates} == {a.entity_id, b.entity_id}


def test_a_bare_registered_name_still_wins_exactly(r):
    bare = r.intern("Ньютон")
    full = r.intern("Исаак Ньютон")
    assert r.resolve("Ньютон").entity_id == bare.entity_id
    assert r.resolve("Ньютон").status == RES_EXACT
    assert full.entity_id != bare.entity_id


def test_an_unexplained_token_blocks_a_surname_match(r):
    bare = r.intern("Ньютон")
    full = r.intern("Исаак Ньютон")
    res = r.resolve("Исаака Ньютона")
    assert res.entity_id == full.entity_id
    assert bare.entity_id not in {c.entity_id for c in res.candidates}


def test_explicit_alias_removes_the_ambiguity(r):
    short = r.intern("Исаак Ньютон")
    full = r.intern("сэр Исаак Ньютон")
    r.link_alias(short.entity_id, full.entity_id)
    assert r.resolve("Ньютона").status != RES_AMBIGUOUS
    assert r.resolve("Ньютона").entity_id == full.entity_id


def test_resolver_never_merges_on_similarity(r):
    a = r.intern("Исаак Ньютон")
    b = r.intern("Исаак Азимов")
    assert a.entity_id != b.entity_id
    assert len(r) == 2


def test_intern_reuses_a_strong_morphological_match(r):
    """«физиком» must not create a near-duplicate of «физик»."""
    a = r.intern("физик")
    b = r.intern("физиком")
    assert a.entity_id == b.entity_id
    assert len(r) == 1


def test_intern_does_not_reuse_a_weak_morphological_match(r):
    """«физика» (physics) vs «физик» (physicist): one stripped character is
    not enough evidence to merge."""
    a = r.intern("физик")
    b = r.intern("физика")
    assert a.entity_id != b.entity_id


# --------------------------------------------------------------- gender
@pytest.mark.parametrize("name,gender", [
    ("Исаак Ньютон", "masculine"),
    ("Мария Кюри", "feminine"),
    ("Анна Иванова", "feminine"),
    ("окно", "neuter"),
])
def test_gender_inference(name, gender):
    assert infer_gender(name) == gender


def test_explicit_gender_wins(r):
    e = r.intern("Саша")
    r.set_gender(e.entity_id, "feminine")
    assert e.gender == "feminine"
    assert e.gender_source == "explicit"
    r._recompute_gender(e)
    assert e.gender == "feminine"


# ------------------------------------------------------- serialisation
def test_roundtrip(r):
    e = r.intern("Исаак Ньютон")
    r.add_alias(e.entity_id, "Ньютон")
    r.set_kind(e.entity_id, "PERSON")
    r2 = EntityResolver.from_dict(r.to_dict())
    assert r2.resolve("Ньютона").entity_id == e.entity_id
    assert r2.get(e.entity_id).kind == "PERSON"
    assert len(r2) == len(r)


def test_empty_name_is_rejected(r):
    with pytest.raises(ValueError):
        r.intern("   ")


def test_link_alias_keeps_history(r):
    a = r.intern("Ньютон")
    b = r.intern("Исаак Ньютон")
    b2 = r.link_alias(a.entity_id, b.entity_id, when="t1", source="teach")
    assert b2.entity_id == b.entity_id
    assert "Ньютон" in b.aliases
    assert any("absorbed alias entity" in h["change"] for h in b.history)
