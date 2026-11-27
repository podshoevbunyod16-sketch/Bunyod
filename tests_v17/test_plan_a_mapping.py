"""Plan A — the mapping table is the integration contract; test it as data."""
from __future__ import annotations

import pytest

from rasm_cim_v16.relations import DEFAULT_SCHEMA, Relation

from rasm_cim_v17.interop.mapping import (ENTITY_KIND_MAP, KIND_POLICY,
                                          PREDICATE_MAP, PredicateMapping,
                                          boolean_value, custom_relations,
                                          map_entity_kind, map_predicate,
                                          unmapped_predicates)


def test_every_custom_definition_is_a_valid_relation():
    for definition in custom_relations():
        rel = Relation.from_dict(definition)          # raises on a bad field
        assert rel.name == definition["name"]
        assert rel.human, f"{rel.name}: relation without a Russian gloss"
        assert "{s}" in rel.template or "{o}" in rel.template or rel.template, \
            f"{rel.name}: template must render the arguments"


def test_custom_relation_names_do_not_shadow_builtins():
    for definition in custom_relations():
        assert definition["name"] not in DEFAULT_SCHEMA, (
            f"{definition['name']} is already a v0.16 relation — "
            f"registering it again would overwrite built-in semantics")


def test_custom_definitions_are_unique():
    names = [d["name"] for d in custom_relations()]
    assert len(names) == len(set(names))


def test_every_mapping_target_is_a_relation_that_exists_or_is_created():
    known = set(DEFAULT_SCHEMA.names()) | {d["name"] for d in custom_relations()}
    for legacy, mapping in PREDICATE_MAP.items():
        assert mapping.target in known, f"{legacy} -> unknown {mapping.target}"
        assert mapping.legacy == legacy


def test_inverse_relations_point_at_mapped_relations():
    known = set(DEFAULT_SCHEMA.names()) | {d["name"] for d in custom_relations()}
    for definition in custom_relations():
        inverse = definition.get("inverse", "")
        if inverse:
            assert inverse in known, (
                f"{definition['name']}.inverse -> {inverse} does not exist")


def test_unknown_predicate_is_not_mapped_and_is_counted():
    assert map_predicate("совершенно_новый_предикат") is None
    assert unmapped_predicates(["capital_of", "нэв", "нэв", "is_a"]) == {"нэв": 2}


def test_builtin_names_pass_through_without_a_table_entry():
    mapping = map_predicate("located_in")
    assert mapping is not None and mapping.target == "located_in"
    passthrough = map_predicate("known_for")
    assert passthrough is not None and passthrough.target == "known_for"
    assert "already a v0.16 relation" in passthrough.note


def test_mapping_is_case_insensitive_but_keeps_the_original():
    m = map_predicate("CAPITAL_OF")
    assert m is not None and m.target == "capital_of"


def test_legacy_is_a_maps_to_instance_of():
    m = map_predicate("is_a")
    assert m is not None and m.target == "instance_of"


def test_boolean_properties_are_normalised_not_edges():
    for legacy in ("может_летать", "существует"):
        m = map_predicate(legacy)
        assert m is not None and m.boolean_object
        assert m.target not in DEFAULT_SCHEMA or m.target in ("has_property",)


def test_boolean_value_parsing():
    assert boolean_value("да") is True
    assert boolean_value("НЕТ") is False
    assert boolean_value("100 цельсия") is None


def test_entity_kind_mapping_is_total_and_reported():
    for legacy in ENTITY_KIND_MAP:
        kind, mapped = map_entity_kind(legacy)
        assert kind and mapped
    kind, mapped = map_entity_kind("инопланетянин")
    assert kind == "ANY" and mapped is False


def test_hypothesis_policy_is_explicit():
    assert KIND_POLICY["hypothesis"] == "skip"
    assert KIND_POLICY["user_assertion"] == KIND_POLICY["extracted_fact"] == "import"


def test_measurement_relations_mark_their_object_as_literal():
    for legacy in ("кипит_при", "population_of", "height_of"):
        assert map_predicate(legacy).object_is_literal is True


def test_functional_legacy_relations_are_declared_functional():
    functional = {d["name"] for d in custom_relations() if d.get("functional")}
    assert {"capital_of", "population_of", "height_of", "boils_at",
            "director_of", "ceo_of", "governor_of"} <= functional


def test_templates_do_not_leak_python_format_placeholders():
    for definition in custom_relations():
        template = definition.get("template", "")
        if not template:
            continue
        rendered = template.format(s="A", o="B")
        assert "{" not in rendered and "}" not in rendered
