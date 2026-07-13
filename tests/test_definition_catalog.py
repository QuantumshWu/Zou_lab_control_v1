"""Explicit catalog composition has no discovery or override back door."""

from dataclasses import dataclass

import pytest

from zlc_neutral_atom.catalog import DefinitionCatalog, DefinitionKey


@dataclass(frozen=True)
class DemoDefinition:
    key: DefinitionKey
    label: str


def definition(name: str) -> DemoDefinition:
    return DemoDefinition(DefinitionKey("tests.demo", name), name)


def test_catalog_composes_explicit_tuples_without_mutation_or_discovery():
    first = definition("first")
    second = definition("second")
    catalog = DefinitionCatalog.compose((first,), (second,))

    assert catalog.definitions == (first, second)
    assert catalog.resolve(first.key, DemoDefinition) is first
    assert tuple(catalog) == (first, second)
    assert not hasattr(catalog, "register")
    assert not hasattr(catalog, "discover")
    with pytest.raises(TypeError):
        catalog.by_key[first.key] = second


def test_catalog_rejects_duplicate_keys():
    first = definition("capture")
    with pytest.raises(ValueError, match="duplicate DefinitionKey"):
        DefinitionCatalog((first, first))


def test_catalog_rejects_implicit_or_mutable_definitions():
    with pytest.raises(TypeError, match="explicit immutable tuple"):
        DefinitionCatalog([definition("list")])

    @dataclass
    class MutableDefinition:
        key: DefinitionKey

    with pytest.raises(TypeError, match="frozen dataclass"):
        DefinitionCatalog((MutableDefinition(DefinitionKey("tests", "mutable")),))

    with pytest.raises(TypeError, match="DefinitionKey"):
        DefinitionCatalog((object(),))


def test_catalog_rejects_callable_state_hidden_in_frozen_definition():
    @dataclass(frozen=True)
    class CallbackDefinition:
        key: DefinitionKey
        callback: object

    with pytest.raises(TypeError, match="declarative data"):
        DefinitionCatalog(
            (
                CallbackDefinition(
                    DefinitionKey("tests", "callback"),
                    lambda: None,
                ),
            )
        )


def test_catalog_resolution_is_typed_and_missing_is_loud():
    item = definition("capture")
    catalog = DefinitionCatalog((item,))
    with pytest.raises(TypeError, match="not str"):
        catalog.resolve(item.key, str)
    with pytest.raises(KeyError, match="absent"):
        catalog.resolve(DefinitionKey("tests.demo", "missing"))
