"""Stable Definition identity and closed metadata contracts."""

from dataclasses import FrozenInstanceError

import pytest

from zlc_neutral_atom.catalog import (
    DefinitionKey,
    MeasurementDefinition,
    ProcessorDefinition,
    TaskDefinition,
    definition_key_from_tree,
    definition_key_to_tree,
)


def test_definition_key_has_one_exact_current_codec() -> None:
    key = DefinitionKey("tests.logic_nodes", "camera")
    tree = definition_key_to_tree(key)

    assert tree == {
        "schema": "zlc_neutral_atom.DefinitionKey",
        "owner_package": "tests.logic_nodes",
        "stable_definition_id": "camera",
    }
    assert definition_key_from_tree(tree) == key

    with pytest.raises(ValueError):
        definition_key_from_tree({**tree, "legacy_alias": "camera"})


def test_definition_records_are_closed_frozen_metadata() -> None:
    key = DefinitionKey("tests.logic_nodes", "probe")
    task = TaskDefinition(key, "Probe task", "tests.ProbeTaskRequest")
    measurement = MeasurementDefinition(
        key,
        "Probe measurement",
        "tests.ProbeMeasurementRequest",
        "tests.ProbeMeasurementBinding",
    )
    processor = ProcessorDefinition(
        key,
        "Probe processor",
        "tests.ProbeProcessorConfig",
    )

    assert task.key is measurement.key is processor.key
    assert not hasattr(measurement, "capture_spec_owner_fingerprint")
    with pytest.raises(FrozenInstanceError):
        task.title = "mutated"


@pytest.mark.parametrize(
    "factory",
    [
        lambda: TaskDefinition(object(), "Probe", "tests.Request"),
        lambda: ProcessorDefinition(
            DefinitionKey("tests", "processor"),
            lambda: None,
            "tests.Config",
        ),
        lambda: MeasurementDefinition(
            DefinitionKey("tests", "measurement"),
            "Probe",
            "tests.Request",
            "tests.Binding",
            "0" * 64,
        ),
    ],
)
def test_definition_records_reject_values_outside_their_closed_schema(factory) -> None:
    with pytest.raises((TypeError, ValueError)):
        factory()
