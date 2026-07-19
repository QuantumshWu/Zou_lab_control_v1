"""A newly registered Definition reaches the TaskConsole without a GUI edit.

`main` derives its Add-Panel entries straight from the measurement/processor/task
spec sequences, so registering a new spec makes it appear with no console change.
The projection here previously pinned the exact key set of the three current
capabilities and raised on anything else, which turned "one more definition" into
a hard failure instead of one more catalog row.
"""

from __future__ import annotations

import pytest

from zlc_neutral_atom.catalog import (
    DefinitionCatalog,
    DefinitionKey,
    MeasurementDefinition,
    StreamProcessorDefinition,
    TaskDefinition,
)
from zlc_workbench.task_console import (
    compose_task_console_catalog,
    task_console_catalog_items,
)


def _extra_task() -> TaskDefinition:
    return TaskDefinition(
        key=DefinitionKey("zlc_neutral_atom", "contract_probe_task"),
        title="Contract probe",
        request_schema_id="zlc_neutral_atom.ContractProbeRequest",
    )


def test_the_three_current_capabilities_still_project_in_catalog_order():
    items = task_console_catalog_items(compose_task_console_catalog())
    assert [item.group for item in items] == ["Task", "Measurement", "Processor"]
    assert all(item.title for item in items)


def test_a_new_definition_appears_without_touching_the_projection():
    catalog = compose_task_console_catalog()
    grown = DefinitionCatalog(catalog.definitions + (_extra_task(),))

    items = task_console_catalog_items(grown)

    assert len(items) == len(catalog.definitions) + 1
    added = items[-1]
    assert added.group == "Task"
    assert added.title == "Contract probe"
    # Order follows the catalog, so existing rows keep their position.
    assert items[: len(catalog.definitions)] == task_console_catalog_items(catalog)


def test_each_definition_kind_decides_its_own_group():
    catalog = DefinitionCatalog(
        (
            StreamProcessorDefinition(
                key=DefinitionKey("zlc_neutral_atom", "contract_probe_processor"),
                title="Probe processor",
                config_schema_id="zlc_neutral_atom.ProbeConfig",
            ),
            MeasurementDefinition(
                key=DefinitionKey("zlc_neutral_atom", "contract_probe_measurement"),
                title="Probe measurement",
                request_schema_id="zlc_neutral_atom.ProbeRequest",
                binding_schema_id="zlc_neutral_atom.ProbeBinding",
                capture_spec_owner_fingerprint="0" * 64,
            ),
        )
    )

    items = task_console_catalog_items(catalog)

    assert [item.group for item in items] == ["Processor", "Measurement"]


def test_a_definition_kind_this_product_cannot_group_fails_closed():
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class UnknownDefinition:
        key: DefinitionKey
        title: str

    catalog = DefinitionCatalog(
        (
            UnknownDefinition(
                key=DefinitionKey("zlc_neutral_atom", "contract_probe_unknown"),
                title="Unknown",
            ),
        )
    )

    with pytest.raises(TypeError, match="cannot group definition"):
        task_console_catalog_items(catalog)


def test_projection_rejects_a_non_catalog_argument():
    with pytest.raises(TypeError, match="catalog must be DefinitionCatalog"):
        task_console_catalog_items(object())
