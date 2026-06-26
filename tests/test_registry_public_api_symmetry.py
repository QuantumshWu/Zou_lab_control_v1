"""Contract: the measurement / processor / task open-registries expose a SYMMETRIC
public API through both `Zou_lab_control.neutral_atom` and `.operations`.

The three catalogs are the same pattern (one `OpenRegistry` each, #H3q-4c); a notebook
must be able to `register_*` / `unregister_*` / list any of the three the same way.
This guard fails if a future tier (or a re-export edit) drops one noun's names -- e.g.
`register_processor` was missing from both packages while `register_measurement` was
present (#H3r-F1), so the documented notebook path raised ImportError.
"""

from __future__ import annotations

import importlib

import pytest

NOUNS = ("measurement", "processor", "task")
MODULES = ("Zou_lab_control.neutral_atom", "Zou_lab_control.neutral_atom.operations")


def _expected_names(noun: str) -> set[str]:
    # the five public names every catalog binds off its OpenRegistry instance
    return {f"register_{noun}", noun, f"unregister_{noun}", f"registered_{noun}s"}


@pytest.mark.parametrize("module_path", MODULES)
@pytest.mark.parametrize("noun", NOUNS)
def test_catalog_public_names_present_and_callable(module_path, noun):
    mod = importlib.import_module(module_path)
    for name in _expected_names(noun):
        assert hasattr(mod, name), f"{module_path} is missing {name!r}"
        assert callable(getattr(mod, name)), f"{module_path}.{name} is not callable"


@pytest.mark.parametrize("module_path", MODULES)
def test_all_three_catalogs_expose_the_same_name_shape(module_path):
    mod = importlib.import_module(module_path)
    shapes = {noun: {n for n in _expected_names(noun) if hasattr(mod, n)} for noun in NOUNS}
    # every noun exposes EXACTLY its five names -- no asymmetry between the tiers
    for noun in NOUNS:
        assert shapes[noun] == _expected_names(noun), f"{module_path}: {noun} name set drifted: {shapes[noun]}"


def test_discovered_specs_entrypoint_per_catalog():
    ops = importlib.import_module("Zou_lab_control.neutral_atom.operations")
    for fn in ("discovered_specs", "discovered_processor_specs", "discovered_task_specs"):
        assert callable(getattr(ops, fn)), f"operations.{fn} missing"


def test_spec_dataclasses_exported_per_catalog():
    ops = importlib.import_module("Zou_lab_control.neutral_atom.operations")
    for spec in ("MeasurementSpec", "ProcessorSpec", "TaskSpec"):
        assert hasattr(ops, spec), f"operations.{spec} missing"
