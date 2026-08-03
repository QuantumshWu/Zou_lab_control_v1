"""Current TaskConsole boundary: descriptors in, one host factory out."""

from __future__ import annotations

from pathlib import Path

from Zou_lab_control.api import WorkspacePaths, connect
from Zou_lab_control.workbench._composition import task_console_dependencies
from zlc_neutral_atom.installation import DeviceCatalogView
from zlc_neutral_atom.logic_node import LogicNodeDescriptor
from zlc_neutral_atom.processing.signal_plane import SignalDataPlane
from zlc_workbench.task_console.logic_node_parameter_panel import (
    _device_choice_projection,
)


def _workspace(root: Path) -> WorkspacePaths:
    return WorkspacePaths.for_workspace(root.resolve())


def test_composition_exposes_only_current_task_console_facts(tmp_path: Path) -> None:
    with connect("virtual", workspace=_workspace(tmp_path / "project")) as exp:
        dependencies = task_console_dependencies(exp)
        assert set(dependencies) == {
            "descriptors",
            "device_catalog",
            "host_factory",
            "data_plane",
            "project_root",
            "pulses_root",
            "tasks_root",
            "figures_root",
        }
        assert dependencies["descriptors"] == exp.nodes.descriptors
        assert all(
            isinstance(value, LogicNodeDescriptor)
            for value in dependencies["descriptors"]
        )
        assert dependencies["device_catalog"] is exp.device_catalog
        assert isinstance(dependencies["device_catalog"], DeviceCatalogView)
        assert isinstance(dependencies["data_plane"], SignalDataPlane)
        assert callable(dependencies["host_factory"])

        catalog = dependencies["device_catalog"]
        for descriptor in dependencies["descriptors"]:
            for projection in _device_choice_projection(
                descriptor,
                catalog,
            ).values():
                values = tuple(choice.value for choice in projection.choices)
                assert values
                assert projection.default in values
                assert all(value in catalog for value in values)


def test_removed_task_console_owners_do_not_exist() -> None:
    root = Path(__file__).resolve().parents[1] / "zlc_workbench" / "task_console"
    for name in (
        "application_ports.py",
        "capability.py",
        "attachment_builders.py",
        "declaration_projection.py",
        "catalog_bridge.py",
        "artifact_resolution.py",
        "input_binding.py",
    ):
        assert not (root / name).exists()
