"""Current explicit WorkspacePaths and stateless path-resolution contract."""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path
import sys
from types import ModuleType

import pytest

from Zou_lab_control.api import WorkspacePaths
from Zou_lab_control.workbench import open_pulse_editor
from zlc_storage.paths import resolve_under


def test_workspace_paths_are_composed_from_one_explicit_project_root(tmp_path) -> None:
    project = (tmp_path / "project").resolve()

    paths = WorkspacePaths.for_workspace(project)

    assert tuple(field.name for field in fields(paths)) == (
        "project_root",
        "pulses_root",
        "tasks_root",
        "runs_root",
        "figures_root",
    )
    assert paths.project_root == project
    assert paths.pulses_root == project / "pulses"
    assert paths.tasks_root == project / "tasks"
    assert paths.runs_root == project / "runs"
    assert paths.figures_root == project / "figures"


def test_workspace_paths_reject_relative_authorities() -> None:
    with pytest.raises(ValueError, match="absolute"):
        WorkspacePaths.for_workspace(Path("relative"))


def test_resolve_under_never_uses_process_cwd(tmp_path, monkeypatch) -> None:
    root = (tmp_path / "root").resolve()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    assert resolve_under(root, "nested/value.json") == (
        root / "nested" / "value.json"
    ).resolve()
    with pytest.raises(ValueError, match="relative"):
        resolve_under(root, (tmp_path / "external.json").resolve())
    with pytest.raises(ValueError, match="escapes"):
        resolve_under(root, "../external.json")


def test_resolve_under_rejects_relative_roots() -> None:
    with pytest.raises(ValueError, match="root must be absolute"):
        resolve_under(Path("relative"), "value.json")


def test_public_workbench_composition_forwards_workspace_roots(
    tmp_path,
    monkeypatch,
) -> None:
    workspace = WorkspacePaths.for_workspace((tmp_path / "project").resolve())
    observed = {}
    fake_app = ModuleType("zlc_workbench.pulse_editor.app")

    def open_from_composition(**kwargs):
        observed.update(kwargs)
        return "opened"

    fake_app.open_pulse_editor = open_from_composition
    monkeypatch.setitem(sys.modules, fake_app.__name__, fake_app)

    assert open_pulse_editor(workspace=workspace) == "opened"
    assert observed["pulses_root"] == workspace.pulses_root
    assert observed["output_root"] == workspace.project_root
    assert observed["initial_connection_mode"] == "offline"
    assert callable(observed["connection_factory"])
