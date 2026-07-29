"""Current explicit WorkspacePaths and stateless path-resolution contract."""

from __future__ import annotations

from pathlib import Path
import sys
from types import ModuleType

import pytest

from Zou_lab_control.api import WorkspacePaths
from Zou_lab_control.workbench import open_pulse_editor
from zlc_storage.paths import resolve_under


def test_workspace_paths_are_composed_from_explicit_absolute_roots(tmp_path) -> None:
    authored = (tmp_path / "authored").resolve()
    repository = (tmp_path / "repository").resolve()

    paths = WorkspacePaths.for_workspace(
        authored,
        repository_root=repository,
    )

    assert paths.pulses_root == authored / "pulses"
    assert paths.tasks_root == authored / "tasks"
    assert paths.output_root == repository / "output"
    assert paths.repository_root == repository


def test_workspace_paths_reject_relative_authorities() -> None:
    with pytest.raises(ValueError, match="absolute"):
        WorkspacePaths.for_workspace(
            Path("relative"),
            repository_root=Path("repository"),
        )


def test_resolve_under_never_uses_process_cwd(tmp_path, monkeypatch) -> None:
    root = (tmp_path / "root").resolve()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    assert resolve_under(root, "nested/value.json") == (
        root / "nested" / "value.json"
    ).resolve()
    absolute = (tmp_path / "external.json").resolve()
    assert resolve_under(root, absolute) == absolute


def test_storage_path_module_exposes_no_workspace_policy() -> None:
    import zlc_storage.paths as paths

    for retired in (
        "PROJECT_ROOT",
        "project_path",
        "user_output_path",
        "resolve_under_project",
        "display_path",
    ):
        assert not hasattr(paths, retired)


def test_public_workbench_composition_forwards_workspace_roots(
    tmp_path,
    monkeypatch,
) -> None:
    workspace = WorkspacePaths.for_workspace(
        (tmp_path / "authored").resolve(),
        repository_root=(tmp_path / "repository").resolve(),
    )
    observed = {}
    fake_app = ModuleType("zlc_workbench.pulse_editor.app")

    def open_from_composition(**kwargs):
        observed.update(kwargs)
        return "opened"

    fake_app.open_pulse_editor = open_from_composition
    monkeypatch.setitem(sys.modules, fake_app.__name__, fake_app)

    assert open_pulse_editor(workspace=workspace) == "opened"
    assert observed["pulses_root"] == workspace.pulses_root
    assert observed["output_root"] == workspace.output_root
    assert observed["initial_connection_mode"] == "offline"
    assert callable(observed["connection_factory"])
