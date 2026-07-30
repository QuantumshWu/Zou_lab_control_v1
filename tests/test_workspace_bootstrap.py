"""Current durable-directory and single-root workspace bootstrap contracts."""

from __future__ import annotations

from pathlib import Path

import pytest

from zlc_storage import durable_makedirs, durable_mkdir


def test_creates_every_missing_level(tmp_path: Path):
    target = tmp_path / "home" / ".zlc" / "task_console"

    created = durable_makedirs(target)

    assert created == target.resolve()
    assert target.is_dir()
    assert (tmp_path / "home" / ".zlc").is_dir()


def test_is_idempotent_and_re_acknowledges_an_existing_path(tmp_path: Path):
    target = tmp_path / "workspace"
    durable_makedirs(target)
    marker = target / "keep.txt"
    marker.write_text("keep", encoding="utf-8")

    again = durable_makedirs(target)

    assert again == target.resolve()
    # Re-running must flush the existing entry, never replace what it holds.
    assert marker.read_text(encoding="utf-8") == "keep"


def test_single_level_mkdir_still_refuses_to_guess_ancestors(tmp_path: Path):
    missing_parent = tmp_path / "absent" / "child"

    with pytest.raises(FileNotFoundError, match="parent directory does not exist"):
        durable_mkdir(missing_parent)


def test_a_file_in_the_way_fails_closed(tmp_path: Path):
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory", encoding="utf-8")

    with pytest.raises(NotADirectoryError):
        durable_makedirs(blocker / "below")


def test_connect_prepares_owned_output_roots_before_composition(tmp_path: Path):
    """Composition creates output roots without inventing authored content."""

    from Zou_lab_control.api import WorkspacePaths, connect

    project = (tmp_path / "project").resolve()
    workspace = WorkspacePaths.for_workspace(project)

    with connect("virtual", workspace=workspace):
        assert workspace.output_root.is_dir()
        assert (workspace.output_root / "captures").is_dir()
        assert (workspace.output_root / "calibrations").is_dir()
        assert not workspace.pulses_root.exists()
        assert not workspace.tasks_root.exists()
