"""A first run on a machine with no workspace root still starts.

The composition root creates exactly one repository below an existing parent and
refuses to guess missing ancestors, which is why the launchers - whose default
repository is two levels below the home directory - must own the levels above
it.  Without that, connecting on a machine that has never held a `~/.zlc` died
with FileNotFoundError before any window appeared.
"""

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


def test_both_launchers_prepare_the_workspace_before_connecting():
    """Neither launcher may hand the composition root a missing ancestor."""

    root = Path(__file__).resolve().parents[1]
    for name in ("task_console.py", "pulse_gui.py"):
        source = (root / name).read_text(encoding="utf-8")
        assert "durable_makedirs(" in source, name
        prepare = source.index("durable_makedirs(")
        connect_call = source.index("experiment = connect(")
        assert prepare < connect_call, name
