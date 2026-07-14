"""Filesystem hierarchy and directory-flush durability contracts."""

from __future__ import annotations

import os

import pytest

from zlc_storage import durable_mkdir, flush_directory


def test_durable_mkdir_flushes_one_child_before_its_existing_parent(
    tmp_path,
    monkeypatch,
):
    import zlc_storage.durability as durability

    observed = []
    monkeypatch.setattr(
        durability,
        "flush_directory",
        lambda directory: observed.append(directory.resolve()),
    )

    child = (tmp_path / "child").resolve()
    assert durable_mkdir(child) == child
    assert observed == [child, child.parent]


def test_durable_mkdir_retry_reacknowledges_visible_child_and_parent(
    tmp_path,
    monkeypatch,
):
    import zlc_storage.durability as durability

    target = (tmp_path / "repository").resolve()
    observed = []
    fail_parent_once = True

    def flush(directory):
        nonlocal fail_parent_once
        resolved = directory.resolve()
        observed.append(resolved)
        if resolved == target.parent and fail_parent_once:
            fail_parent_once = False
            raise durability.DirectoryDurabilityError("parent flush failed")

    monkeypatch.setattr(durability, "flush_directory", flush)
    with pytest.raises(durability.DirectoryDurabilityError, match="parent flush"):
        durable_mkdir(target)
    assert target.is_dir()
    assert observed == [target, target.parent]

    observed.clear()
    assert durable_mkdir(target) == target
    assert observed == [target, target.parent]


def test_durable_mkdir_rejects_a_missing_parent(tmp_path):
    target = tmp_path / "missing" / "child"
    with pytest.raises(FileNotFoundError, match="parent directory"):
        durable_mkdir(target)
    assert not target.parent.exists()


def test_directory_flush_is_real_on_the_current_platform(tmp_path):
    repository = durable_mkdir(tmp_path / "repository")
    target = durable_mkdir(repository / "nested")
    flush_directory(target)


def test_directory_flush_rejects_a_regular_file(tmp_path):
    path = tmp_path / "not-a-directory"
    path.write_bytes(b"payload")
    with pytest.raises(NotADirectoryError):
        flush_directory(path)


def test_durable_mkdir_rejects_a_regular_file_target(tmp_path):
    path = tmp_path / "not-a-directory"
    path.write_bytes(b"payload")
    with pytest.raises(NotADirectoryError):
        durable_mkdir(path)


@pytest.mark.skipif(os.name != "nt", reason="Windows directory-handle smoke test")
def test_windows_directory_handle_flush_smoke(tmp_path):
    import zlc_storage.durability as durability

    target = durable_mkdir(tmp_path / "windows-directory")
    durability._flush_windows_directory(target)
