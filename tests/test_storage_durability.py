"""Filesystem hierarchy and directory-flush durability contracts."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from zlc_storage import durable_mkdir, flush_directory
from zlc_storage.durability import (
    atomic_write_bytes,
    atomic_write_file,
    atomic_write_text,
)


def test_atomic_writers_replace_through_a_same_directory_temporary(
    tmp_path,
    monkeypatch,
):
    import zlc_storage.durability as durability

    target = tmp_path / "record.json"
    target.write_bytes(b"old")
    observed = []
    durability_events = []
    real_fsync = os.fsync
    real_replace = os.replace

    def fsync(descriptor):
        durability_events.append("file-fsync")
        real_fsync(descriptor)

    def replace(source, destination):
        durability_events.append("replace")
        source_path = Path(source)
        destination_path = Path(destination)
        observed.append(
            (
                source_path,
                destination_path,
                source_path.read_bytes(),
                target.read_bytes(),
            )
        )
        real_replace(source, destination)

    def flush(directory):
        durability_events.append("directory-flush")
        observed.append(Path(directory).resolve())

    monkeypatch.setattr(durability.os, "fsync", fsync)
    monkeypatch.setattr(durability.os, "replace", replace)
    monkeypatch.setattr(durability, "flush_directory", flush)

    assert atomic_write_text(target, "new") == target.resolve()
    source, destination, payload, previous = observed[0]
    assert source.parent == target.parent.resolve()
    assert source.name.startswith(f".{target.name}.")
    assert destination == target.resolve()
    assert payload == b"new"
    assert previous == b"old"
    assert observed[1] == target.parent.resolve()
    assert durability_events == ["file-fsync", "replace", "directory-flush"]
    assert target.read_text(encoding="utf-8") == "new"

    durability_events.clear()
    assert atomic_write_bytes(target, b"bytes") == target.resolve()
    _, _, payload, previous = observed[2]
    assert payload == b"bytes"
    assert previous == b"new"
    assert durability_events == ["file-fsync", "replace", "directory-flush"]
    assert target.read_bytes() == b"bytes"


def test_atomic_write_file_cleans_temporary_and_preserves_target_on_failure(
    tmp_path,
):
    target = tmp_path / "array.npy"
    target.write_bytes(b"old")

    def fail(stream):
        stream.write(b"partial")
        raise RuntimeError("writer failed")

    with pytest.raises(RuntimeError, match="writer failed"):
        atomic_write_file(target, fail)

    assert target.read_bytes() == b"old"
    assert not tuple(tmp_path.glob(f".{target.name}.*.tmp"))


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
