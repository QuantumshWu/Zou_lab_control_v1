"""Filesystem hierarchy and directory-flush durability contracts."""

from __future__ import annotations

import os

import pytest

from zlc_storage import durable_mkdir, flush_directory


def test_durable_mkdir_flushes_each_new_child_before_its_parent(
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

    first = (tmp_path / "first").resolve()
    second = (first / "second").resolve()
    assert durable_mkdir(second) == second
    assert observed == [first, first.parent, second, second.parent]


def test_directory_flush_is_real_on_the_current_platform(tmp_path):
    target = durable_mkdir(tmp_path / "repository" / "nested")
    flush_directory(target)


def test_directory_flush_rejects_a_regular_file(tmp_path):
    path = tmp_path / "not-a-directory"
    path.write_bytes(b"payload")
    with pytest.raises(NotADirectoryError):
        flush_directory(path)


@pytest.mark.skipif(os.name != "nt", reason="Windows directory-handle smoke test")
def test_windows_directory_handle_flush_smoke(tmp_path):
    import zlc_storage.durability as durability

    target = durable_mkdir(tmp_path / "windows-directory")
    durability._flush_windows_directory(target)
