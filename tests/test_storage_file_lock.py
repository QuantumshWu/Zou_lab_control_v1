"""Independent durability and error-classification checks for file locks."""

from __future__ import annotations

import errno

import pytest

from zlc_storage import DirectoryDurabilityError
from zlc_storage.file_lock import (
    acquire_file_lock,
    open_durable_lock_file,
    release_file_lock,
)


def test_lock_file_retry_repersists_existing_file_and_parent(tmp_path, monkeypatch):
    import zlc_storage.file_lock as file_lock

    path = tmp_path / "owner.lock"
    real_fsync = file_lock.os.fsync
    real_flush = file_lock.flush_directory
    fsync_calls = 0
    flush_calls = 0

    def counted_fsync(descriptor):
        nonlocal fsync_calls
        fsync_calls += 1
        return real_fsync(descriptor)

    def fail_first_flush(directory):
        nonlocal flush_calls
        flush_calls += 1
        if flush_calls == 1:
            raise DirectoryDurabilityError("parent acknowledgement failed")
        return real_flush(directory)

    monkeypatch.setattr(file_lock.os, "fsync", counted_fsync)
    monkeypatch.setattr(file_lock, "flush_directory", fail_first_flush)
    with pytest.raises(DirectoryDurabilityError, match="acknowledgement"):
        open_durable_lock_file(path)
    assert path.read_bytes() == b"\0"

    stream = open_durable_lock_file(path)
    try:
        acquire_file_lock(stream, blocking=False)
        release_file_lock(stream)
    finally:
        stream.close()
    assert fsync_calls == 2
    assert flush_calls == 2


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (OSError(errno.EACCES, "busy"), True),
        (OSError(errno.EAGAIN, "busy"), True),
        (OSError(errno.EDEADLK, "busy"), True),
        (OSError(errno.EIO, "I/O failure"), False),
        (OSError(errno.EBADF, "bad descriptor"), False),
    ],
)
def test_only_lock_contention_is_classified_as_busy(error, expected):
    import zlc_storage.file_lock as file_lock

    assert file_lock._is_contention(error) is expected
