"""Process-lifetime repository-root ownership contracts."""

from __future__ import annotations

import errno
import os
import pickle
from pathlib import Path
import subprocess
import sys
import threading

import pytest

from zlc_storage import RepositoryRootBusy, RepositoryRootLease


_LEASE_PROBE = """
import sys
from zlc_storage import RepositoryRootBusy, RepositoryRootLease
try:
    lease = RepositoryRootLease(sys.argv[1])
except RepositoryRootBusy:
    print('BUSY')
else:
    lease.close()
    print('ACQUIRED')
"""


def _probe_root(root):
    project_root = Path(__file__).resolve().parents[1]
    return subprocess.run(
        [sys.executable, "-c", _LEASE_PROBE, str(root)],
        cwd=str(project_root),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _assert_second_close_waits(close, entered, release):
    second_returned = threading.Event()
    errors: list[BaseException] = []

    def invoke(done=None):
        try:
            close()
        except BaseException as exc:
            errors.append(exc)
        finally:
            if done is not None:
                done.set()

    first = threading.Thread(target=invoke)
    second = threading.Thread(target=invoke, args=(second_returned,))
    first.start()
    assert entered.wait(2.0)
    second.start()
    assert not second_returned.wait(0.05)
    release.set()
    first.join(2.0)
    second.join(2.0)
    assert not first.is_alive() and not second.is_alive()
    assert errors == []


def test_repository_root_has_one_live_owner_and_can_reopen_after_close(tmp_path):
    root = tmp_path / "repository"
    lease = RepositoryRootLease(root)
    with pytest.raises(RepositoryRootBusy, match="live owner"):
        RepositoryRootLease(root)

    lease.close()
    lease.close()
    with RepositoryRootLease(root) as reopened:
        assert reopened.root == root.resolve()


def test_repository_close_is_rejected_without_entering_half_closed_state(
    tmp_path,
):
    lease = RepositoryRootLease(tmp_path / "repository")
    borrow = lease.borrow()

    with pytest.raises(RuntimeError, match="outstanding operations"):
        lease.close()
    assert lease.active
    assert borrow.active
    borrow.require_active()

    borrow.close()
    lease.close()
    assert not lease.active


def test_repository_lease_and_borrow_are_process_local(tmp_path):
    lease = RepositoryRootLease(tmp_path / "repository")
    borrow = lease.borrow()
    try:
        with pytest.raises(TypeError, match="process-local"):
            pickle.dumps(lease)
        with pytest.raises(TypeError, match="process-local"):
            pickle.dumps(borrow)
    finally:
        borrow.close()
        lease.close()


def test_repository_root_lock_is_exclusive_across_processes(tmp_path):
    root = tmp_path / "repository"
    lease = RepositoryRootLease(root)
    assert _probe_root(root) == "BUSY"

    lease.close()
    assert _probe_root(root) == "ACQUIRED"


@pytest.mark.skipif(not hasattr(os, "fork"), reason="POSIX fork contract")
def test_forked_lease_copy_cannot_unlock_parent_authority(tmp_path):
    root = tmp_path / "repository"
    lease = RepositoryRootLease(root)
    child = os.fork()
    if child == 0:
        try:
            lease.close()
        except RuntimeError:
            os._exit(0)
        os._exit(2)
    try:
        _pid, status = os.waitpid(child, 0)
        assert os.waitstatus_to_exitcode(status) == 0
        assert _probe_root(root) == "BUSY"
    finally:
        lease.close()


def test_repository_lock_io_failure_is_not_misreported_as_live_owner(
    tmp_path,
    monkeypatch,
):
    import zlc_storage.repository_lease as repository_lease

    root = tmp_path / "repository"
    real_acquire = repository_lease.acquire_file_lock

    def fail_io(_stream, *, blocking):
        assert blocking is False
        raise OSError(errno.EIO, "lock backend I/O failure")

    monkeypatch.setattr(repository_lease, "acquire_file_lock", fail_io)
    with pytest.raises(OSError, match="lock backend I/O failure"):
        RepositoryRootLease(root)

    monkeypatch.setattr(repository_lease, "acquire_file_lock", real_acquire)
    with RepositoryRootLease(root) as reopened:
        assert reopened.active


def test_concurrent_close_returns_only_after_os_ownership_is_released(
    tmp_path,
    monkeypatch,
):
    import zlc_storage.repository_lease as repository_lease

    root = tmp_path / "repository"
    lease = RepositoryRootLease(root)
    entered = threading.Event()
    release = threading.Event()
    real_unlock = repository_lease.release_file_lock

    def blocked_unlock(stream):
        entered.set()
        if not release.wait(2.0):
            raise TimeoutError("test did not release the OS unlock")
        real_unlock(stream)

    monkeypatch.setattr(repository_lease, "release_file_lock", blocked_unlock)
    _assert_second_close_waits(lease.close, entered, release)
    with RepositoryRootLease(root) as reopened:
        assert reopened.active


def test_concurrent_borrow_close_returns_only_after_count_is_released(
    tmp_path,
    monkeypatch,
):
    import zlc_storage.repository_lease as repository_lease

    lease = RepositoryRootLease(tmp_path / "repository")
    borrow = lease.borrow()
    entered = threading.Event()
    release = threading.Event()
    real_release = repository_lease.RepositoryRootLease._release_borrow

    def blocked_release(self, token):
        entered.set()
        if not release.wait(2.0):
            raise TimeoutError("test did not release the borrow close")
        return real_release(self, token)

    monkeypatch.setattr(
        repository_lease.RepositoryRootLease,
        "_release_borrow",
        blocked_release,
    )
    _assert_second_close_waits(borrow.close, entered, release)
    lease.close()
