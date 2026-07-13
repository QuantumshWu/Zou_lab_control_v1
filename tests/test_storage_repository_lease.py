"""Process-lifetime repository-root ownership contracts."""

from __future__ import annotations

import pickle
from pathlib import Path
import subprocess
import sys

import pytest

from zlc_storage import RepositoryRootBusy, RepositoryRootLease


def test_repository_root_has_one_live_owner_and_can_reopen_after_close(tmp_path):
    root = tmp_path / "repository"
    lease = RepositoryRootLease(root, owner="first")
    with pytest.raises(RepositoryRootBusy, match="live owner"):
        RepositoryRootLease(root, owner="second")

    lease.close()
    lease.close()
    with RepositoryRootLease(root, owner="second") as reopened:
        assert reopened.root == root.resolve()


def test_repository_close_is_rejected_without_entering_half_closed_state(
    tmp_path,
):
    lease = RepositoryRootLease(tmp_path / "repository", owner="owner")
    borrow = lease.borrow()

    with pytest.raises(RuntimeError, match="outstanding commit authorities"):
        lease.close()
    assert lease.active
    assert borrow.active
    borrow.require_active()

    borrow.close()
    lease.close()
    assert not lease.active


def test_repository_lease_and_borrow_are_process_local(tmp_path):
    lease = RepositoryRootLease(tmp_path / "repository", owner="owner")
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
    lease = RepositoryRootLease(root, owner="parent")
    script = """
import sys
from zlc_storage import RepositoryRootBusy, RepositoryRootLease
try:
    lease = RepositoryRootLease(sys.argv[1], owner='child')
except RepositoryRootBusy:
    print('BUSY')
else:
    lease.close()
    print('ACQUIRED')
"""
    project_root = Path(__file__).resolve().parents[1]

    blocked = subprocess.run(
        [sys.executable, "-c", script, str(root)],
        cwd=str(project_root),
        check=True,
        capture_output=True,
        text=True,
    )
    assert blocked.stdout.strip() == "BUSY"

    lease.close()
    reopened = subprocess.run(
        [sys.executable, "-c", script, str(root)],
        cwd=str(project_root),
        check=True,
        capture_output=True,
        text=True,
    )
    assert reopened.stdout.strip() == "ACQUIRED"
