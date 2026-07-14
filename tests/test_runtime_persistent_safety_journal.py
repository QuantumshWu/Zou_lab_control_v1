"""Restart and recovery contracts for installation-level safety facts."""

from __future__ import annotations

import os
import threading

import pytest

from zlc_neutral_atom.runtime import (
    HazardAppendStatus,
    HazardClaim,
    HazardRecord,
    PersistentSafetyJournal,
    RecoveryBundle,
    RecoveryClaim,
    RecoveryEvidence,
    ResourceArbiter,
    ResourceClaim,
    ResourceKey,
    ResourceLease,
    ResourceQuarantined,
    SafetyAuthorityBusy,
    SafetyDispositionBundle,
    SafetyDispositionRecord,
    SafetyDecision,
    SafetyOutcome,
)


def _assert_second_call_waits(call, entered, release):
    second_returned = threading.Event()
    failures = []

    def invoke(done=None):
        try:
            call()
        except BaseException as exc:
            failures.append(exc)
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
    assert failures == []


def test_hazard_survives_restart_and_blocks_ordinary_acquisition(tmp_path):
    path = tmp_path / "installation-safety.zlcj"
    key = ResourceKey.parse("device/camera/qcmos")
    journal = PersistentSafetyJournal(path)
    journal.append_hazards(
        (
            HazardRecord(
                record_id="hazard-one",
                key=key,
                stable_device_identity=str(key),
                connection_generation="generation-one",
                run_id="run-one",
                activated_at=1.0,
            ),
        )
    )
    journal.close()

    restarted = PersistentSafetyJournal(path)
    assert tuple(record.record_id for record in restarted.snapshot().unresolved_hazards) == (
        "hazard-one",
    )
    outcome = ResourceArbiter(restarted).acquire_all(
        "another-run",
        (ResourceClaim(key),),
    )
    assert isinstance(outcome, ResourceQuarantined)


def test_quarantine_and_verified_recovery_facts_round_trip(tmp_path):
    path = tmp_path / "installation-safety.zlcj"
    key = ResourceKey.parse("device/fpga/pulse")
    journal = PersistentSafetyJournal(path)
    hazard = HazardRecord(
        record_id="hazard-two",
        key=key,
        stable_device_identity=str(key),
        connection_generation="generation-two",
        run_id="run-two",
        activated_at=2.0,
    )
    journal.append_hazards((hazard,))
    journal.append_safety_bundle(
        SafetyDispositionBundle(
            bundle_id="safety-two",
            run_id="run-two",
            records=(
                SafetyDispositionRecord(
                    disposition_id="quarantine-two",
                    key=key,
                    outcome=SafetyOutcome.UNSAFE,
                    hazard_record_id="hazard-two",
                    stable_device_identity=str(key),
                    connection_generation="generation-two",
                    safe_receipt=None,
                    reason="safe state could not be verified",
                    recovery_action="reconnect and verify hardware safe state",
                ),
            ),
            recorded_at=3.0,
        )
    )

    quarantined = journal.snapshot()
    assert quarantined.unresolved_hazards == ()
    assert tuple(record.record_id for record in quarantined.unresolved_quarantines) == (
        "quarantine-two",
    )
    with pytest.raises(ValueError, match="safety records"):
        journal.append_hazards(
            (
                HazardRecord(
                    record_id="hazard-while-quarantined",
                    key=key,
                    stable_device_identity=str(key),
                    connection_generation="generation-three",
                    run_id="run-three",
                    activated_at=3.5,
                ),
            )
        )

    journal.append_recovery_bundle(
        RecoveryBundle(
            bundle_id="recovery-two",
            claim=RecoveryClaim(
                key=key,
                stable_device_identity=str(key),
                quarantine_record_ids=("quarantine-two",),
                hazard_record_ids=(),
            ),
            evidence=RecoveryEvidence(
                stable_device_identity=str(key),
                connection_generation="generation-three",
                health_digest="healthy",
                safe_state_digest="safe",
                verified_at=4.0,
            ),
            recorded_at=4.0,
        )
    )
    assert journal.snapshot().unresolved_quarantines == ()
    assert journal.append_hazards((hazard,)) is HazardAppendStatus.ALREADY_RESOLVED
    assert journal.snapshot().unresolved_hazards == ()


def test_lost_ack_retry_is_idempotent_for_hazard_batch(tmp_path):
    path = tmp_path / "installation-safety.zlcj"
    key = ResourceKey.parse("device/camera/retry")
    record = HazardRecord(
        record_id="hazard-retry",
        key=key,
        stable_device_identity=str(key),
        connection_generation="generation-retry",
        run_id="run-retry",
        activated_at=5.0,
    )
    journal = PersistentSafetyJournal(path)
    journal.append_hazards((record,))
    journal.append_hazards((record,))
    assert journal.snapshot().unresolved_hazards == (record,)


def test_second_installation_safety_authority_is_rejected(tmp_path):
    path = tmp_path / "installation-safety.zlcj"
    first = PersistentSafetyJournal(path)
    with pytest.raises(SafetyAuthorityBusy):
        PersistentSafetyJournal(path)
    first.close()
    with pytest.raises(RuntimeError, match="authority is closed"):
        first.snapshot()
    assert isinstance(PersistentSafetyJournal(path), PersistentSafetyJournal)


def test_concurrent_safety_close_waits_for_actual_lock_release(
    tmp_path,
    monkeypatch,
):
    import zlc_neutral_atom.runtime.safety_journal as safety_journal

    path = tmp_path / "installation-safety.zlcj"
    journal = PersistentSafetyJournal(path)
    real_unlock = safety_journal.release_file_lock
    entered = threading.Event()
    release = threading.Event()

    def blocked_unlock(stream):
        entered.set()
        if not release.wait(2.0):
            raise TimeoutError("test did not release safety unlock")
        real_unlock(stream)

    monkeypatch.setattr(safety_journal, "release_file_lock", blocked_unlock)
    _assert_second_call_waits(journal.close, entered, release)
    PersistentSafetyJournal(path).close()


def test_concurrent_arbiter_shutdown_waits_for_owner_release(tmp_path, monkeypatch):
    import zlc_neutral_atom.runtime.safety_journal as safety_journal

    path = tmp_path / "installation-safety.zlcj"
    arbiter = ResourceArbiter(PersistentSafetyJournal(path))
    entered = threading.Event()
    release = threading.Event()
    real_unlock = safety_journal.release_file_lock

    def blocked_unlock(stream):
        entered.set()
        if not release.wait(2.0):
            raise TimeoutError("test did not release arbiter shutdown")
        real_unlock(stream)

    monkeypatch.setattr(safety_journal, "release_file_lock", blocked_unlock)
    _assert_second_call_waits(arbiter.shutdown, entered, release)
    PersistentSafetyJournal(path).close()


def test_safety_authority_cannot_bind_across_owner_close(tmp_path, monkeypatch):
    import zlc_neutral_atom.runtime.safety_journal as safety_journal

    path = tmp_path / "installation-safety.zlcj"
    journal = PersistentSafetyJournal(path)
    unlock_entered = threading.Event()
    release_unlock = threading.Event()
    bind_returned = threading.Event()
    close_failures = []
    bind_failures = []
    real_unlock = safety_journal.release_file_lock

    def blocked_unlock(stream):
        unlock_entered.set()
        if not release_unlock.wait(2.0):
            raise TimeoutError("test did not release safety owner close")
        real_unlock(stream)

    def close():
        try:
            journal.close()
        except BaseException as exc:
            close_failures.append(exc)

    def bind():
        try:
            ResourceArbiter(journal)
        except BaseException as exc:
            bind_failures.append(exc)
        finally:
            bind_returned.set()

    monkeypatch.setattr(safety_journal, "release_file_lock", blocked_unlock)
    close_thread = threading.Thread(target=close)
    bind_thread = threading.Thread(target=bind)
    close_thread.start()
    assert unlock_entered.wait(2.0)
    bind_thread.start()
    assert not bind_returned.wait(0.05)
    release_unlock.set()
    close_thread.join(2.0)
    bind_thread.join(2.0)

    assert not close_thread.is_alive() and not bind_thread.is_alive()
    assert close_failures == []
    assert len(bind_failures) == 1
    assert isinstance(bind_failures[0], RuntimeError)
    assert "closed" in str(bind_failures[0])
    PersistentSafetyJournal(path).close()


def test_unlock_failure_still_closes_old_safety_authority(tmp_path, monkeypatch):
    import zlc_neutral_atom.runtime.safety_journal as safety_journal

    path = tmp_path / "installation-safety.zlcj"
    journal = PersistentSafetyJournal(path)
    real_unlock = safety_journal.release_file_lock

    def fail_unlock(_stream):
        raise OSError("injected safety unlock failure")

    monkeypatch.setattr(safety_journal, "release_file_lock", fail_unlock)
    with pytest.raises(OSError, match="unlock failure"):
        journal.close()
    with pytest.raises(RuntimeError, match="authority is closed"):
        journal.snapshot()

    monkeypatch.setattr(safety_journal, "release_file_lock", real_unlock)
    PersistentSafetyJournal(path).close()


@pytest.mark.skipif(not hasattr(os, "fork"), reason="POSIX fork contract")
def test_forked_safety_copy_cannot_unlock_parent_authority(tmp_path):
    path = tmp_path / "installation-safety.zlcj"
    journal = PersistentSafetyJournal(path)
    child = os.fork()
    if child == 0:
        try:
            journal.close()
        except RuntimeError:
            os._exit(0)
        os._exit(2)
    try:
        _pid, status = os.waitpid(child, 0)
        assert os.waitstatus_to_exitcode(status) == 0
        with pytest.raises(SafetyAuthorityBusy):
            PersistentSafetyJournal(path)
    finally:
        journal.close()


def test_active_resource_authority_cannot_release_installation_owner_lock(tmp_path):
    path = tmp_path / "installation-safety.zlcj"
    key = ResourceKey.parse("device/fpga/live-owner")
    journal = PersistentSafetyJournal(path)
    arbiter = ResourceArbiter(journal)
    lease = arbiter.acquire_all("live-run", (ResourceClaim(key),))
    assert isinstance(lease, ResourceLease)
    lease.activate_hazards(
        (
            HazardClaim(key, str(key), "live-generation"),
        )
    )
    with pytest.raises(RuntimeError, match="owned by ResourceArbiter"):
        journal.close()
    with pytest.raises(RuntimeError, match="active ownership"):
        arbiter.shutdown()
    with pytest.raises(SafetyAuthorityBusy):
        PersistentSafetyJournal(path)

    lease._commit_safety(
        (
            SafetyDecision.unsafe(
                key,
                reason="safe state unknown",
                recovery_action="verify hardware",
            ),
        )
    )
    lease.release_after_safety(disposition="QUARANTINED")
    arbiter.shutdown()
    restarted = PersistentSafetyJournal(path)
    restarted.close()
