"""Durability and single-owner contracts for the installation safety journal."""

from __future__ import annotations

import os
import threading

import pytest

from zlc_neutral_atom.runtime.resources import (
    DeviceBindingStamp,
    DeviceIdentityEvidenceKind,
    HazardAppendStatus,
    HazardClaim,
    HazardRecord,
    PhysicalDeviceIdentity,
    RecoveryBundle,
    RecoveryClaim,
    RecoveryEvidence,
    ResourceArbiter,
    ResourceClaim,
    ResourceKey,
    ResourceLease,
    ResourceQuarantined,
    SafeReceipt,
    SafetyDecision,
    SafetyDispositionBundle,
    SafetyDispositionRecord,
    SafetyJournalSnapshot,
    SafetyOutcome,
)
from zlc_neutral_atom.runtime.safety_journal import (
    PersistentSafetyJournal,
    SafetyAuthorityBusy,
)
from zlc_storage.framed_journal import FramedJournal


def physical() -> PhysicalDeviceIdentity:
    return PhysicalDeviceIdentity(
        stable_device_identity="physical-device",
        evidence_kind=DeviceIdentityEvidenceKind.HARDWARE_IDENTITY_READBACK,
        evidence_digest="identity-readback",
        asset_map_revision="assets-v1",
    )


def stamp(generation: str = "generation") -> DeviceBindingStamp:
    return DeviceBindingStamp(physical(), generation)


def hazard(
    resource: ResourceKey,
    *,
    record_id: str = "hazard",
    run_id: str = "run",
    generation: str = "generation",
    activated_at: float = 1.0,
) -> HazardRecord:
    return HazardRecord(
        record_id=record_id,
        key=resource,
        binding_stamp=stamp(generation),
        run_id=run_id,
        activated_at=activated_at,
    )


def unsafe_bundle(
    record: HazardRecord,
    *,
    bundle_id: str = "safety",
    disposition_id: str = "quarantine",
    recorded_at: float = 2.0,
) -> SafetyDispositionBundle:
    return SafetyDispositionBundle(
        bundle_id=bundle_id,
        run_id=record.run_id,
        records=(
            SafetyDispositionRecord(
                disposition_id=disposition_id,
                key=record.key,
                outcome=SafetyOutcome.UNSAFE,
                hazard_record_id=record.record_id,
                binding_stamp=record.binding_stamp,
                safe_receipt=None,
                reason="safe state could not be verified",
                recovery_action="re-establish the same asset and verify safe state",
            ),
        ),
        recorded_at=recorded_at,
    )


def safe_bundle(
    record: HazardRecord,
    *,
    bundle_id: str,
    recorded_at: float,
) -> SafetyDispositionBundle:
    return SafetyDispositionBundle(
        bundle_id=bundle_id,
        run_id=record.run_id,
        records=(
            SafetyDispositionRecord(
                disposition_id=f"disposition-{record.record_id}",
                key=record.key,
                outcome=SafetyOutcome.SAFE,
                hazard_record_id=record.record_id,
                binding_stamp=record.binding_stamp,
                safe_receipt=SafeReceipt(
                    record.key,
                    record.binding_stamp,
                    "VERIFY_SAFE_STATE",
                    "safe-state",
                ),
                reason=None,
                recovery_action=None,
            ),
        ),
        recorded_at=recorded_at,
    )


def assert_second_call_waits(call, entered, release) -> None:
    second_returned = threading.Event()
    failures: list[BaseException] = []

    def invoke(done=None) -> None:
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
    assert entered.wait(2)
    second.start()
    assert not second_returned.wait(0.05)
    release.set()
    first.join(2)
    second.join(2)
    assert not first.is_alive() and not second.is_alive()
    assert failures == []


def test_hazard_survives_restart_and_blocks_ordinary_acquisition(tmp_path):
    path = tmp_path / "installation-safety.zlcj"
    resource = ResourceKey.parse("device/camera/qcmos")
    journal = PersistentSafetyJournal(path)
    record = hazard(resource)
    assert journal.append_hazards((record,)) is HazardAppendStatus.APPENDED
    journal.close()

    restarted = PersistentSafetyJournal(path)
    assert restarted.snapshot().unresolved_hazards == (record,)
    arbiter = ResourceArbiter(restarted)
    assert isinstance(
        arbiter.acquire_all("another-run", (ResourceClaim(resource),)),
        ResourceQuarantined,
    )
    arbiter.shutdown()


def test_quarantine_and_recovery_round_trip_with_new_generation_and_clock_rollback(
    tmp_path,
):
    path = tmp_path / "installation-safety.zlcj"
    resource = ResourceKey.parse("device/fpga/pulse")
    journal = PersistentSafetyJournal(path)
    record = hazard(resource, activated_at=10_000.0)
    journal.append_hazards((record,))
    journal.append_safety_bundle(
        unsafe_bundle(record, recorded_at=-10_000.0)
    )
    snapshot = journal.snapshot()
    assert snapshot.unresolved_hazards == ()
    assert tuple(item.record_id for item in snapshot.unresolved_quarantines) == (
        "quarantine",
    )

    recovery = RecoveryBundle(
        bundle_id="recovery",
        claim=RecoveryClaim(
            key=resource,
            physical_identity=physical(),
            blocking_record_id="quarantine",
        ),
        evidence=RecoveryEvidence(
            binding_stamp=stamp("new-generation"),
            safe_state_digest="verified-safe-state",
        ),
        recorded_at=-20_000.0,
    )
    journal.append_recovery_bundle(recovery)
    journal.append_recovery_bundle(recovery)
    assert journal.snapshot() == SafetyJournalSnapshot((), ())
    assert journal.append_hazards((record,)) is HazardAppendStatus.ALREADY_RESOLVED
    journal.close()

    reopened = PersistentSafetyJournal(path)
    assert reopened.snapshot() == SafetyJournalSnapshot((), ())
    reopened.close()


def test_durable_then_lost_ack_retry_does_not_duplicate_or_rescan(tmp_path, monkeypatch):
    import zlc_storage.framed_journal as framed_journal

    path = tmp_path / "installation-safety.zlcj"
    resource = ResourceKey.parse("device/camera/lost-ack")
    journal = PersistentSafetyJournal(path)
    record = hazard(resource)
    real_fsync = framed_journal.os.fsync
    failed = False

    def durable_then_raise(file_descriptor):
        nonlocal failed
        real_fsync(file_descriptor)
        if not failed:
            failed = True
            raise OSError("fsync acknowledgement lost")

    monkeypatch.setattr(framed_journal.os, "fsync", durable_then_raise)
    with pytest.raises(OSError, match="acknowledgement lost"):
        journal.append_hazards((record,))
    assert journal.snapshot().unresolved_hazards == (record,)
    assert (
        journal.append_hazards((record,))
        is HazardAppendStatus.ALREADY_UNRESOLVED_SAME
    )
    journal.close()
    reopened = PersistentSafetyJournal(path)
    assert reopened.snapshot().unresolved_hazards == (record,)
    reopened.close()


def test_torn_tail_is_repaired_without_losing_last_complete_safety_fact(tmp_path):
    path = tmp_path / "installation-safety.zlcj"
    resource = ResourceKey.parse("device/camera/torn-tail")
    record = hazard(resource)
    journal = PersistentSafetyJournal(path)
    journal.append_hazards((record,))
    journal.close()
    complete_size = path.stat().st_size
    with path.open("ab") as stream:
        stream.write(b"ZLCJNL1")
        stream.flush()
        os.fsync(stream.fileno())
    assert path.stat().st_size > complete_size

    repaired = PersistentSafetyJournal(path)
    assert repaired.snapshot().unresolved_hazards == (record,)
    assert path.stat().st_size == complete_size
    repaired.close()


def test_steady_append_uses_one_startup_scan(tmp_path, monkeypatch):
    path = tmp_path / "installation-safety.zlcj"
    scans = 0
    real_scan = FramedJournal._scan

    def counting_scan(self, *args, **kwargs):
        nonlocal scans
        scans += 1
        return real_scan(self, *args, **kwargs)

    monkeypatch.setattr(FramedJournal, "_scan", counting_scan)
    journal = PersistentSafetyJournal(path)
    assert scans == 1
    for index in range(25):
        record = hazard(
            ResourceKey.parse(f"device/camera/unit-{index}"),
            record_id=f"hazard-{index}",
            run_id=f"run-{index}",
            activated_at=float(index),
        )
        journal.append_hazards((record,))
        journal.append_safety_bundle(
            safe_bundle(
                record,
                bundle_id=f"safety-{index}",
                recorded_at=float(index),
            )
        )
    assert journal.snapshot() == SafetyJournalSnapshot((), ())
    assert scans == 1
    journal.close()


@pytest.mark.parametrize(
    "mutate",
    (
        lambda value: value["binding_stamp"]["physical_identity"].pop(
            "evidence_digest"
        ),
        lambda value: value["binding_stamp"]["physical_identity"].update(
            {"unexpected": "field"}
        ),
        lambda value: value["binding_stamp"]["physical_identity"].update(
            {"evidence_kind": "GUESSED"}
        ),
        lambda value: value["binding_stamp"].update(
            {"connection_generation": ""}
        ),
    ),
)
def test_identity_stamp_codec_is_strict_and_fail_closed(tmp_path, mutate):
    path = tmp_path / "malformed-safety.zlcj"
    value = {
        "record_id": "hazard",
        "key": "device/camera/test",
        "binding_stamp": {
            "physical_identity": {
                "stable_device_identity": "camera-serial",
                "evidence_kind": "HARDWARE_IDENTITY_READBACK",
                "evidence_digest": "readback",
                "asset_map_revision": "assets-v1",
            },
            "connection_generation": "generation",
        },
        "run_id": "run",
        "activated_at": 1.0,
    }
    mutate(value)
    FramedJournal(path).append(
        "malformed",
        {"kind": "HAZARD_BATCH", "records": [value]},
    )
    with pytest.raises((TypeError, ValueError)):
        PersistentSafetyJournal(path)
    valid = PersistentSafetyJournal(tmp_path / "valid-safety.zlcj")
    valid.close()


def test_second_installation_authority_is_rejected_until_real_close(tmp_path):
    path = tmp_path / "installation-safety.zlcj"
    first = PersistentSafetyJournal(path)
    with pytest.raises(SafetyAuthorityBusy):
        PersistentSafetyJournal(path)
    first.close()
    with pytest.raises(RuntimeError, match="authority is closed"):
        first.snapshot()
    second = PersistentSafetyJournal(path)
    second.close()


def test_concurrent_close_waits_for_physical_owner_unlock(tmp_path, monkeypatch):
    import zlc_storage.framed_journal as framed_journal

    path = tmp_path / "installation-safety.zlcj"
    journal = PersistentSafetyJournal(path)
    real_unlock = framed_journal.release_file_lock
    entered = threading.Event()
    release = threading.Event()

    def blocked_unlock(stream):
        entered.set()
        assert release.wait(2)
        real_unlock(stream)

    monkeypatch.setattr(framed_journal, "release_file_lock", blocked_unlock)
    assert_second_call_waits(journal.close, entered, release)
    reopened = PersistentSafetyJournal(path)
    reopened.close()


def test_concurrent_arbiter_shutdown_waits_for_physical_owner_unlock(
    tmp_path,
    monkeypatch,
):
    import zlc_storage.framed_journal as framed_journal

    path = tmp_path / "installation-safety.zlcj"
    arbiter = ResourceArbiter(PersistentSafetyJournal(path))
    real_unlock = framed_journal.release_file_lock
    entered = threading.Event()
    release = threading.Event()

    def blocked_unlock(stream):
        entered.set()
        assert release.wait(2)
        real_unlock(stream)

    monkeypatch.setattr(framed_journal, "release_file_lock", blocked_unlock)
    assert_second_call_waits(arbiter.shutdown, entered, release)
    reopened = PersistentSafetyJournal(path)
    reopened.close()


def test_unlock_failure_leaves_old_authority_permanently_closed(tmp_path, monkeypatch):
    import zlc_storage.framed_journal as framed_journal

    path = tmp_path / "installation-safety.zlcj"
    journal = PersistentSafetyJournal(path)
    real_unlock = framed_journal.release_file_lock

    def fail_unlock(_stream):
        raise OSError("injected owner unlock failure")

    monkeypatch.setattr(framed_journal, "release_file_lock", fail_unlock)
    with pytest.raises(OSError, match="unlock failure"):
        journal.close()
    with pytest.raises(RuntimeError, match="authority is closed"):
        journal.snapshot()
    monkeypatch.setattr(framed_journal, "release_file_lock", real_unlock)
    reopened = PersistentSafetyJournal(path)
    reopened.close()


@pytest.mark.skipif(not hasattr(os, "fork"), reason="POSIX fork contract")
def test_forked_copy_cannot_unlock_parent_authority(tmp_path):
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


def test_active_resource_authority_cannot_release_owner_lock(tmp_path):
    path = tmp_path / "installation-safety.zlcj"
    resource = ResourceKey.parse("device/fpga/live-owner")
    journal = PersistentSafetyJournal(path)
    arbiter = ResourceArbiter(journal)
    lease = arbiter.acquire_all("live-run", (ResourceClaim(resource),))
    assert isinstance(lease, ResourceLease)
    lease.activate_hazards((HazardClaim(resource, stamp("live-generation")),))
    with pytest.raises(RuntimeError, match="owned by ResourceArbiter"):
        journal.close()
    with pytest.raises(RuntimeError, match="active ownership"):
        arbiter.shutdown()
    with pytest.raises(SafetyAuthorityBusy):
        PersistentSafetyJournal(path)

    lease._commit_safety(
        (
            SafetyDecision.unsafe(
                resource,
                reason="safe state unknown",
                recovery_action="verify hardware",
            ),
        )
    )
    lease.release_after_safety(disposition="QUARANTINED")
    arbiter.shutdown()
    restarted = PersistentSafetyJournal(path)
    restarted.close()
