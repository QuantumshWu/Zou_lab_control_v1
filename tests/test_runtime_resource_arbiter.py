"""Resource ownership and cancellation contracts for the new runtime spine."""

from __future__ import annotations

import threading
import time

import pytest

from zlc_neutral_atom.runtime import (
    CancellationRequested,
    CancellationToken,
    ClaimMode,
    DeviceBroker,
    DeviceIdentityAck,
    DeviceIdentityEvidenceKind,
    HazardClaim,
    HazardRecord,
    MemoryQuarantineJournal,
    QuarantineJournalError,
    RecoveryEvidence,
    RecoveryBundle,
    RecoveryClaim,
    RecoveryAck,
    RecoveryAttempt,
    RecoveryController,
    ResourceArbiter,
    ResourceBusy,
    ResourceClaim,
    ResourceKey,
    ResourceLease,
    ResourceQuarantined,
    SafeReceipt,
    SafeStateAck,
    SafetyDecision,
    SafetyDispositionBundle,
    SafetyDispositionRecord,
    SafetyOutcome,
)


def claim(path: str, mode: ClaimMode = ClaimMode.EXCLUSIVE) -> ResourceClaim:
    return ResourceClaim(ResourceKey.parse(path), mode)


def new_arbiter() -> ResourceArbiter:
    return ResourceArbiter(MemoryQuarantineJournal())


def safe_receipt(
    key: ResourceKey,
    generation: str,
    *,
    digest: str = "verified-safe-state",
) -> SafeReceipt:
    return SafeReceipt(
        key=key,
        stable_device_identity=str(key),
        connection_generation=generation,
        operation_id="VERIFY_SAFE_STATE",
        acknowledgement_digest=digest,
    )


def hazard(key: ResourceKey, generation: str) -> HazardClaim:
    return HazardClaim(key, str(key), generation)


def verified_identity(
    broker: DeviceBroker,
    stable_device_identity: str,
    evidence_digest: str,
):
    return broker.verify_identity(
        lambda: DeviceIdentityAck(
            stable_device_identity,
            DeviceIdentityEvidenceKind.HARDWARE_IDENTITY_READBACK,
            evidence_digest,
            "test-assets-v1",
        )
    )


def release_safe(
    lease: ResourceLease,
    receipts: tuple[SafeReceipt, ...] = (),
) -> bool:
    receipts = tuple(receipts)
    if lease.released:
        return False
    try:
        active = lease._arbiter._active_hazard_records(lease._capability, lease.run_id)
    except RuntimeError:
        if lease.released:
            return False
        raise
    if not active:
        if receipts:
            lease.activate_hazards(
                tuple(
                    HazardClaim(
                        receipt.key,
                        receipt.stable_device_identity,
                        receipt.connection_generation,
                    )
                    for receipt in receipts
                )
            )
        else:
            return lease._release_unarmed()
    lease._commit_safety(tuple(SafetyDecision.safe(receipt) for receipt in receipts))
    return lease.release_after_safety(disposition="SAFE")


def quarantine_and_release(
    lease: ResourceLease,
    *,
    keys: tuple[ResourceKey, ...],
    reason: str,
    recovery_action: str,
    safe_receipts: tuple[SafeReceipt, ...] = (),
):
    keys = tuple(keys)
    safe_receipts = tuple(safe_receipts)
    active = lease._arbiter._active_hazard_records(lease._capability, lease.run_id)
    if not active:
        lease.activate_hazards(
            tuple(hazard(key, "test-generation") for key in keys)
            + tuple(
                HazardClaim(
                    receipt.key,
                    receipt.stable_device_identity,
                    receipt.connection_generation,
                )
                for receipt in safe_receipts
            )
        )
    decisions = tuple(
        SafetyDecision.unsafe(
            key,
            reason=reason,
            recovery_action=recovery_action,
        )
        for key in keys
    ) + tuple(SafetyDecision.safe(receipt) for receipt in safe_receipts)
    bundle = lease._commit_safety(tuple(sorted(decisions, key=lambda value: value.key)))
    assert bundle is not None
    records = lease._arbiter._quarantines_for_bundle(bundle.bundle_id)
    lease.release_after_safety(disposition="QUARANTINED")
    return records


def complete_recovery(
    arbiter: ResourceArbiter,
    key: ResourceKey,
    *,
    generation: str = "recovery-generation",
) -> object:
    devices = DeviceBroker()
    devices.bind(
        key=key,
        identity=verified_identity(devices, str(key), generation),
        execute_command=lambda command: command,
        cleanup_operations={},
        verify_safe_state=lambda: SafeStateAck("verified-safe-state"),
        recovery_probe=lambda: RecoveryAck(
            stable_device_identity=str(key),
            connection_generation=devices.current_binding(key).connection_generation,
            health_digest="verified-health",
            safe_state_digest="verified-safe-state",
            verified_at=time.time(),
        ),
    )
    recovery = RecoveryController(arbiter, devices).begin(key)
    assert isinstance(recovery, RecoveryAttempt)
    return recovery.complete()


def test_acquire_all_is_atomic_when_one_claim_conflicts():
    arbiter = new_arbiter()
    held = arbiter.acquire_all("run-a", (claim("device/camera/a"),))
    assert isinstance(held, ResourceLease)

    result = arbiter.acquire_all(
        "run-b",
        (claim("device/camera/b"), claim("device/camera/a")),
    )
    assert isinstance(result, ResourceBusy)
    active = arbiter.active_claims()
    assert len(active) == 1
    assert "run-b" not in active


def test_hierarchical_exclusive_claim_conflicts_with_child():
    arbiter = new_arbiter()
    parent = arbiter.acquire_all("parent", (claim("device/camera"),))
    assert isinstance(parent, ResourceLease)
    result = arbiter.acquire_all("child", (claim("device/camera/serial-1"),))
    assert isinstance(result, ResourceBusy)
    assert result.conflicting_run_id == "parent"

    release_safe(parent)
    child = arbiter.acquire_all("child-first", (claim("device/camera/serial-1"),))
    assert isinstance(child, ResourceLease)
    reverse = arbiter.acquire_all("parent-second", (claim("device/camera"),))
    assert isinstance(reverse, ResourceBusy)


def test_observers_share_only_when_no_exclusive_claim_exists():
    arbiter = new_arbiter()
    first = arbiter.acquire_all("observer-a", (claim("telemetry/fpga", ClaimMode.OBSERVE),))
    second = arbiter.acquire_all("observer-b", (claim("telemetry/fpga", ClaimMode.OBSERVE),))
    assert isinstance(first, ResourceLease)
    assert isinstance(second, ResourceLease)
    blocked = arbiter.acquire_all("writer", (claim("telemetry/fpga"),))
    assert isinstance(blocked, ResourceBusy)


def test_one_run_rejects_overlapping_claims_instead_of_hiding_redundancy():
    arbiter = new_arbiter()
    with pytest.raises(ValueError, match="overlapping"):
        arbiter.acquire_all(
            "ambiguous",
            (claim("device/camera"), claim("device/camera/a", ClaimMode.OBSERVE)),
        )


def test_quarantine_survives_lease_release_and_blocks_hierarchy():
    arbiter = new_arbiter()
    key = ResourceKey.parse("device/fpga/board-1")
    lease = arbiter.acquire_all("failed-run", (ResourceClaim(key),))
    assert isinstance(lease, ResourceLease)
    records = quarantine_and_release(lease,
        keys=(key,),
        reason="safe acknowledgement failed",
        recovery_action="verify board safe state",
    )
    assert len(records) == 1
    assert lease.released
    assert lease.disposition == "QUARANTINED"
    assert not release_safe(lease)

    blocked = arbiter.acquire_all("next-run", (claim("device/fpga"),))
    assert isinstance(blocked, ResourceQuarantined)
    complete_recovery(arbiter, records[0].key)
    retry = arbiter.acquire_all("next-run", (claim("device/fpga"),))
    assert isinstance(retry, ResourceLease)


def test_cleanup_can_quarantine_only_the_resource_whose_safe_action_failed():
    arbiter = new_arbiter()
    camera = ResourceKey.parse("device/camera/serial-1")
    sequencer = ResourceKey.parse("device/fpga/board-1")
    lease = arbiter.acquire_all(
        "partial-cleanup",
        (ResourceClaim(camera), ResourceClaim(sequencer)),
    )
    assert isinstance(lease, ResourceLease)
    records = quarantine_and_release(lease,
        keys=(sequencer,),
        reason="sequencer safe failed / transport lost",
        recovery_action="reconnect and verify safe",
    )
    assert tuple(record.key for record in records) == (sequencer,)
    assert isinstance(arbiter.acquire_all("camera-ok", (ResourceClaim(camera),)), ResourceLease)
    assert isinstance(
        arbiter.acquire_all("fpga-blocked", (ResourceClaim(sequencer),)),
        ResourceQuarantined,
    )


def test_concurrent_acquire_has_exactly_one_exclusive_winner():
    arbiter = new_arbiter()
    barrier = threading.Barrier(8)
    results = []
    lock = threading.Lock()

    def compete(index: int) -> None:
        barrier.wait()
        result = arbiter.acquire_all(f"run-{index}", (claim("device/qcmos/serial"),))
        with lock:
            results.append(result)

    threads = [threading.Thread(target=compete, args=(index,)) for index in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert sum(isinstance(result, ResourceLease) for result in results) == 1
    assert sum(isinstance(result, ResourceBusy) for result in results) == 7


def test_same_run_cannot_acquire_a_second_disjoint_lease():
    arbiter = new_arbiter()
    first = arbiter.acquire_all("one-run", (claim("device/camera/a"),))
    assert isinstance(first, ResourceLease)
    with pytest.raises(RuntimeError, match="already owns"):
        arbiter.acquire_all("one-run", (claim("device/camera/b"),))


def test_observe_claim_cannot_quarantine_hardware():
    arbiter = new_arbiter()
    key = ResourceKey.parse("telemetry/fpga")
    lease = arbiter.acquire_all("observer", (ResourceClaim(key, ClaimMode.OBSERVE),))
    assert isinstance(lease, ResourceLease)
    with pytest.raises(ValueError, match="EXCLUSIVE"):
        quarantine_and_release(lease,
            keys=(key,), reason="observer error", recovery_action="none"
        )
    assert not lease.released
    assert release_safe(lease)


def test_quarantine_journal_is_restart_stable_and_keeps_resolution_proof():
    journal = MemoryQuarantineJournal()
    key = ResourceKey.parse("device/qcmos/serial-9")
    first = ResourceArbiter(journal)
    lease = first.acquire_all("failed", (ResourceClaim(key),))
    assert isinstance(lease, ResourceLease)
    records = quarantine_and_release(lease,
        keys=(key,), reason="driver lost", recovery_action="verify camera idle"
    )

    restarted = ResourceArbiter(journal)
    assert isinstance(restarted.acquire_all("blocked", (ResourceClaim(key),)), ResourceQuarantined)
    complete_recovery(restarted, key)
    entries = journal.entries()
    assert entries[-1].claim.quarantine_record_ids == (records[0].record_id,)
    assert entries[-1].evidence.safe_state_digest == "verified-safe-state"
    assert isinstance(ResourceArbiter(journal).acquire_all("recovered", (ResourceClaim(key),)), ResourceLease)


def test_journal_failure_keeps_claim_active_and_cannot_fall_back_to_safe_release():
    class FailingJournal(MemoryQuarantineJournal):
        fail = True

        def append_safety_bundle(self, bundle):
            if self.fail:
                raise OSError("disk unavailable")
            super().append_safety_bundle(bundle)

    journal = FailingJournal()
    arbiter = ResourceArbiter(journal)
    key = ResourceKey.parse("device/fpga/board-x")
    lease = arbiter.acquire_all("unsafe", (ResourceClaim(key),))
    assert isinstance(lease, ResourceLease)
    with pytest.raises(QuarantineJournalError, match="failed to persist") as caught:
        quarantine_and_release(lease,
            keys=(key,), reason="safe failed", recovery_action="repair journal then recover"
        )
    assert isinstance(caught.value.__cause__, OSError)
    assert lease.disposition == "SAFETY_JOURNAL_FAILED"
    assert not lease.released
    with pytest.raises(ValueError, match="original decisions"):
        release_safe(lease)
    assert isinstance(arbiter.acquire_all("blocked", (ResourceClaim(key),)), ResourceBusy)

    journal.fail = False
    quarantine_and_release(lease,
        keys=(key,), reason="safe failed", recovery_action="repair journal then recover"
    )
    assert lease.released


def test_partial_journal_write_retry_reuses_record_ids_and_restart_stays_resolved():
    class PartialJournal(MemoryQuarantineJournal):
        fail_once = True

        def append_safety_bundle(self, bundle):
            if self.fail_once:
                self.fail_once = False
                super().append_safety_bundle(bundle)
                raise OSError("commit acknowledgement lost")
            super().append_safety_bundle(bundle)

    journal = PartialJournal()
    arbiter = ResourceArbiter(journal)
    camera = ResourceKey.parse("device/camera/partial")
    fpga = ResourceKey.parse("device/fpga/partial")
    lease = arbiter.acquire_all(
        "partial-journal",
        (ResourceClaim(camera), ResourceClaim(fpga)),
    )
    assert isinstance(lease, ResourceLease)
    with pytest.raises(QuarantineJournalError):
        quarantine_and_release(lease,
            keys=(camera, fpga),
            reason="cleanup failed",
            recovery_action="verify both devices",
        )
    first_ids = tuple(record.record_id for record in journal.unresolved_quarantines())
    records = quarantine_and_release(lease,
        keys=(camera, fpga),
        reason="cleanup failed",
        recovery_action="verify both devices",
    )
    assert records[0].record_id == first_ids[0]
    assert len({record.record_id for record in journal.unresolved_quarantines()}) == 2

    restarted = ResourceArbiter(journal)
    complete_recovery(restarted, camera)
    complete_recovery(restarted, fpga)
    clean_restart = ResourceArbiter(journal)
    assert not clean_restart.quarantine_records()


def test_release_and_quarantine_choose_one_atomic_terminal_disposition():
    arbiter = new_arbiter()
    key = ResourceKey.parse("device/camera/race")
    lease = arbiter.acquire_all("race", (ResourceClaim(key),))
    assert isinstance(lease, ResourceLease)
    barrier = threading.Barrier(2)
    outcomes = []

    def safe_finish():
        barrier.wait()
        outcomes.append(("safe", release_safe(lease)))

    def unsafe_finish():
        barrier.wait()
        try:
            quarantine_and_release(lease,
                keys=(key,), reason="safe failed", recovery_action="verify camera"
            )
            outcomes.append(("quarantine", True))
        except RuntimeError:
            outcomes.append(("quarantine", False))

    threads = [threading.Thread(target=safe_finish), threading.Thread(target=unsafe_finish)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sum(bool(outcome) for _name, outcome in outcomes) == 1
    assert lease.released


def test_resource_key_constructor_rejects_string_as_segment_sequence():
    with pytest.raises(TypeError, match="tuple"):
        ResourceKey("abc")


def test_cancellation_token_is_monotonic_and_preserves_first_reason():
    token = CancellationToken()
    assert token.request("user requested stop")
    assert not token.request("later reason")
    snapshot = token.snapshot()
    assert snapshot.requested
    assert snapshot.reason == "user requested stop"
    assert snapshot.requested_at is not None
    with pytest.raises(CancellationRequested, match="user requested stop"):
        token.checkpoint()


def test_hazard_active_survives_process_restart_until_explicit_safe_recovery():
    journal = MemoryQuarantineJournal()
    key = ResourceKey.parse("device/qcmos/crash-test")
    first_process = ResourceArbiter(journal)
    lease = first_process.acquire_all("crashed-run", (ResourceClaim(key),))
    assert isinstance(lease, ResourceLease)
    assert lease.activate_hazards((hazard(key, "connection-generation-7"),))
    assert len(journal.unresolved_hazards()) == 1

    restarted = ResourceArbiter(journal)
    blocked = restarted.acquire_all("after-crash", (ResourceClaim(key),))
    assert isinstance(blocked, ResourceQuarantined)
    assert "crashed-run" in blocked.reason
    complete_recovery(restarted, key, generation="connection-generation-8")
    assert not journal.unresolved_hazards()
    assert isinstance(
        ResourceArbiter(journal).acquire_all("recovered", (ResourceClaim(key),)),
        ResourceLease,
    )


def test_safe_release_resolves_write_ahead_hazard_before_unlocking_resource():
    journal = MemoryQuarantineJournal()
    arbiter = ResourceArbiter(journal)
    key = ResourceKey.parse("device/fpga/safe-run")
    lease = arbiter.acquire_all("safe-run", (ResourceClaim(key),))
    assert isinstance(lease, ResourceLease)
    lease.activate_hazards((hazard(key, "connection-generation-2"),))
    assert journal.unresolved_hazards()
    assert release_safe(lease,
        receipts=(safe_receipt(key, "connection-generation-2"),)
    )
    assert not journal.unresolved_hazards()


def test_safe_resolution_partial_write_retry_reuses_the_same_resolution():
    class PartialResolutionJournal(MemoryQuarantineJournal):
        fail_once = True

        def append_safety_bundle(self, bundle):
            if self.fail_once:
                self.fail_once = False
                super().append_safety_bundle(bundle)
                raise OSError("resolution acknowledgement lost")
            super().append_safety_bundle(bundle)

    journal = PartialResolutionJournal()
    arbiter = ResourceArbiter(journal)
    key = ResourceKey.parse("device/fpga/resolution-retry")
    lease = arbiter.acquire_all("safe-retry", (ResourceClaim(key),))
    assert isinstance(lease, ResourceLease)
    lease.activate_hazards((hazard(key, "connection-generation-3"),))
    with pytest.raises(QuarantineJournalError):
        release_safe(lease,
            receipts=(safe_receipt(key, "connection-generation-3"),)
        )
    assert not lease.released
    with pytest.raises(ValueError, match="original decisions"):
        release_safe(lease,
            receipts=(
                safe_receipt(
                    key,
                    "connection-generation-3",
                    digest="different-safe-state",
                ),
            )
        )
    assert release_safe(lease,
        receipts=(safe_receipt(key, "connection-generation-3"),)
    )
    assert not ResourceArbiter(journal).unresolved_hazards()


def test_resource_arbiter_never_defaults_real_safety_state_to_process_memory():
    with pytest.raises(TypeError, match="journal is required"):
        ResourceArbiter(None)


def test_resource_lease_exposes_no_public_forgeable_safe_release_path():
    key = ResourceKey.parse("device/camera/no-public-bypass")
    lease = new_arbiter().acquire_all("run", (ResourceClaim(key),))
    assert isinstance(lease, ResourceLease)
    assert not hasattr(lease, "release_safe")
    assert not hasattr(lease, "quarantine_and_release")
    assert not hasattr(lease, "commit_safety")
    lease._release_unarmed()


def test_device_binding_requires_a_typed_identity_handshake():
    broker = DeviceBroker()
    key = ResourceKey.parse("device/camera/typed-identity")
    with pytest.raises(TypeError, match="DeviceIdentityAck"):
        broker.verify_identity(lambda: "camera-serial")
    with pytest.raises(TypeError, match="VerifiedBoundDeviceIdentity"):
        broker.bind(
            key=key,
            identity="camera-serial",
            execute_command=lambda command: command,
            cleanup_operations={},
            verify_safe_state=lambda: SafeStateAck("safe"),
        )
    identity = verified_identity(broker, "camera-serial", "generation-one")
    with pytest.raises(AttributeError, match="immutable"):
        identity.stable_device_identity = "forged-serial"
    broker.bind(
        key=key,
        identity=identity,
        execute_command=lambda command: command,
        cleanup_operations={},
        verify_safe_state=lambda: SafeStateAck("safe"),
    )
    with pytest.raises(RuntimeError, match="already consumed"):
        broker.bind(
            key=ResourceKey.parse("device/camera/reused-token"),
            identity=identity,
            execute_command=lambda command: command,
            cleanup_operations={},
            verify_safe_state=lambda: SafeStateAck("safe"),
        )
    duplicate_physical_identity = verified_identity(
        broker,
        "camera-serial",
        "generation-two",
    )
    with pytest.raises(ValueError, match="multiple ResourceKeys"):
        broker.bind(
            key=ResourceKey.parse("device/camera/duplicate-physical-device"),
            identity=duplicate_physical_identity,
            execute_command=lambda command: command,
            cleanup_operations={},
            verify_safe_state=lambda: SafeStateAck("safe"),
        )


def test_broker_mints_a_fresh_generation_for_each_endpoint_establishment():
    broker = DeviceBroker()
    key = ResourceKey.parse("device/camera/reconnect-generation")

    first = broker.bind(
        key=key,
        identity=verified_identity(broker, "camera-serial", "readback-one"),
        execute_command=lambda command: command,
        cleanup_operations={},
        verify_safe_state=lambda: SafeStateAck("safe"),
    )
    second = broker.bind(
        key=key,
        identity=verified_identity(broker, "camera-serial", "readback-two"),
        execute_command=lambda command: command,
        cleanup_operations={},
        verify_safe_state=lambda: SafeStateAck("safe"),
    )

    assert first.connection_generation != second.connection_generation
    assert first.connection_generation not in {"readback-one", "readback-two"}
    assert second.connection_generation not in {"readback-one", "readback-two"}
    assert broker.current_binding(key) is second


def test_mixed_safety_bundle_is_atomic_idempotent_and_keeps_claim_until_terminal():
    class LostAcknowledgementJournal(MemoryQuarantineJournal):
        fail_once = True

        def append_safety_bundle(self, bundle):
            super().append_safety_bundle(bundle)
            if self.fail_once:
                self.fail_once = False
                raise OSError("bundle fsync acknowledgement lost")

    journal = LostAcknowledgementJournal()
    arbiter = ResourceArbiter(journal)
    camera = ResourceKey.parse("device/camera/mixed")
    fpga = ResourceKey.parse("device/fpga/mixed")
    lease = arbiter.acquire_all(
        "mixed-run",
        (ResourceClaim(camera), ResourceClaim(fpga)),
    )
    assert isinstance(lease, ResourceLease)
    lease.activate_hazards(
        (
            hazard(camera, "camera-generation"),
            hazard(fpga, "fpga-generation"),
        )
    )
    decisions = (
        SafetyDecision.safe(safe_receipt(camera, "camera-generation")),
        SafetyDecision.unsafe(
            fpga,
            reason="FPGA safe readback unavailable",
            recovery_action="reconnect and verify safe",
        ),
    )
    with pytest.raises(QuarantineJournalError):
        lease._commit_safety(decisions)
    first_bundle = next(
        entry for entry in journal.entries() if isinstance(entry, SafetyDispositionBundle)
    )
    assert lease._commit_safety(decisions) == first_bundle
    assert sum(isinstance(entry, SafetyDispositionBundle) for entry in journal.entries()) == 1
    assert isinstance(
        arbiter.acquire_all("still-held", (ResourceClaim(camera),)), ResourceBusy
    )

    lease.release_after_safety(disposition="FAILED")
    assert isinstance(
        arbiter.acquire_all("camera-reuse", (ResourceClaim(camera),)), ResourceLease
    )
    assert isinstance(
        arbiter.acquire_all("fpga-quarantine", (ResourceClaim(fpga),)),
        ResourceQuarantined,
    )


def test_recovery_bundle_retry_is_one_atomic_idempotent_record():
    class LostRecoveryAcknowledgement(MemoryQuarantineJournal):
        fail_once = True

        def append_recovery_bundle(self, bundle):
            super().append_recovery_bundle(bundle)
            if self.fail_once:
                self.fail_once = False
                raise OSError("recovery bundle acknowledgement lost")

    journal = LostRecoveryAcknowledgement()
    key = ResourceKey.parse("device/camera/recovery-bundle")
    arbiter = ResourceArbiter(journal)
    lease = arbiter.acquire_all("unsafe", (ResourceClaim(key),))
    assert isinstance(lease, ResourceLease)
    quarantine_and_release(lease,
        keys=(key,),
        reason="safe unknown",
        recovery_action="verify camera idle",
    )
    devices = DeviceBroker()
    devices.bind(
        key=key,
        identity=verified_identity(devices, str(key), "recovered-generation"),
        execute_command=lambda command: command,
        cleanup_operations={},
        verify_safe_state=lambda: SafeStateAck("verified-safe-state"),
        recovery_probe=lambda: RecoveryAck(
            stable_device_identity=str(key),
            connection_generation=devices.current_binding(key).connection_generation,
            health_digest="verified-health",
            safe_state_digest="verified-safe-state",
            verified_at=time.time(),
        ),
    )
    recovery = RecoveryController(arbiter, devices).begin(key)
    assert isinstance(recovery, RecoveryAttempt)
    with pytest.raises(QuarantineJournalError):
        recovery.complete()
    recovery.complete()
    recovery_entries = [
        entry for entry in journal.entries() if type(entry).__name__ == "RecoveryBundle"
    ]
    assert len(recovery_entries) == 1
    assert not ResourceArbiter(journal).quarantine_records()


def test_raw_recovery_evidence_cannot_authorize_quarantine_resolution():
    arbiter = new_arbiter()
    key = ResourceKey.parse("device/camera/raw-recovery-evidence")
    lease = arbiter.acquire_all("unsafe", (ResourceClaim(key),))
    assert isinstance(lease, ResourceLease)
    quarantine_and_release(lease,
        keys=(key,),
        reason="safe state unknown",
        recovery_action="verify camera",
    )
    recovery = arbiter.begin_recovery(key)
    assert recovery is not None and not isinstance(recovery, ResourceBusy)
    with pytest.raises(TypeError, match="VerifiedRecoveryProof"):
        recovery.complete(
            RecoveryEvidence(
                stable_device_identity="made-up-device",
                connection_generation="wrong-generation",
                health_digest="made-up-health",
                safe_state_digest="made-up-safe-state",
                verified_at=time.time(),
            )
        )
    recovery.abort()
    assert arbiter.quarantine_records()


def test_recovery_probe_identity_mismatch_keeps_quarantine():
    arbiter = new_arbiter()
    key = ResourceKey.parse("device/camera/recovery-identity")
    lease = arbiter.acquire_all("unsafe", (ResourceClaim(key),))
    assert isinstance(lease, ResourceLease)
    quarantine_and_release(lease,
        keys=(key,),
        reason="safe state unknown",
        recovery_action="verify camera",
    )
    devices = DeviceBroker()
    devices.bind(
        key=key,
        identity=verified_identity(devices, str(key), "current-generation"),
        execute_command=lambda command: command,
        cleanup_operations={},
        verify_safe_state=lambda: SafeStateAck("verified-safe-state"),
        recovery_probe=lambda: RecoveryAck(
            stable_device_identity="different-camera-serial",
            connection_generation=devices.current_binding(key).connection_generation,
            health_digest="verified-health",
            safe_state_digest="verified-safe-state",
            verified_at=time.time(),
        ),
    )

    with pytest.raises(ValueError, match="stable device identity"):
        RecoveryController(arbiter, devices).begin(key)

    assert arbiter.quarantine_records()
    assert not arbiter.active_claims()


def test_different_physical_device_cannot_clear_old_device_hazard():
    arbiter = new_arbiter()
    key = ResourceKey.parse("device/fpga/replaced-device")
    lease = arbiter.acquire_all("unsafe", (ResourceClaim(key),))
    assert isinstance(lease, ResourceLease)
    quarantine_and_release(
        lease,
        keys=(key,),
        reason="safe state unknown",
        recovery_action="recover the original physical device",
    )
    devices = DeviceBroker()
    devices.bind(
        key=key,
        identity=verified_identity(
            devices, "different-physical-serial", "new-generation"
        ),
        execute_command=lambda command: command,
        cleanup_operations={},
        verify_safe_state=lambda: SafeStateAck("safe"),
        recovery_probe=lambda: RecoveryAck(
            stable_device_identity="different-physical-serial",
            connection_generation=devices.current_binding(key).connection_generation,
            health_digest="healthy",
            safe_state_digest="safe",
            verified_at=time.time(),
        ),
    )
    with pytest.raises(ValueError, match="hazardous stable identity"):
        RecoveryController(arbiter, devices).begin(key)
    assert arbiter.quarantine_records()


def test_recovery_binding_lease_blocks_rebind_until_complete_or_abort():
    arbiter = new_arbiter()
    key = ResourceKey.parse("device/camera/recovery-rebind-race")
    lease = arbiter.acquire_all("unsafe", (ResourceClaim(key),))
    assert isinstance(lease, ResourceLease)
    quarantine_and_release(
        lease,
        keys=(key,),
        reason="safe state unknown",
        recovery_action="verify original camera",
    )
    entered = threading.Event()
    allow = threading.Event()
    devices = DeviceBroker()

    def probe():
        entered.set()
        assert allow.wait(2.0)
        return RecoveryAck(
            stable_device_identity=str(key),
            connection_generation=devices.current_binding(key).connection_generation,
            health_digest="healthy",
            safe_state_digest="safe",
            verified_at=time.time(),
        )

    devices.bind(
        key=key,
        identity=verified_identity(devices, str(key), "recovery-generation"),
        execute_command=lambda command: command,
        cleanup_operations={},
        verify_safe_state=lambda: SafeStateAck("safe"),
        recovery_probe=probe,
    )
    attempts = []
    failures = []

    def begin():
        try:
            attempts.append(RecoveryController(arbiter, devices).begin(key))
        except BaseException as exc:
            failures.append(exc)

    worker = threading.Thread(target=begin)
    worker.start()
    assert entered.wait(1.0)
    with pytest.raises(RuntimeError, match="during recovery"):
        devices.bind(
            key=key,
            identity=verified_identity(
                devices, str(key), "replacement-generation"
            ),
            execute_command=lambda command: command,
            cleanup_operations={},
            verify_safe_state=lambda: SafeStateAck("safe"),
        )
    allow.set()
    worker.join()
    assert not failures
    assert isinstance(attempts[0], RecoveryAttempt)
    attempts[0].abort()
    devices.bind(
        key=key,
        identity=verified_identity(
            devices, str(key), "replacement-generation"
        ),
        execute_command=lambda command: command,
        cleanup_operations={},
        verify_safe_state=lambda: SafeStateAck("safe"),
    )


def test_recovery_evidence_must_postdate_the_records_it_clears():
    arbiter = new_arbiter()
    key = ResourceKey.parse("device/camera/recovery-time")
    lease = arbiter.acquire_all("unsafe", (ResourceClaim(key),))
    assert isinstance(lease, ResourceLease)
    quarantine_and_release(lease,
        keys=(key,),
        reason="safe state unknown",
        recovery_action="verify camera",
    )
    devices = DeviceBroker()
    devices.bind(
        key=key,
        identity=verified_identity(devices, str(key), "current-generation"),
        execute_command=lambda command: command,
        cleanup_operations={},
        verify_safe_state=lambda: SafeStateAck("verified-safe-state"),
        recovery_probe=lambda: RecoveryAck(
            stable_device_identity=str(key),
            connection_generation=devices.current_binding(key).connection_generation,
            health_digest="verified-health",
            safe_state_digest="verified-safe-state",
            verified_at=1.0,
        ),
    )
    attempt = RecoveryController(arbiter, devices).begin(key)
    assert isinstance(attempt, RecoveryAttempt)
    with pytest.raises(ValueError, match="predates"):
        attempt.complete()
    attempt.abort()
    assert arbiter.quarantine_records()


def test_cross_key_safety_bundle_is_rejected_without_mutating_journal():
    journal = MemoryQuarantineJournal()
    camera = ResourceKey.parse("device/camera/cross-key")
    fpga = ResourceKey.parse("device/fpga/cross-key")
    hazard = HazardRecord(
        record_id="camera-hazard",
        key=camera,
        stable_device_identity=str(camera),
        connection_generation="camera-generation",
        run_id="camera-run",
        activated_at=1.0,
    )
    journal.append_hazards((hazard,))
    forged = SafetyDispositionBundle(
        bundle_id="forged-bundle",
        run_id="camera-run",
        records=(
            SafetyDispositionRecord(
                disposition_id="forged-disposition",
                key=fpga,
                outcome=SafetyOutcome.SAFE,
                hazard_record_id=hazard.record_id,
                stable_device_identity=str(fpga),
                connection_generation=hazard.connection_generation,
                safe_receipt=safe_receipt(fpga, hazard.connection_generation),
                reason=None,
                recovery_action=None,
            ),
        ),
        recorded_at=2.0,
    )

    with pytest.raises(ValueError, match="run/key/generation"):
        journal.append_safety_bundle(forged)

    assert journal.entries() == (hazard,)
    assert journal.unresolved_hazards() == (hazard,)


def test_partial_safety_bundle_is_rejected_atomically():
    journal = MemoryQuarantineJournal()
    camera = ResourceKey.parse("device/camera/partial-coverage")
    fpga = ResourceKey.parse("device/fpga/partial-coverage")
    hazards = (
        HazardRecord(
            record_id="camera-partial-hazard",
            key=camera,
            stable_device_identity=str(camera),
            connection_generation="camera-generation",
            run_id="partial-run",
            activated_at=1.0,
        ),
        HazardRecord(
            record_id="fpga-partial-hazard",
            key=fpga,
            stable_device_identity=str(fpga),
            connection_generation="fpga-generation",
            run_id="partial-run",
            activated_at=1.0,
        ),
    )
    journal.append_hazards(hazards)
    partial = SafetyDispositionBundle(
        bundle_id="partial-bundle",
        run_id="partial-run",
        records=(
            SafetyDispositionRecord(
                disposition_id="partial-camera-safe",
                key=camera,
                outcome=SafetyOutcome.SAFE,
                hazard_record_id=hazards[0].record_id,
                stable_device_identity=hazards[0].stable_device_identity,
                connection_generation=hazards[0].connection_generation,
                safe_receipt=safe_receipt(camera, "camera-generation"),
                reason=None,
                recovery_action=None,
            ),
        ),
        recorded_at=2.0,
    )

    with pytest.raises(ValueError, match="exactly cover"):
        journal.append_safety_bundle(partial)

    assert journal.entries() == hazards
    assert journal.unresolved_hazards() == hazards


def test_safe_receipt_generation_must_match_active_hazard():
    journal = MemoryQuarantineJournal()
    arbiter = ResourceArbiter(journal)
    key = ResourceKey.parse("device/camera/generation-check")
    lease = arbiter.acquire_all("generation-run", (ResourceClaim(key),))
    assert isinstance(lease, ResourceLease)
    lease.activate_hazards((hazard(key, "active-generation"),))

    with pytest.raises(ValueError, match="key/generation"):
        release_safe(lease,
            receipts=(safe_receipt(key, "stale-generation"),)
        )

    assert not lease.released
    assert release_safe(lease,
        receipts=(safe_receipt(key, "active-generation"),)
    )


def test_blocked_journal_io_does_not_freeze_unrelated_resource_acquire():
    entered = threading.Event()
    release = threading.Event()

    class BlockingJournal(MemoryQuarantineJournal):
        def append_hazards(self, records):
            entered.set()
            assert release.wait(2.0)
            return super().append_hazards(records)

    journal = BlockingJournal()
    arbiter = ResourceArbiter(journal)
    camera = ResourceKey.parse("device/camera/slow-journal")
    fpga = ResourceKey.parse("device/fpga/unrelated")
    camera_lease = arbiter.acquire_all("camera-run", (ResourceClaim(camera),))
    assert isinstance(camera_lease, ResourceLease)
    worker = threading.Thread(
        target=lambda: camera_lease.activate_hazards(
            (hazard(camera, "camera-generation"),)
        )
    )
    worker.start()
    assert entered.wait(1.0)

    unrelated = arbiter.acquire_all("fpga-run", (ResourceClaim(fpga),))
    assert isinstance(unrelated, ResourceLease)
    release_safe(unrelated)

    release.set()
    worker.join(2.0)
    assert not worker.is_alive()
    release_safe(camera_lease,
        receipts=(safe_receipt(camera, "camera-generation"),)
    )
