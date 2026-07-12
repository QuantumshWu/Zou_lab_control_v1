"""Restart and recovery contracts for installation-level safety facts."""

from __future__ import annotations

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
