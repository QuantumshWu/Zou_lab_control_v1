"""Current resource, device-binding, safety, and recovery contracts."""

from __future__ import annotations

import threading
from dataclasses import dataclass

import pytest

import zlc_neutral_atom.runtime.resources as resources_module
from zlc_neutral_atom.runtime.ports import (
    BoundDevice,
    DeviceBroker,
    RecoveryAttempt,
    RecoveryController,
    SafeStateAck,
    VerifiedPhysicalDeviceIdentity,
    _open_device_run,
)
from zlc_neutral_atom.runtime.resources import (
    ClaimMode,
    DeviceBindingStamp,
    DeviceIdentityEvidenceKind,
    HazardClaim,
    PhysicalDeviceIdentity,
    RecoveryBundle,
    ResourceArbiter,
    ResourceBusy,
    ResourceClaim,
    ResourceKey,
    ResourceLease,
    ResourceQuarantined,
    SafeReceipt,
    SafetyDecision,
)
from zlc_neutral_atom.runtime.safety_journal import PersistentSafetyJournal


def key(path: str = "device/camera/test") -> ResourceKey:
    return ResourceKey.parse(path)


def physical(
    stable: str = "camera-serial",
    *,
    digest: str = "identity-readback",
    revision: str = "assets-v1",
) -> PhysicalDeviceIdentity:
    return PhysicalDeviceIdentity(
        stable_device_identity=stable,
        evidence_kind=DeviceIdentityEvidenceKind.HARDWARE_IDENTITY_READBACK,
        evidence_digest=digest,
        asset_map_revision=revision,
    )


def arbiter_at(path) -> ResourceArbiter:
    return ResourceArbiter(PersistentSafetyJournal(path))


def acquire(
    arbiter: ResourceArbiter,
    run_id: str,
    resource: ResourceKey,
) -> ResourceLease:
    result = arbiter.acquire_all(run_id, (ResourceClaim(resource),))
    assert isinstance(result, ResourceLease)
    return result


def bind(
    broker: DeviceBroker,
    resource: ResourceKey,
    identity: PhysicalDeviceIdentity,
    *,
    capability_probe=None,
) -> BoundDevice:
    verified = broker.verify_identity(lambda: identity)
    assert isinstance(verified, VerifiedPhysicalDeviceIdentity)
    return broker.bind(
        key=resource,
        identity=verified,
        execute_command=lambda command: ("executed", command),
        cleanup_operations={},
        verify_safe_state=lambda: SafeStateAck("safe-state"),
        capability_probe=capability_probe,
    )


def quarantine(
    arbiter: ResourceArbiter,
    resource: ResourceKey,
    stamp: DeviceBindingStamp,
    *,
    run_id: str = "unsafe-run",
) -> None:
    lease = acquire(arbiter, run_id, resource)
    assert lease.activate_hazards((HazardClaim(resource, stamp),))
    lease._commit_safety(
        (
            SafetyDecision.unsafe(
                resource,
                reason="safe state is not proven",
                recovery_action="verify the same physical device and safe state",
            ),
        )
    )
    assert lease.release_after_safety(disposition="QUARANTINED")


def test_removed_intermediate_authorities_are_not_restored():
    assert not hasattr(resources_module, "MemorySafetyJournal")
    assert not hasattr(resources_module, "ConnectionEstablishmentLease")
    assert not hasattr(ResourceArbiter, "begin_connection_establishment")
    assert not hasattr(DeviceBroker, "rebind")
    assert not hasattr(DeviceBroker, "current_binding")


def test_atomic_hierarchical_claims_and_observers(tmp_path):
    arbiter = arbiter_at(tmp_path / "safety.journal")
    parent = key("device/camera")
    child = parent.child("serial")
    held = acquire(arbiter, "writer", parent)

    result = arbiter.acquire_all(
        "all-or-nothing",
        (ResourceClaim(child), ResourceClaim(key("device/fpga"))),
    )
    assert isinstance(result, ResourceBusy)
    assert "all-or-nothing" not in arbiter.active_claims()
    held._release_unarmed()

    first = arbiter.acquire_all(
        "observer-1", (ResourceClaim(child, ClaimMode.OBSERVE),)
    )
    second = arbiter.acquire_all(
        "observer-2", (ResourceClaim(child, ClaimMode.OBSERVE),)
    )
    assert isinstance(first, ResourceLease)
    assert isinstance(second, ResourceLease)
    assert isinstance(
        arbiter.acquire_all("exclusive", (ResourceClaim(parent),)), ResourceBusy
    )
    first._release_unarmed()
    second._release_unarmed()
    arbiter.shutdown()


def test_concurrent_exclusive_acquire_has_one_winner(tmp_path):
    arbiter = arbiter_at(tmp_path / "safety.journal")
    resource = key("device/qcmos/serial")
    barrier = threading.Barrier(8)
    results: list[object] = []
    lock = threading.Lock()

    def compete(index: int) -> None:
        barrier.wait()
        result = arbiter.acquire_all(f"run-{index}", (ResourceClaim(resource),))
        with lock:
            results.append(result)

    threads = [threading.Thread(target=compete, args=(index,)) for index in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    winners = [result for result in results if isinstance(result, ResourceLease)]
    assert len(winners) == 1
    assert sum(isinstance(result, ResourceBusy) for result in results) == 7
    winners[0]._release_unarmed()
    arbiter.shutdown()


def test_safe_receipt_requires_exact_binding_stamp(tmp_path):
    arbiter = arbiter_at(tmp_path / "safety.journal")
    resource = key()
    authoritative = DeviceBindingStamp(physical(), "generation-a")
    wrong = DeviceBindingStamp(physical(digest="different-evidence"), "generation-a")
    lease = acquire(arbiter, "run", resource)
    lease.activate_hazards((HazardClaim(resource, authoritative),))

    with pytest.raises(ValueError, match="binding stamp"):
        lease._commit_safety(
            (
                SafetyDecision.safe(
                    SafeReceipt(resource, wrong, "verify", "wrong-ack")
                ),
            )
        )

    receipt = SafeReceipt(resource, authoritative, "verify", "safe-ack")
    bundle = lease._commit_safety((SafetyDecision.safe(receipt),))
    assert bundle is not None
    assert bundle.records[0].safe_receipt == receipt
    lease.release_after_safety(disposition="SUCCEEDED")
    arbiter.shutdown()


def test_unsafe_disposition_survives_restart_and_blocks_overlap(tmp_path):
    path = tmp_path / "safety.journal"
    resource = key("device/qcmos/serial")
    stamp = DeviceBindingStamp(physical("qcmos-serial"), "binding-a")
    first = arbiter_at(path)
    quarantine(first, resource, stamp)
    first.shutdown()

    restarted = arbiter_at(path)
    blocked = restarted.acquire_all("next-run", (ResourceClaim(resource),))
    assert isinstance(blocked, ResourceQuarantined)
    assert restarted.quarantine_records()[0].binding_stamp == stamp
    assert isinstance(
        restarted.acquire_all(
            "parent-run", (ResourceClaim(key("device/qcmos")),)
        ),
        ResourceQuarantined,
    )
    restarted.shutdown()


def test_recovery_requires_same_physical_identity_and_durable_safe_readback(tmp_path):
    path = tmp_path / "safety.journal"
    resource = key("device/qcmos/serial")
    identity = physical("qcmos-serial")
    broker = DeviceBroker()
    device = bind(broker, resource, identity)
    arbiter = arbiter_at(path)
    quarantine(arbiter, resource, device.binding_stamp)

    attempt = RecoveryController(arbiter, broker).begin(resource)
    assert isinstance(attempt, RecoveryAttempt)
    recovered = attempt.complete(device)
    assert isinstance(recovered, RecoveryBundle)
    assert recovered.evidence.binding_stamp == device.binding_stamp
    assert not arbiter.quarantine_records()

    lease = acquire(arbiter, "after-recovery", resource)
    lease._release_unarmed()
    broker.shutdown()
    arbiter.shutdown()


def test_wrong_physical_device_cannot_clear_quarantine(tmp_path):
    resource = key("device/qcmos/serial")
    arbiter = arbiter_at(tmp_path / "safety.journal")
    quarantine(
        arbiter,
        resource,
        DeviceBindingStamp(physical("expected-qcmos"), "old-binding"),
    )
    broker = DeviceBroker()
    wrong = bind(broker, resource, physical("other-qcmos"))
    attempt = RecoveryController(arbiter, broker).begin(resource)
    assert isinstance(attempt, RecoveryAttempt)
    with pytest.raises(ValueError, match="physical identity"):
        attempt.complete(wrong)
    assert attempt.abort()
    assert arbiter.quarantine_records()
    broker.shutdown()
    arbiter.shutdown()


def test_recovery_attempt_is_explicit_not_a_context_manager(tmp_path):
    resource = key()
    arbiter = arbiter_at(tmp_path / "safety.journal")
    quarantine(
        arbiter,
        resource,
        DeviceBindingStamp(physical(), "old-binding"),
    )
    broker = DeviceBroker()
    attempt = RecoveryController(arbiter, broker).begin(resource)
    assert isinstance(attempt, RecoveryAttempt)
    assert not hasattr(attempt, "__enter__")
    assert attempt.abort()
    broker.shutdown()
    arbiter.shutdown()


def test_verified_identity_is_one_use_and_failed_bind_can_discard_proof():
    broker = DeviceBroker()
    resource = key()
    identity = physical()
    proof = broker.verify_identity(lambda: identity)
    device = broker.bind(
        key=resource,
        identity=proof,
        execute_command=lambda command: command,
        cleanup_operations={},
        verify_safe_state=lambda: SafeStateAck("safe"),
    )
    assert device.binding_instance_id == proof.binding_instance_id
    with pytest.raises(RuntimeError, match="already consumed"):
        broker.bind(
            key=key("device/camera/other"),
            identity=proof,
            execute_command=lambda command: command,
            cleanup_operations={},
            verify_safe_state=lambda: SafeStateAck("safe"),
        )

    unused = broker.verify_identity(lambda: physical("other-camera"))
    with pytest.raises(RuntimeError, match="already bound"):
        broker.bind(
            key=resource,
            identity=unused,
            execute_command=lambda command: command,
            cleanup_operations={},
            verify_safe_state=lambda: SafeStateAck("safe"),
        )
    assert broker.discard_verified_identity(unused)
    assert not broker.discard_verified_identity(unused)
    broker.shutdown()


def test_installation_membership_rejects_one_physical_device_under_two_keys():
    broker = DeviceBroker()
    identity = physical()
    bind(broker, key("device/camera/a"), identity)
    duplicate = broker.verify_identity(lambda: identity)
    with pytest.raises(ValueError, match="stable physical identity"):
        broker.bind(
            key=key("device/camera/b"),
            identity=duplicate,
            execute_command=lambda command: command,
            cleanup_operations={},
            verify_safe_state=lambda: SafeStateAck("safe"),
        )
    assert broker.discard_verified_identity(duplicate)
    broker.shutdown()


def test_run_device_authority_is_revoked_as_one_lease():
    broker = DeviceBroker()
    device = bind(broker, key(), physical())
    lease = _open_device_run("run", (device,))
    assert lease.execute(device, "capture") == ("executed", "capture")
    lease.revoke()
    with pytest.raises(RuntimeError, match="revoked"):
        lease.execute(device, "late")
    broker.shutdown()


@dataclass(frozen=True)
class _CapabilitySnapshot:
    revision: int


def test_capability_proofs_are_frozen_and_latest_wins():
    broker = DeviceBroker()
    revision = [1]
    device = bind(
        broker,
        key(),
        physical(),
        capability_probe=lambda: _CapabilitySnapshot(revision[0]),
    )
    first = broker.verify_capability(device)
    assert device.validate_capability(first) == _CapabilitySnapshot(1)
    revision[0] = 2
    second = broker.verify_capability(device)
    assert device.validate_capability(second) == _CapabilitySnapshot(2)
    with pytest.raises(RuntimeError, match="unknown or revoked"):
        device.validate_capability(first)
    broker.shutdown()


def test_shutdown_refuses_to_hide_active_ownership(tmp_path):
    arbiter = arbiter_at(tmp_path / "safety.journal")
    lease = acquire(arbiter, "run", key())
    with pytest.raises(RuntimeError, match="active ownership"):
        arbiter.shutdown()
    lease._release_unarmed()
    arbiter.shutdown()
    with pytest.raises(RuntimeError, match="shut down"):
        arbiter.acquire_all("late", (ResourceClaim(key()),))
