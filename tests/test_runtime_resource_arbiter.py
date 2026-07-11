"""Resource ownership and cancellation contracts for the new runtime spine."""

from __future__ import annotations

import threading

import pytest

from zlc_neutral_atom.runtime import (
    CancellationRequested,
    CancellationToken,
    ClaimMode,
    MemoryQuarantineJournal,
    QuarantineJournalError,
    ResourceArbiter,
    ResourceBusy,
    ResourceClaim,
    ResourceKey,
    ResourceLease,
    ResourceQuarantined,
)


def claim(path: str, mode: ClaimMode = ClaimMode.EXCLUSIVE) -> ResourceClaim:
    return ResourceClaim(ResourceKey.parse(path), mode)


def test_acquire_all_is_atomic_when_one_claim_conflicts():
    arbiter = ResourceArbiter()
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
    arbiter = ResourceArbiter()
    parent = arbiter.acquire_all("parent", (claim("device/camera"),))
    assert isinstance(parent, ResourceLease)
    result = arbiter.acquire_all("child", (claim("device/camera/serial-1"),))
    assert isinstance(result, ResourceBusy)
    assert result.conflicting_run_id == "parent"

    parent.release_safe()
    child = arbiter.acquire_all("child-first", (claim("device/camera/serial-1"),))
    assert isinstance(child, ResourceLease)
    reverse = arbiter.acquire_all("parent-second", (claim("device/camera"),))
    assert isinstance(reverse, ResourceBusy)


def test_observers_share_only_when_no_exclusive_claim_exists():
    arbiter = ResourceArbiter()
    first = arbiter.acquire_all("observer-a", (claim("telemetry/fpga", ClaimMode.OBSERVE),))
    second = arbiter.acquire_all("observer-b", (claim("telemetry/fpga", ClaimMode.OBSERVE),))
    assert isinstance(first, ResourceLease)
    assert isinstance(second, ResourceLease)
    blocked = arbiter.acquire_all("writer", (claim("telemetry/fpga"),))
    assert isinstance(blocked, ResourceBusy)


def test_one_run_rejects_overlapping_claims_instead_of_hiding_redundancy():
    arbiter = ResourceArbiter()
    with pytest.raises(ValueError, match="overlapping"):
        arbiter.acquire_all(
            "ambiguous",
            (claim("device/camera"), claim("device/camera/a", ClaimMode.OBSERVE)),
        )


def test_quarantine_survives_lease_release_and_blocks_hierarchy():
    arbiter = ResourceArbiter()
    key = ResourceKey.parse("device/fpga/board-1")
    lease = arbiter.acquire_all("failed-run", (ResourceClaim(key),))
    assert isinstance(lease, ResourceLease)
    records = lease.quarantine_and_release(
        keys=(key,),
        reason="safe acknowledgement failed",
        recovery_action="verify board safe state",
    )
    assert len(records) == 1
    assert lease.released
    assert lease.disposition == "QUARANTINED"
    assert not lease.release_safe()

    blocked = arbiter.acquire_all("next-run", (claim("device/fpga"),))
    assert isinstance(blocked, ResourceQuarantined)
    assert arbiter.resolve_quarantine(records[0].key, proof="operator verified safe readback")
    retry = arbiter.acquire_all("next-run", (claim("device/fpga"),))
    assert isinstance(retry, ResourceLease)


def test_cleanup_can_quarantine_only_the_resource_whose_safe_action_failed():
    arbiter = ResourceArbiter()
    camera = ResourceKey.parse("device/camera/serial-1")
    sequencer = ResourceKey.parse("device/fpga/board-1")
    lease = arbiter.acquire_all(
        "partial-cleanup",
        (ResourceClaim(camera), ResourceClaim(sequencer)),
    )
    assert isinstance(lease, ResourceLease)
    records = lease.quarantine_and_release(
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
    arbiter = ResourceArbiter()
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
    arbiter = ResourceArbiter()
    first = arbiter.acquire_all("one-run", (claim("device/camera/a"),))
    assert isinstance(first, ResourceLease)
    with pytest.raises(RuntimeError, match="already owns"):
        arbiter.acquire_all("one-run", (claim("device/camera/b"),))


def test_observe_claim_cannot_quarantine_hardware():
    arbiter = ResourceArbiter()
    key = ResourceKey.parse("telemetry/fpga")
    lease = arbiter.acquire_all("observer", (ResourceClaim(key, ClaimMode.OBSERVE),))
    assert isinstance(lease, ResourceLease)
    with pytest.raises(ValueError, match="EXCLUSIVE"):
        lease.quarantine_and_release(
            keys=(key,), reason="observer error", recovery_action="none"
        )
    assert not lease.released
    assert lease.release_safe()


def test_quarantine_journal_is_restart_stable_and_keeps_resolution_proof():
    journal = MemoryQuarantineJournal()
    key = ResourceKey.parse("device/qcmos/serial-9")
    first = ResourceArbiter(journal)
    lease = first.acquire_all("failed", (ResourceClaim(key),))
    assert isinstance(lease, ResourceLease)
    records = lease.quarantine_and_release(
        keys=(key,), reason="driver lost", recovery_action="verify camera idle"
    )

    restarted = ResourceArbiter(journal)
    assert isinstance(restarted.acquire_all("blocked", (ResourceClaim(key),)), ResourceQuarantined)
    assert restarted.resolve_quarantine(key, proof="serial and idle state verified")
    entries = journal.entries()
    assert entries[-1].record_id == records[0].record_id
    assert entries[-1].proof == "serial and idle state verified"
    assert isinstance(ResourceArbiter(journal).acquire_all("recovered", (ResourceClaim(key),)), ResourceLease)


def test_journal_failure_keeps_claim_active_and_cannot_fall_back_to_safe_release():
    class FailingJournal(MemoryQuarantineJournal):
        fail = True

        def append_quarantined(self, records):
            if self.fail:
                raise OSError("disk unavailable")
            super().append_quarantined(records)

    journal = FailingJournal()
    arbiter = ResourceArbiter(journal)
    key = ResourceKey.parse("device/fpga/board-x")
    lease = arbiter.acquire_all("unsafe", (ResourceClaim(key),))
    assert isinstance(lease, ResourceLease)
    with pytest.raises(QuarantineJournalError, match="failed to persist") as caught:
        lease.quarantine_and_release(
            keys=(key,), reason="safe failed", recovery_action="repair journal then recover"
        )
    assert isinstance(caught.value.__cause__, OSError)
    assert lease.disposition == "JOURNAL_WRITE_FAILED"
    assert not lease.released
    assert not lease.release_safe()
    assert isinstance(arbiter.acquire_all("blocked", (ResourceClaim(key),)), ResourceBusy)

    journal.fail = False
    lease.quarantine_and_release(
        keys=(key,), reason="safe failed", recovery_action="repair journal then recover"
    )
    assert lease.released


def test_partial_journal_write_retry_reuses_record_ids_and_restart_stays_resolved():
    class PartialJournal(MemoryQuarantineJournal):
        fail_once = True

        def append_quarantined(self, records):
            if self.fail_once:
                self.fail_once = False
                super().append_quarantined(records[:1])
                raise OSError("commit acknowledgement lost")
            super().append_quarantined(records)

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
        lease.quarantine_and_release(
            keys=(camera, fpga),
            reason="cleanup failed",
            recovery_action="verify both devices",
        )
    first_ids = tuple(record.record_id for record in journal.unresolved())
    records = lease.quarantine_and_release(
        keys=(camera, fpga),
        reason="cleanup failed",
        recovery_action="verify both devices",
    )
    assert records[0].record_id == first_ids[0]
    assert len({record.record_id for record in journal.unresolved()}) == 2

    restarted = ResourceArbiter(journal)
    assert restarted.resolve_quarantine(camera, proof="camera safe")
    assert restarted.resolve_quarantine(fpga, proof="fpga safe")
    clean_restart = ResourceArbiter(journal)
    assert not clean_restart.quarantine_records()


def test_release_and_quarantine_choose_one_atomic_terminal_disposition():
    arbiter = ResourceArbiter()
    key = ResourceKey.parse("device/camera/race")
    lease = arbiter.acquire_all("race", (ResourceClaim(key),))
    assert isinstance(lease, ResourceLease)
    barrier = threading.Barrier(2)
    outcomes = []

    def safe_finish():
        barrier.wait()
        outcomes.append(("safe", lease.release_safe()))

    def unsafe_finish():
        barrier.wait()
        try:
            lease.quarantine_and_release(
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
