"""Current in-process resource ownership contracts."""

from __future__ import annotations

import threading

import pytest

from zlc_neutral_atom.runtime.resources import (
    ClaimMode,
    ResourceArbiter,
    ResourceBusy,
    ResourceClaim,
    ResourceKey,
    ResourceLease,
    _mint_terminal_publication,
)


def key(path: str = "device/camera/test") -> ResourceKey:
    return ResourceKey.parse(path)


def acquire(
    arbiter: ResourceArbiter,
    run_id: str,
    resource: ResourceKey,
) -> ResourceLease:
    result = arbiter.acquire_all(run_id, (ResourceClaim(resource),))
    assert isinstance(result, ResourceLease)
    return result


def test_hierarchical_claims_are_atomic_and_observers_can_share() -> None:
    arbiter = ResourceArbiter()
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
        arbiter.acquire_all("exclusive", (ResourceClaim(parent),)),
        ResourceBusy,
    )
    first._release_unarmed()
    second._release_unarmed()
    arbiter.shutdown()


def test_concurrent_exclusive_acquire_has_one_winner() -> None:
    arbiter = ResourceArbiter()
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


def test_terminal_publication_and_release_are_one_transition() -> None:
    arbiter = ResourceArbiter()
    resource = key()
    lease = acquire(arbiter, "first", resource)
    events: list[str] = []

    publication = _mint_terminal_publication(
        lambda: events.append("published"),
        lambda: events.append("released"),
    )
    assert lease.release_terminal(publication, disposition="SUCCEEDED")
    assert events == ["published", "released"]
    assert lease.released
    assert lease.disposition == "SUCCEEDED"
    assert not lease.release_terminal(publication, disposition="SUCCEEDED")

    next_lease = acquire(arbiter, "next", resource)
    next_lease._release_unarmed()
    arbiter.shutdown()


def test_invalid_claim_sets_and_duplicate_run_ids_are_rejected() -> None:
    arbiter = ResourceArbiter()
    parent = key("device/camera")
    child = parent.child("serial")
    with pytest.raises(ValueError, match="overlapping resources"):
        arbiter.acquire_all(
            "overlap",
            (ResourceClaim(parent), ResourceClaim(child)),
        )

    lease = acquire(arbiter, "same-run", parent)
    with pytest.raises(RuntimeError, match="already owns"):
        arbiter.acquire_all("same-run", (ResourceClaim(key("device/fpga")),))
    lease._release_unarmed()
    arbiter.shutdown()


def test_shutdown_refuses_active_ownership_and_is_terminal() -> None:
    arbiter = ResourceArbiter()
    lease = acquire(arbiter, "run", key())
    with pytest.raises(RuntimeError, match="active ownership"):
        arbiter.shutdown()
    lease._release_unarmed()
    arbiter.shutdown()
    with pytest.raises(RuntimeError, match="shut down"):
        arbiter.acquire_all("late", (ResourceClaim(key()),))
