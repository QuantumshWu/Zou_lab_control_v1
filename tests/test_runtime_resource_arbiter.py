"""Current in-process resource ownership contracts."""

from __future__ import annotations

import threading

import pytest

from zlc_neutral_atom.runtime.resources import (
    ResourceArbiter,
    ResourceBusy,
    ResourceClaim,
    ResourceKey,
    ResourceLease,
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


def test_exact_device_claim_sets_are_atomic() -> None:
    arbiter = ResourceArbiter()
    camera = key("device/camera")
    fpga = key("device/fpga")
    held = acquire(arbiter, "writer", camera)

    result = arbiter.acquire_all(
        "all-or-nothing",
        (ResourceClaim(camera), ResourceClaim(fpga)),
    )
    assert isinstance(result, ResourceBusy)
    assert held.release()

    first = arbiter.acquire_all(
        "retry", (ResourceClaim(camera), ResourceClaim(fpga))
    )
    assert isinstance(first, ResourceLease)
    assert first.release()
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
    assert winners[0].release()
    arbiter.shutdown()


def test_release_is_an_exact_once_resource_transition() -> None:
    arbiter = ResourceArbiter()
    resource = key()
    lease = acquire(arbiter, "first", resource)

    assert lease.release()
    assert lease.released
    assert not lease.release()

    next_lease = acquire(arbiter, "next", resource)
    assert next_lease.release()
    arbiter.shutdown()


def test_invalid_claim_sets_and_duplicate_run_ids_are_rejected() -> None:
    arbiter = ResourceArbiter()
    camera = key("device/camera")
    with pytest.raises(ValueError, match="cannot request resource"):
        arbiter.acquire_all(
            "duplicate",
            (ResourceClaim(camera), ResourceClaim(camera)),
        )

    lease = acquire(arbiter, "same-run", camera)
    with pytest.raises(RuntimeError, match="already owns"):
        arbiter.acquire_all("same-run", (ResourceClaim(key("device/fpga")),))
    assert lease.release()
    arbiter.shutdown()


def test_shutdown_refuses_active_ownership_and_is_terminal() -> None:
    arbiter = ResourceArbiter()
    lease = acquire(arbiter, "run", key())
    with pytest.raises(RuntimeError, match="active ownership"):
        arbiter.shutdown()
    assert lease.release()
    arbiter.shutdown()
    with pytest.raises(RuntimeError, match="shut down"):
        arbiter.acquire_all("late", (ResourceClaim(key()),))
