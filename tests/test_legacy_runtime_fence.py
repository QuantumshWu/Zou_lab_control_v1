from __future__ import annotations

import threading
import time

import pytest

from zlc_neutral_atom.runtime import (
    CleanupStepAck,
    DeviceBroker,
    DeviceIdentityAck,
    MemoryQuarantineJournal,
    ResourceArbiter,
    ResourceKey,
    RunController,
    RunStartRejected,
    RunState,
    SafeStateAck,
    SafetyOperation,
)
from zlc_workbench import (
    LegacyDeviceNotRegistered,
    LegacyDeviceRegistration,
    LegacyDeviceRegistry,
    LegacyNodeAlreadyManaged,
    LegacyRuntimeFence,
    LegacyStopStatus,
)


class _Device:
    def __init__(self, name: str) -> None:
        self.name = name
        self.safe = True
        self.cleanup_calls = 0


class _Node:
    def __init__(
        self,
        device: _Device | None,
        *,
        journal: MemoryQuarantineJournal | None = None,
        helper_gate: threading.Event | None = None,
        ignore_stop_attempts: int = 0,
    ) -> None:
        self.device = device
        self.journal = journal
        self.helper_gate = helper_gate
        self.ignore_stop_attempts = ignore_stop_attempts
        self.stop_calls = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.started_after_hazard = False

    def occupied_devices(self):
        return () if self.device is None else (self.device,)

    def referenced_devices(self):
        return self.occupied_devices()

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self):
        if self.journal is not None:
            self.started_after_hazard = bool(self.journal.unresolved_hazards())
            assert self.started_after_hazard
        if self.device is not None:
            self.device.safe = False
        self._stop.clear()

        def work() -> None:
            while not self._stop.wait(0.005):
                pass

        self._thread = threading.Thread(target=work, daemon=False)
        self._thread.start()
        return self

    def stop(self, timeout: float = 0.1) -> bool:
        self.stop_calls += 1
        if self.stop_calls <= self.ignore_stop_attempts:
            return False
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout)
        if self.helper_gate is not None:
            self.helper_gate.wait()
        if thread is not None and thread.is_alive():
            return False
        self._thread = None
        return True


def _fence_for(*devices: _Device):
    journal = MemoryQuarantineJournal()
    resources = ResourceArbiter(journal)
    broker = DeviceBroker()
    registry = LegacyDeviceRegistry(broker)
    for device in devices:
        key = ResourceKey(("device", device.name))

        def cleanup(device=device):
            device.cleanup_calls += 1
            device.safe = True
            return CleanupStepAck(SafetyOperation.SAFE_STATE, f"safe:{device.name}")

        def verify(device=device):
            if not device.safe:
                raise RuntimeError(f"{device.name} is not safe")
            return SafeStateAck(f"verified:{device.name}")

        registry.register(
            LegacyDeviceRegistration(
                device=device,
                key=key,
                identity_probe=lambda device=device: DeviceIdentityAck(
                    f"fixture:{device.name}", "generation-1"
                ),
                cleanup_operations={SafetyOperation.SAFE_STATE: cleanup},
                cleanup_order=(SafetyOperation.SAFE_STATE,),
                verify_safe_state=verify,
            )
        )
    return LegacyRuntimeFence(RunController(resources), registry), resources, journal


def test_hazard_is_durable_before_legacy_node_start_and_safe_before_release():
    device = _Device("camera")
    fence, resources, journal = _fence_for(device)
    node = _Node(device, journal=journal)

    handle = fence.start(node)
    started = handle.wait_started(1.0)
    assert started.node_type == "_Node"
    assert node.started_after_hazard
    assert resources.active_claims()

    receipt = fence.stop(node, timeout=1.0)
    assert receipt.status is LegacyStopStatus.TERMINATED
    assert receipt.snapshot is not None
    assert receipt.snapshot.state is RunState.CANCELLED
    assert device.cleanup_calls == 1
    assert device.safe
    assert not resources.active_claims()
    assert not journal.unresolved_hazards()


def test_stop_timeout_keeps_claim_while_stop_helper_is_still_alive():
    device = _Device("camera")
    fence, resources, _journal = _fence_for(device)
    gate = threading.Event()
    node = _Node(device, helper_gate=gate)
    handle = fence.start(node)
    handle.wait_started(1.0)

    pending = fence.stop(node, timeout=0.02)
    assert pending.status is LegacyStopStatus.PENDING
    assert pending.snapshot is not None
    assert pending.snapshot.state is RunState.CANCELLING
    assert resources.active_claims()

    gate.set()
    terminal = handle.run_handle.wait(1.0)
    assert terminal.state is RunState.CANCELLED
    assert not resources.active_claims()


def test_serial_stop_retries_do_not_release_for_false_result_while_owner_lives():
    device = _Device("camera")
    fence, resources, _journal = _fence_for(device)
    node = _Node(device, ignore_stop_attempts=2)
    # Do not expose the conventional event to execute(), so only serial stop attempts
    # can make progress after cancellation.
    stop_event = node._stop
    del node._stop

    def start_without_public_stop_event():
        device.safe = False

        def work() -> None:
            while not stop_event.wait(0.005):
                pass

        node._thread = threading.Thread(target=work, daemon=False)
        node._thread.start()
        return node

    def stop_after_retries(timeout=0.1):
        node.stop_calls += 1
        if node.stop_calls <= node.ignore_stop_attempts:
            return False
        stop_event.set()
        node._thread.join(timeout)
        if node._thread.is_alive():
            return False
        node._thread = None
        return True

    node.start = start_without_public_stop_event
    node.stop = stop_after_retries
    handle = fence.start(node)
    handle.wait_started(1.0)
    receipt = fence.stop(node, timeout=1.0)

    assert receipt.terminated
    assert node.stop_calls == 3
    assert not resources.active_claims()


def test_conflicting_legacy_nodes_are_rejected_by_shared_resource_arbiter():
    device = _Device("camera")
    fence, _resources, _journal = _fence_for(device)
    first = _Node(device)
    second = _Node(device)
    first_handle = fence.start(first)
    first_handle.wait_started(1.0)

    with pytest.raises(RunStartRejected):
        fence.start(second)

    assert fence.stop(first, timeout=1.0).terminated
    second_handle = fence.start(second)
    second_handle.wait_started(1.0)
    assert fence.stop(second, timeout=1.0).terminated


def test_unregistered_or_already_running_nodes_fail_closed():
    device = _Device("camera")
    fence, _resources, _journal = _fence_for()
    node = _Node(device)
    with pytest.raises(LegacyDeviceNotRegistered):
        fence.start(node)

    node.start()
    try:
        with pytest.raises(LegacyNodeAlreadyManaged):
            fence.start(node)
        with pytest.raises(LegacyNodeAlreadyManaged):
            fence.stop(node)
    finally:
        node.stop()


def test_device_free_legacy_processor_uses_same_run_lifecycle_without_hazard():
    fence, resources, journal = _fence_for()
    node = _Node(None)
    handle = fence.start(node)
    try:
        handle.wait_started(1.0)
        assert all(not claims for claims in resources.active_claims().values())
        assert not journal.unresolved_hazards()
    finally:
        receipt = fence.stop(node, timeout=1.0)
    assert receipt.terminated
