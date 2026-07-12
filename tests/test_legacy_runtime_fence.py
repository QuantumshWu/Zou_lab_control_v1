from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import pytest
from conftest import raw_device_set

from zlc_neutral_atom.runtime import (
    CleanupStepAck,
    DeviceBroker,
    DeviceIdentityAck,
    DeviceIdentityEvidenceKind,
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
    InstallationAsset,
    InstallationAssetMap,
    LegacyDeviceNotRegistered,
    LegacyDeviceRegistration,
    LegacyDeviceRegistry,
    LegacyNodeAlreadyManaged,
    LegacyRuntimeFence,
    LegacyRuntimeTransition,
    LegacyStopStatus,
)
from zlc_workbench.asset_map import adapter_kind
from Zou_lab_control.neutral_atom import connect
from zlc_workbench.legacy_neutral_atom import LegacyNeutralAtomRuntime
from zlc_workbench.legacy_neutral_atom import registrations_for

LegacyRuntimeServices = LegacyNeutralAtomRuntime


def _asset_map_for(
    role: str,
    device: object,
    *,
    expected_identity: str,
    evidence_kind: DeviceIdentityEvidenceKind,
) -> InstallationAssetMap:
    return InstallationAssetMap(
        (
            InstallationAsset(
                asset_id=f"fixture-{role}",
                role=role,
                resource_key=ResourceKey.parse(f"device/{role}"),
                adapter_kind=adapter_kind(device),
                evidence_kind=evidence_kind,
                expected_identity=expected_identity,
            ),
        )
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
                    f"fixture:{device.name}",
                    DeviceIdentityEvidenceKind.HARDWARE_IDENTITY_READBACK,
                    f"fixture-evidence:{device.name}",
                    "test-assets-v1",
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


def test_lifecycle_reference_tracks_swap_without_creating_an_observe_claim():
    camera = _Device("camera")
    sequencer = _Device("sequencer")
    fence, _resources, _journal = _fence_for(camera, sequencer)

    class _WiredCameraNode(_Node):
        def __init__(self):
            super().__init__(camera)

        def lifecycle_devices(self):
            return (sequencer,)

    monitor = _WiredCameraNode()
    monitor_handle = fence.start(monitor)
    monitor_handle.wait_started(1.0)
    assert {claim.key for claim in monitor_handle.claims} == {
        ResourceKey.parse("device/camera")
    }
    assert set(monitor_handle.reference_keys) == {
        ResourceKey.parse("device/camera"),
        ResourceKey.parse("device/sequencer"),
    }

    sequencer_owner = _Node(sequencer)
    sequencer_handle = fence.start(sequencer_owner)
    sequencer_handle.wait_started(1.0)
    receipts = fence.stop_nodes_using((sequencer,), timeout=1.0)
    assert len(receipts) == 2
    assert all(receipt.terminated for receipt in receipts)


def test_real_observe_reference_still_conflicts_with_an_exclusive_owner():
    camera = _Device("camera")
    sequencer = _Device("sequencer")
    fence, _resources, _journal = _fence_for(camera, sequencer)

    class _ObserverNode(_Node):
        def __init__(self):
            super().__init__(camera)

        def referenced_devices(self):
            return (camera, sequencer)

    observer = _ObserverNode()
    handle = fence.start(observer)
    handle.wait_started(1.0)
    with pytest.raises(RunStartRejected):
        fence.start(_Node(sequencer))
    assert fence.stop(observer, timeout=1.0).terminated


def test_lifecycle_only_reference_blocks_generation_commit_until_termination():
    sequencer = _Device("sequencer")
    fence, _resources, _journal = _fence_for(sequencer)

    class _WiringOnlyNode(_Node):
        def __init__(self):
            super().__init__(None)

        def lifecycle_devices(self):
            return (sequencer,)

    node = _WiringOnlyNode()
    handle = fence.start(node)
    handle.wait_started(1.0)
    assert handle.claims == ()
    assert handle.reference_keys == (ResourceKey.parse("device/sequencer"),)

    transition = fence.begin_device_transition()
    with pytest.raises(RuntimeError, match="device-referencing Run is active"):
        fence.commit_device_transition(transition, fence.devices)
    assert fence.stop(node, timeout=1.0).terminated
    fence.commit_device_transition(transition, fence.devices)


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


def test_virtual_session_composition_binds_real_device_objects_and_safes_them():
    experiment = connect("virtual")
    services = experiment._require_runtime_services()
    node = _Node(raw_device_set(experiment).camera)
    try:
        assert services.persistent is False
        handle = services.fence.start(node)
        handle.wait_started(1.0)
        assert services.resources.active_claims()
        receipt = services.fence.stop(node, timeout=1.0)
        assert receipt.terminated
        assert receipt.snapshot is not None
        assert receipt.snapshot.safety_bundle_id is not None
        assert not services.resources.unresolved_hazards()
    finally:
        services.shutdown(timeout=1.0)
        raw_device_set(experiment).close()


def test_task_console_launcher_enrolls_passed_node_in_runtime_fence():
    from Zou_lab_control.frontend.task_console import show_task_console
    from Zou_lab_control.neutral_atom.core.signals import SignalHub
    from Zou_lab_control.frontend.qt_fluent import ensure_qt_app

    ensure_qt_app()
    experiment = connect("virtual")
    services = experiment._require_runtime_services()
    node = _Node(raw_device_set(experiment).camera)
    console = None
    try:
        console = show_task_console(
            hub=SignalHub(),
            running_nodes=[node],
            session=experiment,
            runtime_fence=services.fence,
            title="legacy-fence-enrollment-test",
        )
        handle = services.fence.handle_for(node)
        assert handle is not None
        handle.wait_started(1.0)
        assert node in console.running_nodes
        assert console.stop_all_nodes(timeout=1.0)
        assert handle.snapshot().state.terminal
    finally:
        if console is not None:
            console.shutdown(timeout=1.0)
            console.window().close()
        services.shutdown(timeout=1.0)
        raw_device_set(experiment).close()


def test_task_console_logic_start_and_stop_use_the_injected_runtime_fence():
    from Zou_lab_control.frontend.qt_fluent import ensure_qt_app
    from Zou_lab_control.frontend.task_console import TaskConsole, default_console_state
    from Zou_lab_control.neutral_atom.core.signals import SignalHub

    ensure_qt_app()
    experiment = connect("virtual")
    services = experiment._require_runtime_services()
    console = TaskConsole(
        hub=SignalHub(),
        state=default_console_state(),
        session=experiment,
        runtime_fence=services.fence,
        measurements=experiment.readout.measurement_specs(),
        processors=experiment.readout.processor_specs(),
        tasks=experiment.readout.task_specs(),
        window_px=(900, 600),
    )
    console._timer.stop()
    try:
        index = next(
            i
            for i in range(console.kind_combo.count())
            if console.kind_combo.itemData(i) == ("camera", "live")
        )
        console.kind_combo.setCurrentIndex(index)
        console._add_panel()
        row = console.logic_nodes[-1]
        console._start_logic_node(row)
        node = console._logic_nodes.get(id(row)) or console._starting_nodes[id(row)]
        handle = services.fence.handle_for(node)
        assert handle is not None
        handle.wait_started(1.0)
        console._poll_logic_nodes()
        assert console._logic_nodes[id(row)] is node
        assert node in console.running_nodes
        assert console._stop_logic_node(row, timeout=1.0)
        assert handle.snapshot().state.terminal
        assert node not in console.running_nodes
    finally:
        console.shutdown(timeout=1.0)
        services.shutdown(timeout=1.0)
        raw_device_set(experiment).close()


def test_session_task_console_composes_one_shared_runtime_authority():
    from Zou_lab_control.frontend.qt_fluent import ensure_qt_app

    ensure_qt_app()
    experiment = connect("virtual")
    console = experiment.task_console()
    services = experiment._require_runtime_services()
    try:
        assert console._current_runtime_fence() is services
        assert services.fence.controller is services.controller
        assert experiment.task_console() is console
    finally:
        console.shutdown(timeout=1.0)
        experiment.close()
        console.window().close()
    assert services.closed


def test_session_config_swap_rebinds_console_fence_after_old_owner_terminates():
    from Zou_lab_control.frontend.qt_fluent import ensure_qt_app

    ensure_qt_app()
    experiment = connect("virtual")
    console = experiment.task_console()
    services = experiment._require_runtime_services()
    try:
        index = next(
            i
            for i in range(console.kind_combo.count())
            if console.kind_combo.itemData(i) == ("camera", "live")
        )
        console.kind_combo.setCurrentIndex(index)
        console._add_panel()
        row = console.logic_nodes[-1]
        console._start_logic_node(row)
        old_node = console._logic_nodes.get(id(row)) or console._starting_nodes[id(row)]
        old_handle = services.fence.handle_for(old_node)
        assert old_handle is not None
        old_handle.wait_started(1.0)

        experiment.load_config("virtual")

        assert old_handle.snapshot().state.terminal
        console._poll_logic_nodes()
        assert console._logic_nodes[id(row)] is None
        assert console._current_runtime_fence() is experiment._require_runtime_services()
        console._start_logic_node(row)
        new_node = console._logic_nodes.get(id(row)) or console._starting_nodes[id(row)]
        assert new_node is not old_node
        new_handle = services.fence.handle_for(new_node)
        assert new_handle is not None
        new_handle.wait_started(1.0)
        assert console._stop_logic_node(row, timeout=1.0)
    finally:
        console.shutdown(timeout=1.0)
        experiment.close()
        console.window().close()


def test_experiment_config_cannot_replace_installation_adapter_kind(tmp_path, monkeypatch):
    from Zou_lab_control.frontend.qt_fluent import ensure_qt_app

    ensure_qt_app()
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    experiment = connect("virtual")
    console = experiment.task_console()
    authority = experiment._require_runtime_services()
    old_devices = raw_device_set(experiment)
    try:
        with pytest.raises(RuntimeError, match="requires adapter"):
            experiment.load_config(
                {
                    "sequencer": {
                        "type": "ManualSequencer",
                        "params": {"channels": ["ch00"]},
                    }
                }
            )
        assert experiment._require_runtime_services() is authority
        assert raw_device_set(experiment) is old_devices
        assert experiment.device_catalog.availability.value == "available"
        assert console._current_runtime_fence() is authority
    finally:
        console.shutdown(timeout=1.0)
        experiment.close()
        console.window().close()


def test_terminal_before_node_start_rolls_console_registry_back_without_ghost():
    from Zou_lab_control.frontend.qt_fluent import ensure_qt_app
    from Zou_lab_control.frontend.task_console import TaskConsole, default_console_state
    from Zou_lab_control.neutral_atom.core.signals import SignalHub

    ensure_qt_app()
    experiment = connect("virtual")
    services = experiment._require_runtime_services()
    console = TaskConsole(
        hub=SignalHub(),
        state=default_console_state(),
        session=experiment,
        runtime_fence=services.fence,
        window_px=(900, 600),
    )
    console._timer.stop()

    class _FailingNode:
        instance_label = ""

        def __init__(self, device):
            self.device = device
            self.running = False

        def occupied_devices(self):
            return (self.device,)

        def referenced_devices(self):
            return (self.device,)

        def published_signals(self):
            return frozenset({"never_published"})

        def start(self):
            raise RuntimeError("start boundary failed")

        def stop(self, timeout=0.1):
            return True

    try:
        index = next(
            i
            for i in range(console.kind_combo.count())
            if console.kind_combo.itemData(i) == ("camera", "live")
        )
        console.kind_combo.setCurrentIndex(index)
        console._add_panel()
        row = console.logic_nodes[-1]
        failing = _FailingNode(raw_device_set(experiment).camera)
        console._build_logic_node = lambda *_args, **_kwargs: failing
        console._start_logic_node(row)
        handle = services.fence.handle_for(failing)
        assert handle is not None
        assert handle.run_handle.wait(1.0).state is RunState.FAILED
        console._poll_logic_nodes()
        assert console._logic_nodes[id(row)] is None
        assert failing not in console.running_nodes
        assert "never_published" not in console.hub.names()
    finally:
        console.shutdown(timeout=1.0)
        services.shutdown(timeout=1.0)
        raw_device_set(experiment).close()


def test_slow_hazard_journal_keeps_starting_node_out_of_provider_registry():
    from Zou_lab_control.frontend.qt_fluent import ensure_qt_app
    from Zou_lab_control.frontend.task_console import TaskConsole, default_console_state
    from Zou_lab_control.neutral_atom.core.signals import SignalHub

    class BlockingJournal(MemoryQuarantineJournal):
        def __init__(self):
            super().__init__()
            self.entered = threading.Event()
            self.allow = threading.Event()

        def append_hazards(self, records):
            self.entered.set()
            assert self.allow.wait(2.0)
            return super().append_hazards(records)

    ensure_qt_app()
    experiment = connect("virtual")
    journal = BlockingJournal()
    resources = ResourceArbiter(journal)
    broker = DeviceBroker()
    registry = LegacyDeviceRegistry(broker)
    for registration in registrations_for(raw_device_set(experiment)):
        registry.register(registration)
    fence = LegacyRuntimeFence(RunController(resources), registry)
    console = TaskConsole(
        hub=SignalHub(),
        state=default_console_state(),
        session=experiment,
        runtime_fence=fence,
        window_px=(900, 600),
    )
    console._timer.stop()
    try:
        index = next(
            i
            for i in range(console.kind_combo.count())
            if console.kind_combo.itemData(i) == ("camera", "live")
        )
        console.kind_combo.setCurrentIndex(index)
        console._add_panel()
        row = console.logic_nodes[-1]
        console._start_logic_node(row)
        assert journal.entered.wait(1.0)
        node = console._starting_nodes[id(row)]
        assert console._logic_nodes[id(row)] is None
        assert node not in console.running_nodes
        assert console._node_for_signal("frame_0") is None

        journal.allow.set()
        fence.handle_for(node).wait_started(1.0)
        console._poll_logic_nodes()
        assert console._logic_nodes[id(row)] is node
        assert node in console.running_nodes
    finally:
        journal.allow.set()
        console.shutdown(timeout=1.0)
        experiment.close()
    resources.shutdown()


def test_synchronous_hardware_call_uses_same_hazard_and_safe_authority():
    device = _Device("camera")
    fence, resources, journal = _fence_for(device)
    observed = []

    def operation():
        observed.append(bool(journal.unresolved_hazards()))
        device.safe = False
        return "frame"

    assert fence.call((device,), operation, name="capture") == "frame"
    assert observed == [True]
    assert device.cleanup_calls == 1
    assert device.safe
    assert resources.active_claims() == {}
    resources.shutdown()


def test_session_hardware_call_establishes_closed_connection_before_identity_bound_run(
    monkeypatch,
):
    experiment = connect("virtual")
    camera = raw_device_set(experiment).camera
    state = {"open": False, "opens": 0}

    monkeypatch.setattr(
        type(camera),
        "is_open",
        property(lambda _self: state["open"]),
    )

    def open_camera():
        state["opens"] += 1
        state["open"] = True
        return camera

    monkeypatch.setattr(camera, "open", open_camera)
    try:
        assert not camera.is_open

        def operation():
            assert camera.is_open
            assert experiment._require_runtime_services().resources.active_claims()
            return "captured"

        assert experiment._run_hardware_call(
            (camera,), operation, name="lazy-open-capture"
        ) == "captured"
        assert state == {"open": True, "opens": 1}
        assert experiment._require_runtime_services().resources.active_claims() == {}
    finally:
        experiment.close()


def test_session_pulse_facade_holds_one_claim_from_prepare_through_stop():
    experiment = connect("virtual")
    try:
        controller = experiment.timing.bind_pulse(experiment.sequence)
        assert not hasattr(controller, "sequencer")
        controller.on_pulse(repeat_forever=True, wait=False)
        runtime = experiment._require_runtime_services()
        assert runtime.resources.active_claims()
        assert raw_device_set(experiment).sequencer.snapshot()["state"] == "running"
        controller.stop()
        assert runtime.resources.active_claims() == {}
        assert raw_device_set(experiment).sequencer.snapshot()["state"] == "safe"
    finally:
        experiment.close()


def test_quarantine_survives_registry_replacement_and_blocks_same_resource():
    first = _Device("camera")
    journal = MemoryQuarantineJournal()
    resources = ResourceArbiter(journal)
    broker = DeviceBroker()

    def registry_for(device, *, verify_fails=False):
        registry = LegacyDeviceRegistry(broker)
        registry.register(
            LegacyDeviceRegistration(
                device=device,
                key=ResourceKey(("device", "camera")),
                identity_probe=lambda: DeviceIdentityAck(
                    "fixture:camera",
                    DeviceIdentityEvidenceKind.HARDWARE_IDENTITY_READBACK,
                    f"fixture-evidence:{device.name}",
                    "test-assets-v1",
                ),
                cleanup_operations={
                    SafetyOperation.SAFE_STATE: lambda: CleanupStepAck(
                        SafetyOperation.SAFE_STATE, "safe-requested"
                    )
                },
                cleanup_order=(SafetyOperation.SAFE_STATE,),
                verify_safe_state=(
                    (lambda: (_ for _ in ()).throw(RuntimeError("readback failed")))
                    if verify_fails
                    else (lambda: SafeStateAck("safe-verified"))
                ),
            )
        )
        return registry

    fence = LegacyRuntimeFence(
        RunController(resources), registry_for(first, verify_fails=True)
    )
    node = _Node(first)
    handle = fence.start(node)
    handle.wait_started(1.0)
    receipt = fence.stop(node, timeout=1.0)
    assert receipt.snapshot is not None
    assert receipt.snapshot.state is RunState.FAILED
    assert resources.quarantine_records()

    replacement = _Device("replacement")
    transition = fence.begin_device_transition()
    fence.commit_device_transition(transition, registry_for(replacement))
    with pytest.raises(RunStartRejected) as rejected:
        fence.start(_Node(replacement))
    assert "ResourceQuarantined" in repr(rejected.value.outcome)


def test_production_console_reports_device_conflict_without_stopping_first_owner():
    from Zou_lab_control.frontend.qt_fluent import ensure_qt_app
    from Zou_lab_control.frontend.task_console import TaskConsole, default_console_state
    from Zou_lab_control.neutral_atom.core.signals import SignalHub

    ensure_qt_app()
    experiment = connect("virtual")
    services = experiment._require_runtime_services()
    console = TaskConsole(
        hub=SignalHub(),
        state=default_console_state(),
        session=experiment,
        runtime_fence=services.fence,
        window_px=(900, 600),
    )
    console._timer.stop()

    def add_camera_row():
        index = next(
            i
            for i in range(console.kind_combo.count())
            if console.kind_combo.itemData(i) == ("camera", "live")
        )
        console.kind_combo.setCurrentIndex(index)
        console._add_panel()
        return console.logic_nodes[-1]

    try:
        first_row = add_camera_row()
        console._start_logic_node(first_row)
        first = (
            console._logic_nodes.get(id(first_row))
            or console._starting_nodes[id(first_row)]
        )
        first_handle = services.fence.handle_for(first)
        assert first_handle is not None
        first_handle.wait_started(1.0)
        console._poll_logic_nodes()

        second_row = add_camera_row()
        console._start_logic_node(second_row)

        assert console._logic_nodes.get(id(second_row)) is None
        assert first in console.running_nodes
        assert first_handle.snapshot().state is RunState.RUNNING
        assert "start failed" in second_row.status_label.text().lower()
        assert console._stop_logic_node(first_row, timeout=1.0)
    finally:
        console.shutdown(timeout=1.0)
        services.shutdown(timeout=1.0)
        raw_device_set(experiment).close()


def test_device_transition_closes_start_admission_until_commit_or_abort():
    device = _Device("camera")
    replacement = _Device("replacement")
    fence, _resources, _journal = _fence_for(device)
    transition = fence.begin_device_transition()

    with pytest.raises(LegacyRuntimeTransition, match="start is closed"):
        fence.start(_Node(device))

    fence.abort_device_transition(transition)
    handle = fence.start(_Node(device))
    handle.wait_started(1.0)
    assert fence.stop(handle.node, timeout=1.0).terminated

    transition = fence.begin_device_transition()
    replacement_fence, _resources, _journal = _fence_for(replacement)
    fence.commit_device_transition(transition, replacement_fence.devices)
    replacement_handle = fence.start(_Node(replacement))
    replacement_handle.wait_started(1.0)
    assert fence.stop(replacement_handle.node, timeout=1.0).terminated


def test_session_swap_gate_blocks_concurrent_start_through_old_generation():
    experiment = connect("virtual")
    runtime = experiment._require_runtime_services()
    node = _Node(raw_device_set(experiment).camera)
    handle = runtime.fence.start(node)
    handle.wait_started(1.0)
    old_devices = raw_device_set(experiment)
    real_close = old_devices.close
    close_entered = threading.Event()
    allow_close = threading.Event()
    failures = []

    def blocked_close():
        close_entered.set()
        assert allow_close.wait(2.0)
        return real_close()

    def swap():
        try:
            experiment.load_config("virtual")
        except BaseException as exc:
            failures.append(exc)

    old_devices.close = blocked_close
    worker = threading.Thread(target=swap)
    worker.start()
    assert close_entered.wait(1.0)
    with pytest.raises(LegacyRuntimeTransition, match="start is closed"):
        runtime.fence.start(_Node(old_devices.camera))
    allow_close.set()
    worker.join(3.0)

    assert not worker.is_alive()
    assert failures == []
    assert handle.snapshot().state.terminal
    experiment.close()


def test_manual_sequencer_registration_has_claim_but_no_hazard_contract():
    from Zou_lab_control.neutral_atom.devices.sequencer import ManualSequencer

    sequencer = ManualSequencer(channels=("ch00",))
    registration = registrations_for(
        SimpleNamespace(
            devices={"sequencer": sequencer},
            config={
                "sequencer": {
                    "type": "ManualSequencer",
                    "params": {"channels": ["ch00"]},
                }
            },
        )
    )[0]

    assert registration.hazardous is False
    assert registration.cleanup_operations == {}
    assert registration.verify_safe_state is None


def test_remote_sequencer_is_no_go_before_raw_hardware_safe_contract_exists():
    from Zou_lab_control.neutral_atom.devices.sequencer import RemoteSequencer

    sequencer = RemoteSequencer(host="127.0.0.1", port=18861)
    asset_map = _asset_map_for(
        "sequencer",
        sequencer,
        expected_identity="installation-endpoint:remote-sequencer:127.0.0.1:18861",
        evidence_kind=DeviceIdentityEvidenceKind.INSTALLATION_ASSERTED_ENDPOINT,
    )
    registration = registrations_for(
        SimpleNamespace(
            devices={"sequencer": sequencer},
            config={
                "sequencer": {
                    "type": "RemoteSequencer",
                    "params": {"host": "127.0.0.1", "port": 18861},
                }
            },
        ),
        asset_map=asset_map,
    )[0]
    broker = DeviceBroker()
    registry = LegacyDeviceRegistry(broker)
    registry.register(registration)
    fence = LegacyRuntimeFence(
        RunController(ResourceArbiter(MemoryQuarantineJournal())), registry
    )

    with pytest.raises(RuntimeError, match="NO-GO"):
        fence.start(_Node(sequencer))


@pytest.mark.parametrize(
    "field",
    ("resource_key", "stable_device_identity", "asset_map_revision"),
)
def test_experiment_config_cannot_override_installation_identity(field):
    experiment = connect("virtual")
    try:
        raw_device_set(experiment).config["camera"][field] = "forged"
        with pytest.raises(ValueError, match="installation-owned identity"):
            registrations_for(raw_device_set(experiment))
    finally:
        raw_device_set(experiment).close()


def test_unknown_concrete_camera_adapter_is_rejected_at_composition():
    from Zou_lab_control.neutral_atom.devices.virtual import VirtualCamera

    class UnqualifiedVirtualCamera(VirtualCamera):
        pass

    camera = object.__new__(UnqualifiedVirtualCamera)
    with pytest.raises(TypeError, match="no explicit identity/safety adapter"):
        registrations_for(
            SimpleNamespace(
                devices={"camera": camera},
                config={
                    "camera": {
                        "type": "UnqualifiedVirtualCamera",
                        "params": {},
                    }
                },
            )
        )


def test_unknown_non_camera_device_subclass_is_rejected_at_composition():
    from Zou_lab_control.neutral_atom.devices.virtual import VirtualLaser

    class UnqualifiedLaser(VirtualLaser):
        pass

    laser = UnqualifiedLaser()
    with pytest.raises(TypeError, match="no explicit identity/safety adapter"):
        registrations_for(
            SimpleNamespace(
                devices={"cooling_laser": laser},
                config={"cooling_laser": {"type": "UnqualifiedLaser", "params": {}}},
            )
        )


def test_virtual_device_set_has_explicit_registration_for_every_base_device():
    experiment = connect("virtual")
    try:
        registrations = registrations_for(raw_device_set(experiment))
        assert len(registrations) == len(raw_device_set(experiment).devices)
        assert {id(registration.device) for registration in registrations} == {
            id(device) for device in raw_device_set(experiment).devices.values()
        }
    finally:
        raw_device_set(experiment).close()


def test_real_adapter_requires_installation_asset_map(tmp_path, monkeypatch):
    from Zou_lab_control.neutral_atom.devices.qcmos import QCMOSCamera

    monkeypatch.setenv("ZLC_ASSET_MAP", str(tmp_path / "missing-asset-map.json"))
    camera = QCMOSCamera()
    with pytest.raises(RuntimeError, match="requires an installation AssetMap"):
        registrations_for(
            SimpleNamespace(
                devices={"camera": camera},
                config={"camera": {"type": "QCMOSCamera", "params": {}}},
            )
        )


def test_asset_map_revision_is_canonical_content_digest():
    from Zou_lab_control.neutral_atom.devices.virtual import VirtualLaser

    camera = VirtualLaser()
    first = _asset_map_for(
        "camera",
        camera,
        expected_identity="installation-endpoint:virtual:camera",
        evidence_kind=DeviceIdentityEvidenceKind.INSTALLATION_ASSERTED_ENDPOINT,
    )
    replay = InstallationAssetMap.from_mapping(first.to_dict())
    changed = _asset_map_for(
        "camera",
        camera,
        expected_identity="installation-endpoint:virtual:other-camera",
        evidence_kind=DeviceIdentityEvidenceKind.INSTALLATION_ASSERTED_ENDPOINT,
    )
    assert replay.revision == first.revision
    assert changed.revision != first.revision


class _PylonNode:
    def __init__(self, value):
        self.value = value

    def GetValue(self):
        return self.value


class _PylonLiveHandle:
    def __init__(self, serial="PX-1", model="acA1920"):
        self.DeviceSerialNumber = _PylonNode(serial)
        self.DeviceModelName = _PylonNode(model)
        self.removed = False
        self.open = True
        self.grabbing = False

    def IsOpen(self):
        return self.open

    def IsCameraDeviceRemoved(self):
        return self.removed

    def IsGrabbing(self):
        return self.grabbing


def _qualified_pylon_registration():
    from Zou_lab_control.neutral_atom.devices.pylon import PylonCamera

    camera = PylonCamera(serial="PX-1", trigger_source="Software")
    camera._camera = _PylonLiveHandle()
    camera._connection_token = object()
    asset_map = _asset_map_for(
        "monitor_camera",
        camera,
        expected_identity="pylon:PX-1",
        evidence_kind=DeviceIdentityEvidenceKind.HARDWARE_IDENTITY_READBACK,
    )
    registration = registrations_for(
        SimpleNamespace(
            devices={"monitor_camera": camera},
            config={
                "monitor_camera": {
                    "type": "PylonCamera",
                    "params": {"serial": "PX-1", "trigger_source": "Software"},
                }
            },
        ),
        asset_map=asset_map,
    )[0]
    return camera, registration


def test_pylon_removed_handle_cannot_reuse_cached_identity_or_mint_safe():
    camera, registration = _qualified_pylon_registration()
    identity = registration.identity_probe()
    assert identity.stable_device_identity == "pylon:PX-1"
    camera._camera.removed = True
    with pytest.raises(RuntimeError, match="removal|removed"):
        registration.identity_probe()
    with pytest.raises(RuntimeError, match="removal|removed"):
        registration.verify_safe_state()


def test_camera_cleanup_attempts_stop_after_disarm_failure():
    camera, registration = _qualified_pylon_registration()
    calls = []

    def fail_disarm():
        calls.append("disarm")
        raise RuntimeError("disarm failed")

    def stop():
        calls.append("stop")

    camera.disarm = fail_disarm
    camera.stop = stop
    with pytest.raises(RuntimeError, match="cleanup did not acknowledge every step"):
        registration.cleanup_operations[SafetyOperation.DISARM]()
    assert calls == ["disarm", "stop"]


def test_device_bearing_logic_node_direct_start_is_mechanically_rejected():
    from Zou_lab_control.neutral_atom.core.signals import SignalHub
    from Zou_lab_control.neutral_atom.operations.logic import CameraMeasurement

    experiment = connect("virtual")
    node = CameraMeasurement(SignalHub(), raw_device_set(experiment).camera)
    try:
        with pytest.raises(RuntimeError, match="LegacyRuntimeFence authority"):
            node.start()
        assert not node.running
    finally:
        raw_device_set(experiment).close()


def test_observe_only_logic_node_direct_start_is_mechanically_rejected():
    from Zou_lab_control.neutral_atom.core.signals import SignalHub
    from Zou_lab_control.neutral_atom.operations.logic import LogicNode, OBSERVE

    class Observer(LogicNode):
        _devices = {"camera": OBSERVE}

        def __init__(self, camera):
            super().__init__(SignalHub())
            self.camera = camera

        def shot(self):
            return {}

    experiment = connect("virtual")
    node = Observer(raw_device_set(experiment).camera)
    try:
        assert node.occupied_devices() == ()
        assert node.referenced_devices() == (raw_device_set(experiment).camera,)
        with pytest.raises(RuntimeError, match="LegacyRuntimeFence authority"):
            node.start()
        assert not node.running
    finally:
        raw_device_set(experiment).close()
