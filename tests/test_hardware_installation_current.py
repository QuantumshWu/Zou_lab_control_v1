"""Contract tests for the real-installation composition using explicit fakes.

These tests prove wiring and active-E0 mechanics only.  They make no claim
about real-camera loss statistics or a deployed FPGA.
"""

from __future__ import annotations

from dataclasses import replace
import math
import threading
import time

import numpy as np
import pytest

from conftest import pulse_backend_completion_for
from zlc_neutral_atom.devices.camera.contract import (
    CameraCaptureTerminalRecord,
    CameraFrameRecord,
    CameraWorkingPoint,
)
from zlc_neutral_atom.device_types import (
    CAPABILITY_CAMERA_CAPTURE,
    CAPABILITY_CAMERA_MONITOR,
    CAPABILITY_PULSE_EXECUTE,
)
from zlc_neutral_atom.devices.camera import device_types as camera_device_types
from zlc_neutral_atom.devices.camera.pylon import PylonCameraAdapter, PylonCameraConfig
from zlc_neutral_atom.devices.sequencer.port import (
    PulseTerminalAck,
    SimulatedPulseReceipt,
)
from zlc_neutral_atom.installation_config import installation_template
from zlc_neutral_atom.installation_runtime import create_installation
from zlc_neutral_atom.logic_nodes.camera_measurement.definition import (
    CameraMeasurementRequest,
)
from zlc_neutral_atom.logic_nodes.camera_measurement.monitor import (
    CameraMonitorViewSpec,
    open_live_camera_measurement,
)
from zlc_neutral_atom.runtime.resources import DeviceIdentityEvidenceKind
from zlc_neutral_atom.runtime.signal_source import SignalAssociationRequest
from zlc_pulse import (
    PreparedPulseRef,
    PulseCompletion,
    PulseDocument,
    PulseExecutionForm,
    PulsePeriod,
    PulseServerSnapshot,
    RemotePulseExecutionClient,
    compile_pulse_artifact,
    load_deployed_geometry_facts,
    load_deployed_pulse_target,
    pulse_target_manifest_from_lanes,
)


def _four_trigger_document(
    *,
    client: RemotePulseExecutionClient,
    trigger_lane: str,
    required_interval_seconds: float,
) -> PulseDocument:
    """Build the public Pulse witness required by association contract tests."""

    snapshot = client.snapshot()
    target = snapshot.target
    lane_index = target.raw_lanes.index(trigger_lane)
    owner = next(port for port in target.ports if trigger_lane in port.lanes)
    tick_ns = 1e9 / snapshot.clock_hz
    high_ticks = 1
    interval_ticks = max(
        high_ticks + 1,
        math.ceil(required_interval_seconds * snapshot.clock_hz),
    )
    low = tuple(0 for _ in target.raw_lanes)
    high_values = list(low)
    high_values[lane_index] = 1
    high = tuple(high_values)
    periods = [PulsePeriod("initial_safe", tick_ns, "ns", "safe", low)]
    for index in range(4):
        periods.extend(
            (
                PulsePeriod(
                    f"trigger_{index}", tick_ns, "ns", "camera trigger", high
                ),
                PulsePeriod(
                    f"safe_{index}",
                    (interval_ticks - high_ticks) * tick_ns,
                    "ns",
                    "safe interval",
                    low,
                ),
            )
        )
    return PulseDocument(
        name=f"four-trigger {trigger_lane} association witness",
        target=target,
        time_step_ns=tick_ns,
        periods=tuple(periods),
        visible_ports=(owner.key,),
    )


class _TriggerBus:
    def __init__(self) -> None:
        self.cameras = {}
        self.fire_count = 0

    def emit(self, artifact) -> None:
        self.fire_count += 1
        for schedule in artifact.trigger_schedules:
            self.cameras[schedule.channel].emit(schedule.total)


class _FakeCamera:
    def __init__(self, config, bus: _TriggerBus) -> None:
        self.config = config
        self.lane = config.capture_trigger_channels[0]
        self.bus = bus
        bus.cameras[self.lane] = self
        self.armed = False
        self.records = []
        self.delivered = 0
        self.arm_calls = []
        self.closed = False
        self.hardware_stamps_enabled = True
        self.read_gate = threading.Event()
        self.read_gate.set()
        self.blocked_read_started = threading.Event()

    @property
    def timeout(self):
        return 1.0

    def capture_working_point(self):
        return CameraWorkingPoint(
            "EXTERNAL_TRIGGERED",
            (4, 5),
            (4, 5),
            (0, 0),
            (4, 5),
            (1, 1),
            np.dtype("u1"),
            "count",
            (self.lane,),
            0.001,
            0.0001,
            None,
            1.0,
            "fake-current-camera",
        )

    def configure_exposure_seconds(self, exposure_seconds):
        return None

    def arm(self, frames, *, source_group_sizes, buffer_frame_count, timeout):
        self.arm_calls.append((frames, source_group_sizes, buffer_frame_count))
        self.armed = True
        self.records = []
        self.delivered = 0

    def emit(self, count):
        assert self.armed
        start = len(self.records)
        self.records.extend(
            CameraFrameRecord(
                np.full((4, 5), source_ordinal, dtype=np.uint8),
                source_ordinal,
                source_ordinal + 1,
                100 + source_ordinal if self.hardware_stamps_enabled else None,
                200 + source_ordinal if self.hardware_stamps_enabled else None,
                None,
                None,
                time.time_ns() + source_ordinal,
            )
            for source_ordinal in range(start, start + count)
        )

    def read_frame_records(self, n, *, timeout, exact):
        if not self.read_gate.is_set():
            self.blocked_read_started.set()
            self.read_gate.wait(timeout)
        result = self.records[self.delivered : self.delivered + n]
        self.delivered += len(result)
        if exact and len(result) != n:
            raise TimeoutError("fake camera short read")
        return result

    def finish_record_capture(self):
        self.armed = False
        return CameraCaptureTerminalRecord(self.delivered, True, True, True)

    def capture_state(self):
        return self.armed, len(self.records)

    def observed_produced_count(self):
        if not self.armed:
            raise RuntimeError("fake camera is not armed")
        return len(self.records)

    def close(self):
        self.closed = True


class _FakeRemoteClient(RemotePulseExecutionClient):
    def __init__(self, bus: _TriggerBus) -> None:
        self.bus = bus
        self._target = load_deployed_pulse_target()
        self._manifest = pulse_target_manifest_from_lanes(self._target)
        self._generation = "fake-current-generation"
        self._state = "SAFE"
        self._prepared = None
        self._closed = False

    @property
    def transport_timeout_seconds(self):
        return 10.0

    @property
    def connection_generation(self):
        return self._generation

    def snapshot(self):
        geometry = load_deployed_geometry_facts()
        safe = self._state == "SAFE"
        prepared_digest = (
            None if safe or self._prepared is None else self._prepared.fingerprint
        )
        return PulseServerSnapshot(
            connection_generation=self._generation,
            manifest=self._manifest,
            clock_hz=geometry.clock_hz,
            geometry_fingerprint=geometry.geometry_fingerprint,
            state=self._state,
            prepared_ref=None,
            physical_state="SAFE" if safe else self._state,
            physical_prepared_artifact_digest=prepared_digest,
            physical_scan_point_count=(
                0
                if self._prepared is None
                else len(self._prepared.target_ir.scan_points)
            ),
            physical_scan_cursor=None,
            physical_cursor_sample_count=0,
            physical_underflow_observed=False,
            safe_status_word=0 if safe else None,
            safe_clock_enable_words=(
                tuple(
                    0
                    for _ in range(
                        (len(self._manifest.target.raw_lanes) + 31) // 32
                    )
                )
                if safe
                else None
            ),
        )

    def prepare(self, artifact):
        self._prepared = artifact
        self._state = "PREPARED"
        return PreparedPulseRef(self._generation, artifact.fingerprint)

    def fire(self, reference):
        assert reference.artifact_digest == self._prepared.fingerprint
        self.bus.emit(self._prepared)
        self._state = "RUNNING"

    def complete(self, reference, *, timeout):
        backend = pulse_backend_completion_for(self._prepared, transport_id="fake-e0")
        counts = tuple((item.channel, item.total) for item in self._prepared.trigger_schedules)
        self._state = "DONE"
        return PulseCompletion(
            reference,
            backend.hardware_terminal,
            backend.post_terminal_tail,
            counts,
        )

    def safe_state(self, *, timeout=None):
        self._state = "SAFE"
        self._prepared = None
        return self.snapshot()

    def close(self):
        self._closed = True


class _GapCamera(_FakeCamera):
    def emit(self, count):
        super().emit(count)
        self.records = [
            CameraFrameRecord(
                record.image,
                record.source_ordinal,
                record.produced_count,
                100 + record.source_ordinal * 2,
                record.camera_stamp,
                record.timestamp_seconds,
                record.timestamp_microseconds,
                record.host_received_at_ns,
            )
            for record in self.records
        ]


def _hardware_document():
    return installation_template(
        "hardware",
        host="test-host",
        serial="test-basler",
    )


def _patch_hardware_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    client: _FakeRemoteClient,
    camera_factory,
) -> None:
    monkeypatch.setattr(
        RemotePulseExecutionClient,
        "connect",
        lambda *_args, **_kwargs: client,
    )
    monkeypatch.setattr(
        camera_device_types,
        "DcamCameraAdapter",
        camera_factory,
    )
    monkeypatch.setattr(
        camera_device_types,
        "PylonCameraAdapter",
        camera_factory,
    )


class _LiveView:
    def __init__(self, spec: CameraMonitorViewSpec) -> None:
        self.spec = spec
        self.dataset = None
        self.failure = None

    def bind(self, dataset):
        self.dataset = dataset

    def updated(self):
        return None

    def notification_failed(self, message):
        self.failure = message

    def fail(self, message):
        self.failure = message

    def source_terminal(self):
        return None


def test_fake_real_installation_runs_both_active_e0_paths_and_binds_ports(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bus = _TriggerBus()
    client = _FakeRemoteClient(bus)
    cameras = []

    def camera_factory(config):
        camera = _FakeCamera(config, bus)
        cameras.append(camera)
        return camera

    _patch_hardware_dependencies(monkeypatch, client, camera_factory)
    composition = create_installation(_hardware_document())
    try:
        assert bus.fire_count == 2
        assert [camera.lane for camera in cameras] == ["ch11", "ch06"]
        assert all(camera.arm_calls == [(4, (4,), 4)] for camera in cameras)
        catalog = composition.runtime.device_catalog
        assert catalog.roles("camera") == ("camera", "mot_camera")
        physical_identities = {}
        for instance_id in ("camera", "mot-camera"):
            reference = catalog.require(instance_id).ref
            port = composition.runtime.require_capability(
                reference,
                CAPABILITY_CAMERA_CAPTURE,
            )
            physical_identities[instance_id] = (
                port.capability.binding_stamp.physical_identity
            )
            evidence = port.capability.camera_capability_evidence
            assert evidence.exact_external_trigger_qualified
            assert port.capability.payload_contract.value_schema.dtype == np.dtype("u1")
        pulse_port = composition.runtime.require_capability(
            catalog.require("sequencer").ref,
            CAPABILITY_PULSE_EXECUTE,
        )
        physical_identities["sequencer"] = (
            pulse_port.device.binding_stamp.physical_identity
        )
        assert {
            role: identity.stable_device_identity
            for role, identity in physical_identities.items()
        } == {
            "camera": "dcam-device-index:0",
            "mot-camera": "pylon-serial:test-basler",
            "sequencer": "remote-pulse-endpoint:test-host:18861",
        }
        assert {
            identity.evidence_kind for identity in physical_identities.values()
        } == {DeviceIdentityEvidenceKind.INSTALLATION_ASSERTED_ENDPOINT}
        assert tuple(catalog) == ("sequencer", "camera", "mot-camera")
        assert catalog.require("camera").resource_key == "device/camera"
        assert catalog.require("mot-camera").resource_key == "device/mot-camera"
        assert composition.readout_installation_bindings[0].trigger_channel == "ch11"
        assert composition.readout_installation_bindings[0].camera_instance_id == "camera"
        assert (
            composition.readout_installation_bindings[0].sequencer_instance_id
            == "sequencer"
        )
        assert tuple(
            role for role, _authority in composition.camera_signal_association_authorities
        ) == ("camera",)
    finally:
        assert composition.runtime.shutdown(timeout=2.0)
    assert client._closed
    assert all(camera.closed for camera in cameras)


def test_real_installation_rejects_an_e0_hardware_stamp_gap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bus = _TriggerBus()
    client = _FakeRemoteClient(bus)
    cameras = []

    def gap_factory(config):
        camera = _GapCamera(config, bus)
        cameras.append(camera)
        return camera

    _patch_hardware_dependencies(monkeypatch, client, gap_factory)
    with pytest.raises(RuntimeError, match="frame_stamp has a gap"):
        create_installation(_hardware_document())
    assert client._closed
    assert cameras and all(camera.closed for camera in cameras)


def test_fake_real_camera_signal_association_uses_hardware_terminal_and_ordinals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise the real composition's sole Camera->PulseScan evidence path."""

    bus = _TriggerBus()
    client = _FakeRemoteClient(bus)
    cameras = []

    def camera_factory(config):
        camera = _FakeCamera(config, bus)
        cameras.append(camera)
        return camera

    _patch_hardware_dependencies(monkeypatch, client, camera_factory)
    composition = create_installation(_hardware_document())
    runtime = composition.runtime
    camera_ref = runtime.device_catalog.require("camera").ref
    authority = dict(composition.camera_signal_association_authorities)["camera"]
    sources = []
    views = []

    def open_dataset(spec, *, event_source, **_kwargs):
        view = _LiveView(spec)
        views.append(view)
        sources.append(event_source)
        return view

    plan = open_live_camera_measurement(
        CameraMeasurementRequest("camera", repeat=0),
        monitor_port=runtime.require_capability(
            camera_ref,
            CAPABILITY_CAMERA_MONITOR,
        ),
        open_dataset=open_dataset,
        association_authority=authority,
    )
    handle = runtime.start(plan)
    source = sources[0]
    try:
        deadline = time.monotonic() + 2.0
        while handle.snapshot().phase != "monitoring-camera":
            if handle.snapshot().state.terminal or time.monotonic() >= deadline:
                raise AssertionError(handle.snapshot())
            time.sleep(0.005)

        document = _four_trigger_document(
            client=client,
            trigger_lane="ch11",
            required_interval_seconds=0.0001,
        )
        snapshot = client.snapshot()
        artifact = compile_pulse_artifact(
            document,
            clock_hz=snapshot.clock_hz,
            execution_form=PulseExecutionForm.STATIC_ONCE,
            trigger_channels=("ch11",),
            live_target=snapshot.target,
        )
        session_id = "fake-hardware-association"
        cursor = source.open_associated_signal_cursor("frame_0")
        token = cursor.arm_signal_association(
            SignalAssociationRequest(
                session_id,
                artifact.fingerprint,
                4,
                artifact.trigger_schedules[0].fingerprint,
                artifact.trigger_schedules[0].channel,
                artifact.trigger_schedules[0].total,
                artifact.trigger_schedules[0].minimum_interval_ticks,
                artifact.target_ir.clock_hz,
            )
        )
        reference = client.prepare(artifact)
        client.fire(reference)
        completion = client.complete(reference, timeout=1.0)
        terminal = PulseTerminalAck(session_id, "fake-sequencer-binding", completion)
        extra_channel_terminal = PulseTerminalAck(
            session_id,
            "fake-sequencer-binding",
            replace(
                completion,
                expected_trigger_counts_from_completed_schedule=(
                    *completion.expected_trigger_counts_from_completed_schedule,
                    ("ch06", 1),
                ),
            ),
        )
        with pytest.raises(
            RuntimeError,
            match="hardware pulse-terminal trigger count differs",
        ):
            cursor.bind_signal_association(token, extra_channel_terminal)
        cursor.bind_signal_association(token, terminal)
        events = tuple(cursor.next_associated_signal(token, 1.0) for _ in range(4))
        cursor.finish_signal_association(token)
        cursor.close()

        assert tuple(event.event_ref.sequence for event in events) == (0, 1, 2, 3)

        rejected = source.open_associated_signal_cursor("frame_0")
        bad_token = rejected.arm_signal_association(
            SignalAssociationRequest(
                "simulated-cause",
                artifact.fingerprint,
                4,
                artifact.trigger_schedules[0].fingerprint,
                artifact.trigger_schedules[0].channel,
                artifact.trigger_schedules[0].total,
                artifact.trigger_schedules[0].minimum_interval_ticks,
                artifact.target_ir.clock_hz,
            )
        )
        simulated = PulseTerminalAck(
            "simulated-cause",
            "fake-sequencer-binding",
            SimulatedPulseReceipt(
                artifact.fingerprint,
                "fake-simulator",
                (("ch11", 4),),
                0.0,
                0.0,
            ),
        )
        try:
            with np.testing.assert_raises_regex(ValueError, "hardware pulse terminal"):
                rejected.bind_signal_association(bad_token, simulated)
        finally:
            rejected.close()

        qcamera = cameras[0]
        qcamera.hardware_stamps_enabled = False
        missing_stamp = source.open_associated_signal_cursor("frame_0")
        missing_session_id = "missing-hardware-stamp"
        missing_token = missing_stamp.arm_signal_association(
            SignalAssociationRequest(
                missing_session_id,
                artifact.fingerprint,
                4,
                artifact.trigger_schedules[0].fingerprint,
                artifact.trigger_schedules[0].channel,
                artifact.trigger_schedules[0].total,
                artifact.trigger_schedules[0].minimum_interval_ticks,
                artifact.target_ir.clock_hz,
            )
        )
        reference = client.prepare(artifact)
        client.fire(reference)
        completion = client.complete(reference, timeout=1.0)
        try:
            deadline = time.monotonic() + 1.0
            while not handle.snapshot().state.terminal:
                if time.monotonic() >= deadline:
                    raise AssertionError(handle.snapshot())
                time.sleep(0.005)
            assert views[0].failure is not None
            assert "E0-qualified hardware stamp" in views[0].failure
            with pytest.raises(RuntimeError, match="E0-qualified hardware stamp"):
                missing_stamp.bind_signal_association(
                    missing_token,
                    PulseTerminalAck(
                        missing_session_id,
                        "fake-sequencer-binding",
                        completion,
                    ),
                )
        finally:
            missing_stamp.close()
    finally:
        if not handle.snapshot().state.terminal:
            handle.cancel("hardware association test complete")
        handle.wait(2.0)
        assert runtime.shutdown(timeout=2.0)


def test_real_camera_association_rejects_an_undrained_pre_fire_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A driver-ring frame cannot become ordinal zero of a later FPGA FIRE."""

    bus = _TriggerBus()
    client = _FakeRemoteClient(bus)
    cameras = []

    def camera_factory(config):
        camera = _FakeCamera(config, bus)
        cameras.append(camera)
        return camera

    _patch_hardware_dependencies(monkeypatch, client, camera_factory)
    composition = create_installation(_hardware_document())
    runtime = composition.runtime
    camera_ref = runtime.device_catalog.require("camera").ref
    authority = dict(composition.camera_signal_association_authorities)["camera"]
    sources = []

    def open_dataset(spec, *, event_source, **_kwargs):
        sources.append(event_source)
        return _LiveView(spec)

    plan = open_live_camera_measurement(
        CameraMeasurementRequest("camera", repeat=0),
        monitor_port=runtime.require_capability(
            camera_ref,
            CAPABILITY_CAMERA_MONITOR,
        ),
        open_dataset=open_dataset,
        association_authority=authority,
    )
    handle = runtime.start(plan)
    source = sources[0]
    try:
        deadline = time.monotonic() + 2.0
        while handle.snapshot().phase != "monitoring-camera":
            if handle.snapshot().state.terminal or time.monotonic() >= deadline:
                raise AssertionError(handle.snapshot())
            time.sleep(0.005)
        qcamera = cameras[0]
        qcamera.read_gate.clear()
        assert qcamera.blocked_read_started.wait(1.0)
        qcamera.emit(1)
        document = _four_trigger_document(
            client=client,
            trigger_lane="ch11",
            required_interval_seconds=0.0001,
        )
        snapshot = client.snapshot()
        artifact = compile_pulse_artifact(
            document,
            clock_hz=snapshot.clock_hz,
            execution_form=PulseExecutionForm.STATIC_ONCE,
            trigger_channels=("ch11",),
            live_target=snapshot.target,
        )
        cursor = source.open_associated_signal_cursor("frame_0")
        try:
            with pytest.raises(RuntimeError, match="has produced frames"):
                cursor.arm_signal_association(
                    SignalAssociationRequest(
                        "undrained-cause",
                        artifact.fingerprint,
                        4,
                        artifact.trigger_schedules[0].fingerprint,
                        artifact.trigger_schedules[0].channel,
                        artifact.trigger_schedules[0].total,
                        artifact.trigger_schedules[0].minimum_interval_ticks,
                        artifact.target_ir.clock_hz,
                    )
                )
        finally:
            cursor.close()
            qcamera.read_gate.set()
    finally:
        if not handle.snapshot().state.terminal:
            handle.cancel("undrained-frame boundary test complete")
        handle.wait(2.0)
        assert runtime.shutdown(timeout=2.0)


class _Node:
    def __init__(self, value, *, minimum=0, maximum=10_000, increment=1):
        self.value = value
        self.minimum = minimum
        self.maximum = maximum
        self.increment = increment

    def SetValue(self, value):
        self.value = value

    def GetValue(self):
        return self.value

    def GetMin(self):
        return self.minimum

    def GetMax(self):
        return self.maximum

    def GetInc(self):
        return self.increment


class _GrabResult:
    def __init__(self, value, stamp):
        self.Array = np.asarray(value, dtype=np.uint8)
        self.stamp = stamp
        self.released = False

    def GrabSucceeded(self):
        return True

    def GetBlockID(self):
        return self.stamp

    def GetImageNumber(self):
        return self.stamp + 100

    def Release(self):
        self.released = True


class _PylonDeviceInfo:
    def GetSerialNumber(self):
        return "basler-1"


class _PylonHardware:
    def __init__(self):
        self.PixelFormat = _Node("Mono8")
        self.ExposureTime = _Node(5_000.0)
        self.TriggerSelector = _Node("FrameStart")
        self.TriggerMode = _Node("On")
        self.TriggerSource = _Node("Line1")
        self.TriggerActivation = _Node("RisingEdge")
        self.Width = _Node(5, minimum=1, maximum=5)
        self.Height = _Node(4, minimum=1, maximum=4)
        self.OffsetX = _Node(0, minimum=0, maximum=4)
        self.OffsetY = _Node(0, minimum=0, maximum=3)
        self.WidthMax = _Node(5)
        self.HeightMax = _Node(4)
        self.ResultingFrameRate = _Node(1_000.0)
        self.results = []
        self.grabbing = False
        self.closed = False
        self.strategy = None

    def Open(self):
        return None

    def Close(self):
        self.closed = True

    def IsGrabbing(self):
        return self.grabbing

    def StartGrabbing(self, strategy):
        self.strategy = strategy
        self.grabbing = True

    def StartGrabbingMax(self, _frames, strategy):
        self.strategy = strategy
        self.grabbing = True

    def StopGrabbing(self):
        self.grabbing = False

    def RetrieveResult(self, _timeout_ms, _handling):
        return None if not self.results else self.results.pop(0)


class _PylonFactory:
    def __init__(self, hardware):
        self.hardware = hardware
        self.info = _PylonDeviceInfo()

    def EnumerateDevices(self):
        return (self.info,)

    def CreateDevice(self, _info):
        return self.hardware


class _PylonModule:
    GrabStrategy_LatestImageOnly = "latest"
    GrabStrategy_OneByOne = "ordered"
    TimeoutHandling_Return = "return"

    def __init__(self, hardware):
        factory = _PylonFactory(hardware)

        class TlFactory:
            @staticmethod
            def GetInstance():
                return factory

        self.TlFactory = TlFactory
        self.InstantCamera = lambda device: device


def test_pylon_adapter_preserves_mono8_for_finite_and_live_modes() -> None:
    hardware = _PylonHardware()
    adapter = PylonCameraAdapter(
        PylonCameraConfig("basler-1", ("ch06",)),
        pylon_module=_PylonModule(hardware),
    )
    try:
        assert adapter.capture_working_point().dtype == np.dtype("u1")
        hardware.results.extend(
            (_GrabResult(np.ones((4, 5)), 10), _GrabResult(np.ones((4, 5)), 11))
        )
        adapter.arm(2, source_group_sizes=(2,), buffer_frame_count=2, timeout=1.0)
        finite = adapter.read_frame_records(2, timeout=1.0, exact=True)
        assert [record.image.dtype for record in finite] == [np.dtype("u1")] * 2
        assert [record.produced_count for record in finite] == [1, 2]
        assert adapter.finish_record_capture().produced_count == 2

        hardware.results.append(_GrabResult(np.zeros((4, 5)), 20))
        adapter.arm(None, source_group_sizes=None, buffer_frame_count=8, timeout=1.0)
        live = adapter.read_frame_records(1, timeout=1.0, exact=False)
        assert live[0].image.dtype == np.dtype("u1")
        assert live[0].produced_count is None
        assert hardware.strategy == "latest"
        adapter.finish_record_capture()
        assert hardware.TriggerMode.GetValue() == "On"
    finally:
        adapter.close()
    assert hardware.closed
