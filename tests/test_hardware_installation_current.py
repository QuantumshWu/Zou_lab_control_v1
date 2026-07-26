"""Contract tests for the real-installation composition using explicit fakes.

These tests prove wiring and active-E0 mechanics only.  They make no claim
about real-camera loss statistics or a deployed FPGA.
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from conftest import pulse_backend_completion_for
from fpga.pulse_streamer.host.image import StreamerParams, build_fingerprint
from zlc_neutral_atom.devices.camera.contract import (
    CameraCaptureTerminalRecord,
    CameraFrameRecord,
    CameraWorkingPoint,
)
from zlc_neutral_atom.devices.hardware.config import HardwareInstallationConfig
from zlc_neutral_atom.devices.hardware.installation import create_hardware_installation
from zlc_neutral_atom.devices.hardware.qualification import _qualification_document
from zlc_neutral_atom.devices.camera.pylon import PylonCameraAdapter, PylonCameraConfig
from zlc_neutral_atom.devices.sequencer.port import (
    PulseTerminalAck,
    SimulatedPulseReceipt,
)
from zlc_neutral_atom.installation_plan import InstallationDevicePlan
from zlc_neutral_atom.logic_nodes.camera_measurement.definition import (
    CameraMeasurementRequest,
)
from zlc_neutral_atom.logic_nodes.camera_measurement.monitor import (
    CameraMonitorViewSpec,
    prepare_live_camera_measurement,
)
from zlc_neutral_atom.runtime.signal_source import SignalAssociationRequest
from zlc_pulse import (
    PreparedPulseRef,
    PulseCompletion,
    PulseExecutionForm,
    PulseServerSnapshot,
    RemotePulseExecutionClient,
    compile_pulse_artifact,
    load_deployed_pulse_target,
    pulse_target_manifest_from_lanes,
)
from zlc_storage import canonical_digest, decode


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
        self.read_gate = threading.Event()
        self.read_gate.set()
        self.blocked_read_started = threading.Event()

    @property
    def timeout(self):
        return 1.0

    def capture_working_point(self):
        primitive = {"fake-camera": self.lane, "dtype": "u1"}
        return CameraWorkingPoint(
            canonical_digest(primitive),
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
        self.records = [
            CameraFrameRecord(
                np.full((4, 5), index, dtype=np.uint8),
                index,
                index + 1,
                100 + index,
                200 + index,
                None,
                None,
                time.time_ns() + index,
            )
            for index in range(count)
        ]

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
        return self.armed, max(0, len(self.records) - self.delivered)

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
        return PulseServerSnapshot(
            self._generation,
            self._manifest,
            200e6,
            build_fingerprint(StreamerParams()),
            self._state,
            None,
            {},
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


def _kind(cls) -> str:
    return f"{cls.__module__}.{cls.__qualname__}"


class _LiveView:
    def __init__(self, spec: CameraMonitorViewSpec) -> None:
        self.spec = spec
        self.dataset = None
        self.failure = None

    def bind(self, dataset, *, run_id, causation_domain_id):
        assert run_id and causation_domain_id
        self.dataset = dataset

    def updated(self):
        return None

    def notification_failed(self, message):
        self.failure = message

    def fail(self, message):
        self.failure = message

    def source_terminal(self):
        return None


def test_fake_real_installation_runs_both_active_e0_paths_and_binds_ports() -> None:
    bus = _TriggerBus()
    client = _FakeRemoteClient(bus)
    cameras = []

    def camera_factory(config):
        camera = _FakeCamera(config, bus)
        cameras.append(camera)
        return camera

    plan = (
        InstallationDevicePlan("sequencer", "sequencer", _kind(_FakeRemoteClient), "fake remote"),
        InstallationDevicePlan("camera", "camera", _kind(_FakeCamera), "fake qCMOS"),
        InstallationDevicePlan("mot_camera", "camera", _kind(_FakeCamera), "fake Basler"),
    )
    config = HardwareInstallationConfig(
        pulse_host="test-host",
        pylon_serial="test-basler",
        readout_site_centers_xy=((1.0, 1.0),),
        readout_grid_shape_yx=(1, 1),
    )
    composition = create_hardware_installation(
        config,
        device_plan=plan,
        remote_client_factory=lambda *_args, **_kwargs: client,
        dcam_factory=camera_factory,
        pylon_factory=camera_factory,
    )
    try:
        assert bus.fire_count == 2
        assert [camera.lane for camera in cameras] == ["ch11", "ch06"]
        assert all(camera.arm_calls == [(4, (4,), 4)] for camera in cameras)
        catalog = composition.runtime.device_catalog
        assert catalog.roles("camera") == ("camera", "mot_camera")
        for role in ("camera", "mot_camera"):
            reference = catalog.require(role).ref
            port = composition.runtime.camera_port(reference)
            evidence = port.capability.camera_capability_evidence
            assert evidence.exact_external_trigger_qualification_digest is not None
            assert port.capability.payload_contract.value_schema.dtype == np.dtype("u1")
        assert composition.readout_apparatus_facts[0].trigger_channel == "ch11"
        assert tuple(
            role for role, _authority in composition.camera_signal_association_authorities
        ) == ("camera",)
    finally:
        assert composition.runtime.shutdown(timeout=2.0)
    assert client._closed
    assert all(camera.closed for camera in cameras)


def test_real_installation_rejects_an_e0_hardware_stamp_gap() -> None:
    bus = _TriggerBus()
    client = _FakeRemoteClient(bus)
    cameras = []

    def gap_factory(config):
        camera = _GapCamera(config, bus)
        cameras.append(camera)
        return camera

    plan = (
        InstallationDevicePlan("sequencer", "sequencer", _kind(_FakeRemoteClient), "fake remote"),
        InstallationDevicePlan("camera", "camera", _kind(_GapCamera), "fake qCMOS"),
        InstallationDevicePlan("mot_camera", "camera", _kind(_GapCamera), "fake Basler"),
    )
    with pytest.raises(RuntimeError, match="frame_stamp has a gap"):
        create_hardware_installation(
            HardwareInstallationConfig(
                pulse_host="test-host",
                pylon_serial="test-basler",
                readout_site_centers_xy=((1.0, 1.0),),
                readout_grid_shape_yx=(1, 1),
            ),
            device_plan=plan,
            remote_client_factory=lambda *_args, **_kwargs: client,
            dcam_factory=gap_factory,
            pylon_factory=gap_factory,
        )
    assert client._closed
    assert cameras and all(camera.closed for camera in cameras)


def test_fake_real_camera_signal_association_uses_hardware_terminal_and_ordinals() -> None:
    """Exercise the real composition's sole Camera->PulseScan evidence path."""

    bus = _TriggerBus()
    client = _FakeRemoteClient(bus)

    def camera_factory(config):
        return _FakeCamera(config, bus)

    plan = (
        InstallationDevicePlan("sequencer", "sequencer", _kind(_FakeRemoteClient), "fake remote"),
        InstallationDevicePlan("camera", "camera", _kind(_FakeCamera), "fake qCMOS"),
        InstallationDevicePlan("mot_camera", "camera", _kind(_FakeCamera), "fake Basler"),
    )
    composition = create_hardware_installation(
        HardwareInstallationConfig(
            pulse_host="test-host",
            pylon_serial="test-basler",
            readout_site_centers_xy=((1.0, 1.0),),
            readout_grid_shape_yx=(1, 1),
        ),
        device_plan=plan,
        remote_client_factory=lambda *_args, **_kwargs: client,
        dcam_factory=camera_factory,
        pylon_factory=camera_factory,
    )
    runtime = composition.runtime
    camera_ref = runtime.device_catalog.require("camera").ref
    authority = dict(composition.camera_signal_association_authorities)["camera"]
    prepared = prepare_live_camera_measurement(
        CameraMeasurementRequest(camera_ref, repeat=0, history_cycles=8),
        monitor_port=runtime.camera_monitor_port(camera_ref),
        start_run=runtime.start,
        association_authority=authority,
    )
    views = []

    def view_factory(spec):
        view = _LiveView(spec)
        views.append(view)
        return view

    handle = prepared.start_with_view(factory=view_factory)
    try:
        deadline = time.monotonic() + 2.0
        while handle.snapshot().phase != "monitoring-camera":
            if handle.snapshot().state.terminal or time.monotonic() >= deadline:
                raise AssertionError(handle.snapshot())
            time.sleep(0.005)

        document = _qualification_document(
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
        cursor = prepared.open_associated_signal_cursor("frame_0")
        token = cursor.arm_signal_association(
            SignalAssociationRequest(
                "fake-hardware-window",
                session_id,
                artifact.fingerprint,
                4,
            )
        )
        reference = client.prepare(artifact)
        client.fire(reference)
        completion = client.complete(reference, timeout=1.0)
        terminal = PulseTerminalAck(session_id, "fake-sequencer-binding", completion)
        cursor.bind_signal_association(token, terminal)
        events = tuple(cursor.next_associated_signal(token, 1.0) for _ in range(4))
        evidence = cursor.finish_signal_association(token)
        cursor.close()

        assert tuple(event.event_ref.sequence for event in events) == (0, 1, 2, 3)
        payload = decode(evidence.canonical_evidence)
        assert payload["schema"].endswith("camera-measurement.pulse-association")
        assert payload["terminal_evidence_kind"] == "HARDWARE_RAW_REGISTERS"
        assert payload["physical_start_ordinal"] == 0
        assert payload["physical_end_ordinal"] == 4

        rejected = prepared.open_associated_signal_cursor("frame_0")
        bad_token = rejected.arm_signal_association(
            SignalAssociationRequest(
                "reject-simulated-terminal",
                "simulated-cause",
                artifact.fingerprint,
                1,
            )
        )
        simulated = PulseTerminalAck(
            "simulated-cause",
            "fake-sequencer-binding",
            SimulatedPulseReceipt(
                artifact.fingerprint,
                "fake-simulator",
                (("ch11", 1),),
                0.0,
                0.0,
            ),
        )
        try:
            with np.testing.assert_raises_regex(ValueError, "hardware pulse terminal"):
                rejected.bind_signal_association(bad_token, simulated)
        finally:
            rejected.close()
    finally:
        if not handle.snapshot().state.terminal:
            handle.cancel("hardware association test complete")
        handle.wait(2.0)
        assert runtime.shutdown(timeout=2.0)


def test_real_camera_association_rejects_an_undrained_pre_fire_frame() -> None:
    """A driver-ring frame cannot become ordinal zero of a later FPGA FIRE."""

    bus = _TriggerBus()
    client = _FakeRemoteClient(bus)
    cameras = []

    def camera_factory(config):
        camera = _FakeCamera(config, bus)
        cameras.append(camera)
        return camera

    plan = (
        InstallationDevicePlan(
            "sequencer", "sequencer", _kind(_FakeRemoteClient), "fake remote"
        ),
        InstallationDevicePlan(
            "camera", "camera", _kind(_FakeCamera), "fake qCMOS"
        ),
        InstallationDevicePlan(
            "mot_camera", "camera", _kind(_FakeCamera), "fake Basler"
        ),
    )
    composition = create_hardware_installation(
        HardwareInstallationConfig(
            pulse_host="test-host",
            pylon_serial="test-basler",
            readout_site_centers_xy=((1.0, 1.0),),
            readout_grid_shape_yx=(1, 1),
        ),
        device_plan=plan,
        remote_client_factory=lambda *_args, **_kwargs: client,
        dcam_factory=camera_factory,
        pylon_factory=camera_factory,
    )
    runtime = composition.runtime
    camera_ref = runtime.device_catalog.require("camera").ref
    authority = dict(composition.camera_signal_association_authorities)["camera"]
    prepared = prepare_live_camera_measurement(
        CameraMeasurementRequest(camera_ref, repeat=0, history_cycles=8),
        monitor_port=runtime.camera_monitor_port(camera_ref),
        start_run=runtime.start,
        association_authority=authority,
    )
    handle = prepared.start_with_view(factory=_LiveView)
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
        cursor = prepared.open_associated_signal_cursor("frame_0")
        try:
            with pytest.raises(RuntimeError, match="has produced frames"):
                cursor.arm_signal_association(
                    SignalAssociationRequest(
                        "reject-undrained-frame",
                        "undrained-cause",
                        "0" * 64,
                        1,
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
