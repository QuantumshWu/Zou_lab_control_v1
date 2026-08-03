"""Camera adapter SPI tests at its devices.camera.contract owner."""

from __future__ import annotations

import numpy as np
import pytest

from zlc_neutral_atom.devices.camera.contract import (
    CameraAcquisitionMode,
    CameraCaptureSpec,
)
from zlc_neutral_atom.devices.camera.contract import (
    CameraCaptureTerminalRecord,
    CameraFrameRecord,
    CameraWorkingPoint,
)
from zlc_neutral_atom.devices.camera.endpoint import CameraCaptureEndpoint
from zlc_neutral_atom.devices.camera.capture_port import (
    CompleteCaptureCommand,
    PrepareCaptureCommand,
    ReadCaptureCommand,
    StartCaptureCommand,
)
from zlc_neutral_atom.runtime.ports import DeviceBroker, SafetyOperation
from zlc_neutral_atom.runtime.resources import (
    DeviceIdentityEvidenceKind,
    PhysicalDeviceIdentity,
    ResourceKey,
)


class _ReusingRingCamera:
    timeout = 1.0

    def __init__(self, *, ordinals: tuple[int, ...] = (0, 1)) -> None:
        self.ring = np.zeros((3, 4), dtype=np.uint16)
        self.ordinals = ordinals
        self.expected = 0
        self.read_index = 0
        self.armed = False

    def capture_working_point(self) -> CameraWorkingPoint:
        return CameraWorkingPoint(
            "EXTERNAL_TRIGGERED",
            (3, 4),
            (3, 4),
            (0, 0),
            (3, 4),
            (1, 1),
            np.dtype("<u2"),
            "count",
            0.001,
            0.001,
            0.0,
            1.0,
            "fake-ring",
        )

    def arm(
        self,
        frames: int,
        *,
        source_group_sizes: tuple[int, ...] | None,
        buffer_frame_count: int,
        timeout: float,
    ) -> None:
        assert source_group_sizes is not None
        assert sum(source_group_sizes) == frames
        assert buffer_frame_count == frames
        assert timeout > 0
        self.expected = frames
        self.read_index = 0
        self.armed = True

    def read_frame_records(
        self,
        n: int,
        *,
        timeout: float,
        exact: bool,
    ) -> list[CameraFrameRecord]:
        assert n == 1 and exact and timeout > 0 and self.armed
        ordinal = self.ordinals[self.read_index]
        self.ring.fill(self.read_index + 1)
        record = CameraFrameRecord(
            self.ring,
            ordinal,
            self.expected,
            100 + self.read_index,
            200 + self.read_index,
            1,
            1_000 + self.read_index,
            10_000 + self.read_index,
            self.read_index % 2,
        )
        self.ring.fill(65_535)
        self.read_index += 1
        return [record]

    def finish_record_capture(self) -> CameraCaptureTerminalRecord:
        self.armed = False
        return CameraCaptureTerminalRecord(self.expected, True, True, True)

    def capture_state(self) -> tuple[bool, int]:
        return self.armed, 0


def _bound(camera: _ReusingRingCamera):
    endpoint = CameraCaptureEndpoint(
        camera,
        "camera",
    )
    broker = DeviceBroker()
    identity = PhysicalDeviceIdentity(
        stable_device_identity="fake-camera-identity",
        evidence_kind=DeviceIdentityEvidenceKind.INSTALLATION_ASSERTED_ENDPOINT,
    )
    proof = broker.verify_identity(lambda: identity)
    binding = None

    def current():
        assert binding is not None
        return binding

    binding = broker.bind(
        key=ResourceKey.parse("device/camera"),
        identity=proof,
        execute_command=lambda command: endpoint.execute_command(current(), command),
        capability_probe=lambda: endpoint.capability_probe(current()),
        close_session=lambda command: endpoint.close_session(current(), command),
        interrupt_operations={SafetyOperation.DISARM: endpoint.interrupt},
    )
    capability = broker.verify_capability(binding).snapshot
    return endpoint, binding, capability, broker


def _prepare_started(camera: _ReusingRingCamera):
    endpoint, binding, _capability, broker = _bound(camera)
    command = _prepare_command()
    session_id = command.session_id
    endpoint.execute_command(binding, command)
    endpoint.execute_command(binding, StartCaptureCommand(session_id, 1.0))
    return endpoint, binding, session_id, broker


def _prepare_command() -> PrepareCaptureCommand:
    spec = CameraCaptureSpec(
        CameraAcquisitionMode.EXTERNAL_TRIGGERED,
        2,
        (1, 1),
    )
    session_id = "fake-session"
    return PrepareCaptureCommand(
        session_id,
        spec,
        2,
        1.0,
    )


def test_nonvirtual_adapter_owns_ring_bytes_and_preserves_batch_count() -> None:
    camera = _ReusingRingCamera()
    endpoint, binding, session_id, broker = _prepare_started(camera)
    try:
        first = endpoint.execute_command(
            binding, ReadCaptureCommand(session_id, 1.0)
        ).payload
        second = endpoint.execute_command(
            binding, ReadCaptureCommand(session_id, 1.0)
        ).payload
        terminal = endpoint.execute_command(
            binding, CompleteCaptureCommand(session_id, 2, 1.0)
        )
        assert np.all(first.image.values == 1)
        assert np.all(second.image.values == 2)
        assert np.all(camera.ring == 65_535)
        assert first.metadata.produced_count == second.metadata.produced_count == 2
        assert (first.metadata.source_ordinal, second.metadata.source_ordinal) == (0, 1)
        assert terminal.produced_count == terminal.drained_count == 2
        assert terminal.joined
    finally:
        if camera.armed:
            endpoint.interrupt()
        broker.shutdown()


def test_exact_endpoint_rejects_adapter_ordinal_gap() -> None:
    camera = _ReusingRingCamera(ordinals=(0, 2))
    endpoint, binding, session_id, broker = _prepare_started(camera)
    try:
        endpoint.execute_command(binding, ReadCaptureCommand(session_id, 1.0))
        with pytest.raises(RuntimeError, match="ordinal 2 differs from expected 1"):
            endpoint.execute_command(binding, ReadCaptureCommand(session_id, 1.0))
    finally:
        endpoint.interrupt()
        broker.shutdown()


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("produced_count", -1),
        ("timestamp_seconds", -1),
        ("timestamp_microseconds", 1_000_000),
        ("driver_buffer_index", -1),
    ),
)
def test_frame_record_rejects_invalid_adapter_metadata(field: str, value: int) -> None:
    values = {
        "image": np.zeros((2, 3), dtype=np.uint16),
        "source_ordinal": 0,
        "produced_count": 1,
        "frame_stamp": 1,
        "camera_stamp": 1,
        "timestamp_seconds": 1,
        "timestamp_microseconds": 0,
        "host_received_at_ns": 1,
        "driver_buffer_index": 0,
    }
    values[field] = value
    with pytest.raises((TypeError, ValueError)):
        CameraFrameRecord(**values)
