"""Current adapter-SDK camera endpoint contracts."""

from __future__ import annotations

from dataclasses import replace
import threading

import numpy as np
import pytest

from zlc_neutral_atom.acquisition import (
    CameraAcquisitionMode,
    CameraCaptureSpec,
    freeze_camera_capture_spec,
)
from zlc_neutral_atom.adapter_sdk import (
    CameraCaptureTerminalRecord,
    CameraFrameRecord,
    CameraWorkingPoint,
)
from zlc_neutral_atom.bootstrap._camera_endpoint import CameraCaptureEndpoint
from zlc_neutral_atom.runtime.capture import (
    CompleteCaptureCommand,
    PrepareCaptureCommand,
    ReadCaptureCommand,
    StartCaptureCommand,
    camera_capability_evidence_from_tree,
    camera_capability_evidence_to_tree,
)
from zlc_neutral_atom.runtime.ports import (
    DeviceBroker,
    SafetyOperation,
    SessionCloseCommand,
)
from zlc_neutral_atom.runtime.resources import (
    DeviceIdentityEvidenceKind,
    PhysicalDeviceIdentity,
    ResourceKey,
)
from zlc_storage import canonical_digest


class _Camera:
    max_pending_records = 2
    timeout = 1.0

    def __init__(
        self,
        *,
        ordinals: tuple[int, ...] = (0, 1),
        block_arm: bool = False,
        pause_after_arm: bool = False,
        metadata_overrides: tuple[dict[str, int | None], ...] = (),
    ) -> None:
        self.ring = np.zeros((3, 4), dtype=np.uint16)
        self.ordinals = ordinals
        self.expected = 0
        self.read_index = 0
        self.armed = False
        self.settings_generation = 0
        self.block_arm = block_arm
        self.pause_after_arm = pause_after_arm
        self.metadata_overrides = tuple(dict(item) for item in metadata_overrides)
        self.trigger_channels = ("ch11",)
        self.arm_entered = threading.Event()
        self.arm_installed = threading.Event()
        self.release_arm = threading.Event()
        self.finish_calls = 0
        self._condition = threading.Condition()

    def capture_working_point(self) -> CameraWorkingPoint:
        return CameraWorkingPoint(
            canonical_digest(
                {
                    "fixture": "camera-endpoint",
                    "generation": self.settings_generation,
                }
            ),
            "EXTERNAL_TRIGGERED",
            (3, 4),
            (3, 4),
            (0, 0),
            (3, 4),
            (1, 1),
            np.dtype("<u2"),
            "count",
            self.trigger_channels,
            0.001,
            0.001,
            0.0,
            1.0,
            "fixture-readout",
        )

    def arm(
        self,
        frames: int,
        *,
        source_group_sizes: tuple[int, ...] | None,
        max_inflight_frames: int,
        timeout: float,
    ) -> None:
        assert source_group_sizes is not None
        assert sum(source_group_sizes) == frames
        assert max_inflight_frames == 2
        assert timeout > 0
        self.arm_entered.set()
        if self.block_arm and not self.release_arm.wait(2.0):
            raise TimeoutError("fixture did not release arm")
        with self._condition:
            self.expected = frames
            self.read_index = 0
            self.armed = True
            self._condition.notify_all()
        self.arm_installed.set()
        if self.pause_after_arm and not self.release_arm.wait(2.0):
            raise TimeoutError("fixture did not release post-arm pause")

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
        metadata = {
            "source_ordinal": ordinal,
            "produced_count": self.expected,
            "frame_stamp": 100 + self.read_index,
            "camera_stamp": 200 + self.read_index,
            "timestamp_seconds": 1,
            "timestamp_microseconds": 1_000 + self.read_index,
            "host_received_at_ns": 10_000 + self.read_index,
            "driver_buffer_index": self.read_index % 2,
        }
        if self.read_index < len(self.metadata_overrides):
            metadata.update(self.metadata_overrides[self.read_index])
        record = CameraFrameRecord(self.ring, **metadata)
        self.ring.fill(65_535)
        self.read_index += 1
        return [record]

    def finish_record_capture(self) -> CameraCaptureTerminalRecord:
        self.finish_calls += 1
        with self._condition:
            self.armed = False
            self._condition.notify_all()
        return CameraCaptureTerminalRecord(self.expected, True, True, True)

    def capture_state(self) -> tuple[bool, int]:
        with self._condition:
            return self.armed, 0


def _bound(camera: _Camera, *, qualified: bool = True):
    endpoint = CameraCaptureEndpoint(
        camera,
        "camera",
        exact_external_trigger_qualification_digest=(
            canonical_digest({"qualification": "fixture"})
            if qualified
            else None
        ),
    )
    broker = DeviceBroker()
    identity = PhysicalDeviceIdentity(
        "fixture-camera",
        DeviceIdentityEvidenceKind.INSTALLATION_ASSERTED_ENDPOINT,
        "fixture-camera-evidence",
        "fixture-assets-v1",
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


def _prepare_command(capability, *, session_id: str = "fixture-session"):
    frozen = freeze_camera_capture_spec(
        CameraCaptureSpec(
            CameraAcquisitionMode.EXTERNAL_TRIGGERED,
            2,
            (1, 1),
            capability.settings_fingerprint,
        )
    )
    return PrepareCaptureCommand(
        session_id,
        frozen.payload,
        frozen.owner_fingerprint,
        frozen.digest,
        capability.capability_fingerprint,
        capability.settings_fingerprint,
        2,
        1.0,
    )


def _prepare_started(camera: _Camera):
    endpoint, binding, capability, broker = _bound(camera)
    command = _prepare_command(capability)
    endpoint.execute_command(binding, command)
    endpoint.execute_command(binding, StartCaptureCommand(command.session_id, 1.0))
    return endpoint, binding, command, broker


def test_endpoint_owns_adapter_ring_bytes_and_terminal_count() -> None:
    camera = _Camera()
    endpoint, binding, command, broker = _prepare_started(camera)
    try:
        first = endpoint.execute_command(
            binding,
            ReadCaptureCommand(command.session_id, 1.0),
        ).payload
        second = endpoint.execute_command(
            binding,
            ReadCaptureCommand(command.session_id, 1.0),
        ).payload
        terminal = endpoint.execute_command(
            binding,
            CompleteCaptureCommand(command.session_id, 2, 1.0),
        )
        assert np.all(first.image.values == 1)
        assert np.all(second.image.values == 2)
        assert np.all(camera.ring == 65_535)
        assert (first.metadata.source_ordinal, second.metadata.source_ordinal) == (0, 1)
        assert terminal.produced_count == terminal.drained_count == 2
        assert terminal.joined
    finally:
        if camera.armed:
            endpoint.interrupt()
        broker.shutdown()


def test_capability_evidence_is_the_public_owner_of_physical_facts() -> None:
    camera = _Camera()
    endpoint, binding, capability, broker = _bound(camera)
    try:
        evidence = capability.camera_capability_evidence
        assert evidence.physical_facts.output_shape_yx == (3, 4)
        assert evidence.physical_facts.capture_trigger_channels == ("ch11",)
        assert capability.payload_contract.fingerprint == (
            evidence.payload_contract_fingerprint
        )
        tree = camera_capability_evidence_to_tree(evidence)
        assert tree["schema"] == "zlc_neutral_atom.CameraCapabilityEvidence"
        assert tree["physical_facts"]["capture_trigger_channels"] == ["ch11"]
        assert camera_capability_evidence_from_tree(tree) == evidence
        assert not hasattr(endpoint, "_payload_contract")
        assert endpoint.payload_contract(binding) is capability.payload_contract
    finally:
        broker.shutdown()


def test_endpoint_rejects_settings_drift_before_arm() -> None:
    camera = _Camera()
    endpoint, binding, capability, broker = _bound(camera)
    try:
        camera.settings_generation += 1
        with pytest.raises(RuntimeError, match="working point changed"):
            endpoint.execute_command(binding, _prepare_command(capability))
        assert camera.capture_state() == (False, 0)
    finally:
        broker.shutdown()


def test_endpoint_revalidates_working_point_after_arm_and_disarms() -> None:
    camera = _Camera(pause_after_arm=True)
    endpoint, binding, capability, broker = _bound(camera)
    command = _prepare_command(capability)
    endpoint.execute_command(binding, command)
    outcome: dict[str, object] = {}

    def start() -> None:
        try:
            outcome["ack"] = endpoint.execute_command(
                binding,
                StartCaptureCommand(command.session_id, 1.0),
            )
        except BaseException as error:
            outcome["error"] = error

    worker = threading.Thread(target=start)
    worker.start()
    assert camera.arm_installed.wait(1.0)
    camera.settings_generation += 1
    camera.trigger_channels = ("ch12",)
    camera.release_arm.set()
    worker.join(2.0)
    try:
        assert not worker.is_alive()
        assert "ack" not in outcome
        assert "working point changed during capture" in str(outcome["error"])
        assert camera.finish_calls >= 1
        assert camera.capture_state() == (False, 0)
    finally:
        broker.shutdown()


def test_endpoint_rejects_adapter_ordinal_gap() -> None:
    camera = _Camera(ordinals=(0, 2))
    endpoint, binding, command, broker = _prepare_started(camera)
    try:
        endpoint.execute_command(
            binding,
            ReadCaptureCommand(command.session_id, 1.0),
        )
        with pytest.raises(RuntimeError, match="ordinal 2 differs from expected 1"):
            endpoint.execute_command(
                binding,
                ReadCaptureCommand(command.session_id, 1.0),
            )
    finally:
        endpoint.interrupt()
        broker.shutdown()


@pytest.mark.parametrize(
    "metadata_overrides, expected_error",
    (
        (
            ({"produced_count": 3}, {"produced_count": 2}),
            "produced-count moved backwards",
        ),
        (({}, {"frame_stamp": 100}), "frame stamp is not strictly increasing"),
        (({}, {"camera_stamp": 200}), "camera stamp is not strictly increasing"),
        (
            ({"timestamp_seconds": 2}, {"timestamp_seconds": 1}),
            "capture timestamp moved backwards",
        ),
    ),
)
def test_endpoint_rejects_nonmonotonic_physical_frame_metadata(
    metadata_overrides,
    expected_error,
) -> None:
    camera = _Camera(metadata_overrides=metadata_overrides)
    endpoint, binding, command, broker = _prepare_started(camera)
    try:
        endpoint.execute_command(
            binding,
            ReadCaptureCommand(command.session_id, 1.0),
        )
        with pytest.raises(RuntimeError, match=expected_error):
            endpoint.execute_command(
                binding,
                ReadCaptureCommand(command.session_id, 1.0),
            )
    finally:
        endpoint.interrupt()
        broker.shutdown()


def test_unqualified_adapter_cannot_self_grant_exact_capture() -> None:
    camera = _Camera()
    endpoint, binding, capability, broker = _bound(camera, qualified=False)
    try:
        with pytest.raises(ValueError, match="requires E0-qualified"):
            endpoint.execute_command(binding, _prepare_command(capability))
        assert camera.capture_state() == (False, 0)
    finally:
        broker.shutdown()


def test_pre_arm_interrupt_supersedes_start_and_close_waits_for_join() -> None:
    camera = _Camera(block_arm=True)
    endpoint, binding, capability, broker = _bound(camera)
    command = _prepare_command(capability)
    endpoint.execute_command(binding, command)
    outcome: dict[str, object] = {}

    def start() -> None:
        try:
            outcome["ack"] = endpoint.execute_command(
                binding,
                StartCaptureCommand(command.session_id, 1.0),
            )
        except BaseException as error:
            outcome["error"] = error

    worker = threading.Thread(target=start)
    worker.start()
    assert camera.arm_entered.wait(1.0)
    endpoint.interrupt()

    close_done = threading.Event()
    close_outcome: dict[str, object] = {}

    def close() -> None:
        try:
            close_outcome["ack"] = endpoint.close_session(
                binding,
                SessionCloseCommand(command.session_id, 1.0),
            )
        except BaseException as error:
            close_outcome["error"] = error
        finally:
            close_done.set()

    closer = threading.Thread(target=close)
    closer.start()
    assert not close_done.wait(0.05)
    camera.release_arm.set()
    worker.join(2.0)
    closer.join(2.0)
    try:
        assert not worker.is_alive() and not closer.is_alive()
        assert "ack" not in outcome
        assert "superseded" in str(outcome["error"])
        assert "error" not in close_outcome
        assert close_outcome["ack"].is_terminal
    finally:
        broker.shutdown()


def test_post_arm_interrupt_never_returns_started_ack_or_double_stops() -> None:
    camera = _Camera(pause_after_arm=True)
    endpoint, binding, capability, broker = _bound(camera)
    command = _prepare_command(capability)
    endpoint.execute_command(binding, command)
    outcome: dict[str, object] = {}

    def start() -> None:
        try:
            outcome["ack"] = endpoint.execute_command(
                binding,
                StartCaptureCommand(command.session_id, 1.0),
            )
        except BaseException as error:
            outcome["error"] = error

    worker = threading.Thread(target=start)
    worker.start()
    assert camera.arm_installed.wait(1.0)
    endpoint.interrupt()
    camera.release_arm.set()
    worker.join(2.0)
    try:
        assert not worker.is_alive()
        assert "ack" not in outcome
        assert "superseded" in str(outcome["error"])
        assert camera.finish_calls >= 1
        assert camera.capture_state() == (False, 0)
    finally:
        broker.shutdown()


def test_session_identity_is_binding_scoped_and_cannot_be_replayed() -> None:
    camera = _Camera()
    endpoint, binding, capability, broker = _bound(camera)
    command = _prepare_command(capability)
    endpoint.execute_command(binding, command)
    try:
        replay = replace(command, session_id="other-session")
        with pytest.raises(RuntimeError, match="active session"):
            endpoint.execute_command(binding, replay)
    finally:
        endpoint.close_session(
            binding,
            SessionCloseCommand(command.session_id, 1.0),
        )
        broker.shutdown()
