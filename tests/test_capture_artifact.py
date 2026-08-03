"""Direct-output persistence for exact pulse-triggered camera datasets."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import zlc_neutral_atom.capture.artifact as capture_artifact_impl
from zlc_data import AxisId, AxisSpec, BlockId, REPEAT
from zlc_neutral_atom.devices.camera.contract import (
    CameraCaptureTerminalRecord,
    CameraFrameRecord,
    CameraWorkingPoint,
)
from zlc_neutral_atom.capture.artifact import (
    compile_capture_artifact_pipeline,
    load_capture_artifact,
)
from zlc_neutral_atom.devices.camera.endpoint import CameraCaptureEndpoint
from zlc_neutral_atom.devices.simulation.sequencer_endpoint import (
    VirtualSequencerExecutionEndpoint,
)
from zlc_neutral_atom.capture.binding import (
    TriggeredCameraLayout,
    bind_triggered_camera_acquisition,
)
from zlc_neutral_atom.devices.simulation.apparatus import VirtualSequencer
from zlc_neutral_atom.capture.reference import (
    CaptureArtifactRef,
    capture_artifact_ref_from_tree,
    capture_artifact_ref_to_tree,
)
from zlc_neutral_atom.devices.camera.capture_port import BoundCapturePort
from zlc_neutral_atom.capture.pipeline import MinimalPipelineSpec
from zlc_neutral_atom.runtime.ports import DeviceBroker, SafetyOperation
from zlc_neutral_atom.runtime.resources import (
    DeviceIdentityEvidenceKind,
    PhysicalDeviceIdentity,
    ResourceArbiter,
    ResourceKey,
)
from zlc_neutral_atom.runtime.run import RunController, RunFailed
from zlc_neutral_atom.capture.triggered import TriggeredCaptureSpec
from zlc_neutral_atom.devices.sequencer.port import BoundPulsePort
from zlc_pulse import (
    PulseExecutionForm,
    load_deployed_geometry_facts,
    load_deployed_pulse_target,
    load_pulse_document,
    pulse_target_manifest_from_lanes,
)
from zlc_storage import canonical_digest


_ROOT = Path(__file__).parents[1]


class _Camera:
    timeout = 1.0

    def __init__(self, *, terminal_count_delta: int = 0) -> None:
        self.expected = 0
        self.ordinal = 0
        self.armed = False
        self.terminal_count_delta = terminal_count_delta

    def capture_working_point(self) -> CameraWorkingPoint:
        return CameraWorkingPoint(
            canonical_digest({"fixture": "capture-artifact-camera"}),
            "EXTERNAL_TRIGGERED",
            (3, 4),
            (3, 4),
            (0, 0),
            (3, 4),
            (1, 1),
            np.dtype("<u2"),
            "count",
            ("ch11",),
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
        buffer_frame_count: int,
        timeout: float,
    ) -> None:
        assert source_group_sizes is not None
        assert sum(source_group_sizes) == frames
        assert buffer_frame_count == frames and timeout > 0
        self.expected = frames
        self.ordinal = 0
        self.armed = True

    def read_frame_records(
        self,
        n: int,
        *,
        timeout: float,
        exact: bool,
    ) -> list[CameraFrameRecord]:
        assert n == 1 and exact and timeout > 0 and self.armed
        ordinal = self.ordinal
        self.ordinal += 1
        curve = np.array((1200, 800, 520, 340), dtype=np.uint16)
        image = np.tile(curve + ordinal * 10, (3, 1))
        return [
            CameraFrameRecord(
                image,
                ordinal,
                self.expected,
                100 + ordinal,
                200 + ordinal,
                1,
                1_000 + ordinal,
                10_000 + ordinal,
                ordinal % 2,
            )
        ]

    def finish_record_capture(self) -> CameraCaptureTerminalRecord:
        self.armed = False
        return CameraCaptureTerminalRecord(
            self.expected + self.terminal_count_delta,
            True,
            True,
            True,
        )

    def capture_state(self) -> tuple[bool, int]:
        return self.armed, 0

    def close(self) -> None:
        self.armed = False


def _identity(name: str) -> PhysicalDeviceIdentity:
    return PhysicalDeviceIdentity(
        stable_device_identity=name,
        evidence_kind=DeviceIdentityEvidenceKind.INSTALLATION_ASSERTED_ENDPOINT,
    )


def _bind_endpoint(
    broker: DeviceBroker,
    *,
    key: str,
    identity: str,
    endpoint,
    cleanup_operation: SafetyOperation,
):
    binding = None

    def current():
        assert binding is not None
        return binding

    proof = broker.verify_identity(lambda: _identity(identity))
    binding = broker.bind(
        key=ResourceKey.parse(key),
        identity=proof,
        execute_command=lambda command: endpoint.execute_command(current(), command),
        capability_probe=lambda: endpoint.capability_probe(current()),
        close_session=lambda command: endpoint.close_session(current(), command),
        interrupt_operations={cleanup_operation: endpoint.interrupt},
    )
    return broker.verify_capability(binding)


class _CaptureCase:
    def __init__(
        self,
        tmp_path,
        *,
        camera: _Camera | None = None,
    ) -> None:
        self.camera = _Camera() if camera is None else camera
        self.broker = DeviceBroker()
        camera_endpoint = CameraCaptureEndpoint(
            self.camera,
            "camera",
            exact_external_trigger_qualification_digest=canonical_digest(
                {"qualification": "deterministic fixture adapter"}
            ),
        )
        camera_port = BoundCapturePort(
            _bind_endpoint(
                self.broker,
                key="device/camera",
                identity="fixture-camera",
                endpoint=camera_endpoint,
                cleanup_operation=SafetyOperation.DISARM,
            )
        )
        target = load_deployed_pulse_target()
        geometry = load_deployed_geometry_facts()
        self.sequencer = VirtualSequencer(
            target,
            clock_hz=geometry.clock_hz,
            sleep_scale=0,
        )
        pulse_endpoint = VirtualSequencerExecutionEndpoint(
            self.sequencer,
            pulse_target_manifest_from_lanes(target),
            geometry_fingerprint=geometry.geometry_fingerprint,
        )
        pulse_port = BoundPulsePort(
            _bind_endpoint(
                self.broker,
                key="device/sequencer",
                identity="fixture-sequencer",
                endpoint=pulse_endpoint,
                cleanup_operation=SafetyOperation.SAFE_STATE,
            ),
            "virtual",
            (),
        )
        repeat_axis = AxisSpec(AxisId("repeat"), "repeat", REPEAT, 1, (0,))
        binding = bind_triggered_camera_acquisition(
            pulse_port,
            camera_port,
            pulse_document=load_pulse_document(
                _ROOT / "pulses" / "imaging_template.json"
            ),
            execution_form=PulseExecutionForm.STATIC_ONCE,
            trigger_channel="ch11",
            layout=TriggeredCameraLayout(
                repeat_axis,
                AxisId("readout-event"),
                AxisId("scan-ordinal"),
                readout_events_per_repeat=3,
            ),
        )
        pipeline = MinimalPipelineSpec(
            "persist current exact capture",
            binding.capture,
            BlockId("capture-artifact-test"),
        )
        self.triggered = TriggeredCaptureSpec(
            pipeline,
            binding.pulse_port,
            binding.pulse_request,
            binding.trigger_channel,
            binding.cell_plan,
        )
        self.captures_root = tmp_path / "captures"
        self.resources = ResourceArbiter()
        self.controller = RunController(self.resources)
        self.handle = None

    def run(self) -> CaptureArtifactRef:
        self.handle = self.controller.start(
            compile_capture_artifact_pipeline(
                self.triggered,
                self.captures_root,
            )
        )
        return self.handle.result(5.0)

    def close(self) -> None:
        assert self.controller.shutdown(2.0)
        self.broker.shutdown()
        self.resources.shutdown()
        self.sequencer.close()
        self.camera.close()


def test_exact_pipeline_commits_and_reloads_multidimensional_capture(tmp_path) -> None:
    case = _CaptureCase(tmp_path)
    try:
        reference = case.run()
        assert isinstance(reference, CaptureArtifactRef)
        artifact = load_capture_artifact(case.captures_root, reference)
        block = artifact.materialize_snapshot().block
        assert block.values.shape == (1, 3, 3, 4)
        assert tuple(
            axis.axis_id.value for axis in block.schema.cell_schema.data_axes
        ) == ("camera.y", "camera.x")
        assert np.array_equal(
            block.values[0, 0, 0],
            np.array((1200, 800, 520, 340), dtype=np.uint16),
        )
        assert tuple(
            metadata.source_ordinal
            for _cell, metadata in artifact.frame_source.iter_event_records()
        ) == (0, 1, 2)
        assert artifact.pulse_evidence is not None
        assert artifact.pulse_evidence.expected_trigger_count == 3
    finally:
        case.close()


def test_capture_record_is_the_only_visibility_boundary(
    tmp_path,
    monkeypatch,
) -> None:
    record_writes = []
    write_record = capture_artifact_impl.atomic_write_text

    def assert_payload_precedes_record(path, text):
        directory = Path(path).parent
        assert (directory / "frames.npy").is_file()
        assert (directory / "pulse.bin").is_file()
        record_writes.append(Path(path))
        return write_record(path, text)

    monkeypatch.setattr(
        capture_artifact_impl,
        "atomic_write_text",
        assert_payload_precedes_record,
    )
    case = _CaptureCase(tmp_path)
    try:
        reference = case.run()
        record_path = case.captures_root / reference.record_path
        assert record_writes == [record_path]
        record = json.loads(record_path.read_text(encoding="utf-8"))
        assert set(record) == {
            "schema",
            "run_id",
            "frames",
            "provenance",
            "terminal",
            "camera_provenance",
            "camera_capability_evidence",
            "camera_arm_spec",
            "compiled_pulse_file",
            "pulse_evidence",
        }
        assert record["camera_provenance"]["binding_stamp"][
            "physical_identity"
        ] == {
            "stable_device_identity": "fixture-camera",
            "evidence_kind": "INSTALLATION_ASSERTED_ENDPOINT",
        }
        assert (record_path.parent / "frames.npy").is_file()
        assert (record_path.parent / "pulse.bin").is_file()
        frames = np.load(record_path.parent / "frames.npy", allow_pickle=False)
        assert frames.dtype == np.dtype("<u2")
    finally:
        case.close()


def test_failed_terminal_count_never_publishes_a_capture_record(tmp_path) -> None:
    case = _CaptureCase(tmp_path, camera=_Camera(terminal_count_delta=-1))
    try:
        with pytest.raises(RunFailed, match="terminal"):
            case.run()
        assert not tuple((tmp_path / "captures").glob("*/capture.json"))
    finally:
        case.close()


def test_lazy_frame_read_rejects_a_broken_array(tmp_path) -> None:
    case = _CaptureCase(tmp_path)
    try:
        reference = case.run()
        artifact = load_capture_artifact(case.captures_root, reference)
        (case.captures_root / reference.record_path).parent.joinpath(
            "frames.npy"
        ).write_bytes(b"broken")
        with pytest.raises((OSError, ValueError)):
            artifact.materialize_snapshot()
    finally:
        case.close()


def test_capture_ref_has_one_strict_leaf_owner() -> None:
    reference = CaptureArtifactRef("run-001/capture.json")
    assert capture_artifact_ref_from_tree(
        capture_artifact_ref_to_tree(reference)
    ) == reference
    tree = {
        "schema": "zlc_neutral_atom.capture-artifact-ref",
        "record_path": reference.record_path,
    }
    assert capture_artifact_ref_from_tree(tree) == reference
    with pytest.raises(ValueError, match="unknown field"):
        capture_artifact_ref_from_tree({**tree, "legacy_generation": 1})
def test_capture_record_unknown_fields_are_rejected(tmp_path) -> None:
    case = _CaptureCase(tmp_path)
    try:
        reference = case.run()
        record_path = case.captures_root / reference.record_path
        record = json.loads(record_path.read_text(encoding="utf-8"))
        record["legacy_checkpoint_kind"] = "CHECKPOINT"
        record_path.write_text(json.dumps(record), encoding="utf-8")
        with pytest.raises(ValueError, match="CaptureArtifact"):
            load_capture_artifact(case.captures_root, reference)
    finally:
        case.close()


def test_capture_provenance_rejects_a_legacy_identity_digest(tmp_path) -> None:
    case = _CaptureCase(tmp_path)
    try:
        reference = case.run()
        record_path = case.captures_root / reference.record_path
        record = json.loads(record_path.read_text(encoding="utf-8"))
        record["camera_provenance"]["binding_stamp"]["physical_identity"][
            "evidence_digest"
        ] = "retired-identity-digest"
        record_path.write_text(json.dumps(record), encoding="utf-8")
        with pytest.raises(ValueError, match="physical device identity"):
            load_capture_artifact(case.captures_root, reference)
    finally:
        case.close()
