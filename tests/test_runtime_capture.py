"""Current exact camera runtime contracts without legacy compatibility."""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
import pytest

from fpga.pulse_streamer.host.image import DEFAULT_CONFIG_PATH, default_clock_hz
from zlc_data import (
    AxisId,
    AxisSpec,
    BlockId,
    REPEAT,
)
from zlc_neutral_atom.acquisition import CameraAcquisitionMode
from zlc_neutral_atom.adapter_sdk import (
    CameraCaptureTerminalRecord,
    CameraFrameRecord,
    CameraWorkingPoint,
)
from zlc_neutral_atom.bootstrap._camera_endpoint import CameraCaptureEndpoint
from zlc_neutral_atom.bootstrap._sequencer_endpoint import (
    VirtualSequencerExecutionEndpoint,
)
from zlc_neutral_atom.bootstrap._triggered_capture import (
    TriggeredCameraLayout,
    bind_triggered_camera_acquisition,
)
from zlc_neutral_atom.bootstrap._virtual_hardware import VirtualSequencer
from zlc_neutral_atom.runtime.capture import BoundCapturePort
from zlc_neutral_atom.runtime.pipeline import MinimalPipelineSpec
from zlc_neutral_atom.runtime.ports import DeviceBroker, SafetyOperation
from zlc_neutral_atom.runtime.resources import (
    DeviceIdentityEvidenceKind,
    PhysicalDeviceIdentity,
    ResourceArbiter,
    ResourceKey,
)
from zlc_neutral_atom.runtime.run import RunCancelled, RunController, RunFailed
from zlc_neutral_atom.timing.capture import (
    TriggeredCaptureSpec,
    compile_triggered_pipeline,
)
from zlc_neutral_atom.timing.pulse import BoundPulsePort
from zlc_pulse import PulseExecutionForm, load_deployed_pulse_target, load_pulse_document
from zlc_storage import canonical_digest


_ROOT = Path(__file__).parents[1]


class _Camera:
    max_pending_records = 2
    timeout = 1.0

    def __init__(
        self,
        *,
        terminal_count_delta: int = 0,
        block_reads: bool = False,
    ) -> None:
        self.expected = 0
        self.read_index = 0
        self.armed = False
        self.terminal_count_delta = terminal_count_delta
        self.block_reads = block_reads
        self.read_entered = threading.Event()
        self._condition = threading.Condition()

    def capture_working_point(self) -> CameraWorkingPoint:
        return CameraWorkingPoint(
            canonical_digest({"fixture": "runtime-camera-working-point"}),
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
        max_inflight_frames: int,
        timeout: float,
    ) -> None:
        assert max_inflight_frames == min(frames, 2)
        assert timeout > 0
        with self._condition:
            self.expected = frames
            self.read_index = 0
            self.armed = True
            self._condition.notify_all()

    def read_frame_records(
        self,
        n: int,
        *,
        timeout: float,
        exact: bool,
    ) -> list[CameraFrameRecord]:
        assert n == 1 and exact and timeout > 0
        with self._condition:
            self.read_entered.set()
            while self.block_reads and self.armed:
                self._condition.wait(0.05)
            if not self.armed:
                raise RuntimeError("fixture camera was interrupted")
            ordinal = self.read_index
            self.read_index += 1
        image = np.full((3, 4), 10 + ordinal, dtype=np.uint16)
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
        with self._condition:
            self.armed = False
            self._condition.notify_all()
        return CameraCaptureTerminalRecord(
            self.expected + self.terminal_count_delta,
            True,
            True,
            True,
        )

    def capture_state(self) -> tuple[bool, int]:
        with self._condition:
            return self.armed, 0


class _RuntimeFixture:
    def __init__(
        self,
        tmp_path,
        *,
        camera: _Camera | None = None,
        transport_memory_limit_bytes: int = 8 << 20,
    ) -> None:
        self.camera = _Camera() if camera is None else camera
        self.broker = DeviceBroker()
        self.endpoint = CameraCaptureEndpoint(
            self.camera,
            "camera",
            exact_external_trigger_qualification_digest=canonical_digest(
                {"qualification": "deterministic fixture adapter"}
            ),
        )
        identity = PhysicalDeviceIdentity(
            "fixture-camera",
            DeviceIdentityEvidenceKind.INSTALLATION_ASSERTED_ENDPOINT,
            "fixture-camera-evidence",
            "fixture-assets-v1",
        )
        proof = self.broker.verify_identity(lambda: identity)
        binding = None

        def current():
            assert binding is not None
            return binding

        binding = self.broker.bind(
            key=ResourceKey.parse("device/camera"),
            identity=proof,
            execute_command=lambda command: self.endpoint.execute_command(
                current(),
                command,
            ),
            capability_probe=lambda: self.endpoint.capability_probe(current()),
            close_session=lambda command: self.endpoint.close_session(
                current(),
                command,
            ),
            interrupt_operations={SafetyOperation.DISARM: self.endpoint.interrupt},
        )
        capture_port = BoundCapturePort(
            self.broker.verify_capability(binding),
            (SafetyOperation.DISARM,),
        )
        pulse_target = load_deployed_pulse_target()
        self.sequencer = VirtualSequencer(
            pulse_target,
            clock_hz=default_clock_hz(DEFAULT_CONFIG_PATH),
            sleep_scale=0,
        )
        pulse_endpoint = VirtualSequencerExecutionEndpoint(self.sequencer)
        pulse_identity = PhysicalDeviceIdentity(
            "fixture-sequencer",
            DeviceIdentityEvidenceKind.INSTALLATION_ASSERTED_ENDPOINT,
            "fixture-sequencer-evidence",
            "fixture-assets-v1",
        )
        pulse_proof = self.broker.verify_identity(lambda: pulse_identity)
        pulse_binding = None

        def current_pulse():
            assert pulse_binding is not None
            return pulse_binding

        pulse_binding = self.broker.bind(
            key=ResourceKey.parse("device/sequencer"),
            identity=pulse_proof,
            execute_command=lambda command: pulse_endpoint.execute_command(
                current_pulse(),
                command,
            ),
            capability_probe=lambda: pulse_endpoint.capability_probe(current_pulse()),
            close_session=lambda command: pulse_endpoint.close_session(
                current_pulse(),
                command,
            ),
            interrupt_operations={
                SafetyOperation.SAFE_STATE: pulse_endpoint.interrupt
            },
        )
        pulse_port = BoundPulsePort(
            self.broker.verify_capability(pulse_binding),
            (),
        )
        repeat_axis = AxisSpec(AxisId("repeat"), "repeat", REPEAT, 1, (0,))
        binding_result = bind_triggered_camera_acquisition(
            pulse_port,
            capture_port,
            pulse_document=load_pulse_document(
                _ROOT / "zlc_neutral_atom" / "assets" / "imaging_template.json"
            ),
            execution_form=PulseExecutionForm.STATIC_ONCE,
            trigger_channel="ch11",
            layout=TriggeredCameraLayout(
                repeat_axis,
                AxisId("readout-event"),
                AxisId("scan-ordinal"),
                readout_events_per_repeat=3,
            ),
            transport_memory_limit_bytes=transport_memory_limit_bytes,
        )
        self.measurement = binding_result.measurement
        self.spec = MinimalPipelineSpec(
            "current direct camera capture",
            self.measurement,
            BlockId("runtime-capture"),
            16 << 20,
            timeout_seconds=2.0,
        )
        self.triggered = TriggeredCaptureSpec(
            self.spec,
            binding_result.pulse_port,
            binding_result.pulse_request,
            binding_result.trigger_channel,
            binding_result.cell_plan,
        )
        self.resources = ResourceArbiter()
        self.controller = RunController(self.resources)

    def close(self) -> None:
        assert self.controller.shutdown(2.0)
        self.broker.shutdown()
        self.resources.shutdown()


def test_direct_runtime_preserves_every_declared_data_axis(tmp_path) -> None:
    fixture = _RuntimeFixture(tmp_path)
    try:
        result = fixture.controller.start(
            compile_triggered_pipeline(fixture.triggered)
        ).result(3.0)
        capture = result.capture
        assert capture.dataset.block.values.shape == (1, 3, 3, 4)
        assert tuple(
            axis.axis_id.value
            for axis in capture.dataset.block.schema.cell_schema.data_axes
        ) == ("camera.y", "camera.x")
        assert np.all(capture.dataset.block.values[0, 0] == 10)
        assert np.all(capture.dataset.block.values[0, 1] == 11)
        assert np.all(capture.dataset.block.values[0, 2] == 12)
        assert tuple(
            item.source_ordinal for item in capture.dataset.event_metadata
        ) == (0, 1, 2)
    finally:
        fixture.close()


def test_capture_contract_has_one_capability_owner_and_result_has_no_mirrors(
    tmp_path,
) -> None:
    fixture = _RuntimeFixture(tmp_path)
    try:
        contract = fixture.measurement.capture_contract
        assert contract.capability is fixture.measurement.capture_port.capability
        result = fixture.controller.start(
            compile_triggered_pipeline(fixture.triggered)
        ).result(3.0)
        capture = result.capture
        assert capture.__slots__ == (
            "_dataset",
            "_capture_completion",
            "_direct_raw_capture",
        )
        completion = capture._capture_completion
        assert completion._session is None
        assert completion._terminal_reservation is None
        assert not hasattr(capture, "camera")
        assert not hasattr(capture, "aggregate_peak_bytes")
    finally:
        fixture.close()


def test_terminal_count_mismatch_fails_the_run_without_a_dataset(tmp_path) -> None:
    fixture = _RuntimeFixture(
        tmp_path,
        camera=_Camera(terminal_count_delta=-1),
    )
    try:
        handle = fixture.controller.start(
            compile_triggered_pipeline(fixture.triggered)
        )
        with pytest.raises(RunFailed, match="terminal"):
            handle.result(3.0)
        assert not handle.snapshot().final_committed
        assert fixture.camera.capture_state() == (False, 0)
    finally:
        fixture.close()


def test_transport_budget_is_rejected_before_hardware_arm(tmp_path) -> None:
    camera = _Camera()
    with pytest.raises(MemoryError, match="transport budget"):
        _RuntimeFixture(
            tmp_path,
            camera=camera,
            transport_memory_limit_bytes=1,
        )
    assert camera.capture_state() == (False, 0)


def test_cancel_interrupts_blocked_capture_and_releases_hardware(tmp_path) -> None:
    camera = _Camera(block_reads=True)
    fixture = _RuntimeFixture(tmp_path, camera=camera)
    try:
        handle = fixture.controller.start(
            compile_triggered_pipeline(fixture.triggered)
        )
        assert camera.read_entered.wait(1.0)
        handle.cancel("fixture cancellation")
        with pytest.raises((RunCancelled, RunFailed)):
            handle.result(3.0)
        assert camera.capture_state() == (False, 0)
    finally:
        fixture.close()


def test_compiled_plan_is_reusable_only_with_fresh_runtime_authority(tmp_path) -> None:
    fixture = _RuntimeFixture(tmp_path)
    try:
        plan = compile_triggered_pipeline(fixture.triggered)
        first = fixture.controller.start(plan).result(3.0)
        second = fixture.controller.start(plan).result(3.0)
        assert np.array_equal(
            first.capture.dataset.block.values,
            second.capture.dataset.block.values,
        )
        assert first.capture.dataset.provenance.trace_binding.run_id != (
            second.capture.dataset.provenance.trace_binding.run_id
        )
    finally:
        fixture.close()
