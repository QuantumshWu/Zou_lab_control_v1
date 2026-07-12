"""Camera arm and one finite FPGA fire share one flat exact RunPlan."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from Zou_lab_control.neutral_atom.devices.registry import DeviceSet
from Zou_lab_control.neutral_atom.devices.virtual import (
    VirtualCamera,
    VirtualSequencer,
    VirtualTrapArray,
)
from zlc_data import AxisId, AxisSpec, BlockId, PointLayout, REPEAT, SCAN_POINT
from zlc_neutral_atom.acquisition import CameraAcquisitionMode
from zlc_neutral_atom.artifacts import (
    CaptureRepository,
    compile_capture_artifact_pipeline,
)
from zlc_neutral_atom.runtime import (
    DatasetMaterializerSpec,
    MinimalPipelineSpec,
    PipelineMemoryProfile,
)
from zlc_neutral_atom.timing import (
    FinitePulseExecutionRequest,
    TriggeredCaptureSpec,
    compile_triggered_pipeline,
)
from zlc_pulse import (
    PulseExecutionForm,
    bind_pulse_document_target,
    compile_pulse_artifact,
    load_pulse_document,
)
from zlc_pulse.target import pulse_target_from_legacy_tree
from zlc_workbench.camera_capture import CameraCaptureBindingRequest
from zlc_workbench.legacy_neutral_atom import LegacyNeutralAtomRuntime


ROOT = Path(__file__).parents[1]


def _axis(name, role, size):
    return AxisSpec(AxisId(name), name, role, size, tuple(range(size)))


def _runtime(point_count=3):
    document = load_pulse_document(ROOT / "pulses" / "imaging_template.json")
    sequencer = VirtualSequencer(sleep_scale=0)
    document = bind_pulse_document_target(
        document,
        pulse_target_from_legacy_tree(sequencer.port_catalog.to_dict()),
    )
    trap = VirtualTrapArray(grid_shape=(2, 2), image_shape=(6, 8), seed=7)
    camera = VirtualCamera(trap, exposure=1e-3, sequencer=sequencer)
    device_set = DeviceSet(
        {"trap": trap, "sequencer": sequencer, "readout": camera},
        {
            "trap": {"type": "VirtualTrapArray", "params": {}},
            "sequencer": {"type": "VirtualSequencer", "params": {}},
            "readout": {"type": "VirtualCamera", "params": {}},
        },
    )
    runtime = LegacyNeutralAtomRuntime(device_set)
    measurement = runtime.bind_camera_measurement(
        CameraCaptureBindingRequest(
            "readout",
            _axis("repeat", REPEAT, 1),
            (_axis("frame", SCAN_POINT, point_count),),
            PointLayout.rect_c((point_count,)),
            CameraAcquisitionMode.EXTERNAL_TRIGGERED,
            0,
            4 << 20,
        )
    )
    capture = MinimalPipelineSpec(
        "finite triggered capture",
        measurement,
        DatasetMaterializerSpec(
            BlockId("triggered-capture"),
            PipelineMemoryProfile.for_current_runtime(8 << 20),
        ),
        timeout_seconds=3.0,
    )
    pulse_port = runtime.bind_sequencer_port()
    artifact = compile_pulse_artifact(
        document,
        clock_hz=sequencer.clock_hz,
        execution_form=PulseExecutionForm.STATIC_ONCE,
        trigger_channels=("emCCD",),
        live_target=document.target,
    )
    return runtime, camera, sequencer, capture, document, pulse_port, artifact


def test_camera_is_armed_before_one_fire_and_all_frames_are_exactly_materialized():
    runtime, camera, sequencer, capture, document, pulse_port, artifact = _runtime()
    armed_at_fire = []

    def observe_fire(_program):
        state = camera._recent_state()
        with state["cond"]:
            armed_at_fire.append(bool(state["armed"]))

    sequencer.add_fire_listener(observe_fire)
    spec = TriggeredCaptureSpec(
        capture,
        pulse_port,
        FinitePulseExecutionRequest(document, artifact),
        "emCCD",
    )
    try:
        result = runtime.controller.run(compile_triggered_pipeline(spec))
        assert armed_at_fire == [True]
        assert result.dataset.block.values.shape == (1, 3, 6, 8)
        assert np.all(result.dataset.block.validity)
        assert [item.source_ordinal for item in result.dataset.event_metadata] == [0, 1, 2]
        assert [item.produced_count for item in result.dataset.event_metadata] == [1, 2, 3]
        assert result.capture_terminal.produced_count == 3
        assert result.pulse_terminal.completed_schedule_trigger_counts == (("emCCD", 3),)
        assert [item["action"] for item in sequencer.history] == [
            "prepare",
            "fire",
            "wait_done",
            "safe",
        ]
        assert dict(sequencer.snapshot())["state"] == "safe"
        state = camera._recent_state()
        with state["cond"]:
            assert not state["armed"] and not state["pending"]
    finally:
        assert runtime.shutdown(timeout=2.0)


def test_trigger_cardinality_mismatch_is_rejected_before_hardware_run():
    runtime, _camera, sequencer, capture, document, pulse_port, artifact = _runtime(2)
    try:
        with pytest.raises(ValueError, match="trigger count 3"):
            TriggeredCaptureSpec(
                capture,
                pulse_port,
                FinitePulseExecutionRequest(document, artifact),
                "emCCD",
            )
        assert sequencer.history == []
    finally:
        assert runtime.shutdown(timeout=2.0)


def test_triggered_capture_artifact_persists_pulse_lineage(tmp_path):
    runtime, _camera, sequencer, capture, document, pulse_port, artifact = _runtime()
    repository = CaptureRepository(tmp_path / "captures")
    spec = TriggeredCaptureSpec(
        capture,
        pulse_port,
        FinitePulseExecutionRequest(document, artifact),
        "emCCD",
    )
    try:
        reference = runtime.controller.run(
            compile_capture_artifact_pipeline(spec, repository)
        )
        stored = repository.load(reference)
        lineage = stored.pulse_lineage
        assert lineage is not None
        assert lineage.compiled_artifact_digest == artifact.fingerprint
        assert lineage.source_document_digest == document.fingerprint
        assert lineage.execution_form is PulseExecutionForm.STATIC_ONCE
        assert lineage.trigger_channel == "emCCD"
        assert lineage.expected_trigger_count == 3
        assert lineage.terminal.logical_done
        assert stored.terminal.produced_count == lineage.expected_trigger_count
        assert dict(sequencer.snapshot())["state"] == "safe"
    finally:
        assert runtime.shutdown(timeout=2.0)
