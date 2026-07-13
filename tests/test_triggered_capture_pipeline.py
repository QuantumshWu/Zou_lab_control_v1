"""Camera arm and one finite FPGA fire share one flat exact RunPlan."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import pickle
import threading
import time

import numpy as np
import pytest

from Zou_lab_control.neutral_atom.devices.registry import DeviceSet
from Zou_lab_control.neutral_atom.devices.virtual import (
    VirtualCamera,
    VirtualSequencer,
    VirtualTrapArray,
)
from Zou_lab_control.neutral_atom.ports import PortCatalog, PortSpec
from zlc_data import AxisId, AxisSpec, BlockId, PointLayout, READOUT_EVENT, REPEAT
from zlc_neutral_atom.acquisition import CameraAcquisitionMode
from zlc_neutral_atom.artifacts import (
    CaptureArtifactRef,
    CaptureRepository,
    compile_capture_artifact_pipeline,
)
from zlc_neutral_atom.runtime import (
    CleanupReport,
    DatasetCellAddress,
    DatasetMaterializerSpec,
    MinimalPipelineSpec,
    PipelineMemoryProfile,
    RunCancelled,
    RunFailed,
)
from zlc_neutral_atom.timing._coordination import run_cleanup_steps
from zlc_neutral_atom.timing import (
    FinitePulseExecutionRequest,
    PulseTerminalEvidenceKind,
    SimulatedPulseReceipt,
    TriggeredCaptureSpec,
    TriggeredPipelineResult,
    compile_capture_cell_plan,
    compile_triggered_pipeline,
)
from zlc_pulse import (
    PulseExecutionForm,
    RepeatRegion,
    compile_pulse_artifact,
    load_pulse_document,
)
from zlc_storage import decode, encode
from zlc_workbench.camera_capture import CameraCaptureBindingRequest
from zlc_workbench.legacy_neutral_atom import LegacyNeutralAtomRuntime


ROOT = Path(__file__).parents[1]


def _axis(name, role, size):
    return AxisSpec(AxisId(name), name, role, size, tuple(range(size)))


def test_ordered_cleanup_runs_later_physical_steps_after_an_earlier_exception():
    calls = []
    failure = RuntimeError("sequencer cleanup failed")
    later_failure = RuntimeError("camera cleanup reported an error")

    def pulse_cleanup():
        calls.append("pulse")
        raise failure

    def camera_cleanup():
        calls.append("camera")
        return CleanupReport(errors=(later_failure,))

    report = run_cleanup_steps(pulse_cleanup, camera_cleanup)

    assert calls == ["pulse", "camera"]
    assert report.errors == (failure, later_failure)


def _runtime(
    point_count=3,
    repeat_count=1,
    *,
    capture_trigger_channels=("ch11",),
):
    document = load_pulse_document(ROOT / "pulses" / "imaging_template.json")
    if repeat_count > 1:
        document = replace(
            document,
            repeat=RepeatRegion(
                document.periods[0].period_id,
                document.periods[-1].period_id,
                repeat_count,
            ),
        )
    catalog = PortCatalog(
        document.target.raw_lanes,
        tuple(
            PortSpec(
                port.key,
                port.kind,
                port.lanes,
                port.label,
                port.bus_index,
                port.width,
                port.encoding,
                port.safe_value,
                port.latch_clock,
            )
            for port in document.target.ports
        ),
    )
    sequencer = VirtualSequencer(sleep_scale=0, port_catalog=catalog)
    trap = VirtualTrapArray(grid_shape=(2, 2), image_shape=(6, 8), seed=7)
    camera = VirtualCamera(
        trap,
        exposure=1e-3,
        capture_trigger_channels=capture_trigger_channels,
        sequencer=sequencer,
    )
    device_set = DeviceSet(
        {"trap": trap, "sequencer": sequencer, "readout": camera},
        {
            "trap": {"type": "VirtualTrapArray", "params": {}},
            "sequencer": {"type": "VirtualSequencer", "params": {}},
            "readout": {"type": "VirtualCamera", "params": {}},
        },
    )
    runtime = LegacyNeutralAtomRuntime(device_set)
    description = runtime.describe_camera("readout")
    measurement = runtime.bind_camera_measurement(
        CameraCaptureBindingRequest(
            "readout",
            _axis("repeat", REPEAT, repeat_count),
            (_axis("frame", READOUT_EVENT, point_count),),
            PointLayout.rect_c((point_count,)),
            tuple(
                DatasetCellAddress(repeat, point)
                for repeat in range(repeat_count)
                for point in range(point_count)
            ),
            CameraAcquisitionMode.EXTERNAL_TRIGGERED,
            0,
            4 << 20,
            tuple(description.event_setting(index) for index in range(point_count)),
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
        trigger_channels=("ch11",),
        live_target=document.target,
    )
    plan = None
    if artifact.trigger_schedules[0].total == repeat_count * point_count:
        plan = compile_capture_cell_plan(
            artifact,
            "ch11",
            measurement.capture_contract.dataset_schema,
            readout_event_axis_id=AxisId("frame"),
            scan_point_layout=PointLayout.rect_c(()),
            within_point_grouping=(
                tuple(
                    (repeat, event)
                    for repeat in range(repeat_count)
                    for event in range(point_count)
                )
                if repeat_count > 1 and point_count > 1
                else None
            ),
        )
    return runtime, camera, sequencer, capture, document, pulse_port, artifact, plan


def test_camera_is_armed_before_one_fire_and_all_frames_are_exactly_materialized():
    runtime, camera, sequencer, capture, document, pulse_port, artifact, plan = _runtime()
    assert plan is not None
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
        "ch11",
        plan,
    )
    try:
        result = runtime.controller.run(compile_triggered_pipeline(spec))
        assert armed_at_fire == [True]
        assert result.dataset.block.values.shape == (1, 3, 6, 8)
        assert np.all(result.dataset.block.validity)
        assert [item.source_ordinal for item in result.dataset.event_metadata] == [0, 1, 2]
        assert [item.produced_count for item in result.dataset.event_metadata] == [1, 2, 3]
        assert result.capture_terminal.produced_count == 3
        assert (
            result.pulse_terminal.expected_trigger_counts_from_completed_schedule
            == (("ch11", 3),)
        )
        assert result.pulse_terminal.evidence_kind is PulseTerminalEvidenceKind.SIMULATED
        assert [item["action"] for item in sequencer.history] == [
            "prepare",
            "fire",
            "wait_done",
            "safe",
        ]
        with pytest.raises(TypeError, match="process-local"):
            pickle.dumps(result)
        with pytest.raises(PermissionError, match="only be minted"):
            TriggeredPipelineResult(
                object(),
                capture=result.capture,
                pulse_session_id=result.pulse_session_id,
                pulse_terminal=result.pulse_terminal,
                trigger_channel=result.trigger_channel,
                compiled_artifact=result.compiled_artifact,
                cell_plan=result.cell_plan,
            )
        assert dict(sequencer.snapshot())["state"] == "safe"
        state = camera._recent_state()
        with state["cond"]:
            assert not state["armed"] and not state["pending"]
    finally:
        assert runtime.shutdown(timeout=2.0)


def test_triggered_capture_rejects_a_result_from_another_run():
    runtime, _camera, _sequencer, capture, document, pulse_port, artifact, plan = (
        _runtime()
    )
    assert plan is not None
    spec = TriggeredCaptureSpec(
        capture,
        pulse_port,
        FinitePulseExecutionRequest(document, artifact),
        "ch11",
        plan,
    )
    compiled = compile_triggered_pipeline(spec)
    try:
        first = runtime.controller.run(compiled)
        stale_plan = replace(
            compiled,
            execute=lambda _context, _prepared: first,
        )
        with pytest.raises(RunFailed) as failure:
            runtime.controller.run(stale_plan)

        assert failure.value.primary is not None
        assert failure.value.primary.original_type == "ValueError"
        assert "another Run" in str(failure.value.primary)
    finally:
        assert runtime.shutdown(timeout=2.0)


def test_triggered_capture_rejects_a_terminal_from_another_pulse_session(
    monkeypatch,
):
    import zlc_neutral_atom.timing.capture as timing_capture

    runtime, _camera, _sequencer, capture, document, pulse_port, artifact, plan = (
        _runtime()
    )
    assert plan is not None
    spec = TriggeredCaptureSpec(
        capture,
        pulse_port,
        FinitePulseExecutionRequest(document, artifact),
        "ch11",
        plan,
    )
    compiled = compile_triggered_pipeline(spec)
    try:
        first = runtime.controller.run(compiled)
        original_complete = timing_capture.PulseSession.complete

        def stale_complete(self, context):
            current = original_complete(self, context)
            assert current.session_id != first.pulse_terminal.session_id
            return first.pulse_terminal

        monkeypatch.setattr(
            timing_capture.PulseSession,
            "complete",
            stale_complete,
        )
        with pytest.raises(RunFailed) as failure:
            runtime.controller.run(compiled)

        assert failure.value.primary is not None
        assert failure.value.primary.original_type == "PermissionError"
        assert "not minted" in str(failure.value.primary)
    finally:
        assert runtime.shutdown(timeout=2.0)


def test_triggered_capture_rejects_a_stale_receipt_rebadged_to_current_session(
    monkeypatch,
):
    import zlc_neutral_atom.timing.capture as timing_capture

    runtime, _camera, _sequencer, capture, document, pulse_port, artifact, plan = (
        _runtime()
    )
    assert plan is not None
    spec = TriggeredCaptureSpec(
        capture,
        pulse_port,
        FinitePulseExecutionRequest(document, artifact),
        "ch11",
        plan,
    )
    compiled = compile_triggered_pipeline(spec)
    try:
        first = runtime.controller.run(compiled)
        original_complete = timing_capture.PulseSession.complete

        def rebadged_complete(self, context):
            current = original_complete(self, context)
            return replace(
                first.pulse_terminal,
                session_id=current.session_id,
            )

        monkeypatch.setattr(
            timing_capture.PulseSession,
            "complete",
            rebadged_complete,
        )
        with pytest.raises(RunFailed) as failure:
            runtime.controller.run(compiled)

        assert failure.value.primary is not None
        assert failure.value.primary.original_type == "PermissionError"
        assert "not minted" in str(failure.value.primary)
    finally:
        assert runtime.shutdown(timeout=2.0)


def test_triggered_capture_cancel_after_fire_cleans_pulse_before_camera(
    monkeypatch,
):
    import zlc_neutral_atom.timing.capture as timing_capture

    runtime, camera, sequencer, capture, document, pulse_port, artifact, plan = (
        _runtime()
    )
    assert plan is not None
    spec = TriggeredCaptureSpec(
        capture,
        pulse_port,
        FinitePulseExecutionRequest(document, artifact),
        "ch11",
        plan,
    )
    entered_capture = threading.Event()
    cleanup_order = []
    original_pulse_cleanup = timing_capture.PulseSession.cleanup
    original_capture_cleanup = timing_capture.ExactCaptureTransaction.cleanup

    def blocked_capture_all(self, context):
        entered_capture.set()
        while True:
            context.checkpoint()
            time.sleep(0.005)

    def observed_pulse_cleanup(self, context):
        cleanup_order.append("pulse")
        return original_pulse_cleanup(self, context)

    def observed_capture_cleanup(self, context):
        cleanup_order.append("camera")
        return original_capture_cleanup(self, context)

    monkeypatch.setattr(
        timing_capture.ExactCaptureTransaction,
        "capture_all",
        blocked_capture_all,
    )
    monkeypatch.setattr(
        timing_capture.PulseSession,
        "cleanup",
        observed_pulse_cleanup,
    )
    monkeypatch.setattr(
        timing_capture.ExactCaptureTransaction,
        "cleanup",
        observed_capture_cleanup,
    )
    sequencer.history.clear()
    try:
        handle = runtime.controller.start(compile_triggered_pipeline(spec))
        assert entered_capture.wait(3.0)
        handle.cancel("cancel raw exact capture after FIRE")
        with pytest.raises(RunCancelled):
            handle.result(10.0)

        assert cleanup_order == ["pulse", "camera"]
        assert "fire" in {item["action"] for item in sequencer.history}
        assert dict(sequencer.snapshot())["state"] == "safe"
        state = camera._recent_state()
        with state["cond"]:
            assert not state["armed"] and not state["pending"]
    finally:
        assert runtime.shutdown(timeout=2.0)


def test_repeated_three_event_capture_streams_all_frames_through_bounded_ring():
    runtime, camera, _sequencer, capture, document, pulse_port, artifact, plan = _runtime(
        repeat_count=12
    )
    assert plan is not None
    assert artifact.trigger_schedules[0].loop_count == 12
    assert plan.within_point_grouping == tuple(
        (repeat, event)
        for repeat in range(12)
        for event in range(3)
    )
    spec = TriggeredCaptureSpec(
        capture,
        pulse_port,
        FinitePulseExecutionRequest(document, artifact),
        "ch11",
        plan,
    )
    try:
        result = runtime.controller.run(compile_triggered_pipeline(spec))
        assert result.dataset.block.values.shape == (12, 3, 6, 8)
        assert result.capture_terminal.produced_count == 36
        assert result.capture_terminal.drained_count == 36
        assert tuple(item.source_ordinal for item in result.dataset.event_metadata) == tuple(
            range(36)
        )
        state = camera._recent_state()
        with state["cond"]:
            assert state["pending_capacity"] < 36
            assert not state["armed"] and not state["pending"]
    finally:
        assert runtime.shutdown(timeout=2.0)


def test_trigger_cardinality_mismatch_is_rejected_before_hardware_run():
    runtime, _camera, sequencer, capture, document, pulse_port, artifact, plan = _runtime(2)
    assert plan is None
    try:
        with pytest.raises(ValueError, match=r"R \* E"):
            compile_capture_cell_plan(
                artifact,
                "ch11",
                capture.measurement.capture_contract.dataset_schema,
                readout_event_axis_id=AxisId("frame"),
                scan_point_layout=PointLayout.rect_c(()),
            )
        assert sequencer.history == []
    finally:
        assert runtime.shutdown(timeout=2.0)


def test_triggered_capture_rejects_a_schedule_on_an_unwired_camera_channel():
    runtime, _camera, sequencer, capture, document, pulse_port, _artifact, _plan = (
        _runtime()
    )
    source_index = document.target.raw_lanes.index("ch11")
    unwired_index = document.target.raw_lanes.index("ch12")
    periods = []
    for period in document.periods:
        states = list(period.states)
        states[unwired_index] = states[source_index]
        states[source_index] = 0
        periods.append(replace(period, states=tuple(states)))
    unwired_document = replace(document, periods=tuple(periods))
    unwired_artifact = compile_pulse_artifact(
        unwired_document,
        clock_hz=sequencer.clock_hz,
        execution_form=PulseExecutionForm.STATIC_ONCE,
        trigger_channels=("ch12",),
        live_target=unwired_document.target,
    )
    unwired_plan = compile_capture_cell_plan(
        unwired_artifact,
        "ch12",
        capture.measurement.capture_contract.dataset_schema,
        readout_event_axis_id=AxisId("frame"),
        scan_point_layout=PointLayout.rect_c(()),
    )
    try:
        with pytest.raises(ValueError, match="not.*wired"):
            TriggeredCaptureSpec(
                capture,
                pulse_port,
                FinitePulseExecutionRequest(unwired_document, unwired_artifact),
                "ch12",
                unwired_plan,
            )
        assert sequencer.history == []
    finally:
        assert runtime.shutdown(timeout=2.0)


def test_exact_triggered_capture_rejects_multi_line_camera_wiring():
    runtime, _camera, sequencer, capture, document, pulse_port, artifact, plan = (
        _runtime(capture_trigger_channels=("ch11", "ch12"))
    )
    assert plan is not None
    try:
        with pytest.raises(ValueError, match="exactly one"):
            TriggeredCaptureSpec(
                capture,
                pulse_port,
                FinitePulseExecutionRequest(document, artifact),
                "ch11",
                plan,
            )
        assert sequencer.history == []
    finally:
        assert runtime.shutdown(timeout=2.0)


def test_triggered_capture_artifact_persists_pulse_lineage(tmp_path):
    runtime, _camera, sequencer, capture, document, pulse_port, artifact, plan = _runtime()
    assert plan is not None
    repository = CaptureRepository(tmp_path / "captures")
    spec = TriggeredCaptureSpec(
        capture,
        pulse_port,
        FinitePulseExecutionRequest(document, artifact),
        "ch11",
        plan,
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
        assert lineage.trigger_channel == "ch11"
        assert lineage.expected_trigger_count == 3
        assert lineage.cell_plan == plan
        assert lineage.cell_plan.dataset_schema_fingerprint == stored.block.schema.fingerprint
        assert lineage.cell_plan.cell_permutation_digest == stored.provenance.join_plan_digest
        assert isinstance(lineage.terminal.receipt, SimulatedPulseReceipt)
        assert lineage.terminal.evidence_kind is PulseTerminalEvidenceKind.SIMULATED
        assert stored.terminal.produced_count == lineage.expected_trigger_count
        assert (
            stored.camera_provenance
            == capture.measurement.capture_contract.camera_provenance
        )
        assert stored.source_cell_schedule == plan.expected_cells
        assert stored.run_id == stored.provenance.trace_binding.run_id
        assert stored.safety_bundle_id is not None
        assert len(stored.chain_contract_digest) == 64
        assert dict(sequencer.snapshot())["state"] == "safe"

        forged_facts = replace(
            stored.camera_capability_evidence.physical_facts,
            capture_trigger_channels=("ch12",),
        )
        forged_evidence = replace(
            stored.camera_capability_evidence,
            physical_facts=forged_facts,
        )
        with pytest.raises(ValueError, match="not.*wired"):
            replace(
                stored,
                camera_capability_evidence=forged_evidence,
                terminal=replace(
                    stored.terminal,
                    capability_fingerprint=forged_evidence.fingerprint,
                ),
                camera_provenance=replace(
                    stored.camera_provenance,
                    capability_fingerprint=forged_evidence.fingerprint,
                ),
            )

        manifest = decode(
            repository._store.read_manifest("capture", reference.manifest_digest)
        )
        manifest["camera_capability_evidence"]["physical_facts"][
            "capture_trigger_channels"
        ] = ["ch12"]
        manifest["camera_capability_evidence"][
            "physical_facts_fingerprint"
        ] = forged_facts.fingerprint
        manifest["terminal"][
            "capability_fingerprint"
        ] = forged_evidence.fingerprint
        manifest["camera_provenance"][
            "capability_fingerprint"
        ] = forged_evidence.fingerprint
        forged_manifest = repository._store.publish_manifest(
            "capture",
            encode(manifest),
        )
        with pytest.raises(ValueError, match="not.*wired"):
            repository.load(
                CaptureArtifactRef(
                    repository.repository_id,
                    forged_manifest.content.digest,
                )
            )
    finally:
        assert runtime.shutdown(timeout=2.0)
