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
from zlc_data import (
    AxisId,
    AxisSpec,
    BlockId,
    DatasetSchema,
    PointLayout,
    READOUT_EVENT,
    REPEAT,
    SCAN_POINT,
)
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
from zlc_neutral_atom.timing._coordination import (
    execute_autonomous_single_fire,
    run_cleanup_steps,
    validate_single_trigger_capture_binding,
)
from zlc_neutral_atom.timing.capture import (
    TriggeredCaptureSpec,
    TriggeredPipelineResult,
    compile_triggered_pipeline,
)
from zlc_neutral_atom.timing.capture_plan import compile_capture_cell_plan
from zlc_neutral_atom.timing.lineage import PulseCaptureBinding, PulseCaptureLineage
from zlc_neutral_atom.timing.pulse import (
    FinitePulseExecutionRequest,
    PulseTerminalEvidenceKind,
    SimulatedPulseReceipt,
)
from zlc_pulse import (
    PulseExecutionForm,
    PulseFieldRef,
    RepeatRegion,
    ScanParameter,
    compile_pulse_artifact,
    freeze_scan_table,
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


def test_single_fire_failure_attempts_both_software_poisons():
    calls = []
    primary = RuntimeError("pulse prepare failed")

    class Pulse:
        def prepare(self, _context):
            calls.append("prepare")
            raise primary

        def fail(self):
            calls.append("pulse-fail")

    class Capture:
        def fail(self, error):
            assert error is primary
            calls.append("capture-fail")
            raise RuntimeError("capture poison failed")

    with pytest.raises(RuntimeError) as failure:
        execute_autonomous_single_fire(
            object(),
            pulse=Pulse(),
            capture=Capture(),
        )

    assert failure.value is primary
    assert calls == ["prepare", "capture-fail", "pulse-fail"]


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
            PipelineMemoryProfile(8 << 20),
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


def _contract_with_required_trigger_interval(capture, seconds):
    """Re-mint the canonical evidence chain with one changed physical fact."""

    contract = capture.measurement.capture_contract
    evidence = contract.capability.camera_capability_evidence
    assert evidence is not None
    facts = replace(
        evidence.physical_facts,
        required_external_trigger_interval_seconds=seconds,
    )
    evidence = replace(evidence, physical_facts=facts)
    capability = replace(
        contract.capability,
        capability_fingerprint=evidence.fingerprint,
        camera_capability_evidence=evidence,
    )
    provenance = contract.camera_provenance
    assert provenance is not None
    provenance = replace(
        provenance,
        capability_fingerprint=evidence.fingerprint,
    )
    return replace(
        contract,
        capability=capability,
        camera_provenance=provenance,
    )


def test_trigger_interval_gate_rejects_only_strictly_shorter_schedules():
    runtime, _camera, sequencer, capture, _document, _port, artifact, plan = (
        _runtime()
    )
    assert plan is not None
    schedule = artifact.trigger_schedules[0]
    assert schedule.minimum_interval_ticks is not None
    actual = schedule.minimum_interval_ticks / artifact.target_ir.clock_hz
    pulse_binding = PulseCaptureBinding(artifact, "ch11", plan)
    try:
        for required in (0.0, actual / 2, actual):
            accepted = validate_single_trigger_capture_binding(
                capture_spec=capture.measurement.capture_spec,
                contract=_contract_with_required_trigger_interval(
                    capture,
                    required,
                ),
                pulse_binding=pulse_binding,
            )
            assert accepted is schedule

        with pytest.raises(ValueError, match="shorter.*required external trigger"):
            validate_single_trigger_capture_binding(
                capture_spec=capture.measurement.capture_spec,
                contract=_contract_with_required_trigger_interval(
                    capture,
                    float(np.nextafter(actual, np.inf)),
                ),
                pulse_binding=pulse_binding,
            )
        assert sequencer.history == []
    finally:
        assert runtime.shutdown(timeout=2.0)


def test_single_camera_trigger_edge_has_no_interval_to_reject():
    runtime, _camera, sequencer, capture, document, _port, _artifact, _plan = (
        _runtime(point_count=1)
    )
    trigger_index = document.target.raw_lanes.index("ch11")
    periods = []
    for index, period in enumerate(document.periods):
        states = list(period.states)
        if index > 1:
            states[trigger_index] = 0
        periods.append(replace(period, states=tuple(states)))
    single_edge_document = replace(document, periods=tuple(periods))
    single_edge_artifact = compile_pulse_artifact(
        single_edge_document,
        clock_hz=sequencer.clock_hz,
        execution_form=PulseExecutionForm.STATIC_ONCE,
        trigger_channels=("ch11",),
        live_target=single_edge_document.target,
    )
    assert single_edge_artifact.trigger_schedules[0].minimum_interval_ticks is None
    single_edge_plan = compile_capture_cell_plan(
        single_edge_artifact,
        "ch11",
        capture.measurement.capture_contract.dataset_schema,
        readout_event_axis_id=AxisId("frame"),
        scan_point_layout=PointLayout.rect_c(()),
    )
    try:
        accepted = validate_single_trigger_capture_binding(
            capture_spec=capture.measurement.capture_spec,
            contract=_contract_with_required_trigger_interval(capture, 1000.0),
            pulse_binding=PulseCaptureBinding(
                single_edge_artifact,
                "ch11",
                single_edge_plan,
            ),
        )
        assert accepted.total == 1
        assert sequencer.history == []
    finally:
        assert runtime.shutdown(timeout=2.0)


def test_trigger_interval_gate_includes_the_shortest_scan_point_boundary():
    runtime, _camera, sequencer, capture, document, _port, _artifact, plan = (
        _runtime()
    )
    assert plan is not None
    document = replace(
        document,
        periods=document.periods[:3],
        api_parameters=(),
    )
    first = document.periods[0]
    parameter = ScanParameter(
        "capture_scan",
        PulseFieldRef("duration", first.period_id),
        "capture scan",
        first.unit,
    )
    document = replace(document, scan_parameters=(parameter,))
    table, _report = freeze_scan_table(
        document,
        (parameter.parameter_id,),
        ((0.003,), (0.0002,), (0.004,)),
    )
    document = replace(document, scan_table=table)
    artifact = compile_pulse_artifact(
        document,
        clock_hz=sequencer.clock_hz,
        execution_form=PulseExecutionForm.AUTONOMOUS_SCAN_ONCE,
        trigger_channels=("ch11",),
        live_target=document.target,
    )
    schedule = artifact.trigger_schedules[0]
    assert tuple(edge.point_index for edge in schedule.edges) == (0, 1, 2)
    intervals = tuple(
        right.tick_from_run_start - left.tick_from_run_start
        for left, right in zip(schedule.edges, schedule.edges[1:])
    )
    assert schedule.minimum_interval_ticks == min(intervals)
    required = float(
        np.nextafter(
            min(intervals) / artifact.target_ir.clock_hz,
            np.inf,
        )
    )
    current_schema = capture.measurement.capture_contract.dataset_schema
    scan_axis = _axis("scan", SCAN_POINT, 3)
    event_axis = _axis("frame", READOUT_EVENT, 1)
    schedule_schema = DatasetSchema(
        current_schema.repeat_axis,
        (scan_axis, event_axis),
        PointLayout.rect_c((3, 1)),
        current_schema.cell_schema,
    )
    plan = compile_capture_cell_plan(
        artifact,
        "ch11",
        schedule_schema,
        readout_event_axis_id=event_axis.axis_id,
        scan_point_layout=PointLayout.rect_c((3,)),
    )
    try:
        with pytest.raises(ValueError, match="shorter.*required external trigger"):
            validate_single_trigger_capture_binding(
                capture_spec=capture.measurement.capture_spec,
                contract=_contract_with_required_trigger_interval(capture, required),
                pulse_binding=PulseCaptureBinding(artifact, "ch11", plan),
            )
        assert sequencer.history == []
    finally:
        assert runtime.shutdown(timeout=2.0)


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
        assert result.capture.dataset.block.values.shape == (1, 3, 6, 8)
        assert np.all(result.capture.dataset.block.validity)
        assert [item.source_ordinal for item in result.capture.dataset.event_metadata] == [0, 1, 2]
        assert [item.produced_count for item in result.capture.dataset.event_metadata] == [1, 2, 3]
        assert result.capture.capture_terminal.produced_count == 3
        assert (
            result.lineage.terminal.expected_trigger_counts_from_completed_schedule
            == (("ch11", 3),)
        )
        assert result.lineage.terminal.evidence_kind is PulseTerminalEvidenceKind.SIMULATED
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
                lineage=result.lineage,
            )
        tampered_receipt = replace(
            result.lineage.terminal.receipt,
            expected_trigger_counts_from_completed_schedule=(("ch11", 2),),
        )
        with pytest.raises(ValueError, match="expected counts differ"):
            PulseCaptureLineage(
                result.lineage.binding,
                replace(result.lineage.terminal, receipt=tampered_receipt),
            )
        with pytest.raises(ValueError, match="trigger channel differs"):
            PulseCaptureBinding(
                artifact,
                "ch11",
                replace(plan, trigger_channel="ch12"),
            )
        with pytest.raises(ValueError, match="schedule digest differs"):
            PulseCaptureBinding(
                artifact,
                "ch11",
                replace(plan, trigger_schedule_digest="0" * 64),
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
            assert current.session_id != first.lineage.terminal.session_id
            return first.lineage.terminal

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
                first.lineage.terminal,
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
        assert result.capture.dataset.block.values.shape == (12, 3, 6, 8)
        assert result.capture.capture_terminal.produced_count == 36
        assert result.capture.capture_terminal.drained_count == 36
        assert tuple(item.source_ordinal for item in result.capture.dataset.event_metadata) == tuple(
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
        assert isinstance(lineage, PulseCaptureLineage)
        assert lineage.compiled_artifact.fingerprint == artifact.fingerprint
        assert lineage.compiled_artifact.source_document_digest == document.fingerprint
        assert lineage.compiled_artifact.execution_form is PulseExecutionForm.STATIC_ONCE
        assert lineage.trigger_channel == "ch11"
        assert lineage.expected_trigger_count == 3
        assert lineage.cell_plan == plan
        assert (
            lineage.cell_plan.dataset_schema_fingerprint
            == stored.frame_source.schema.fingerprint
        )
        assert lineage.cell_plan.cell_permutation_digest == stored.provenance.join_plan_digest
        assert isinstance(lineage.terminal.receipt, SimulatedPulseReceipt)
        assert lineage.terminal.evidence_kind is PulseTerminalEvidenceKind.SIMULATED
        assert stored.terminal.produced_count == lineage.expected_trigger_count
        assert (
            stored.camera_provenance
            == capture.measurement.capture_contract.camera_provenance
        )
        assert stored.frame_source.cell_schedule == plan.expected_cells
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
