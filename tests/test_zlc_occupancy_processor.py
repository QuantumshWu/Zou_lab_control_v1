"""Vertical contracts for admitted-calibration camera occupancy.

The fixture deliberately crosses the real capture and calibration repositories.
Tests assert physical outcomes and public result surfaces; they do not reproduce
the processor's formulas, proof objects, or internal authority machinery.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
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
    READOUT_EVENT,
    REPEAT,
    SCAN_POINT,
    AxisId,
    AxisSpec,
    BlockId,
    ComponentValidity,
    PointLayout,
    VALID,
    Value,
)
from zlc_neutral_atom.acquisition import (
    CameraAcquisitionMode,
    CameraFrameMetadata,
    CameraSample,
)
from zlc_neutral_atom.artifacts import (
    CaptureArtifactRef,
    CaptureRepository,
    compile_capture_artifact_pipeline,
)
from zlc_neutral_atom.readout.analysis import CalibrationAnalysisRequest
from zlc_neutral_atom.readout.calibration import (
    BoxReducer,
    ReadoutModelKind,
    ResolvedCalibration,
)
from zlc_neutral_atom.readout.calibration_reference import CalibrationArtifactRef
from zlc_neutral_atom.readout.calibration_repository import (
    CalibrationRepository,
    compile_calibration_artifact_plan,
)
from zlc_neutral_atom.readout.contracts import CalibrationCaptureLayout
from zlc_neutral_atom.readout.occupancy import (
    OccupancyStreamProcessorSpec,
    bind_occupancy_stream_processor,
)
from zlc_neutral_atom.readout.occupancy_pipeline import (
    OccupancyPipelineResult,
    OccupancyPipelineSpec,
    compile_occupancy_pipeline,
)
from zlc_neutral_atom.runtime import (
    DatasetCellAddress,
    DatasetMaterializerSpec,
    MinimalPipelineSpec,
    PipelineMemoryProfile,
    RunCancelled,
    RunFailed,
)
from zlc_neutral_atom.runtime.streams import StreamId, TraceBinding
from zlc_neutral_atom.timing.capture_plan import compile_capture_cell_plan
from zlc_neutral_atom.timing.capture import TriggeredCaptureSpec
from zlc_neutral_atom.timing.occupancy import (
    TriggeredOccupancyPipelineResult,
    TriggeredOccupancySpec,
    compile_triggered_occupancy_pipeline,
)
from zlc_neutral_atom.timing.pulse import FinitePulseExecutionRequest
from zlc_pulse import (
    PulseExecutionForm,
    RepeatRegion,
    compile_pulse_artifact,
    load_pulse_document,
)
from zlc_workbench.camera_capture import CameraCaptureBindingRequest
from zlc_workbench.legacy_neutral_atom import LegacyNeutralAtomRuntime


ROOT = Path(__file__).parents[1]
_CENTERS = ((7, 7), (24, 7), (7, 24), (24, 24))
_SPOT = np.array(
    ((0.42, 0.60, 0.42), (0.60, 1.00, 0.60), (0.42, 0.60, 0.42)),
    dtype=np.float64,
)


def _axis(name: str, role, size: int) -> AxisSpec:
    return AxisSpec(AxisId(name), name, role, size, tuple(range(size)))


def _frame(repeat: int, event: int, point: int = 0) -> np.ndarray:
    """A known four-site scene; site 3 has deliberately reversed readout polarity."""

    image = np.zeros((32, 32), dtype=np.uint16)
    for site, (x, y) in enumerate(_CENTERS):
        occupied = (repeat + point + site) % 2 == 0
        if event in (1, 2):
            level = 2000.0 if occupied else 200.0
        elif site == 3:
            level = 100.0 if occupied else 1000.0
        else:
            level = 1000.0 if occupied else 100.0
        image[y - 1 : y + 2, x - 1 : x + 2] = np.rint(
            level * _SPOT
        ).astype(np.uint16)
    return image


def _deliver_when_armed(camera: VirtualCamera, images: list[np.ndarray]):
    """Drive the virtual sensor only after the real capture session is armed."""

    failures: list[BaseException] = []

    def source() -> None:
        try:
            deadline = time.monotonic() + 5.0
            state = camera._recent_state()
            with state["cond"]:
                while not state["armed"]:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("camera was not armed")
                    state["cond"].wait(remaining)
            camera._deliver(images)
        except BaseException as error:
            failures.append(error)

    thread = threading.Thread(target=source, daemon=False)
    thread.start()
    return thread, failures


@dataclass(frozen=True)
class _Rig:
    runtime: LegacyNeutralAtomRuntime
    camera: VirtualCamera
    sequencer: VirtualSequencer
    capture_repository: CaptureRepository
    capture_ref: CaptureArtifactRef
    calibration_repository: CalibrationRepository
    calibration_ref: CalibrationArtifactRef
    admitted: ResolvedCalibration


def _pulse_catalog() -> PortCatalog:
    document = load_pulse_document(ROOT / "pulses" / "imaging_template.json")
    return PortCatalog(
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


@pytest.fixture(scope="module")
def rig(tmp_path_factory):
    root = tmp_path_factory.mktemp("occupancy-integration")
    sequencer = VirtualSequencer(sleep_scale=0, port_catalog=_pulse_catalog())
    trap = VirtualTrapArray(grid_shape=(2, 2), image_shape=(32, 32), seed=11)
    camera = VirtualCamera(
        trap,
        exposure=1e-3,
        capture_trigger_channels=("ch11",),
    )
    camera.recent_capacity = 256
    runtime = LegacyNeutralAtomRuntime(
        DeviceSet(
            {"trap": trap, "readout": camera, "sequencer": sequencer},
            {
                "trap": {"type": "VirtualTrapArray", "params": {}},
                "readout": {"type": "VirtualCamera", "params": {}},
                "sequencer": {"type": "VirtualSequencer", "params": {}},
            },
        )
    )
    capture_repository = CaptureRepository(root / "captures", repository_id="captures")
    calibration_repository = CalibrationRepository(
        root / "calibrations",
        repository_id="calibrations",
    )
    source_thread = None
    try:
        description = runtime.describe_camera("readout")
        repeat_axis = _axis("calibration-repeat", REPEAT, 24)
        event_axis = _axis("calibration-event", READOUT_EVENT, 3)
        context_axis = _axis("calibration-context", SCAN_POINT, 1)
        layout = PointLayout.rect_c((3, 1))
        cells = tuple(
            DatasetCellAddress(repeat, point)
            for repeat in range(repeat_axis.size)
            for point in range(layout.storage_size)
        )
        measurement = runtime.bind_camera_measurement(
            CameraCaptureBindingRequest(
                "readout",
                repeat_axis,
                (event_axis, context_axis),
                layout,
                cells,
                CameraAcquisitionMode.EXTERNAL_TRIGGERED,
                len(cells),
                64 << 20,
                tuple(description.event_setting(index) for index in range(3)),
            )
        )
        capture = MinimalPipelineSpec(
                "occupancy calibration source",
                measurement,
                DatasetMaterializerSpec(
                    BlockId("occupancy-calibration-source"),
                    PipelineMemoryProfile(128 << 20),
                ),
        )
        document = load_pulse_document(ROOT / "pulses" / "imaging_template.json")
        document = replace(
            document,
            repeat=RepeatRegion(
                document.periods[0].period_id,
                document.periods[-1].period_id,
                repeat_axis.size,
            ),
        )
        pulse_port = runtime.bind_sequencer_port()
        pulse_artifact = compile_pulse_artifact(
            document,
            clock_hz=pulse_port.capability.clock_hz,
            execution_form=PulseExecutionForm.STATIC_ONCE,
            trigger_channels=("ch11",),
            live_target=pulse_port.capability.target,
        )
        cell_plan = compile_capture_cell_plan(
            pulse_artifact,
            "ch11",
            measurement.capture_contract.dataset_schema,
            readout_event_axis_id=event_axis.axis_id,
            scan_point_layout=PointLayout.rect_c((1,)),
            within_point_grouping=tuple(
                (repeat, event)
                for repeat in range(repeat_axis.size)
                for event in range(event_axis.size)
            ),
        )
        capture_plan = compile_capture_artifact_pipeline(
            TriggeredCaptureSpec(
                capture,
                pulse_port,
                FinitePulseExecutionRequest(document, pulse_artifact),
                "ch11",
                cell_plan,
            ),
            capture_repository,
        )
        images = [
            _frame(repeat, *layout.multi_index(point))
            for repeat in range(repeat_axis.size)
            for point in range(layout.storage_size)
        ]
        source_thread, source_failures = _deliver_when_armed(camera, images)
        capture_ref = runtime.controller.start(capture_plan).result(15.0)
        source_thread.join(5.0)
        assert not source_thread.is_alive()
        assert source_failures == []

        request = CalibrationAnalysisRequest(
            CalibrationCaptureLayout(AxisId("calibration-event"), (1, 2), 0),
            (2, 2),
            box_radius=1,
            box_reducer=BoxReducer.SUM,
            model_kinds=(ReadoutModelKind.BOX,),
            default_model_kind=ReadoutModelKind.BOX,
            train_fraction=0.5,
            histogram_bins=24,
            max_drop=0,
            detector_threshold_rel=0.2,
            expected_centers_xy=np.asarray(_CENTERS, dtype="<f8"),
            maximum_site_residual_px=2.0,
        )
        calibration_ref = runtime.controller.start(
            compile_calibration_artifact_plan(
                capture_ref,
                capture_repository,
                calibration_repository,
                request,
                memory_limit_bytes=512 << 20,
            )
        ).result(20.0)
        admitted = calibration_repository.admit(
            calibration_ref,
            capture_repository,
        )
        camera._wire_to(sequencer)
        yield _Rig(
            runtime,
            camera,
            sequencer,
            capture_repository,
            capture_ref,
            calibration_repository,
            calibration_ref,
            admitted,
        )
    finally:
        if source_thread is not None and source_thread.is_alive():
            camera.finish_record_capture()
            source_thread.join(2.0)
        assert runtime.shutdown(timeout=3.0)
        calibration_repository.close()
        capture_repository.close()


def _source_measurement(
    rig: _Rig,
    *,
    repeats: int = 1,
    point_shape: tuple[int, ...] = (1,),
    readout_axis: bool = False,
):
    description = rig.runtime.describe_camera("readout")
    repeat_axis = _axis("occupancy-repeat", REPEAT, repeats)
    scan_axes = tuple(
        _axis(f"occupancy-point-{position}", SCAN_POINT, size)
        for position, size in enumerate(point_shape)
    )
    if readout_axis:
        point_axes = (_axis("occupancy-readout-event", READOUT_EVENT, 1), *scan_axes)
        layout = PointLayout.rect_c((1, *point_shape))
    else:
        point_axes = scan_axes
        layout = PointLayout.rect_c(point_shape)
    cells = tuple(
        DatasetCellAddress(repeat, point)
        for repeat in range(repeats)
        for point in range(layout.storage_size)
    )
    return rig.runtime.bind_camera_measurement(
        CameraCaptureBindingRequest(
            "readout",
            repeat_axis,
            point_axes,
            layout,
            cells,
            CameraAcquisitionMode.EXTERNAL_TRIGGERED,
            len(cells),
            64 << 20,
            (description.event_setting(0),),
        )
    )


def _processor_spec(rig: _Rig) -> OccupancyStreamProcessorSpec:
    return OccupancyStreamProcessorSpec(
        calibration=rig.admitted,
        output_stream_id=StreamId("occupancy.output"),
        output_source_id="occupancy",
    )


def _pipeline_spec(
    rig: _Rig,
    measurement,
    *,
    memory_limit: int = 128 << 20,
    timeout_seconds: float = 5.0,
) -> OccupancyPipelineSpec:
    return OccupancyPipelineSpec(
        "camera to occupancy",
        measurement,
        _processor_spec(rig),
        BlockId("occupancy-counts"),
        BlockId("occupancy-occupied"),
        PipelineMemoryProfile(memory_limit),
        timeout_seconds,
    )


def _run_with_frames(
    rig: _Rig,
    spec: OccupancyPipelineSpec,
    images: list[np.ndarray],
) -> OccupancyPipelineResult:
    thread, failures = _deliver_when_armed(rig.camera, images)
    try:
        result = rig.runtime.controller.start(
            compile_occupancy_pipeline(spec)
        ).result(10.0)
    finally:
        if thread.is_alive():
            rig.camera.finish_record_capture()
        thread.join(3.0)
    assert not thread.is_alive() and failures == []
    assert isinstance(result, OccupancyPipelineResult)
    return result


def _single_trigger_document(trigger_channel: str = "ch11"):
    document = load_pulse_document(ROOT / "pulses" / "imaging_template.json")
    source_index = document.target.raw_lanes.index("ch11")
    trigger_index = document.target.raw_lanes.index(trigger_channel)
    kept = False
    periods = []
    for period in document.periods:
        states = list(period.states)
        source_high = bool(states[source_index])
        states[source_index] = 0
        states[trigger_index] = 0
        if source_high and not kept:
            states[trigger_index] = 1
            kept = True
        periods.append(replace(period, states=tuple(states)))
    assert kept
    return replace(
        document,
        name="occupancy-single-trigger",
        periods=tuple(periods),
        repeat=None,
    )


def _triggered_spec(rig: _Rig, *, document=None) -> TriggeredOccupancySpec:
    measurement = _source_measurement(rig, readout_axis=True)
    occupancy = _pipeline_spec(rig, measurement)
    pulse_port = rig.runtime.bind_sequencer_port()
    document = _single_trigger_document() if document is None else document
    artifact = compile_pulse_artifact(
        document,
        clock_hz=pulse_port.capability.clock_hz,
        execution_form=PulseExecutionForm.STATIC_ONCE,
        trigger_channels=("ch11",),
        live_target=pulse_port.capability.target,
    )
    cell_plan = compile_capture_cell_plan(
        artifact,
        "ch11",
        measurement.capture_contract.dataset_schema,
        readout_event_axis_id=AxisId("occupancy-readout-event"),
        scan_point_layout=PointLayout.rect_c((1,)),
    )
    return TriggeredOccupancySpec(
        occupancy,
        pulse_port,
        FinitePulseExecutionRequest(document, artifact),
        "ch11",
        cell_plan,
    )


def test_triggered_occupancy_rejects_current_pulse_context_before_arm_or_fire(
    rig: _Rig,
) -> None:
    document = _single_trigger_document()
    probe_period = next(
        period
        for period in document.periods
        if period.states[document.target.raw_lanes.index("ch11")]
    )
    ch00 = document.target.raw_lanes.index("ch00")
    changed_states = list(probe_period.states)
    changed_states[ch00] = 1
    changed_period = replace(probe_period, states=tuple(changed_states))
    changed_document = replace(
        document,
        periods=tuple(
            changed_period if period.period_id == probe_period.period_id else period
            for period in document.periods
        ),
    )

    rig.sequencer.history.clear()
    with pytest.raises(ValueError, match="pulse context differs from calibration"):
        _triggered_spec(rig, document=changed_document)

    assert rig.sequencer.history == []
    state = rig.camera._recent_state()
    with state["cond"]:
        assert not state["armed"] and not state["pending"]


def _metadata(index: int) -> CameraFrameMetadata:
    return CameraFrameMetadata(
        source_ordinal=index,
        produced_count=index + 1,
        frame_stamp=100 + index,
        camera_stamp=200 + index,
        timestamp_seconds=10,
        timestamp_microseconds=index,
        host_received_at_ns=10_000_000_000 + index,
        driver_buffer_index=index,
        correlation_id=f"occupancy-shot-{index}",
    )


def test_real_admitted_calibration_binds_and_preserves_invalid_sites(rig: _Rig) -> None:
    """One physical frame yields one coherent public sample."""

    admitted_again = rig.calibration_repository.admit(
        rig.calibration_ref,
        rig.capture_repository,
    )
    assert admitted_again.reference == rig.calibration_ref
    assert (
        admitted_again.artifact.source_binding.source_capture_ref
        == rig.capture_ref
    )

    measurement = _source_measurement(rig)
    contract = measurement.capture_contract
    session = measurement.capture_port.open_session(
        contract,
        TraceBinding("occupancy-bind", contract.source_id),
        measurement.capture_spec,
    )
    bound = bind_occupancy_stream_processor(
        _processor_spec(rig),
        session.processor_input_binding,
    )
    schema = session.processor_input_binding.payload_contract.value_schema
    sample = CameraSample(Value(_frame(0, 0), VALID, schema), _metadata(0))

    observed = bound.evaluate(sample)

    model = rig.admitted.artifact.select_model()
    expected_validity = model.usable_sites.mask
    assert np.any(expected_validity) and np.any(~expected_validity)
    assert isinstance(observed.counts.validity, ComponentValidity)
    assert observed.counts.validity is observed.occupied.validity
    np.testing.assert_array_equal(observed.counts.validity.mask, expected_validity)
    np.testing.assert_array_equal(
        observed.occupied.values[expected_validity],
        np.array([site % 2 == 0 for site in range(4)])[expected_validity],
    )
    invalid = ~expected_validity
    np.testing.assert_array_equal(observed.counts.values[invalid], 0.0)
    assert not np.any(np.signbit(observed.counts.values[invalid]))
    np.testing.assert_array_equal(observed.occupied.values[invalid], False)
    assert observed.metadata is sample.metadata


def test_normal_exact_pipeline_preserves_multiaxis_points_and_terminal_order(
    rig: _Rig,
) -> None:
    measurement = _source_measurement(rig, repeats=2, point_shape=(2, 3))
    spec = _pipeline_spec(rig, measurement)
    cells = measurement.capture_contract.expected_cells
    frames = [
        _frame(cell.repeat_index, 0, cell.point_storage_index)
        for cell in cells
    ]

    result = _run_with_frames(rig, spec, frames)

    counts = result.dataset.counts.block
    occupied = result.dataset.occupied.block
    site_count = rig.admitted.artifact.site_map.site_axis.size
    assert counts.schema.point_layout.logical_shape == (2, 3)
    assert counts.values.shape == occupied.values.shape == (2, 6, site_count)
    assert counts.validity is occupied.validity
    assert isinstance(counts.validity, ComponentValidity)
    assert result.dataset.cell_schedule == cells
    assert tuple(
        metadata.source_ordinal for _cell, metadata in result.dataset.events
    ) == tuple(range(len(cells)))

    for cell in cells:
        location = (cell.repeat_index, cell.point_storage_index)
        valid = counts.validity.mask[location]
        expected = np.array(
            [
                (cell.repeat_index + cell.point_storage_index + site) % 2 == 0
                for site in range(site_count)
            ]
        )
        np.testing.assert_array_equal(occupied.values[location][valid], expected[valid])
        np.testing.assert_array_equal(counts.values[location][~valid], 0.0)
        np.testing.assert_array_equal(occupied.values[location][~valid], False)

    terminal = result.pipeline.capture_terminal
    span = result.pipeline.source_event_span
    assert terminal.produced_count == terminal.drained_count == len(cells)
    assert terminal.source_stopped and terminal.no_more_frames and terminal.joined
    assert span.count == span.end_sequence - span.start_sequence == len(cells)
    assert result.pipeline.source_cell_schedule == cells
    assert result.pipeline.aggregate_peak_bytes <= spec.memory.memory_limit_bytes
    assert result.calibration_reference == rig.calibration_ref


def test_missing_frame_is_gap_fatal_and_cleanup_allows_the_next_run(
    rig: _Rig,
) -> None:
    measurement = _source_measurement(rig, point_shape=(2,))
    spec = _pipeline_spec(rig, measurement, timeout_seconds=3.0)
    thread, source_failures = _deliver_when_armed(
        rig.camera,
        [_frame(0, 0, 0)],
    )
    handle = rig.runtime.controller.start(compile_occupancy_pipeline(spec))
    with pytest.raises(RunFailed) as failure:
        handle.result(8.0)
    thread.join(3.0)

    assert not thread.is_alive() and source_failures == []
    assert failure.value.primary is not None
    assert failure.value.primary.original_type == "RuntimeError"
    assert "short read for an exact capture" in str(failure.value.primary)

    recovery_measurement = _source_measurement(rig)
    recovery = _run_with_frames(
        rig,
        _pipeline_spec(rig, recovery_measurement),
        [_frame(0, 0, 0)],
    )
    assert recovery.pipeline.capture_terminal.produced_count == 1
    assert len(recovery.dataset.events) == 1


def test_memory_rejection_happens_before_camera_prepare(
    rig: _Rig,
    monkeypatch,
) -> None:
    arm_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    original_arm = rig.camera.arm

    def observed_arm(*args, **kwargs):
        arm_calls.append((args, kwargs))
        return original_arm(*args, **kwargs)

    monkeypatch.setattr(rig.camera, "arm", observed_arm)
    measurement = _source_measurement(rig)
    handle = rig.runtime.controller.start(
        compile_occupancy_pipeline(
            _pipeline_spec(rig, measurement, memory_limit=1)
        )
    )

    with pytest.raises(RunFailed) as failure:
        handle.result(5.0)

    assert failure.value.primary is not None
    assert failure.value.primary.original_type == "MemoryError"
    assert arm_calls == []


def test_triggered_pipeline_readies_full_chain_before_its_single_fire(
    rig: _Rig,
    monkeypatch,
) -> None:
    import zlc_neutral_atom.timing.occupancy as timing_occupancy

    spec = _triggered_spec(rig)
    software_ready = threading.Event()
    camera_armed = threading.Event()
    readiness_at_fire: list[tuple[bool, bool]] = []
    original_open = timing_occupancy.open_exact_occupancy
    original_arm = rig.camera.arm

    def observed_open(occupancy_spec, context):
        transaction = original_open(occupancy_spec, context)
        assert transaction.worker is not None
        transaction.worker.exact_readiness()
        software_ready.set()
        return transaction

    def observed_arm(*args, **kwargs):
        result = original_arm(*args, **kwargs)
        camera_armed.set()
        return result

    def observe_fire(_playback):
        readiness_at_fire.append((software_ready.is_set(), camera_armed.is_set()))

    monkeypatch.setattr(timing_occupancy, "open_exact_occupancy", observed_open)
    monkeypatch.setattr(rig.camera, "arm", observed_arm)
    rig.sequencer.history.clear()
    rig.sequencer.add_fire_listener(observe_fire)
    try:
        result = rig.runtime.controller.start(
            compile_triggered_occupancy_pipeline(spec)
        ).result(10.0)
    finally:
        rig.sequencer.remove_fire_listener(observe_fire)

    assert isinstance(result, TriggeredOccupancyPipelineResult)
    assert readiness_at_fire == [(True, True)]
    actions = [item["action"] for item in rig.sequencer.history]
    assert actions.count("fire") == 1
    assert actions == ["prepare", "fire", "wait_done", "safe"]

    terminal = result.occupancy.pipeline.capture_terminal
    trigger_counts = dict(
        result.lineage.terminal.expected_trigger_counts_from_completed_schedule
    )
    assert trigger_counts["ch11"] == 1
    assert terminal.produced_count == terminal.drained_count == 1
    assert (
        result.lineage.cell_plan.total_events
        == len(result.occupancy.dataset.events)
        == 1
    )
    assert (
        result.occupancy.dataset.cell_schedule
        == result.lineage.cell_plan.expected_cells
    )
    assert (
        result.occupancy.pipeline.source_cell_schedule
        == result.lineage.cell_plan.expected_cells
    )
    assert len(result.occupancy.pipeline.processor_stages) == 1
    assert result.lineage.compiled_artifact is spec.pulse_request.artifact
    assert result.occupancy.calibration_reference == rig.calibration_ref


def test_triggered_occupancy_rejects_an_executed_value_from_another_run(
    rig: _Rig,
) -> None:
    plan = compile_triggered_occupancy_pipeline(_triggered_spec(rig))
    executed_values = []

    def recording_execute(context, prepared):
        executed = plan.execute(context, prepared)
        executed_values.append(executed)
        return executed

    rig.runtime.controller.run(replace(plan, execute=recording_execute))
    assert len(executed_values) == 1

    def stale_execute(context, prepared):
        current = plan.execute(context, prepared)
        assert current is not executed_values[0]
        return executed_values[0]

    with pytest.raises(RunFailed) as failure:
        rig.runtime.controller.run(replace(plan, execute=stale_execute))

    assert failure.value.primary is not None
    assert failure.value.primary.original_type == "ValueError"
    assert "another Run" in str(failure.value.primary)


def test_triggered_occupancy_finalize_rejects_a_rebound_terminal(
    rig: _Rig,
) -> None:
    plan = compile_triggered_occupancy_pipeline(_triggered_spec(rig))
    first = rig.runtime.controller.run(plan)

    def rebind_terminal(context, prepared):
        executed = plan.execute(context, prepared)
        executed.pulse_terminal = first.lineage.terminal
        return executed

    with pytest.raises(RunFailed) as failure:
        rig.runtime.controller.run(replace(plan, execute=rebind_terminal))

    assert failure.value.primary is not None
    assert failure.value.primary.original_type == "PermissionError"
    assert "terminal changed" in str(failure.value.primary)


def test_triggered_occupancy_cancel_after_fire_cleans_pulse_before_camera(
    rig: _Rig,
    monkeypatch,
) -> None:
    import zlc_neutral_atom.timing.occupancy as timing_occupancy

    entered_capture = threading.Event()
    cleanup_order = []
    original_pulse_cleanup = timing_occupancy.PulseSession.cleanup
    original_occupancy_cleanup = timing_occupancy.ExactOccupancyTransaction.cleanup

    def blocked_capture_all(self, context):
        entered_capture.set()
        while True:
            context.checkpoint()
            time.sleep(0.005)

    def observed_pulse_cleanup(self, context):
        cleanup_order.append("pulse")
        return original_pulse_cleanup(self, context)

    def observed_occupancy_cleanup(self, context):
        cleanup_order.append("camera")
        return original_occupancy_cleanup(self, context)

    monkeypatch.setattr(
        timing_occupancy.ExactOccupancyTransaction,
        "capture_all",
        blocked_capture_all,
    )
    monkeypatch.setattr(
        timing_occupancy.PulseSession,
        "cleanup",
        observed_pulse_cleanup,
    )
    monkeypatch.setattr(
        timing_occupancy.ExactOccupancyTransaction,
        "cleanup",
        observed_occupancy_cleanup,
    )
    rig.sequencer.history.clear()

    handle = rig.runtime.controller.start(
        compile_triggered_occupancy_pipeline(_triggered_spec(rig))
    )
    assert entered_capture.wait(3.0)
    handle.cancel("cancel triggered occupancy after FIRE")
    with pytest.raises(RunCancelled):
        handle.result(10.0)

    assert cleanup_order == ["pulse", "camera"]
    assert "fire" in {item["action"] for item in rig.sequencer.history}
    assert dict(rig.sequencer.snapshot())["state"] == "safe"
    state = rig.camera._recent_state()
    with state["cond"]:
        assert not state["armed"] and not state["pending"]
