"""One-stage admitted-calibration camera -> occupancy contracts."""

from __future__ import annotations

import gc
from dataclasses import dataclass, replace
from pathlib import Path
import pickle
import threading
import time
import weakref

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
    StreamGenerationId,
    INVALID,
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
from zlc_neutral_atom.processing.stream import (
    ExactStreamProcessorWorker,
    StreamProcessorError,
)
from zlc_neutral_atom.readout.analysis import (
    BoxAnalysisConfig,
    CalibrationAnalysisPlanningAssumption,
    CalibrationAnalysisRequest,
    CalibrationBracketSamplingAssumption,
    PsfAnalysisConfig,
    ReferenceClassOrientation,
    ReferenceLabelSource,
)
from zlc_neutral_atom.readout.calibration import (
    BackgroundMode,
    BoxReadoutModel,
    BoxReducer,
    CalibrationResourceExceeded,
    ReadoutModelKind,
    apply_readout_model,
    bind_readout_feature_spec,
    calibration_retained_array_nbytes,
    readout_application_scratch_nbytes,
)
from zlc_neutral_atom.readout.calibration_repository import (
    AdmittedCalibration,
    CalibrationRepository,
    compile_calibration_artifact_plan,
)
from zlc_neutral_atom.readout.calibration_reference import (
    CalibrationArtifactRef,
    calibration_artifact_input_ref,
)
from zlc_neutral_atom.readout.contracts import (
    CalibrationCaptureLayout,
    ReadoutBindingKey,
)
from zlc_neutral_atom.readout.occupancy import (
    BoundOccupancyStreamProcessor,
    OccupancyDatasetEventAdapter,
    OccupancyDatasetField,
    OccupancyDatasetMetadata,
    OccupancySample,
    OccupancySampleContract,
    OccupancyStreamProcessorSpec,
    bind_occupancy_stream_processor,
)
from zlc_neutral_atom.readout.occupancy_pipeline import (
    FrozenOccupancyDataset,
    OccupancyDatasetMaterializerSpec,
    OccupancyPipelineResult,
    OccupancyPipelineSpec,
    compile_occupancy_pipeline,
)
from zlc_neutral_atom.runtime import (
    DatasetBuilder,
    DatasetCellAddress,
    DatasetMaterializerSpec,
    DatasetMode,
    FrozenDatasetEdge,
    CleanupReport,
    MinimalPipelineSpec,
    PipelineMemoryProfile,
    PostSafetyContext,
    RunContext,
    RunCancelled,
    RunFailed,
    RunMode,
    RunPlan,
)
from zlc_neutral_atom.runtime.streams import (
    AcquisitionStream,
    ProducerFlowControl,
    ReservationState,
    StreamId,
    TraceBinding,
    TraceContext,
)
from zlc_neutral_atom.timing import (
    FinitePulseExecutionRequest,
    TriggeredOccupancyPipelineResult,
    TriggeredOccupancySpec,
    compile_capture_cell_plan,
    compile_triggered_occupancy_pipeline,
)
from zlc_pulse import (
    PulseExecutionForm,
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


def _analysis_request() -> CalibrationAnalysisRequest:
    return CalibrationAnalysisRequest(
        CalibrationCaptureLayout(AxisId("readout-event"), (0, 2), 1),
        (2, 2),
        ReferenceLabelSource.UNSUPERVISED_REFERENCE_VALLEY,
        ReferenceClassOrientation.ABOVE_IS_OCCUPIED,
        CalibrationBracketSamplingAssumption.INDEPENDENT_STATIONARY_BRACKETS,
        CalibrationAnalysisPlanningAssumption.PRECOMMITTED_BEFORE_SOURCE_INSPECTION,
        box=BoxAnalysisConfig(1, BoxReducer.SUM),
        model_kinds=(
            ReadoutModelKind.UNIFORM_PSF,
            ReadoutModelKind.BOX,
            ReadoutModelKind.PER_SITE_PSF,
        ),
        default_model_kind=ReadoutModelKind.BOX,
        psf=PsfAnalysisConfig(1, BackgroundMode.NONE, 0),
        train_fraction=0.35,
        minimum_train_samples_per_class=1,
        minimum_test_samples_per_class=1,
        minimum_held_out_class_accuracy_lower_bound=0.0,
    )


def _frame(repeat: int, event: int, context: int) -> np.ndarray:
    image = np.zeros((32, 32), dtype=np.uint16)
    for site, (x, y) in enumerate(_CENTERS):
        occupied = (repeat + context + site) % 2 == 0
        if event in (0, 2):
            level = 2000.0 if occupied else 200.0
        else:
            level = 1000.0 if occupied else 100.0
        image[y - 1 : y + 2, x - 1 : x + 2] = np.rint(level * _SPOT).astype(
            np.uint16
        )
    return image


def _deliver_when_armed(camera: VirtualCamera, images: list[np.ndarray]):
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


@dataclass
class _TrustedCalibration:
    runtime: LegacyNeutralAtomRuntime
    camera: VirtualCamera
    sequencer: VirtualSequencer
    capture_repository: CaptureRepository
    capture_ref: CaptureArtifactRef
    calibration_repository: CalibrationRepository
    calibration_ref: CalibrationArtifactRef
    admitted: AdmittedCalibration


@pytest.fixture(scope="module")
def trusted_calibration(tmp_path_factory):
    root = tmp_path_factory.mktemp("occupancy-trusted-calibration")
    pulse_document = load_pulse_document(ROOT / "pulses" / "imaging_template.json")
    pulse_catalog = PortCatalog(
        pulse_document.target.raw_lanes,
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
            for port in pulse_document.target.ports
        ),
    )
    sequencer = VirtualSequencer(sleep_scale=0, port_catalog=pulse_catalog)
    trap = VirtualTrapArray(grid_shape=(2, 2), image_shape=(32, 32), seed=11)
    camera = VirtualCamera(
        trap,
        exposure=1e-3,
        capture_trigger_channels=("ch11",),
        sequencer=sequencer,
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
    description = runtime.describe_camera("readout")
    repeat_axis = _axis("repeat", REPEAT, 32)
    event_axis = _axis("readout-event", READOUT_EVENT, 3)
    context_axis = _axis("context", SCAN_POINT, 2)
    layout = PointLayout.rect_c((3, 2))
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
    capture_spec = MinimalPipelineSpec(
        "occupancy trusted calibration raw capture",
        measurement,
        DatasetMaterializerSpec(
            BlockId("occupancy-calibration-source"),
            PipelineMemoryProfile(128 << 20),
        ),
    )
    images = [
        _frame(repeat, *layout.multi_index(point))
        for repeat in range(repeat_axis.size)
        for point in range(layout.storage_size)
    ]
    capture_repository = CaptureRepository(root / "captures", repository_id="captures")
    source_thread, source_failures = _deliver_when_armed(camera, images)
    calibration_repository = None
    try:
        capture_ref = runtime.controller.start(
            compile_capture_artifact_pipeline(capture_spec, capture_repository)
        ).result(15.0)
        source_thread.join(5.0)
        assert not source_thread.is_alive()
        assert source_failures == []
        calibration_repository = CalibrationRepository(
            root / "calibrations",
            repository_id="calibrations",
        )
        calibration_ref = runtime.controller.start(
            compile_calibration_artifact_plan(
                capture_ref,
                capture_repository,
                calibration_repository,
                _analysis_request(),
            )
        ).result(20.0)
        admitted = calibration_repository.admit(
            calibration_ref,
            capture_repository,
        )
        yield _TrustedCalibration(
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
        if source_thread.is_alive():
            camera.finish_record_capture()
            source_thread.join(2.0)
        assert runtime.shutdown(timeout=3.0)
        if calibration_repository is not None:
            calibration_repository.close()
        capture_repository.close()


def _source_session(
    trusted: _TrustedCalibration,
    *,
    event_settings=None,
    readout_events: int | None = None,
    points: int = 2,
    repeat_count: int = 1,
    point_layout: PointLayout | None = None,
    repeat_axis_id: str = "occupancy-repeat",
):
    measurement = _source_measurement(
        trusted,
        event_settings=event_settings,
        readout_events=readout_events,
        points=points,
        repeat_count=repeat_count,
        point_layout=point_layout,
        repeat_axis_id=repeat_axis_id,
    )
    contract = measurement.capture_contract
    return measurement.capture_port.open_session(
        contract,
        TraceBinding("occupancy-bind-run", contract.source_id),
        measurement.capture_spec,
    )


def _source_measurement(
    trusted: _TrustedCalibration,
    *,
    event_settings=None,
    readout_events: int | None = None,
    points: int = 2,
    repeat_count: int = 1,
    point_layout: PointLayout | None = None,
    repeat_axis_id: str = "occupancy-repeat",
    cell_schedule: tuple[DatasetCellAddress, ...] | None = None,
):
    description = trusted.runtime.describe_camera("readout")
    repeat_axis = _axis(repeat_axis_id, REPEAT, repeat_count)
    if readout_events is None:
        point_axes = (_axis("occupancy-point", SCAN_POINT, points),)
        layout = point_layout or PointLayout.rect_c((points,))
        settings = (
            (description.event_setting(0),)
            if event_settings is None
            else tuple(event_settings)
        )
    else:
        point_axes = (
            _axis("occupancy-readout-event", READOUT_EVENT, readout_events),
            _axis("occupancy-context", SCAN_POINT, points),
        )
        layout = point_layout or PointLayout.rect_c((readout_events, points))
        settings = (
            tuple(description.event_setting(index) for index in range(readout_events))
            if event_settings is None
            else tuple(event_settings)
        )
    cells = (
        tuple(
            DatasetCellAddress(repeat, point)
            for repeat in range(repeat_axis.size)
            for point in range(layout.storage_size)
        )
        if cell_schedule is None
        else tuple(cell_schedule)
    )
    measurement = trusted.runtime.bind_camera_measurement(
        CameraCaptureBindingRequest(
            "readout",
            repeat_axis,
            point_axes,
            layout,
            cells,
            CameraAcquisitionMode.EXTERNAL_TRIGGERED,
            len(cells),
            64 << 20,
            settings,
        )
    )
    return measurement


def _bind(
    trusted: _TrustedCalibration,
    session,
    *,
    model_kind: ReadoutModelKind | None = None,
):
    return bind_occupancy_stream_processor(
        _processor_spec(trusted, model_kind=model_kind),
        session.processor_input_binding,
    )


def _processor_spec(
    trusted: _TrustedCalibration,
    *,
    model_kind: ReadoutModelKind | None = None,
    output_stream_id: str = "occupancy.output",
    output_source_id: str = "occupancy",
) -> OccupancyStreamProcessorSpec:
    return OccupancyStreamProcessorSpec(
        calibration=trusted.admitted,
        readout_binding=ReadoutBindingKey("readout"),
        model_kind=model_kind,
        output_stream_id=StreamId(output_stream_id),
        output_source_id=output_source_id,
    )


def _occupancy_pipeline_spec(
    trusted: _TrustedCalibration,
    measurement,
    *,
    memory_limit: int = 256 << 20,
) -> OccupancyPipelineSpec:
    return OccupancyPipelineSpec(
        name="typed camera to occupancy",
        measurement=measurement,
        processor=_processor_spec(trusted),
        materializer=OccupancyDatasetMaterializerSpec(
            BlockId("occupancy-counts"),
            BlockId("occupancy-occupied"),
            PipelineMemoryProfile(memory_limit),
        ),
        timeout_seconds=12.0,
    )


def _single_trigger_document(
    trusted: _TrustedCalibration,
    trigger_channel: str = "ch11",
):
    document = load_pulse_document(ROOT / "pulses" / "imaging_template.json")
    source_index = document.target.raw_lanes.index("ch11")
    trigger_index = document.target.raw_lanes.index(trigger_channel)
    kept_trigger = False
    periods = []
    for period in document.periods:
        states = list(period.states)
        source_high = bool(states[source_index])
        states[source_index] = 0
        states[trigger_index] = 0
        if source_high and not kept_trigger:
            states[trigger_index] = 1
            kept_trigger = True
        periods.append(replace(period, states=tuple(states)))
    assert kept_trigger
    return replace(
        document,
        name="occupancy-single-trigger",
        periods=tuple(periods),
        repeat=None,
    )


def _triggered_occupancy_spec(
    trusted: _TrustedCalibration,
    *,
    trigger_channel: str = "ch11",
) -> TriggeredOccupancySpec:
    measurement = _source_measurement(
        trusted,
        readout_events=1,
        points=1,
    )
    occupancy = _occupancy_pipeline_spec(trusted, measurement)
    pulse_port = trusted.runtime.bind_sequencer_port()
    document = _single_trigger_document(trusted, trigger_channel)
    artifact = compile_pulse_artifact(
        document,
        clock_hz=pulse_port.capability.clock_hz,
        execution_form=PulseExecutionForm.STATIC_ONCE,
        trigger_channels=(trigger_channel,),
        live_target=pulse_port.capability.target,
    )
    cell_plan = compile_capture_cell_plan(
        artifact,
        trigger_channel,
        measurement.capture_contract.dataset_schema,
        readout_event_axis_id=AxisId("occupancy-readout-event"),
        scan_point_layout=PointLayout.rect_c((1,)),
    )
    return TriggeredOccupancySpec(
        occupancy,
        pulse_port,
        FinitePulseExecutionRequest(document, artifact),
        trigger_channel,
        cell_plan,
    )


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


def _terminal_output(
    bundle: BoundOccupancyStreamProcessor,
    capture_input,
    *,
    run_id: str,
    edge: FrozenDatasetEdge | None = None,
):
    output_contract = bundle.output_payload_contract
    events = len(capture_input.capture_contract.expected_cells)
    output_budget = events * output_contract.max_retained_nbytes
    output, output_producer = AcquisitionStream.create(
        bundle.output_stream_id,
        output_contract,
        flow_control=ProducerFlowControl.BACKPRESSURE_CAPABLE,
        retention_events=events,
        retention_bytes=output_budget,
        join_key_contract=capture_input.join_key_contract,
    )
    output_reservation = output.reserve(
        total_events=events,
        max_inflight_events=events,
        max_inflight_bytes=output_budget,
        trace_binding=TraceBinding(run_id, bundle.output_source_id),
    )
    output_cursor = output_reservation.activate()
    builder = DatasetBuilder(
        BlockId(f"occupancy-authority-{run_id}"),
        output_reservation,
        bundle.output_edge if edge is None else edge,
        DatasetMode.FINITE_EXACT,
    )
    return output_producer, output_cursor, builder


def _sample(session, image: np.ndarray, index: int) -> CameraSample:
    schema = session.processor_input_binding.payload_contract.value_schema
    return CameraSample(Value(image, VALID, schema), _metadata(index))


def test_bind_requires_admitted_calibration_and_freezes_default_model(trusted_calibration):
    session = _source_session(trusted_calibration, points=1)
    spec = _processor_spec(trusted_calibration)
    assert spec.model_kind is ReadoutModelKind.BOX
    bound = bind_occupancy_stream_processor(
        spec,
        session.processor_input_binding,
    )
    assert isinstance(bound, BoundOccupancyStreamProcessor)
    assert bound.model_kind is ReadoutModelKind.BOX
    assert bound.calibration_reference == trusted_calibration.calibration_ref
    assert bound.calibration_artifact_fingerprint == (
        trusted_calibration.admitted.artifact_fingerprint
    )
    assert bound.calibration_admission_evidence_digest == (
        trusted_calibration.admitted.evidence_digest
    )
    assert bound.artifact_inputs == (
        calibration_artifact_input_ref(trusted_calibration.calibration_ref),
    )
    assert not hasattr(bound, "processor")
    assert not hasattr(bound, "config")

    artifact = trusted_calibration.admitted.artifact
    selected_model = artifact.select_model(kind=bound.model_kind)
    assert bound.operator_scratch_nbytes == readout_application_scratch_nbytes(
        bind_readout_feature_spec(selected_model, artifact.site_map),
        artifact.frame_contract.frame_schema,
    )
    with pytest.raises(TypeError, match="AdmittedCalibration"):
        OccupancyStreamProcessorSpec(
            calibration=artifact,
            readout_binding=ReadoutBindingKey("readout"),
            model_kind=ReadoutModelKind.BOX,
            output_stream_id=StreamId("bad.raw-artifact"),
            output_source_id="bad-raw-artifact",
        )

    forged = object.__new__(AdmittedCalibration)
    for slot in AdmittedCalibration.__slots__:
        if slot == "__weakref__":
            continue
        object.__setattr__(
            forged,
            slot,
            object.__getattribute__(trusted_calibration.admitted, slot),
        )
    object.__setattr__(
        forged,
        "_reference",
        CalibrationArtifactRef(
            trusted_calibration.calibration_ref.repository_id,
            "0" * 64,
        ),
    )
    with pytest.raises(PermissionError, match="authority is invalid"):
        bind_occupancy_stream_processor(
            OccupancyStreamProcessorSpec(
                calibration=forged,
                readout_binding=ReadoutBindingKey("readout"),
                model_kind=ReadoutModelKind.BOX,
                output_stream_id=StreamId("bad-forged-admission"),
                output_source_id="bad-forged-admission",
            ),
            session.processor_input_binding,
        )


def test_admitted_calibration_model_arrays_are_intrinsically_immutable(
    trusted_calibration,
):
    admission = trusted_calibration.calibration_repository.admit(
        trusted_calibration.calibration_ref,
        trusted_calibration.capture_repository,
    )
    artifact = admission.artifact
    model = artifact.select_model(kind=ReadoutModelKind.BOX)
    thresholds = model.header.thresholds
    with pytest.raises(ValueError):
        thresholds.setflags(write=True)
    with pytest.raises(ValueError):
        thresholds[0] += 1.0
    assert admission.artifact_fingerprint == artifact.fingerprint


def test_calibration_memory_owner_counts_every_unique_array(trusted_calibration):
    artifact = trusted_calibration.admitted.artifact
    arrays = [artifact.site_map.coordinates_xy, artifact.site_map.validity.mask]
    quality_fields = (
        "dark_training_sample_counts",
        "bright_training_sample_counts",
        "held_out_dark_success_counts",
        "held_out_dark_total_counts",
        "held_out_dark_labeled_counts",
        "held_out_bright_success_counts",
        "held_out_bright_total_counts",
        "held_out_bright_labeled_counts",
        "held_out_dark_accuracy_lower_bounds",
        "held_out_bright_accuracy_lower_bounds",
        "held_out_fidelity",
    )
    for model in artifact.models:
        header = model.header
        arrays.extend(
            (
                header.thresholds,
                header.occupied_above_thresholds,
                header.quality.usable_sites.mask,
                header.quality.held_out_validity.mask,
                model.boxes_xywh,
            )
        )
        arrays.extend(getattr(header.quality, name) for name in quality_fields)
        if hasattr(model, "kernels"):
            arrays.append(model.kernels)
        if hasattr(model, "kernel"):
            arrays.append(model.kernel)

    allocations = {}
    for array in arrays:
        owner = array
        while isinstance(owner, np.ndarray) and owner.base is not None:
            owner = owner.base
        while isinstance(owner, memoryview):
            owner = owner.obj
        allocations.setdefault(
            id(owner),
            len(owner) if isinstance(owner, bytes) else int(array.nbytes),
        )

    assert calibration_retained_array_nbytes(artifact) == sum(allocations.values())


def test_reusable_processor_spec_binds_each_capture_session_generation(
    trusted_calibration,
):
    spec = OccupancyStreamProcessorSpec(
        calibration=trusted_calibration.admitted,
        readout_binding=ReadoutBindingKey("readout"),
        model_kind=None,
        output_stream_id=StreamId("occupancy.reusable-output"),
        output_source_id="occupancy-reusable",
    )
    first_session = _source_session(trusted_calibration, points=1)
    second_session = _source_session(trusted_calibration, points=1)

    first = bind_occupancy_stream_processor(
        spec,
        first_session.processor_input_binding,
    )
    second = bind_occupancy_stream_processor(
        spec,
        second_session.processor_input_binding,
    )

    assert first.model_kind is second.model_kind is spec.model_kind
    assert first.output_payload_contract.fingerprint == (
        second.output_payload_contract.fingerprint
    )
    assert first.fingerprint != second.fingerprint


def test_bind_rejects_mixed_physical_readout_events(trusted_calibration):
    description = trusted_calibration.runtime.describe_camera("readout")
    base = description.event_setting(0)
    uniform = _source_session(
        trusted_calibration,
        readout_events=2,
        points=1,
        event_settings=(base, replace(base, event_index=1)),
    )
    with pytest.raises(ValueError, match="exactly one physical READOUT_EVENT"):
        _bind(trusted_calibration, uniform)


def test_bind_rejects_binding_and_resource_drift(trusted_calibration):
    session = _source_session(trusted_calibration, points=1)
    with pytest.raises(ValueError, match="another readout binding"):
        bind_occupancy_stream_processor(
            OccupancyStreamProcessorSpec(
                calibration=trusted_calibration.admitted,
                readout_binding=ReadoutBindingKey("another-readout"),
                model_kind=ReadoutModelKind.BOX,
                output_stream_id=StreamId("occupancy.wrong-binding"),
                output_source_id="occupancy-wrong-binding",
            ),
            session.processor_input_binding,
        )
    restrictive = replace(
        trusted_calibration.calibration_repository.resource_policy,
        max_sites=1,
    )
    with pytest.raises(CalibrationResourceExceeded, match="site count"):
        bind_occupancy_stream_processor(
            OccupancyStreamProcessorSpec(
                calibration=trusted_calibration.admitted,
                readout_binding=ReadoutBindingKey("readout"),
                model_kind=ReadoutModelKind.BOX,
                output_stream_id=StreamId("occupancy.resource-reject"),
                output_source_id="occupancy-resource-reject",
                resource_policy=restrictive,
            ),
            session.processor_input_binding,
        )


def test_one_operator_call_preserves_metadata_counts_occupancy_and_invalid_sites(
    trusted_calibration,
    monkeypatch,
):
    session = _source_session(trusted_calibration, points=1)
    bound = _bind(trusted_calibration, session)
    artifact = trusted_calibration.admitted.artifact
    model = artifact.select_model(kind=bound.model_kind)
    assert isinstance(model, BoxReadoutModel)
    image = _frame(0, 1, 0)
    schema = session.processor_input_binding.payload_contract.value_schema
    sample = CameraSample(
        Value(image, INVALID, schema),
        _metadata(0),
    )

    def forbidden_runtime_lookup(*_args, **_kwargs):
        raise AssertionError("bound operator attempted a repository lookup")

    monkeypatch.setattr(CalibrationRepository, "admit", forbidden_runtime_lookup)
    output = bound.evaluate(sample)
    assert isinstance(output, OccupancySample)
    assert output.metadata is sample.metadata
    assert output.counts.schema.data_axes == (artifact.site_map.site_axis,)
    assert output.occupied.schema.data_axes == (artifact.site_map.site_axis,)
    assert output.counts.values.shape == output.occupied.values.shape == (4,)
    assert not np.any(output.counts.validity.mask)
    assert not np.any(output.occupied.validity.mask)
    assert not np.any(output.counts.values)
    assert not np.any(output.occupied.values)
    bound.output_payload_contract.validate(output)


def test_occupancy_payload_digest_binds_both_fields_and_physical_metadata(
    trusted_calibration,
):
    session = _source_session(trusted_calibration, points=1)
    bound = _bind(trusted_calibration, session)
    output = bound.evaluate(_sample(session, _frame(0, 1, 0), 0))
    contract = bound.output_payload_contract
    baseline = contract.digest(output)
    assert contract.digest_components(
        output.occupied.values,
        output.occupied.validity,
        output.counts.values,
        output.counts.validity,
        output.metadata,
    ) == baseline
    valid_indices = np.flatnonzero(output.occupied.validity.mask)
    assert valid_indices.size
    site = int(valid_indices[0])

    changed_occupied_values = np.array(output.occupied.values, copy=True)
    changed_occupied_values[site] = ~changed_occupied_values[site]
    changed_counts_values = np.array(output.counts.values, copy=True)
    changed_counts_values[site] += 1.0
    variants = (
        OccupancySample(
            Value(
                changed_occupied_values,
                output.occupied.validity,
                output.occupied.schema,
            ),
            output.counts,
            output.metadata,
        ),
        OccupancySample(
            output.occupied,
            Value(
                changed_counts_values,
                output.counts.validity,
                output.counts.schema,
            ),
            output.metadata,
        ),
        OccupancySample(
            output.occupied,
            output.counts,
            replace(output.metadata, frame_stamp=output.metadata.frame_stamp + 1),
        ),
    )

    assert all(contract.digest(variant) != baseline for variant in variants)


def test_dataset_adapter_has_one_canonical_counts_projection_and_occupied_metadata(
    trusted_calibration,
):
    session = _source_session(trusted_calibration, points=1)
    bound = _bind(trusted_calibration, session)
    assert bound.output_schema.cell_schema.dtype == np.dtype("<f8")
    assert bound.output_payload_contract is bound.output_edge.payload_contract

    sample = _sample(session, _frame(0, 1, 0), 0)
    output = bound.evaluate(sample)
    assert bound.output_adapter.value(output) is output.counts
    metadata = bound.output_edge.metadata_contract.snapshot(output)
    assert isinstance(metadata, OccupancyDatasetMetadata)
    assert metadata.occupied is output.occupied
    assert metadata.source_metadata is output.metadata
    digest = bound.output_edge.metadata_contract.digest(metadata)
    assert bound.output_edge.metadata_contract.digest_components(
        output.occupied.values,
        output.occupied.validity,
        output.metadata,
    ) == digest
    changed_occupied = Value(
        ~output.occupied.values,
        output.occupied.validity,
        output.occupied.schema,
    )
    changed = OccupancyDatasetMetadata(
        changed_occupied,
        output.metadata,
    )
    assert bound.output_edge.metadata_contract.digest(changed) != digest
    scalar_validity_metadata = OccupancyDatasetMetadata(
        Value(output.occupied.values, VALID, output.occupied.schema),
        output.metadata,
    )
    with pytest.raises(TypeError, match="ComponentValidity"):
        bound.output_edge.metadata_contract.validate(scalar_validity_metadata)
    with pytest.raises(TypeError, match="ComponentValidity"):
        bound.output_edge.metadata_contract.digest(scalar_validity_metadata)


def test_bound_authority_hides_generic_processor_and_validates_replacements(
    trusted_calibration,
):
    session = _source_session(trusted_calibration, points=1)
    bundle = _bind(trusted_calibration, session)
    processor = object.__getattribute__(bundle, "_processor")
    config = processor.config
    artifact = trusted_calibration.admitted.artifact

    with pytest.raises(PermissionError, match="only be minted by its binder"):
        BoundOccupancyStreamProcessor(
            object(),
            processor=processor,
            capture_input=session.processor_input_binding,
            output_edge=bundle.output_edge,
        )
    with pytest.raises(TypeError, match="final"):
        type("ForgedOccupancyBinding", (BoundOccupancyStreamProcessor,), {})
    with pytest.raises(AttributeError, match="immutable"):
        bundle.output_source_id = "forged"
    with pytest.raises(TypeError, match="process-local"):
        pickle.dumps(bundle)

    other_model = artifact.select_model(kind=ReadoutModelKind.UNIFORM_PSF)
    other_feature_spec = bind_readout_feature_spec(other_model, artifact.site_map)
    with pytest.raises(ValueError, match="selected kind"):
        replace(config, model_kind=ReadoutModelKind.UNIFORM_PSF)
    with pytest.raises(ValueError, match="feature spec does not match"):
        replace(config, feature_spec=other_feature_spec)
    with pytest.raises(ValueError, match="another FrameContract"):
        replace(
            config,
            frame_contract=replace(
                config.frame_contract,
                gain=config.frame_contract.gain + 1.0,
            ),
        )
    with pytest.raises(ValueError, match="another SiteMap"):
        replace(
            config,
            site_map=replace(
                config.site_map,
                detection_lineage_digest="0" * 64,
            ),
        )
    with pytest.raises(ValueError, match="counts schema unit"):
        replace(
            config,
            counts_schema=replace(config.counts_schema, value_unit="electron"),
        )

    assert len(bundle.fingerprint) == 64


def test_domain_worker_factory_rejects_equivalent_cloned_capture_stream(
    trusted_calibration,
):
    session = _source_session(trusted_calibration, points=1)
    capture_input = session.processor_input_binding
    bundle = _bind(trusted_calibration, session)
    source = capture_input.stream
    clone, _clone_producer = AcquisitionStream.create(
        source.stream_id,
        capture_input.payload_contract,
        flow_control=source.flow_control,
        retention_events=source.retention_events,
        retention_bytes=source.retention_bytes,
        join_key_contract=capture_input.join_key_contract,
    )
    clone_reservation = clone.reserve(
        total_events=1,
        max_inflight_events=1,
        max_inflight_bytes=capture_input.payload_contract.max_retained_nbytes,
        trace_binding=TraceBinding("occupancy-clone", "readout"),
    )
    clone_cursor = clone_reservation.activate()
    output_contract = bundle.output_payload_contract
    _output, output_producer = AcquisitionStream.create(
        bundle.output_stream_id,
        output_contract,
        flow_control=ProducerFlowControl.BACKPRESSURE_CAPABLE,
        retention_events=1,
        retention_bytes=output_contract.max_retained_nbytes,
        join_key_contract=capture_input.join_key_contract,
    )

    assert capture_input.input_edge.validate_stream(clone) is None
    with pytest.raises(PermissionError, match="not minted by this CaptureSession"):
        bundle.create_exact_worker(
            clone_reservation,
            clone_cursor,
            output_producer=output_producer,
            deadline_monotonic=time.monotonic() + 2.0,
        )

    clone_reservation.abort(cancelled=True)
    clone_reservation.release()


def test_raw_generic_worker_cannot_bypass_capture_session_reservation_authority(
    trusted_calibration,
):
    session = _source_session(trusted_calibration, points=1)
    capture_input = session.processor_input_binding
    bundle = _bind(trusted_calibration, session)
    processor = object.__getattribute__(bundle, "_processor")
    source = capture_input.stream
    clone, _clone_producer = AcquisitionStream.create(
        source.stream_id,
        capture_input.payload_contract,
        flow_control=source.flow_control,
        retention_events=source.retention_events,
        retention_bytes=source.retention_bytes,
        join_key_contract=capture_input.join_key_contract,
    )
    clone_reservation = clone.reserve(
        total_events=1,
        max_inflight_events=1,
        max_inflight_bytes=capture_input.payload_contract.max_retained_nbytes,
        trace_binding=TraceBinding("occupancy-raw-bypass", "readout"),
    )
    clone_cursor = clone_reservation.activate()
    output_producer, output_cursor, builder = _terminal_output(
        bundle,
        capture_input,
        run_id="occupancy-raw-bypass",
    )

    try:
        with pytest.raises(
            PermissionError,
            match="not minted by this CaptureSession",
        ):
            ExactStreamProcessorWorker(
                processor,
                clone_reservation,
                clone_cursor,
                input_edge=capture_input.input_edge,
                output_producer=output_producer,
                output_cursor=output_cursor,
                output_builder=builder,
                deadline_monotonic=time.monotonic() + 2.0,
            )
        assert not clone_reservation.consumer_bound
    finally:
        builder.close()
        clone_reservation.abort(cancelled=True)
        clone_reservation.release()


def test_raw_generic_worker_cannot_substitute_an_equivalent_cloned_output_edge(
    trusted_calibration,
):
    session = _source_session(trusted_calibration, points=1)
    capture_input = session.processor_input_binding
    bundle = _bind(trusted_calibration, session)
    processor = object.__getattribute__(bundle, "_processor")
    input_reservation = session.reserve_exact()
    input_cursor = input_reservation.activate()
    wrong_adapter = OccupancyDatasetEventAdapter(bundle.output_payload_contract)
    wrong_edge = FrozenDatasetEdge(
        bundle.output_schema,
        wrong_adapter,
        bundle.output_edge.expected_cells,
    )
    output_producer, output_cursor, builder = _terminal_output(
        bundle,
        capture_input,
        run_id=input_reservation.trace_binding.run_id,
        edge=wrong_edge,
    )

    try:
        with pytest.raises(
            PermissionError,
            match="not the occupancy bundle's frozen edge",
        ):
            ExactStreamProcessorWorker(
                processor,
                input_reservation,
                input_cursor,
                input_edge=capture_input.input_edge,
                output_producer=output_producer,
                output_cursor=output_cursor,
                output_builder=builder,
                deadline_monotonic=time.monotonic() + 2.0,
            )
        assert not input_reservation.consumer_bound
    finally:
        builder.close()
        input_reservation.abort(cancelled=True)
        input_reservation.release()


def test_raw_generic_worker_cannot_reuse_guard_on_a_cloned_processor_binding(
    trusted_calibration,
):
    session = _source_session(trusted_calibration, points=1)
    capture_input = session.processor_input_binding
    bundle = _bind(trusted_calibration, session)
    processor = object.__getattribute__(bundle, "_processor")
    guard = processor.execution_guard
    cloned_processor = replace(
        processor,
        artifact_inputs=(),
        execution_guard=guard,
    )
    input_reservation = session.reserve_exact()
    input_cursor = input_reservation.activate()
    output_producer, output_cursor, builder = _terminal_output(
        bundle,
        capture_input,
        run_id=input_reservation.trace_binding.run_id,
    )

    try:
        with pytest.raises(
            PermissionError,
            match="not the one bound by the occupancy binder",
        ):
            ExactStreamProcessorWorker(
                cloned_processor,
                input_reservation,
                input_cursor,
                input_edge=capture_input.input_edge,
                output_producer=output_producer,
                output_cursor=output_cursor,
                output_builder=builder,
                deadline_monotonic=time.monotonic() + 2.0,
            )
        assert not input_reservation.consumer_bound
    finally:
        builder.close()
        input_reservation.abort(cancelled=True)
        input_reservation.release()


@pytest.mark.parametrize("termination", ("cancel", "deadline"))
def test_occupancy_worker_failure_releases_both_exact_reservations(
    trusted_calibration,
    termination,
):
    session = _source_session(trusted_calibration, points=1)
    capture_input = session.processor_input_binding
    bundle = _bind(trusted_calibration, session)
    input_reservation = session.reserve_exact()
    input_cursor = input_reservation.activate()
    output_producer, output_cursor, builder = _terminal_output(
        bundle,
        capture_input,
        run_id=input_reservation.trace_binding.run_id,
    )
    output_reservation = builder._reservation
    assert output_reservation is not None
    deadline = time.monotonic() + (0.25 if termination == "deadline" else 2.0)
    worker = bundle.create_exact_worker(
        input_reservation,
        input_cursor,
        output_producer=output_producer,
        output_cursor=output_cursor,
        output_builder=builder,
        deadline_monotonic=deadline,
    )

    worker.start()
    if termination == "cancel":
        worker.close(2.0)
    else:
        worker.wait(2.0)
        with pytest.raises(StreamProcessorError):
            worker.raise_if_failed()

    assert not worker.is_alive
    assert input_reservation.state is ReservationState.RELEASED
    assert output_reservation.state is ReservationState.RELEASED


def test_real_exact_worker_materializes_both_fields_by_frozen_cell_schedule(
    trusted_calibration,
):
    repeats = 2
    context_points = 3
    layout = PointLayout.explicit(
        (1, context_points),
        ((0, 2), (0, 0), (0, 1)),
    )
    measurement = _source_measurement(
        trusted_calibration,
        readout_events=1,
        points=context_points,
        repeat_count=repeats,
        point_layout=layout,
    )
    contract = measurement.capture_contract
    events = len(contract.expected_cells)

    def preflight(context: RunContext):
        session = measurement.capture_port.open_session(
            contract,
            TraceBinding(context.run_id.value, contract.source_id),
            measurement.capture_spec,
        )
        worker = None
        monitor = None
        try:
            capture_input = session.processor_input_binding
            bundle = _bind(trusted_calibration, session)
            source_reservation = session.reserve_exact()
            source_cursor = source_reservation.activate()
            output_contract = bundle.output_payload_contract
            output_budget = events * output_contract.max_retained_nbytes
            output, output_producer = AcquisitionStream.create(
                bundle.output_stream_id,
                output_contract,
                flow_control=ProducerFlowControl.BACKPRESSURE_CAPABLE,
                retention_events=events,
                retention_bytes=output_budget,
                join_key_contract=capture_input.join_key_contract,
            )
            output_reservation = output.reserve(
                total_events=events,
                max_inflight_events=events,
                max_inflight_bytes=output_budget,
                trace_binding=TraceBinding(
                    context.run_id.value,
                    bundle.output_source_id,
                ),
            )
            output_cursor = output_reservation.activate()
            builder = DatasetBuilder(
                BlockId("occupancy-output"),
                output_reservation,
                bundle.output_edge,
                DatasetMode.FINITE_EXACT,
            )
            worker = bundle.create_exact_worker(
                source_reservation,
                source_cursor,
                output_producer=output_producer,
                output_cursor=output_cursor,
                output_builder=builder,
                deadline_monotonic=time.monotonic() + 8.0,
            )
            assert isinstance(worker, ExactStreamProcessorWorker)
            monitor = output.monitor(max_events=events, max_bytes=output_budget)
            worker.start()
            session.bind_exact_consumer(worker.exact_readiness())
            session.prepare(context)
            prepared = (session, bundle, worker, monitor)
            return prepared
        except BaseException as error:
            try:
                session.fail(error)
            except BaseException:
                pass
            if worker is not None:
                worker.close(2.0)
            if monitor is not None:
                monitor.close()
            session.cleanup(context)
            raise

    def execute(context: RunContext, prepared):
        session, bundle, worker, monitor = prepared
        try:
            session.start(context)
            for _ in contract.expected_cells:
                context.checkpoint()
                session.capture_next(context)
            completion = session.complete(context)
            sealed = worker.finish(completion.eos, 8.0)
            envelopes = tuple(monitor.next().envelope for _ in range(events))
            return sealed, envelopes, bundle
        except BaseException as error:
            try:
                session.fail(error)
            except BaseException as failure_error:
                error.add_note(f"capture poison also failed: {failure_error!r}")
            worker.close(2.0)
            raise

    def cleanup(context: RunContext, prepared, _primary):
        if prepared is None:
            return measurement.capture_port.verify_idle(context)
        session, _bundle, worker, monitor = prepared
        software_errors: list[BaseException] = []
        for close in (lambda: monitor.close(), lambda: worker.close(2.0)):
            try:
                close()
            except BaseException as error:
                software_errors.append(error)
        report = session.cleanup(context)
        if not software_errors:
            return report
        return CleanupReport(
            safety_proofs=report.safety_proofs,
            decisions=report.decisions,
            errors=(*report.errors, *software_errors),
        )

    plan = RunPlan(
        name="one-stage exact camera to occupancy",
        mode=RunMode.FINITE_EXACT,
        resource_claims=(measurement.capture_port.resource_claim,),
        hazard_claims=(measurement.capture_port.hazard_claim,),
        bound_devices=(measurement.capture_port.device,),
        preflight=preflight,
        execute=execute,
        cleanup=cleanup,
        finalize=lambda _context, result: result,
        interrupt_operations=measurement.capture_port.interrupt_operations,
        timeout_seconds=12.0,
    )
    source_thread, source_failures = _deliver_when_armed(
        trusted_calibration.camera,
        [_frame(index, 1, index) for index in range(events)],
    )
    try:
        sealed, envelopes, bundle = trusted_calibration.runtime.controller.start(
            plan
        ).result(15.0)
    finally:
        if source_thread.is_alive():
            trusted_calibration.camera.finish_record_capture()
        source_thread.join(3.0)
    assert source_failures == []
    assert not source_thread.is_alive()

    assert sealed.block.schema == bundle.output_schema
    assert sealed.block.schema.repeat_axis == contract.dataset_schema.repeat_axis
    assert sealed.block.schema.point_axes == contract.dataset_schema.point_axes
    assert sealed.block.schema.point_layout == layout
    assert sealed.block.schema.cell_schema.dtype == np.dtype("<f8")
    assert sealed.block.values.shape == (repeats, layout.storage_size, 4)
    assert sealed.block.validity.mask.shape == (repeats, layout.storage_size, 4)
    assert all(
        isinstance(metadata, OccupancyDatasetMetadata)
        for metadata in sealed.event_metadata
    )
    assert all(
        metadata.occupied.values.shape == (4,)
        and metadata.source_metadata.correlation_id
        for metadata in sealed.event_metadata
    )
    assert sealed.provenance.derivation is not None
    assert sealed.provenance.derivation.artifact_inputs == (
        calibration_artifact_input_ref(trusted_calibration.calibration_ref),
    )
    keys = contract.expected_cells
    assert [envelope.join_key for envelope in envelopes] == list(keys)
    assert all(isinstance(envelope.payload, OccupancySample) for envelope in envelopes)
    for key, envelope, metadata in zip(
        keys,
        envelopes,
        sealed.event_metadata,
        strict=True,
    ):
        assert isinstance(metadata, OccupancyDatasetMetadata)
        payload = envelope.payload
        assert isinstance(payload, OccupancySample)
        np.testing.assert_array_equal(
            sealed.block.values[key.repeat_index, key.point_storage_index],
            payload.counts.values,
        )
        np.testing.assert_array_equal(
            sealed.block.validity.mask[
                key.repeat_index,
                key.point_storage_index,
            ],
            payload.counts.validity.mask,
        )
        np.testing.assert_array_equal(
            metadata.occupied.values,
            payload.occupied.values,
        )
        np.testing.assert_array_equal(
            metadata.occupied.validity.mask,
            payload.occupied.validity.mask,
        )
        assert envelope.trace.source_id == bundle.output_source_id
        assert envelope.trace.correlation_id == metadata.source_metadata.correlation_id
        assert envelope.trace.causation_refs[1] == calibration_artifact_input_ref(
            trusted_calibration.calibration_ref
        )


def test_compiled_occupancy_pipeline_returns_two_coherent_standard_snapshots(
    trusted_calibration,
):
    repeats = 2
    points = 3
    layout = PointLayout.explicit((points,), ((2,), (0,), (1,)))
    event_schedule = (
        DatasetCellAddress(0, 2),
        DatasetCellAddress(1, 0),
        DatasetCellAddress(0, 0),
        DatasetCellAddress(1, 2),
        DatasetCellAddress(0, 1),
        DatasetCellAddress(1, 1),
    )
    measurement = _source_measurement(
        trusted_calibration,
        points=points,
        repeat_count=repeats,
        point_layout=layout,
        cell_schedule=event_schedule,
    )
    spec = _occupancy_pipeline_spec(trusted_calibration, measurement)
    events = measurement.capture_contract.total_events
    frames = []
    for ordinal in range(events):
        image = np.array(_frame(ordinal, 1, 0), copy=True)
        x, y = _CENTERS[0]
        site = image[y - 1 : y + 2, x - 1 : x + 2].astype(np.uint32)
        image[y - 1 : y + 2, x - 1 : x + 2] = (
            site + ordinal * 11
        ).astype(np.uint16)
        frames.append(image)
    source_thread, source_failures = _deliver_when_armed(
        trusted_calibration.camera,
        frames,
    )
    try:
        result = trusted_calibration.runtime.controller.start(
            compile_occupancy_pipeline(spec)
        ).result(15.0)
    finally:
        if source_thread.is_alive():
            trusted_calibration.camera.finish_record_capture()
        source_thread.join(3.0)

    assert source_failures == []
    assert not source_thread.is_alive()
    assert isinstance(result, OccupancyPipelineResult)
    assert isinstance(result.dataset, FrozenOccupancyDataset)
    assert result.aggregate_peak_bytes <= (
        spec.materializer.memory.memory_limit_bytes
    )
    assert result.source_dataset_schema is measurement.capture_contract.dataset_schema
    assert not hasattr(result, "pipeline")
    assert result.capture_terminal.produced_count == events
    assert result.terminal_provenance.generation == (
        result.dataset.counts.ref.stream_generation
    )
    assert result.model_kind is spec.processor.model_kind
    assert result.calibration_reference == trusted_calibration.calibration_ref
    assert result.calibration_artifact_fingerprint == (
        trusted_calibration.admitted.artifact_fingerprint
    )
    assert result.calibration_admission_evidence_digest == (
        trusted_calibration.admitted.evidence_digest
    )
    assert result.processor_binding_digest == (
        result.terminal_provenance.derivation.stages[0].processor_binding_digest
    )
    with pytest.raises(TypeError, match="process-local"):
        pickle.dumps(result)
    with pytest.raises(TypeError, match="process-local"):
        pickle.dumps(result.dataset)
    with pytest.raises(AttributeError, match="immutable"):
        result.run_id = "forged-run"
    with pytest.raises(PermissionError, match="occupancy finalization"):
        FrozenOccupancyDataset(
            None,
            counts=result.dataset.counts,
            occupied=result.dataset.occupied,
            source_metadata=result.dataset.source_metadata_in_event_order,
            cell_schedule=result.dataset.cell_schedule,
        )
    with pytest.raises(PermissionError, match="compiler"):
        OccupancyPipelineResult(
            None,
            pipeline=object(),
            dataset=result.dataset,
            bound=object(),
        )
    assert result.dataset.counts is result.dataset.field(OccupancyDatasetField.COUNTS)
    assert result.dataset.occupied is result.dataset.field(
        OccupancyDatasetField.OCCUPIED
    )
    assert result.dataset.counts.ref.block_id == spec.materializer.counts_block_id
    assert result.dataset.occupied.ref.block_id == spec.materializer.occupied_block_id
    assert result.dataset.counts.ref.stream_generation == (
        result.dataset.occupied.ref.stream_generation
    )
    assert result.dataset.counts.block.schema.point_layout == layout
    assert result.dataset.occupied.block.schema.point_layout == layout
    assert result.dataset.counts.block.values.shape == (repeats, points, 4)
    assert result.dataset.occupied.block.values.shape == (repeats, points, 4)
    assert result.dataset.counts.block.values.dtype == np.dtype("<f8")
    assert result.dataset.occupied.block.values.dtype == np.dtype(bool)
    assert result.dataset.cell_schedule == measurement.capture_contract.expected_cells
    assert tuple(
        item.source_ordinal
        for item in result.dataset.source_metadata_in_event_order
    ) == tuple(range(events))
    assert result.dataset.events == tuple(
        zip(
            result.dataset.cell_schedule,
            result.dataset.source_metadata_in_event_order,
            strict=True,
        )
    )

    model = trusted_calibration.admitted.artifact.select_model(
        kind=spec.processor.model_kind
    )
    artifact = trusted_calibration.admitted.artifact
    frame_schema = measurement.capture_contract.dataset_schema.cell_schema
    expected_counts = tuple(
        apply_readout_model(
            model,
            frame_contract=artifact.frame_contract,
            site_map=artifact.site_map,
            frame=Value(frame, VALID, frame_schema),
        ).signals.values
        for frame in frames
    )
    assert len({float(values[0]) for values in expected_counts}) == events
    counts = result.dataset.counts.block
    occupied = result.dataset.occupied.block
    assert isinstance(counts.validity, ComponentValidity)
    assert isinstance(occupied.validity, ComponentValidity)
    assert counts.validity is occupied.validity
    for ordinal, ((cell, metadata), expected) in enumerate(
        zip(result.dataset.events, expected_counts, strict=True)
    ):
        assert metadata.source_ordinal == ordinal
        location = (cell.repeat_index, cell.point_storage_index)
        np.testing.assert_allclose(counts.values[location], expected)
    for cell in result.dataset.cell_schedule:
        location = (cell.repeat_index, cell.point_storage_index)
        np.testing.assert_array_equal(
            counts.validity.mask[location],
            occupied.validity.mask[location],
        )
        valid = counts.validity.mask[location]
        expected = np.zeros(valid.shape, dtype=bool)
        above = valid & model.header.occupied_above_thresholds
        below = valid & ~model.header.occupied_above_thresholds
        expected[above] = counts.values[location][above] > model.header.thresholds[above]
        expected[below] = counts.values[location][below] < model.header.thresholds[below]
        np.testing.assert_array_equal(occupied.values[location], expected)


def test_triggered_occupancy_arms_full_chain_before_fire_and_retains_lineage(
    trusted_calibration,
):
    spec = _triggered_occupancy_spec(trusted_calibration)
    artifact = spec.pulse_request.artifact
    cell_plan = spec.cell_plan
    armed_at_fire = []

    def observe_fire(_playback):
        state = trusted_calibration.camera._recent_state()
        with state["cond"]:
            armed_at_fire.append(bool(state["armed"]))

    trusted_calibration.sequencer.history.clear()
    trusted_calibration.sequencer.add_fire_listener(observe_fire)
    try:
        result = trusted_calibration.runtime.controller.start(
            compile_triggered_occupancy_pipeline(spec)
        ).result(15.0)
    finally:
        trusted_calibration.sequencer.remove_fire_listener(observe_fire)

    assert type(result) is TriggeredOccupancyPipelineResult
    assert type(result.occupancy) is OccupancyPipelineResult
    assert armed_at_fire == [True]
    assert result.dataset.counts.block.values.shape == (1, 1, 4)
    assert result.dataset.occupied.block.values.shape == (1, 1, 4)
    assert result.dataset.cell_schedule == cell_plan.expected_cells
    assert result.compiled_artifact is artifact
    assert result.compiled_artifact_digest == artifact.fingerprint
    assert (
        result.pulse_terminal.expected_trigger_counts_from_completed_schedule
        == (("ch11", 1),)
    )
    assert result.capture_terminal.produced_count == 1
    assert result.capture_terminal.drained_count == 1
    assert [item["action"] for item in trusted_calibration.sequencer.history] == [
        "prepare",
        "fire",
        "wait_done",
        "safe",
    ]
    with pytest.raises(TypeError, match="process-local"):
        pickle.dumps(result)
    with pytest.raises(PermissionError, match="only be minted"):
        TriggeredOccupancyPipelineResult(
            object(),
            occupancy=result.occupancy,
            pulse_terminal=result.pulse_terminal,
            trigger_channel=result.trigger_channel,
            compiled_artifact=result.compiled_artifact,
            cell_plan=result.cell_plan,
        )
    with pytest.raises(AttributeError, match="immutable"):
        result._trigger_channel = "other"


def test_triggered_occupancy_rejects_an_unwired_camera_channel_before_run(
    trusted_calibration,
):
    trusted_calibration.sequencer.history.clear()
    with pytest.raises(ValueError, match="not.*wired"):
        _triggered_occupancy_spec(
            trusted_calibration,
            trigger_channel="ch12",
        )
    assert trusted_calibration.sequencer.history == []


def test_triggered_occupancy_camera_start_failure_never_fires_and_cleans_both(
    trusted_calibration,
    monkeypatch,
):
    spec = _triggered_occupancy_spec(trusted_calibration)
    camera = trusted_calibration.camera
    sequencer = trusted_calibration.sequencer
    original_finish = camera.finish_record_capture
    camera_cleanup_after_actions = []

    def fail_arm(*_args, **_kwargs):
        raise RuntimeError("injected camera arm failure")

    def observed_finish():
        camera_cleanup_after_actions.append(
            tuple(item["action"] for item in sequencer.history)
        )
        return original_finish()

    monkeypatch.setattr(camera, "arm", fail_arm)
    monkeypatch.setattr(camera, "finish_record_capture", observed_finish)
    sequencer.history.clear()

    with pytest.raises(RunFailed) as failure:
        trusted_calibration.runtime.controller.start(
            compile_triggered_occupancy_pipeline(spec)
        ).result(10.0)

    assert failure.value.primary is not None
    assert failure.value.primary.original_type == "RuntimeError"
    actions = [item["action"] for item in sequencer.history]
    assert actions == ["prepare", "safe"]
    assert "fire" not in actions
    assert camera_cleanup_after_actions
    assert camera_cleanup_after_actions[-1][-1] == "safe"
    state = camera._recent_state()
    with state["cond"]:
        assert not state["armed"] and not state["pending"]


def test_triggered_occupancy_cancel_after_fire_cleans_pulse_before_camera(
    trusted_calibration,
    monkeypatch,
):
    import zlc_neutral_atom.timing.occupancy as timing_occupancy

    spec = _triggered_occupancy_spec(trusted_calibration)
    entered_capture = threading.Event()
    cleanup_order = []
    original_pulse_cleanup = timing_occupancy.PulseSession.cleanup
    original_occupancy_cleanup = timing_occupancy._OccupancyTransaction.cleanup

    def blocked_capture_all(self, context):
        entered_capture.set()
        while True:
            context.checkpoint()
            time.sleep(0.005)

    def observed_pulse_cleanup(self, context):
        cleanup_order.append("pulse")
        return original_pulse_cleanup(self, context)

    def observed_occupancy_cleanup(self, context):
        cleanup_order.append("occupancy")
        return original_occupancy_cleanup(self, context)

    monkeypatch.setattr(
        timing_occupancy._OccupancyTransaction,
        "capture_all",
        blocked_capture_all,
    )
    monkeypatch.setattr(
        timing_occupancy.PulseSession,
        "cleanup",
        observed_pulse_cleanup,
    )
    monkeypatch.setattr(
        timing_occupancy._OccupancyTransaction,
        "cleanup",
        observed_occupancy_cleanup,
    )
    trusted_calibration.sequencer.history.clear()
    handle = trusted_calibration.runtime.controller.start(
        compile_triggered_occupancy_pipeline(spec)
    )
    assert entered_capture.wait(3.0)
    handle.cancel("cancel after occupancy FIRE")
    with pytest.raises(RunCancelled):
        handle.result(10.0)

    assert cleanup_order == ["pulse", "occupancy"]
    assert "fire" in {
        item["action"] for item in trusted_calibration.sequencer.history
    }
    state = trusted_calibration.camera._recent_state()
    with state["cond"]:
        assert not state["armed"] and not state["pending"]
    assert dict(trusted_calibration.sequencer.snapshot())["state"] == "safe"


def test_triggered_occupancy_rejects_tampered_pulse_terminal_before_publication(
    trusted_calibration,
    monkeypatch,
):
    import zlc_neutral_atom.timing.occupancy as timing_occupancy

    spec = _triggered_occupancy_spec(trusted_calibration)
    original_complete = timing_occupancy.PulseSession.complete

    def tampering_complete(self, context):
        terminal = original_complete(self, context)
        receipt = replace(
            terminal.receipt,
            expected_trigger_counts_from_completed_schedule=(("ch11", 2),),
        )
        forged = replace(terminal, receipt=receipt)
        # Deliberately bypass session identity so this test still exercises the
        # independent artifact/count validation at the result boundary.
        self._terminal = forged
        return forged

    monkeypatch.setattr(
        timing_occupancy.PulseSession,
        "complete",
        tampering_complete,
    )
    trusted_calibration.sequencer.history.clear()
    with pytest.raises(RunFailed) as failure:
        trusted_calibration.runtime.controller.start(
            compile_triggered_occupancy_pipeline(spec)
        ).result(15.0)

    assert failure.value.primary is not None
    assert failure.value.primary.original_type == "ValueError"
    assert "expected counts differ" in str(failure.value.primary)


def test_triggered_occupancy_rejects_a_terminal_from_another_run(
    trusted_calibration,
    monkeypatch,
):
    import zlc_neutral_atom.timing.occupancy as timing_occupancy

    spec = _triggered_occupancy_spec(trusted_calibration)
    first = trusted_calibration.runtime.controller.run(
        compile_triggered_occupancy_pipeline(spec)
    )
    original_complete = timing_occupancy.PulseSession.complete

    def stale_complete(self, context):
        current = original_complete(self, context)
        assert current.session_id != first.pulse_terminal.session_id
        return first.pulse_terminal

    monkeypatch.setattr(
        timing_occupancy.PulseSession,
        "complete",
        stale_complete,
    )
    with pytest.raises(RunFailed) as failure:
        trusted_calibration.runtime.controller.run(
            compile_triggered_occupancy_pipeline(spec)
        )

    assert failure.value.primary is not None
    assert failure.value.primary.original_type == "PermissionError"
    assert "not minted" in str(failure.value.primary)


def test_triggered_occupancy_rejects_a_stale_receipt_rebadged_to_current_session(
    trusted_calibration,
    monkeypatch,
):
    import zlc_neutral_atom.timing.occupancy as timing_occupancy

    spec = _triggered_occupancy_spec(trusted_calibration)
    first = trusted_calibration.runtime.controller.run(
        compile_triggered_occupancy_pipeline(spec)
    )
    original_complete = timing_occupancy.PulseSession.complete

    def rebadged_complete(self, context):
        current = original_complete(self, context)
        return replace(
            first.pulse_terminal,
            session_id=current.session_id,
        )

    monkeypatch.setattr(
        timing_occupancy.PulseSession,
        "complete",
        rebadged_complete,
    )
    with pytest.raises(RunFailed) as failure:
        trusted_calibration.runtime.controller.run(
            compile_triggered_occupancy_pipeline(spec)
        )

    assert failure.value.primary is not None
    assert failure.value.primary.original_type == "PermissionError"
    assert "not minted" in str(failure.value.primary)


def test_triggered_occupancy_rejects_an_executed_value_from_another_run(
    trusted_calibration,
):
    spec = _triggered_occupancy_spec(trusted_calibration)
    plan = compile_triggered_occupancy_pipeline(spec)
    original_execute = plan.execute
    executed_values = []

    def recording_execute(context, prepared):
        executed = original_execute(context, prepared)
        executed_values.append(executed)
        return executed

    trusted_calibration.runtime.controller.run(
        replace(plan, execute=recording_execute)
    )
    assert len(executed_values) == 1

    def stale_execute(context, prepared):
        original_execute(context, prepared)
        return executed_values[0]

    with pytest.raises(RunFailed) as failure:
        trusted_calibration.runtime.controller.run(
            replace(
                plan,
                execute=stale_execute,
            )
        )

    assert failure.value.primary is not None
    assert failure.value.primary.original_type == "ValueError"
    assert "another Run" in str(failure.value.primary)


def test_occupancy_pipeline_memory_rejects_before_camera_prepare(
    trusted_calibration,
):
    measurement = _source_measurement(trusted_calibration, points=1)
    state = trusted_calibration.camera._recent_state()
    with state["cond"]:
        assert not state["armed"]
    handle = trusted_calibration.runtime.controller.start(
        compile_occupancy_pipeline(
            _occupancy_pipeline_spec(
                trusted_calibration,
                measurement,
                memory_limit=1,
            )
        )
    )
    with pytest.raises(RunFailed) as failure:
        handle.result(5.0)
    assert failure.value.primary is not None
    assert failure.value.primary.original_type == "MemoryError"
    assert failure.value.primary.__traceback__ is None
    assert tuple(
        note
        for note in getattr(failure.value.primary, "__notes__", ())
        if note.startswith("detached run traceback: ")
    )
    assert not any(
        "teardown" in note.lower()
        for note in getattr(failure.value.primary, "__notes__", ())
    )
    with state["cond"]:
        assert not state["armed"]


def test_occupancy_pipeline_admits_retained_calibration_memory_before_prepare(
    trusted_calibration,
    monkeypatch,
):
    import zlc_neutral_atom.readout.occupancy_pipeline as occupancy_pipeline

    measurement = _source_measurement(trusted_calibration, points=1)
    state = trusted_calibration.camera._recent_state()
    with state["cond"]:
        assert not state["armed"]
    monkeypatch.setattr(
        occupancy_pipeline,
        "calibration_retained_array_nbytes",
        lambda _artifact: 1 << 40,
    )
    handle = trusted_calibration.runtime.controller.start(
        compile_occupancy_pipeline(
            _occupancy_pipeline_spec(
                trusted_calibration,
                measurement,
                memory_limit=256 << 20,
            )
        )
    )
    with pytest.raises(RunFailed) as failure:
        handle.result(5.0)

    assert failure.value.primary is not None
    assert failure.value.primary.original_type == "MemoryError"
    with state["cond"]:
        assert not state["armed"]


def test_occupancy_pipeline_spec_and_plan_are_reusable_across_runs(
    trusted_calibration,
):
    measurement = _source_measurement(trusted_calibration, points=2)
    plan = compile_occupancy_pipeline(
        _occupancy_pipeline_spec(trusted_calibration, measurement)
    )
    results = []
    for run_index in range(2):
        source_thread, source_failures = _deliver_when_armed(
            trusted_calibration.camera,
            [_frame(run_index, event, run_index) for event in range(2)],
        )
        try:
            result = trusted_calibration.runtime.controller.start(plan).result(15.0)
        finally:
            if source_thread.is_alive():
                trusted_calibration.camera.finish_record_capture()
            source_thread.join(3.0)
        assert source_failures == []
        assert not source_thread.is_alive()
        results.append(result)

    assert results[0].run_id != results[1].run_id
    assert (
        results[0].dataset.counts.ref.stream_generation
        != results[1].dataset.counts.ref.stream_generation
    )


def test_occupancy_pipeline_releases_live_graph_without_cyclic_gc(
    trusted_calibration,
):
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        measurement = _source_measurement(trusted_calibration, points=1)
        base = compile_occupancy_pipeline(
            _occupancy_pipeline_spec(trusted_calibration, measurement)
        )
        references = {}
        original_preflight = base.preflight

        def observed_preflight(context):
            prepared = original_preflight(context)
            worker = prepared.worker
            assert worker is not None
            builder = worker._output_sink._builder
            references.update(
                session=weakref.ref(prepared.session),
                worker=weakref.ref(worker),
                builder=weakref.ref(builder),
            )
            return prepared

        plan = replace(base, preflight=observed_preflight)
        source_thread, source_failures = _deliver_when_armed(
            trusted_calibration.camera,
            [_frame(0, 0, 0)],
        )
        try:
            result = trusted_calibration.runtime.controller.start(plan).result(15.0)
        finally:
            if source_thread.is_alive():
                trusted_calibration.camera.finish_record_capture()
            source_thread.join(3.0)

        assert isinstance(result, OccupancyPipelineResult)
        assert source_failures == []
        assert all(reference() is None for reference in references.values())
    finally:
        if was_enabled:
            gc.enable()


def test_occupancy_finalizer_checkpoints_each_event(
    trusted_calibration,
    monkeypatch,
):
    measurement = _source_measurement(trusted_calibration, points=4)
    plan = compile_occupancy_pipeline(
        _occupancy_pipeline_spec(trusted_calibration, measurement)
    )
    original_checkpoint = PostSafetyContext.checkpoint
    calls = []

    def cancelling_checkpoint(context):
        calls.append(len(calls))
        if len(calls) == 3:
            context.cancellation.request("cancel occupancy finalization")
        return original_checkpoint(context)

    monkeypatch.setattr(PostSafetyContext, "checkpoint", cancelling_checkpoint)
    source_thread, source_failures = _deliver_when_armed(
        trusted_calibration.camera,
        [_frame(0, event, 0) for event in range(4)],
    )
    try:
        handle = trusted_calibration.runtime.controller.start(plan)
        with pytest.raises((RunCancelled, RunFailed)):
            handle.result(15.0)
    finally:
        if source_thread.is_alive():
            trusted_calibration.camera.finish_record_capture()
        source_thread.join(3.0)

    assert source_failures == []
    assert len(calls) == 3


def test_occupancy_finalizer_rejects_source_metadata_not_proven_by_camera(
    trusted_calibration,
):
    measurement = _source_measurement(trusted_calibration, points=2)
    plan = compile_occupancy_pipeline(
        _occupancy_pipeline_spec(trusted_calibration, measurement)
    )
    trusted_finalize = plan.finalize

    def tampering_finalize(context, executed):
        terminal = executed.pipeline.dataset
        metadata = terminal.event_metadata
        first = metadata[0]
        assert isinstance(first, OccupancyDatasetMetadata)
        forged_source = replace(
            first.source_metadata,
            correlation_id="forged-processor-source-metadata",
        )
        object.__setattr__(
            terminal,
            "_event_metadata",
            (replace(first, source_metadata=forged_source), *metadata[1:]),
        )
        return trusted_finalize(context, executed)

    source_thread, source_failures = _deliver_when_armed(
        trusted_calibration.camera,
        [_frame(0, event, 0) for event in range(2)],
    )
    try:
        handle = trusted_calibration.runtime.controller.start(
            replace(plan, finalize=tampering_finalize)
        )
        with pytest.raises(RunFailed) as failure:
            handle.result(15.0)
    finally:
        if source_thread.is_alive():
            trusted_calibration.camera.finish_record_capture()
        source_thread.join(3.0)

    assert source_failures == []
    assert failure.value.primary is not None
    assert failure.value.primary.original_type == "RuntimeError"
    assert "source metadata differs from physical capture" in str(
        failure.value.primary
    )


def test_occupancy_finalizer_rejects_terminal_ref_generation_mismatch(
    trusted_calibration,
):
    measurement = _source_measurement(trusted_calibration, points=1)
    plan = compile_occupancy_pipeline(
        _occupancy_pipeline_spec(trusted_calibration, measurement)
    )
    trusted_finalize = plan.finalize

    def tampering_finalize(context, executed):
        terminal = executed.pipeline.dataset
        object.__setattr__(
            terminal,
            "_provenance",
            replace(
                terminal.provenance,
                generation=StreamGenerationId("forged-occupancy-generation"),
            ),
        )
        return trusted_finalize(context, executed)

    source_thread, source_failures = _deliver_when_armed(
        trusted_calibration.camera,
        [_frame(0, 0, 0)],
    )
    try:
        handle = trusted_calibration.runtime.controller.start(
            replace(plan, finalize=tampering_finalize)
        )
        with pytest.raises(RunFailed) as failure:
            handle.result(15.0)
    finally:
        if source_thread.is_alive():
            trusted_calibration.camera.finish_record_capture()
        source_thread.join(3.0)

    assert source_failures == []
    assert failure.value.primary is not None
    assert failure.value.primary.original_type == "RuntimeError"
    assert "ref generation differs from provenance" in str(failure.value.primary)


def test_contract_rejects_divergent_validity_and_noncanonical_invalid_fillers(
    trusted_calibration,
):
    session = _source_session(trusted_calibration, points=1)
    bundle = _bind(trusted_calibration, session)
    contract = bundle.output_payload_contract
    assert contract.finalization_scratch_nbytes == 32 * 4
    site_axis = contract.occupied_schema.data_axes[0]
    validity = ComponentValidity(
        (site_axis.axis_id,),
        np.array([True, False, True, True]),
    )
    occupied = Value(
        np.array([True, False, False, True]),
        validity,
        contract.occupied_schema,
    )
    bad_counts = Value(
        np.array([1.0, 2.0, 3.0, 4.0]),
        validity,
        contract.counts_schema,
    )
    with pytest.raises(ValueError, match="canonical zero"):
        contract.validate(OccupancySample(occupied, bad_counts, _metadata(0)))
    with pytest.raises(ValueError, match="canonical zero"):
        contract.digest_components(
            occupied.values,
            occupied.validity,
            bad_counts.values,
            bad_counts.validity,
            _metadata(0),
        )
    negative_zero_counts = Value(
        np.array([1.0, -0.0, 3.0, 4.0]),
        validity,
        contract.counts_schema,
    )
    with pytest.raises(ValueError, match="canonical zero"):
        contract.validate(
            OccupancySample(occupied, negative_zero_counts, _metadata(0))
        )
    nonfinite_counts = Value(
        np.array([1.0, 0.0, np.nan, 4.0]),
        validity,
        contract.counts_schema,
    )
    with pytest.raises(ValueError, match="must be finite"):
        contract.validate(
            OccupancySample(occupied, nonfinite_counts, _metadata(0))
        )
    with pytest.raises(ValueError, match="must be finite"):
        contract.digest_components(
            occupied.values,
            occupied.validity,
            nonfinite_counts.values,
            nonfinite_counts.validity,
            _metadata(0),
        )
    bad_occupied = Value(
        np.array([True, True, False, True]),
        validity,
        contract.occupied_schema,
    )
    with pytest.raises(ValueError, match="canonical False"):
        contract.digest_components(
            bad_occupied.values,
            bad_occupied.validity,
            np.array([1.0, 0.0, 3.0, 4.0]),
            validity,
            _metadata(0),
        )
    other_validity = ComponentValidity(
        (site_axis.axis_id,),
        np.array([True, True, True, True]),
    )
    with pytest.raises(ValueError, match="identical component validity"):
        OccupancySample(
            occupied,
            Value(np.arange(4.0), other_validity, contract.counts_schema),
            _metadata(0),
        )


def test_site_axis_cannot_collide_with_capture_sampling_axes(trusted_calibration):
    artifact = trusted_calibration.admitted.artifact
    session = _source_session(
        trusted_calibration,
        points=1,
        repeat_axis_id=artifact.site_map.site_axis.axis_id.value,
    )
    with pytest.raises(ValueError, match="site AxisId collides"):
        _bind(trusted_calibration, session)


def test_public_processor_surface_has_no_two_stage_api():
    import zlc_neutral_atom.readout.occupancy as occupancy

    assert not hasattr(occupancy, "OccupancyStreamProcessorBindRequest")
    assert "capture_input" not in occupancy.OccupancyStreamProcessorSpec.__dataclass_fields__
    assert "output_field" not in occupancy.OccupancyStreamProcessorSpec.__dataclass_fields__
    assert not hasattr(occupancy, "ReadoutSignalProcessorBindRequest")
    assert not hasattr(occupancy, "bind_readout_signal_processor")
    assert not hasattr(occupancy, "bind_occupancy_processor")
    assert not hasattr(occupancy, "occupancy_stream_operator")
    assert "repository" not in occupancy._occupancy_stream_operator.__code__.co_names
    assert "session" not in occupancy._occupancy_stream_operator.__code__.co_names
