"""One-stage admitted-calibration camera -> occupancy contracts."""

from __future__ import annotations

from dataclasses import dataclass, replace
import pickle
import threading
import time

import numpy as np
import pytest

from Zou_lab_control.neutral_atom.devices.registry import DeviceSet
from Zou_lab_control.neutral_atom.devices.virtual import VirtualCamera, VirtualTrapArray
from zlc_data import (
    READOUT_EVENT,
    REPEAT,
    SCAN_POINT,
    AxisId,
    AxisSpec,
    BlockId,
    ComponentValidity,
    PointLayout,
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
    CalibrationAnalysisRequest,
    PsfAnalysisConfig,
)
from zlc_neutral_atom.readout.calibration import (
    BackgroundMode,
    BoxReadoutModel,
    BoxReducer,
    CalibrationResourceExceeded,
    ReadoutModelKind,
    bind_readout_feature_spec,
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
    OccupancyDatasetMetadata,
    OccupancyModelSelection,
    OccupancySample,
    OccupancySampleContract,
    OccupancyStreamProcessorSpec,
    bind_occupancy_stream_processor,
    resolve_occupancy_model_selection,
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
    RunContext,
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
from zlc_workbench.camera_capture import CameraCaptureBindingRequest
from zlc_workbench.legacy_neutral_atom import LegacyNeutralAtomRuntime


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
        box=BoxAnalysisConfig(1, BoxReducer.SUM),
        model_kinds=(
            ReadoutModelKind.UNIFORM_PSF,
            ReadoutModelKind.BOX,
            ReadoutModelKind.PER_SITE_PSF,
        ),
        default_model_kind=ReadoutModelKind.BOX,
        psf=PsfAnalysisConfig(1, BackgroundMode.NONE, 0),
        train_fraction=0.6,
        random_seed=3817,
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
    capture_repository: CaptureRepository
    capture_ref: CaptureArtifactRef
    calibration_repository: CalibrationRepository
    calibration_ref: CalibrationArtifactRef
    admitted: AdmittedCalibration


@pytest.fixture(scope="module")
def trusted_calibration(tmp_path_factory):
    root = tmp_path_factory.mktemp("occupancy-trusted-calibration")
    camera = VirtualCamera(
        VirtualTrapArray(grid_shape=(2, 2), image_shape=(32, 32), seed=11),
        exposure=1e-3,
    )
    camera.recent_capacity = 64
    runtime = LegacyNeutralAtomRuntime(
        DeviceSet(
            {"readout": camera},
            {"readout": {"type": "VirtualCamera", "params": {}}},
        )
    )
    description = runtime.describe_camera("readout")
    repeat_axis = _axis("repeat", REPEAT, 10)
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
            PipelineMemoryProfile.for_current_runtime(128 << 20),
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
    cells = tuple(
        DatasetCellAddress(repeat, point)
        for repeat in range(repeat_axis.size)
        for point in range(layout.storage_size)
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
    selection: OccupancyModelSelection | None = None,
):
    if selection is None:
        selection = resolve_occupancy_model_selection(trusted.admitted)
    return bind_occupancy_stream_processor(
        OccupancyStreamProcessorSpec(
            calibration=trusted.admitted,
            readout_binding=ReadoutBindingKey("readout"),
            model=selection,
            output_stream_id=StreamId("occupancy.output"),
            output_source_id="occupancy",
        ),
        session.processor_input_binding,
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
    selection = resolve_occupancy_model_selection(trusted_calibration.admitted)
    assert selection.model_kind is ReadoutModelKind.BOX
    bound = _bind(trusted_calibration, session, selection=selection)
    assert isinstance(bound, BoundOccupancyStreamProcessor)
    assert bound.model_selection == selection
    assert bound.calibration_reference == trusted_calibration.calibration_ref
    assert bound.calibration_admission_evidence_digest == (
        trusted_calibration.admitted.evidence_digest
    )
    assert bound.artifact_inputs == (
        calibration_artifact_input_ref(trusted_calibration.calibration_ref),
    )
    assert not hasattr(bound, "processor")
    assert not hasattr(bound, "config")

    artifact = trusted_calibration.admitted.artifact
    selected_model = artifact.select_model(model_id=selection.model_id)
    assert bound.operator_scratch_nbytes == readout_application_scratch_nbytes(
        bind_readout_feature_spec(selected_model, artifact.site_map),
        artifact.frame_contract.frame_schema,
    )
    with pytest.raises(TypeError, match="AdmittedCalibration"):
        OccupancyStreamProcessorSpec(
            calibration=artifact,
            readout_binding=ReadoutBindingKey("readout"),
            model=selection,
            output_stream_id=StreamId("bad.raw-artifact"),
            output_source_id="bad-raw-artifact",
        )
    with pytest.raises(TypeError, match="AdmittedCalibration"):
        resolve_occupancy_model_selection(trusted_calibration.calibration_ref)

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
                model=selection,
                output_stream_id=StreamId("bad-forged-admission"),
                output_source_id="bad-forged-admission",
            ),
            session.processor_input_binding,
        )


def test_reusable_processor_spec_binds_each_capture_session_generation(
    trusted_calibration,
):
    selection = resolve_occupancy_model_selection(trusted_calibration.admitted)
    spec = OccupancyStreamProcessorSpec(
        calibration=trusted_calibration.admitted,
        readout_binding=ReadoutBindingKey("readout"),
        model=selection,
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

    assert first.model_selection == second.model_selection == selection
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


def test_bind_rejects_binding_model_version_and_resource_drift(trusted_calibration):
    session = _source_session(trusted_calibration, points=1)
    selection = resolve_occupancy_model_selection(trusted_calibration.admitted)
    with pytest.raises(ValueError, match="another readout binding"):
        bind_occupancy_stream_processor(
            OccupancyStreamProcessorSpec(
                calibration=trusted_calibration.admitted,
                readout_binding=ReadoutBindingKey("another-readout"),
                model=selection,
                output_stream_id=StreamId("occupancy.wrong-binding"),
                output_source_id="occupancy-wrong-binding",
            ),
            session.processor_input_binding,
        )
    with pytest.raises(ValueError, match="version/kind"):
        _bind(
            trusted_calibration,
            session,
            selection=replace(selection, model_version="wrong-version"),
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
                model=selection,
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
    model = artifact.select_model(model_id=bound.model_selection.model_id)
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


def test_bound_authority_is_sealed_and_rejects_config_drift(
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

    original_inputs = processor.artifact_inputs
    object.__setattr__(processor, "artifact_inputs", ())
    with pytest.raises(PermissionError, match="processor semantics changed"):
        _ = bundle.fingerprint
    object.__setattr__(processor, "artifact_inputs", original_inputs)
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


def test_contract_rejects_divergent_validity_and_noncanonical_invalid_fillers(
    trusted_calibration,
):
    session = _source_session(trusted_calibration, points=1)
    bundle = _bind(trusted_calibration, session)
    contract = bundle.output_payload_contract
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
