"""Current PulseScan contract: sequence an external signal, never acquire it."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import threading
import time

import numpy as np
import pytest

import Zou_lab_control.api as zlc
from zlc_data import (
    SITE,
    VALID,
    AxisId,
    AxisSpec,
    ComponentValidity,
    DataTransformSpec,
    IndexSelection,
    Selection,
    ValidityContract,
    Value,
    ValuePayloadContract,
    ValueSchema,
)
from zlc_neutral_atom.catalog import DefinitionKey
from zlc_neutral_atom.dataset_output import DatasetOutputDeclaration
from zlc_neutral_atom.logic_nodes.camera_measurement import (
    CAMERA_MEASUREMENT_KEY,
    CameraMonitorViewSpec,
    camera_frame_output_declarations,
)
from zlc_neutral_atom.timing.pulse_parameter_scan import (
    ApiSegmentTable,
    ApiSlotSegmentedProgram,
    AutonomousScanSlotProgram,
)
from zlc_neutral_atom.pulse_catalog import PROBE_PULSE_PATH
from zlc_neutral_atom.logic_nodes.readout.occupancy.processor import (
    OCCUPANCY_LIVE_OUTPUT_DECLARATIONS,
)
from zlc_neutral_atom.logic_nodes.readout.occupancy.signal_source import (
    AssociatedRunningOccupancySignalSource,
    OccupancySignalValues,
    OccupancySignalValuesContract,
)
from zlc_neutral_atom.logic_nodes.pulse_scan.source_binding import (
    PulseScanBoundRequest,
    ScanSignalBinding,
)
from zlc_neutral_atom.devices.sequencer.port import PulseSession
from zlc_neutral_atom.runtime.signal_source import (
    SignalAssociationUnavailable,
    SignalEventAssociationSource,
    SignalOutputProjection,
    StreamSignalEventSource,
)
from zlc_neutral_atom.runtime.streams import (
    AcquisitionStream,
    ArtifactInputRef,
    ProcessorStageProvenance,
    StreamId,
)
from zlc_pulse import FrozenScanTable, load_pulse_document
from zlc_storage import canonical_digest, decode, encode


ROOT = Path(__file__).resolve().parents[1]
_CAMERA_SCAN_PULSE = ROOT / "pulses" / "camera_imaging_address_switch.json"


class _OpeningSource:
    """Ordering-only source used to prove that the association gate refuses it."""

    def __init__(self, source, opened: threading.Event) -> None:
        self._source = source
        self._opened = opened

    def value_schema(self, output_name: str):
        return self._source.value_schema(output_name)

    def open_signal_cursor(self, output_name: str):
        cursor = self._source.open_signal_cursor(output_name)
        self._opened.set()
        return cursor


class _LiveCameraView:
    """Minimal view port; the production Camera command still owns the stream."""

    def __init__(self, spec: CameraMonitorViewSpec) -> None:
        self.spec = spec
        self.dataset = None
        self.failure: str | None = None

    def bind(self, dataset, *, run_id: str, causation_domain_id: str) -> None:
        assert run_id and causation_domain_id
        self.dataset = dataset

    def updated(self) -> None:
        return None

    def notification_failed(self, message: str) -> None:
        self.failure = message

    def fail(self, message: str) -> None:
        self.failure = message

    def source_terminal(self) -> None:
        return None


def _start_virtual_readout_camera(experiment, *, frames_per_cycle: int = 1):
    request = experiment.nodes.camera_measurement.camera_measurement_request(
        camera_role="camera",
        repeat=0,
        frames_per_cycle=frames_per_cycle,
    )
    source = experiment.nodes.camera_measurement.prepare_camera_measurement(request)
    assert isinstance(source, SignalEventAssociationSource)
    views: list[_LiveCameraView] = []

    def factory(spec: CameraMonitorViewSpec) -> _LiveCameraView:
        view = _LiveCameraView(spec)
        views.append(view)
        return view

    handle = source.start_with_view(factory=factory)
    deadline = time.monotonic() + 5.0
    while handle.snapshot().phase != "monitoring-camera":
        snapshot = handle.snapshot()
        if snapshot.state.terminal or time.monotonic() >= deadline:
            raise AssertionError(snapshot)
        time.sleep(0.005)
    assert views and views[0].failure is None
    return source, handle


def _stop_virtual_readout_camera(handle) -> None:
    if not handle.snapshot().state.terminal:
        handle.cancel("PulseScan production-source test complete")
    terminal = handle.wait(5.0)
    assert terminal.state.terminal, terminal


def _autonomous_camera_program(rows: tuple[tuple[int], ...]):
    document = load_pulse_document(_CAMERA_SCAN_PULSE)
    columns = tuple(
        parameter.parameter_id for parameter in document.scan_parameters
    )
    return AutonomousScanSlotProgram(
        replace(document, scan_table=FrozenScanTable(columns, rows))
    )


def _camera_scan_binding(*, transform: DataTransformSpec | None = None):
    return ScanSignalBinding(
        CAMERA_MEASUREMENT_KEY,
        camera_frame_output_declarations(1)[0],
        transform,
    )


def test_pulse_scan_consumes_virtual_camera_without_claiming_producer(
    tmp_path,
) -> None:
    program = _autonomous_camera_program(((50_000_000,), (60_000_000,)))
    request = PulseScanBoundRequest(
        program,
        _camera_scan_binding(),
    )

    with zlc.connect("virtual", repository=tmp_path / "workspace") as experiment:
        source, camera_handle = _start_virtual_readout_camera(experiment)
        try:
            prepared = experiment.nodes.pulse_scan.prepare_scan_source(request, source)
            assert len(prepared.descriptor.resource_claims) == 1
            assert "sequencer" in prepared.descriptor.resource_claims[0]
            reference = prepared.start().result(5.0)
            materialized = experiment.nodes.pulse_scan.materialize_scan(reference)
            artifact = experiment.nodes.pulse_scan.load_scan(reference)
            assert not camera_handle.snapshot().state.terminal
        finally:
            _stop_virtual_readout_camera(camera_handle)

    assert materialized.values.shape == (1, 2, 96, 128)
    assert materialized.values.dtype == np.dtype("<u2")
    assert artifact.execution.source.source_run_id == camera_handle.run_id.value
    assert artifact.execution.source.source_id == "camera"
    associations = artifact.execution.source.associations
    assert len(associations) == 1
    assert associations[0].evidence_schema_id.endswith(
        "camera-measurement.pulse-association"
    )
    assert associations[0].request.expected_event_count == 2
    assert associations[0].request.cause_id == artifact.execution.terminal.session_id
    assert (
        associations[0].request.cause_digest
        == artifact.execution.artifact.fingerprint
    )
    evidence = decode(associations[0].canonical_evidence)
    assert evidence["trigger_channel"] == "ch11"
    assert evidence["physical_end_ordinal"] - evidence["physical_start_ordinal"] == 2


def test_camera_association_is_unavailable_before_the_producer_is_armed(
    tmp_path,
) -> None:
    with zlc.connect("virtual", repository=tmp_path / "workspace") as experiment:
        request = experiment.nodes.camera_measurement.camera_measurement_request(
            camera_role="camera",
            repeat=0,
            frames_per_cycle=1,
        )
        source = experiment.nodes.camera_measurement.prepare_camera_measurement(request)
        assert isinstance(source, SignalEventAssociationSource)
        with pytest.raises(SignalAssociationUnavailable, match="already-running"):
            source.open_associated_signal_cursor("frame_0")


def test_occupancy_scan_artifact_round_trips_expandable_same_shot_lineage(
    tmp_path,
) -> None:
    site_axis = AxisSpec(AxisId("scan-live-site"), "site", SITE, 1, (0,))
    site_validity = ValidityContract.components(site_axis.axis_id)
    counts_schema = ValueSchema(
        (site_axis,),
        site_validity,
        np.dtype("<f8"),
        "count",
    )
    occupied_schema = ValueSchema(
        (site_axis,),
        site_validity,
        np.dtype(bool),
        "occupation",
    )
    rate_schema = ValueSchema.scalar(np.dtype("<f8"), None)
    occupancy_contract = OccupancySignalValuesContract(
        counts_schema,
        occupied_schema,
        rate_schema,
    )
    calibration_input = ArtifactInputRef(
        "tests.calibration-ref",
        encode(
            {
                "schema": "tests.calibration-ref",
                "artifact_id": "scan-calibration",
            }
        ),
        canonical_digest({"artifact": "scan-calibration-content"}),
    )
    occupancy_stage = ProcessorStageProvenance(
        canonical_digest({"processor": "occupancy-live-fixture"}),
        (calibration_input,),
    )

    def classify(frame: Value) -> OccupancySignalValues:
        mean_count = float(np.mean(frame.values))
        occupied_state = mean_count > 0.0
        validity = ComponentValidity(
            (site_axis.axis_id,),
            np.asarray((True,), dtype=bool),
        )
        return OccupancySignalValues(
            Value(
                np.asarray((mean_count,), dtype="<f8"),
                validity,
                counts_schema,
            ),
            Value(
                np.asarray((occupied_state,), dtype=bool),
                validity,
                occupied_schema,
            ),
            Value(
                np.asarray((float(occupied_state),), dtype="<f8"),
                VALID,
                rate_schema,
            ),
        )

    program = _autonomous_camera_program(((50_000_000,), (60_000_000,)))
    request = PulseScanBoundRequest(
        program,
        ScanSignalBinding(
            DefinitionKey(
                "zlc_neutral_atom.logic_nodes.readout.occupancy",
                "occupancy-processor",
            ),
            OCCUPANCY_LIVE_OUTPUT_DECLARATIONS[2],
        ),
    )
    repository = tmp_path / "workspace"
    with zlc.connect("virtual", repository=repository) as experiment:
        camera_source, camera_handle = _start_virtual_readout_camera(experiment)
        frame_schema = camera_source.value_schema("frame_0")
        camera_binding = camera_source.dataset_output_binding("frame_0")
        occupancy = AssociatedRunningOccupancySignalSource(
            camera_source,
            source_output_name="frame_0",
            frame_schema=frame_schema,
            contract=occupancy_contract,
            classify=classify,
            artifact_input=calibration_input,
            processor_stage=occupancy_stage,
            expected_source_stream_id=camera_binding.stream_id,
            expected_source_stream_generation=camera_binding.stream_generation,
        )
        try:
            source = occupancy
            prepared = experiment.nodes.pulse_scan.prepare_scan_source(request, source)
            reference = prepared.start().result(5.0)
            materialized = experiment.nodes.pulse_scan.materialize_scan(reference)
            artifact = experiment.nodes.pulse_scan.load_scan(reference)
            assert not camera_handle.snapshot().state.terminal
        finally:
            occupancy.request_close()
            deadline = time.monotonic() + 2.0
            while not occupancy.worker_idle and time.monotonic() < deadline:
                time.sleep(0.005)
            assert occupancy.worker_idle
            occupancy.join_closed()
            _stop_virtual_readout_camera(camera_handle)

    assert materialized.values.shape == (1, 2, 1)
    assert np.all(materialized.values >= 0.0)
    sequence = artifact.execution.source
    assert tuple(item.sequence for item in sequence.event_refs) == (0, 1)
    assert all(len(refs) == 1 for refs in sequence.direct_input_event_refs)
    assert sequence.processor_stages == (occupancy_stage,)
    assert sequence.artifact_inputs == (calibration_input,)
    assert sequence.source_run_id == camera_handle.run_id.value
    assert sequence.source_id.startswith("occupancy-associated:")
    assert len(sequence.associations) == 1
    association_payload = decode(sequence.associations[0].canonical_evidence)
    assert association_payload["schema"].endswith("occupancy.signal-association")
    assert association_payload["processor_stage"]["processor_binding_digest"] == (
        occupancy_stage.processor_binding_digest
    )
    assert (
        association_payload["upstream_evidence"]["evidence_schema_id"]
        .endswith("camera-measurement.pulse-association")
    )

    with zlc.connect("virtual", repository=repository) as experiment:
        reloaded = experiment.nodes.pulse_scan.load_scan(reference)
    assert reloaded.execution.source == sequence


def test_api_segmented_scan_uses_one_terminal_bound_association_per_cell(
    tmp_path,
) -> None:
    document = load_pulse_document(PROBE_PULSE_PATH)
    program = ApiSlotSegmentedProgram(
        document,
        ApiSegmentTable(
            ("probe_exposure",),
            ((2e-8,), (4e-8,)),
        ),
        "test one physical terminal-bound association per API segment",
    )
    request = PulseScanBoundRequest(
        program,
        _camera_scan_binding(),
    )

    with zlc.connect("virtual", repository=tmp_path / "workspace") as experiment:
        source, camera_handle = _start_virtual_readout_camera(experiment)
        try:
            prepared = experiment.nodes.pulse_scan.prepare_scan_source(request, source)
            reference = prepared.start().result(5.0)
            artifact = experiment.nodes.pulse_scan.load_scan(reference)
            materialized = experiment.nodes.pulse_scan.materialize_scan(reference)
            assert not camera_handle.snapshot().state.terminal
        finally:
            _stop_virtual_readout_camera(camera_handle)

    assert materialized.values.shape == (1, 2, 96, 128)
    assert materialized.values.dtype == np.dtype("<u2")
    associations = artifact.execution.source.associations
    segments = artifact.execution.segments
    assert len(associations) == len(segments) == 2
    assert tuple(item.request.expected_event_count for item in associations) == (1, 1)
    assert tuple(item.request.cause_id for item in associations) == tuple(
        item.terminal.session_id for item in segments
    )
    assert tuple(item.request.cause_digest for item in associations) == tuple(
        item.artifact.fingerprint for item in segments
    )
    assert len({item.request.cause_id for item in associations}) == 2
    assert all(
        item.evidence_schema_id.endswith(
            "camera-measurement.pulse-association"
        )
        for item in associations
    )


def test_scan_persists_the_single_committed_signal_projection_authority(
    tmp_path,
) -> None:
    program = _autonomous_camera_program(((50_000_000,),))
    repository = tmp_path / "workspace"
    with zlc.connect("virtual", repository=repository) as experiment:
        source, camera_handle = _start_virtual_readout_camera(experiment)
        image_schema = source.value_schema("frame_0")
        transform = DataTransformSpec(
            (
                Selection(
                    tuple(
                        IndexSelection(axis.axis_id, 1)
                        for axis in image_schema.data_axes
                    )
                ),
            )
        )
        request = PulseScanBoundRequest(
            program,
            _camera_scan_binding(transform=transform),
        )
        try:
            prepared = experiment.nodes.pulse_scan.prepare_scan_source(request, source)
            reference = prepared.start().result(5.0)
            materialized = experiment.nodes.pulse_scan.materialize_scan(reference)
        finally:
            _stop_virtual_readout_camera(camera_handle)

    with zlc.connect("virtual", repository=repository) as experiment:
        artifact = experiment.nodes.pulse_scan.load_scan(reference)
    projection = artifact.execution.source.projection_authority
    assert projection.input_value_schema == image_schema
    assert projection.input_schema_fingerprint == image_schema.fingerprint
    assert projection.committed_transform is not None
    assert projection.committed_transform.spec == transform
    assert projection.output_value_schema == artifact.source_dataset_schema.cell_schema
    assert (
        projection.output_schema_fingerprint
        == artifact.source_dataset_schema.cell_schema.fingerprint
    )
    assert artifact.output_contract.committed_transform is None
    assert materialized.values.shape == (1, 1, 1)
    assert materialized.values.dtype == np.dtype("<u2")


def test_pulse_scan_refuses_ordering_only_source_before_any_fire(
    tmp_path,
    monkeypatch,
) -> None:
    scalar = ValueSchema.scalar(np.dtype("<f8"), "count")
    stream, _producer = AcquisitionStream.create(
        StreamId("ordering-only-running-y"),
        ValuePayloadContract(scalar),
    )
    opened = threading.Event()
    source = _OpeningSource(
        StreamSignalEventSource(
            stream,
            {
                "y": SignalOutputProjection(
                    scalar,
                    lambda envelope: envelope.payload,
                )
            },
        ),
        opened,
    )
    document = load_pulse_document(ROOT / "pulses" / "mot_field_template.json")
    columns = tuple(parameter.parameter_id for parameter in document.scan_parameters)
    request = PulseScanBoundRequest(
        AutonomousScanSlotProgram(
            replace(
                document,
                scan_table=FrozenScanTable(columns, ((0, 0, 0),)),
            )
        ),
        ScanSignalBinding(
            DefinitionKey("tests", "ordering-only-running-y"),
            DatasetOutputDeclaration("y", "tests.ordering-only.dataset"),
        ),
    )
    fire_calls: list[str] = []
    original_fire = PulseSession.fire

    def record_fire(self, context) -> None:
        fire_calls.append(self.session_id)
        original_fire(self, context)

    monkeypatch.setattr(PulseSession, "fire", record_fire)
    with zlc.connect("virtual", repository=tmp_path / "workspace") as experiment:
        with pytest.raises(
            SignalAssociationUnavailable,
            match="software order only",
        ):
            experiment.nodes.pulse_scan.prepare_scan_source(request, source)

    assert not opened.is_set()
    assert fire_calls == []
