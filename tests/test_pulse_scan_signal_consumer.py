"""Current PulseScan contract: sequence an external signal, never acquire it."""

from __future__ import annotations

from dataclasses import replace
import math
from pathlib import Path
import threading
import time

import numpy as np
import pytest

import Zou_lab_control.api as zlc
import Zou_lab_control.api._application_services as application_services_impl
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
from zlc_neutral_atom.devices.camera.contract import (
    CameraFrameMetadata,
    CameraSample,
    CameraSampleContract,
)
from zlc_neutral_atom.devices.sequencer.port import (
    PulseSession,
    PulseTerminalAck,
    SimulatedPulseReceipt,
)
from zlc_neutral_atom.installation import DeviceRef
from zlc_neutral_atom.logic_nodes.camera_measurement.definition import (
    CAMERA_MEASUREMENT_KEY,
    camera_frame_output_declarations,
)
from zlc_neutral_atom.logic_nodes.camera_measurement.monitor import (
    CameraMonitorViewSpec,
)
from zlc_neutral_atom.logic_nodes.camera_measurement.definition import (
    CameraMeasurementRequest,
)
from zlc_neutral_atom.logic_nodes.camera_measurement.signal_source import (
    camera_signal_event_source,
)
from zlc_neutral_atom.timing.pulse_parameter_scan import (
    ApiSegmentTable,
    ApiSlotSegmentedProgram,
    AutonomousScanSlotProgram,
)
from zlc_neutral_atom.logic_nodes.pulse_scan.authoring import (
    DEFAULT_PULSE_SCAN_PULSE_PATH,
)
from zlc_neutral_atom.logic_nodes.readout.occupancy.declaration import (
    OCCUPANCY_LOGIC_NODE,
)
from zlc_neutral_atom.logic_nodes.pulse_scan.source_binding import (
    PulseScanBoundRequest,
    ScanSignalBinding,
)
from zlc_neutral_atom.logic_nodes.pulse_scan.lineage import (
    ApiSegmentedScanExecution,
    pulse_scan_execution_to_tree,
)
from zlc_neutral_atom.runtime.signal_source import (
    SignalAssociationRequest,
    SignalAssociationUnavailable,
    SignalEventAssociationSource,
    SignalOutputProjection,
    StreamSignalEventSource,
)
from zlc_neutral_atom.runtime.streams import (
    AcquisitionStream,
    SourceFailed,
    StreamId,
)
from zlc_pulse import FrozenScanTable, load_pulse_document


ROOT = Path(__file__).resolve().parents[1]
_CAMERA_SCAN_PULSE = ROOT / "pulses" / "camera_imaging_address_switch.json"


def _workspace(project: Path) -> zlc.WorkspacePaths:
    project = project.resolve()
    return zlc.WorkspacePaths(
        project_root=project,
        pulses_root=(ROOT / "pulses").resolve(),
        tasks_root=(ROOT / "tasks").resolve(),
        output_root=project / "_output",
    )


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

    def bind(self, dataset) -> None:
        self.dataset = dataset

    def updated(self) -> None:
        return None

    def notification_failed(self, message: str) -> None:
        self.failure = message

    def fail(self, message: str) -> None:
        self.failure = message

    def source_terminal(self) -> None:
        return None


def _start_virtual_readout_camera(
    experiment,
    *,
    frames_per_cycle: int = 1,
    exposure: float | None = None,
):
    request = experiment.nodes.camera_measurement.camera_measurement_request(
        camera_role="camera",
        repeat=0,
        frames_per_cycle=frames_per_cycle,
        exposure=exposure,
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


def _autonomous_camera_program(
    rows: tuple[tuple[int], ...],
    *,
    sweep_count: int = 1,
):
    document = load_pulse_document(_CAMERA_SCAN_PULSE)
    columns = tuple(
        parameter.parameter_id for parameter in document.scan_parameters
    )
    return AutonomousScanSlotProgram(
        replace(
            document,
            scan_table=FrozenScanTable(columns, rows),
            scan_sweep_count=sweep_count,
        )
    )


def _camera_scan_binding(*, transform: DataTransformSpec | None = None):
    return ScanSignalBinding(
        CAMERA_MEASUREMENT_KEY,
        camera_frame_output_declarations(1)[0],
        transform,
    )


def _association_request(**changes) -> SignalAssociationRequest:
    values = {
        "cause_id": "pulse-session",
        "cause_digest": "a" * 64,
        "expected_event_count": 1,
        "trigger_schedule_fingerprint": "b" * 64,
        "trigger_channel": "ch11",
        "trigger_count": 3,
        "minimum_trigger_interval_ticks": 1,
        "clock_hz": 1,
    }
    values.update(changes)
    return SignalAssociationRequest(**values)


class _PublicationAuthority:
    """Small physical-boundary witness; stream publication remains production code."""

    def __init__(self, physical_end: int) -> None:
        self.physical_end = physical_end
        self.token = object()
        self.terminal_bound = False
        self.finish_entered = threading.Event()

    def arm_signal_event_association(
        self,
        request,
        trigger_group_size,
        expected_group_count,
    ):
        assert request.trigger_count == self.physical_end
        assert trigger_group_size * expected_group_count == self.physical_end
        return self.token, 0

    def bind_signal_event_association(
        self,
        token,
        *,
        artifact_digest,
        trigger_counts,
        terminal_evidence_kind,
    ):
        assert token is self.token
        assert artifact_digest == "a" * 64
        assert trigger_counts == (("ch11", self.physical_end),)
        assert terminal_evidence_kind == "SIMULATED"
        self.terminal_bound = True
        return "ch11", 0, self.physical_end

    def finish_signal_event_association(self, token):
        assert token is self.token
        assert self.terminal_bound
        self.finish_entered.set()
        return "ch11", 0, self.physical_end

    def cancel_signal_event_association(self, token) -> None:
        assert token is self.token


_ASSOCIATED_CAMERA_SCHEMA = ValueSchema.scalar(np.dtype("<f8"), "count")


def _associated_camera_cursor(*, operation_deadline_seconds: float = 1.0):
    contract = CameraSampleContract(_ASSOCIATED_CAMERA_SCHEMA)
    stream, producer = AcquisitionStream.create(
        StreamId("camera-associated-publication"),
        contract,
    )
    authority = _PublicationAuthority(3)
    source = camera_signal_event_source(
        stream,
        CameraMeasurementRequest(
            DeviceRef("installation", "runtime", "camera"),
            repeat=0,
            frames_per_cycle=3,
        ),
        contract,
        association_authority=authority,
        trigger_channel="ch11",
        operation_deadline_seconds=operation_deadline_seconds,
    )
    source.mark_association_running()
    cursor = source.open_associated_signal_cursor("frame_0")
    request = _association_request()
    token = cursor.arm_signal_association(request)
    terminal = PulseTerminalAck(
        request.cause_id,
        "pulse-binding",
        SimulatedPulseReceipt(
            request.cause_digest,
            "test-simulator",
            (("ch11", 3),),
            0.0,
            0.0,
        ),
    )
    cursor.bind_signal_association(token, terminal)
    return stream, producer, authority, cursor, token


def _emit_associated_camera_sample(producer, sequence: int) -> None:
    sample = CameraSample(
        Value(
            np.asarray([float(sequence)], dtype="<f8"),
            VALID,
            _ASSOCIATED_CAMERA_SCHEMA,
        ),
        CameraFrameMetadata(
            source_ordinal=sequence,
            produced_count=sequence + 1,
            frame_stamp=sequence,
            camera_stamp=sequence,
            timestamp_seconds=None,
            timestamp_microseconds=None,
            host_received_at_ns=sequence + 1,
            driver_buffer_index=sequence,
        ),
    )
    producer.emit(
        sample,
        captured_at=sample.metadata.captured_at,
    )


def test_signal_association_request_requires_physical_spacing_for_multiple_triggers() -> None:
    with pytest.raises(ValueError, match="requires its minimum interval"):
        _association_request(minimum_trigger_interval_ticks=None)

    single = _association_request(
        trigger_count=1,
        minimum_trigger_interval_ticks=None,
    )
    assert single.minimum_trigger_interval_ticks is None


@pytest.mark.parametrize(
    ("invalid_interval", "error_type"),
    (
        (0, ValueError),
        (-1, ValueError),
        (True, TypeError),
        (1.0, TypeError),
    ),
)
def test_signal_association_request_rejects_nonpositive_or_noninteger_spacing(
    invalid_interval,
    error_type,
) -> None:
    with pytest.raises(error_type, match="minimum_trigger_interval_ticks"):
        _association_request(
            minimum_trigger_interval_ticks=invalid_interval,
        )


def test_virtual_camera_preflight_uses_the_current_exposure_working_point(
    tmp_path,
) -> None:
    exposure_seconds = 0.02
    with zlc.connect(
        "virtual",
        workspace=_workspace(tmp_path / "workspace"),
    ) as experiment:
        source, camera_handle = _start_virtual_readout_camera(
            experiment,
            exposure=exposure_seconds,
        )
        cursor = source.open_associated_signal_cursor("frame_0")
        clock_hz = int(experiment.pulse.target.clock_hz)
        required_ticks = math.ceil(exposure_seconds * clock_hz)
        try:
            with pytest.raises(ValueError, match="faster than the camera working point"):
                cursor.arm_signal_association(
                    _association_request(
                        expected_event_count=2,
                        trigger_count=2,
                        minimum_trigger_interval_ticks=required_ticks - 1,
                        clock_hz=clock_hz,
                    )
                )
            token = cursor.arm_signal_association(
                _association_request(
                    expected_event_count=2,
                    trigger_count=2,
                    minimum_trigger_interval_ticks=required_ticks,
                    clock_hz=clock_hz,
                )
            )
            assert token is not None
        finally:
            cursor.close()
            _stop_virtual_readout_camera(camera_handle)


def test_camera_association_waits_for_every_trailing_sibling_publication() -> None:
    _stream, producer, authority, cursor, token = _associated_camera_cursor()
    _emit_associated_camera_sample(producer, 0)
    selected = cursor.next_associated_signal(token, 0.2)
    assert selected.event_ref.sequence == 0

    started = threading.Event()
    completed = threading.Event()
    results = []
    errors = []

    def finish() -> None:
        started.set()
        try:
            results.append(cursor.finish_signal_association(token))
        except BaseException as error:
            errors.append(error)
        finally:
            completed.set()

    worker = threading.Thread(target=finish, name="camera-association-finish")
    worker.start()
    try:
        assert started.wait(0.2)
        assert not authority.finish_entered.wait(0.05)
        _emit_associated_camera_sample(producer, 1)
        assert not authority.finish_entered.wait(0.05)
        _emit_associated_camera_sample(producer, 2)
        assert completed.wait(0.5)
        assert authority.finish_entered.is_set()
        assert not errors
        assert results == [None]
    finally:
        worker.join(1.0)
        cursor.close()
        producer.finish()


def test_camera_association_rejects_a_terminal_stream_at_reached_frontier() -> None:
    _stream, producer, authority, cursor, token = _associated_camera_cursor()
    for sequence in range(3):
        _emit_associated_camera_sample(producer, sequence)
    selected = cursor.next_associated_signal(token, 0.2)
    assert selected.event_ref.sequence == 0
    failure = SourceFailed("camera publication failed")
    producer.fail(failure)

    try:
        with pytest.raises(SourceFailed) as caught:
            cursor.finish_signal_association(token)
        assert caught.value is failure
        assert not authority.finish_entered.is_set()
    finally:
        cursor.close()


def test_pulse_scan_consumes_virtual_camera_without_claiming_producer(
    tmp_path,
    monkeypatch,
) -> None:
    started_plans = []
    start_run = application_services_impl.application_start_run

    def observe_start_run(services, plan, **kwargs):
        started_plans.append(plan)
        return start_run(services, plan, **kwargs)

    monkeypatch.setattr(
        application_services_impl,
        "application_start_run",
        observe_start_run,
    )

    with zlc.connect(
        "virtual",
        workspace=_workspace(tmp_path / "workspace"),
    ) as experiment:
        program = experiment.nodes.pulse_scan.scan_slot_program(
            "camera_imaging_address_switch.json",
            rows=((50_000_000,), (60_000_000,)),
            scan_sweep_count=1,
        )
        source, camera_handle = _start_virtual_readout_camera(experiment)
        request = experiment.nodes.pulse_scan.bind_scan(
            program,
            source,
            output_name="frame_0",
        )
        try:
            reference = experiment.nodes.pulse_scan.run_scan(request, source)
            plans = [
                plan
                for plan in started_plans
                if plan.name.startswith("Pulse scan")
            ]
            assert len(plans) == 1
            plan = plans[0]
            assert tuple(str(claim.key) for claim in plan.resource_claims) == (
                "device/sequencer",
            )
            assert tuple(str(device.key) for device in plan.bound_devices) == (
                "device/sequencer",
            )
            materialized = experiment.nodes.pulse_scan.materialize_scan(reference)
            artifact = experiment.nodes.pulse_scan.load_scan(reference)
            assert not camera_handle.snapshot().state.terminal
        finally:
            _stop_virtual_readout_camera(camera_handle)

    assert materialized.values.shape == (1, 2, 96, 128)
    assert materialized.values.dtype == np.dtype("<u2")
    assert artifact.execution.source.count == 2
    assert artifact.execution.terminal.session_id
    assert artifact.execution.artifact.fingerprint
    assert artifact.execution.program == program
    assert "program_fingerprint" not in pulse_scan_execution_to_tree(
        artifact.execution
    )


def test_public_scan_binding_requires_one_declared_owned_output(tmp_path) -> None:
    with zlc.connect(
        "virtual",
        workspace=_workspace(tmp_path / "workspace"),
    ) as experiment:
        program = experiment.nodes.pulse_scan.scan_slot_program(
            "camera_imaging_address_switch.json",
            rows=((50_000_000,),),
            scan_sweep_count=1,
        )
        camera_request = (
            experiment.nodes.camera_measurement.camera_measurement_request(
                camera_role="camera",
                repeat=0,
                frames_per_cycle=1,
            )
        )
        camera_source = (
            experiment.nodes.camera_measurement.prepare_camera_measurement(
                camera_request
            )
        )
        with pytest.raises(KeyError, match="no unique Dataset output"):
            experiment.nodes.pulse_scan.bind_scan(
                program,
                camera_source,
                output_name="missing",
            )

        scalar = ValueSchema.scalar(np.dtype("<f8"), "count")
        stream, producer = AcquisitionStream.create(
            StreamId("unowned-running-y"),
            ValuePayloadContract(scalar),
        )
        source = StreamSignalEventSource(
            stream,
            {
                "y": SignalOutputProjection(
                    scalar,
                    lambda envelope: envelope.payload,
                )
            },
        )
        try:
            with pytest.raises(TypeError, match="DefinitionKey"):
                experiment.nodes.pulse_scan.bind_scan(
                    program,
                    source,
                    output_name="y",
                )
        finally:
            producer.finish()


def test_virtual_camera_rejects_a_terminal_with_an_extra_trigger_channel(
    tmp_path,
    monkeypatch,
) -> None:
    original_complete = PulseSession.complete

    def complete_with_extra_channel(self, context):
        terminal = original_complete(self, context)
        receipt = replace(
            terminal.receipt,
            expected_trigger_counts_from_completed_schedule=(
                *terminal.expected_trigger_counts_from_completed_schedule,
                ("ch06", 1),
            ),
        )
        return replace(terminal, receipt=receipt)

    monkeypatch.setattr(PulseSession, "complete", complete_with_extra_channel)
    program = _autonomous_camera_program(((50_000_000,),))
    request = PulseScanBoundRequest(program, _camera_scan_binding())

    with zlc.connect(
        "virtual",
        workspace=_workspace(tmp_path / "workspace"),
    ) as experiment:
        source, camera_handle = _start_virtual_readout_camera(experiment)
        try:
            prepared = experiment.nodes.pulse_scan.prepare_scan(request, source)
            with pytest.raises(
                RuntimeError,
                match="virtual pulse terminal trigger count differs",
            ):
                prepared.start().result(5.0)
            assert not camera_handle.snapshot().state.terminal
        finally:
            _stop_virtual_readout_camera(camera_handle)


def test_camera_association_is_unavailable_before_the_producer_is_armed(
    tmp_path,
) -> None:
    with zlc.connect(
        "virtual",
        workspace=_workspace(tmp_path / "workspace"),
    ) as experiment:
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
    program = _autonomous_camera_program(((50_000_000,), (60_000_000,)))
    repository = tmp_path / "workspace"
    with zlc.connect("virtual", workspace=_workspace(repository)) as experiment:
        calibration_ref = experiment.nodes.calibration.sitemap(frames=4)
        assert (
            experiment.nodes.calibration.current_calibration_ref
            == calibration_ref
        )
        camera_source, camera_handle = _start_virtual_readout_camera(experiment)
        camera_binding = camera_source.dataset_output_binding("frame_0")
        prepared_occupancy = experiment.nodes.occupancy.prepare_occupancy_processor(
            camera_binding,
        )
        assert prepared_occupancy.request.calibration_ref == calibration_ref
        occupancy = prepared_occupancy.start_signal_events(camera_source)
        request = experiment.nodes.pulse_scan.bind_scan(
            program,
            occupancy,
            output_name="rate",
        )
        cursors = tuple(
            occupancy.open_signal_cursor(name)
            for name in ("counts", "occupied", "rate")
        )
        try:
            prepared = experiment.nodes.pulse_scan.prepare_scan(
                request,
                occupancy,
            )
            reference = prepared.start().result(5.0)
            sibling_events = tuple(
                tuple(cursor.next(timeout=1.0) for cursor in cursors)
                for _index in range(2)
            )
            materialized = experiment.nodes.pulse_scan.materialize_scan(reference)
            artifact = experiment.nodes.pulse_scan.load_scan(reference)
            assert not camera_handle.snapshot().state.terminal
        finally:
            for cursor in cursors:
                cursor.close()
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
    for index, (counts, occupied, rate) in enumerate(sibling_events):
        assert counts.event_ref is occupied.event_ref is rate.event_ref
        assert counts.direct_parent_refs == occupied.direct_parent_refs
        assert counts.direct_parent_refs == rate.direct_parent_refs
        assert counts.direct_parent_refs == sequence.direct_input_event_refs[index]

    with zlc.connect("virtual", workspace=_workspace(repository)) as experiment:
        reloaded = experiment.nodes.pulse_scan.load_scan(reference)
    assert reloaded.execution.source == sequence


def test_api_segmented_scan_uses_one_terminal_bound_association_per_cell(
    tmp_path,
) -> None:
    with zlc.connect(
        "virtual",
        workspace=_workspace(tmp_path / "workspace"),
    ) as experiment:
        program = experiment.nodes.pulse_scan.api_slot_program(
            DEFAULT_PULSE_SCAN_PULSE_PATH,
            rows=((2e-8,), (4e-8,)),
            scan_sweep_count=2,
        )
        source, camera_handle = _start_virtual_readout_camera(experiment)
        request = experiment.nodes.pulse_scan.bind_scan(
            program,
            source,
            output_name="frame_0",
        )
        try:
            prepared = experiment.nodes.pulse_scan.prepare_scan(request, source)
            reference = prepared.start().result(5.0)
            artifact = experiment.nodes.pulse_scan.load_scan(reference)
            materialized = experiment.nodes.pulse_scan.materialize_scan(reference)
            assert not camera_handle.snapshot().state.terminal
        finally:
            _stop_virtual_readout_camera(camera_handle)

    assert materialized.values.shape == (2, 2, 96, 128)
    assert materialized.values.dtype == np.dtype("<u2")
    segments = artifact.execution.segments
    assert artifact.execution.program == program
    assert "program_fingerprint" not in pulse_scan_execution_to_tree(
        artifact.execution
    )
    assert len(segments) == artifact.execution.source.count == 4
    assert len({item.terminal.session_id for item in segments}) == 4
    with pytest.raises(
        ValueError,
        match="repeat-major and point-fast",
    ):
        ApiSegmentedScanExecution(
            artifact.execution.program,
            tuple(reversed(segments)),
            artifact.execution.source,
        )


def test_scan_persists_the_single_committed_signal_projection_authority(
    tmp_path,
) -> None:
    program = _autonomous_camera_program(((50_000_000,),))
    repository = tmp_path / "workspace"
    with zlc.connect("virtual", workspace=_workspace(repository)) as experiment:
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
        request = experiment.nodes.pulse_scan.bind_scan(
            program,
            source,
            output_name="frame_0",
            transform=transform,
        )
        try:
            prepared = experiment.nodes.pulse_scan.prepare_scan(request, source)
            reference = prepared.start().result(5.0)
            materialized = experiment.nodes.pulse_scan.materialize_scan(reference)
        finally:
            _stop_virtual_readout_camera(camera_handle)

    with zlc.connect("virtual", workspace=_workspace(repository)) as experiment:
        artifact = experiment.nodes.pulse_scan.load_scan(reference)
    projection = artifact.execution.source.projection_authority
    assert projection.input_value_schema == image_schema
    assert projection.committed_transform is not None
    assert projection.committed_transform.spec == transform
    assert projection.output_value_schema == artifact.dataset_schema.cell_schema
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
                scan_sweep_count=1,
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
    with zlc.connect(
        "virtual",
        workspace=_workspace(tmp_path / "workspace"),
    ) as experiment:
        with pytest.raises(
            SignalAssociationUnavailable,
            match="software order only",
        ):
            experiment.nodes.pulse_scan.prepare_scan(request, source)

    assert not opened.is_set()
    assert fire_calls == []
