"""Direct Fit entry gets one typed display cell without narrowing authority."""

from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PyQt5 import QtCore, QtWidgets

from zlc_frontend.qt_widgets import ensure_qt_app
import pytest

import Zou_lab_control.notebook as zlc
from zlc_data import (
    READOUT_EVENT,
    REPEAT,
    SCAN_POINT,
    SPATIAL_X,
    SPATIAL_Y,
    AxisId,
    AxisSpec,
    BlockId,
    CoordinateFrameId,
    DataBlock,
    DatasetRevision,
    DatasetSchema,
    OwnedSnapshot,
    PointLayout,
    Selection,
    StreamGenerationId,
    VALID,
    ValidityContract,
    Value,
    ValueSchema,
    suggest_fit_draft,
)
from zlc_frontend.figure import (
    AxisViewRole,
    DatasetDescriptor,
    DatasetId,
    FigureDocument,
    FigureEvaluator,
    FigureLayer,
    RepeatViewMode,
    ResolvedDataset,
    ResolvedDatasetMap,
    SuggestionStatus,
    ViewIntent,
    fit_single_panel_presentation,
    suggest_view,
)
from zlc_neutral_atom.catalog import DefinitionKey
from zlc_neutral_atom.dataset_output import DatasetOutputDeclaration
from zlc_neutral_atom.devices.sequencer.port import (
    PulseTerminalAck,
    pulse_terminal_ack_to_tree,
)
from zlc_neutral_atom.logic_nodes.pulse_scan.source_binding import (
    PulseScanBoundRequest,
    ScanSignalBinding,
)
from zlc_neutral_atom.logic_nodes.readout.calibration.sitemap import load_sitemap_pulse
from zlc_neutral_atom.runtime.signal_source import (
    SignalAssociationEvidence,
    SignalAssociationRequest,
    SignalAssociationScheduleRequirement,
    SignalEvent,
)
from zlc_neutral_atom.runtime.streams import (
    EventRef,
    StreamId,
    TraceContext,
    event_id_for_sequence,
)
from zlc_neutral_atom.timing.pulse_parameter_scan import AutonomousScanSlotProgram
from zlc_pulse import FrozenScanTable, RepeatRegion, ScanParameter
from zlc_storage import canonical_digest, encode


ROOT = Path(__file__).resolve().parents[1]
PULSE = ROOT / "pulses" / "imaging_template.json"


def _axis(identity: str, role, size: int) -> AxisSpec:
    return AxisSpec(
        AxisId(identity),
        identity,
        role,
        size,
        tuple(range(size)),
        None,
        None,
    )


def _sparse_image_schema() -> DatasetSchema:
    repeat = _axis("repeat", REPEAT, 3)
    event = _axis("readout-event", READOUT_EVENT, 4)
    scan = _axis("scan-point", SCAN_POINT, 3)
    y_axis = _axis("camera-y", SPATIAL_Y, 3)
    x_axis = _axis("camera-x", SPATIAL_X, 5)
    return DatasetSchema(
        repeat,
        (event, scan),
        # Logical event zero is deliberately absent.  The first published
        # tuple is (event two, scan one), followed by (event one, scan two).
        PointLayout.from_mapping(
            (event.size, scan.size),
            ((2, 1), (1, 2)),
        ),
        ValueSchema(
            (y_axis, x_axis),
            ValidityContract.value(),
            np.dtype("<f8"),
            "count",
        ),
    )


def test_fit_single_panel_presentation_uses_first_physical_sparse_tuple() -> None:
    schema = _sparse_image_schema()
    seed = suggest_view(schema, ViewIntent.IMAGE)
    assert seed.status is SuggestionStatus.RESOLVED
    assert seed.spec is not None
    repeat_id = schema.repeat_axis.axis_id
    event_id = schema.point_axes[0].axis_id
    scan_id = schema.point_axes[1].axis_id
    assert seed.spec.binding(repeat_id).role is AxisViewRole.REDUCED
    assert seed.spec.binding(event_id).role is AxisViewRole.FACET
    assert seed.spec.binding(scan_id).role is AxisViewRole.SLIDER
    assert seed.spec.binding(scan_id).selector.index == 1

    selection, preferences = fit_single_panel_presentation(schema, seed.spec)
    assert selection is not None
    selected = {term.axis_id: term.index for term in selection.terms}
    assert selected == {repeat_id: 0, event_id: 2}
    assert preferences.repeat_mode is RepeatViewMode.LATEST

    resolved = suggest_view(
        schema,
        ViewIntent.IMAGE,
        selection,
        preferences,
    )
    assert resolved.status is SuggestionStatus.RESOLVED
    assert resolved.spec is not None
    assert resolved.spec.binding(repeat_id).role is AxisViewRole.SELECTED
    assert resolved.spec.binding(event_id).role is AxisViewRole.SELECTED

    values = np.arange(np.prod(schema.physical_shape), dtype=np.float64).reshape(
        schema.physical_shape
    )
    block = DataBlock(
        BlockId("sparse-image"),
        DatasetRevision(1),
        values,
        VALID,
        schema,
    )
    snapshot = OwnedSnapshot(
        block.ref(StreamGenerationId("sparse-image-generation")),
        block,
    )
    dataset_id = DatasetId("source")
    document = FigureDocument(
        "single-fit-panel",
        0,
        (DatasetDescriptor(dataset_id, "source", schema.fingerprint),),
        (FigureLayer("data", dataset_id, resolved.spec),),
    )
    evaluated = FigureEvaluator().evaluate(
        document,
        ResolvedDatasetMap((ResolvedDataset(dataset_id, snapshot),)),
    )
    assert len(evaluated.layers) == 1
    assert len(evaluated.layers[0].cells) == 1
    assert len(evaluated.layers[0].cells[0].series) == 1

    bound = suggest_fit_draft(
        schema,
        "radial_gaussian_center",
        fit_axis_ids=(
            schema.cell_schema.data_axes[1].axis_id,
            schema.cell_schema.data_axes[0].axis_id,
        ),
    )
    assert bound.spec.committed_transform is None
    assert bound.spec.batch_axis_ids == (repeat_id, event_id, scan_id)


def test_fit_single_panel_presentation_rejects_empty_sparse_selection() -> None:
    schema = _sparse_image_schema()
    seed = suggest_view(schema, ViewIntent.IMAGE)
    assert seed.spec is not None
    impossible = replace(
        seed.spec,
        display_selections=(Selection.index(schema.point_axes[0].axis_id, 0),),
    )
    with pytest.raises(ValueError, match="physical point"):
        fit_single_panel_presentation(schema, impossible)


@pytest.fixture(scope="module")
def application():
    return ensure_qt_app()


@pytest.fixture(scope="module")
def experiment(tmp_path_factory):
    with zlc.connect(
        "virtual",
        repository=tmp_path_factory.mktemp("u03d-direct-fit"),
    ) as connected:
        yield connected


def _until(application, predicate, *, timeout: float = 45.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        application.processEvents(QtCore.QEventLoop.AllEvents, 20)
        QtCore.QCoreApplication.sendPostedEvents(
            None,
            QtCore.QEvent.DeferredDelete,
        )
        time.sleep(0.005)
    assert predicate()


def _two_point_scan_document():
    document = load_sitemap_pulse()
    scanned_api = document.api_parameters[0]
    scanned_period = next(
        period
        for period in document.periods
        if period.period_id == scanned_api.field.period_id
    )
    scan_parameter = ScanParameter(
        "reference_settle",
        scanned_api.field,
        "reference settle",
        scanned_api.unit,
    )
    start = scanned_period.duration
    step = 1 if isinstance(start, int) else 1e-6
    return replace(
        document,
        name="fit-single-panel-scan",
        api_parameters=tuple(
            parameter
            for parameter in document.api_parameters
            if parameter is not scanned_api
        ),
        scan_parameters=(scan_parameter,),
        scan_table=FrozenScanTable(
            (scan_parameter.parameter_id,),
            ((start,), (start + step,)),
        ),
        repeat=RepeatRegion(
            document.periods[0].period_id,
            document.periods[-1].period_id,
            2,
        ),
    )


_SYNTHETIC_IMAGE_OUTPUT = DatasetOutputDeclaration(
    "image",
    "tests.synthetic-associated-image",
)
_SYNTHETIC_IMAGE_DEFINITION = DefinitionKey(
    "tests",
    "synthetic-associated-image",
)
_SYNTHETIC_IMAGE_FRAME = CoordinateFrameId("synthetic-image-pixels")
_SYNTHETIC_IMAGE_SCHEMA = ValueSchema(
    (
        AxisSpec(
            AxisId("synthetic-y"),
            "synthetic y",
            SPATIAL_Y,
            3,
            (0.0, 1.0, 2.0),
            "pixel",
            _SYNTHETIC_IMAGE_FRAME,
        ),
        AxisSpec(
            AxisId("synthetic-x"),
            "synthetic x",
            SPATIAL_X,
            5,
            (0.0, 1.0, 2.0, 3.0, 4.0),
            "pixel",
            _SYNTHETIC_IMAGE_FRAME,
        ),
    ),
    ValidityContract.value(),
    np.dtype("<f8"),
    "count",
)
_SYNTHETIC_ASSOCIATION_SCHEMA = "tests.synthetic-signal-association"


class _SyntheticAssociatedImageCursor:
    """Test producer proving one exact group without naming a device domain."""

    def __init__(self, values: tuple[Value, ...]) -> None:
        self._values = values
        self._stream_id = StreamId("synthetic-image-stream")
        self._generation = StreamGenerationId("synthetic-image-generation")
        self._request: SignalAssociationRequest | None = None
        self._terminal_digest: str | None = None
        self._delivered = 0
        self._closed = False

    @property
    def value_schema(self) -> ValueSchema:
        return _SYNTHETIC_IMAGE_SCHEMA

    @property
    def stream_id(self) -> StreamId:
        return self._stream_id

    @property
    def stream_generation(self) -> StreamGenerationId:
        return self._generation

    @property
    def start_sequence(self) -> int:
        return 0

    def arm_signal_association(self, request: SignalAssociationRequest) -> object:
        assert not self._closed
        assert self._request is None
        assert request.expected_event_count == len(self._values)
        self._request = request
        return request

    def bind_signal_association(
        self,
        token: object,
        terminal_evidence: object,
    ) -> None:
        assert token is self._request
        assert isinstance(terminal_evidence, PulseTerminalAck)
        assert terminal_evidence.session_id == self._request.cause_id
        assert terminal_evidence.artifact_digest == self._request.cause_digest
        self._terminal_digest = canonical_digest(
            pulse_terminal_ack_to_tree(terminal_evidence)
        )

    def next_associated_signal(
        self,
        token: object,
        timeout: float | None = None,
    ) -> SignalEvent:
        del timeout
        assert token is self._request
        assert self._terminal_digest is not None
        sequence = self._delivered
        value = self._values[sequence]
        self._delivered += 1
        reference = EventRef(
            self._stream_id,
            self._generation,
            sequence,
            event_id_for_sequence(self._stream_id, self._generation, sequence),
            canonical_digest({"synthetic_image_sequence": sequence}),
        )
        return SignalEvent(
            value,
            reference,
            TraceContext(
                "synthetic-image-producer-run",
                "synthetic-image-source",
                f"synthetic-image:{sequence}",
            ),
            float(sequence),
        )

    def finish_signal_association(self, token: object) -> SignalAssociationEvidence:
        assert token is self._request
        assert self._request is not None
        assert self._terminal_digest is not None
        assert self._delivered == len(self._values)
        request = self._request
        return SignalAssociationEvidence(
            request,
            self._terminal_digest,
            _SYNTHETIC_ASSOCIATION_SCHEMA,
            encode(
                {
                    "schema": _SYNTHETIC_ASSOCIATION_SCHEMA,
                    "association_id": request.association_id,
                    "cause_id": request.cause_id,
                    "cause_digest": request.cause_digest,
                    "expected_event_count": request.expected_event_count,
                    "terminal_evidence_digest": self._terminal_digest,
                    "producer": "synthetic-associated-image",
                }
            ),
        )

    def close(self) -> None:
        self._closed = True


class _SyntheticAssociatedImageSource:
    def __init__(self, values: tuple[Value, ...]) -> None:
        self._values = values

    def value_schema(self, output_name: str) -> ValueSchema:
        assert output_name == _SYNTHETIC_IMAGE_OUTPUT.name
        return _SYNTHETIC_IMAGE_SCHEMA

    def open_signal_cursor(self, output_name: str):
        return self.open_associated_signal_cursor(output_name)

    def open_associated_signal_cursor(self, output_name: str):
        assert output_name == _SYNTHETIC_IMAGE_OUTPUT.name
        return _SyntheticAssociatedImageCursor(self._values)

    def signal_association_schedule_requirement(
        self,
        output_name: str,
    ) -> SignalAssociationScheduleRequirement:
        assert output_name == _SYNTHETIC_IMAGE_OUTPUT.name
        return SignalAssociationScheduleRequirement()


def _synthetic_image_source(count: int) -> _SyntheticAssociatedImageSource:
    y, x = np.mgrid[-1.0:1.0:3j, -1.0:1.0:5j]
    values = tuple(
        Value(
            np.asarray(
                5.0
                + (20.0 + index)
                * np.exp(-((x - 0.1) ** 2 + (y + 0.1) ** 2) / 0.6),
                dtype=np.dtype("<f8"),
            ),
            VALID,
            _SYNTHETIC_IMAGE_SCHEMA,
        )
        for index in range(count)
    )
    return _SyntheticAssociatedImageSource(values)


def test_direct_capture_fit_entry_is_typed_and_keeps_every_batch_axis(
    application,
    experiment,
) -> None:
    reference = experiment.readout.capture(PULSE)
    window = experiment.fit_gui(
        reference,
        model="radial_gaussian_center",
        timeout_seconds=30.0,
    )
    try:
        _until(
            application,
            lambda: window.worker_idle and bool(window.fit_models),
        )
        assert window.raster_ready
        assert window._view_family == "image"
        assert window._fit_pane is not None
        bound = window._fit_pane.current_option()
        assert bound.spec.model_id == "radial_gaussian_center"
        assert bound.spec.committed_transform is None
        assert set(bound.spec.fit_axis_ids) == set(window._fit_axis_ids)
        assert set(bound.spec.batch_axis_ids) == {
            axis_id
            for axis_id, _role in window._fit_axis_roles
            if axis_id not in bound.spec.fit_axis_ids
        }
        assert all(
            role
            in (
                AxisViewRole.BATCH,
                AxisViewRole.FACET,
                AxisViewRole.SELECTED,
                AxisViewRole.SLIDER,
            )
            or (
                role is AxisViewRole.REDUCED
                and dict(bound.batch_axis_sizes)[axis_id] == 1
            )
            for axis_id, role in window._fit_axis_roles
            if axis_id in bound.spec.batch_axis_ids
        )
        assert any(
            role is AxisViewRole.SELECTED
            for _axis_id, role in window._fit_axis_roles
        )
    finally:
        window.close()
        _until(application, lambda: window.closed and not window.isVisible())


def test_direct_scan_fit_entry_is_typed_and_repeat_remains_authoritative_batch(
    application,
    experiment,
) -> None:
    document = _two_point_scan_document()
    program = AutonomousScanSlotProgram(
        document,
        tuple(
            (
                parameter.parameter_id,
                document.field_value(parameter.field)[0],
            )
            for parameter in document.api_parameters
        ),
    )
    request = PulseScanBoundRequest(
        program,
        ScanSignalBinding(
            _SYNTHETIC_IMAGE_DEFINITION,
            _SYNTHETIC_IMAGE_OUTPUT,
        ),
    )
    source = _synthetic_image_source(
        program.repeat_count * program.point_table.point_layout.storage_size
    )
    reference = experiment.readout.prepare_scan_source(
        request,
        source,
    ).start().result(20.0)
    window = experiment.fit_gui(
        reference,
        model="radial_gaussian_center",
        timeout_seconds=30.0,
    )
    try:
        _until(
            application,
            lambda: window.worker_idle and bool(window.fit_models),
        )
        assert window.raster_ready
        assert window._view_family == "image"
        assert window._fit_pane is not None
        bound = window._fit_pane.current_option()
        assert bound.spec.committed_transform is None
        assert set(bound.spec.batch_axis_ids) == {
            axis_id
            for axis_id, _role in window._fit_axis_roles
            if axis_id not in bound.spec.fit_axis_ids
        }
        repeat_axis = experiment.readout.load_scan(
            reference
        ).output_schema.repeat_axis
        assert repeat_axis.role == REPEAT
        repeat_id = repeat_axis.axis_id
        assert repeat_id in bound.spec.batch_axis_ids
        assert dict(window._fit_axis_roles)[repeat_id] is AxisViewRole.SELECTED
        assert dict(bound.batch_axis_sizes)[repeat_id] == 2
        window._fit_pane.fit_button.click()
        _until(
            application,
            lambda: window.worker_idle
            and window.draft_ready
            and window.raster_ready,
        )
        assert window._fit_draft is not None
        assert window._fit_draft.result.spec.committed_transform is None
        assert (
            window._fit_draft.result.spec.batch_axis_ids
            == bound.spec.batch_axis_ids
        )
        assert window._visible_fit_result_identity is not None
    finally:
        window.close()
        _until(application, lambda: window.closed and not window.isVisible())
