"""Direct Fit entry gets one typed display cell without narrowing authority."""

from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PyQt5 import QtCore, QtWidgets
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
    DataBlock,
    DatasetRevision,
    DatasetSchema,
    OwnedSnapshot,
    PointLayout,
    Selection,
    StreamGenerationId,
    VALID,
    ValidityContract,
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
from zlc_neutral_atom.readout.sitemap import load_packaged_sitemap_pulse
from zlc_pulse import FrozenScanTable, RepeatRegion, ScanParameter


ROOT = Path(__file__).resolve().parents[1]
PULSE = ROOT / "zlc_neutral_atom" / "assets" / "imaging_template.json"


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
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


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


def _two_point_image_scan_document():
    document = load_packaged_sitemap_pulse()
    camera_port = next(
        port for port in document.target.ports if port.label == "emCCD"
    )
    trigger_index = document.target.raw_lanes.index(camera_port.lanes[0])
    segment = -1
    previous = 0
    periods = []
    for period in document.periods:
        states = list(period.states)
        current = int(states[trigger_index])
        if current and not previous:
            segment += 1
        states[trigger_index] = int(bool(current and segment == 1))
        periods.append(replace(period, states=tuple(states)))
        previous = current
    scanned_api = document.api_parameters[0]
    scanned_period = next(
        period
        for period in periods
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
        periods=tuple(periods),
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
            periods[0].period_id,
            periods[-1].period_id,
            2,
        ),
    )


def _fixed_api_values(document):
    return {
        parameter.parameter_id: document.field_value(parameter.field)[0]
        for parameter in document.api_parameters
    }


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
    document = _two_point_image_scan_document()
    reference = experiment.readout.scan(
        document,
        api_values=_fixed_api_values(document),
        timeout_seconds=20.0,
    )
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
