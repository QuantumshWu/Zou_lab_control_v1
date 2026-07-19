"""W8a exact authority-to-presentation projection for transformed Capture Fit."""

from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PyQt5 import QtCore, QtWidgets

import Zou_lab_control.notebook as zlc
from zlc_data import (
    REPEAT,
    SCAN_POINT,
    SITE,
    SPATIAL_X,
    SPATIAL_Y,
    AxisId,
    AxisSpec,
    BlockId,
    ComponentValidity,
    CoordinateFrameId,
    DataBlock,
    DataTransformSpec,
    DatasetRevision,
    DatasetSchema,
    IndexRangeSelection,
    MissingPolicy,
    OwnedSnapshot,
    PointLayout,
    ReductionMethod,
    ReductionSpec,
    Selection,
    StreamGenerationId,
    ValidityContract,
    ValidityPolicy,
    ValueSchema,
    apply_transform,
    bind_fit,
    commit_transform,
    fit_spec_for,
)
from zlc_data.fit_model import evaluate_fit_model
from zlc_frontend import DataFigure
from zlc_frontend.fit_grid import FitGridModel
from zlc_frontend.figure import (
    AxisViewRole,
    DatasetDescriptor,
    DatasetId,
    FigureDocument,
    FigureLayer,
    ResolvedDataset,
    ResolvedDatasetMap,
    SuggestionStatus,
    suggest_fit_view,
)
from zlc_neutral_atom.artifacts import CaptureFitResultRepository
from Zou_lab_control.workbench._frozen_raster import FrozenRasterWindow


ROOT = Path(__file__).resolve().parents[1]
PULSE = ROOT / "zlc_neutral_atom" / "assets" / "imaging_template.json"


@pytest.fixture(scope="module")
def application():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _until(application, predicate, *, timeout: float = 45.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        application.processEvents(QtCore.QEventLoop.AllEvents, 20)
        time.sleep(0.005)
    assert predicate()


def _axis(
    name,
    role,
    size,
    coordinates=None,
    *,
    coordinate_frame=None,
) -> AxisSpec:
    return AxisSpec(
        AxisId(name),
        name,
        role,
        size,
        tuple(range(size)) if coordinates is None else tuple(coordinates),
        "pixel" if role in (SPATIAL_X, SPATIAL_Y) else None,
        coordinate_frame,
    )


def _selection_image_product():
    repeat = _axis("repeat", REPEAT, 1)
    event = _axis("event", SCAN_POINT, 3, (100.0, 200.0, 300.0))
    site = _axis("site", SITE, 2, ("left", "right"))
    frame = CoordinateFrameId("camera")
    y_axis = _axis(
        "camera.y",
        SPATIAL_Y,
        5,
        (10, 11, 12, 13, 14),
        coordinate_frame=frame,
    )
    x_axis = _axis(
        "camera.x",
        SPATIAL_X,
        7,
        (20, 21, 22, 23, 24, 25, 26),
        coordinate_frame=frame,
    )
    point_mapping = ((2,), (0,))
    point_layout = PointLayout.explicit((event.size,), point_mapping)
    y_values, x_values = np.meshgrid(
        np.asarray(y_axis.coordinates, dtype=np.float64),
        np.asarray(x_axis.coordinates, dtype=np.float64),
        indexing="ij",
    )
    values = np.empty(
        (repeat.size, point_layout.storage_size, site.size, y_axis.size, x_axis.size),
        dtype=np.float64,
    )
    for storage, (logical_event,) in enumerate(point_mapping):
        for site_index in range(site.size):
            values[0, storage, site_index] = evaluate_fit_model(
                "radial_gaussian_center",
                (x_values, y_values),
                (
                    20.0 + logical_event + site_index,
                    0.5,
                    23.0 + 0.1 * site_index,
                    12.0 - 0.1 * logical_event,
                    1.4,
                ),
            )
    validity = np.ones(values.shape, dtype=bool)
    validity[0, 0, 0, 1, 3] = False
    values[0, 0, 0, 1, 3] = 1e12
    validity[0, 1, 1, 2:4, 2:5] = False
    values[0, 1, 1, 2:4, 2:5] = 1e12
    schema = DatasetSchema(
        repeat,
        (event,),
        point_layout,
        ValueSchema(
            (site, y_axis, x_axis),
            ValidityContract.components(
                site.axis_id,
                y_axis.axis_id,
                x_axis.axis_id,
            ),
            np.dtype("<f8"),
            "count",
        ),
    )
    block = DataBlock(
        BlockId("w8-transform-source"),
        DatasetRevision(4),
        values,
        ComponentValidity(
            (site.axis_id, y_axis.axis_id, x_axis.axis_id),
            validity,
        ),
        schema,
    )
    snapshot = OwnedSnapshot(
        block.ref(StreamGenerationId("w8-transform-generation")),
        block,
    )
    roi = Selection.rectangle(
        x_axis.axis_id,
        y_axis.axis_id,
        22,
        25,
        11,
        13,
        coordinate_frame=frame,
    )
    committed = commit_transform(
        schema,
        DataTransformSpec((roi,)),
    )
    result = bind_fit(
        fit_spec_for(
            schema,
            "radial_gaussian_center",
            committed_transform=committed,
        ),
        schema,
    ).run(snapshot)
    return snapshot, roi, committed, result, site, y_axis, x_axis


def _data_figure(snapshot, result, suggestion):
    assert suggestion.spec is not None
    dataset_id = DatasetId("source")
    document = FigureDocument(
        "w8-transformed-fit",
        0,
        (
            DatasetDescriptor(
                dataset_id,
                "transformed fit",
                snapshot.block.schema.fingerprint,
            ),
        ),
        (FigureLayer("data", dataset_id, suggestion.spec),),
    )
    return DataFigure(
        document,
        ResolvedDatasetMap((ResolvedDataset(dataset_id, snapshot),)),
        fit_results={"data": result},
        evaluation_memory_limit_bytes=128 << 20,
        render_memory_limit_bytes=128 << 20,
    )


def test_spatial_selection_projection_matches_authoritative_values_and_validity() -> None:
    snapshot, roi, committed, result, _site, y_axis, x_axis = (
        _selection_image_product()
    )
    model = FitGridModel.from_result("fit-result/" + "8" * 64, result)
    page = model.page()
    suggestion = suggest_fit_view(
        snapshot.block.schema,
        result,
        page.selection,
        page.preferences,
    )
    assert suggestion.status is SuggestionStatus.RESOLVED
    assert suggestion.spec is not None
    view = suggestion.spec
    assert view.schema_fingerprint == snapshot.block.schema.fingerprint
    roi_axis_ids = {term.axis_id for term in roi.terms}
    assert tuple(
        term
        for selection in view.display_selections
        for term in selection.terms
        if term.axis_id in roi_axis_ids
    ) == roi.terms
    assert view.binding(x_axis.axis_id).role is AxisViewRole.IMAGE_X
    assert view.binding(y_axis.axis_id).role is AxisViewRole.IMAGE_Y
    assert all(
        binding.role is not AxisViewRole.REDUCED
        for binding in view.axis_bindings
    )

    figure = _data_figure(snapshot, result, suggestion)
    transformed = apply_transform(snapshot, committed)
    expected_values = transformed.values.reshape(
        result.batch_layout.storage_size,
        3,
        4,
    )
    expected_validity = transformed.expanded_validity().reshape(
        expected_values.shape
    )
    present = 0
    holes = 0
    for cell in figure.evaluated.layers[0].cells:
        for item in cell.series:
            addresses = {
                address.axis_id: address
                for address in (*cell.facet_address, *item.batch_address)
            }
            multi = tuple(
                addresses[axis.axis_id].index
                for axis in result.batch_axis_specs
            )
            try:
                storage = result.batch_layout.storage_index(multi)
            except KeyError:
                holes += 1
                assert not np.any(item.data.validity)
                continue
            present += 1
            assert item.data.x_axis.coordinates == (22, 23, 24, 25)
            assert item.data.y_axis.coordinates == (11, 12, 13)
            np.testing.assert_array_equal(
                item.data.values,
                expected_values[storage],
            )
            np.testing.assert_array_equal(
                item.data.validity,
                expected_validity[storage],
            )
    assert present == result.batch_layout.storage_size
    assert holes == 2
    assert snapshot.block.values.shape == (1, 2, 2, 5, 7)
    assert transformed.values.shape == (2, 2, 3, 4)
    assert result.source_ref == snapshot.ref
    assert result.effective_schema_fingerprint == committed.output_schema_fingerprint


def test_projection_rejects_reduction_conflict_and_tampering_fail_closed() -> None:
    snapshot, roi, _committed, result, site, y_axis, x_axis = (
        _selection_image_product()
    )
    reduction = ReductionSpec(
        (site.axis_id,),
        ReductionMethod.MEAN,
        MissingPolicy.REQUIRE_ALL,
        ValidityPolicy.OMIT_INVALID,
    )
    unsupported_transform = commit_transform(
        snapshot.block.schema,
        DataTransformSpec((roi, reduction)),
    )
    unsupported_result = bind_fit(
        fit_spec_for(
            snapshot.block.schema,
            "radial_gaussian_center",
            committed_transform=unsupported_transform,
        ),
        snapshot.block.schema,
    ).run(snapshot)
    unavailable = suggest_fit_view(snapshot.block.schema, unsupported_result)
    assert unavailable.status is SuggestionStatus.NEEDS_INPUT
    assert unavailable.spec is None
    assert unavailable.reasons[0].code == "TRANSFORMED_FIT_DISPLAY_UNAVAILABLE"
    valid_suggestion = suggest_fit_view(snapshot.block.schema, result)
    with pytest.raises(ValueError, match="faithfully displayable"):
        _data_figure(snapshot, unsupported_result, valid_suggestion)

    conflicting = suggest_fit_view(
        snapshot.block.schema,
        result,
        Selection.index(x_axis.axis_id, 0),
    )
    assert conflicting.status is SuggestionStatus.NEEDS_INPUT
    assert conflicting.reasons[0].code == "TRANSFORMED_FIT_SELECTION_CONFLICT"

    suggestion = valid_suggestion
    assert suggestion.spec is not None
    wrong_roi = Selection(
        (
            IndexRangeSelection(y_axis.axis_id, 0, 4),
            IndexRangeSelection(x_axis.axis_id, 2, 6),
        )
    )
    tampered_view = replace(suggestion.spec, display_selections=(wrong_roi,))
    dataset_id = DatasetId("source")
    document = FigureDocument(
        "w8-tampered-fit",
        0,
        (
            DatasetDescriptor(
                dataset_id,
                "tampered",
                snapshot.block.schema.fingerprint,
            ),
        ),
        (FigureLayer("data", dataset_id, tampered_view),),
    )
    with pytest.raises(ValueError, match="committed transform"):
        DataFigure(
            document,
            ResolvedDatasetMap((ResolvedDataset(dataset_id, snapshot),)),
            fit_results={"data": result},
        )
    roi_axis_ids = {term.axis_id for term in roi.terms}
    assert tuple(
        term
        for selection in suggestion.spec.display_selections
        for term in selection.terms
        if term.axis_id in roi_axis_ids
    ) == roi.terms


def test_public_execution_and_saved_ref_reopen_the_same_selection_transform(
    application,
    tmp_path,
    monkeypatch,
) -> None:
    with zlc.connect("virtual", repository=tmp_path / "w8-public") as experiment:
        capture_ref = experiment.readout.capture(PULSE)
        artifact = experiment.readout.load_capture(capture_ref)
        schema = artifact.frame_source.schema
        spatial = {
            axis.role: axis for axis in schema.cell_schema.data_axes
        }
        y_axis, x_axis = spatial[SPATIAL_Y], spatial[SPATIAL_X]
        roi = Selection(
            (
                IndexRangeSelection(y_axis.axis_id, 1, y_axis.size - 1),
                IndexRangeSelection(x_axis.axis_id, 1, x_axis.size - 1),
            )
        )
        committed = commit_transform(schema, DataTransformSpec((roi,)))
        execution = experiment.fit(
            capture_ref,
            model="radial_gaussian_center",
            committed_transform=committed,
        )
        draft_figure = experiment.figure(execution)
        draft_window = experiment.figure_gui(execution)
        try:
            _until(
                application,
                lambda: draft_window.worker_idle and draft_window.raster_ready,
            )
            assert isinstance(draft_window, FrozenRasterWindow)
            assert draft_window._view_family == "encoded"
        finally:
            draft_window.close()
            _until(
                application,
                lambda: draft_window.closed and not draft_window.isVisible(),
            )
        saved_ref = execution.save()
        saved = experiment.load_fit(saved_ref)
        saved_figure = experiment.figure(saved_ref)

        assert saved.result.spec.committed_transform == committed
        assert saved.result.source_ref == execution.result.source_ref
        assert draft_figure.document.layers[0].view == saved_figure.document.layers[0].view
        assert tuple(
            term
            for selection in saved_figure.document.layers[0].view.display_selections
            for term in selection.terms
            if term.axis_id in (x_axis.axis_id, y_axis.axis_id)
        ) == roi.terms

        def forbidden_execute(*_args, **_kwargs):
            raise AssertionError("saved transformed Fit must never re-run the solver")

        monkeypatch.setattr(
            CaptureFitResultRepository,
            "execute",
            forbidden_execute,
        )
        window = experiment.figure_gui(saved_ref)
        try:
            _until(application, lambda: window.worker_idle and window.raster_ready)
            assert type(window).__name__ == "SavedFitGridWindow"
            assert window._model is not None
            assert window._page is not None
        finally:
            window.close()
            _until(application, lambda: window.closed and not window.isVisible())
